//! FSM Calibration Executor
//!
//! Orchestrates the full FSM calibration workflow by coordinating command generation,
//! measurement collection, and transform matrix computation.

use nalgebra::Vector2;
use std::time::Duration;
use thiserror::Error;

use super::{
    build_transform_matrix, fitter, FitError, FsmAxisCalibration, FsmCalibrationConfig,
    FsmDegenerateAxesError, SinusoidGenerator,
};

/// Error during FSM calibration execution
#[derive(Error, Debug)]
pub enum CalibrationError {
    /// Failed to fit sinusoid to axis response
    #[error("axis {axis} fit failed: {source}")]
    FitFailed { axis: u8, source: FitError },

    /// FSM axes are degenerate (too parallel)
    #[error("degenerate axes: {0}")]
    DegenerateAxes(#[from] FsmDegenerateAxesError),

    /// FSM communication error
    #[error("FSM error: {0}")]
    FsmError(String),

    /// Camera/centroid measurement error
    #[error("camera error: {0}")]
    CameraError(String),

    /// Calibration was aborted
    #[error("calibration aborted")]
    Aborted,
}

/// Centroid measurement from camera system
#[derive(Debug, Clone, Copy)]
pub struct CentroidMeasurement {
    /// X position in pixels
    pub x: f64,
    /// Y position in pixels
    pub y: f64,
    /// Timestamp in seconds since calibration start
    pub timestamp_s: f64,
}

/// Interface for FSM control
///
/// Abstracts the FSM hardware for testability.
pub trait FsmInterface {
    /// Send a command to move both FSM axes
    ///
    /// # Arguments
    /// * `axis1_urad` - Axis 1 command in microradians
    /// * `axis2_urad` - Axis 2 command in microradians
    fn move_to(&mut self, axis1_urad: f64, axis2_urad: f64) -> Result<(), String>;

    /// Get the current FSM position
    fn get_position(&self) -> Result<(f64, f64), String>;

    /// Get the sample rate at which commands should be sent
    fn command_rate_hz(&self) -> f64;
}

/// Interface for centroid measurement
///
/// Abstracts the camera/centroid system for testability.
pub trait CentroidSource {
    /// Get the current centroid position
    ///
    /// Returns None if no valid centroid is available (e.g., lost track)
    fn get_centroid(&mut self) -> Result<Option<CentroidMeasurement>, String>;

    /// Get the measurement rate in Hz
    fn measurement_rate_hz(&self) -> f64;
}

/// Progress callback for calibration updates
pub trait ProgressCallback: Send {
    fn report(&mut self, progress: CalibrationProgress);
}

/// No-op progress callback
pub struct NoProgress;

impl ProgressCallback for NoProgress {
    fn report(&mut self, _progress: CalibrationProgress) {}
}

/// Calibration progress information
#[derive(Debug, Clone)]
pub enum CalibrationProgress {
    /// Starting calibration
    Starting,
    /// Wiggling axis 1
    WigglingAxis1 { progress: f64 },
    /// Fitting axis 1 response
    FittingAxis1,
    /// Wiggling axis 2
    WigglingAxis2 { progress: f64 },
    /// Fitting axis 2 response
    FittingAxis2,
    /// Building transform matrix
    BuildingMatrix,
    /// Calibration complete
    Complete,
}

/// Executes FSM calibration workflow
pub struct CalibrationExecutor<F: FsmInterface, C: CentroidSource> {
    fsm: F,
    camera: C,
    config: FsmCalibrationConfig,
}

impl<F: FsmInterface, C: CentroidSource> CalibrationExecutor<F, C> {
    /// Create a new calibration executor
    pub fn new(fsm: F, camera: C, config: FsmCalibrationConfig) -> Self {
        Self {
            fsm,
            camera,
            config,
        }
    }

    /// Run full calibration for both axes
    ///
    /// # Returns
    /// * `Ok(FsmAxisCalibration)` - Completed calibration with transform matrices
    /// * `Err(CalibrationError)` - If calibration fails
    pub fn run_calibration<P: ProgressCallback>(
        &mut self,
        progress: &mut P,
    ) -> Result<FsmAxisCalibration, CalibrationError> {
        progress.report(CalibrationProgress::Starting);

        // Calibrate axis 1 (wiggle axis 1, hold axis 2 at zero)
        let (axis1_fit, axis1_measurements) = self.calibrate_single_axis(1, progress)?;

        // Calibrate axis 2 (wiggle axis 2, hold axis 1 at zero)
        let (axis2_fit, axis2_measurements) = self.calibrate_single_axis(2, progress)?;

        // Build transform matrix from response vectors
        progress.report(CalibrationProgress::BuildingMatrix);

        let axis1_response = self.compute_response_vector(&axis1_fit, &axis1_measurements);
        let axis2_response = self.compute_response_vector(&axis2_fit, &axis2_measurements);

        let fsm_to_sensor = build_transform_matrix(axis1_response, axis2_response)?;

        let sensor_to_fsm = super::invert_matrix(&fsm_to_sensor).map_err(|e| {
            CalibrationError::DegenerateAxes(FsmDegenerateAxesError {
                angle_degrees: e.determinant.abs().acos().to_degrees(),
            })
        })?;

        progress.report(CalibrationProgress::Complete);

        Ok(FsmAxisCalibration {
            fsm_to_sensor,
            sensor_to_fsm,
            axis1_r_squared: axis1_fit.r_squared,
            axis2_r_squared: axis2_fit.r_squared,
            verification_rms_error_pixels: None,
            verification_max_error_pixels: None,
            config: self.config.clone(),
        })
    }

    /// Calibrate a single FSM axis
    fn calibrate_single_axis<P: ProgressCallback>(
        &mut self,
        axis: u8,
        progress: &mut P,
    ) -> Result<(fitter::SinusoidFit, AxisMeasurements), CalibrationError> {
        let wiggle_duration_s = self.config.wiggle_cycles as f64 / self.config.wiggle_frequency_hz;

        let generator = SinusoidGenerator::new(
            self.config.wiggle_amplitude_urad,
            self.config.wiggle_frequency_hz,
            self.fsm.command_rate_hz(),
        );

        let commands = generator.generate(wiggle_duration_s);
        let mut measurements = AxisMeasurements::new();

        let progress_event = if axis == 1 {
            CalibrationProgress::WigglingAxis1 { progress: 0.0 }
        } else {
            CalibrationProgress::WigglingAxis2 { progress: 0.0 }
        };
        progress.report(progress_event);

        let start_time = std::time::Instant::now();

        for (i, &cmd) in commands.iter().enumerate() {
            // Send command to FSM (wiggle one axis, hold other at zero)
            let (axis1_cmd, axis2_cmd) = if axis == 1 { (cmd, 0.0) } else { (0.0, cmd) };

            self.fsm
                .move_to(axis1_cmd, axis2_cmd)
                .map_err(CalibrationError::FsmError)?;

            // Collect centroid measurement
            if let Some(centroid) = self
                .camera
                .get_centroid()
                .map_err(CalibrationError::CameraError)?
            {
                measurements.push(cmd, centroid);
            }

            // Report progress periodically
            if i % 10 == 0 {
                let frac = i as f64 / commands.len() as f64;
                let progress_event = if axis == 1 {
                    CalibrationProgress::WigglingAxis1 { progress: frac }
                } else {
                    CalibrationProgress::WigglingAxis2 { progress: frac }
                };
                progress.report(progress_event);
            }

            // Maintain sample rate timing
            let expected_time =
                Duration::from_secs_f64((i + 1) as f64 / self.fsm.command_rate_hz());
            let actual_time = start_time.elapsed();
            if expected_time > actual_time {
                std::thread::sleep(expected_time - actual_time);
            }
        }

        // Return to center
        self.fsm
            .move_to(0.0, 0.0)
            .map_err(CalibrationError::FsmError)?;

        // Fit sinusoid to collected data
        let fit_progress = if axis == 1 {
            CalibrationProgress::FittingAxis1
        } else {
            CalibrationProgress::FittingAxis2
        };
        progress.report(fit_progress);

        // Fit X and Y separately - one may have zero variance if axes are aligned
        // with sensor coordinates. We use unchecked fitting here and validate
        // the combined response amplitude instead.
        let fit_x = fitter::fit_sinusoid(
            &measurements.centroid_x,
            &measurements.timestamps,
            self.config.wiggle_frequency_hz,
        );

        let fit_y = fitter::fit_sinusoid(
            &measurements.centroid_y,
            &measurements.timestamps,
            self.config.wiggle_frequency_hz,
        );

        // Handle zero variance case - axis might only move in one direction
        let (fit_x, fit_y) = match (fit_x, fit_y) {
            (Ok(x), Ok(y)) => (x, y),
            (Ok(x), Err(FitError::ZeroVariance)) => {
                // Y has no variance - axis only moves in X
                (
                    x,
                    fitter::SinusoidFit {
                        amplitude: 0.0,
                        phase: 0.0,
                        offset: measurements.centroid_y.first().copied().unwrap_or(0.0),
                        r_squared: 1.0, // Perfect fit to constant
                    },
                )
            }
            (Err(FitError::ZeroVariance), Ok(y)) => {
                // X has no variance - axis only moves in Y
                (
                    fitter::SinusoidFit {
                        amplitude: 0.0,
                        phase: 0.0,
                        offset: measurements.centroid_x.first().copied().unwrap_or(0.0),
                        r_squared: 1.0,
                    },
                    y,
                )
            }
            (Err(e), _) => return Err(CalibrationError::FitFailed { axis, source: e }),
            (_, Err(e)) => return Err(CalibrationError::FitFailed { axis, source: e }),
        };

        // Validate that at least one axis has meaningful motion
        let combined_amplitude = (fit_x.amplitude.powi(2) + fit_y.amplitude.powi(2)).sqrt();
        if combined_amplitude < 1e-10 {
            return Err(CalibrationError::FitFailed {
                axis,
                source: FitError::ZeroVariance,
            });
        }

        // Check R² quality for the axis(es) with variance
        let effective_r_squared = if fit_x.amplitude > fit_y.amplitude {
            fit_x.r_squared
        } else {
            fit_y.r_squared
        };

        if effective_r_squared < self.config.min_fit_r_squared {
            return Err(CalibrationError::FitFailed {
                axis,
                source: FitError::LowFitQuality {
                    r_squared: effective_r_squared,
                    threshold: self.config.min_fit_r_squared,
                },
            });
        }

        // Combined fit uses the R² from the dominant axis
        let combined_fit = fitter::SinusoidFit {
            amplitude: combined_amplitude,
            phase: if fit_x.amplitude > fit_y.amplitude {
                fit_x.phase
            } else {
                fit_y.phase
            },
            offset: 0.0,
            r_squared: effective_r_squared,
        };

        // Store individual X/Y amplitudes and phases for response vector
        let measurements_with_fits = AxisMeasurements {
            fit_x: Some(fit_x),
            fit_y: Some(fit_y),
            ..measurements
        };

        Ok((combined_fit, measurements_with_fits))
    }

    /// Compute response vector from fit results
    fn compute_response_vector(
        &self,
        _overall_fit: &fitter::SinusoidFit,
        measurements: &AxisMeasurements,
    ) -> Vector2<f64> {
        let fit_x = measurements.fit_x.as_ref().expect("fit_x should be set");
        let fit_y = measurements.fit_y.as_ref().expect("fit_y should be set");

        // Response vector: pixels of motion per µrad of command
        // Amplitude is in pixels, command was in µrad
        let scale = 1.0 / self.config.wiggle_amplitude_urad;

        // Account for phase difference between X and Y response
        let dx = fit_x.amplitude * fit_x.phase.cos() * scale;
        let dy = fit_y.amplitude * fit_y.phase.cos() * scale;

        Vector2::new(dx, dy)
    }
}

/// Collected measurements during axis calibration
#[derive(Debug, Clone)]
struct AxisMeasurements {
    commands: Vec<f64>,
    centroid_x: Vec<f64>,
    centroid_y: Vec<f64>,
    timestamps: Vec<f64>,
    fit_x: Option<fitter::SinusoidFit>,
    fit_y: Option<fitter::SinusoidFit>,
}

impl AxisMeasurements {
    fn new() -> Self {
        Self {
            commands: Vec::new(),
            centroid_x: Vec::new(),
            centroid_y: Vec::new(),
            timestamps: Vec::new(),
            fit_x: None,
            fit_y: None,
        }
    }

    fn push(&mut self, command: f64, centroid: CentroidMeasurement) {
        self.commands.push(command);
        self.centroid_x.push(centroid.x);
        self.centroid_y.push(centroid.y);
        self.timestamps.push(centroid.timestamp_s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Mock FSM that applies a known transform to commands
    struct MockFsm {
        /// Transform matrix: actual_position = transform * command
        transform: nalgebra::Matrix2<f64>,
        /// Current position
        position: (f64, f64),
        /// Command rate
        rate_hz: f64,
    }

    impl MockFsm {
        fn new(transform: nalgebra::Matrix2<f64>) -> Self {
            Self {
                transform,
                position: (0.0, 0.0),
                rate_hz: 100.0,
            }
        }

        fn identity() -> Self {
            Self::new(nalgebra::Matrix2::identity())
        }

        fn rotated(angle_rad: f64) -> Self {
            Self::new(super::super::matrix::rotation_matrix(angle_rad))
        }

        fn scaled(sx: f64, sy: f64) -> Self {
            Self::new(super::super::matrix::scale_matrix(sx, sy))
        }
    }

    impl FsmInterface for MockFsm {
        fn move_to(&mut self, axis1_urad: f64, axis2_urad: f64) -> Result<(), String> {
            let cmd = Vector2::new(axis1_urad, axis2_urad);
            let pos = self.transform * cmd;
            self.position = (pos.x, pos.y);
            Ok(())
        }

        fn get_position(&self) -> Result<(f64, f64), String> {
            Ok(self.position)
        }

        fn command_rate_hz(&self) -> f64 {
            self.rate_hz
        }
    }

    /// Coupled FSM+Camera mock that simulates a known transform
    ///
    /// This mock couples the FSM commands directly to the camera measurements,
    /// simulating a system where FSM commands immediately produce centroid motion.
    struct CoupledMockSystem {
        /// Transform from FSM commands to sensor position: sensor = transform * fsm
        transform: nalgebra::Matrix2<f64>,
        /// Current FSM command
        current_command: (f64, f64),
        /// Command/measurement rate
        rate_hz: f64,
        /// Measurement counter for timestamps
        measurement_count: usize,
    }

    impl CoupledMockSystem {
        fn with_identity(rate_hz: f64) -> Self {
            Self {
                transform: nalgebra::Matrix2::identity(),
                current_command: (0.0, 0.0),
                rate_hz,
                measurement_count: 0,
            }
        }

        fn with_transform(transform: nalgebra::Matrix2<f64>, rate_hz: f64) -> Self {
            Self {
                transform,
                current_command: (0.0, 0.0),
                rate_hz,
                measurement_count: 0,
            }
        }

        fn get_sensor_position(&self) -> (f64, f64) {
            let cmd = Vector2::new(self.current_command.0, self.current_command.1);
            let pos = self.transform * cmd;
            (pos.x, pos.y)
        }
    }

    impl FsmInterface for CoupledMockSystem {
        fn move_to(&mut self, axis1_urad: f64, axis2_urad: f64) -> Result<(), String> {
            self.current_command = (axis1_urad, axis2_urad);
            Ok(())
        }

        fn get_position(&self) -> Result<(f64, f64), String> {
            Ok(self.current_command)
        }

        fn command_rate_hz(&self) -> f64 {
            self.rate_hz
        }
    }

    impl CentroidSource for CoupledMockSystem {
        fn get_centroid(&mut self) -> Result<Option<CentroidMeasurement>, String> {
            self.measurement_count += 1;
            let (x, y) = self.get_sensor_position();

            Ok(Some(CentroidMeasurement {
                x,
                y,
                timestamp_s: self.measurement_count as f64 / self.rate_hz,
            }))
        }

        fn measurement_rate_hz(&self) -> f64 {
            self.rate_hz
        }
    }

    /// Wrapper that splits a CoupledMockSystem into FSM and Camera halves
    /// This is needed because CalibrationExecutor takes ownership of both
    struct SplitMockFsm {
        inner: std::sync::Arc<std::sync::Mutex<CoupledMockSystem>>,
    }

    struct SplitMockCamera {
        inner: std::sync::Arc<std::sync::Mutex<CoupledMockSystem>>,
    }

    impl FsmInterface for SplitMockFsm {
        fn move_to(&mut self, axis1_urad: f64, axis2_urad: f64) -> Result<(), String> {
            self.inner.lock().unwrap().move_to(axis1_urad, axis2_urad)
        }

        fn get_position(&self) -> Result<(f64, f64), String> {
            self.inner.lock().unwrap().get_position()
        }

        fn command_rate_hz(&self) -> f64 {
            self.inner.lock().unwrap().command_rate_hz()
        }
    }

    impl CentroidSource for SplitMockCamera {
        fn get_centroid(&mut self) -> Result<Option<CentroidMeasurement>, String> {
            self.inner.lock().unwrap().get_centroid()
        }

        fn measurement_rate_hz(&self) -> f64 {
            self.inner.lock().unwrap().measurement_rate_hz()
        }
    }

    fn create_split_mock(system: CoupledMockSystem) -> (SplitMockFsm, SplitMockCamera) {
        let inner = std::sync::Arc::new(std::sync::Mutex::new(system));
        (
            SplitMockFsm {
                inner: inner.clone(),
            },
            SplitMockCamera { inner },
        )
    }

    #[test]
    fn test_calibration_identity_transform() {
        // FSM with identity transform: axis1 moves sensor X, axis2 moves sensor Y
        let rate_hz = 1000.0; // High rate to make test fast (no real sleep)
        let system = CoupledMockSystem::with_identity(rate_hz);
        let (fsm, camera) = create_split_mock(system);

        let config = FsmCalibrationConfig {
            wiggle_amplitude_urad: 100.0,
            wiggle_frequency_hz: 10.0, // Fast wiggle
            wiggle_cycles: 3,
            min_fit_r_squared: 0.9,
            ..Default::default()
        };

        let mut executor = CalibrationExecutor::new(fsm, camera, config);
        let mut progress = NoProgress;

        let result = executor.run_calibration(&mut progress);
        assert!(result.is_ok(), "Calibration should succeed: {:?}", result);

        let calibration = result.unwrap();

        // For identity transform, fsm_to_sensor should be close to identity
        let fsm_to_sensor = calibration.fsm_to_sensor;
        assert!(
            (fsm_to_sensor[(0, 0)] - 1.0).abs() < 0.1,
            "M[0,0] should be ~1.0, got {}",
            fsm_to_sensor[(0, 0)]
        );
        assert!(
            fsm_to_sensor[(0, 1)].abs() < 0.1,
            "M[0,1] should be ~0.0, got {}",
            fsm_to_sensor[(0, 1)]
        );
        assert!(
            fsm_to_sensor[(1, 0)].abs() < 0.1,
            "M[1,0] should be ~0.0, got {}",
            fsm_to_sensor[(1, 0)]
        );
        assert!(
            (fsm_to_sensor[(1, 1)] - 1.0).abs() < 0.1,
            "M[1,1] should be ~1.0, got {}",
            fsm_to_sensor[(1, 1)]
        );

        // Check R² values are high
        assert!(
            calibration.axis1_r_squared > 0.9,
            "Axis 1 R² should be high: {}",
            calibration.axis1_r_squared
        );
        assert!(
            calibration.axis2_r_squared > 0.9,
            "Axis 2 R² should be high: {}",
            calibration.axis2_r_squared
        );
    }

    #[test]
    fn test_calibration_scaled_transform() {
        // FSM with scaled transform: 0.01 pixels per µrad
        let rate_hz = 1000.0;
        let scale = 0.01;
        let transform = super::super::matrix::scale_matrix(scale, scale);
        let system = CoupledMockSystem::with_transform(transform, rate_hz);
        let (fsm, camera) = create_split_mock(system);

        let config = FsmCalibrationConfig {
            wiggle_amplitude_urad: 100.0,
            wiggle_frequency_hz: 10.0,
            wiggle_cycles: 3,
            min_fit_r_squared: 0.9,
            ..Default::default()
        };

        let mut executor = CalibrationExecutor::new(fsm, camera, config);
        let mut progress = NoProgress;

        let result = executor.run_calibration(&mut progress);
        assert!(result.is_ok(), "Calibration should succeed: {:?}", result);

        let calibration = result.unwrap();

        // For scaled transform, diagonal elements should be ~scale
        let fsm_to_sensor = calibration.fsm_to_sensor;
        assert!(
            (fsm_to_sensor[(0, 0)] - scale).abs() < 0.01,
            "M[0,0] should be ~{}, got {}",
            scale,
            fsm_to_sensor[(0, 0)]
        );
        assert!(
            (fsm_to_sensor[(1, 1)] - scale).abs() < 0.01,
            "M[1,1] should be ~{}, got {}",
            scale,
            fsm_to_sensor[(1, 1)]
        );

        // Inverse should recover 1/scale
        let sensor_to_fsm = calibration.sensor_to_fsm;
        let expected_inv = 1.0 / scale;
        assert!(
            (sensor_to_fsm[(0, 0)] - expected_inv).abs() < 10.0,
            "Inverse M[0,0] should be ~{}, got {}",
            expected_inv,
            sensor_to_fsm[(0, 0)]
        );
    }

    #[test]
    fn test_calibration_rotated_transform() {
        // FSM with 45-degree rotation
        let rate_hz = 1000.0;
        let angle = PI / 4.0;
        let transform = super::super::matrix::rotation_matrix(angle);
        let system = CoupledMockSystem::with_transform(transform, rate_hz);
        let (fsm, camera) = create_split_mock(system);

        let config = FsmCalibrationConfig {
            wiggle_amplitude_urad: 100.0,
            wiggle_frequency_hz: 10.0,
            wiggle_cycles: 3,
            min_fit_r_squared: 0.9,
            ..Default::default()
        };

        let mut executor = CalibrationExecutor::new(fsm, camera, config);
        let mut progress = NoProgress;

        let result = executor.run_calibration(&mut progress);
        assert!(result.is_ok(), "Calibration should succeed: {:?}", result);

        let calibration = result.unwrap();

        // Verify the transform roundtrip works
        let test_cmd = Vector2::new(50.0, 30.0);
        let expected_sensor = transform * test_cmd;
        let actual_sensor = calibration.fsm_to_sensor * test_cmd;

        assert!(
            (expected_sensor - actual_sensor).norm() < 5.0,
            "Transform roundtrip failed: expected {:?}, got {:?}",
            expected_sensor,
            actual_sensor
        );
    }

    #[test]
    fn test_mock_fsm_identity() {
        let mut fsm = MockFsm::identity();

        fsm.move_to(100.0, 50.0).unwrap();
        let (x, y) = fsm.get_position().unwrap();

        assert!((x - 100.0).abs() < 1e-10);
        assert!((y - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_mock_fsm_rotated() {
        let mut fsm = MockFsm::rotated(PI / 2.0); // 90 degree rotation

        fsm.move_to(100.0, 0.0).unwrap();
        let (x, y) = fsm.get_position().unwrap();

        // 90 degree rotation: (100, 0) -> (0, 100)
        assert!(x.abs() < 1e-10);
        assert!((y - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_mock_fsm_scaled() {
        let mut fsm = MockFsm::scaled(2.0, 0.5);

        fsm.move_to(100.0, 100.0).unwrap();
        let (x, y) = fsm.get_position().unwrap();

        assert!((x - 200.0).abs() < 1e-10);
        assert!((y - 50.0).abs() < 1e-10);
    }
}

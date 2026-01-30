//! Static Step FSM Calibration Executor
//!
//! Calibrates FSM-to-sensor transform using static step positions instead of
//! sinusoidal wiggle. Moves to discrete positions, waits for settle, then
//! collects tracking messages.

use crate::tracking_collector::TrackingCollector;
use hardware::FsmInterface;
use nalgebra::Vector2;
use shared_wasm::{CalibrateServerClient, FgsServerClient, StatsScan, TrackingState};
use std::mem::discriminant;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::info;

use super::{
    build_transform_matrix, FsmAxisCalibration, FsmCalibrationConfig, FsmDegenerateAxesError,
};

/// Error during static step calibration
#[derive(Error, Debug)]
pub enum StaticCalibrationError {
    /// FSM communication error
    #[error("FSM error: {0}")]
    FsmError(String),

    /// Centroid measurement error
    #[error("centroid source error: {0}")]
    CentroidError(String),

    /// Not enough samples collected
    #[error("insufficient samples at position: got {got}, need {need}")]
    InsufficientSamples { got: usize, need: usize },

    /// FSM axes are degenerate (too parallel)
    #[error("degenerate axes: {0}")]
    DegenerateAxes(#[from] FsmDegenerateAxesError),

    /// Zero motion detected
    #[error("no motion detected for axis {axis}")]
    NoMotionDetected { axis: u8 },

    /// Tracking control error
    #[error("tracking control error: {0}")]
    TrackingError(String),

    /// Tracking failed to reacquire
    #[error("tracking failed to reacquire within timeout")]
    TrackingReacquireFailed,

    /// Display control error
    #[error("display control error: {0}")]
    DisplayError(String),
}

/// A single raw sample measurement
#[derive(Debug, Clone)]
pub struct RawSample {
    /// Axis being calibrated (1 or 2)
    pub axis: u8,
    /// FSM command that was sent (µrad)
    pub fsm_command_urad: f64,
    /// Sample index within this position
    pub sample_index: usize,
    /// Centroid X position (pixels)
    pub centroid_x: f64,
    /// Centroid Y position (pixels)
    pub centroid_y: f64,
    /// Timestamp since calibration start (seconds)
    pub timestamp_s: f64,
}

/// All raw data collected during calibration
#[derive(Debug, Clone, Default)]
pub struct CalibrationRawData {
    /// All raw samples collected
    pub samples: Vec<RawSample>,
}

/// Measured position at a static step
#[derive(Debug, Clone)]
pub struct StepMeasurement {
    /// FSM command that was sent
    pub fsm_command_urad: f64,
    /// Mean centroid X position
    pub mean_x: f64,
    /// Mean centroid Y position
    pub mean_y: f64,
    /// Standard deviation in X
    pub std_x: f64,
    /// Standard deviation in Y
    pub std_y: f64,
}

/// Compute response vector from static step measurements.
///
/// Takes measurements at different FSM positions and computes the
/// centroid displacement per µrad of FSM motion.
pub fn compute_response_vector(
    measurements: &[StepMeasurement],
) -> Result<Vector2<f64>, StaticCalibrationError> {
    if measurements.len() < 2 {
        return Err(StaticCalibrationError::InsufficientSamples {
            got: measurements.len(),
            need: 2,
        });
    }

    // Find min and max positions
    let min = measurements
        .iter()
        .min_by(|a, b| a.fsm_command_urad.partial_cmp(&b.fsm_command_urad).unwrap())
        .ok_or(StaticCalibrationError::InsufficientSamples { got: 0, need: 1 })?;
    let max = measurements
        .iter()
        .max_by(|a, b| a.fsm_command_urad.partial_cmp(&b.fsm_command_urad).unwrap())
        .ok_or(StaticCalibrationError::InsufficientSamples { got: 0, need: 1 })?;

    // Compute delta in FSM command and delta in centroid
    let delta_cmd = max.fsm_command_urad - min.fsm_command_urad;
    let delta_x = max.mean_x - min.mean_x;
    let delta_y = max.mean_y - min.mean_y;

    // Response: pixels per µrad
    let response_x = delta_x / delta_cmd;
    let response_y = delta_y / delta_cmd;

    // Check for zero motion
    let response_magnitude = (response_x.powi(2) + response_y.powi(2)).sqrt();
    if response_magnitude < 1e-10 {
        return Err(StaticCalibrationError::NoMotionDetected { axis: 0 });
    }

    Ok(Vector2::new(response_x, response_y))
}

/// Extract the intercept (centroid position when FSM at center) from measurements.
///
/// Finds the measurement closest to the center position and returns its centroid.
pub fn extract_intercept(
    measurements: &[StepMeasurement],
    center_position_urad: f64,
) -> Result<Vector2<f64>, StaticCalibrationError> {
    let center_measurement = measurements
        .iter()
        .min_by(|a, b| {
            let dist_a = (a.fsm_command_urad - center_position_urad).abs();
            let dist_b = (b.fsm_command_urad - center_position_urad).abs();
            dist_a.partial_cmp(&dist_b).unwrap()
        })
        .ok_or(StaticCalibrationError::InsufficientSamples { got: 0, need: 1 })?;

    Ok(Vector2::new(
        center_measurement.mean_x,
        center_measurement.mean_y,
    ))
}

/// Static step calibration executor
pub struct StaticStepExecutor<F: FsmInterface> {
    fsm: F,
    collector: TrackingCollector,
    config: FsmCalibrationConfig,
    tracking: FgsServerClient,
    display: CalibrateServerClient,
    raw_data: CalibrationRawData,
    calibration_start: Option<Instant>,
}

impl<F: FsmInterface> StaticStepExecutor<F> {
    /// Create a new static step calibration executor
    pub fn new(
        fsm: F,
        collector: TrackingCollector,
        config: FsmCalibrationConfig,
        tracking: FgsServerClient,
        display: CalibrateServerClient,
    ) -> Self {
        Self {
            fsm,
            collector,
            config,
            tracking,
            display,
            raw_data: CalibrationRawData::default(),
            calibration_start: None,
        }
    }

    /// Enable tracking and wait for it to lock on.
    ///
    /// Returns Ok(true) if tracking locked on within timeout, Ok(false) if timeout expired.
    async fn enable_tracking_and_wait(&self, axis: u8) -> Result<bool, StaticCalibrationError> {
        info!("Axis {}: enabling tracking", axis);
        self.tracking
            .set_tracking_enabled(true)
            .await
            .map_err(|e| StaticCalibrationError::TrackingError(e.to_string()))?;

        let timeout = Duration::from_secs_f64(self.config.lock_on_time_secs);

        // Use a dummy TrackingState::Tracking to match on variant
        match self
            .wait_for_tracking_state(
                TrackingState::Tracking {
                    frames_processed: 0,
                },
                timeout,
            )
            .await
        {
            Ok(_) => Ok(true),
            Err(StaticCalibrationError::TrackingReacquireFailed) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Disable tracking and wait for idle state.
    async fn disable_tracking(&self) -> Result<(), StaticCalibrationError> {
        self.tracking
            .set_tracking_enabled(false)
            .await
            .map_err(|e| StaticCalibrationError::TrackingError(e.to_string()))?;

        // Wait for tracking to become idle (up to 5 seconds)
        self.wait_for_tracking_state(TrackingState::Idle, Duration::from_secs(5))
            .await?;

        Ok(())
    }

    /// Wait for tracking to reach a specific state.
    ///
    /// Polls the tracking status every 250ms until the state matches or timeout is reached.
    ///
    /// # Arguments
    /// * `target_state` - The tracking state to wait for (variant only, inner values ignored)
    /// * `max_wait` - Maximum duration to wait before returning timeout error
    ///
    /// # Returns
    /// * `Ok(TrackingState)` - The actual state when matched (includes inner values)
    /// * `Err(TrackingReacquireFailed)` - If timeout reached without matching state
    pub async fn wait_for_tracking_state(
        &self,
        target_state: TrackingState,
        max_wait: Duration,
    ) -> Result<TrackingState, StaticCalibrationError> {
        let start = Instant::now();

        while start.elapsed() < max_wait {
            let status = self
                .tracking
                .get_tracking_status()
                .await
                .map_err(|e| StaticCalibrationError::TrackingError(e.to_string()))?;

            // Compare enum variants only (ignoring inner values)
            if discriminant(&status.state) == discriminant(&target_state) {
                return Ok(status.state);
            }

            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        Err(StaticCalibrationError::TrackingReacquireFailed)
    }

    /// Refresh the display pattern to prevent OLED timeout.
    async fn refresh_display(&self) -> Result<(), StaticCalibrationError> {
        self.display
            .set_pixel_pattern()
            .await
            .map_err(|e| StaticCalibrationError::DisplayError(e.to_string()))?;
        Ok(())
    }

    /// Run the static step calibration.
    ///
    /// Returns the calibration result and raw sample data on success.
    pub async fn run_calibration(
        &mut self,
    ) -> Result<(FsmAxisCalibration, CalibrationRawData), StaticCalibrationError> {
        self.calibration_start = Some(Instant::now());
        self.raw_data = CalibrationRawData::default();
        info!("Starting FSM calibration...");

        // Calibrate axis 1: move axis 1, hold axis 2 at zero
        let axis1_measurements = self.calibrate_axis(1).await?;
        let axis1_response = compute_response_vector(&axis1_measurements)?;

        // Calibrate axis 2: move axis 2, hold axis 1 at zero
        let axis2_measurements = self.calibrate_axis(2).await?;
        let axis2_response = compute_response_vector(&axis2_measurements)?;

        // Build transform matrix
        info!("Building transform matrix...");

        let fsm_to_sensor = build_transform_matrix(axis1_response, axis2_response)?;

        let sensor_to_fsm = super::invert_matrix(&fsm_to_sensor).map_err(|e| {
            StaticCalibrationError::DegenerateAxes(FsmDegenerateAxesError {
                angle_degrees: e.determinant.abs().acos().to_degrees(),
            })
        })?;

        // Extract intercept from center position measurements
        let intercept_pixels =
            extract_intercept(&axis1_measurements, self.config.center_position_urad)?;

        info!("Calibration complete!");

        // Compute mean standard deviation from measurements
        let axis1_std = (axis1_measurements
            .iter()
            .map(|m| m.std_x + m.std_y)
            .sum::<f64>())
            / (axis1_measurements.len() as f64 * 2.0);
        let axis2_std = (axis2_measurements
            .iter()
            .map(|m| m.std_x + m.std_y)
            .sum::<f64>())
            / (axis2_measurements.len() as f64 * 2.0);

        let calibration = FsmAxisCalibration {
            fsm_to_sensor,
            sensor_to_fsm,
            intercept_pixels,
            axis1_std_pixels: axis1_std,
            axis2_std_pixels: axis2_std,
            config: self.config.clone(),
        };

        let raw_data = std::mem::take(&mut self.raw_data);
        Ok((calibration, raw_data))
    }

    /// Calibrate a single axis using static steps
    async fn calibrate_axis(
        &mut self,
        axis: u8,
    ) -> Result<Vec<StepMeasurement>, StaticCalibrationError> {
        let mut measurements = Vec::new();

        // Generate positions across the swing range, centered on center_position
        let center = self.config.center_position_urad;
        let half_swing = self.config.swing_range_urad / 2.0;
        let num_steps = self.config.num_steps;

        let positions: Vec<f64> = if num_steps <= 1 {
            vec![center]
        } else {
            (0..num_steps)
                .map(|i| {
                    let t = i as f64 / (num_steps - 1) as f64;
                    center - half_swing + self.config.swing_range_urad * t
                })
                .collect()
        };

        info!(
            "Axis {}: sweeping {} positions from {:.1} to {:.1} µrad",
            axis,
            positions.len(),
            positions.first().unwrap_or(&center),
            positions.last().unwrap_or(&center)
        );

        for &pos in &positions {
            // Refresh display pattern to prevent OLED timeout
            info!("Axis {}: refreshing display pattern", axis);
            self.refresh_display().await?;

            // Disable tracking before move
            info!("Axis {}: disabling tracking", axis);
            self.disable_tracking().await?;

            // Move to position
            let center = self.config.center_position_urad;
            let (axis1_pos, axis2_pos) = if axis == 1 {
                (pos, center)
            } else {
                (center, pos)
            };

            info!(
                "Axis {}: moving to ({:.1}, {:.1}) µrad",
                axis, axis1_pos, axis2_pos
            );

            self.fsm
                .move_to(axis1_pos, axis2_pos)
                .map_err(StaticCalibrationError::FsmError)?;

            // Enable tracking and wait for lock
            let locked = self.enable_tracking_and_wait(axis).await?;

            if !locked {
                return Err(StaticCalibrationError::TrackingReacquireFailed);
            }

            // Wait for stale messages to arrive, then clear
            // 200ms at 60Hz = ~12 frames - enough for buffered SSE data to arrive
            tokio::time::sleep(Duration::from_millis(200)).await;
            self.collector.clear();

            // Collect samples
            let measurement = self.collect_samples_at_position(axis, pos)?;
            measurements.push(measurement);
        }

        // Disable tracking before returning to center
        self.disable_tracking().await?;

        // Return to center
        let center = self.config.center_position_urad;
        self.fsm
            .move_to(center, center)
            .map_err(StaticCalibrationError::FsmError)?;

        info!("Axis {}: processing data", axis);

        Ok(measurements)
    }

    /// Collect samples at current position using TrackingCollector
    fn collect_samples_at_position(
        &mut self,
        axis: u8,
        fsm_command: f64,
    ) -> Result<StepMeasurement, StaticCalibrationError> {
        let start_time = self.calibration_start.unwrap_or_else(Instant::now);
        let total_samples = self.config.discard_samples + self.config.samples_per_position;

        info!(
            "Axis {}: collecting {} samples (discarding first {})...",
            axis, self.config.samples_per_position, self.config.discard_samples
        );

        // Collect all samples using TrackingCollector
        let timeout = Duration::from_secs(30); // Generous timeout
        let messages = self
            .collector
            .collect_n(total_samples, timeout)
            .map_err(|e| StaticCalibrationError::CentroidError(e.to_string()))?;

        if messages.len() < total_samples {
            return Err(StaticCalibrationError::InsufficientSamples {
                got: messages.len(),
                need: total_samples,
            });
        }

        // Discard initial transient samples
        let messages: Vec<_> = messages
            .into_iter()
            .skip(self.config.discard_samples)
            .collect();

        info!("Axis {}: collected {} samples", axis, messages.len());

        // Record raw samples
        for (i, msg) in messages.iter().enumerate() {
            self.raw_data.samples.push(RawSample {
                axis,
                fsm_command_urad: fsm_command,
                sample_index: i,
                centroid_x: msg.x,
                centroid_y: msg.y,
                timestamp_s: start_time.elapsed().as_secs_f64(),
            });
        }

        // Extract x and y for statistics
        let x_samples: Vec<f64> = messages.iter().map(|m| m.x).collect();
        let y_samples: Vec<f64> = messages.iter().map(|m| m.y).collect();

        let x_stats = StatsScan::new(&x_samples);
        let y_stats = StatsScan::new(&y_samples);

        let mean_x = x_stats.mean().unwrap_or(0.0);
        let mean_y = y_stats.mean().unwrap_or(0.0);
        let std_x = x_stats.std_dev(&x_samples).unwrap_or(0.0);
        let std_y = y_stats.std_dev(&y_samples).unwrap_or(0.0);

        Ok(StepMeasurement {
            fsm_command_urad: fsm_command,
            mean_x,
            mean_y,
            std_x,
            std_y,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsm_calibration::build_transform_matrix;

    fn make_measurement(fsm_cmd: f64, x: f64, y: f64) -> StepMeasurement {
        StepMeasurement {
            fsm_command_urad: fsm_cmd,
            mean_x: x,
            mean_y: y,
            std_x: 0.1,
            std_y: 0.1,
        }
    }

    #[test]
    fn test_response_vector_identity_axis1() {
        // Axis 1 moves in +X direction: 1 pixel per µrad
        let measurements = vec![
            make_measurement(900.0, 0.0, 0.0),
            make_measurement(1000.0, 100.0, 0.0),
            make_measurement(1100.0, 200.0, 0.0),
        ];

        let response = compute_response_vector(&measurements).unwrap();
        assert!((response.x - 1.0).abs() < 1e-10, "X response should be 1.0");
        assert!(response.y.abs() < 1e-10, "Y response should be 0.0");
    }

    #[test]
    fn test_response_vector_identity_axis2() {
        // Axis 2 moves in +Y direction: 1 pixel per µrad
        let measurements = vec![
            make_measurement(900.0, 0.0, 0.0),
            make_measurement(1000.0, 0.0, 100.0),
            make_measurement(1100.0, 0.0, 200.0),
        ];

        let response = compute_response_vector(&measurements).unwrap();
        assert!(response.x.abs() < 1e-10, "X response should be 0.0");
        assert!((response.y - 1.0).abs() < 1e-10, "Y response should be 1.0");
    }

    #[test]
    fn test_response_vector_scaled() {
        // 0.01 pixels per µrad (typical real value)
        let scale = 0.01;
        let measurements = vec![
            make_measurement(900.0, 0.0, 0.0),
            make_measurement(1000.0, 1.0, 0.0),
            make_measurement(1100.0, 2.0, 0.0),
        ];

        let response = compute_response_vector(&measurements).unwrap();
        assert!(
            (response.x - scale).abs() < 1e-10,
            "X response should be {}",
            scale
        );
    }

    #[test]
    fn test_response_vector_rotated() {
        // Axis moves at 45 degrees: equal X and Y motion
        let measurements = vec![
            make_measurement(900.0, 0.0, 0.0),
            make_measurement(1000.0, 100.0, 100.0),
            make_measurement(1100.0, 200.0, 200.0),
        ];

        let response = compute_response_vector(&measurements).unwrap();
        assert!((response.x - 1.0).abs() < 1e-10, "X response should be 1.0");
        assert!((response.y - 1.0).abs() < 1e-10, "Y response should be 1.0");
    }

    #[test]
    fn test_response_vector_insufficient_samples() {
        let measurements = vec![make_measurement(1000.0, 100.0, 50.0)];

        let result = compute_response_vector(&measurements);
        assert!(matches!(
            result,
            Err(StaticCalibrationError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_response_vector_no_motion() {
        // FSM moves but centroid doesn't
        let measurements = vec![
            make_measurement(900.0, 100.0, 50.0),
            make_measurement(1000.0, 100.0, 50.0),
            make_measurement(1100.0, 100.0, 50.0),
        ];

        let result = compute_response_vector(&measurements);
        assert!(matches!(
            result,
            Err(StaticCalibrationError::NoMotionDetected { .. })
        ));
    }

    #[test]
    fn test_extract_intercept() {
        let measurements = vec![
            make_measurement(900.0, 10.0, 20.0),
            make_measurement(1000.0, 512.0, 384.0), // center
            make_measurement(1100.0, 1014.0, 748.0),
        ];

        let intercept = extract_intercept(&measurements, 1000.0).unwrap();
        assert!((intercept.x - 512.0).abs() < 1e-10);
        assert!((intercept.y - 384.0).abs() < 1e-10);
    }

    #[test]
    fn test_full_calibration_identity() {
        // Simulate identity transform: axis1 -> +X, axis2 -> +Y
        let center = 1000.0;
        let amp = 100.0;

        let axis1_measurements = vec![
            make_measurement(center - amp, 0.0, 0.0),
            make_measurement(center, 100.0, 0.0),
            make_measurement(center + amp, 200.0, 0.0),
        ];

        let axis2_measurements = vec![
            make_measurement(center - amp, 100.0, 0.0),
            make_measurement(center, 100.0, 100.0),
            make_measurement(center + amp, 100.0, 200.0),
        ];

        let axis1_response = compute_response_vector(&axis1_measurements).unwrap();
        let axis2_response = compute_response_vector(&axis2_measurements).unwrap();

        let fsm_to_sensor = build_transform_matrix(axis1_response, axis2_response).unwrap();

        // Should be identity (within tolerance)
        assert!(
            (fsm_to_sensor[(0, 0)] - 1.0).abs() < 1e-10,
            "M[0,0] = {}",
            fsm_to_sensor[(0, 0)]
        );
        assert!(
            fsm_to_sensor[(0, 1)].abs() < 1e-10,
            "M[0,1] = {}",
            fsm_to_sensor[(0, 1)]
        );
        assert!(
            fsm_to_sensor[(1, 0)].abs() < 1e-10,
            "M[1,0] = {}",
            fsm_to_sensor[(1, 0)]
        );
        assert!(
            (fsm_to_sensor[(1, 1)] - 1.0).abs() < 1e-10,
            "M[1,1] = {}",
            fsm_to_sensor[(1, 1)]
        );
    }

    #[test]
    fn test_full_calibration_scaled() {
        // 0.01 pixels per µrad
        let scale = 0.01;
        let center = 1000.0;
        let amp = 100.0;

        let axis1_measurements = vec![
            make_measurement(center - amp, 0.0, 0.0),
            make_measurement(center, scale * amp, 0.0),
            make_measurement(center + amp, scale * 2.0 * amp, 0.0),
        ];

        let axis2_measurements = vec![
            make_measurement(center - amp, scale * amp, 0.0),
            make_measurement(center, scale * amp, scale * amp),
            make_measurement(center + amp, scale * amp, scale * 2.0 * amp),
        ];

        let axis1_response = compute_response_vector(&axis1_measurements).unwrap();
        let axis2_response = compute_response_vector(&axis2_measurements).unwrap();

        let fsm_to_sensor = build_transform_matrix(axis1_response, axis2_response).unwrap();

        assert!(
            (fsm_to_sensor[(0, 0)] - scale).abs() < 1e-10,
            "M[0,0] = {}, expected {}",
            fsm_to_sensor[(0, 0)],
            scale
        );
        assert!(
            (fsm_to_sensor[(1, 1)] - scale).abs() < 1e-10,
            "M[1,1] = {}, expected {}",
            fsm_to_sensor[(1, 1)],
            scale
        );
    }

    #[test]
    fn test_full_calibration_rotated_45deg() {
        // Both axes at 45 degrees
        let scale = 1.0;
        let center = 1000.0;
        let amp = 100.0;
        let diag = scale / std::f64::consts::SQRT_2;

        // Axis 1 moves at +45 deg
        let axis1_measurements = vec![
            make_measurement(center - amp, 0.0, 0.0),
            make_measurement(center, diag * amp, diag * amp),
            make_measurement(center + amp, diag * 2.0 * amp, diag * 2.0 * amp),
        ];

        // Axis 2 moves at -45 deg (perpendicular)
        let axis2_measurements = vec![
            make_measurement(center - amp, diag * amp, diag * amp),
            make_measurement(center, diag * 2.0 * amp, 0.0),
            make_measurement(center + amp, diag * 3.0 * amp, -diag * amp),
        ];

        let axis1_response = compute_response_vector(&axis1_measurements).unwrap();
        let axis2_response = compute_response_vector(&axis2_measurements).unwrap();

        let fsm_to_sensor = build_transform_matrix(axis1_response, axis2_response).unwrap();

        // Check that both columns have magnitude = scale
        let col1_mag = (fsm_to_sensor[(0, 0)].powi(2) + fsm_to_sensor[(1, 0)].powi(2)).sqrt();
        let col2_mag = (fsm_to_sensor[(0, 1)].powi(2) + fsm_to_sensor[(1, 1)].powi(2)).sqrt();

        assert!(
            (col1_mag - scale).abs() < 1e-10,
            "Column 1 magnitude = {}, expected {}",
            col1_mag,
            scale
        );
        assert!(
            (col2_mag - scale).abs() < 1e-10,
            "Column 2 magnitude = {}, expected {}",
            col2_mag,
            scale
        );
    }
}

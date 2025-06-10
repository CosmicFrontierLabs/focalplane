use nalgebra::{Unit, UnitQuaternion, Vector3};
use rand::prelude::*;
use rustfft::{num_complex::Complex64, FftPlanner};
use std::{f64::consts::PI, time::Duration};

/// Validate PSD curve inputs for physical correctness.
///
/// # Arguments
/// * `points` - PSD points to validate
/// * `axis` - Axis vector to validate
///
/// # Panics
/// * If any frequency or amplitude values are NaN, infinite, or negative
/// * If the axis vector is zero or contains NaN/infinite values
/// * If frequency points are not sorted in ascending order
fn validate_psd_inputs(points: &[PsdPoint], axis: &Vector3<f64>) {
    // Validate axis vector
    if !axis.iter().all(|&x| x.is_finite()) {
        panic!(
            "PSD axis vector contains NaN or infinite values: {:?}",
            axis
        );
    }
    if axis.norm() == 0.0 {
        panic!("PSD axis vector cannot be zero vector");
    }

    // Validate PSD points
    for (i, point) in points.iter().enumerate() {
        if !point.frequency.is_finite() {
            panic!("PSD point {} has invalid frequency: {}", i, point.frequency);
        }
        if point.frequency < 0.0 {
            panic!(
                "PSD point {} has negative frequency: {}",
                i, point.frequency
            );
        }
        if !point.amplitude_squared.is_finite() {
            panic!(
                "PSD point {} has invalid amplitude_squared: {}",
                i, point.amplitude_squared
            );
        }
        if point.amplitude_squared < 0.0 {
            panic!(
                "PSD point {} has negative amplitude_squared: {}",
                i, point.amplitude_squared
            );
        }
    }

    // Validate frequency sorting
    for i in 1..points.len() {
        if points[i].frequency <= points[i - 1].frequency {
            panic!(
                "PSD points are not sorted by frequency: point {} frequency {} <= point {} frequency {}",
                i,
                points[i].frequency,
                i - 1,
                points[i - 1].frequency
            );
        }
    }
}

/// A single point in a Power Spectral Density curve.
///
/// Represents the magnitude of angular vibrations at a specific frequency,
/// typically used to characterize mechanical disturbances in telescopes.
#[derive(Debug, Clone)]
pub struct PsdPoint {
    /// Frequency in Hz
    pub frequency: f64,
    /// Power spectral density in rad²/Hz
    pub amplitude_squared: f64,
}

/// Power Spectral Density curve defining angular vibrations along a specific axis.
///
/// This characterizes how mechanical disturbances (like telescope mount vibrations)
/// vary with frequency. The PSD is used to generate realistic time-domain angular
/// displacements via inverse FFT methods.
#[derive(Debug, Clone)]
pub struct PsdCurve {
    /// Frequency points defining the PSD curve
    pub points: Vec<PsdPoint>,
    /// Unit vector defining the axis of rotation for this PSD
    pub axis: Vector3<f64>,
}

impl PsdCurve {
    /// Create a new PSD curve with axis normalization.
    ///
    /// # Arguments
    /// * `points` - Frequency-amplitude pairs defining the PSD curve (must be sorted by frequency)
    /// * `axis` - Rotation axis (will be normalized to unit vector)
    ///
    /// # Panics
    /// * If any frequency or amplitude values are NaN, infinite, or negative
    /// * If the axis vector is zero or contains NaN/infinite values
    /// * If frequency points are not sorted in ascending order
    pub fn new(points: Vec<PsdPoint>, axis: Vector3<f64>) -> Self {
        validate_psd_inputs(&points, &axis);

        Self {
            points,
            axis: axis.normalize(),
        }
    }

    /// Interpolate PSD value at a given frequency using log-log interpolation.
    ///
    /// Returns constant extrapolation beyond the defined frequency range.
    /// Uses log-log interpolation for better accuracy with typical PSD curves
    /// that follow power-law relationships.
    pub fn interpolate(&self, freq: f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }

        if freq <= self.points[0].frequency {
            return self.points[0].amplitude_squared;
        }

        if freq >= self.points.last().unwrap().frequency {
            return self.points.last().unwrap().amplitude_squared;
        }

        // Linear interpolation in log-log space for better accuracy
        for i in 0..self.points.len() - 1 {
            if freq >= self.points[i].frequency && freq <= self.points[i + 1].frequency {
                let x0 = self.points[i].frequency.ln();
                let x1 = self.points[i + 1].frequency.ln();
                let y0 = self.points[i].amplitude_squared.ln();
                let y1 = self.points[i + 1].amplitude_squared.ln();

                let log_freq = freq.ln();
                let log_psd = y0 + (y1 - y0) * (log_freq - x0) / (x1 - x0);
                return log_psd.exp();
            }
        }

        0.0
    }

    /// Calculate RMS angular displacement from this PSD curve via numerical integration.
    ///
    /// Integrates the PSD over frequency using the trapezoidal rule to compute
    /// the total variance, then returns the square root (RMS).
    ///
    /// # Returns
    /// RMS angular displacement in radians
    pub fn calculate_rms(&self) -> f64 {
        // Integrate PSD over frequency to get variance
        let mut variance = 0.0;

        // Numerical integration using trapezoidal rule
        for i in 0..self.points.len() - 1 {
            let f1 = self.points[i].frequency;
            let f2 = self.points[i + 1].frequency;
            let p1 = self.points[i].amplitude_squared;
            let p2 = self.points[i + 1].amplitude_squared;

            variance += 0.5 * (p1 + p2) * (f2 - f1);
        }

        variance.sqrt()
    }
}

/// Vibration simulator that converts PSD curves to time-domain quaternion orientations.
///
/// This simulator uses inverse FFT methods to generate realistic time-domain angular
/// displacements from frequency-domain PSD specifications. It supports multiple PSD
/// curves along different axes to model complex 3D vibration environments.
///
/// The generated orientations can be used to simulate mechanical disturbances in
/// telescope tracking systems, star tracker jitter, or other precision pointing systems.
pub struct VibrationSimulator {
    /// PSD curves defining vibrations along different axes
    psd_curves: Vec<PsdCurve>,
    /// Sampling rate for time-domain generation in Hz
    sample_rate: f64,
}

impl VibrationSimulator {
    /// Create a new vibration simulator.
    ///
    /// # Arguments
    /// * `psd_curves` - Vector of PSD curves defining vibrations along different axes
    /// * `sample_rate` - Sampling rate in Hz for time-domain generation
    pub fn new(psd_curves: Vec<PsdCurve>, sample_rate: f64) -> Self {
        Self {
            psd_curves,
            sample_rate,
        }
    }

    /// Generate time-domain angular displacement from PSD using inverse FFT approach.
    ///
    /// This method converts frequency-domain PSD specifications into realistic
    /// time-domain angular displacements by:
    /// 1. Creating frequency-domain spectrum with appropriate amplitudes
    /// 2. Adding random phases to each frequency component
    /// 3. Using inverse FFT to generate time-domain signals
    ///
    /// # Arguments
    /// * `duration` - Simulation duration
    /// * `seed` - Optional random seed. If None, uses random seed
    ///
    /// # Returns
    /// Vector of 3D angular displacements in radians, one per time sample
    pub fn generate_angular_displacement(
        &self,
        duration: Duration,
        seed: Option<u64>,
    ) -> Vec<Vector3<f64>> {
        let n_samples = (duration.as_secs_f64() * self.sample_rate) as usize;
        // Ensure power of 2 for FFT efficiency
        let n_fft = n_samples.next_power_of_two();

        // Initialize RNG at the top
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let mut angular_displacements = vec![Vector3::zeros(); n_samples];

        // Process each PSD curve
        for psd in &self.psd_curves {
            // Generate frequency-domain representation
            let mut spectrum = vec![Complex64::new(0.0, 0.0); n_fft];

            // Fill spectrum based on PSD
            for i in 0..n_fft / 2 + 1 {
                let freq = i as f64 * self.sample_rate / n_fft as f64;

                if freq > 0.0 && freq < self.sample_rate / 2.0 {
                    let psd_value = psd.interpolate(freq);

                    // Convert PSD to amplitude spectrum
                    // PSD = |X(f)|² / Δf, so |X(f)| = sqrt(PSD * Δf)
                    let df = self.sample_rate / n_fft as f64;
                    let amplitude = (psd_value * df).sqrt();

                    // Random phase
                    let phase = rng.gen_range(0.0..2.0 * PI);
                    spectrum[i] = Complex64::from_polar(amplitude, phase);

                    // Hermitian symmetry for real signal
                    if i > 0 && i < n_fft / 2 {
                        spectrum[n_fft - i] = spectrum[i].conj();
                    }
                }
            }

            // Inverse FFT to get time-domain signal
            let mut planner = FftPlanner::new();
            let inverse_fft = planner.plan_fft_inverse(n_fft);
            inverse_fft.process(&mut spectrum);

            // Scale and extract real part
            let scale = 1.0 / (n_fft as f64).sqrt();

            // Add contribution from this PSD curve along its axis
            for i in 0..n_samples {
                angular_displacements[i] += psd.axis * (spectrum[i].re * scale);
            }
        }

        angular_displacements
    }

    /// Generate quaternion orientations from angular displacements.
    ///
    /// Converts angular displacement vectors into unit quaternions representing
    /// absolute orientations. For small angles, uses axis-angle representation.
    ///
    /// # Arguments
    /// * `duration` - Simulation duration
    /// * `seed` - Optional random seed. If None, uses random seed
    ///
    /// # Returns
    /// Vector of unit quaternions representing absolute orientations
    pub fn generate_orientations(
        &self,
        duration: Duration,
        seed: Option<u64>,
    ) -> Vec<UnitQuaternion<f64>> {
        let angular_displacements = self.generate_angular_displacement(duration, seed);

        angular_displacements
            .iter()
            .map(|theta| {
                // For small angles, quaternion from axis-angle
                let angle = theta.norm();
                if angle < 1e-10 {
                    UnitQuaternion::identity()
                } else {
                    let axis = Unit::new_normalize(*theta);
                    UnitQuaternion::from_axis_angle(&axis, angle)
                }
            })
            .collect()
    }

    /// Generate quaternion orientation deltas (frame-to-frame rotations).
    ///
    /// Computes the relative rotation between consecutive time steps.
    /// Useful for simulating incremental tracking errors or for integration
    /// into existing tracking systems.
    ///
    /// # Arguments
    /// * `duration` - Simulation duration
    /// * `seed` - Optional random seed. If None, uses random seed
    ///
    /// # Returns
    /// Vector of unit quaternions representing frame-to-frame rotations
    pub fn generate_orientation_deltas(
        &self,
        duration: Duration,
        seed: Option<u64>,
    ) -> Vec<UnitQuaternion<f64>> {
        let orientations = self.generate_orientations(duration, seed);
        let mut deltas = Vec::with_capacity(orientations.len());

        // First delta is from identity to first orientation
        if !orientations.is_empty() {
            deltas.push(orientations[0]);
        }

        // Subsequent deltas are frame-to-frame
        for i in 1..orientations.len() {
            let delta = orientations[i - 1].inverse() * orientations[i];
            deltas.push(delta);
        }

        deltas
    }

    /// Generate time vector for the simulation.
    ///
    /// Creates equally-spaced time samples from 0 to duration based on sample rate.
    ///
    /// # Arguments
    /// * `duration` - Simulation duration
    ///
    /// # Returns
    /// Vector of time values in seconds
    pub fn time_vector(&self, duration: Duration) -> Vec<f64> {
        let n_samples = (duration.as_secs_f64() * self.sample_rate) as usize;
        (0..n_samples)
            .map(|i| i as f64 / self.sample_rate)
            .collect()
    }

    /// Get the configured sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Calculate expected RMS angular displacement from all PSD curves.
    ///
    /// Computes the theoretical RMS value for each axis by integrating
    /// all PSD curves and projecting onto their respective axes.
    ///
    /// # Returns
    /// 3D vector of RMS angular displacements in radians for each axis
    pub fn calculate_total_rms(&self) -> Vector3<f64> {
        let mut rms = Vector3::zeros();

        for psd in &self.psd_curves {
            rms += psd.axis * psd.calculate_rms();
        }

        rms
    }
}

/// N-dimensional PSD curve that combines multiple orthonormal 1D PSD curves.
///
/// This represents vibrations in an n-dimensional space defined by orthonormal axes.
/// The resulting quaternions are composed by applying rotations from all
/// axes sequentially. This is useful for modeling telescope mount vibrations
/// that have coupled motion in multiple orthogonal pointing directions.
#[derive(Debug, Clone)]
pub struct PsdCurveND {
    /// Vector of PSD curves along orthonormal axes
    pub curves: Vec<PsdCurve>,
}

/// Validate that a set of PSD curves have orthonormal axes.
///
/// # Arguments
/// * `curves` - Vector of PSD curves to validate
///
/// # Panics
/// * If any pair of axes are not orthonormal (dot product not near zero)
/// * If any axis is not unit length
fn validate_orthonormal_axes(curves: &[PsdCurve]) {
    for (i, curve_i) in curves.iter().enumerate() {
        // Check unit length
        if (curve_i.axis.norm() - 1.0).abs() > 1e-10 {
            panic!(
                "PSD curve {} axis is not unit length: norm = {}",
                i,
                curve_i.axis.norm()
            );
        }

        // Check orthogonality with all other curves
        for (j, curve_j) in curves.iter().enumerate() {
            if i != j {
                let dot_product = curve_i.axis.dot(&curve_j.axis);
                if dot_product.abs() > 1e-10 {
                    panic!(
                        "PSD curve axes {} and {} are not orthogonal: dot product = {}, expected near 0",
                        i, j, dot_product
                    );
                }
            }
        }
    }
}

impl PsdCurveND {
    /// Create a new N-dimensional PSD curve with validation for orthonormal axes.
    ///
    /// # Arguments
    /// * `curves` - Vector of PSD curves with orthonormal axes
    ///
    /// # Panics
    /// * If any pair of axes are not orthonormal (dot product not near zero)
    /// * If any axis is not unit length
    /// * If the curves vector is empty
    pub fn new(curves: Vec<PsdCurve>) -> Self {
        if curves.is_empty() {
            panic!("PsdCurveND requires at least one curve");
        }

        validate_orthonormal_axes(&curves);
        Self { curves }
    }

    /// Get the number of dimensions (number of curves).
    pub fn dimensions(&self) -> usize {
        self.curves.len()
    }

    /// Generate N-dimensional angular displacements from all PSD curves.
    ///
    /// Creates independent time-domain signals for all axes and combines them
    /// into a single vector of 3D angular displacements.
    ///
    /// # Arguments
    /// * `duration` - Simulation duration
    /// * `sample_rate` - Sampling rate in Hz
    /// * `seed` - Optional random seed for reproducible results
    ///
    /// # Returns
    /// Vector of 3D angular displacements combining all axes
    pub fn generate_angular_displacement(
        &self,
        duration: Duration,
        sample_rate: f64,
        seed: Option<u64>,
    ) -> Vec<Vector3<f64>> {
        let n_samples = (duration.as_secs_f64() * sample_rate) as usize;
        let mut combined_displacements = vec![Vector3::zeros(); n_samples];

        // Generate and combine displacements from all curves
        for curve in &self.curves {
            let sim = VibrationSimulator::new(vec![curve.clone()], sample_rate);
            let displacements = sim.generate_angular_displacement(duration, seed);

            for (i, displacement) in displacements.into_iter().enumerate() {
                if i < combined_displacements.len() {
                    combined_displacements[i] += displacement;
                }
            }
        }

        combined_displacements
    }

    /// Generate quaternion orientations by composing rotations from all axes.
    ///
    /// Creates independent angular displacements for all axes and composes
    /// them into unit quaternions by applying rotations sequentially.
    ///
    /// # Arguments
    /// * `duration` - Simulation duration
    /// * `sample_rate` - Sampling rate in Hz
    /// * `seed` - Optional random seed for reproducible results
    ///
    /// # Returns
    /// Vector of unit quaternions representing composed orientations
    pub fn generate_orientations(
        &self,
        duration: Duration,
        sample_rate: f64,
        seed: Option<u64>,
    ) -> Vec<UnitQuaternion<f64>> {
        let n_samples = (duration.as_secs_f64() * sample_rate) as usize;
        let mut combined_orientations = vec![UnitQuaternion::identity(); n_samples];

        // Generate orientations for each curve and compose them
        for curve in &self.curves {
            let sim = VibrationSimulator::new(vec![curve.clone()], sample_rate);
            let orientations = sim.generate_orientations(duration, seed);

            for (i, orientation) in orientations.into_iter().enumerate() {
                if i < combined_orientations.len() {
                    combined_orientations[i] = combined_orientations[i] * orientation;
                }
            }
        }

        combined_orientations
    }

    /// Generate quaternion orientation deltas by composing frame-to-frame rotations.
    ///
    /// Computes composed orientation deltas from all axes. This is useful for
    /// simulating incremental tracking errors in n-dimensional space.
    ///
    /// # Arguments
    /// * `duration` - Simulation duration
    /// * `sample_rate` - Sampling rate in Hz
    /// * `seed` - Optional random seed for reproducible results
    ///
    /// # Returns
    /// Vector of unit quaternions representing composed frame-to-frame rotations
    pub fn generate_orientation_deltas(
        &self,
        duration: Duration,
        sample_rate: f64,
        seed: Option<u64>,
    ) -> Vec<UnitQuaternion<f64>> {
        let orientations = self.generate_orientations(duration, sample_rate, seed);
        let mut deltas = Vec::with_capacity(orientations.len());

        // First delta is from identity to first orientation
        if !orientations.is_empty() {
            deltas.push(orientations[0]);
        }

        // Subsequent deltas are frame-to-frame
        for i in 1..orientations.len() {
            let delta = orientations[i - 1].inverse() * orientations[i];
            deltas.push(delta);
        }

        deltas
    }

    /// Calculate expected RMS angular displacement from all PSD curves.
    ///
    /// Computes the theoretical RMS value by combining the RMS from all curves.
    /// Since the axes are orthonormal, the total RMS is the vector sum.
    ///
    /// # Returns
    /// 3D vector of RMS angular displacements in radians
    pub fn calculate_total_rms(&self) -> Vector3<f64> {
        let mut rms = Vector3::zeros();
        for curve in &self.curves {
            rms += curve.axis * curve.calculate_rms();
        }
        rms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    #[test]
    fn test_psd_curve_creation() {
        let points = vec![
            PsdPoint {
                frequency: 1.0,
                amplitude_squared: 0.01,
            },
            PsdPoint {
                frequency: 10.0,
                amplitude_squared: 0.1,
            },
        ];
        let axis = Vector3::new(1.0, 0.0, 0.0);

        let curve = PsdCurve::new(points, axis);

        assert_eq!(curve.points.len(), 2);
        assert_relative_eq!(curve.axis.norm(), 1.0);
        assert_relative_eq!(curve.points[0].frequency, 1.0);
        assert_relative_eq!(curve.points[1].frequency, 10.0);
    }

    #[test]
    fn test_psd_interpolation() {
        let points = vec![
            PsdPoint {
                frequency: 1.0,
                amplitude_squared: 0.01,
            },
            PsdPoint {
                frequency: 10.0,
                amplitude_squared: 0.1,
            },
        ];
        let axis = Vector3::new(0.0, 1.0, 0.0);
        let curve = PsdCurve::new(points, axis);

        // Test boundary values
        assert_relative_eq!(curve.interpolate(0.5), 0.01);
        assert_relative_eq!(curve.interpolate(15.0), 0.1);

        // Test interpolation in log-log space
        // At geometric midpoint sqrt(1*10) = 3.16, value should be sqrt(0.01*0.1) but in log space
        let expected = (0.01_f64.ln() + 0.1_f64.ln()) / 2.0;
        assert_relative_eq!(
            curve.interpolate(3.16227766017).ln(),
            expected,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_rms_calculation() {
        // Simple rectangle PSD for easy integration check
        let points = vec![
            PsdPoint {
                frequency: 1.0,
                amplitude_squared: 0.04,
            },
            PsdPoint {
                frequency: 2.0,
                amplitude_squared: 0.04,
            },
        ];
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let curve = PsdCurve::new(points, axis);

        // RMS = sqrt(∫PSD df) = sqrt(0.04 * (2-1)) = sqrt(0.04) = 0.2
        assert_relative_eq!(curve.calculate_rms(), 0.2);
    }

    #[test]
    fn test_vibration_simulator_creation() {
        let psd_curve = PsdCurve::new(
            vec![PsdPoint {
                frequency: 1.0,
                amplitude_squared: 0.01,
            }],
            Vector3::new(1.0, 0.0, 0.0),
        );
        let simulator = VibrationSimulator::new(vec![psd_curve], 100.0);

        assert_eq!(simulator.psd_curves.len(), 1);
        assert_relative_eq!(simulator.sample_rate, 100.0);
    }

    #[test]
    fn test_time_vector() {
        let psd_curve = PsdCurve::new(
            vec![PsdPoint {
                frequency: 1.0,
                amplitude_squared: 0.01,
            }],
            Vector3::new(1.0, 0.0, 0.0),
        );
        let simulator = VibrationSimulator::new(vec![psd_curve], 10.0);

        let times = simulator.time_vector(Duration::from_millis(500));

        assert_eq!(times.len(), 5); // 0.5 seconds * 10 Hz = 5 samples
        assert_relative_eq!(times[0], 0.0);
        assert_relative_eq!(times[4], 0.4);
    }

    #[test]
    fn test_angular_displacement_generation() {
        let psd_curve = PsdCurve::new(
            vec![
                PsdPoint {
                    frequency: 1.0,
                    amplitude_squared: 0.01,
                },
                PsdPoint {
                    frequency: 10.0,
                    amplitude_squared: 0.01,
                },
            ],
            Vector3::new(1.0, 0.0, 0.0),
        );
        let simulator = VibrationSimulator::new(vec![psd_curve], 100.0);

        let displacements =
            simulator.generate_angular_displacement(Duration::from_millis(100), None);

        assert_eq!(displacements.len(), 10); // 0.1s * 100Hz = 10 samples

        // Check that displacements are primarily along the x-axis
        let mean_abs_x =
            displacements.iter().map(|d| d.x.abs()).sum::<f64>() / displacements.len() as f64;
        let mean_abs_y =
            displacements.iter().map(|d| d.y.abs()).sum::<f64>() / displacements.len() as f64;
        let mean_abs_z =
            displacements.iter().map(|d| d.z.abs()).sum::<f64>() / displacements.len() as f64;

        assert!(mean_abs_x > mean_abs_y);
        assert!(mean_abs_x > mean_abs_z);
    }

    #[test]
    fn test_orientation_generation() {
        let psd_curve = PsdCurve::new(
            vec![
                PsdPoint {
                    frequency: 1.0,
                    amplitude_squared: 0.01,
                },
                PsdPoint {
                    frequency: 10.0,
                    amplitude_squared: 0.01,
                },
            ],
            Vector3::new(1.0, 0.0, 0.0),
        );
        let simulator = VibrationSimulator::new(vec![psd_curve], 100.0);

        let orientations = simulator.generate_orientations(Duration::from_millis(100), None);

        assert_eq!(orientations.len(), 10); // 0.1s * 100Hz = 10 samples

        // All quaternions should be unit quaternions
        for q in &orientations {
            assert_relative_eq!(q.as_vector().norm(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_orientation_deltas() {
        let psd_curve = PsdCurve::new(
            vec![
                PsdPoint {
                    frequency: 1.0,
                    amplitude_squared: 0.01,
                },
                PsdPoint {
                    frequency: 10.0,
                    amplitude_squared: 0.01,
                },
            ],
            Vector3::new(1.0, 0.0, 0.0),
        );
        let simulator = VibrationSimulator::new(vec![psd_curve], 100.0);

        let deltas = simulator.generate_orientation_deltas(Duration::from_millis(100), None);

        assert_eq!(deltas.len(), 10); // 0.1s * 100Hz = 10 samples

        // All deltas should be unit quaternions
        for q in &deltas {
            assert_relative_eq!(q.as_vector().norm(), 1.0, epsilon = 1e-10);
        }
    }

    mod validation_tests {
        use super::*;

        #[test]
        #[should_panic(expected = "PSD axis vector contains NaN or infinite values")]
        fn test_validate_psd_inputs_axis_nan() {
            let points = vec![PsdPoint {
                frequency: 1.0,
                amplitude_squared: 0.01,
            }];
            let axis = Vector3::new(f64::NAN, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        #[should_panic(expected = "PSD axis vector contains NaN or infinite values")]
        fn test_validate_psd_inputs_axis_inf() {
            let points = vec![PsdPoint {
                frequency: 1.0,
                amplitude_squared: 0.01,
            }];
            let axis = Vector3::new(f64::INFINITY, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        #[should_panic(expected = "PSD axis vector cannot be zero vector")]
        fn test_validate_psd_inputs_zero_axis() {
            let points = vec![PsdPoint {
                frequency: 1.0,
                amplitude_squared: 0.01,
            }];
            let axis = Vector3::new(0.0, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        #[should_panic(expected = "PSD point 0 has invalid frequency: NaN")]
        fn test_validate_psd_inputs_nan_frequency() {
            let points = vec![PsdPoint {
                frequency: f64::NAN,
                amplitude_squared: 0.01,
            }];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        #[should_panic(expected = "PSD point 0 has invalid frequency: inf")]
        fn test_validate_psd_inputs_inf_frequency() {
            let points = vec![PsdPoint {
                frequency: f64::INFINITY,
                amplitude_squared: 0.01,
            }];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        #[should_panic(expected = "PSD point 0 has negative frequency: -1")]
        fn test_validate_psd_inputs_negative_frequency() {
            let points = vec![PsdPoint {
                frequency: -1.0,
                amplitude_squared: 0.01,
            }];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        #[should_panic(expected = "PSD point 0 has invalid amplitude_squared: NaN")]
        fn test_validate_psd_inputs_nan_amplitude() {
            let points = vec![PsdPoint {
                frequency: 1.0,
                amplitude_squared: f64::NAN,
            }];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        #[should_panic(expected = "PSD point 0 has invalid amplitude_squared: inf")]
        fn test_validate_psd_inputs_inf_amplitude() {
            let points = vec![PsdPoint {
                frequency: 1.0,
                amplitude_squared: f64::INFINITY,
            }];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        #[should_panic(expected = "PSD point 0 has negative amplitude_squared: -0.01")]
        fn test_validate_psd_inputs_negative_amplitude() {
            let points = vec![PsdPoint {
                frequency: 1.0,
                amplitude_squared: -0.01,
            }];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        #[should_panic(expected = "PSD points are not sorted by frequency")]
        fn test_validate_psd_inputs_unsorted_frequencies() {
            let points = vec![
                PsdPoint {
                    frequency: 10.0,
                    amplitude_squared: 0.1,
                },
                PsdPoint {
                    frequency: 1.0,
                    amplitude_squared: 0.01,
                },
            ];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        #[should_panic(expected = "PSD points are not sorted by frequency")]
        fn test_validate_psd_inputs_duplicate_frequencies() {
            let points = vec![
                PsdPoint {
                    frequency: 1.0,
                    amplitude_squared: 0.01,
                },
                PsdPoint {
                    frequency: 1.0,
                    amplitude_squared: 0.02,
                },
            ];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        fn test_validate_psd_inputs_valid_data() {
            let points = vec![
                PsdPoint {
                    frequency: 1.0,
                    amplitude_squared: 0.01,
                },
                PsdPoint {
                    frequency: 10.0,
                    amplitude_squared: 0.1,
                },
            ];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            // Should not panic
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        fn test_validate_psd_inputs_empty_points() {
            let points = vec![];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            // Should not panic - empty points list is valid
            validate_psd_inputs(&points, &axis);
        }

        #[test]
        fn test_validate_psd_inputs_single_point() {
            let points = vec![PsdPoint {
                frequency: 1.0,
                amplitude_squared: 0.01,
            }];
            let axis = Vector3::new(1.0, 0.0, 0.0);
            // Should not panic - single point is trivially sorted
            validate_psd_inputs(&points, &axis);
        }
    }

    #[test]
    fn test_total_rms() {
        let psd_curve1 = PsdCurve::new(
            vec![
                PsdPoint {
                    frequency: 1.0,
                    amplitude_squared: 0.04,
                },
                PsdPoint {
                    frequency: 2.0,
                    amplitude_squared: 0.04,
                },
            ],
            Vector3::new(1.0, 0.0, 0.0),
        );

        let psd_curve2 = PsdCurve::new(
            vec![
                PsdPoint {
                    frequency: 1.0,
                    amplitude_squared: 0.09,
                },
                PsdPoint {
                    frequency: 2.0,
                    amplitude_squared: 0.09,
                },
            ],
            Vector3::new(0.0, 1.0, 0.0),
        );

        let simulator = VibrationSimulator::new(vec![psd_curve1, psd_curve2], 100.0);

        let total_rms = simulator.calculate_total_rms();

        // RMS for curve1 = 0.2 along x-axis, curve2 = 0.3 along y-axis
        assert_relative_eq!(total_rms.x, 0.2);
        assert_relative_eq!(total_rms.y, 0.3);
        assert_relative_eq!(total_rms.z, 0.0);
    }
}

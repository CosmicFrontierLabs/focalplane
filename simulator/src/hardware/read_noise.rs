//! Read noise estimation for sensor temperature and exposure parameters
//!
//! This module provides various strategies for estimating read noise values based on
//! temperature and exposure time. Read noise is a fundamental limitation in
//! electronic image sensors, representing the uncertainty in the measurement
//! of pixel values due to thermal fluctuations and electronic readout.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use meter_math::bilinear::{BilinearInterpolator, InterpolationError};

/// Error types for read noise estimation
#[derive(Debug, Clone, PartialEq)]
pub enum ReadNoiseError {
    /// Temperature is outside the valid range
    TemperatureOutOfBounds { value: f64, min: f64, max: f64 },
    /// Frame rate is outside the valid range
    FrameRateOutOfBounds { value: f64, min: f64, max: f64 },
}

impl std::fmt::Display for ReadNoiseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadNoiseError::TemperatureOutOfBounds { value, min, max } => {
                write!(
                    f,
                    "Temperature {value:.1}°C is outside valid range [{min:.1}°C, {max:.1}°C]"
                )
            }
            ReadNoiseError::FrameRateOutOfBounds { value, min, max } => {
                write!(
                    f,
                    "Frame rate {value:.1} Hz is outside valid range [{min:.1} Hz, {max:.1} Hz]"
                )
            }
        }
    }
}

impl std::error::Error for ReadNoiseError {}

/// Read noise estimator using bilinear interpolation over temperature and frame rate
///
/// Models read noise characteristics with:
/// - X-axis: Frame rate in Hz
/// - Y-axis: Temperature in degrees Celsius
/// - Output: Read noise in electrons RMS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadNoiseEstimator {
    /// Bilinear interpolator for noise values
    interpolator: BilinearInterpolator,
}

impl ReadNoiseEstimator {
    /// Access the underlying bilinear interpolator over
    /// (frame_rate_hz, temperature_c). Lets consumers introspect the
    /// calibration grid + noise surface via `BilinearInterpolator`'s
    /// `x_coords()` / `y_coords()` / `data()` accessors — e.g. to dump
    /// the full read-noise table into render metadata.
    pub fn interpolator(&self) -> &BilinearInterpolator {
        &self.interpolator
    }

    /// Create a constant read noise estimator that returns the same value regardless of conditions
    /// This is a model for sensors where the read noise is not a strong function of temperature
    /// or frame rate.
    pub fn constant(noise_value: f64) -> Self {
        // Only need corner points for constant value
        let frame_rates = vec![5.0, 1000.0];
        let temperatures = vec![-20.0, 20.0];
        let data = Array2::from_elem((2, 2), noise_value);

        let interpolator = BilinearInterpolator::new(frame_rates, temperatures, data)
            .expect("Failed to create constant noise interpolator");

        Self { interpolator }
    }

    /// Create a read noise estimator from a bilinear interpolator over
    /// (frame rate in Hz, temperature in °C) producing read noise in electrons RMS.
    pub fn from_interpolator(interpolator: BilinearInterpolator) -> Self {
        Self { interpolator }
    }

    /// Estimate read noise for given temperature and exposure time
    ///
    /// # Arguments
    /// * `temperature` - Sensor temperature in degrees Celsius
    /// * `exposure_time` - Integration time for the exposure
    ///
    /// # Returns
    /// * `Ok(f64)` - Read noise in electrons RMS per pixel
    /// * `Err(ReadNoiseError)` - If temperature or frame rate is outside calibration bounds
    ///
    /// # Units
    /// The returned value represents the RMS (root mean square) read noise in electrons.
    /// This is the standard deviation of the noise distribution added by the readout
    /// electronics, independent of photon shot noise or dark current.
    pub fn estimate(
        &self,
        temperature: f64,
        exposure_time: Duration,
    ) -> Result<f64, ReadNoiseError> {
        // Convert exposure time to frame rate (Hz = 1/seconds)
        let frame_rate = 1.0 / exposure_time.as_secs_f64();

        // At frame rates lower than 5Hz we use the 5Hz value
        let frame_rate = frame_rate.max(5.0);

        // Use bilinear interpolator
        match self.interpolator.interpolate(frame_rate, temperature) {
            Ok(value) => Ok(value),
            Err(InterpolationError::OutOfBounds {
                axis,
                value: _,
                min,
                max,
            }) => {
                if axis == "X" {
                    Err(ReadNoiseError::FrameRateOutOfBounds {
                        value: frame_rate,
                        min,
                        max,
                    })
                } else {
                    Err(ReadNoiseError::TemperatureOutOfBounds {
                        value: temperature,
                        min,
                        max,
                    })
                }
            }
            Err(_) => Err(ReadNoiseError::TemperatureOutOfBounds {
                value: temperature,
                min: -20.0,
                max: 20.0,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Representative (frame rate, temperature) calibration table used to exercise
    /// the generic interpolation, clamping, and bounds behaviour of `estimate()`.
    fn sample_estimator() -> ReadNoiseEstimator {
        let frame_rates = vec![5.0, 15.0, 30.0, 60.0, 120.0, 1000.0];
        let temperatures = vec![-20.0, 20.0];

        // Data indexed as [temp_idx, rate_idx]
        let data = Array2::from_shape_vec(
            (2, 6),
            vec![
                // -20°C row
                0.233, 0.263, 0.279, 0.334, 0.381, 0.381, // +20°C row
                0.301, 0.301, 0.305, 0.371, 0.404, 0.404,
            ],
        )
        .expect("Failed to create sample noise data");

        let interpolator = BilinearInterpolator::new(frame_rates, temperatures, data)
            .expect("Failed to create sample interpolator");

        ReadNoiseEstimator::from_interpolator(interpolator)
    }

    #[test]
    fn test_exact_corner_values() {
        let interp = sample_estimator();

        // Test exact corner values using Duration
        assert_eq!(
            interp
                .estimate(-20.0, Duration::from_secs_f64(1.0 / 5.0))
                .unwrap(),
            0.233
        );
        assert_eq!(
            interp
                .estimate(-20.0, Duration::from_secs_f64(1.0 / 1000.0))
                .unwrap(),
            0.381
        );
        assert_eq!(
            interp
                .estimate(20.0, Duration::from_secs_f64(1.0 / 5.0))
                .unwrap(),
            0.301
        );
        assert_eq!(
            interp
                .estimate(20.0, Duration::from_secs_f64(1.0 / 1000.0))
                .unwrap(),
            0.404
        );
    }

    #[test]
    fn test_temperature_out_of_bounds() {
        let interp = sample_estimator();

        // Test below minimum temperature
        let result = interp.estimate(-30.0, Duration::from_secs_f64(1.0 / 50.0));
        assert!(matches!(
            result,
            Err(ReadNoiseError::TemperatureOutOfBounds { value, min, max })
            if value == -30.0 && min == -20.0 && max == 20.0
        ));

        // Test above maximum temperature
        let result = interp.estimate(25.0, Duration::from_secs_f64(1.0 / 50.0));
        assert!(matches!(
            result,
            Err(ReadNoiseError::TemperatureOutOfBounds { value, min, max })
            if value == 25.0 && min == -20.0 && max == 20.0
        ));
    }

    #[test]
    fn test_frame_rate_out_of_bounds() {
        let interp = sample_estimator();

        // Test above maximum frame rate (too short exposure)
        let result = interp.estimate(0.0, Duration::from_secs_f64(1.0 / 1500.0));
        assert!(matches!(
            result,
            Err(ReadNoiseError::FrameRateOutOfBounds { value, min, max })
            if value > 1000.0 && value < 1501.0 && min == 5.0 && max == 1000.0
        ));
    }

    #[test]
    fn test_temperature_interpolation() {
        let interp = sample_estimator();

        // At 0°C (midpoint), at 30 fps, should interpolate between 0.279 and 0.305
        let result = interp
            .estimate(0.0, Duration::from_secs_f64(1.0 / 30.0))
            .unwrap();
        let expected = (0.279 + 0.305) / 2.0; // Average at midpoint
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_frame_rate_interpolation() {
        let interp = sample_estimator();

        // Test interpolation between frame rates at fixed temperature
        let result = interp
            .estimate(-20.0, Duration::from_secs_f64(1.0 / 20.0))
            .unwrap(); // 20 Hz between 15 and 30
        assert!(result > 0.263 && result < 0.279); // Should be between the two values

        // Test exact midpoint between 60 and 120 Hz
        let result = interp
            .estimate(-20.0, Duration::from_secs_f64(1.0 / 90.0))
            .unwrap(); // 90 Hz
        let expected = (0.334 + 0.381) / 2.0; // Average of 60Hz and 120Hz values
        assert_relative_eq!(result, expected, epsilon = 0.01);
    }

    #[test]
    fn test_bilinear_interpolation() {
        let interp = sample_estimator();

        // Test center point interpolation
        let result = interp
            .estimate(0.0, Duration::from_secs_f64(1.0 / 45.0))
            .unwrap(); // 45 Hz between 30 and 60

        // Should be between all four corner values
        let min_noise = 0.233; // Minimum in table
        let max_noise = 0.404; // Maximum in table
        assert!(result > min_noise && result < max_noise);
    }

    #[test]
    fn test_error_display() {
        let temp_err = ReadNoiseError::TemperatureOutOfBounds {
            value: -30.0,
            min: -20.0,
            max: 20.0,
        };
        assert_eq!(
            temp_err.to_string(),
            "Temperature -30.0°C is outside valid range [-20.0°C, 20.0°C]"
        );

        let rate_err = ReadNoiseError::FrameRateOutOfBounds {
            value: 250.0,
            min: 1.0,
            max: 200.0,
        };
        assert_eq!(
            rate_err.to_string(),
            "Frame rate 250.0 Hz is outside valid range [1.0 Hz, 200.0 Hz]"
        );
    }

    #[test]
    fn test_boundary_values_exact() {
        let interp = sample_estimator();

        // Test exact boundary values (should be OK)
        assert!(interp
            .estimate(-20.0, Duration::from_secs_f64(1.0 / 5.0))
            .is_ok());
        assert!(interp
            .estimate(-20.0, Duration::from_secs_f64(1.0 / 1000.0))
            .is_ok());
        assert!(interp
            .estimate(20.0, Duration::from_secs_f64(1.0 / 5.0))
            .is_ok());
        assert!(interp
            .estimate(20.0, Duration::from_secs_f64(1.0 / 1000.0))
            .is_ok());

        // Test just outside boundaries (should fail)
        assert!(interp
            .estimate(-20.1, Duration::from_secs_f64(1.0 / 100.0))
            .is_err());
        assert!(interp
            .estimate(20.1, Duration::from_secs_f64(1.0 / 100.0))
            .is_err());
        assert!(interp
            .estimate(0.0, Duration::from_secs_f64(1.0 / 1001.0))
            .is_err());
    }

    #[test]
    fn test_constant_read_noise() {
        let constant = ReadNoiseEstimator::constant(2.5);

        // Should always return same value regardless of parameters
        // Using exposure times that map to frame rates within 5-1000 Hz bounds
        assert_eq!(
            constant
                .estimate(-20.0, Duration::from_secs_f64(0.2))
                .unwrap(), // 5 Hz
            2.5
        );
        assert_eq!(
            constant
                .estimate(20.0, Duration::from_secs_f64(0.01))
                .unwrap(), // 100 Hz
            2.5
        );
        assert_eq!(
            constant
                .estimate(0.0, Duration::from_secs_f64(0.001))
                .unwrap(), // 1000 Hz
            2.5
        );
    }

    #[test]
    fn test_different_estimators() {
        // Test both an interpolated estimator and a constant one
        let varying = sample_estimator();
        let constant = ReadNoiseEstimator::constant(3.0);

        // The interpolated estimator should vary with conditions
        let cold = varying
            .estimate(-20.0, Duration::from_secs_f64(0.2))
            .unwrap();
        let hot = varying
            .estimate(20.0, Duration::from_secs_f64(0.2))
            .unwrap();
        assert_ne!(cold, hot);

        // Constant should always be same
        let const_cold = constant
            .estimate(-20.0, Duration::from_secs_f64(0.2))
            .unwrap();
        let const_hot = constant
            .estimate(20.0, Duration::from_secs_f64(0.2))
            .unwrap();
        assert_eq!(const_cold, const_hot);
        assert_eq!(const_cold, 3.0);
    }

    #[test]
    fn test_frame_rate_clamping_below_minimum() {
        // The slowest framerate in the table is 5 hz, which corresponds to a 200 ms exposure
        let interp = sample_estimator();
        // Test below minimum frame rate (too long exposure)
        let result_capped = interp.estimate(0.0, Duration::from_secs_f64(1.0)); // 1 Hz
        println!("Result: {result_capped:?}");
        let expected = interp.estimate(0.0, Duration::from_millis(200));

        assert_eq!(result_capped, expected);
    }
}

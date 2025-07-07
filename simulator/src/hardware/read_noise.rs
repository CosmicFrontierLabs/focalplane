//! Read noise estimation for sensor temperature and exposure parameters
//!
//! This module provides various strategies for estimating read noise values based on
//! temperature and exposure time. Read noise is a fundamental limitation in
//! electronic image sensors, representing the uncertainty in the measurement
//! of pixel values due to thermal fluctuations and electronic readout.

use ndarray::Array2;
use std::time::Duration;

use crate::algo::bilinear::{BilinearInterpolator, InterpolationError};

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
                    "Temperature {:.1}°C is outside valid range [{:.1}°C, {:.1}°C]",
                    value, min, max
                )
            }
            ReadNoiseError::FrameRateOutOfBounds { value, min, max } => {
                write!(
                    f,
                    "Frame rate {:.1} Hz is outside valid range [{:.1} Hz, {:.1} Hz]",
                    value, min, max
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
#[derive(Debug, Clone)]
pub struct ReadNoiseEstimator {
    /// Bilinear interpolator for noise values
    interpolator: BilinearInterpolator,
}

impl ReadNoiseEstimator {
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

    /// Create a new HWK4123 read noise estimator with factory calibration data
    /// Data is from here:
    /// - https://drive.google.com/file/d/1hhnfMxPQs3cXautEpjHpIDGkb_bArWdE
    /// - https://docs.google.com/spreadsheets/d/16WdFvMo3rj3Z9252pq32agsLV-wm7YNacvqfkEgSOAI
    pub fn hwk4123() -> Self {
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
        .expect("Failed to create HWK4123 noise data");

        let interpolator = BilinearInterpolator::new(frame_rates, temperatures, data)
            .expect("Failed to create HWK4123 interpolator");

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

    fn test_hwk4123() -> ReadNoiseEstimator {
        ReadNoiseEstimator::hwk4123()
    }

    #[test]
    fn test_exact_corner_values() {
        let interp = test_hwk4123();

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
        let interp = test_hwk4123();

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
        let interp = test_hwk4123();

        // Test below minimum frame rate (too long exposure)
        let result = interp.estimate(0.0, Duration::from_secs_f64(1.0)); // 1 Hz
        assert!(matches!(
            result,
            Err(ReadNoiseError::FrameRateOutOfBounds { value, min, max })
            if value == 1.0 && min == 5.0 && max == 1000.0
        ));

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
        let interp = test_hwk4123();

        // At 0°C (midpoint), at 30 fps, should interpolate between 0.279 and 0.305
        let result = interp
            .estimate(0.0, Duration::from_secs_f64(1.0 / 30.0))
            .unwrap();
        let expected = (0.279 + 0.305) / 2.0; // Average at midpoint
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_frame_rate_interpolation() {
        let interp = test_hwk4123();

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
        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_bilinear_interpolation() {
        let interp = test_hwk4123();

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
        let interp = test_hwk4123();

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
            .estimate(0.0, Duration::from_secs_f64(1.0 / 4.99))
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
        // Test both HWK4123 and constant estimators
        let hwk = ReadNoiseEstimator::hwk4123();
        let constant = ReadNoiseEstimator::constant(3.0);

        // HWK4123 should vary with conditions
        let hwk_cold = hwk.estimate(-20.0, Duration::from_secs_f64(0.2)).unwrap();
        let hwk_hot = hwk.estimate(20.0, Duration::from_secs_f64(0.2)).unwrap();
        assert_ne!(hwk_cold, hwk_hot);

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
}

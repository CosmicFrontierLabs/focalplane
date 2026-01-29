//! Configuration types for FSM calibration

use nalgebra::Matrix2;

/// Configuration for FSM axis calibration procedure
#[derive(Debug, Clone)]
pub struct FsmCalibrationConfig {
    /// Wiggle amplitude in microradians
    pub wiggle_amplitude_urad: f64,
    /// Wiggle frequency in Hz
    pub wiggle_frequency_hz: f64,
    /// Number of complete cycles per axis
    pub wiggle_cycles: usize,
    /// Verification circle radius in microradians
    pub verify_radius_urad: f64,
    /// Minimum acceptable R² for sinusoid fit
    pub min_fit_r_squared: f64,
}

impl Default for FsmCalibrationConfig {
    fn default() -> Self {
        Self {
            wiggle_amplitude_urad: 100.0,
            wiggle_frequency_hz: 1.0,
            wiggle_cycles: 5,
            verify_radius_urad: 150.0,
            min_fit_r_squared: 0.95,
        }
    }
}

/// Result of FSM axis calibration
#[derive(Debug, Clone)]
pub struct FsmAxisCalibration {
    /// Transform from FSM µrad to sensor pixels: [pixels/µrad]
    /// sensor_delta = fsm_to_sensor * fsm_command
    pub fsm_to_sensor: Matrix2<f64>,

    /// Transform from sensor pixels to FSM µrad: [µrad/pixel]
    /// fsm_command = sensor_to_fsm * sensor_delta
    pub sensor_to_fsm: Matrix2<f64>,

    /// Fit quality metric for axis 1 (0.0 to 1.0)
    pub axis1_r_squared: f64,

    /// Fit quality metric for axis 2 (0.0 to 1.0)
    pub axis2_r_squared: f64,

    /// Verification RMS error in pixels (populated after verification)
    pub verification_rms_error_pixels: Option<f64>,

    /// Verification max error in pixels (populated after verification)
    pub verification_max_error_pixels: Option<f64>,

    /// Configuration used for this calibration
    pub config: FsmCalibrationConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_default_config() {
        let config = FsmCalibrationConfig::default();

        assert_abs_diff_eq!(config.wiggle_amplitude_urad, 100.0, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(config.wiggle_frequency_hz, 1.0, epsilon = f64::EPSILON);
        assert_eq!(config.wiggle_cycles, 5);
        assert_abs_diff_eq!(config.verify_radius_urad, 150.0, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(config.min_fit_r_squared, 0.95, epsilon = f64::EPSILON);
    }
}

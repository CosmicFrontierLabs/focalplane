//! Configuration types for FSM calibration

use clap::Args;
use nalgebra::{Matrix2, Vector2};

/// Configuration for static step FSM calibration.
///
/// This struct can be embedded in CLI args via `#[command(flatten)]` or constructed
/// directly for programmatic use.
#[derive(Debug, Clone, Args)]
pub struct FsmCalibrationConfig {
    /// Center position in microradians (FSM range is typically 0-2000 µrad).
    /// This is auto-detected from the FSM travel range and set programmatically.
    #[arg(skip)]
    pub center_position_urad: f64,

    /// Total end-to-end swing range in microradians.
    #[arg(
        long,
        default_value = "200.0",
        help = "Total swing range in microradians",
        long_help = "Total end-to-end swing range for calibration steps. Steps are evenly \
            distributed across this range, centered on the center position. For example, \
            with swing_range=200 and num_steps=3, positions would be at center-100, \
            center, center+100. Typical range: 100-2000 µrad."
    )]
    pub swing_range_urad: f64,

    /// Number of steps per axis distributed across the swing range.
    #[arg(
        long,
        default_value = "3",
        help = "Number of step positions per axis",
        long_help = "Number of discrete positions to measure per axis. Positions are evenly \
            spaced across the swing range. More steps give better linearity assessment \
            but take longer. Minimum: 2, typical: 3-10."
    )]
    pub num_steps: usize,

    /// Maximum time in seconds to wait for tracking to reacquire after a move.
    #[arg(
        long,
        default_value = "5.0",
        help = "Maximum time to wait for tracking lock-on (seconds)"
    )]
    pub lock_on_time_secs: f64,

    /// Number of tracking messages to collect at each position.
    #[arg(
        long,
        default_value = "30",
        help = "Number of tracking samples per position"
    )]
    pub samples_per_position: usize,

    /// Number of initial samples to discard after lock-on (transient rejection).
    #[arg(
        long,
        default_value = "10",
        help = "Number of samples to discard after settling"
    )]
    pub discard_samples: usize,
}

impl Default for FsmCalibrationConfig {
    fn default() -> Self {
        Self {
            center_position_urad: 1000.0,
            swing_range_urad: 200.0,
            num_steps: 3,
            lock_on_time_secs: 5.0,
            samples_per_position: 30,
            discard_samples: 10,
        }
    }
}

/// Result of FSM axis calibration
///
/// The full transform model is:
///   sensor_position = fsm_to_sensor * fsm_command + intercept_pixels
///
/// Where:
///   - fsm_command is in µrad (absolute position in 0-2000 range)
///   - sensor_position is in pixels
///   - intercept_pixels is the centroid position when FSM is at center_position
#[derive(Debug, Clone)]
pub struct FsmAxisCalibration {
    /// Transform from FSM µrad to sensor pixels: [pixels/µrad]
    /// sensor_delta = fsm_to_sensor * fsm_command
    pub fsm_to_sensor: Matrix2<f64>,

    /// Transform from sensor pixels to FSM µrad: [µrad/pixel]
    /// fsm_command = sensor_to_fsm * sensor_delta
    pub sensor_to_fsm: Matrix2<f64>,

    /// Intercept: sensor position when FSM is at center position
    /// This is the centroid position measured at the configured center_position_urad
    pub intercept_pixels: Vector2<f64>,

    /// Mean standard deviation in centroid measurements for axis 1
    pub axis1_std_pixels: f64,

    /// Mean standard deviation in centroid measurements for axis 2
    pub axis2_std_pixels: f64,

    /// Configuration used for this calibration
    pub config: FsmCalibrationConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_default_config() {
        let config = FsmCalibrationConfig::default();

        assert_relative_eq!(config.center_position_urad, 1000.0);
        assert_relative_eq!(config.swing_range_urad, 200.0);
        assert_eq!(config.samples_per_position, 30);
        assert_eq!(config.discard_samples, 10);
    }
}

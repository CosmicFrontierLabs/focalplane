//! System information types for auto-discovery.
//!
//! These types allow calibration tools to discover display and sensor
//! properties from running services without manual configuration.

use serde::{Deserialize, Serialize};

/// Display system information from calibrate_serve.
///
/// Returned via HTTP GET /info endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayInfo {
    /// Display width in pixels
    pub width: u32,
    /// Display height in pixels
    pub height: u32,
    /// Pixel pitch in microns (None if unknown/unavailable)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub pixel_pitch_um: Option<f64>,
    /// Display name/identifier (e.g., "OLED 2560x2560")
    pub name: String,
}

/// Sensor system information from cam_track.
///
/// Included as optional field in TrackingMessage, sent on first
/// message and periodically thereafter for late joiners.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorInfo {
    /// Sensor width in pixels
    pub width: u32,
    /// Sensor height in pixels
    pub height: u32,
    /// Pixel pitch in microns
    pub pixel_pitch_um: f64,
    /// Camera/sensor name (e.g., "IMX455")
    pub name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_display_info_serialization() {
        let info = DisplayInfo {
            width: 2560,
            height: 2560,
            pixel_pitch_um: Some(9.6),
            name: "OLED".to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("pixel_pitch_um"));
        let parsed: DisplayInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.width, 2560);
        assert_eq!(parsed.height, 2560);
        assert_abs_diff_eq!(parsed.pixel_pitch_um.unwrap(), 9.6, epsilon = 1e-10);
        assert_eq!(parsed.name, "OLED");
    }

    #[test]
    fn test_display_info_without_pixel_pitch() {
        let info = DisplayInfo {
            width: 1920,
            height: 1080,
            pixel_pitch_um: None,
            name: "Unknown Display".to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        // pixel_pitch_um should be skipped when None
        assert!(!json.contains("pixel_pitch_um"));

        let parsed: DisplayInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.width, 1920);
        assert!(parsed.pixel_pitch_um.is_none());
    }

    #[test]
    fn test_sensor_info_serialization() {
        let info = SensorInfo {
            width: 9576,
            height: 6388,
            pixel_pitch_um: 3.76,
            name: "IMX455".to_string(),
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: SensorInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.width, 9576);
        assert_eq!(parsed.height, 6388);
        assert_abs_diff_eq!(parsed.pixel_pitch_um, 3.76, epsilon = 1e-10);
        assert_eq!(parsed.name, "IMX455");
    }
}

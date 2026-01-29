//! Tracking message types for inter-process communication.
//!
//! Used for communicating tracked target positions between processes
//! (e.g., cam_track publishing to calibration subscribers).

use serde::{Deserialize, Serialize};
use test_bench_shared::{SpotShape, Timestamp};

use crate::system_info::SensorInfo;

/// A tracking update message containing the position of a tracked target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingMessage {
    /// Unique identifier for this track
    pub track_id: u32,
    /// X position in sensor coordinates (pixels)
    pub x: f64,
    /// Y position in sensor coordinates (pixels)
    pub y: f64,
    /// Timestamp when this measurement was taken
    pub timestamp: Timestamp,
    /// Spot shape characterization (flux, moments, diameter).
    /// Used for defocus mapping, PSF characterization, and radiometric calibration.
    pub shape: SpotShape,
    /// Sensor info - included on first message and periodically for late joiners.
    /// Allows clients to auto-discover sensor dimensions without manual config.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub sensor_info: Option<SensorInfo>,
}

impl TrackingMessage {
    /// Create a new tracking message with position and shape data.
    pub fn new(track_id: u32, x: f64, y: f64, timestamp: Timestamp, shape: SpotShape) -> Self {
        Self {
            track_id,
            x,
            y,
            timestamp,
            shape,
            sensor_info: None,
        }
    }

    /// Create a tracking message with sensor info attached.
    pub fn with_sensor_info(
        track_id: u32,
        x: f64,
        y: f64,
        timestamp: Timestamp,
        shape: SpotShape,
        sensor_info: SensorInfo,
    ) -> Self {
        Self {
            track_id,
            x,
            y,
            timestamp,
            shape,
            sensor_info: Some(sensor_info),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_shape() -> SpotShape {
        SpotShape {
            flux: 42000.0,
            m_xx: 2.5,
            m_yy: 3.0,
            m_xy: 0.1,
            aspect_ratio: 1.2,
            diameter: 5.0,
        }
    }

    #[test]
    fn test_tracking_message_serialization() {
        let shape = make_test_shape();
        let msg = TrackingMessage::new(1, 100.5, 200.5, Timestamp::new(12345, 6789), shape);

        assert_eq!(msg.track_id, 1);
        assert_relative_eq!(msg.x, 100.5);
        assert_relative_eq!(msg.y, 200.5);
        assert_eq!(msg.timestamp.seconds, 12345);
        assert_relative_eq!(msg.shape.flux, 42000.0);
        assert_relative_eq!(msg.shape.diameter, 5.0);
        assert!(msg.sensor_info.is_none());

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("shape"));
        assert!(json.contains("flux"));
        assert!(json.contains("diameter"));
        // sensor_info should be skipped when None
        assert!(!json.contains("sensor_info"));

        let parsed: TrackingMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.track_id, 1);
        assert_relative_eq!(parsed.shape.flux, 42000.0);
        assert_relative_eq!(parsed.shape.diameter, 5.0);
        assert!(parsed.sensor_info.is_none());
    }

    #[test]
    fn test_tracking_message_with_sensor_info() {
        let shape = make_test_shape();
        let sensor_info = SensorInfo {
            width: 9576,
            height: 6388,
            pixel_pitch_um: 3.76,
            name: "IMX455".to_string(),
        };
        let msg = TrackingMessage::with_sensor_info(
            1,
            100.5,
            200.5,
            Timestamp::new(12345, 6789),
            shape,
            sensor_info,
        );

        assert!(msg.sensor_info.is_some());
        let info = msg.sensor_info.as_ref().unwrap();
        assert_eq!(info.width, 9576);
        assert_eq!(info.height, 6388);

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("sensor_info"));
        assert!(json.contains("pixel_pitch_um"));

        let parsed: TrackingMessage = serde_json::from_str(&json).unwrap();
        assert!(parsed.sensor_info.is_some());
        let parsed_info = parsed.sensor_info.unwrap();
        assert_eq!(parsed_info.width, 9576);
        assert_eq!(parsed_info.name, "IMX455");
    }
}

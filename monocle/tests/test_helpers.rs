use ndarray::Array2;
use shared::camera_interface::{mock::MockCameraInterface, CameraConfig, Timestamp};
use std::time::Duration;

/// Helper function to create a test timestamp
pub fn test_timestamp() -> Timestamp {
    Timestamp::from_duration(Duration::from_millis(100))
}

/// Helper function to create a test timestamp with specific milliseconds
pub fn test_timestamp_ms(ms: u64) -> Timestamp {
    Timestamp::from_duration(Duration::from_millis(ms))
}

/// Helper function to create a mock camera with a repeating frame
pub fn create_mock_camera(frame: Array2<u16>) -> MockCameraInterface {
    let (height, width) = frame.dim();
    let config = CameraConfig {
        width,
        height,
        exposure: Duration::from_millis(10),
        bit_depth: 16,
    };
    MockCameraInterface::new_repeating(config, frame)
}

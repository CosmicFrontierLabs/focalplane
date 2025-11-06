use ndarray::Array2;
use shared::camera_interface::{mock::MockCameraInterface, CameraInterface, Timestamp};
use std::time::Duration;

/// Helper function to create a test timestamp
pub fn test_timestamp() -> Timestamp {
    Timestamp::from_duration(Duration::from_millis(100))
}

/// Helper function to create a mock camera with a repeating frame
pub fn create_mock_camera(frame: Array2<u16>) -> MockCameraInterface {
    let (height, width) = frame.dim();
    let mut camera = MockCameraInterface::new_repeating((width, height), 16, frame);
    camera.set_exposure(Duration::from_millis(10)).unwrap();
    camera
}

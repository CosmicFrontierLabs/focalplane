use shared::camera_interface::Timestamp;
use std::time::Duration;

/// Helper function to create a test timestamp
pub fn test_timestamp() -> Timestamp {
    Timestamp::from_duration(Duration::from_millis(100))
}

/// Helper function to create a test timestamp with specific milliseconds
pub fn test_timestamp_ms(ms: u64) -> Timestamp {
    Timestamp::from_duration(Duration::from_millis(ms))
}

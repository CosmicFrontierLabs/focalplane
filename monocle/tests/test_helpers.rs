use std::time::Duration;
use test_bench_shared::Timestamp;

/// Helper function to create a test timestamp
pub fn test_timestamp() -> Timestamp {
    Timestamp::from_duration(Duration::from_millis(100))
}

//! Communication utilities for sensor discovery.

use std::time::{Duration, Instant};

use crate::tracking_collector::TrackingCollector;
use shared::system_info::SensorInfo;

/// Wait for sensor info from tracking messages.
///
/// Polls the tracking collector for up to `timeout` duration to receive a
/// TrackingMessage with sensor_info. Returns the discovered SensorInfo
/// or None if no sensor info was received within the timeout.
pub fn discover_sensor_info(
    collector: &TrackingCollector,
    timeout: Duration,
) -> Option<SensorInfo> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        match collector.poll() {
            Ok(msgs) => {
                for msg in msgs {
                    if let Some(info) = msg.sensor_info {
                        return Some(info);
                    }
                }
            }
            Err(_) => return None,
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    None
}

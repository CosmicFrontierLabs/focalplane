//! Communication utilities for ZMQ and HTTP pattern control.

use std::time::{Duration, Instant};

use shared::pattern_command::PatternCommand;
use shared::system_info::SensorInfo;
use shared::tracking_collector::TrackingCollector;

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
        for msg in collector.poll() {
            if let Some(info) = msg.sensor_info {
                return Some(info);
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    None
}

/// Set the pattern to RemoteControlled mode via HTTP API.
///
/// This must be called before ZMQ pattern commands will have any effect.
/// The display server needs to be in RemoteControlled mode to accept
/// external pattern commands.
pub fn enable_remote_controlled_mode(http_endpoint: &str) -> Result<(), String> {
    let url = format!("{http_endpoint}/config");
    let body = r#"{"pattern_id": "RemoteControlled", "values": {}, "invert": false}"#;

    let response = ureq::post(&url)
        .header("Content-Type", "application/json")
        .send(body.as_bytes())
        .map_err(|e| format!("Failed to set RemoteControlled mode: {e}"))?;

    if response.status() != 200 {
        return Err(format!(
            "Failed to set RemoteControlled mode: HTTP {}",
            response.status()
        ));
    }

    Ok(())
}

/// Send a pattern command and wait for acknowledgment.
///
/// Sends the command via ZMQ REQ socket and waits for the reply.
/// Returns an error if the send fails, receive times out, or server
/// returns an error response.
pub fn send_pattern_command(socket: &zmq::Socket, cmd: &PatternCommand) -> Result<(), String> {
    let json =
        serde_json::to_string(cmd).map_err(|e| format!("Failed to serialize command: {e}"))?;
    socket
        .send(&json, 0)
        .map_err(|e| format!("Failed to send command: {e}"))?;

    // Wait for reply
    let reply = socket
        .recv_string(0)
        .map_err(|e| format!("Failed to receive reply: {e}"))?
        .map_err(|_| "Reply was not valid UTF-8".to_string())?;

    if reply.starts_with("error") {
        return Err(format!("Server error: {reply}"));
    }

    Ok(())
}

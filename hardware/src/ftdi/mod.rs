//! FTDI device utilities and mock gyro packet generation
//!
//! Provides helpers for working with FTDI devices and generating
//! Exail Asterix NS gyro packets for hardware-in-the-loop testing.

mod mock_gyro_packets;

pub use mock_gyro_packets::{build_full_packet, build_raw_packet, HEALTH_OK};

use anyhow::{Context, Result};
use libftd2xx::{list_devices, list_devices_fs, DeviceInfo};

/// List connected FTDI devices.
///
/// Tries filesystem-based listing first (more reliable on Linux),
/// then falls back to D2XX API.
pub fn list_ftdi_devices() -> Result<Vec<DeviceInfo>> {
    // Try filesystem-based listing first (more reliable on Linux)
    if let Ok(devices) = list_devices_fs() {
        if !devices.is_empty() {
            return Ok(devices);
        }
    }
    // Fall back to D2XX API
    let devices = list_devices().context("Failed to list FTDI devices")?;
    Ok(devices)
}

/// Print device info to stdout in a formatted way.
pub fn print_device_info(index: usize, info: &DeviceInfo) {
    println!(
        "  [{}] Serial: {:?}, Description: {:?}, Type: {:?}, VID:PID: {:04x}:{:04x}, Open: {}",
        index,
        info.serial_number,
        info.description,
        info.device_type,
        info.vendor_id,
        info.product_id,
        info.port_open
    );
}

//! FTDI device utilities and mock gyro packet generation
//!
//! Provides helpers for working with FTDI devices and generating
//! Exail Asterix NS gyro packets for hardware-in-the-loop testing.

#[cfg(feature = "exail")]
mod mock_gyro_packets;

#[cfg(feature = "exail")]
pub use mock_gyro_packets::{build_full_packet, build_raw_packet, HEALTH_OK};

use anyhow::{Context, Result};
use clap::Args;
use libftd2xx::{list_devices, list_devices_fs, DeviceInfo, Ftdi, FtdiCommon};
use std::time::Duration;

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

/// Shared CLI arguments for FTDI device selection.
///
/// Use with `#[command(flatten)]` in clap structs to add standard FTDI options.
///
/// # Example
/// ```ignore
/// use clap::Parser;
/// use hardware::ftdi::FtdiArgs;
///
/// #[derive(Parser)]
/// struct Args {
///     #[command(flatten)]
///     ftdi: FtdiArgs,
/// }
/// ```
#[derive(Args, Debug, Clone)]
pub struct FtdiArgs {
    /// List available FTDI devices and exit
    #[arg(long, help = "List available FTDI devices and exit")]
    pub list_ftdi: bool,

    /// FTDI device index (0 = first device)
    #[arg(
        long,
        help = "FTDI device index (0 = first device)",
        long_help = "Index of the FTDI device to use. Use --list-ftdi to see available \
            devices. If not specified, FTDI functionality may be disabled depending \
            on the application."
    )]
    pub ftdi_device: Option<i32>,

    /// Baud rate for FTDI communication
    #[arg(long, default_value = "1000000", help = "Baud rate in bits per second")]
    pub ftdi_baud: u32,
}

impl FtdiArgs {
    /// Handle --list-ftdi flag by printing devices and returning true if listing was requested.
    ///
    /// Call this early in main() to exit after listing:
    /// ```ignore
    /// if args.ftdi.handle_list_ftdi()? {
    ///     return Ok(());
    /// }
    /// ```
    pub fn handle_list_ftdi(&self) -> Result<bool> {
        if self.list_ftdi {
            self.print_devices()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Print available FTDI devices to stdout.
    pub fn print_devices(&self) -> Result<()> {
        let devices = list_ftdi_devices()?;
        println!("Found {} FTDI device(s):", devices.len());
        for (i, dev) in devices.iter().enumerate() {
            print_device_info(i, dev);
        }
        Ok(())
    }

    /// Open the selected FTDI device with standard configuration.
    ///
    /// Returns an error if no device index is specified or device not found.
    pub fn open_device(&self) -> Result<Ftdi> {
        let index = self
            .ftdi_device
            .ok_or_else(|| anyhow::anyhow!("No FTDI device specified (use --ftdi-device)"))?;

        let devices = list_ftdi_devices()?;

        if devices.is_empty() {
            anyhow::bail!("No FTDI devices found");
        }

        if index as usize >= devices.len() {
            anyhow::bail!(
                "FTDI device index {} out of range (found {} devices)",
                index,
                devices.len()
            );
        }

        let selected = &devices[index as usize];
        log::info!(
            "Opening FTDI device {} - Serial: {:?}, Description: {:?}",
            index,
            selected.serial_number,
            selected.description
        );

        let mut ftdi = Ftdi::with_index(index).context("Failed to open FTDI device")?;

        log::info!("Setting baud rate to {} bps", self.ftdi_baud);
        ftdi.set_baud_rate(self.ftdi_baud)
            .context("Failed to set baud rate")?;

        log::info!("Configuring for low latency");
        ftdi.set_latency_timer(Duration::from_millis(1))
            .context("Failed to set latency timer")?;
        ftdi.set_usb_parameters(64)
            .context("Failed to set USB parameters")?;
        ftdi.set_timeouts(Duration::from_millis(100), Duration::from_millis(100))
            .context("Failed to set timeouts")?;

        Ok(ftdi)
    }

    /// Check if FTDI device is configured (index specified).
    pub fn is_configured(&self) -> bool {
        self.ftdi_device.is_some()
    }

    /// Open the selected FTDI device, defaulting to device 0 if not specified.
    ///
    /// Use this for binaries where FTDI is the primary input and device 0
    /// is a reasonable default.
    pub fn open_device_or_default(&self) -> Result<Ftdi> {
        let index = self.ftdi_device.unwrap_or(0);

        let devices = list_ftdi_devices()?;

        if devices.is_empty() {
            anyhow::bail!("No FTDI devices found");
        }

        if index as usize >= devices.len() {
            anyhow::bail!(
                "FTDI device index {} out of range (found {} devices)",
                index,
                devices.len()
            );
        }

        let selected = &devices[index as usize];
        log::info!(
            "Opening FTDI device {} - Serial: {:?}, Description: {:?}",
            index,
            selected.serial_number,
            selected.description
        );

        let mut ftdi = Ftdi::with_index(index).context("Failed to open FTDI device")?;

        log::info!("Setting baud rate to {} bps", self.ftdi_baud);
        ftdi.set_baud_rate(self.ftdi_baud)
            .context("Failed to set baud rate")?;

        log::info!("Configuring for low latency");
        ftdi.set_latency_timer(Duration::from_millis(1))
            .context("Failed to set latency timer")?;
        ftdi.set_usb_parameters(64)
            .context("Failed to set USB parameters")?;
        ftdi.set_timeouts(Duration::from_millis(100), Duration::from_millis(100))
            .context("Failed to set timeouts")?;

        Ok(ftdi)
    }
}

/// Get formatted list of FTDI devices for display purposes.
pub fn list_ftdi_devices_info() -> Result<Vec<String>> {
    let devices = list_ftdi_devices()?;
    Ok(devices
        .iter()
        .enumerate()
        .map(|(i, dev)| {
            format!(
                "{}: Serial={:?}, Description={:?}",
                i, dev.serial_number, dev.description
            )
        })
        .collect())
}

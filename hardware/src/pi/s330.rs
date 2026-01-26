//! PI S-330 Fast Steering Mirror Driver
//!
//! This module provides a high-level interface to the PI S-330 fast steering mirror,
//! which is controlled via an E-727 digital piezo controller.
//!
//! The S-330 is a 2-axis tip/tilt mirror with piezo actuators. This driver wraps
//! the E727 controller and provides S-330-specific initialization and configuration.

use std::time::{Duration, Instant};

use tracing::info;

use super::e727::{Axis, SpaParam, E727};
use super::gcs::{GcsResult, DEFAULT_PORT};

/// Polling interval when waiting for voltage changes.
const VOLTAGE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Timeout for axis 3 bias voltage to reach target.
const BIAS_VOLTAGE_TIMEOUT: Duration = Duration::from_secs(2);

/// Timeout for piezo voltages to reach 0V during shutdown.
const SHUTDOWN_VOLTAGE_TIMEOUT: Duration = Duration::from_secs(5);

/// PI S-330 Fast Steering Mirror driver.
///
/// Wraps an E727 controller with S-330-specific initialization and cleanup.
/// By default, servos are disabled on drop for safety.
pub struct S330 {
    e727: E727,
    poweroff_on_drop: bool,
}

impl S330 {
    /// Connect to an S-330 via E-727 controller at the given IP.
    ///
    /// Uses the default GCS port (50000).
    pub fn connect_ip(ip: &str) -> GcsResult<Self> {
        Self::connect(ip, DEFAULT_PORT)
    }

    /// Connect to an S-330 via E-727 controller at the given IP and port.
    pub fn connect(ip: &str, port: u16) -> GcsResult<Self> {
        let e727 = E727::connect(format!("{ip}:{port}"))?;
        let mut s330 = Self {
            e727,
            poweroff_on_drop: true,
        };
        s330.init()?;
        Ok(s330)
    }

    /// Initialize the S-330 for operation.
    ///
    /// Checks that axis 3 piezo driving parameters are correctly configured.
    /// If not, runs the shutdown sequence to safely power down, then sets
    /// the correct parameters.
    fn init(&mut self) -> GcsResult<()> {
        info!("Initializing S-330...");

        // Check axis 3 piezo driving parameters
        // Expected: piezo1=0, piezo2=0, piezo3=1
        let expected_params: [(SpaParam, f64); 3] = [
            (SpaParam::DrivingFactorPiezo1, 0.0),
            (SpaParam::DrivingFactorPiezo2, 0.0),
            (SpaParam::DrivingFactorPiezo3, 1.0),
        ];

        let mut params_correct = true;
        for (param, expected) in &expected_params {
            let value = self.e727.get_param(Axis::Axis3, *param)?;
            // Use tolerance for float comparison (values like -0.02 should match 0)
            if (value - expected).abs() > 0.1 {
                tracing::warn!(
                    "SPA 3 {param} (0x{:08X}): was {value}, expected {expected}",
                    param.address()
                );
                params_correct = false;
            }
        }

        if !params_correct {
            // Run shutdown sequence to safely depower before changing params
            self.shutdown_sequence()?;

            // Elevate command level to 1 for parameter write access
            info!("Setting command level 1 for parameter access...");
            self.e727.device_mut().command("CCL 1 advanced")?;

            // Set correct parameters
            for (param, expected) in &expected_params {
                info!(
                    "Setting SPA 3 {param} (0x{:08X}) = {expected}",
                    param.address()
                );
                self.e727.set_param(Axis::Axis3, *param, *expected)?;
            }

            // Restore command level to 0
            info!("Restoring command level 0...");
            self.e727.device_mut().command("CCL 0")?;

            tracing::warn!("S-330 axis 3 parameters have been reset");
        } else {
            info!("S-330 axis 3 parameters verified OK");
        }

        // Check axis 3 bias voltage (should be 100V)
        let axis3_voltage = self.get_axis_voltage(Axis::Axis3)?;
        if (axis3_voltage - 100.0).abs() > 1.0 {
            info!("Setting axis 3 bias voltage to 100V (currently {axis3_voltage:.1}V)...");
            self.e727.device_mut().command("SVA 3 100")?;

            // Poll until voltage reaches 100V
            let start = Instant::now();
            loop {
                if start.elapsed() > BIAS_VOLTAGE_TIMEOUT {
                    tracing::warn!("Timeout waiting for axis 3 voltage to reach 100V");
                    break;
                }

                let voltage = self.get_axis_voltage(Axis::Axis3)?;
                if (voltage - 100.0).abs() < 1.0 {
                    info!("Axis 3 bias voltage at {voltage:.1}V");
                    break;
                }

                std::thread::sleep(VOLTAGE_POLL_INTERVAL);
            }
        } else {
            info!("Axis 3 bias voltage OK ({axis3_voltage:.1}V)");
        }

        // Check if axes 1 and 2 need autozero
        let axis1_zeroed = self.e727.is_autozeroed(Axis::Axis1)?;
        let axis2_zeroed = self.e727.is_autozeroed(Axis::Axis2)?;

        if !axis1_zeroed || !axis2_zeroed {
            info!("Performing autozero on axes 1 and 2...");
            self.e727.autozero(&[Axis::Axis1, Axis::Axis2], true)?;
        } else {
            info!("Axes 1 and 2 already autozeroed");
        }

        // Enable servos on axes 1 and 2
        info!("Enabling servos on axes 1 and 2...");
        self.e727.set_servo(Axis::Axis1, true)?;
        self.e727.set_servo(Axis::Axis2, true)?;

        info!("S-330 initialization complete");
        Ok(())
    }

    /// Get the current voltage for an axis.
    fn get_axis_voltage(&mut self, axis: Axis) -> GcsResult<f64> {
        self.e727.get_voltage(axis)
    }

    /// Get a reference to the underlying E727 controller.
    pub fn e727(&self) -> &E727 {
        &self.e727
    }

    /// Get a mutable reference to the underlying E727 controller.
    pub fn e727_mut(&mut self) -> &mut E727 {
        &mut self.e727
    }

    /// Set whether to disable servos when the S330 is dropped.
    ///
    /// Default is `true` for safety. Set to `false` if you want the mirror
    /// to maintain its position after the driver is dropped.
    pub fn set_poweroff_on_drop(&mut self, poweroff: bool) {
        self.poweroff_on_drop = poweroff;
    }

    // ==================== Motion Commands ====================

    /// Move both tilt axes (1 and 2) to absolute positions.
    ///
    /// # Arguments
    /// * `axis1` - Target position for axis 1 in µrad (microradians)
    /// * `axis2` - Target position for axis 2 in µrad (microradians)
    ///
    /// This is a fast move command without error checking, suitable for
    /// high-speed motion loops.
    pub fn move_to(&mut self, axis1: f64, axis2: f64) -> GcsResult<()> {
        self.e727.move_to_fast(Some(axis1), Some(axis2), None, None)
    }

    /// Get current positions of both tilt axes.
    ///
    /// Returns `(axis1, axis2)` positions in µrad (microradians).
    pub fn get_position(&mut self) -> GcsResult<(f64, f64)> {
        let axis1 = self.e727.get_position(Axis::Axis1)?;
        let axis2 = self.e727.get_position(Axis::Axis2)?;
        Ok((axis1, axis2))
    }

    /// Get the center positions (midpoint of travel range) for both tilt axes.
    ///
    /// Returns `(center_axis1, center_axis2)` in µrad (microradians).
    pub fn get_centers(&mut self) -> GcsResult<(f64, f64)> {
        self.e727.get_xy_centers()
    }

    /// Get the travel range for both tilt axes.
    ///
    /// Returns `((min_axis1, max_axis1), (min_axis2, max_axis2))` in µrad (microradians).
    pub fn get_travel_ranges(&mut self) -> GcsResult<((f64, f64), (f64, f64))> {
        let range_axis1 = self.e727.get_travel_range(Axis::Axis1)?;
        let range_axis2 = self.e727.get_travel_range(Axis::Axis2)?;
        Ok((range_axis1, range_axis2))
    }

    /// Get the physical unit string for the tilt axes (typically "µrad").
    pub fn get_unit(&mut self) -> GcsResult<String> {
        self.e727.get_unit(Axis::Axis1)
    }
}

impl Drop for S330 {
    fn drop(&mut self) {
        if self.poweroff_on_drop {
            if let Err(e) = self.shutdown_sequence() {
                tracing::warn!("S-330 shutdown sequence failed: {}", e);
            }
        }
    }
}

impl S330 {
    /// Execute the S-330 shutdown sequence.
    ///
    /// This performs a safe shutdown:
    /// 1. Disable servo mode for axes 1 and 2
    /// 2. Set piezo output voltage to 0V for axes 1 and 2
    /// 3. Wait for voltages to reach 0V
    /// 4. Set fixed voltage to 0V for axis 3
    fn shutdown_sequence(&mut self) -> GcsResult<()> {
        info!("S-330 shutdown sequence starting...");

        // a. Disable servo mode for axes 1 and 2
        info!("Disabling servos on axes 1 and 2...");
        self.e727.set_servo(Axis::Axis1, false)?;
        self.e727.set_servo(Axis::Axis2, false)?;

        // b. Set piezo output voltage to 0V for axes 1 and 2
        info!("Setting piezo voltage to 0V on axes 1 and 2...");
        self.e727.set_voltage(Axis::Axis1, 0.0)?;
        self.e727.set_voltage(Axis::Axis2, 0.0)?;

        // c. Wait for piezo output voltages to reach 0V
        info!("Waiting for piezo voltages to settle...");
        let start = Instant::now();

        loop {
            if start.elapsed() > SHUTDOWN_VOLTAGE_TIMEOUT {
                tracing::warn!("Timeout waiting for piezo voltages to reach 0V, continuing...");
                break;
            }

            let v1 = self.e727.get_voltage(Axis::Axis1)?;
            let v2 = self.e727.get_voltage(Axis::Axis2)?;

            if v1.abs() < 0.1 && v2.abs() < 0.1 {
                info!("Piezo voltages at 0V");
                break;
            }

            std::thread::sleep(VOLTAGE_POLL_INTERVAL);
        }

        // d. Set fixed voltage to 0V for axis 3
        info!("Setting axis 3 voltage to 0V...");
        self.e727.set_voltage(Axis::Axis3, 0.0)?;

        info!("S-330 shutdown sequence complete");
        Ok(())
    }
}

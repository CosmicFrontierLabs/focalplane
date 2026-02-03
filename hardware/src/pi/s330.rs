//! PI S-330 Fast Steering Mirror Driver
//!
//! This module provides a high-level interface to the PI S-330 fast steering mirror,
//! which is controlled via an E-727 digital piezo controller.
//!
//! The S-330 is a 2-axis tip/tilt mirror with piezo actuators. This driver wraps
//! the E727 controller and provides S-330-specific initialization and configuration.

use std::cell::OnceCell;
use std::time::{Duration, Instant};

use clap::Args;
use tracing::info;

use super::e727::{Axis, SpaParam, E727};
use super::gcs::{GcsResult, DEFAULT_FSM_IP, DEFAULT_PORT};

/// Polling interval when waiting for voltage changes.
const VOLTAGE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Timeout for axis 3 bias voltage to reach target.
const BIAS_VOLTAGE_TIMEOUT: Duration = Duration::from_secs(2);

/// Timeout for piezo voltages to reach 0V during shutdown.
const SHUTDOWN_VOLTAGE_TIMEOUT: Duration = Duration::from_secs(5);

/// Duration for slewing to center during shutdown.
const SHUTDOWN_SLEW_DURATION: Duration = Duration::from_millis(250);

/// Interval between position commands during shutdown slew.
const SHUTDOWN_SLEW_INTERVAL: Duration = Duration::from_millis(10);

/// Result of a clamped move operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MoveResult {
    /// Position was within limits, moved as requested.
    Ok { x: f64, y: f64 },
    /// Position was clamped to travel limits.
    Clamped {
        requested_x: f64,
        requested_y: f64,
        actual_x: f64,
        actual_y: f64,
    },
}

impl MoveResult {
    /// Get the actual position that was commanded.
    pub fn position(&self) -> (f64, f64) {
        match self {
            MoveResult::Ok { x, y } => (*x, *y),
            MoveResult::Clamped {
                actual_x, actual_y, ..
            } => (*actual_x, *actual_y),
        }
    }

    /// Check if the move was clamped.
    pub fn was_clamped(&self) -> bool {
        matches!(self, MoveResult::Clamped { .. })
    }
}

/// PI S-330 Fast Steering Mirror driver.
///
/// Wraps an E727 controller with S-330-specific initialization and cleanup.
/// By default, servos are disabled on drop for safety.
pub struct S330 {
    e727: E727,
    poweroff_on_drop: bool,
    /// Cached travel range: ((x_min, x_max), (y_min, y_max)) in µrad
    cached_travel_ranges: OnceCell<((f64, f64), (f64, f64))>,
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
            cached_travel_ranges: OnceCell::new(),
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
            // Disable servos before changing params (shutdown_sequence handles cold start gracefully)
            info!("Axis 3 parameters need correction, preparing controller...");
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
        self.e727.get_axis12_centers()
    }

    /// Get the travel range for both tilt axes.
    ///
    /// Returns `((min_axis1, max_axis1), (min_axis2, max_axis2))` in µrad (microradians).
    ///
    /// The result is cached after the first call since travel ranges don't change.
    pub fn get_travel_ranges(&mut self) -> GcsResult<((f64, f64), (f64, f64))> {
        if let Some(&ranges) = self.cached_travel_ranges.get() {
            return Ok(ranges);
        }

        let range_axis1 = self.e727.get_travel_range(Axis::Axis1)?;
        let range_axis2 = self.e727.get_travel_range(Axis::Axis2)?;
        let ranges = (range_axis1, range_axis2);
        let _ = self.cached_travel_ranges.set(ranges);
        Ok(ranges)
    }

    /// Get the physical unit string for the tilt axes (typically "µrad").
    pub fn get_unit(&mut self) -> GcsResult<String> {
        self.e727.get_unit(Axis::Axis1)
    }

    /// Move both tilt axes to positions, clamping to travel limits.
    ///
    /// # Arguments
    /// * `axis1` - Target position for axis 1 in µrad (microradians)
    /// * `axis2` - Target position for axis 2 in µrad (microradians)
    ///
    /// # Returns
    /// `MoveResult::Ok` if within limits, `MoveResult::Clamped` if clamped.
    pub fn move_clamped(&mut self, axis1: f64, axis2: f64) -> GcsResult<MoveResult> {
        let ((x_min, x_max), (y_min, y_max)) = self.get_travel_ranges()?;
        let clamped_axis1 = axis1.clamp(x_min, x_max);
        let clamped_axis2 = axis2.clamp(y_min, y_max);
        self.move_to(clamped_axis1, clamped_axis2)?;

        if clamped_axis1 != axis1 || clamped_axis2 != axis2 {
            Ok(MoveResult::Clamped {
                requested_x: axis1,
                requested_y: axis2,
                actual_x: clamped_axis1,
                actual_y: clamped_axis2,
            })
        } else {
            Ok(MoveResult::Ok {
                x: clamped_axis1,
                y: clamped_axis2,
            })
        }
    }
}

impl crate::FsmInterface for S330 {
    fn move_to(&mut self, axis1_urad: f64, axis2_urad: f64) -> Result<(), String> {
        S330::move_to(self, axis1_urad, axis2_urad).map_err(|e| format!("FSM move failed: {e}"))
    }

    fn get_position(&mut self) -> Result<(f64, f64), String> {
        S330::get_position(self).map_err(|e| format!("FSM get_position failed: {e}"))
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
    /// 1. Smoothly slew to center position over 250ms (if servos are on)
    /// 2. Disable servo mode for axes 1 and 2
    /// 3. Set piezo output voltage to 0V for axes 1 and 2
    /// 4. Wait for voltages to reach 0V
    /// 5. Set fixed voltage to 0V for axis 3
    fn shutdown_sequence(&mut self) -> GcsResult<()> {
        info!("S-330 shutdown sequence starting...");

        // Check if servos are enabled before attempting to slew
        let servo1_on = self.e727.get_servo(Axis::Axis1).unwrap_or(false);
        let servo2_on = self.e727.get_servo(Axis::Axis2).unwrap_or(false);

        if servo1_on && servo2_on {
            // Slew smoothly to center position before disabling servos
            self.slew_to_center()?;
        } else {
            info!("Servos not enabled, skipping slew to center");
        }

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

    /// Smoothly slew to center position over SHUTDOWN_SLEW_DURATION.
    ///
    /// Sends position commands every SHUTDOWN_SLEW_INTERVAL, linearly interpolating
    /// from current position to center. This prevents jarring motion when powering down
    /// or when tracking is lost.
    pub fn slew_to_center(&mut self) -> GcsResult<()> {
        // Get current position and travel ranges
        let (start_axis1, start_axis2) = self.get_position()?;
        let ((min_axis1, max_axis1), (min_axis2, max_axis2)) = self.get_travel_ranges()?;

        // Calculate center position
        let center_axis1 = (min_axis1 + max_axis1) / 2.0;
        let center_axis2 = (min_axis2 + max_axis2) / 2.0;

        info!(
            "Slewing to center ({:.1}, {:.1}) from ({:.1}, {:.1}) µrad...",
            center_axis1, center_axis2, start_axis1, start_axis2
        );

        let start_time = Instant::now();
        let total_duration_secs = SHUTDOWN_SLEW_DURATION.as_secs_f64();

        loop {
            let elapsed = start_time.elapsed();
            if elapsed >= SHUTDOWN_SLEW_DURATION {
                break;
            }

            // Linear interpolation factor (0.0 to 1.0)
            let t = elapsed.as_secs_f64() / total_duration_secs;

            // Interpolate position
            let axis1 = start_axis1 + (center_axis1 - start_axis1) * t;
            let axis2 = start_axis2 + (center_axis2 - start_axis2) * t;

            self.move_to(axis1, axis2)?;
            std::thread::sleep(SHUTDOWN_SLEW_INTERVAL);
        }

        // Final move to exact center
        self.move_to(center_axis1, center_axis2)?;
        info!("Slew to center complete");

        Ok(())
    }
}

/// Command-line arguments for FSM (Fast Steering Mirror) connection.
///
/// Use with `#[command(flatten)]` in your CLI args struct.
#[derive(Args, Debug, Clone)]
pub struct FsmArgs {
    /// PI E-727 FSM controller IP address.
    #[arg(
        long,
        default_value = DEFAULT_FSM_IP,
        help = "PI E-727 FSM controller IP address",
        long_help = "IP address of the PI E-727 piezo controller for the S-330 Fast Steering \
            Mirror. The controller connects via ethernet on GCS port 50000."
    )]
    pub fsm_ip: String,

    /// Disable FSM servos on program exit for safety.
    #[arg(
        long,
        default_value = "true",
        help = "Disable FSM servos on program exit",
        long_help = "When true (default), the FSM servos are disabled on program exit for safety. \
            Set to false to keep the FSM powered and holding position after the program exits."
    )]
    pub fsm_shutdown_on_exit: bool,
}

impl FsmArgs {
    /// Connect to the FSM using the configured IP and shutdown settings.
    ///
    /// Returns the connected S330 instance with poweroff-on-drop configured
    /// according to `fsm_shutdown_on_exit`.
    pub fn connect(&self) -> Result<S330, String> {
        let mut fsm = S330::connect_ip(&self.fsm_ip)
            .map_err(|e| format!("Failed to connect to FSM at {}: {}", self.fsm_ip, e))?;

        if !self.fsm_shutdown_on_exit {
            fsm.set_poweroff_on_drop(false);
        }

        Ok(fsm)
    }
}

//! PI E-727 Digital Multi-Channel Piezo Controller Driver
//!
//! This module provides a high-level interface to the PI E-727 controller,
//! commonly used with fast steering mirrors (FSM) for optical beam pointing.
//!
//! # Overview
//!
//! The E-727 is a digital piezo controller supporting up to 4 axes of closed-loop
//! servo control. This driver exposes the most commonly used commands for FSM operation:
//!
//! - **Position control**: [`move_to`](E727::move_to), [`move_relative`](E727::move_relative),
//!   [`get_position`](E727::get_position)
//! - **Servo control**: [`set_servo`](E727::set_servo), [`get_servo`](E727::get_servo)
//! - **Motion queries**: [`is_on_target`](E727::is_on_target), [`wait_on_target`](E727::wait_on_target)
//! - **Emergency stop**: [`stop_all`](E727::stop_all), [`halt`](E727::halt)
//!
//! # Axis Configuration
//!
//! The E-727 typically controls a 2-axis fast steering mirror with additional
//! axes for focus or other adjustments:
//!
//! - **Axis 1, 2**: Tilt axes (typically 0-2000 µrad range)
//! - **Axis 3**: Additional tilt or unused (±2500 µrad range on some configs)
//! - **Axis 4**: Piston/focus axis (typically 0-100 µm range)
//!
//! Query axis configuration with [`axes()`](E727::axes) and [`get_travel_range()`](E727::get_travel_range).
//!
//! # Servo Control
//!
//! The E-727 uses closed-loop servo control for precision positioning. Before
//! moving, you must enable the servo on each axis with `set_all_servos(true)`.
//!
//! # Connection
//!
//! The E-727 is connected via TCP/IP on port 50000. The default IP is
//! 192.168.168.10, but DHCP is enabled so the device may acquire a different
//! address on your network.
//!
//! **Note:** USB transport was attempted but has firmware bugs causing
//! communication failures after 2-3 short packet reads.
//!
//! # Safety
//!
//! The E-727 has built-in position limits, but care should still be taken:
//!
//! - Always check [`get_travel_range()`](E727::get_travel_range) before commanding large moves
//! - Use [`stop_all()`](E727::stop_all) for emergency stops (sends Ctrl+X)
//! - Disable servos when not actively controlling the mirror
//!
//! # References
//!
//! - [PI E-727 Documentation (Google Drive)](https://drive.google.com/drive/u/0/folders/1ebFyabBYmZ5Ts942VnFBqXl_U1nlaOlV)

mod errors;
mod params;
mod recorder;

pub use errors::PiErrorCode;
pub use params::{Axis, SpaParam};
pub use recorder::{RecordChannel, RecordTrigger};

use std::collections::HashMap;
use std::net::ToSocketAddrs;
use std::time::Duration;

use ndarray::Array2;
use tracing::{debug, info};

use super::gcs::{GcsDevice, GcsError, GcsResult, DEFAULT_PORT};

/// Data recorder sample rate in Hz (50 kHz).
pub const RECORDER_SAMPLE_RATE_HZ: f64 = 50_000.0;

/// Polling interval for autozero completion.
const AUTOZERO_POLL_INTERVAL: Duration = Duration::from_millis(1000);

/// High-level driver for the PI E-727 digital piezo controller.
///
/// Provides convenient methods for fast steering mirror control including
/// position commands, servo control, and motion monitoring.
pub struct E727 {
    device: GcsDevice,
}

impl E727 {
    /// Connect to an E-727 at the given address.
    ///
    /// # Arguments
    ///
    /// * `addr` - Socket address (IP:port). For just IP, use [`connect_ip`](Self::connect_ip).
    ///
    pub fn connect<A: ToSocketAddrs + ToString>(addr: A) -> GcsResult<Self> {
        let device = GcsDevice::connect(addr)?;
        Self::init(device)
    }

    /// Connect to an E-727 at the given IP using the default port (50000).
    ///
    /// This is the recommended connection method for most use cases.
    pub fn connect_ip(ip: &str) -> GcsResult<Self> {
        let device = GcsDevice::connect(format!("{ip}:{DEFAULT_PORT}"))?;
        Self::init(device)
    }

    /// Initialize the E727 driver after connection.
    fn init(mut device: GcsDevice) -> GcsResult<Self> {
        let idn = device.query("*IDN?")?;
        info!("Connected to: {}", idn.trim());

        Ok(Self { device })
    }

    /// Set the timeout for operations.
    ///
    /// The default is 7 seconds. Increase for long moves or homing operations.
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.device.set_timeout(timeout);
    }

    /// Reconnect to the controller.
    ///
    /// Use this to recover from connection errors or socket timeouts.
    pub fn reconnect(&mut self) -> GcsResult<()> {
        self.device.reconnect()
    }

    /// Get mutable access to the underlying GCS device.
    ///
    /// Use this for low-level commands not exposed by the E727 API,
    /// such as data recorder configuration.
    pub fn device_mut(&mut self) -> &mut GcsDevice {
        &mut self.device
    }

    /// Query device identification string.
    ///
    /// Returns a string like:
    /// `(c)2015 Physik Instrumente (PI) GmbH & Co. KG, E-727, 0116044408, 13.21.00.09`
    pub fn idn(&mut self) -> GcsResult<String> {
        let response = self.device.query("*IDN?")?;
        Ok(response.trim().to_string())
    }

    /// Get the list of all possible axis identifiers.
    ///
    /// Returns all 4 axes for the E-727. Use [`connected_axes()`](Self::connected_axes)
    /// to get only axes with valid sensor readings.
    pub fn axes(&self) -> [Axis; 4] {
        [Axis::Axis1, Axis::Axis2, Axis::Axis3, Axis::Axis4]
    }

    /// Get axes that appear to be connected (position within valid range).
    ///
    /// Filters out axes where the position reading is outside the travel range,
    /// which typically indicates a disconnected or malfunctioning sensor.
    pub fn connected_axes(&mut self) -> GcsResult<Vec<Axis>> {
        let mut connected = Vec::new();
        for axis in self.axes() {
            let pos = self.get_position(axis)?;
            let (min, max) = self.get_travel_range(axis)?;
            // Allow 10% margin outside range for drift
            let margin = (max - min) * 0.1;
            if pos >= min - margin && pos <= max + margin {
                connected.push(axis);
            } else {
                debug!(
                    "Axis {axis} appears disconnected: position {pos:.1} outside [{min:.1}, {max:.1}]"
                );
            }
        }
        Ok(connected)
    }

    // ==================== Position Queries ====================

    /// Get current position of an axis in physical units (µrad or µm).
    pub fn get_position(&mut self, axis: Axis) -> GcsResult<f64> {
        let response = self.device.query(&format!("POS? {axis}"))?;
        GcsDevice::parse_single_value(&response)
    }

    /// Get current positions of all axes.
    ///
    /// Returns a HashMap mapping axis to position value.
    pub fn get_all_positions(&mut self) -> GcsResult<HashMap<Axis, f64>> {
        let mut result = HashMap::new();
        for axis in [Axis::Axis1, Axis::Axis2, Axis::Axis3, Axis::Axis4] {
            if let Ok(pos) = self.get_position(axis) {
                result.insert(axis, pos);
            }
        }
        Ok(result)
    }

    /// Get target (commanded) position of an axis.
    ///
    /// This is the position commanded by the last `MOV` command, which may
    /// differ from the actual position if motion is in progress.
    pub fn get_target(&mut self, axis: Axis) -> GcsResult<f64> {
        let response = self.device.query(&format!("MOV? {axis}"))?;
        GcsDevice::parse_single_value(&response)
    }

    /// Get target positions of all axes.
    pub fn get_all_targets(&mut self) -> GcsResult<HashMap<Axis, f64>> {
        let mut result = HashMap::new();
        for axis in [Axis::Axis1, Axis::Axis2, Axis::Axis3, Axis::Axis4] {
            if let Ok(pos) = self.get_target(axis) {
                result.insert(axis, pos);
            }
        }
        Ok(result)
    }

    // ==================== Motion Commands ====================

    /// Build GCS axis arguments string from optional per-axis values.
    ///
    /// Returns None if all values are None, otherwise returns a string like "1 100.5 2 200.3".
    fn build_axis_args(
        axis1: Option<f64>,
        axis2: Option<f64>,
        axis3: Option<f64>,
        axis4: Option<f64>,
    ) -> Option<String> {
        let mut args = Vec::new();
        if let Some(v) = axis1 {
            args.push(format!("1 {v}"));
        }
        if let Some(v) = axis2 {
            args.push(format!("2 {v}"));
        }
        if let Some(v) = axis3 {
            args.push(format!("3 {v}"));
        }
        if let Some(v) = axis4 {
            args.push(format!("4 {v}"));
        }
        if args.is_empty() {
            None
        } else {
            Some(args.join(" "))
        }
    }

    /// Move axes to absolute positions.
    ///
    /// The servo must be enabled on each axis before motion commands will work.
    /// The command returns immediately; use [`wait_on_target`](Self::wait_on_target)
    /// to wait for motion completion.
    ///
    /// # Arguments
    ///
    /// * `axis1` - Target position for axis 1, or None to skip
    /// * `axis2` - Target position for axis 2, or None to skip
    /// * `axis3` - Target position for axis 3, or None to skip
    /// * `axis4` - Target position for axis 4, or None to skip
    ///
    /// # Errors
    ///
    /// Returns `ControllerError` with code 7 if position is out of limits,
    /// or code 5 if servo is not enabled.
    pub fn move_to(
        &mut self,
        axis1: Option<f64>,
        axis2: Option<f64>,
        axis3: Option<f64>,
        axis4: Option<f64>,
    ) -> GcsResult<()> {
        match Self::build_axis_args(axis1, axis2, axis3, axis4) {
            Some(args) => self.device.command(&format!("MOV {args}")),
            None => Ok(()),
        }
    }

    /// Move a single axis to an absolute position.
    ///
    /// Convenience method for single-axis moves. The servo must be enabled
    /// before motion commands will work.
    pub fn move_axis(&mut self, axis: Axis, position: f64) -> GcsResult<()> {
        self.device.command(&format!("MOV {axis} {position}"))
    }

    /// Move axes by relative distances.
    ///
    /// # Arguments
    ///
    /// * `axis1` - Distance to move axis 1, or None to skip
    /// * `axis2` - Distance to move axis 2, or None to skip
    /// * `axis3` - Distance to move axis 3, or None to skip
    /// * `axis4` - Distance to move axis 4, or None to skip
    pub fn move_relative(
        &mut self,
        axis1: Option<f64>,
        axis2: Option<f64>,
        axis3: Option<f64>,
        axis4: Option<f64>,
    ) -> GcsResult<()> {
        match Self::build_axis_args(axis1, axis2, axis3, axis4) {
            Some(args) => self.device.command(&format!("MVR {args}")),
            None => Ok(()),
        }
    }

    /// Fast move without error checking.
    ///
    /// Sends a single MOV command without querying ERR? afterward.
    /// Use this for high-speed motion loops where latency matters.
    ///
    /// # Safety
    ///
    /// No error checking is performed. Use [`move_to`](Self::move_to) for
    /// commands that need verification.
    pub fn move_to_fast(
        &mut self,
        axis1: Option<f64>,
        axis2: Option<f64>,
        axis3: Option<f64>,
        axis4: Option<f64>,
    ) -> GcsResult<()> {
        match Self::build_axis_args(axis1, axis2, axis3, axis4) {
            Some(args) => self.device.send(&format!("MOV {args}")),
            None => Ok(()),
        }
    }

    // ==================== Servo Control ====================

    /// Enable or disable servo (closed-loop) control for an axis.
    ///
    /// Servo must be enabled before motion commands will work. When disabled,
    /// the piezo operates in open-loop mode.
    pub fn set_servo(&mut self, axis: Axis, enabled: bool) -> GcsResult<()> {
        let state = if enabled { 1 } else { 0 };
        self.device.command(&format!("SVO {axis} {state}"))
    }

    /// Enable or disable servo control for all axes.
    ///
    /// Convenience method to set all axes at once.
    pub fn set_all_servos(&mut self, enabled: bool) -> GcsResult<()> {
        for axis in [Axis::Axis1, Axis::Axis2, Axis::Axis3, Axis::Axis4] {
            self.set_servo(axis, enabled)?;
        }
        Ok(())
    }

    /// Get servo (closed-loop) state for an axis.
    ///
    /// Returns `true` if servo is enabled, `false` if disabled (open-loop).
    pub fn get_servo(&mut self, axis: Axis) -> GcsResult<bool> {
        let response = self.device.query(&format!("SVO? {axis}"))?;
        let values = GcsDevice::parse_axis_bools(&response)?;
        values
            .into_values()
            .next()
            .ok_or_else(|| GcsError::ParseError("No servo state in response".to_string()))
    }

    /// Get servo states for all axes.
    pub fn get_all_servos(&mut self) -> GcsResult<HashMap<Axis, bool>> {
        let mut result = HashMap::new();
        for axis in [Axis::Axis1, Axis::Axis2, Axis::Axis3, Axis::Axis4] {
            if let Ok(state) = self.get_servo(axis) {
                result.insert(axis, state);
            }
        }
        Ok(result)
    }

    // ==================== Voltage Control ====================

    /// Get the current output voltage for an axis.
    ///
    /// Uses the `VOL?` command to read the piezo output voltage in volts.
    pub fn get_voltage(&mut self, axis: Axis) -> GcsResult<f64> {
        let response = self.device.query(&format!("VOL? {axis}"))?;
        GcsDevice::parse_single_value(&response)
    }

    /// Set the output voltage for an axis (open-loop mode).
    ///
    /// Uses the `SVA` command to set the piezo output voltage.
    /// The servo must be disabled for this to take effect.
    pub fn set_voltage(&mut self, axis: Axis, voltage: f64) -> GcsResult<()> {
        self.device.command(&format!("SVA {axis} {voltage}"))
    }

    // ==================== Motion Status ====================

    /// Check if an axis is on target (motion complete).
    ///
    /// Returns `true` if the axis has reached its commanded position within
    /// the configured settling window.
    pub fn is_on_target(&mut self, axis: Axis) -> GcsResult<bool> {
        let response = self.device.query(&format!("ONT? {axis}"))?;
        let values = GcsDevice::parse_axis_bools(&response)?;
        values
            .into_values()
            .next()
            .ok_or_else(|| GcsError::ParseError("No on-target state in response".to_string()))
    }

    /// Get on-target state for all axes.
    pub fn all_on_target(&mut self) -> GcsResult<HashMap<Axis, bool>> {
        let mut result = HashMap::new();
        for axis in [Axis::Axis1, Axis::Axis2, Axis::Axis3, Axis::Axis4] {
            if let Ok(state) = self.is_on_target(axis) {
                result.insert(axis, state);
            }
        }
        Ok(result)
    }

    /// Check if any axis is currently moving (via control byte query).
    ///
    /// This uses the special `\x05` control byte command.
    ///
    /// **Note:** This command may return unexpected data over TCP. Prefer
    /// checking [`all_on_target()`](Self::all_on_target) instead.
    pub fn is_moving(&mut self) -> GcsResult<bool> {
        self.device.send("\x05")?;
        let response = self.device.read()?;
        let status: u8 = response
            .trim()
            .parse()
            .map_err(|_| GcsError::InvalidResponse(format!("Invalid motion status: {response}")))?;
        Ok(status != 0)
    }

    /// Check if controller is ready (via control byte query).
    ///
    /// This uses the special `\x07` control byte command.
    ///
    /// **Note:** This command may return unexpected data over TCP. Prefer
    /// checking [`all_on_target()`](Self::all_on_target) instead.
    pub fn is_ready(&mut self) -> GcsResult<bool> {
        self.device.send("\x07")?;
        let response = self.device.read()?;
        let byte = response.bytes().next().unwrap_or(0);
        Ok(byte == 0xB1)
    }

    // ==================== Emergency Stop ====================

    /// Emergency stop all axes (sends Ctrl+X).
    ///
    /// This immediately stops all motion. Use in emergency situations or
    /// when you need to abort motion quickly.
    pub fn stop_all(&mut self) -> GcsResult<()> {
        self.device.send("\x18")?;
        Ok(())
    }

    /// Halt a specific axis.
    ///
    /// Stops motion on the specified axis while leaving other axes unaffected.
    pub fn halt(&mut self, axis: Axis) -> GcsResult<()> {
        self.device.command(&format!("HLT {axis}"))
    }

    // ==================== Configuration Queries ====================

    /// Get the travel range limits for an axis.
    ///
    /// Returns `(min, max)` position values in physical units (µrad or µm).
    pub fn get_travel_range(&mut self, axis: Axis) -> GcsResult<(f64, f64)> {
        let min_response = self.device.query(&format!("TMN? {axis}"))?;
        let max_response = self.device.query(&format!("TMX? {axis}"))?;
        let min = GcsDevice::parse_single_value(&min_response)?;
        let max = GcsDevice::parse_single_value(&max_response)?;
        Ok((min, max))
    }

    /// Get the physical unit for an axis.
    ///
    /// Returns strings like `"µrad"` (microradians) or `"µm"` (micrometers).
    pub fn get_unit(&mut self, axis: Axis) -> GcsResult<String> {
        let response = self.device.query(&format!("PUN? {axis}"))?;
        for line in response.lines() {
            if let Some((_axis, value)) = line.split_once('=') {
                return Ok(value.trim().to_string());
            }
        }
        Err(GcsError::ParseError("No unit in response".to_string()))
    }

    /// Get the center position of an axis (midpoint of travel range).
    pub fn get_center(&mut self, axis: Axis) -> GcsResult<f64> {
        let (min, max) = self.get_travel_range(axis)?;
        Ok((min + max) / 2.0)
    }

    /// Get center positions for axes 1 and 2.
    ///
    /// Returns `(center_axis1, center_axis2)` tuple.
    pub fn get_axis12_centers(&mut self) -> GcsResult<(f64, f64)> {
        let center_axis1 = self.get_center(Axis::Axis1)?;
        let center_axis2 = self.get_center(Axis::Axis2)?;
        Ok((center_axis1, center_axis2))
    }

    // ==================== Parameter Access (SPA) ====================

    /// Query a parameter value for an axis.
    ///
    /// Uses the `SPA?` command to read controller parameters.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to query
    /// * `param` - The parameter to read
    ///
    /// # Returns
    ///
    /// The parameter value as a float (some SPA params return scientific notation).
    pub fn get_param(&mut self, axis: Axis, param: SpaParam) -> GcsResult<f64> {
        let addr = param.address();
        let response = self.device.query(&format!("SPA? {axis} 0x{addr:08X}"))?;

        // Parse response like "3 0x09000000=0" or "3 0x09000000=-1.999999955e-02"
        response
            .split('=')
            .nth(1)
            .and_then(|v| v.trim().parse::<f64>().ok())
            .ok_or_else(|| GcsError::ParseError(format!("Invalid SPA response: {response}")))
    }

    /// Set a parameter value for an axis.
    ///
    /// Uses the `SPA` command to write controller parameters.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to configure
    /// * `param` - The parameter to write
    /// * `value` - The value to set
    pub fn set_param(&mut self, axis: Axis, param: SpaParam, value: f64) -> GcsResult<()> {
        let addr = param.address();
        self.device
            .command(&format!("SPA {axis} 0x{addr:08X} {value}"))
    }

    // ==================== Utility Methods ====================

    /// Wait until all axes are on target or timeout.
    ///
    /// Polls [`all_on_target()`](Self::all_on_target) every 10ms until all axes
    /// report being on target, or the timeout expires.
    pub fn wait_on_target(&mut self, timeout: Duration) -> GcsResult<()> {
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(GcsError::Timeout);
            }

            let on_target = self.all_on_target()?;
            if on_target.values().all(|&v| v) {
                return Ok(());
            }

            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Query the last error code from the controller.
    ///
    /// Returns the raw error code (0 = no error).
    /// Use [`last_error_decoded`](Self::last_error_decoded) for a decoded error.
    pub fn last_error(&mut self) -> GcsResult<i32> {
        self.device.send("ERR?")?;
        let response = self.device.read()?;
        response
            .trim()
            .parse()
            .map_err(|_| GcsError::InvalidResponse(format!("Invalid error code: {response}")))
    }

    /// Query and decode the last error from the controller.
    ///
    /// Returns the decoded `PiErrorCode` enum value, including `NoError` for code 0.
    /// For unknown error codes, returns the raw code in the error tuple.
    pub fn last_error_decoded(&mut self) -> GcsResult<(i32, PiErrorCode)> {
        let code = self.last_error()?;
        let err = PiErrorCode::from_code(code).unwrap_or(PiErrorCode::UnknownError);
        Ok((code, err))
    }

    // ==================== Autozero ====================

    /// Check if an axis has been autozeroed.
    ///
    /// Returns `true` if the axis has completed autozero successfully.
    pub fn is_autozeroed(&mut self, axis: Axis) -> GcsResult<bool> {
        let response = self.device.query(&format!("ATZ? {axis}"))?;
        Ok(response.contains("=1"))
    }

    /// Perform autozero on specified axes.
    ///
    /// Autozero calibrates the piezo sensors and should be performed after power-on
    /// or when position accuracy degrades. This operation takes several seconds.
    ///
    /// # Arguments
    ///
    /// * `axes` - Axes to autozero
    /// * `force` - If `true`, always run autozero. If `false`, skip if already done.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Autozero was performed
    /// * `Ok(false)` - Autozero was skipped (already done and force=false)
    /// * `Err(_)` - Autozero failed
    pub fn autozero(&mut self, axes: &[Axis], force: bool) -> GcsResult<bool> {
        // Check if already autozeroed
        let mut all_zeroed = true;
        for &axis in axes {
            if !self.is_autozeroed(axis)? {
                all_zeroed = false;
                break;
            }
        }
        if !force && all_zeroed {
            return Ok(false);
        }

        info!("Starting autozero...");

        for &axis in axes {
            self.device.send(&format!("ATZ {axis} NaN"))?;

            // Poll ATZ? until axis is autozeroed
            loop {
                match self.is_autozeroed(axis) {
                    Ok(true) => {
                        break;
                    }
                    Ok(false) => {
                        debug!("ATZ? returned false, waiting...");
                    }
                    Err(GcsError::ControllerError {
                        error: Some(PiErrorCode::AutozeroRunning),
                        ..
                    }) => {
                        debug!("ATZ in progress...");
                    }
                    Err(err) => {
                        return Err(err);
                    }
                }
                std::thread::sleep(AUTOZERO_POLL_INTERVAL);
            }

            info!("Axis {axis} autozeroed");
        }

        info!("Autozero complete");
        Ok(true)
    }

    // ==================== Data Recording ====================

    /// Record position error and current position data for an axis.
    ///
    /// Uses the E-727's built-in data recorder to capture high-speed (50kHz)
    /// position data. Returns vectors of (error, position) samples.
    ///
    /// # Arguments
    ///
    /// * `axis` - Axis to record
    /// * `duration` - How long to record
    ///
    /// # Returns
    ///
    /// Tuple of (position_errors, positions) vectors, sampled at 50kHz (20µs intervals).
    pub fn record_position(
        &mut self,
        axis: Axis,
        duration: Duration,
    ) -> GcsResult<(Vec<f64>, Vec<f64>)> {
        const RECORD_POSITION_ERROR: u8 = 3;
        const RECORD_CURRENT_POSITION: u8 = 2;
        const TRIGGER_IMMEDIATE: u8 = 4;

        // Configure recorder: table 1=error, table 2=position
        self.device.send("RTR 1")?; // Max sample rate (50kHz)
        self.device
            .send(&format!("DRC 1 {axis} {RECORD_POSITION_ERROR}"))?;
        self.device
            .send(&format!("DRC 2 {axis} {RECORD_CURRENT_POSITION}"))?;

        // Start recording (immediate trigger)
        self.device.send(&format!("DRT 1 {TRIGGER_IMMEDIATE} 0"))?;
        self.device.send(&format!("DRT 2 {TRIGGER_IMMEDIATE} 0"))?;

        // Wait for recording duration
        std::thread::sleep(duration);

        // Stop recording
        self.device.send("DRT 1 0 0")?;
        self.device.send("DRT 2 0 0")?;

        // Get sample count
        let response = self.device.query("DRL? 1")?;
        let num_points: usize = response
            .split('=')
            .nth(1)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(0);

        if num_points == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // Read data from both tables
        let resp1 = self.device.query(&format!("DRR? 1 {num_points} 1"))?;
        let resp2 = self.device.query(&format!("DRR? 1 {num_points} 2"))?;

        // Parse data (skip header lines starting with #)
        let parse_data = |s: &str| -> Vec<f64> {
            s.lines()
                .filter_map(|l| {
                    let t = l.trim();
                    if t.is_empty() || t.starts_with('#') {
                        None
                    } else {
                        t.parse().ok()
                    }
                })
                .collect()
        };

        let errors = parse_data(&resp1);
        let positions = parse_data(&resp2);

        debug!(
            "Recorded {} error samples, {} position samples",
            errors.len(),
            positions.len()
        );

        Ok((errors, positions))
    }

    // ==================== Recording Configuration ====================

    /// Configure data recording channels and trigger.
    ///
    /// Sets up the data recorder with the specified channels and trigger mode.
    /// Recording will start based on the trigger type:
    /// - `Immediate`: Recording starts immediately
    /// - `OnMove`: Recording starts when a motion command is sent
    /// - `Default`: Recording starts with IMP/STE/WGO/WGR commands
    /// - `ExternalInput`: Recording starts from digital input signal
    ///
    /// # Arguments
    ///
    /// * `trigger` - What triggers recording to start
    /// * `channels` - What data to record (up to 8 channels)
    /// * `sample_rate_divider` - Divides the 50kHz base rate (1 = full rate)
    pub fn configure_recording(
        &mut self,
        trigger: RecordTrigger,
        channels: &[RecordChannel],
        sample_rate_divider: u32,
    ) -> GcsResult<()> {
        if channels.is_empty() {
            return Err(GcsError::InvalidArgument(
                "At least one channel must be specified".to_string(),
            ));
        }
        if channels.len() > 8 {
            return Err(GcsError::InvalidArgument(
                "Maximum 8 channels supported".to_string(),
            ));
        }

        // Set sample rate divider
        self.device.send(&format!("RTR {sample_rate_divider}"))?;

        // Configure each channel (table)
        for (table_id, &channel) in channels.iter().enumerate() {
            let table = table_id + 1; // Tables are 1-indexed
            let (source, option) = channel.to_drc_args();
            self.device
                .send(&format!("DRC {table} {source} {option}"))?;
        }

        // Set trigger for all configured tables
        let (trigger_source, trigger_value) = trigger.to_drt_args();
        for table_id in 0..channels.len() {
            let table = table_id + 1;
            self.device
                .send(&format!("DRT {table} {trigger_source} {trigger_value}"))?;
        }

        debug!(
            "Configured {} recording channels with trigger {:?}",
            channels.len(),
            trigger
        );

        Ok(())
    }

    /// Stop any active recording.
    ///
    /// Disables the trigger on all tables, stopping data collection.
    pub fn stop_recording(&mut self) -> GcsResult<()> {
        // Query number of tables
        let response = self.device.query("TNR?")?;
        let num_tables: usize = response
            .split('=')
            .nth(1)
            .unwrap_or("8")
            .trim()
            .parse()
            .unwrap_or(8);

        // Disable trigger on all tables
        for table in 1..=num_tables {
            self.device.send(&format!("DRT {table} 0 0"))?;
        }

        Ok(())
    }

    /// Get the number of points recorded so far.
    ///
    /// Useful for checking recording progress or determining when
    /// the recording buffer is full.
    pub fn recording_length(&mut self) -> GcsResult<usize> {
        let response = self.device.query("DRL? 1")?;
        let num_points: usize = response
            .split('=')
            .nth(1)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(0);
        Ok(num_points)
    }

    /// Get the maximum number of points per recording table.
    ///
    /// The E-727 has 262144 total points shared equally among configured tables.
    /// With 2 tables, each can hold 131072 points (~2.6s at 50kHz).
    pub fn recording_max_points(&mut self, num_tables: usize) -> GcsResult<usize> {
        const TOTAL_RECORDER_POINTS: usize = 262144;
        let tables = num_tables.max(1);
        Ok(TOTAL_RECORDER_POINTS / tables)
    }

    /// Wait for recording buffer to fill completely.
    ///
    /// Polls `DRL?` until the recorded points reach the maximum capacity
    /// for the given number of tables, or until the timeout expires.
    ///
    /// # Arguments
    ///
    /// * `num_tables` - Number of recording tables configured
    /// * `timeout` - Maximum time to wait
    /// * `poll_interval` - How often to check progress
    pub fn wait_recording_complete(
        &mut self,
        num_tables: usize,
        timeout: Duration,
        poll_interval: Duration,
    ) -> GcsResult<usize> {
        let max_points = self.recording_max_points(num_tables)?;
        let start = std::time::Instant::now();

        loop {
            let current = self.recording_length()?;
            if current >= max_points {
                return Ok(current);
            }

            if start.elapsed() > timeout {
                return Err(GcsError::Timeout);
            }

            std::thread::sleep(poll_interval);
        }
    }

    /// Read recorded data as a 2D array.
    ///
    /// Returns an Array2<f64> with shape (num_samples, num_channels).
    /// Each column contains data from one recording table.
    ///
    /// # Arguments
    ///
    /// * `num_channels` - Number of channels that were configured
    pub fn read_recording(&mut self, num_channels: usize) -> GcsResult<Array2<f64>> {
        let num_points = self.recording_length()?;

        if num_points == 0 {
            return Ok(Array2::zeros((0, num_channels)));
        }

        // Read data from each table
        let mut columns: Vec<Vec<f64>> = Vec::with_capacity(num_channels);
        for table in 1..=num_channels {
            let response = self.device.query(&format!("DRR? 1 {num_points} {table}"))?;
            let data = Self::parse_recorder_data(&response);
            columns.push(data);
        }

        // Find minimum length (all should be equal, but be safe)
        let min_len = columns.iter().map(|c| c.len()).min().unwrap_or(0);
        if min_len == 0 {
            return Ok(Array2::zeros((0, num_channels)));
        }

        // Build Array2 with shape (samples, channels)
        let mut array = Array2::zeros((min_len, num_channels));
        for (col_idx, col_data) in columns.iter().enumerate() {
            for (row_idx, &value) in col_data.iter().take(min_len).enumerate() {
                array[[row_idx, col_idx]] = value;
            }
        }

        debug!(
            "Read recording: {} samples x {} channels",
            min_len, num_channels
        );

        Ok(array)
    }

    /// Parse data recorder response into vector of floats.
    fn parse_recorder_data(response: &str) -> Vec<f64> {
        response
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    None
                } else {
                    trimmed.parse().ok()
                }
            })
            .collect()
    }

    /// Record data with immediate trigger for a fixed duration.
    ///
    /// Convenience method that configures recording, waits for the specified
    /// duration, stops recording, and returns the data as an Array2.
    ///
    /// # Arguments
    ///
    /// * `channels` - What data to record
    /// * `duration` - How long to record
    /// * `sample_rate_divider` - Divides the 50kHz base rate (1 = full rate)
    ///
    /// # Returns
    ///
    /// Array2<f64> with shape (num_samples, num_channels).
    /// For 100ms at 50kHz with 2 channels, shape would be (5000, 2).
    pub fn record(
        &mut self,
        channels: &[RecordChannel],
        duration: Duration,
        sample_rate_divider: u32,
    ) -> GcsResult<Array2<f64>> {
        let num_channels = channels.len();

        self.configure_recording(RecordTrigger::Immediate, channels, sample_rate_divider)?;

        std::thread::sleep(duration);

        self.stop_recording()?;

        self.read_recording(num_channels)
    }

    /// Record data triggered by the next motion command.
    ///
    /// Configures recording to start when a motion command is sent,
    /// then calls the provided closure to perform the motion.
    /// After motion completes, stops recording and returns the data.
    ///
    /// # Arguments
    ///
    /// * `channels` - What data to record
    /// * `sample_rate_divider` - Divides the 50kHz base rate (1 = full rate)
    /// * `motion` - Closure that performs the motion (receives &mut E727)
    /// * `settle_time` - Additional time to record after motion starts
    ///
    /// # Returns
    ///
    /// Array2<f64> with shape (num_samples, num_channels)
    pub fn record_motion<F>(
        &mut self,
        channels: &[RecordChannel],
        sample_rate_divider: u32,
        motion: F,
        settle_time: Duration,
    ) -> GcsResult<Array2<f64>>
    where
        F: FnOnce(&mut Self) -> GcsResult<()>,
    {
        let num_channels = channels.len();

        self.configure_recording(RecordTrigger::OnMove, channels, sample_rate_divider)?;

        // Execute motion - this triggers recording
        motion(self)?;

        // Wait for settling
        std::thread::sleep(settle_time);

        self.stop_recording()?;

        self.read_recording(num_channels)
    }
}

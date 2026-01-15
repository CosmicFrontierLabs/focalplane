//! GPIO control for Jetson Orin Nano using BOARD pin numbering.
//!
//! This module provides BOARD-mode pin numbering (physical header pins) for GPIO control
//! on Jetson Orin Nano/NX hardware. It uses the gpiod library for reliable GPIO access
//! and includes a lookup table to map BOARD pin numbers to GPIO line offsets.
//!
//! # Pin Numbering
//!
//! BOARD mode uses physical pin numbers on the 40-pin header (e.g., pin 33 = GPIO13).
//! This is more user-friendly than raw GPIO line numbers.
//!
//! # Temporary Implementation Note
//!
//! The current implementation uses a manual lookup table extracted from the Python
//! Jetson.GPIO library. This is a temporary solution because the jetson-gpio-rust
//! crate doesn't recognize the Jetson Orin Nano Super model.
//!
//! See TODO.md "Modernize jetson-gpio-rust integration" for long-term improvements.

use anyhow::{Context, Result};
use gpiod::{Chip, Lines, Options, Output};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum GpioConfig {
    BoardPin(u32),
    DirectLine(u32),
}

fn get_hostname() -> Result<String> {
    let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .context("Failed to read hostname")?
        .trim()
        .to_string();
    Ok(hostname)
}

/// Detects GPIO configuration based on hostname.
///
/// Auto-detects GPIO configuration for known devices:
/// - `orin-005` (Neutralino/NSV): Direct line 127
/// - `ubuntu` (Orin Nano/POA): BOARD pin 33
///
/// # Returns
///
/// Returns `Ok(GpioConfig)` with the detected configuration, or an error if
/// the hostname is not recognized.
pub fn detect_gpio_config() -> Result<GpioConfig> {
    let hostname = get_hostname()?;

    match hostname.as_str() {
        "orin-005" => Ok(GpioConfig::DirectLine(127)),
        "ubuntu" => Ok(GpioConfig::BoardPin(33)),
        _ => {
            anyhow::bail!(
                "Unknown device hostname '{hostname}'. Please specify GPIO configuration explicitly.\n\
                Known hostnames: 'orin-005' (Neutralino/NSV), 'ubuntu' (Orin Nano/POA)"
            )
        }
    }
}

pub struct GpioController {
    chip: Chip,
    line_offset: u32,
    request: Option<Lines<Output>>,
}

/// Maps BOARD pin numbers to GPIO line offsets for Jetson Orin NX/Nano.
///
/// This lookup table provides the mapping from physical header pin numbers
/// (BOARD mode) to gpiod GPIO line offsets on gpiochip0.
///
/// # Data Source
///
/// Extracted from Python Jetson.GPIO library:
/// `ext_ref/jetson-gpio/lib/python/Jetson/GPIO/gpio_pin_data.py`
/// Lines 55-78: JETSON_ORIN_NX_PIN_DEFS
///
/// # Pin Mapping Format
///
/// Each entry maps: (BOARD pin number, GPIO line offset)
/// Example: Pin 33 (GPIO13) → line 43 on gpiochip0
///
/// # TODO
///
/// This is a temporary solution. See TODO.md for plans to:
/// - Fork/contribute to jetson-gpio-rust for Orin Nano Super support
/// - Extract pin data programmatically from Python library
/// - Support multiple Jetson board models dynamically
fn board_pin_to_line(pin: u32) -> Option<u32> {
    let pin_map: HashMap<u32, u32> = [
        (7, 144),
        (11, 112),
        (12, 50),
        (13, 122),
        (15, 85),
        (16, 126),
        (18, 125),
        (19, 135),
        (21, 134),
        (22, 123),
        (23, 133),
        (24, 136),
        (26, 137),
        (29, 105),
        (31, 106),
        (32, 41),
        (33, 43),
        (35, 53),
        (36, 113),
        (37, 124),
        (38, 52),
        (40, 51),
    ]
    .iter()
    .cloned()
    .collect();

    pin_map.get(&pin).copied()
}

impl GpioController {
    /// Creates a new GPIO controller for the specified BOARD pin number.
    ///
    /// # Arguments
    ///
    /// * `board_pin` - Physical header pin number (BOARD mode, e.g., 33 for GPIO13)
    ///
    /// # Returns
    ///
    /// Returns `Ok(GpioController)` if the pin is valid and gpiochip0 is accessible,
    /// or an error if the pin number is invalid or GPIO chip cannot be opened.
    pub fn new(board_pin: u32) -> Result<Self> {
        let line_offset = board_pin_to_line(board_pin)
            .ok_or_else(|| anyhow::anyhow!("Invalid BOARD pin number: {board_pin}"))?;

        let chip =
            Chip::new("gpiochip0").with_context(|| "Failed to open GPIO chip 'gpiochip0'")?;

        Ok(Self {
            chip,
            line_offset,
            request: None,
        })
    }

    /// Creates a new GPIO controller using a raw GPIO line number.
    ///
    /// This bypasses the BOARD pin lookup and directly uses the GPIO line offset.
    /// Useful for hardware that doesn't follow standard Jetson pinouts (e.g., Neutralino).
    ///
    /// # Arguments
    ///
    /// * `line_offset` - Direct GPIO line number on gpiochip0 (e.g., 127)
    ///
    /// # Returns
    ///
    /// Returns `Ok(GpioController)` if gpiochip0 is accessible.
    pub fn new_from_line(line_offset: u32) -> Result<Self> {
        let chip =
            Chip::new("gpiochip0").with_context(|| "Failed to open GPIO chip 'gpiochip0'")?;

        Ok(Self {
            chip,
            line_offset,
            request: None,
        })
    }

    /// Configures the GPIO pin as an output with an initial value.
    ///
    /// # Arguments
    ///
    /// * `consumer` - Descriptive name for this GPIO consumer (shows in kernel logs)
    /// * `initial_value` - Initial pin state (0 = LOW, non-zero = HIGH)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, or an error if the line cannot be requested.
    pub fn request_output(&mut self, consumer: &str, initial_value: u8) -> Result<()> {
        let options = Options::output([self.line_offset])
            .values([initial_value != 0])
            .consumer(consumer);

        let request = self
            .chip
            .request_lines(options)
            .with_context(|| "Failed to request GPIO line as output")?;

        self.request = Some(request);
        Ok(())
    }

    /// Sets the GPIO pin output value.
    ///
    /// # Arguments
    ///
    /// * `value` - Output value (0 = LOW, non-zero = HIGH)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, or an error if the GPIO was not configured
    /// as output or the value cannot be set.
    pub fn set_value(&mut self, value: u8) -> Result<()> {
        if let Some(ref mut request) = self.request {
            request
                .set_values([value != 0])
                .with_context(|| format!("Failed to set GPIO value to {value}"))?;
        } else {
            anyhow::bail!("GPIO line not requested as output");
        }
        Ok(())
    }

    /// Convenience method to set pin HIGH.
    pub fn on(&mut self) -> Result<()> {
        self.set_value(1)
    }

    /// Convenience method to set pin LOW.
    pub fn off(&mut self) -> Result<()> {
        self.set_value(0)
    }
}

/// Default GPIO pin for latency measurement (pin 33 = GPIO13 on Jetson Orin Nano).
pub const ORIN_BOARD_PIN_33: u32 = 33;

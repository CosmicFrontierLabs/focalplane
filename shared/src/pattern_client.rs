//! HTTP client for controlling OLED display patterns.
//!
//! Provides a simple API to send pattern commands to calibrate_serve.

use crate::pattern_command::PatternCommand;
use crate::system_info::DisplayInfo;

/// Client for controlling OLED display patterns via REST API.
///
/// Usage:
///   let client = PatternClient::new("http://test-bench-pi:3001");
///   client.enable_remote_mode()?;
///   client.spot(1280.0, 1280.0, 5.0, 1.0)?;
///   client.clear()?;
pub struct PatternClient {
    base_url: String,
}

impl PatternClient {
    /// Create a new pattern client.
    ///
    /// # Arguments
    /// * `base_url` - Base URL of calibrate_serve (e.g., "http://test-bench-pi:3001")
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Enable RemoteControlled mode on the display.
    ///
    /// This must be called before pattern commands will be visible.
    pub fn enable_remote_mode(&self) -> Result<(), String> {
        let url = format!("{}/config", self.base_url);
        let body = r#"{"pattern_id": "RemoteControlled", "values": {}, "invert": false}"#;

        let response = ureq::post(&url)
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
            .map_err(|e| format!("Failed to enable RemoteControlled mode: {e}"))?;

        if response.status() != 200 {
            return Err(format!(
                "Failed to enable RemoteControlled mode: HTTP {}",
                response.status()
            ));
        }

        Ok(())
    }

    /// Send a pattern command.
    pub fn send(&self, cmd: &PatternCommand) -> Result<(), String> {
        let url = format!("{}/pattern", self.base_url);
        let json =
            serde_json::to_string(cmd).map_err(|e| format!("Failed to serialize command: {e}"))?;

        let response = ureq::post(&url)
            .header("Content-Type", "application/json")
            .send(json.as_bytes())
            .map_err(|e| format!("Failed to send pattern command: {e}"))?;

        if response.status() != 200 {
            return Err(format!(
                "Pattern command failed: HTTP {}",
                response.status()
            ));
        }

        Ok(())
    }

    /// Display a single Gaussian spot.
    ///
    /// # Arguments
    /// * `x` - X position in display pixels
    /// * `y` - Y position in display pixels
    /// * `fwhm` - Full-width at half-maximum in pixels
    /// * `intensity` - Peak intensity (0.0 to 1.0)
    pub fn spot(&self, x: f64, y: f64, fwhm: f64, intensity: f64) -> Result<(), String> {
        self.send(&PatternCommand::Spot {
            x,
            y,
            fwhm,
            intensity,
        })
    }

    /// Display a grid of spots.
    ///
    /// # Arguments
    /// * `positions` - List of (x, y) positions in display pixels
    /// * `fwhm` - Full-width at half-maximum in pixels
    /// * `intensity` - Peak intensity (0.0 to 1.0)
    pub fn spot_grid(
        &self,
        positions: Vec<(f64, f64)>,
        fwhm: f64,
        intensity: f64,
    ) -> Result<(), String> {
        self.send(&PatternCommand::SpotGrid {
            positions,
            fwhm,
            intensity,
        })
    }

    /// Display uniform gray level.
    ///
    /// # Arguments
    /// * `level` - Gray level (0 = black, 255 = white)
    pub fn uniform(&self, level: u8) -> Result<(), String> {
        self.send(&PatternCommand::Uniform { level })
    }

    /// Clear display to black.
    pub fn clear(&self) -> Result<(), String> {
        self.send(&PatternCommand::Clear)
    }

    /// Get display info (dimensions, pixel pitch).
    pub fn display_info(&self) -> Result<DisplayInfo, String> {
        let url = format!("{}/info", self.base_url);

        let response = ureq::get(&url)
            .call()
            .map_err(|e| format!("Failed to get display info: {e}"))?;

        if response.status() != 200 {
            return Err(format!(
                "Failed to get display info: HTTP {}",
                response.status()
            ));
        }

        response
            .into_body()
            .read_json()
            .map_err(|e| format!("Failed to parse display info: {e}"))
    }

    /// Get current pattern command state.
    pub fn current_pattern(&self) -> Result<PatternCommand, String> {
        let url = format!("{}/pattern", self.base_url);

        let response = ureq::get(&url)
            .call()
            .map_err(|e| format!("Failed to get current pattern: {e}"))?;

        if response.status() != 200 {
            return Err(format!(
                "Failed to get current pattern: HTTP {}",
                response.status()
            ));
        }

        response
            .into_body()
            .read_json()
            .map_err(|e| format!("Failed to parse pattern: {e}"))
    }
}

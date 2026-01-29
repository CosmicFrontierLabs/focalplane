//! HTTP client for interacting with calibrate_serve API.
//!
//! This module provides a unified client that works in both native Rust
//! and WASM (frontend) environments. All API interactions are consolidated
//! here for consistent error handling and type safety.

use serde::Serialize;

use crate::http_client::{HttpClient, HttpClientError};
use crate::{DisplayInfo, PatternCommand, PatternConfigResponse, SchemaResponse};

/// Error type for calibrate server operations.
pub type CalibrateError = HttpClientError;

/// Client for interacting with calibrate_serve HTTP API.
///
/// Works in both native Rust and WASM environments.
#[derive(Debug, Clone)]
pub struct CalibrateServerClient {
    http: HttpClient,
}

impl CalibrateServerClient {
    /// Create a new client pointing to the given base URL.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL of the calibrate_serve (e.g., "http://localhost:3001")
    pub fn new(base_url: &str) -> Self {
        Self {
            http: HttpClient::new(base_url),
        }
    }

    /// Create a client for same-origin web requests.
    ///
    /// Uses relative URLs (empty base) which works in WASM when the frontend
    /// is served from the same origin as the API. Panics if called outside WASM.
    pub fn for_web() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Self::new("")
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            unreachable!("for_web() is only available in WASM builds")
        }
    }

    /// Get the base URL this client is configured for.
    pub fn base_url(&self) -> &str {
        self.http.base_url()
    }

    // === Display Info ===

    /// Get display information.
    pub async fn get_display_info(&self) -> Result<DisplayInfo, CalibrateError> {
        self.http.get("/info").await
    }

    // === Schema ===

    /// Get pattern schema.
    pub async fn get_schema(&self) -> Result<SchemaResponse, CalibrateError> {
        self.http.get("/schema").await
    }

    // === Pattern Configuration ===

    /// Get current pattern configuration.
    pub async fn get_config(&self) -> Result<PatternConfigResponse, CalibrateError> {
        self.http.get("/config").await
    }

    /// Update pattern configuration.
    pub async fn set_config(&self, config: &PatternConfigRequest) -> Result<(), CalibrateError> {
        self.http.post_no_response("/config", config).await
    }

    // === Pattern Commands (RemoteControlled mode) ===

    /// Get current pattern command.
    pub async fn get_pattern(&self) -> Result<PatternCommand, CalibrateError> {
        self.http.get("/pattern").await
    }

    /// Send pattern command.
    pub async fn set_pattern(&self, command: &PatternCommand) -> Result<(), CalibrateError> {
        self.http.post_no_response("/pattern", command).await
    }

    // === Convenience Methods (wrapping PatternCommand) ===

    /// Enable RemoteControlled mode on the display.
    ///
    /// This must be called before pattern commands will be visible.
    pub async fn enable_remote_mode(&self) -> Result<(), CalibrateError> {
        let config = PatternConfigRequest {
            pattern_id: "RemoteControlled".to_string(),
            values: serde_json::Map::new(),
            invert: Some(false),
            emit_gyro: None,
            plate_scale: None,
        };
        self.set_config(&config).await
    }

    /// Display a single Gaussian spot.
    ///
    /// # Arguments
    /// * `x` - X position in display pixels
    /// * `y` - Y position in display pixels
    /// * `fwhm` - Full-width at half-maximum in pixels
    /// * `intensity` - Peak intensity (0.0 to 1.0)
    pub async fn spot(
        &self,
        x: f64,
        y: f64,
        fwhm: f64,
        intensity: f64,
    ) -> Result<(), CalibrateError> {
        self.set_pattern(&PatternCommand::Spot {
            x,
            y,
            fwhm,
            intensity,
        })
        .await
    }

    /// Display a grid of spots.
    ///
    /// # Arguments
    /// * `positions` - List of (x, y) positions in display pixels
    /// * `fwhm` - Full-width at half-maximum in pixels
    /// * `intensity` - Peak intensity (0.0 to 1.0)
    pub async fn spot_grid(
        &self,
        positions: Vec<(f64, f64)>,
        fwhm: f64,
        intensity: f64,
    ) -> Result<(), CalibrateError> {
        self.set_pattern(&PatternCommand::SpotGrid {
            positions,
            fwhm,
            intensity,
        })
        .await
    }

    /// Display uniform gray level.
    ///
    /// # Arguments
    /// * `level` - Gray level (0 = black, 255 = white)
    pub async fn uniform(&self, level: u8) -> Result<(), CalibrateError> {
        self.set_pattern(&PatternCommand::Uniform { level }).await
    }

    /// Clear display to black.
    pub async fn clear(&self) -> Result<(), CalibrateError> {
        self.set_pattern(&PatternCommand::Clear).await
    }

    // === Pattern Config Helpers ===

    /// Set display to center pixel pattern.
    ///
    /// Shows a single bright pixel at the center of the display.
    /// Useful for calibration where a point source is needed.
    pub async fn set_pixel_pattern(&self) -> Result<(), CalibrateError> {
        let config = PatternConfigRequest {
            pattern_id: "Pixel".to_string(),
            values: serde_json::Map::new(),
            invert: Some(false),
            emit_gyro: None,
            plate_scale: None,
        };
        self.set_config(&config).await
    }

    /// Turn off display by setting uniform black.
    ///
    /// Uses the Uniform pattern with level 0 via /config endpoint,
    /// which works regardless of remote control mode.
    pub async fn set_blank(&self) -> Result<(), CalibrateError> {
        let config = PatternConfigRequest {
            pattern_id: "Uniform".to_string(),
            values: serde_json::json!({"level": 0})
                .as_object()
                .cloned()
                .unwrap_or_default(),
            invert: Some(false),
            emit_gyro: None,
            plate_scale: None,
        };
        self.set_config(&config).await
    }

    // === Image URLs ===

    /// Get the URL for JPEG pattern endpoint.
    pub fn jpeg_url(&self) -> String {
        format!("{}/jpeg", self.http.base_url())
    }
}

/// Request body for updating pattern configuration.
#[derive(Debug, Clone, Serialize)]
pub struct PatternConfigRequest {
    /// Pattern type identifier
    pub pattern_id: String,
    /// Pattern-specific parameter values
    pub values: serde_json::Map<String, serde_json::Value>,
    /// Whether to invert colors
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invert: Option<bool>,
    /// Enable gyro emission (if FTDI configured)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emit_gyro: Option<bool>,
    /// Plate scale in arcsec/pixel for gyro emission
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plate_scale: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_url_construction() {
        let client = CalibrateServerClient::new("http://localhost:3001");
        assert_eq!(client.base_url(), "http://localhost:3001");
        assert_eq!(client.jpeg_url(), "http://localhost:3001/jpeg");
    }

    #[test]
    fn test_client_strips_trailing_slash() {
        let client = CalibrateServerClient::new("http://localhost:3001/");
        assert_eq!(client.base_url(), "http://localhost:3001");
    }
}

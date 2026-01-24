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

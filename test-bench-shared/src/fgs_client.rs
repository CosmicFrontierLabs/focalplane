//! HTTP client for interacting with fgs_server API.
//!
//! This module provides a unified client that works in both native Rust
//! and WASM (frontend) environments. All API interactions are consolidated
//! here for consistent error handling and type safety.

use crate::http_client::{HttpClient, HttpClientError};
use crate::{
    CameraStats, ExportSettings, ExportStatus, FsmMoveRequest, FsmStatus, RawFrameResponse,
    StarDetectionSettings, TrackingEnableRequest, TrackingSettings, TrackingState, TrackingStatus,
};

/// Error type for FGS server operations.
pub type FgsError = HttpClientError;

/// Client for interacting with fgs_server HTTP API.
///
/// Works in both native Rust and WASM environments.
#[derive(Debug, Clone)]
pub struct FgsServerClient {
    http: HttpClient,
}

impl FgsServerClient {
    /// Create a new client pointing to the given base URL.
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL of the fgs_server (e.g., "http://localhost:3000")
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

    // === Tracking Control ===

    /// Get current tracking status.
    pub async fn get_tracking_status(&self) -> Result<TrackingStatus, FgsError> {
        self.http.get("/tracking/status").await
    }

    /// Enable or disable tracking.
    pub async fn set_tracking_enabled(&self, enabled: bool) -> Result<TrackingStatus, FgsError> {
        self.http
            .post("/tracking/enable", &TrackingEnableRequest { enabled })
            .await
    }

    /// Get tracking settings.
    pub async fn get_tracking_settings(&self) -> Result<TrackingSettings, FgsError> {
        self.http.get("/tracking/settings").await
    }

    /// Update tracking settings.
    pub async fn set_tracking_settings(&self, settings: &TrackingSettings) -> Result<(), FgsError> {
        self.http
            .post_no_response("/tracking/settings", settings)
            .await
    }

    /// Check if tracking is currently active (has lock).
    pub async fn is_tracking_active(&self) -> Result<bool, FgsError> {
        let status = self.get_tracking_status().await?;
        Ok(matches!(status.state, TrackingState::Tracking { .. }))
    }

    // === Export Control ===

    /// Get export status.
    pub async fn get_export_status(&self) -> Result<ExportStatus, FgsError> {
        self.http.get("/tracking/export").await
    }

    /// Update export settings.
    pub async fn set_export_settings(&self, settings: &ExportSettings) -> Result<(), FgsError> {
        self.http
            .post_no_response("/tracking/export", settings)
            .await
    }

    // === Camera Control ===

    /// Get camera statistics.
    pub async fn get_camera_stats(&self) -> Result<CameraStats, FgsError> {
        self.http.get("/stats").await
    }

    /// Get raw frame data.
    pub async fn get_raw_frame(&self) -> Result<RawFrameResponse, FgsError> {
        self.http.get("/raw").await
    }

    // === FSM Control ===

    /// Get FSM (Fast Steering Mirror) status.
    pub async fn get_fsm_status(&self) -> Result<FsmStatus, FgsError> {
        self.http.get("/fsm/status").await
    }

    /// Move FSM to a position.
    pub async fn move_fsm(&self, x_urad: f64, y_urad: f64) -> Result<(), FgsError> {
        self.http
            .post_no_response("/fsm/move", &FsmMoveRequest { x_urad, y_urad })
            .await
    }

    // === Star Detection Control ===

    /// Get star detection settings.
    pub async fn get_star_detection_settings(&self) -> Result<StarDetectionSettings, FgsError> {
        self.http.get("/detection/settings").await
    }

    /// Update star detection settings.
    pub async fn set_star_detection_settings(
        &self,
        settings: &StarDetectionSettings,
    ) -> Result<StarDetectionSettings, FgsError> {
        self.http.post("/detection/settings", settings).await
    }

    // === Image URLs ===

    /// Get the URL for JPEG frame endpoint.
    pub fn jpeg_url(&self) -> String {
        format!("{}/jpeg", self.http.base_url())
    }

    /// Get the URL for annotated frame endpoint.
    pub fn annotated_url(&self) -> String {
        format!("{}/annotated", self.http.base_url())
    }

    /// Get the URL for zoom-annotated frame endpoint.
    pub fn zoom_annotated_url(&self) -> String {
        format!("{}/zoom-annotated", self.http.base_url())
    }

    /// Get the URL for SVG overlay endpoint.
    pub fn overlay_svg_url(&self) -> String {
        format!("{}/overlay-svg", self.http.base_url())
    }

    /// Get the URL for tracking events SSE endpoint.
    pub fn tracking_events_url(&self) -> String {
        format!("{}/tracking/events", self.http.base_url())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_url_construction() {
        let client = FgsServerClient::new("http://localhost:3000");
        assert_eq!(client.base_url(), "http://localhost:3000");
        assert_eq!(client.jpeg_url(), "http://localhost:3000/jpeg");
        assert_eq!(
            client.tracking_events_url(),
            "http://localhost:3000/tracking/events"
        );
    }

    #[test]
    fn test_client_strips_trailing_slash() {
        let client = FgsServerClient::new("http://localhost:3000/");
        assert_eq!(client.base_url(), "http://localhost:3000");
    }
}

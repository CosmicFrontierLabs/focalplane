//! Shared types for test-bench backend and frontend.
//!
//! This crate contains lightweight serialization types that can be used
//! by both the Rust backend (test-bench) and WASM frontend (test-bench-frontend).
//! All types here must be WASM-compatible (no threading, no C bindings).

mod calibrate_client;
mod calibration;
mod camera;
mod fsm;
mod http_client;
mod log;
mod pattern;
mod star_detection;
pub mod stats_scan;
mod tracking;
mod types;

pub use calibrate_client::{CalibrateError, CalibrateServerClient, PatternConfigRequest};
pub use calibration::{
    ControlSpec, DisplayInfo, PatternConfigResponse, PatternSpec, SchemaResponse,
};
pub use camera::{CameraStats, CameraTimingStats, RawFrameResponse};
pub use fsm::{FsmMoveRequest, FsmStatus};
pub use http_client::HttpClientError;
pub use log::{LogEntry, LogLevel};
pub use pattern::{generate_centered_grid, PatternCommand};
pub use star_detection::{DetectedStar, StarDetectionResult, StarDetectionSettings};
pub use stats_scan::{StatsError, StatsScan};
pub use tracking::{
    TrackingEnableRequest, TrackingPosition, TrackingSettings, TrackingState, TrackingStatus,
};
pub use types::{HealthInfo, SpotShape, Timestamp};

use serde::{Deserialize, Serialize};

/// Typed WebSocket message for real-time FGS status streaming.
///
/// Sent from backend to frontend via `/ws/status` endpoint, replacing
/// HTTP polling. The tagged enum serializes as `{"type": "CameraStats", "data": {...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum FgsWsMessage {
    /// Camera statistics (fps, histogram, temperatures)
    CameraStats(CameraStats),
    /// Tracking system state and position
    TrackingStatus(TrackingStatus),
    /// Tracking algorithm settings
    TrackingSettings(TrackingSettings),
    /// FSM position and connection state
    FsmStatus(FsmStatus),
    /// Star detection algorithm settings
    DetectionSettings(StarDetectionSettings),
    /// Error response sent only to the commanding client
    CommandError(CommandError),
}

/// Client-to-server command sent over `/ws/status` WebSocket.
///
/// Sent as JSON text frames using the same tagged enum format as `FgsWsMessage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum FgsWsCommand {
    /// Enable or disable tracking
    SetTrackingEnabled(TrackingEnableRequest),
    /// Update tracking algorithm settings
    SetTrackingSettings(TrackingSettings),
    /// Update star detection settings
    SetDetectionSettings(StarDetectionSettings),
    /// Move FSM to position
    MoveFsm(FsmMoveRequest),
}

/// Error from a WebSocket command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandError {
    /// Which command failed (matches `FgsWsCommand` variant name)
    pub command: String,
    /// Human-readable error message
    pub message: String,
}

//! Shared types for test-bench backend and frontend.
//!
//! This crate contains lightweight serialization types that can be used
//! by both the Rust backend (test-bench) and WASM frontend (test-bench-frontend).
//! All types here must be WASM-compatible (no threading, no C bindings).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Current pattern configuration from calibrate_serve.
///
/// Returned via HTTP GET /config endpoint. Used by frontend to sync
/// state with server (e.g., after ZMQ commands or idle timeout).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatternConfigResponse {
    /// Pattern type identifier (e.g., "Crosshair", "Uniform")
    pub pattern_id: String,
    /// Pattern-specific parameter values (JSON object)
    pub values: serde_json::Value,
    /// Whether pattern colors are inverted
    pub invert: bool,
}

/// Display system information from calibrate_serve.
///
/// Returned via HTTP GET /info endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DisplayInfo {
    /// Display width in pixels
    pub width: u32,
    /// Display height in pixels
    pub height: u32,
    /// Pixel pitch in microns (None if unknown/unavailable)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub pixel_pitch_um: Option<f64>,
    /// Display name/identifier (e.g., "OLED 2560x2560")
    pub name: String,
}

/// Control specification for the frontend UI.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ControlSpec {
    IntRange {
        id: String,
        label: String,
        min: i64,
        max: i64,
        step: i64,
        default: i64,
    },
    FloatRange {
        id: String,
        label: String,
        min: f64,
        max: f64,
        step: f64,
        default: f64,
    },
    Bool {
        id: String,
        label: String,
        default: bool,
    },
    Text {
        id: String,
        label: String,
        default: String,
        placeholder: String,
    },
}

/// Pattern specification for the frontend UI.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatternSpec {
    pub id: String,
    pub name: String,
    pub controls: Vec<ControlSpec>,
}

/// Schema response containing all patterns and global controls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SchemaResponse {
    pub patterns: Vec<PatternSpec>,
    pub global_controls: Vec<ControlSpec>,
}

/// Pipeline timing statistics from camera_server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CameraTimingStats {
    pub avg_capture_ms: f32,
    pub avg_analysis_ms: f32,
    pub avg_render_ms: f32,
    pub avg_total_pipeline_ms: f32,
    pub capture_samples: usize,
    pub analysis_samples: usize,
}

/// Camera statistics from camera_server /stats endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CameraStats {
    pub total_frames: u64,
    pub avg_fps: f32,
    pub temperatures: HashMap<String, f64>,
    pub histogram: Vec<u32>,
    pub histogram_mean: f64,
    pub histogram_max: u16,
    /// Pipeline timing info (optional, frontend may ignore)
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<CameraTimingStats>,
    /// Camera device name
    pub device_name: String,
    /// Camera resolution width in pixels
    pub width: u32,
    /// Camera resolution height in pixels
    pub height: u32,
}

/// Raw frame response from camera_server /raw endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawFrameResponse {
    pub width: usize,
    pub height: usize,
    pub timestamp_sec: u64,
    pub timestamp_nanos: u64,
    pub temperatures: HashMap<String, f64>,
    pub exposure_us: u128,
    pub frame_number: u64,
    pub image_base64: String,
}

/// Health check response from orin_monitor /health endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthInfo {
    pub status: String,
    pub service: String,
    pub timestamp: u64,
}

/// Tracking state enum for the camera unified server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrackingState {
    /// System is idle, not tracking
    #[default]
    Idle,
    /// Acquiring frames to detect guide stars
    Acquiring { frames_collected: usize },
    /// Calibrating detected guide stars
    Calibrating,
    /// Actively tracking targets
    Tracking { frames_processed: usize },
    /// Lost track, attempting to reacquire
    Reacquiring { attempts: usize },
}

/// Current tracking position information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrackingPosition {
    /// Current track ID
    pub track_id: u32,
    /// X position in pixels
    pub x: f64,
    /// Y position in pixels
    pub y: f64,
    /// Signal-to-noise ratio of tracked target
    pub snr: f64,
    /// Timestamp of position measurement (seconds since epoch)
    pub timestamp_sec: u64,
    /// Nanoseconds component of timestamp
    pub timestamp_nanos: u64,
}

/// Full tracking status response from /tracking/status endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrackingStatus {
    /// Whether tracking mode is enabled
    pub enabled: bool,
    /// Current tracking state
    pub state: TrackingState,
    /// Current tracked position (if tracking)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<TrackingPosition>,
    /// Number of guide stars being tracked
    pub num_guide_stars: usize,
    /// Total tracking updates since tracking started
    pub total_updates: u64,
}

impl Default for TrackingStatus {
    fn default() -> Self {
        Self {
            enabled: false,
            state: TrackingState::Idle,
            position: None,
            num_guide_stars: 0,
            total_updates: 0,
        }
    }
}

/// Request to enable/disable tracking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrackingEnableRequest {
    pub enabled: bool,
}

/// Tracking algorithm settings that can be adjusted at runtime.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrackingSettings {
    /// Number of frames to collect during acquisition phase
    pub acquisition_frames: usize,
    /// Size of ROI around detected stars (pixels)
    pub roi_size: usize,
    /// Detection threshold in standard deviations above background
    pub detection_threshold_sigma: f64,
    /// Minimum SNR required to start tracking a star
    pub snr_min: f64,
    /// SNR threshold below which tracking is considered lost
    pub snr_dropout_threshold: f64,
    /// Expected Full Width at Half Maximum of stars (pixels)
    pub fwhm: f64,
}

impl Default for TrackingSettings {
    fn default() -> Self {
        Self {
            acquisition_frames: 5,
            roi_size: 64,
            detection_threshold_sigma: 5.0,
            snr_min: 10.0,
            snr_dropout_threshold: 3.0,
            fwhm: 7.0,
        }
    }
}

/// Export settings for tracking data recording.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExportSettings {
    /// Enable CSV export of tracking data
    pub csv_enabled: bool,
    /// CSV output filename (relative to working directory)
    pub csv_filename: String,
    /// Enable frame export as PNG + JSON metadata
    pub frames_enabled: bool,
    /// Directory for frame export (relative to working directory)
    pub frames_directory: String,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            csv_enabled: false,
            csv_filename: "tracking_data.csv".to_string(),
            frames_enabled: false,
            frames_directory: "frames".to_string(),
        }
    }
}

/// Export status showing current export statistics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ExportStatus {
    /// Number of CSV records written
    pub csv_records_written: u64,
    /// Number of frames exported
    pub frames_exported: u64,
    /// Current export settings
    pub settings: ExportSettings,
    /// Last export error (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

/// Metadata for an exported frame (written alongside PNG files).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FrameExportMetadata {
    /// Frame sequence number
    pub frame_number: usize,
    /// Timestamp seconds component
    pub timestamp_sec: u64,
    /// Timestamp nanoseconds component
    pub timestamp_nanos: u64,
    /// Current track ID (if tracking)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track_id: Option<u32>,
    /// Centroid X position (if tracking)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub centroid_x: Option<f64>,
    /// Centroid Y position (if tracking)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub centroid_y: Option<f64>,
    /// Frame width in pixels
    pub width: usize,
    /// Frame height in pixels
    pub height: usize,
}

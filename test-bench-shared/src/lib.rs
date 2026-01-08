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

//! Shared types for test-bench backend and frontend.
//!
//! This crate contains lightweight serialization types that can be used
//! by both the Rust backend (test-bench) and WASM frontend (test-bench-frontend).
//! All types here must be WASM-compatible (no threading, no C bindings).

mod calibrate_client;
mod fgs_client;
mod http_client;
mod ring_buffer;

pub use calibrate_client::{CalibrateError, CalibrateServerClient, PatternConfigRequest};
pub use fgs_client::{FgsError, FgsServerClient};
pub use http_client::HttpClientError;
pub use ring_buffer::RingBuffer;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

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

// ==================== FSM Control Types ====================

/// FSM controller status response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FsmStatus {
    /// Whether FSM is connected and initialized
    pub connected: bool,
    /// Current X position in microradians
    pub x_urad: f64,
    /// Current Y position in microradians
    pub y_urad: f64,
    /// Minimum X position in microradians
    pub x_min: f64,
    /// Maximum X position in microradians
    pub x_max: f64,
    /// Minimum Y position in microradians
    pub y_min: f64,
    /// Maximum Y position in microradians
    pub y_max: f64,
    /// Last error message (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

impl Default for FsmStatus {
    fn default() -> Self {
        Self {
            connected: false,
            x_urad: 0.0,
            y_urad: 0.0,
            x_min: 0.0,
            x_max: 2000.0,
            y_min: 0.0,
            y_max: 2000.0,
            last_error: None,
        }
    }
}

/// Request to move FSM to a position.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FsmMoveRequest {
    /// Target X position in microradians
    pub x_urad: f64,
    /// Target Y position in microradians
    pub y_urad: f64,
}

// ==================== Star Detection Settings ====================

/// Settings for real-time star detection overlay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StarDetectionSettings {
    /// Enable star detection overlay
    pub enabled: bool,
    /// Detection threshold in sigma above background (3.0 - 20.0)
    pub detection_sigma: f64,
    /// Minimum distance from bad pixels to consider valid (pixels)
    pub min_bad_pixel_distance: f64,
    /// Maximum aspect ratio for valid star (rejects elongated artifacts)
    pub max_aspect_ratio: f64,
    /// Minimum flux for valid detection
    pub min_flux: f64,
}

impl Default for StarDetectionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            detection_sigma: 5.0,
            min_bad_pixel_distance: 5.0,
            max_aspect_ratio: 2.5,
            min_flux: 100.0,
        }
    }
}

/// A detected star with position and shape info.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DetectedStar {
    /// X position (sub-pixel)
    pub x: f64,
    /// Y position (sub-pixel)
    pub y: f64,
    /// Total flux
    pub flux: f64,
    /// Aspect ratio (1.0 = circular)
    pub aspect_ratio: f64,
    /// Estimated diameter in pixels
    pub diameter: f64,
    /// Whether star passes quality filters (not near bad pixel, good shape)
    pub valid: bool,
}

/// Star detection result for a frame.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct StarDetectionResult {
    /// Detected stars
    pub stars: Vec<DetectedStar>,
    /// Detection time in milliseconds
    pub detection_time_ms: f32,
    /// Frame number this detection applies to
    pub frame_number: u64,
}

// ==================== Pattern Commands ====================

/// Commands for remote control of calibration displays.
///
/// Used to command what pattern the OLED display should show
/// via REST API or ZMQ.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum PatternCommand {
    /// Display a single Gaussian spot at the specified position.
    Spot {
        /// X position in display pixels (0 = left edge)
        x: f64,
        /// Y position in display pixels (0 = top edge)
        y: f64,
        /// Full-width at half-maximum in pixels
        fwhm: f64,
        /// Peak intensity (0.0 to 1.0, where 1.0 = white)
        intensity: f64,
    },

    /// Display multiple spots simultaneously.
    SpotGrid {
        /// List of (x, y) positions in display pixels
        positions: Vec<(f64, f64)>,
        /// Full-width at half-maximum in pixels (same for all spots)
        fwhm: f64,
        /// Peak intensity (0.0 to 1.0)
        intensity: f64,
    },

    /// Display uniform gray level across entire screen.
    Uniform {
        /// Gray level (0 = black, 255 = white)
        level: u8,
    },

    /// Clear display to black.
    Clear,
}

impl Default for PatternCommand {
    fn default() -> Self {
        Self::Clear
    }
}

impl PatternCommand {
    /// Create a spot command at the center of the display.
    pub fn centered_spot(width: u32, height: u32, fwhm: f64, intensity: f64) -> Self {
        Self::Spot {
            x: width as f64 / 2.0,
            y: height as f64 / 2.0,
            fwhm,
            intensity,
        }
    }

    /// Create a grid of spots centered on the display.
    pub fn centered_grid(
        width: u32,
        height: u32,
        grid_size: usize,
        spacing: f64,
        fwhm: f64,
        intensity: f64,
    ) -> Self {
        let positions = generate_centered_grid(grid_size, spacing, width, height);
        Self::SpotGrid {
            positions,
            fwhm,
            intensity,
        }
    }
}

/// Generate a centered grid of spot positions.
///
/// Creates a grid of `grid_size × grid_size` positions centered on the display,
/// with each position separated by `grid_spacing` pixels.
pub fn generate_centered_grid(
    grid_size: usize,
    grid_spacing: f64,
    display_width: u32,
    display_height: u32,
) -> Vec<(f64, f64)> {
    let center_x = display_width as f64 / 2.0;
    let center_y = display_height as f64 / 2.0;
    let half_extent = (grid_size - 1) as f64 / 2.0;

    let mut positions = Vec::with_capacity(grid_size * grid_size);
    for row in 0..grid_size {
        for col in 0..grid_size {
            let offset_x = (col as f64 - half_extent) * grid_spacing;
            let offset_y = (row as f64 - half_extent) * grid_spacing;
            positions.push((center_x + offset_x, center_y + offset_y));
        }
    }
    positions
}

// ==================== Core Types (moved from shared) ====================

/// Timestamp structure aligned with V4L2 format.
/// Represents time as seconds and nanoseconds since an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timestamp {
    /// Seconds component
    pub seconds: u64,
    /// Nanoseconds component (0-999,999,999)
    pub nanos: u64,
}

impl Timestamp {
    /// Create a new timestamp
    pub fn new(seconds: u64, nanos: u64) -> Self {
        Self { seconds, nanos }
    }

    /// Create a timestamp from a Duration since epoch
    pub fn from_duration(duration: Duration) -> Self {
        let total_nanos = duration.as_nanos();
        let seconds = (total_nanos / 1_000_000_000) as u64;
        let nanos = (total_nanos % 1_000_000_000) as u64;
        Self { seconds, nanos }
    }

    /// Convert to Duration
    pub fn to_duration(&self) -> Duration {
        Duration::new(self.seconds, self.nanos as u32)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{:09}", self.seconds, self.nanos)
    }
}

/// Spot shape characterization without position.
///
/// Contains flux, shape moments, and size measurements extracted from a centroid
/// calculation. Used for transmitting shape data separately from position
/// (e.g., in tracking messages where frame-relative position is stored separately).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotShape {
    /// Total flux (sum of all pixel intensities)
    pub flux: f64,
    /// Second central moment μ₂₀ (variance in x-direction)
    pub m_xx: f64,
    /// Second central moment μ₀₂ (variance in y-direction)
    pub m_yy: f64,
    /// Second central moment μ₁₁ (covariance between x and y)
    pub m_xy: f64,
    /// Aspect ratio (λ₁/λ₂) from eigenvalues of moment matrix
    pub aspect_ratio: f64,
    /// Estimated object diameter in pixels
    pub diameter: f64,
}

//! Tracking system types.

use serde::{Deserialize, Serialize};

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

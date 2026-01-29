use ndarray::Array2;
use std::sync::Arc;
use test_bench_shared::{SpotShape, Timestamp};

/// Position estimate for tracking callbacks
#[derive(Debug, Clone)]
pub struct PositionEstimate {
    /// X position in pixels
    pub x: f64,
    /// Y position in pixels
    pub y: f64,
    /// Timestamp from camera frame
    pub timestamp: Timestamp,
    /// Spot shape characterization (flux, moments, diameter)
    pub shape: SpotShape,
    /// Signal-to-noise ratio of tracked target
    pub snr: f64,
}

/// Events emitted for external callbacks
#[derive(Debug, Clone)]
pub enum FgsCallbackEvent {
    /// Tracking has started
    TrackingStarted {
        track_id: u32,
        initial_position: PositionEstimate,
        num_guide_stars: usize,
    },
    /// Tracking position update
    TrackingUpdate {
        track_id: u32,
        position: PositionEstimate,
    },
    /// Tracking has been lost
    TrackingLost {
        track_id: u32,
        reason: TrackingLostReason,
    },
    /// Frame size doesn't match expected ROI
    FrameSizeMismatch {
        expected_width: usize,
        expected_height: usize,
        actual_width: usize,
        actual_height: usize,
    },
    /// Frame processed (fired after tracking work completes)
    /// frame_data wrapped in Arc for efficient sharing - clone Arc to keep frame alive
    FrameProcessed {
        frame_number: usize,
        timestamp: Timestamp,
        frame_data: Arc<Array2<u16>>,
        track_id: Option<u32>,
        position: Option<PositionEstimate>,
    },
}

/// Reasons for tracking loss
#[derive(Debug, Clone)]
pub enum TrackingLostReason {
    /// Signal too weak to track
    SignalTooWeak,
    /// Target moved out of bounds
    OutOfBounds,
    /// User requested stop
    UserRequested,
    /// System error occurred
    SystemError(String),
}

/// Callback ID for registration/deregistration
pub type CallbackId = u64;

/// Callback function type
pub type FgsCallback = Arc<dyn Fn(&FgsCallbackEvent) + Send + Sync>;

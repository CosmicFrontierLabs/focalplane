use std::sync::Arc;

/// Position estimate for tracking callbacks
#[derive(Debug, Clone)]
pub struct PositionEstimate {
    /// X position in pixels
    pub x: f64,
    /// Y position in pixels
    pub y: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
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
        last_position: PositionEstimate,
        reason: TrackingLostReason,
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

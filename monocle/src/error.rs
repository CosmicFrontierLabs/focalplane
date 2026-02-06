use thiserror::Error;

/// Errors produced by the Fine Guidance System state machine.
#[derive(Error, Debug)]
pub enum FgsError {
    /// Frame dimensions do not match the accumulated frame.
    #[error("frame dimensions mismatch")]
    FrameDimensionMismatch,

    /// No accumulated frame available for calibration.
    #[error("no accumulated frame available for calibration")]
    NoAccumulatedFrame,

    /// No guide star selected for tracking.
    #[error("no guide star available for tracking")]
    NoGuideStar,

    /// No ROI available for tracking.
    #[error("no ROI available for tracking")]
    NoRoi,

    /// Could not compute an aligned ROI for the given star position.
    #[error("could not compute aligned ROI for star at ({x:.1}, {y:.1})")]
    RoiComputation {
        /// Star X position.
        x: f64,
        /// Star Y position.
        y: f64,
    },

    /// SNR dropped below the tracking threshold.
    #[error("SNR dropout: {measured:.2} < {threshold:.2}")]
    SnrDropout {
        /// Measured SNR value.
        measured: f64,
        /// Configured threshold.
        threshold: f64,
    },

    /// SNR calculation failed during tracking.
    #[error("SNR calculation failed: {0}")]
    SnrCalculation(String),

    /// Configuration validation failure.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

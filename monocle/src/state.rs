use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};
use test_bench_shared::Timestamp;

/// Fine Guidance System states
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FgsState {
    /// Waiting for START_FGS command
    Idle,
    /// Collecting frames for averaging
    Acquiring { frames_collected: usize },
    /// Detecting stars, selecting guides, setting references
    Calibrating,
    /// Continuous centroiding and FSM commanding
    Tracking { frames_processed: usize },
    /// Attempting to recover lost stars
    Reacquiring { attempts: usize },
}

/// Events that trigger state transitions
#[derive(Debug, Clone)]
pub enum FgsEvent<'a> {
    /// Start the FGS
    StartFgs,
    /// Abort current operation
    Abort,
    /// Stop FGS (graceful shutdown)
    StopFgs,
    /// Process a new image frame with timestamp
    ProcessFrame(ArrayView2<'a, u16>, Timestamp),
}

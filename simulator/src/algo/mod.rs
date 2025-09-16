//! Algorithm module specific to simulator.

pub mod misc;
pub mod selection;

// Re-export all shared algo functionality
pub use shared::algo::*;

// Simulator-specific exports
pub use selection::{GuideStarQuality, GuideStarSelector};

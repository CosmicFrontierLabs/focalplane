//! Image processing functionality specific to simulator.
//!
//! This module contains image processing functions that depend on
//! simulator-specific types and are not suitable for the shared module.

pub mod render;

// Re-export all shared image_proc functionality
pub use shared::image_proc::*;

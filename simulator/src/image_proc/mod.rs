//! Image processing module for telescope simulation
//!
//! This module provides image processing utilities for the telescope simulator,
//! including convolution, filtering, and other operations needed for
//! realistic image generation.

pub mod convolve2d;

// Re-export key functionality for easier access
pub use convolve2d::{convolve2d, ConvolveOptions};
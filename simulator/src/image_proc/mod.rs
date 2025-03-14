//! Image processing module for telescope simulation
//!
//! This module provides image processing utilities for the telescope simulator,
//! including convolution, filtering, thresholding, centroid calculation, and
//! other operations needed for realistic image generation and analysis.

pub mod centroid;
pub mod convolve2d;
pub mod overlay;
pub mod thresholding;

// Re-export key functionality for easier access
pub use centroid::{detect_stars, get_centroids, StarDetection};
pub use convolve2d::{convolve2d, gaussian_kernel, ConvolveMode, ConvolveOptions};
pub use overlay::{
    draw_bounding_boxes, draw_simple_boxes, draw_stars_with_sizes, overlay_to_image,
};
pub use thresholding::{
    apply_threshold, connected_components, get_bounding_boxes, merge_overlapping_boxes,
    otsu_threshold,
};

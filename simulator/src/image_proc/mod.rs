//! Image processing module for telescope simulation
//!
//! This module provides image processing utilities for the telescope simulator,
//! including convolution, filtering, thresholding, centroid calculation, and
//! other operations needed for realistic image generation and analysis.

pub mod aabb;
pub mod centroid;
pub mod convolve2d;
pub mod histogram_stretch;
pub mod image;
pub mod io;
pub mod noise;
pub mod overlay;
pub mod render;
pub mod segment;
pub mod smear;
pub mod thresholding;

// Re-export key functionality for easier access
pub use aabb::{aabbs_to_tuples, merge_overlapping_aabbs, tuples_to_aabbs, union_aabbs, AABB};
pub use centroid::{detect_stars, get_centroids, StarDetection};
pub use convolve2d::{convolve2d, gaussian_kernel, ConvolveMode, ConvolveOptions};
pub use histogram_stretch::stretch_histogram;
pub use io::{save_u8_image, u16_to_u8_auto_scale, u16_to_u8_scaled};
pub use noise::{generate_noise_with_precomputed_params, generate_sensor_noise};
pub use overlay::{
    draw_bounding_boxes, draw_simple_boxes, draw_stars_with_sizes, draw_stars_with_x_markers,
    overlay_to_image,
};
pub use thresholding::{apply_threshold, connected_components, get_bounding_boxes, otsu_threshold};

//! Comprehensive image processing pipeline for astronomical telescope simulation.
//!
//! This module provides a complete suite of image processing algorithms specifically
//! designed for astronomical applications. From raw sensor simulation to sophisticated
//! object detection, it handles the entire pipeline of space telescope image analysis.
//!
//! # Module Organization
//!
//! ## Core Algorithms
//! - **airy**: Point spread function modeling for diffraction-limited optics
//! - **convolve2d**: 2D convolution with Gaussian kernels for PSF application
//! - **noise**: Realistic sensor noise models (read noise, dark current, shot noise)
//!
//! ## Object Detection
//! - **detection**: Multi-algorithm star detection (DAO, IRAF, naive centroiding)
//! - **detection::thresholding**: Otsu thresholding and connected component analysis
//! - **detection::aabb**: Bounding box management for detected objects
//!
//! ## Image Enhancement
//! - **histogram_stretch**: Contrast enhancement for faint object visibility
//! - **render**: High-quality rendering of astronomical scenes
//! - **overlay**: Visualization overlays for detection results
//!
//! ## Data I/O
//! - **io**: FITS and standard image format support with bit depth conversion
//! - **image**: Format conversions between ndarray and image crate types
//!
//! ## Specialized Effects
//! - **smear**: Pixel smear simulation for realistic sensor effects
//!
//! # Performance Considerations
//!
//! - **Memory efficiency**: Operates on array views to minimize copying
//! - **Vectorized operations**: Uses ndarray for optimized numerical computing
//! - **Algorithm selection**: Multiple detection algorithms for different use cases
//! - **Bit depth optimization**: Efficient conversions between u16 (sensor) and u8 (display)

pub mod airy;
pub mod aperture_photometry;
pub mod centroid;
pub mod contamination;
pub mod convolve2d;
pub mod detection;
pub mod histogram_stretch;
pub mod image;
pub mod io;
pub mod noise;
pub mod overlay;
pub mod smear;
pub mod source_snr;
pub mod test_patterns;

// Re-export key functionality for easier access
pub use airy::AiryDisk;
pub use aperture_photometry::collect_aperture_pixels;
pub use convolve2d::{convolve2d, gaussian_kernel, ConvolveMode, ConvolveOptions};
pub use detection::{
    aabbs_to_tuples, apply_threshold, connected_components, detect_stars, detect_stars_unified,
    get_bounding_boxes, get_centroids, merge_overlapping_aabbs, otsu_threshold, tuples_to_aabbs,
    union_aabbs, StarDetection, StarFinder, AABB,
};
pub use histogram_stretch::stretch_histogram;
pub use image::{
    array2_to_gray16_image, array2_to_gray_image, gray16_image_to_array2, gray_image_to_array2,
    u16_to_gray_image, Gray16Image,
};
pub use io::{save_u8_image, u16_to_u8_auto_scale, u16_to_u8_scaled};
pub use noise::generate_noise_with_precomputed_params;
pub use overlay::{
    draw_bounding_boxes, draw_simple_boxes, draw_stars_with_sizes, draw_stars_with_x_markers,
    overlay_to_image,
};
pub use source_snr::{calculate_snr, filter_by_snr};

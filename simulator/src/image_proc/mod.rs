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
//! # Processing Pipeline Example
//!
//! ```rust
//! use simulator::image_proc::{
//!     gaussian_kernel, convolve2d, ConvolveMode, ConvolveOptions,
//!     generate_sensor_noise, detect_stars_unified, StarFinder,
//!     stretch_histogram, save_u8_image, u16_to_u8_auto_scale
//! };
//! use simulator::image_proc::airy::PixelScaledAiryDisk;
//! use simulator::hardware::sensor::models::GSENSE6510BSI;
//! use simulator::units::{LengthExt, Wavelength};
//! use ndarray::Array2;
//! use std::time::Duration;
//!
//! // 1. Start with clean astronomical image
//! let mut image = Array2::from_elem((64, 64), 1000u16);  // Sky background
//! image[[32, 32]] = 45000;  // Bright star
//! image[[16, 16]] = 8000;   // Faint star
//!
//! // 2. Add realistic telescope PSF
//! let image_f64 = image.mapv(|x| x as f64);
//! let psf_kernel = gaussian_kernel(5, 1.2);  // 2.5 pixel FWHM
//! let psf_convolved = convolve2d(
//!     &image_f64.view(),
//!     &psf_kernel.view(),
//!     Some(ConvolveOptions { mode: ConvolveMode::Same })
//! );
//!
//! // 3. Add sensor noise
//! let sensor = GSENSE6510BSI.clone().with_dimensions(64, 64);
//! let exposure = Duration::from_secs(10);
//! let noise = generate_sensor_noise(&sensor, &exposure, -20.0, Some(42));
//! let with_noise = psf_convolved.mapv(|x| x as u16) + noise.mapv(|x| x as u16);
//!
//! // 4. Detect astronomical sources
//! let airy_disk = PixelScaledAiryDisk::with_fwhm(2.5, Wavelength::from_nanometers(550.0));
//! let stars = detect_stars_unified(
//!     with_noise.view(),
//!     StarFinder::Dao,
//!     &airy_disk,   // Airy disk
//!     20.0,  // Background RMS
//!     5.0    // 5-sigma detection
//! ).unwrap();
//!
//! // 5. Enhance for visualization
//! let enhanced = stretch_histogram(with_noise.view(), 2.0, 98.0);
//! let display = u16_to_u8_auto_scale(&enhanced);
//!
//! println!("Processed image: detected {} stars", stars.len());
//! // save_u8_image(&display, "processed_starfield.png").unwrap();
//! ```
//!
//! # Performance Considerations
//!
//! - **Memory efficiency**: Operates on array views to minimize copying
//! - **Vectorized operations**: Uses ndarray for optimized numerical computing
//! - **Algorithm selection**: Multiple detection algorithms for different use cases
//! - **Bit depth optimization**: Efficient conversions between u16 (sensor) and u8 (display)

pub mod airy;
pub mod convolve2d;
pub mod detection;
pub mod histogram_stretch;
pub mod image;
pub mod io;
pub mod noise;
pub mod overlay;
pub mod render;
pub mod smear;
pub mod test_patterns;

// Re-export key functionality for easier access
pub use airy::AiryDisk;
pub use convolve2d::{convolve2d, gaussian_kernel, ConvolveMode, ConvolveOptions};
pub use detection::{
    aabbs_to_tuples, apply_threshold, connected_components, detect_stars, detect_stars_unified,
    do_detections, get_bounding_boxes, get_centroids, merge_overlapping_aabbs, otsu_threshold,
    tuples_to_aabbs, union_aabbs, StarDetection, StarFinder, AABB,
};
pub use histogram_stretch::stretch_histogram;
pub use io::{save_u8_image, u16_to_u8_auto_scale, u16_to_u8_scaled};
pub use noise::{generate_noise_with_precomputed_params, generate_sensor_noise};
pub use overlay::{
    draw_bounding_boxes, draw_simple_boxes, draw_stars_with_sizes, draw_stars_with_x_markers,
    overlay_to_image,
};

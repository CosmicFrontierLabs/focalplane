//! Astronomical object detection algorithms and utilities.
//!
//! This module provides a comprehensive suite of star and object detection algorithms
//! for astronomical image analysis. Includes both sophisticated detection methods
//! (DAO, IRAF) and simple approaches (naive centroiding, thresholding) suitable for
//! different astronomical applications and performance requirements.
//!
//! # Module Organization
//!
//! - **detection**: Unified interface for multiple detection algorithms
//! - **config**: Pre-optimized configurations for space telescope observations
//! - **naive**: Simple centroiding-based detection for basic applications
//! - **thresholding**: Threshold-based detection and connected component analysis
//! - **aabb**: Axis-aligned bounding boxes for region management
//!
//! # Algorithm Comparison
//!
//! | Algorithm | Speed | Accuracy | Best Use Case |
//! |-----------|-------|----------|---------------|
//! | DAO       | Slow  | Highest  | Crowded fields, faint sources |
//! | IRAF      | Medium| High     | General-purpose detection |
//! | Naive     | Fast  | Medium   | Bright, isolated sources |
//! | Threshold | Fastest| Basic   | Quick analysis, preprocessing |
//!
//! # Examples
//!
//! ```rust
//! use simulator::image_proc::detection::{detect_stars_unified, StarFinder, detect_stars};
//! use simulator::image_proc::airy::PixelScaledAiryDisk;
//! use ndarray::Array2;
//!
//! // Create a test image
//! let image = Array2::from_elem((100, 100), 10u16);
//!
//! // Method 1: Unified interface with automatic parameter tuning
//! let airy_disk = PixelScaledAiryDisk::with_fwhm(2.5);
//! let stars = detect_stars_unified(
//!     image.view(),
//!     StarFinder::Dao,
//!     &airy_disk,  // Airy disk
//!     1.0,  // Background RMS
//!     5.0   // 5-sigma threshold
//! ).unwrap();
//!
//! // Method 2: Direct naive detection for speed
//! let image_f64 = image.mapv(|x| x as f64);
//! let quick_stars = detect_stars(&image_f64.view(), Some(50.0));
//!
//! println!("Sophisticated: {} stars, Quick: {} stars",
//!          stars.len(), quick_stars.len());
//! ```

pub mod aabb;
pub mod config;
pub mod naive;
pub mod thresholding;
pub mod unified;

pub use aabb::*;
pub use naive::{
    calculate_star_centroid, detect_stars, do_detections, get_centroids, StarDetection,
};
pub use thresholding::*;
pub use unified::{detect_stars as detect_stars_unified, StarFinder};

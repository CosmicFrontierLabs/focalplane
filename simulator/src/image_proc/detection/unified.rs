//! Unified star detection interface for astronomical image analysis.
//!
//! This module provides a single, consistent interface for multiple star detection
//! algorithms, including professional-grade DAO and IRAF methods as well as simple
//! centroiding approaches. Automatically handles algorithm-specific parameter tuning
//! and provides performance monitoring for large-scale astronomical surveys.
//!
//! # Available Algorithms
//!
//! ## DAOStarFinder
//! Based on Stetson's DAOPHOT package, the gold standard for stellar photometry.
//! - **Best for**: Crowded fields, high-precision photometry
//! - **Features**: Shape filtering, PSF fitting, artifact rejection
//! - **Performance**: Slower but most accurate for complex fields
//!
//! ## IRAFStarFinder  
//! Based on IRAF's DAOFIND task, simpler than DAO but still robust.
//! - **Best for**: Well-separated stars, faster processing
//! - **Features**: Basic shape filtering, threshold detection
//! - **Performance**: Moderate speed and accuracy
//!
//! ## Naive Centroiding
//! Simple center-of-mass detection for basic applications.
//! - **Best for**: Bright, isolated stars, quick analysis
//! - **Features**: Fast threshold + centroiding
//! - **Performance**: Fastest but least sophisticated
//!
//! # Usage
//!
//! Provides unified interface for multiple star detection algorithms
//! with automatic parameter optimization for space telescope observations.

use ndarray::ArrayView2;
use starfield::image::starfinders::{DAOStarFinder, IRAFStarFinder, StellarSource};

use super::config::{dao_autoconfig, iraf_autoconfig};
use crate::image_proc::airy::PixelScaledAiryDisk;

/// Available star detection algorithms with different complexity/performance tradeoffs.
///
/// Each algorithm represents a different approach to stellar source detection,
/// from simple thresholding to sophisticated PSF fitting methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarFinder {
    /// DAO (DAOPHOT) photometry algorithm - most sophisticated detection.
    ///
    /// Based on Stetson's DAOPHOT package. Provides excellent rejection of
    /// cosmic rays and image artifacts through shape filtering.
    /// Best for crowded fields and high-precision photometry.
    Dao,
    /// IRAF DAOFIND algorithm - balanced performance and accuracy.
    ///
    /// Simpler than DAO but still includes basic shape filtering.
    /// Good compromise between speed and detection quality.
    Iraf,
    /// Naive centroiding algorithm - fastest but simplest detection.
    ///
    /// Basic threshold detection followed by center-of-mass calculation.
    /// Suitable for bright, well-separated stars or quick analysis.
    Naive,
}

impl std::str::FromStr for StarFinder {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dao" => Ok(StarFinder::Dao),
            "iraf" => Ok(StarFinder::Iraf),
            "naive" => Ok(StarFinder::Naive),
            _ => Err(format!(
                "Unknown star finder: {}. Valid options: dao, iraf, naive",
                s
            )),
        }
    }
}

/// Detect stars in an astronomical image using the specified algorithm.
///
/// Provides a unified interface for multiple star detection algorithms with automatic
/// parameter optimization based on telescope characteristics. Includes performance
/// monitoring and detailed logging for survey operations.
///
/// # Algorithm Selection Guide
/// - **DAO**: Use for crowded fields, faint sources, or when high precision is needed
/// - **IRAF**: Good balance of speed and accuracy for most applications
/// - **Naive**: Fast processing for bright, isolated stars or quick analysis
///
/// # Arguments
/// * `image` - The input image as a 2D array view
/// * `algorithm` - The star detection algorithm to use
/// * `scaled_airy_disk` - ScaledAiryDisk representing the PSF characteristics
/// * `background_rms` - RMS noise level of the background
/// * `detection_sigma` - Detection threshold in units of sigma (typically 5.0)
///
/// # Returns
/// Result containing vector of detected stellar sources, or error message
///
/// # Performance
/// Automatically logs detection timing and efficiency metrics at debug level.
/// Typical performance on modern hardware:
/// - DAO: ~50-100 ns/pixel
/// - IRAF: ~20-50 ns/pixel  
/// - Naive: ~5-10 ns/pixel
///
/// # Usage
/// Unified interface for DAO, IRAF, and naive star detection algorithms
/// with automatic parameter optimization and performance monitoring.
pub fn detect_stars(
    image: ArrayView2<u16>,
    algorithm: StarFinder,
    scaled_airy_disk: &PixelScaledAiryDisk,
    background_rms: f64,
    detection_sigma: f64,
) -> Result<Vec<Box<dyn StellarSource>>, String> {
    let start_time = std::time::Instant::now();
    let (height, width) = image.dim();
    let total_pixels = height * width;

    let result = match algorithm {
        StarFinder::Dao => detect_dao(image, scaled_airy_disk, background_rms, detection_sigma),
        StarFinder::Iraf => detect_iraf(image, scaled_airy_disk, background_rms, detection_sigma),
        StarFinder::Naive => detect_naive(image, detection_sigma * background_rms),
    };

    let duration = start_time.elapsed();
    let time_per_pixel_ns = duration.as_nanos() as f64 / total_pixels as f64;
    let stars_found = result.as_ref().map(|s| s.len()).unwrap_or(0);

    log::debug!("Star detection: algorithm={:?}, duration={:.3}ms, stars_found={}, time_per_pixel={:.2}ns/pixel", 
               algorithm, duration.as_secs_f64() * 1000.0, stars_found, time_per_pixel_ns);

    result
}

/// Internal DAO star detection implementation.
///
/// Uses the DAOStarFinder algorithm with space-telescope optimized parameters.
/// Includes automatic type conversion and error handling.
fn detect_dao(
    image: ArrayView2<u16>,
    scaled_airy_disk: &PixelScaledAiryDisk,
    background_rms: f64,
    detection_sigma: f64,
) -> Result<Vec<Box<dyn StellarSource>>, String> {
    // Convert u16 image to f64 for DAO algorithm
    let image_f64 = image.mapv(|x| x as f64);

    // Use optimized configuration for space telescope
    let config = dao_autoconfig(scaled_airy_disk, background_rms, detection_sigma);

    // Create DAO star finder and detect sources
    let star_finder = DAOStarFinder::new(config)
        .map_err(|e| format!("DAO star finder creation failed: {}", e))?;

    let stars = star_finder
        .find_stars(&image_f64, None)
        .into_iter()
        .map(|star| Box::new(star) as Box<dyn StellarSource>)
        .collect();

    Ok(stars)
}

/// Internal IRAF star detection implementation.
///
/// Uses the IRAFStarFinder algorithm with space-telescope optimized parameters.
/// Includes automatic type conversion and error handling.
fn detect_iraf(
    image: ArrayView2<u16>,
    scaled_airy_disk: &PixelScaledAiryDisk,
    background_rms: f64,
    detection_sigma: f64,
) -> Result<Vec<Box<dyn StellarSource>>, String> {
    // Convert u16 image to f64 for IRAF algorithm
    let image_f64 = image.mapv(|x| x as f64);

    // Use optimized configuration for space telescope
    let config = iraf_autoconfig(scaled_airy_disk, background_rms, detection_sigma);

    // Create IRAF star finder and detect sources
    let star_finder = IRAFStarFinder::new(config)
        .map_err(|e| format!("IRAF star finder creation failed: {}", e))?;

    let stars = star_finder
        .find_stars(&image_f64, None)
        .into_iter()
        .map(|star| Box::new(star) as Box<dyn StellarSource>)
        .collect();

    Ok(stars)
}

/// Internal naive star detection implementation.
///
/// Uses simple threshold detection followed by center-of-mass centroiding.
/// Fastest algorithm but least sophisticated.
fn detect_naive(
    image: ArrayView2<u16>,
    threshold: f64,
) -> Result<Vec<Box<dyn StellarSource>>, String> {
    // Convert u16 image to f64 for centroiding algorithm
    let image_f64 = image.mapv(|x| x as f64);
    let image_view = image_f64.view();

    // Use the existing centroiding detection from the centroid module
    let detections = super::naive::detect_stars(&image_view, Some(threshold));

    // Convert StarDetection objects to boxed StellarSource
    let stars = detections
        .into_iter()
        .map(|detection| Box::new(detection) as Box<dyn StellarSource>)
        .collect();

    Ok(stars)
}

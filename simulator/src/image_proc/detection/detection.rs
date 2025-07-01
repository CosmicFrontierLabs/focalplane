//! Star detection algorithms for astronomical images
//!
//! This module provides a unified interface for different star detection algorithms
//! including DAO, IRAF, and naive centroiding approaches.

use ndarray::ArrayView2;
use starfield::image::starfinders::{DAOStarFinder, IRAFStarFinder, StellarSource};

use super::config::{dao_autoconfig, iraf_autoconfig};

/// Enumeration of available star detection algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarFinder {
    /// DAO (Daophot) photometry algorithm
    Dao,
    /// IRAF-style photometry algorithm  
    Iraf,
    /// Naive centroiding algorithm (center of mass)
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

/// Detect stars in an image using the specified algorithm
///
/// # Arguments
/// * `image` - The input image as a 2D array view
/// * `algorithm` - The star detection algorithm to use
/// * `airy_disk_pixels` - Airy disk diameter in pixels
/// * `background_rms` - RMS noise level of the background
/// * `detection_sigma` - Detection threshold in units of sigma (typically 5.0)
///
/// # Returns
/// A Result containing a vector of objects implementing the StellarSource trait
pub fn detect_stars(
    image: ArrayView2<u16>,
    algorithm: StarFinder,
    airy_disk_pixels: f64,
    background_rms: f64,
    detection_sigma: f64,
) -> Result<Vec<Box<dyn StellarSource>>, String> {
    let start_time = std::time::Instant::now();
    let (height, width) = image.dim();
    let total_pixels = height * width;

    let result = match algorithm {
        StarFinder::Dao => detect_dao(image, airy_disk_pixels, background_rms, detection_sigma),
        StarFinder::Iraf => detect_iraf(image, airy_disk_pixels, background_rms, detection_sigma),
        StarFinder::Naive => detect_naive(image, detection_sigma * background_rms),
    };

    let duration = start_time.elapsed();
    let time_per_pixel_ns = duration.as_nanos() as f64 / total_pixels as f64;
    let stars_found = result.as_ref().map(|s| s.len()).unwrap_or(0);

    log::debug!("Star detection: algorithm={:?}, duration={:.3}ms, stars_found={}, time_per_pixel={:.2}ns/pixel", 
               algorithm, duration.as_secs_f64() * 1000.0, stars_found, time_per_pixel_ns);

    result
}

/// Internal function for DAO star detection using DAOStarFinder
fn detect_dao(
    image: ArrayView2<u16>,
    airy_disk_pixels: f64,
    background_rms: f64,
    detection_sigma: f64,
) -> Result<Vec<Box<dyn StellarSource>>, String> {
    // Convert u16 image to f64 for DAO algorithm
    let image_f64 = image.mapv(|x| x as f64);

    // Use optimized configuration for space telescope
    let config = dao_autoconfig(airy_disk_pixels, background_rms, detection_sigma);

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

/// Internal function for IRAF star detection using IRAFStarFinder
fn detect_iraf(
    image: ArrayView2<u16>,
    airy_disk_pixels: f64,
    background_rms: f64,
    detection_sigma: f64,
) -> Result<Vec<Box<dyn StellarSource>>, String> {
    // Convert u16 image to f64 for IRAF algorithm
    let image_f64 = image.mapv(|x| x as f64);

    // Use optimized configuration for space telescope
    let config = iraf_autoconfig(airy_disk_pixels, background_rms, detection_sigma);

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

/// Internal function for naive star detection using centroiding
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

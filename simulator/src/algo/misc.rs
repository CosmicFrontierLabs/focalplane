//! Miscellaneous mathematical and utility algorithms.
//!
//! This module provides general-purpose mathematical functions and utilities
//! that don't fit into more specific algorithm categories. Currently includes:
//!
//! - **Linear interpolation**: Fast 1D interpolation with error handling
//! - **Numerical utilities**: Common mathematical operations for scientific computing
//! - **WWT utilities**: Functions for generating WorldWide Telescope overlay URLs
//!
//! These functions are designed for performance and robustness in scientific
//! applications, with comprehensive error handling and input validation.

use crate::hardware::SatelliteConfig;
use crate::star_math::field_diameter;
use starfield::Equatorial;
use std::f64::consts::PI;
use thiserror::Error;
use url::Url;

/// Errors that can occur during interpolation operations.
///
/// This enum provides detailed error information for interpolation failures,
/// allowing callers to handle different error conditions appropriately.
#[derive(Error, Debug)]
pub enum InterpError {
    #[error("Value {0} is out of bounds for interpolation range [{1}, {2}]")]
    OutOfBounds(f64, f64, f64),
    #[error("Input vectors must have at least 2 points")]
    InsufficientData,
    #[error("Input vectors must have the same length")]
    MismatchedLengths,
    #[error("X values must be sorted in ascending order")]
    UnsortedData,
}

/// Performs linear interpolation on 1D data using binary search for efficiency.
///
/// This function implements fast linear interpolation by:
/// 1. Validating input data (lengths, sorting, sufficient points)
/// 2. Using binary search to find the correct interval (O(log n))
/// 3. Applying linear interpolation formula: y = y₁ + t(y₂ - y₁)
///
/// where t = (x - x₁)/(x₂ - x₁) is the interpolation parameter.
///
/// # Arguments
///
/// * `x` - The x-coordinate at which to interpolate
/// * `xs` - Array of x-coordinates (must be sorted in ascending order)
/// * `ys` - Array of corresponding y-values (must match length of xs)
///
/// # Returns
///
/// * `Ok(f64)` - The interpolated y-value at position x
/// * `Err(InterpError)` - Detailed error if interpolation fails
///
/// # Performance
///
/// - Time complexity: O(log n) due to binary search
/// - Space complexity: O(1)
/// - Suitable for repeated queries on the same dataset
///
///
/// # Errors
///
/// * `InterpError::OutOfBounds` - x is outside the range \\[xs\\[0\\], xs\\[n-1\\]\\]
/// * `InterpError::InsufficientData` - Less than 2 data points provided
/// * `InterpError::MismatchedLengths` - xs and ys have different lengths
/// * `InterpError::UnsortedData` - xs array is not sorted in ascending order
pub fn interp(x: f64, xs: &[f64], ys: &[f64]) -> Result<f64, InterpError> {
    if xs.len() != ys.len() {
        return Err(InterpError::MismatchedLengths);
    }

    if xs.len() < 2 {
        return Err(InterpError::InsufficientData);
    }

    // Check if xs is sorted
    for i in 1..xs.len() {
        if xs[i] <= xs[i - 1] {
            return Err(InterpError::UnsortedData);
        }
    }

    let min_x = xs[0];
    let max_x = xs[xs.len() - 1];

    if x < min_x || x > max_x {
        return Err(InterpError::OutOfBounds(x, min_x, max_x));
    }

    // Binary search for the correct interval
    let idx = match xs.binary_search_by(|probe| probe.partial_cmp(&x).unwrap()) {
        Ok(exact_idx) => return Ok(ys[exact_idx]), // Exact match
        Err(insert_idx) => insert_idx,
    };

    // Linear interpolation between points
    let i1 = idx - 1;
    let i2 = idx;

    let x1 = xs[i1];
    let x2 = xs[i2];
    let y1 = ys[i1];
    let y2 = ys[i2];

    let t = (x - x1) / (x2 - x1);
    Ok(y1 + t * (y2 - y1))
}

/// Normalizes a vector of values to the range [0, 1] based on its maximum value.
///
/// This function finds the maximum value in the input vector and divides all
/// elements by this maximum, resulting in a normalized vector where the largest
/// value becomes 1.0. If all values are zero or negative, returns the original
/// vector unchanged.
///
/// # Arguments
///
/// * `pts` - Vector of floating point values to normalize
///
/// # Returns
///
/// A new vector with normalized values in the range [0, 1]
///
/// # Example
///
/// ```
/// use simulator::algo::misc::normalize;
///
/// let data = vec![0.5, 2.0, 1.0, 3.0];
/// let normalized = normalize(data);
/// assert_eq!(normalized, vec![0.5/3.0, 2.0/3.0, 1.0/3.0, 1.0]);
/// ```
pub fn normalize(pts: Vec<f64>) -> Vec<f64> {
    // Find maximum value using iterator max_by
    let max_val = pts
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(0.0);

    // If max is zero or negative, return original vector
    if max_val <= 0.0 {
        return pts;
    }

    // Normalize all values by the maximum
    pts.into_iter().map(|val| val / max_val).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0];
        assert_eq!(interp(2.0, &xs, &ys).unwrap(), 20.0);
    }

    #[test]
    fn test_linear_interpolation() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        assert_eq!(interp(1.5, &xs, &ys).unwrap(), 15.0);
        assert_eq!(interp(2.5, &xs, &ys).unwrap(), 25.0);
    }

    #[test]
    fn test_out_of_bounds() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        assert!(matches!(
            interp(0.5, &xs, &ys),
            Err(InterpError::OutOfBounds(_, _, _))
        ));
        assert!(matches!(
            interp(3.5, &xs, &ys),
            Err(InterpError::OutOfBounds(_, _, _))
        ));
    }

    #[test]
    fn test_mismatched_lengths() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0];
        assert!(matches!(
            interp(1.5, &xs, &ys),
            Err(InterpError::MismatchedLengths)
        ));
    }

    #[test]
    fn test_insufficient_data() {
        let xs = vec![1.0];
        let ys = vec![10.0];
        assert!(matches!(
            interp(1.0, &xs, &ys),
            Err(InterpError::InsufficientData)
        ));
    }

    #[test]
    fn test_unsorted_data() {
        let xs = vec![2.0, 1.0, 3.0];
        let ys = vec![20.0, 10.0, 30.0];
        assert!(matches!(
            interp(1.5, &xs, &ys),
            Err(InterpError::UnsortedData)
        ));
    }

    #[test]
    fn test_normalize() {
        // Basic normalization
        let data = vec![0.5, 2.0, 1.0, 3.0];
        let normalized = normalize(data);
        assert_eq!(normalized, vec![0.5 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0]);

        // All zeros
        let zeros = vec![0.0, 0.0, 0.0];
        let result = normalize(zeros.clone());
        assert_eq!(result, zeros);

        // Negative values
        let negative = vec![-1.0, -2.0, -3.0];
        let result = normalize(negative.clone());
        assert_eq!(result, negative);

        // Single value
        let single = vec![5.0];
        let result = normalize(single);
        assert_eq!(result, vec![1.0]);
    }
}

/// Converts Right Ascension from Hours, Minutes, Seconds to Decimal Degrees.
/// RA (Hours) * 15 = RA (Degrees)
pub fn ra_hms_to_deg(hours: f64, minutes: f64, seconds: f64) -> f64 {
    (hours + minutes / 60.0 + seconds / 3600.0) * 15.0
}

/// Converts Declination from Degrees, Minutes, Seconds to Decimal Degrees.
/// Handles negative declination correctly.
pub fn dec_dms_to_deg(degrees: f64, minutes: f64, seconds: f64) -> f64 {
    let sign = if degrees < 0.0 { -1.0 } else { 1.0 };
    (degrees.abs() + minutes / 60.0 + seconds / 3600.0) * sign
}

/// Generates a WorldWide Telescope ShowImage.aspx URL with a transparent overlay.
///
/// # Arguments
/// * `object_name` - A descriptive name for the target.
/// * `coordinates` - Equatorial coordinates (RA/Dec) of the center of the box.
/// * `satellite` - Satellite configuration containing telescope and sensor parameters.
/// * `rotation_deg` - Rotation of the image on the sky in degrees (0 = North up, East left).
/// * `text` - Optional text to display in the center of the overlay.
/// * `text_color` - Hex color code for the text (e.g., "FFFFFF" for white).
/// * `background_color` - Hex color code for the background (e.g., "00000000" for transparent).
///
/// # Returns
/// A `Result` containing the generated URL as a `String` or an `url::ParseError`.
pub fn generate_wwt_overlay_url(
    object_name: &str,
    coordinates: &Equatorial,
    satellite: &SatelliteConfig,
    rotation_deg: f64,
    text: Option<&str>,
    text_color: &str,
    background_color: &str,
) -> Result<String, url::ParseError> {
    // Calculate field of view from satellite configuration
    let fov_deg = field_diameter(&satellite.telescope, &satellite.sensor);
    let fov_rad = fov_deg * PI / 180.0;

    // Use sensor dimensions for image shape
    let image_pixel_width = satellite.sensor.width_px;
    let image_pixel_height = satellite.sensor.height_px;

    // Ensure minimum dimensions for placehold.co
    let image_pixel_width = image_pixel_width.max(1);
    let image_pixel_height = image_pixel_height.max(1);

    // Generate Placehold.co URL for the overlay image
    let display_text = text.unwrap_or("");
    let placehold_url_str = format!(
        "https://placehold.co/{}x{}/{}/{}?text={}",
        image_pixel_width,
        image_pixel_height,
        background_color,
        text_color,
        urlencoding::encode(display_text)
    );
    let placehold_url = Url::parse(&placehold_url_str)?;

    // Calculate WWT 'scale' parameter (width of image in degrees)
    let scale_deg = fov_rad * (180.0 / PI);

    // Calculate center pixel for the image (assuming image is centered)
    let center_x_px = image_pixel_width as f64 / 2.0;
    let center_y_px = image_pixel_height as f64 / 2.0;

    // Construct the WWT ShowImage.aspx URL
    let mut wwt_url = Url::parse("http://www.worldwidetelescope.org/wwtweb/ShowImage.aspx")?;
    wwt_url
        .query_pairs_mut()
        .append_pair("reverseparity", "False")
        .append_pair("scale", &scale_deg.to_string())
        .append_pair("name", object_name)
        .append_pair("imageurl", placehold_url.as_str())
        .append_pair("credits", "Generated via Rust")
        .append_pair("creditsUrl", "https://github.com/meter-sim")
        .append_pair("ra", &coordinates.ra_degrees().to_string())
        .append_pair("dec", &coordinates.dec_degrees().to_string())
        .append_pair("x", &center_x_px.to_string())
        .append_pair("y", &center_y_px.to_string())
        .append_pair("rotation", &rotation_deg.to_string());

    Ok(wwt_url.to_string())
}

#[cfg(test)]
mod wwt_tests {
    use super::*;
    use crate::hardware::{sensor::models::GSENSE6510BSI, telescope::models::DEMO_50CM};

    #[test]
    fn test_ra_hms_to_deg() {
        // Test Andromeda Galaxy coordinates: 0h 42m 44.3s
        let ra_deg = ra_hms_to_deg(0.0, 42.0, 44.3);
        assert!((ra_deg - 10.6845833).abs() < 0.0001);
    }

    #[test]
    fn test_dec_dms_to_deg() {
        // Test positive declination: 41° 16' 9"
        let dec_deg = dec_dms_to_deg(41.0, 16.0, 9.0);
        assert!((dec_deg - 41.2691667).abs() < 0.0001);

        // Test negative declination: -23° 30' 0"
        let dec_deg_neg = dec_dms_to_deg(-23.0, 30.0, 0.0);
        assert!((dec_deg_neg - (-23.5)).abs() < 0.0001);
    }

    #[test]
    fn test_generate_wwt_overlay_url() {
        // Test basic overlay without text
        let coordinates = Equatorial::from_degrees(10.6845833, 41.2691667); // Andromeda
        let satellite = SatelliteConfig::new(
            DEMO_50CM.clone(),
            GSENSE6510BSI.clone(),
            -10.0, // temperature
            550.0, // wavelength
        );

        let url = generate_wwt_overlay_url(
            "Test Overlay",
            &coordinates,
            &satellite,
            30.0, // 30 degree rotation
            None,
            "FFFFFF",
            "00000000",
        )
        .unwrap();

        assert!(url.contains("worldwidetelescope.org"));
        assert!(url.contains("ra=10.6845833"));
        assert!(url.contains("dec=41.2691667"));
        assert!(url.contains("rotation=30"));
        assert!(url.contains("placehold.co"));
    }

    #[test]
    fn test_generate_wwt_overlay_url_with_text() {
        // Test overlay with text
        let coordinates = Equatorial::from_degrees(56.871, 24.105); // Pleiades
        let satellite = SatelliteConfig::new(
            DEMO_50CM.clone(),
            GSENSE6510BSI.clone(),
            -10.0, // temperature
            550.0, // wavelength
        );

        let url = generate_wwt_overlay_url(
            "Pleiades Cluster",
            &coordinates,
            &satellite,
            0.0, // No rotation
            Some("M45"),
            "FFFF00",   // Yellow text
            "00000080", // Semi-transparent black background
        )
        .unwrap();

        // Print URL for debugging
        println!("Generated URL: {}", url);

        // Check that basic WWT parameters are present
        assert!(url.contains("worldwidetelescope.org"));
        assert!(url.contains("placehold.co"));

        // Check colors are in the placehold URL (URL encoded)
        assert!(url.contains("00000080") && url.contains("FFFF00")); // Both colors present

        // Check that the text parameter is present (URL encoded as %3D)
        assert!(url.contains("M45"));
    }
}

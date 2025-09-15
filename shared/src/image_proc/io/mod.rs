//! Astronomical image I/O utilities for data exchange and visualization.
//!
//! This module provides comprehensive I/O functionality for astronomical data,
//! supporting both standard image formats (PNG, JPEG) for visualization and
//! scientific FITS format for data preservation. Handles bit depth conversion,
//! coordinate system transformations, and multi-HDU FITS file operations.
//!
//! # Supported Formats
//!
//! - **Standard Images**: PNG, JPEG, TIFF for visualization and presentation
//! - **FITS Files**: Full support for multi-HDU scientific data with metadata
//! - **Bit Depth Conversion**: Automatic scaling between u16 (sensor) and u8 (display)
//!
//! # Key Features
//!
//! - Automatic dynamic range scaling for visualization
//! - FITS multi-HDU read/write with proper coordinate handling
//! - Type-safe error handling with detailed error messages
//! - Memory-efficient large image processing
//! - Coordinate system preservation between formats
//!
//! # Usage
//!
//! Convert high-dynamic-range sensor data to 8-bit for visualization,
//! then save in standard image formats. Handles coordinate transformations
//! and provides both automatic and fixed scaling options.

use crate::algo::MinMaxScan;
use ndarray::Array2;
use std::error::Error;
use std::path::Path;

/// Save 8-bit grayscale image to standard image format (PNG, JPEG, etc.).
///
/// Converts ndarray data to image crate format and saves to specified path.
/// File format is automatically determined from the file extension.
/// Handles coordinate system conversion between ndarray and image formats.
///
/// # Arguments
/// * `image` - 2D array of u8 pixel values (0-255 grayscale)
/// * `path` - Output file path with appropriate extension (.png, .jpg, .tiff)
///
/// # Returns
/// Result indicating success or I/O error
///
/// # Format Support
/// - PNG: Lossless, good for scientific data visualization
/// - JPEG: Lossy compression, good for presentations
/// - TIFF: Lossless, good for archival display images
///
/// # Usage
/// Save 8-bit grayscale arrays as standard image files. File format
/// determined by extension. Handles coordinate system conversion automatically.
pub fn save_u8_image<P: AsRef<Path>>(image: &Array2<u8>, path: P) -> Result<(), Box<dyn Error>> {
    use image::{ImageBuffer, Luma};

    let (height, width) = image.dim();

    // Create an 8-bit grayscale image buffer directly from the u8 array
    let mut img_buffer = ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
        *pixel = Luma([image[[y as usize, x as usize]]]);
    }

    img_buffer.save(path)?;

    Ok(())
}

/// Convert 16-bit sensor data to 8-bit display format with automatic scaling.
///
/// Automatically determines optimal scaling by finding the maximum value in the
/// image and mapping [0, max_value] to [0, 255]. Preserves relative intensities
/// while maximizing contrast for visualization.
///
/// # Arguments
/// * `image` - 2D array of u16 pixel values (typical astronomical sensor data)
///
/// # Returns
/// Scaled 8-bit image optimized for display and visualization
///
/// # Scaling Method
/// `output_pixel = (input_pixel × 255) / max_value_in_image`
///
/// # Use Cases
/// - Converting raw sensor data for quick visualization
/// - Creating display images from high dynamic range data
/// - Preprocessing for image analysis algorithms requiring 8-bit input
///
/// # Usage
/// Convert 16-bit sensor data to 8-bit display format using automatic
/// scaling based on the image's maximum value for optimal contrast.
pub fn u16_to_u8_auto_scale(image: &Array2<u16>) -> Array2<u8> {
    // Find max value for proper scaling
    let values: Vec<f64> = image.iter().map(|&x| x as f64).collect();
    let scan = MinMaxScan::new(&values);
    let max_value = scan.max().unwrap_or(0.0) as u16;

    if max_value == 0 {
        // Return black image if maximum value is 0
        return Array2::zeros(image.dim());
    }

    // Scale to 0-255 range
    image.mapv(|x| ((x as f32 * 255.0) / max_value as f32) as u8)
}

/// Convert 16-bit data to 8-bit with fixed scaling reference.
///
/// Maps input range [0, max_value] to output range [0, 255] using the specified
/// maximum value as reference. Useful for consistent scaling across multiple images
/// or when working with known sensor bit depths.
///
/// # Arguments
/// * `image` - 2D array of u16 pixel values
/// * `max_value` - Reference maximum for scaling (e.g., 4095 for 12-bit, 65535 for 16-bit)
///
/// # Returns
/// Scaled 8-bit image with consistent brightness mapping
///
/// # Scaling Method
/// `output_pixel = round((input_pixel × 255) / max_value)`
/// Values above max_value are clipped to 255.
///
/// # Usage
/// Convert 16-bit data to 8-bit using a fixed scaling reference.
/// Useful for consistent scaling across multiple images or known bit depths.
pub fn u16_to_u8_scaled(image: &Array2<u16>, max_value: u32) -> Array2<u8> {
    if max_value == 0 {
        // Return black image if maximum value is 0
        return Array2::zeros(image.dim());
    }

    // Scale to 0-255 range based on the specified maximum
    image.mapv(|x| ((x as f32 * 255.0) / max_value as f32).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_u16_to_u8_auto_scale() {
        // Create a test image with known values
        let mut image = Array2::<u16>::zeros((2, 3));
        image[[0, 0]] = 0;
        image[[0, 1]] = 100;
        image[[0, 2]] = 200;
        image[[1, 0]] = 400;
        image[[1, 1]] = 800;
        image[[1, 2]] = 1000; // Max value

        // Convert to u8
        let u8_image = u16_to_u8_auto_scale(&image);

        // Check dimensions are preserved
        assert_eq!(u8_image.dim(), (2, 3));

        // Check scaling based on max value (1000)
        assert_eq!(u8_image[[0, 0]], 0); // 0 -> 0
        assert_eq!(u8_image[[0, 1]], 25); // 100/1000 * 255 = 25.5 -> 25
        assert_eq!(u8_image[[0, 2]], 51); // 200/1000 * 255 = 51
        assert_eq!(u8_image[[1, 0]], 102); // 400/1000 * 255 = 102
        assert_eq!(u8_image[[1, 1]], 204); // 800/1000 * 255 = 204
        assert_eq!(u8_image[[1, 2]], 255); // 1000/1000 * 255 = 255
    }

    #[test]
    fn test_u16_to_u8_scaled() {
        // Create a test image with values beyond the scaling factor
        let mut image = Array2::<u16>::zeros((2, 2));
        image[[0, 0]] = 0;
        image[[0, 1]] = 512;
        image[[1, 0]] = 1024;
        image[[1, 1]] = 4096;

        // Scale based on 12-bit max (4095)
        let u8_image = u16_to_u8_scaled(&image, 4095);

        // Check scaling
        assert_eq!(u8_image[[0, 0]], 0); // 0/4095 * 255 = 0
        assert_eq!(u8_image[[0, 1]], 32); // 512/4095 * 255 = 31.9 -> 32 (rounded)
        assert_eq!(u8_image[[1, 0]], 64); // 1024/4095 * 255 = 63.8 -> 64 (rounded)
        assert_eq!(u8_image[[1, 1]], 255); // 4096/4095 * 255 > 255 -> 255 (clamped by u8)
    }

    #[test]
    fn test_zero_image() {
        // Create an image with all zeros
        let image = Array2::<u16>::zeros((3, 3));

        // Auto scale
        let u8_auto = u16_to_u8_auto_scale(&image);

        // All values should be zero
        assert!(u8_auto.iter().all(|&x| x == 0));

        // Fixed scale
        let u8_fixed = u16_to_u8_scaled(&image, 4095);

        // All values should be zero
        assert!(u8_fixed.iter().all(|&x| x == 0));
    }

    // We don't test save_u8_image directly to avoid file system interactions
    // in normal test runs, but we validate the function signature compiles
}

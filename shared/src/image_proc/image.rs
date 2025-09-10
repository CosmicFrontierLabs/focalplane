//! Image format conversion utilities for astronomical data processing.
//!
//! This module provides conversion functions between ndarray Array2 structures
//! (commonly used in scientific computing) and image crate types (used for
//! visualization and file I/O). Essential for exporting processed astronomical
//! data to standard image formats.
//!
//! # Coordinate System Conversions
//!
//! - **ndarray**: Uses matrix indexing [row, col] = [y, x] with (height, width) dimensions
//! - **image crate**: Uses graphics indexing (x, y) with (width, height) dimensions
//!
//! # Common Use Cases
//!
//! - Export processed star field images for visualization
//! - Convert detection maps to standard image formats
//! - Save analysis results for presentations and reports
//! - Interface with image processing libraries
//!

use image::{GrayImage, Luma};
use ndarray::Array2;

/// Convert ndarray `Array2<u8>` to image crate GrayImage for visualization and I/O.
///
/// Performs efficient conversion between scientific computing array format and
/// standard image format while correctly handling coordinate system differences.
/// The conversion preserves all pixel data and spatial relationships.
///
/// # Coordinate Mapping
/// - Array index [row, col] → Image pixel (col, row)
/// - Array dimensions (height, width) → Image dimensions (width, height)
/// - This ensures proper image orientation is maintained
///
/// # Arguments
/// * `arr` - Reference to 2D grayscale array with u8 pixel values (0-255)
///
/// # Returns
/// GrayImage with identical pixel data suitable for saving or display
///
/// # Performance
/// - Time: O(width × height) - single pass through all pixels
/// - Space: O(width × height) - creates new image buffer
/// - Memory layout optimized for row-major traversal
///
pub fn array2_to_gray_image(arr: &Array2<u8>) -> GrayImage {
    // Get array dimensions (height, width)
    let (height, width) = arr.dim();

    // Create GrayImage with swapped dimensions (width, height)
    let mut img = GrayImage::new(width as u32, height as u32);

    // Copy data with coordinate system conversion
    for y in 0..height {
        for x in 0..width {
            // Map array[y, x] to image pixel (x, y)
            img.put_pixel(x as u32, y as u32, Luma([arr[[y, x]]]));
        }
    }

    img
}

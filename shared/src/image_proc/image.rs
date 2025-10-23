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

use image::{GrayImage, ImageBuffer, Luma};
use ndarray::Array2;

/// Invert a monochrome image given its bit depth.
///
/// Performs pixel value inversion by subtracting each pixel from the maximum
/// value for the given bit depth. Commonly used for converting negative images
/// to positive or vice versa (e.g., inverting AprilTag patterns for detection).
///
/// # Arguments
/// * `frame` - Reference to 2D grayscale array with u16 pixel values
/// * `bit_depth` - ADC bit depth (typically 8, 12, 14, or 16 bits)
///
/// # Returns
/// New Array2<u16> with inverted pixel values
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// use shared::image_proc::image::invert_monochrome;
///
/// let frame = Array2::from_elem((2, 2), 100u16);
/// let inverted = invert_monochrome(&frame, 8);
/// assert_eq!(inverted[[0, 0]], 155); // 255 - 100
/// ```
///
/// # Performance
/// - Time: O(width × height) - single pass through all pixels
/// - Space: O(width × height) - creates new array
///
pub fn invert_monochrome(frame: &Array2<u16>, bit_depth: u8) -> Array2<u16> {
    let max_value = (1u32 << bit_depth) - 1;
    frame.mapv(|pixel| {
        let pixel_u32 = pixel as u32;
        if pixel_u32 > max_value {
            0
        } else {
            (max_value - pixel_u32) as u16
        }
    })
}

/// Convert u16 Array2 to u8 GrayImage with auto-scaling based on max value.
///
/// Scales pixel values from the u16 range to u8 (0-255) by finding the maximum
/// value in the frame and scaling all pixels proportionally. Useful for converting
/// high bit-depth sensor data to 8-bit images for visualization or processing.
///
/// # Arguments
/// * `frame` - Reference to 2D grayscale array with u16 pixel values
///
/// # Returns
/// GrayImage with pixel values scaled to 0-255 range
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// use shared::image_proc::image::u16_to_gray_image;
///
/// let frame = Array2::from_elem((2, 2), 1000u16);
/// let img = u16_to_gray_image(&frame);
/// assert_eq!(img.width(), 2);
/// assert_eq!(img.height(), 2);
/// ```
///
pub fn u16_to_gray_image(frame: &Array2<u16>) -> GrayImage {
    let (height, width) = frame.dim();
    let max_val = *frame.iter().max().unwrap_or(&1) as f32;
    let scale = if max_val > 0.0 { 255.0 / max_val } else { 1.0 };

    let mut pixels = Vec::with_capacity(height * width);

    for y in 0..height {
        for x in 0..width {
            let val = frame[[y, x]];
            pixels.push(((val as f32) * scale) as u8);
        }
    }

    ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width as u32, height as u32, pixels)
        .expect("Failed to create image from raw data")
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert_monochrome_8bit() {
        let frame = Array2::from_elem((2, 2), 100u16);
        let inverted = invert_monochrome(&frame, 8);
        assert_eq!(inverted[[0, 0]], 155);
        assert_eq!(inverted[[1, 1]], 155);
    }

    #[test]
    fn test_invert_monochrome_12bit() {
        let frame = Array2::from_elem((3, 3), 1000u16);
        let inverted = invert_monochrome(&frame, 12);
        assert_eq!(inverted[[0, 0]], 3095);
        assert_eq!(inverted[[2, 2]], 3095);
    }

    #[test]
    fn test_invert_monochrome_14bit() {
        let frame = Array2::from_elem((2, 3), 8000u16);
        let inverted = invert_monochrome(&frame, 14);
        assert_eq!(inverted[[0, 0]], 8383);
        assert_eq!(inverted[[1, 2]], 8383);
    }

    #[test]
    fn test_invert_monochrome_16bit() {
        let frame = Array2::from_elem((4, 4), 30000u16);
        let inverted = invert_monochrome(&frame, 16);
        assert_eq!(inverted[[0, 0]], 35535);
        assert_eq!(inverted[[3, 3]], 35535);
    }

    #[test]
    fn test_invert_monochrome_zeros() {
        let frame = Array2::zeros((2, 2));
        let inverted = invert_monochrome(&frame, 8);
        assert_eq!(inverted[[0, 0]], 255);
        assert_eq!(inverted[[1, 1]], 255);
    }

    #[test]
    fn test_invert_monochrome_max_value() {
        let frame = Array2::from_elem((2, 2), 255u16);
        let inverted = invert_monochrome(&frame, 8);
        assert_eq!(inverted[[0, 0]], 0);
        assert_eq!(inverted[[1, 1]], 0);
    }

    #[test]
    fn test_invert_monochrome_roundtrip() {
        let original = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as u16 * 10);
        let inverted = invert_monochrome(&original, 8);
        let double_inverted = invert_monochrome(&inverted, 8);

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(original[[i, j]], double_inverted[[i, j]]);
            }
        }
    }

    #[test]
    fn test_invert_monochrome_clamps_overflow() {
        let mut frame = Array2::from_elem((2, 2), 65535u16);
        frame[[0, 0]] = 300;
        let inverted = invert_monochrome(&frame, 8);
        assert_eq!(inverted[[0, 0]], 0);
        assert_eq!(inverted[[1, 1]], 0);
    }
}

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

/// 16-bit grayscale image type alias for convenience.
pub type Gray16Image = ImageBuffer<Luma<u16>, Vec<u16>>;

/// Invert a monochrome image given its bit depth.
///
/// Performs pixel value inversion by subtracting each pixel from the maximum
/// value for the given bit depth. Commonly used for converting negative images
/// to positive or vice versa.
///
/// # Arguments
/// * `frame` - Reference to 2D grayscale array with u16 pixel values
/// * `bit_depth` - ADC bit depth (typically 8, 12, 14, or 16 bits)
///
/// # Returns
/// New Array2<u16> with inverted pixel values
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

/// Convert ndarray `Array2<u16>` to 16-bit grayscale image without scaling.
///
/// Performs direct conversion preserving raw sensor values. No scaling or
/// normalization is applied - pixel values are transferred exactly.
///
/// # Coordinate Mapping
/// - Array index [row, col] → Image pixel (col, row)
/// - Array dimensions (height, width) → Image dimensions (width, height)
///
/// # Arguments
/// * `arr` - Reference to 2D array with u16 pixel values (raw sensor data)
///
/// # Returns
/// Gray16Image with identical pixel data suitable for saving as 16-bit PNG
///
pub fn array2_to_gray16_image(arr: &Array2<u16>) -> Gray16Image {
    let (height, width) = arr.dim();
    ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        Luma([arr[[y as usize, x as usize]]])
    })
}

/// Convert GrayImage back to ndarray `Array2<u8>`.
///
/// Reverses the conversion from `array2_to_gray_image`, extracting pixel data
/// from an image buffer back into a 2D array format for scientific processing.
///
/// # Coordinate Mapping
/// - Image pixel (x, y) → Array index [y, x]
/// - Image dimensions (width, height) → Array dimensions (height, width)
///
/// # Arguments
/// * `img` - Reference to 8-bit grayscale image
///
/// # Returns
/// Array2<u8> with identical pixel data
///
pub fn gray_image_to_array2(img: &GrayImage) -> Array2<u8> {
    let (width, height) = img.dimensions();
    Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
        img.get_pixel(x as u32, y as u32).0[0]
    })
}

/// Convert 16-bit grayscale image back to ndarray `Array2<u16>`.
///
/// Reverses the conversion from `array2_to_gray16_image`, extracting raw
/// pixel data back into array format for scientific analysis.
///
/// # Coordinate Mapping
/// - Image pixel (x, y) → Array index [y, x]
/// - Image dimensions (width, height) → Array dimensions (height, width)
///
/// # Arguments
/// * `img` - Reference to 16-bit grayscale image
///
/// # Returns
/// Array2<u16> with identical pixel data
///
pub fn gray16_image_to_array2(img: &Gray16Image) -> Array2<u16> {
    let (width, height) = img.dimensions();
    Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
        img.get_pixel(x as u32, y as u32).0[0]
    })
}

/// Downsample an image by sampling every Nth pixel.
///
/// Creates a smaller image by taking every `factor`th pixel in both dimensions.
/// This is a fast decimation operation (no filtering/interpolation) useful for
/// speeding up algorithms where full resolution is unnecessary.
///
/// # Arguments
/// * `image` - 2D array of pixel values
/// * `factor` - Downsampling factor (1 = no change, 2 = half size in each dimension, etc.)
///
/// # Returns
/// Downsampled image array. If factor is 1, returns a copy of the input.
///
/// # Performance
/// With factor=N, the output has ~1/N² pixels of the input.
/// For a 9576x6388 image (61MP) with factor=16, returns a 599x400 image (~240k pixels).
///
/// # Use Cases
/// - Fast noise estimation on large images (noise is a global sensor property)
/// - Quick preview generation
/// - Reducing computation for algorithms that don't need full resolution
///
pub fn downsample_f64(image: &ndarray::ArrayView2<f64>, factor: usize) -> Array2<f64> {
    let factor = factor.max(1);
    if factor == 1 {
        return image.to_owned();
    }

    let (height, width) = image.dim();
    let new_height = height.div_ceil(factor);
    let new_width = width.div_ceil(factor);

    Array2::from_shape_fn((new_height, new_width), |(i, j)| {
        image[[i * factor, j * factor]]
    })
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

    #[test]
    fn test_array2_to_gray16_image_roundtrip() {
        let arr = Array2::from_shape_fn((3, 4), |(y, x)| (y * 4 + x) as u16 * 100);
        let img = array2_to_gray16_image(&arr);
        let back = gray16_image_to_array2(&img);

        assert_eq!(arr.dim(), back.dim());
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(arr[[y, x]], back[[y, x]]);
            }
        }
    }

    #[test]
    fn test_array2_to_gray_image_roundtrip() {
        let arr = Array2::from_shape_fn((3, 4), |(y, x)| ((y * 4 + x) * 20) as u8);
        let img = array2_to_gray_image(&arr);
        let back = gray_image_to_array2(&img);

        assert_eq!(arr.dim(), back.dim());
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(arr[[y, x]], back[[y, x]]);
            }
        }
    }

    #[test]
    fn test_gray16_image_dimensions() {
        let arr = Array2::from_elem((100, 200), 1000u16);
        let img = array2_to_gray16_image(&arr);

        assert_eq!(img.width(), 200);
        assert_eq!(img.height(), 100);
    }

    #[test]
    fn test_gray_image_dimensions() {
        let arr = Array2::from_elem((50, 75), 128u8);
        let img = array2_to_gray_image(&arr);

        assert_eq!(img.width(), 75);
        assert_eq!(img.height(), 50);
    }

    #[test]
    fn test_downsample_f64_factor_1() {
        // Factor 1 should return a copy of the input
        let arr = Array2::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);
        let result = downsample_f64(&arr.view(), 1);

        assert_eq!(result.dim(), arr.dim());
        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(result[[i, j]], arr[[i, j]]);
            }
        }
    }

    #[test]
    fn test_downsample_f64_factor_0_clamped() {
        // Factor 0 should be clamped to 1
        let arr = Array2::from_elem((10, 10), 42.0);
        let result = downsample_f64(&arr.view(), 0);

        assert_eq!(result.dim(), (10, 10));
    }

    #[test]
    fn test_downsample_f64_factor_2() {
        // 10x10 with factor 2 -> 5x5
        let arr = Array2::from_shape_fn((10, 10), |(i, j)| (i * 10 + j) as f64);
        let result = downsample_f64(&arr.view(), 2);

        assert_eq!(result.dim(), (5, 5));

        // Check sampled values
        assert_eq!(result[[0, 0]], arr[[0, 0]]); // 0
        assert_eq!(result[[0, 1]], arr[[0, 2]]); // 2
        assert_eq!(result[[1, 0]], arr[[2, 0]]); // 20
        assert_eq!(result[[2, 2]], arr[[4, 4]]); // 44
    }

    #[test]
    fn test_downsample_f64_non_divisible() {
        // 7x9 with factor 3 -> ceil(7/3) x ceil(9/3) = 3x3
        let arr = Array2::from_shape_fn((7, 9), |(i, j)| (i * 9 + j) as f64);
        let result = downsample_f64(&arr.view(), 3);

        assert_eq!(result.dim(), (3, 3));

        // Check sampled values at (0,0), (0,3), (0,6), (3,0), etc.
        assert_eq!(result[[0, 0]], arr[[0, 0]]); // 0
        assert_eq!(result[[0, 1]], arr[[0, 3]]); // 3
        assert_eq!(result[[0, 2]], arr[[0, 6]]); // 6
        assert_eq!(result[[1, 0]], arr[[3, 0]]); // 27
        assert_eq!(result[[2, 0]], arr[[6, 0]]); // 54
    }

    #[test]
    fn test_downsample_f64_large_factor() {
        // 100x100 with factor 10 -> 10x10
        let arr = Array2::from_elem((100, 100), 3.14159);
        let result = downsample_f64(&arr.view(), 10);

        assert_eq!(result.dim(), (10, 10));

        // All values should still be pi
        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(result[[i, j]], 3.14159);
            }
        }
    }

    #[test]
    fn test_downsample_f64_preserves_pattern() {
        // Create a checkerboard pattern and verify downsampling picks correct pixels
        let arr = Array2::from_shape_fn((8, 8), |(i, j)| if (i + j) % 2 == 0 { 1.0 } else { 0.0 });
        let result = downsample_f64(&arr.view(), 2);

        assert_eq!(result.dim(), (4, 4));

        // With factor 2, we sample (0,0), (0,2), (0,4), etc.
        // All of these have (i+j) even, so all should be 1.0
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(result[[i, j]], 1.0);
            }
        }
    }

    #[test]
    fn test_downsample_f64_small_image() {
        // 3x3 with factor 2 -> 2x2 (ceil(3/2) = 2)
        let arr = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as f64);
        let result = downsample_f64(&arr.view(), 2);

        assert_eq!(result.dim(), (2, 2));
        assert_eq!(result[[0, 0]], 0.0); // arr[0,0]
        assert_eq!(result[[0, 1]], 2.0); // arr[0,2]
        assert_eq!(result[[1, 0]], 6.0); // arr[2,0]
        assert_eq!(result[[1, 1]], 8.0); // arr[2,2]
    }
}

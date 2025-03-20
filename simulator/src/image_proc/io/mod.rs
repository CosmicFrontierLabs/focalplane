//! Image I/O utilities for saving and loading image data
//!
//! This module provides functions for converting ndarray data to image formats
//! and saving/loading them to the filesystem.

use ndarray::Array2;
use std::error::Error;
use std::path::Path;

/// Save an 8-bit image to a file
///
/// # Arguments
/// * `image` - 2D array of u8 pixel values
/// * `path` - Output path for the image file
///
/// # Returns
/// * `Result<(), Box<dyn Error>>` - Success or error
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// use simulator::image_proc::io::save_u8_image;
/// use std::path::Path;
///
/// // Create a simple gradient image
/// let mut image = Array2::zeros((100, 100));
/// for (i, j) in (0..100).flat_map(|i| (0..100).map(move |j| (i, j))) {
///     image[[i, j]] = ((i + j) / 2) as u8;
/// }
///
/// // Save the image (note: this will create a file on disk)
/// # #[cfg(feature = "io_test")]
/// save_u8_image(&image, Path::new("gradient.png")).unwrap();
/// ```
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

/// Convert a u16 image to u8 by scaling based on the maximum value
///
/// # Arguments
/// * `image` - 2D array of u16 pixel values
///
/// # Returns
/// * `Array2<u8>` - Scaled 8-bit image
///
/// This function automatically scales the image based on the maximum value
/// present in the image to utilize the full 8-bit range (0-255).
pub fn u16_to_u8_auto_scale(image: &Array2<u16>) -> Array2<u8> {
    // Find max value for proper scaling
    let max_value = image.iter().fold(0, |max_val, &x| max_val.max(x));

    if max_value == 0 {
        // Return black image if maximum value is 0
        return Array2::zeros(image.dim());
    }

    // Scale to 0-255 range
    image.mapv(|x| ((x as f32 * 255.0) / max_value as f32) as u8)
}

/// Convert a u16 image to u8 by scaling based on a specific maximum value
///
/// # Arguments
/// * `image` - 2D array of u16 pixel values
/// * `max_value` - The reference maximum value for scaling
///
/// # Returns
/// * `Array2<u8>` - Scaled 8-bit image
///
/// This function scales the image based on the specified maximum value,
/// useful when converting data with a known bit depth or range.
pub fn u16_to_u8_scaled(image: &Array2<u16>, max_value: u16) -> Array2<u8> {
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

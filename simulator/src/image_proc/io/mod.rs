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
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use thiserror::Error;

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

/// Errors that can occur during FITS file operations
#[derive(Error, Debug)]
pub enum FitsError {
    #[error("FITS I/O error: {0}")]
    FitsIo(#[from] fitsio::errors::Error),
    #[error("HDU not found: {0}")]
    HduNotFound(String),
    #[error("Invalid data type in HDU: {0}")]
    InvalidDataType(String),
}

/// Read FITS file and return HashMap of HDU names to `Array2<f64>` data
///
/// # Arguments
/// * `path` - Path to the FITS file
///
/// # Returns
/// * `Result<HashMap<String, Array2<f64>>, FitsError>` - Map of HDU names to 2D arrays
///
/// # Usage
/// Read FITS files and return all image HDUs as a HashMap mapping
/// HDU names to 2D arrays. Handles coordinate system conversion automatically.
pub fn read_fits_to_hashmap<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, Array2<f64>>, FitsError> {
    use fitsio::FitsFile;

    let mut fptr = FitsFile::open(&path)?;
    let mut data_map = HashMap::new();

    let mut hdu_idx = 0;
    while let Ok(hdu) = fptr.hdu(hdu_idx) {
        // Get HDU name, fallback to index-based name if no EXTNAME
        let hdu_name = match hdu.read_key::<String>(&mut fptr, "EXTNAME") {
            Ok(name) => name,
            Err(_) => format!("HDU_{}", hdu_idx),
        };

        // Try to read as image data
        if let Ok(image_data) = hdu.read_image::<Vec<f64>>(&mut fptr) {
            // Get image dimensions
            let naxis = hdu.read_key::<i64>(&mut fptr, "NAXIS").unwrap_or(0);

            if naxis == 2 {
                let naxis1 = hdu.read_key::<i64>(&mut fptr, "NAXIS1").unwrap_or(0) as usize;
                let naxis2 = hdu.read_key::<i64>(&mut fptr, "NAXIS2").unwrap_or(0) as usize;

                // Reshape 1D vector to 2D array (FITS format)
                let fits_array =
                    Array2::from_shape_vec((naxis2, naxis1), image_data).map_err(|_| {
                        FitsError::InvalidDataType(format!(
                            "Cannot reshape image data for HDU '{}'",
                            hdu_name
                        ))
                    })?;

                // Transpose to match ndarray convention
                let array = fits_array.t().to_owned();
                data_map.insert(hdu_name, array);
            }
        }

        hdu_idx += 1;
    }

    Ok(data_map)
}

/// Write HashMap of `Array2<f64>` data to FITS file
///
/// # Arguments
/// * `data` - HashMap mapping HDU names to 2D arrays
/// * `path` - Output path for the FITS file
///
/// # Returns
/// * `Result<(), FitsError>` - Success or error
///
/// # Usage
/// Write HashMap of 2D arrays to FITS file with proper HDU organization
/// and coordinate system handling. Sets EXTNAME headers automatically.
pub fn write_hashmap_to_fits<P: AsRef<Path>>(
    data: &HashMap<String, Array2<f64>>,
    path: P,
) -> Result<(), FitsError> {
    use fitsio::{
        images::{ImageDescription, ImageType},
        FitsFile,
    };

    let mut fptr = FitsFile::create(&path).overwrite().open()?;

    for (name, array) in data {
        let (height, width) = array.dim();

        // Create image description (FITS uses different axis ordering)
        let image_description = ImageDescription {
            data_type: ImageType::Double,
            dimensions: &[width, height],
        };

        // Create new HDU
        let hdu = fptr.create_image(name.clone(), &image_description)?;

        // Write the array data (transpose for FITS format)
        let transposed = array.t().to_owned();
        let flat_data: Vec<f64> = transposed.into_raw_vec();
        hdu.write_image(&mut fptr, &flat_data)?;

        // Set EXTNAME header
        hdu.write_key(&mut fptr, "EXTNAME", name.clone())?;
    }

    Ok(())
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

    #[test]
    fn test_fits_error_display() {
        let error = FitsError::HduNotFound("TEST".to_string());
        assert!(error.to_string().contains("HDU not found: TEST"));

        let error = FitsError::InvalidDataType("bad data".to_string());
        assert!(error
            .to_string()
            .contains("Invalid data type in HDU: bad data"));
    }

    #[test]
    fn test_fits_roundtrip() {
        use tempfile::NamedTempFile;

        // Create test data
        let mut data = HashMap::new();
        let array1 = Array2::from_elem((3, 4), 1.5);
        let mut array2 = Array2::zeros((2, 2));
        array2[[0, 0]] = 2.5;
        array2[[1, 1]] = 3.7;

        data.insert("IMAGE1".to_string(), array1.clone());
        data.insert("IMAGE2".to_string(), array2.clone());

        // Write and read back
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        write_hashmap_to_fits(&data, path).unwrap();
        let read_data = read_fits_to_hashmap(path).unwrap();

        // Verify we got the same data back
        assert_eq!(read_data.len(), 2);
        assert!(read_data.contains_key("IMAGE1"));
        assert!(read_data.contains_key("IMAGE2"));

        let read_array1 = &read_data["IMAGE1"];
        let read_array2 = &read_data["IMAGE2"];

        assert_eq!(read_array1.dim(), (3, 4));
        assert_eq!(read_array2.dim(), (2, 2));

        // Check values (allowing for floating point precision)
        assert!((read_array1[[0, 0]] - 1.5).abs() < 1e-10);
        assert!((read_array2[[0, 0]] - 2.5).abs() < 1e-10);
        assert!((read_array2[[1, 1]] - 3.7).abs() < 1e-10);
    }
}

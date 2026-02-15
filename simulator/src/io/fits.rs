//! FITS file I/O utilities for astronomical data
//!
//! Provides reading and writing of FITS (Flexible Image Transport System) files
//! with support for multiple data types and HDU (Header Data Unit) organization.

use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

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

/// Supported FITS data types with their array data
#[derive(Debug)]
pub enum FitsDataType {
    /// 8-bit unsigned integer data
    UInt8(Array2<u8>),
    /// 16-bit unsigned integer data
    UInt16(Array2<u16>),
    /// 32-bit unsigned integer data
    UInt32(Array2<u32>),
    /// 32-bit signed integer data
    Int32(Array2<i32>),
    /// 64-bit signed integer data
    Int64(Array2<i64>),
    /// 32-bit floating point data
    Float32(Array2<f32>),
    /// 64-bit floating point data (double precision)
    Float64(Array2<f64>),
}

impl FitsDataType {
    /// Get dimensions of the array
    fn dimensions(&self) -> (usize, usize) {
        match self {
            FitsDataType::UInt8(arr) => arr.dim(),
            FitsDataType::UInt16(arr) => arr.dim(),
            FitsDataType::UInt32(arr) => arr.dim(),
            FitsDataType::Int32(arr) => arr.dim(),
            FitsDataType::Int64(arr) => arr.dim(),
            FitsDataType::Float32(arr) => arr.dim(),
            FitsDataType::Float64(arr) => arr.dim(),
        }
    }

    /// Get FITS image type
    fn image_type(&self) -> fitsio::images::ImageType {
        use fitsio::images::ImageType;
        match self {
            FitsDataType::UInt8(_) => ImageType::UnsignedByte,
            FitsDataType::UInt16(_) => ImageType::UnsignedShort,
            FitsDataType::UInt32(_) => ImageType::UnsignedLong,
            FitsDataType::Int32(_) => ImageType::Long,
            FitsDataType::Int64(_) => ImageType::LongLong,
            FitsDataType::Float32(_) => ImageType::Float,
            FitsDataType::Float64(_) => ImageType::Double,
        }
    }

    /// Write array data to FITS file
    fn write_data(
        &self,
        fptr: &mut fitsio::FitsFile,
        hdu: &fitsio::hdu::FitsHdu,
    ) -> Result<(), FitsError> {
        match self {
            FitsDataType::UInt8(arr) => arr.write_to_fits(fptr, hdu),
            FitsDataType::UInt16(arr) => arr.write_to_fits(fptr, hdu),
            FitsDataType::UInt32(arr) => arr.write_to_fits(fptr, hdu),
            FitsDataType::Int32(arr) => arr.write_to_fits(fptr, hdu),
            FitsDataType::Int64(arr) => arr.write_to_fits(fptr, hdu),
            FitsDataType::Float32(arr) => arr.write_to_fits(fptr, hdu),
            FitsDataType::Float64(arr) => arr.write_to_fits(fptr, hdu),
        }
    }
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
            Err(_) => format!("HDU_{hdu_idx}"),
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
                            "Cannot reshape image data for HDU '{hdu_name}'"
                        ))
                    })?;

                // Flip vertically to match ndarray convention (FITS origin is bottom-left)
                // Manually collect to ensure proper standard layout
                let flipped_view = fits_array.slice(ndarray::s![..;-1, ..]);
                let flipped = Array2::from_shape_vec(
                    (naxis2, naxis1),
                    flipped_view.iter().copied().collect(),
                )
                .map_err(|_| {
                    FitsError::InvalidDataType(format!(
                        "Cannot create flipped array for HDU '{hdu_name}'"
                    ))
                })?;
                data_map.insert(hdu_name, flipped);
            }
        }

        hdu_idx += 1;
    }

    Ok(data_map)
}

/// Helper trait to write array data to FITS
trait WriteFitsData {
    fn write_to_fits(
        &self,
        fptr: &mut fitsio::FitsFile,
        hdu: &fitsio::hdu::FitsHdu,
    ) -> Result<(), FitsError>;
}

/// Macro to implement WriteFitsData for different types
macro_rules! impl_write_fits_data {
    ($($t:ty),*) => {
        $(
            impl WriteFitsData for Array2<$t> {
                fn write_to_fits(
                    &self,
                    fptr: &mut fitsio::FitsFile,
                    hdu: &fitsio::hdu::FitsHdu,
                ) -> Result<(), FitsError> {
                    // Flip vertically to match FITS convention (origin at bottom-left)
                    // Collect into Vec manually to ensure proper row-major order
                    let flipped = self.slice(ndarray::s![..;-1, ..]);
                    let flat_data: Vec<$t> = flipped.iter().copied().collect();
                    hdu.write_image(fptr, &flat_data)?;
                    Ok(())
                }
            }
        )*
    };
}

// Implement for all supported types
impl_write_fits_data!(u8, u16, u32, i32, i64, f32, f64);

/// Write HashMap of FitsDataType enums to FITS file
///
/// # Arguments
/// * `data` - HashMap mapping HDU names to FitsDataType enum variants
/// * `path` - Output path for the FITS file
///
/// # Returns
/// * `Result<(), FitsError>` - Success or error
///
/// # Usage
/// Write HashMap of typed arrays to FITS file with proper HDU organization
/// and coordinate system handling. Sets EXTNAME headers automatically.
/// Supports multiple data types through the FitsDataType enum.
pub fn write_typed_fits<P: AsRef<Path>>(
    data: &HashMap<String, FitsDataType>,
    path: P,
) -> Result<(), FitsError> {
    use fitsio::{images::ImageDescription, FitsFile};

    if data.len() == 1 {
        // Single image: write directly to primary HDU so simple readers can find it
        let (name, fits_data) = data.iter().next().unwrap();
        let (height, width) = fits_data.dimensions();
        let primary_desc = ImageDescription {
            data_type: fits_data.image_type(),
            dimensions: &[height, width],
        };

        let mut fptr = FitsFile::create(&path)
            .overwrite()
            .with_custom_primary(&primary_desc)
            .open()?;

        let hdu = fptr.hdu(0)?;
        fits_data.write_data(&mut fptr, &hdu)?;
        hdu.write_key(&mut fptr, "EXTNAME", name.clone())?;
    } else {
        // Multiple images: use extension HDUs with EXTNAME for lookup
        let mut fptr = FitsFile::create(&path).overwrite().open()?;

        for (name, fits_data) in data {
            let (height, width) = fits_data.dimensions();
            let image_description = ImageDescription {
                data_type: fits_data.image_type(),
                dimensions: &[height, width],
            };

            let hdu = fptr.create_image(name.clone(), &image_description)?;
            fits_data.write_data(&mut fptr, &hdu)?;
            hdu.write_key(&mut fptr, "EXTNAME", name.clone())?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;
    use shared::image_proc::test_patterns::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    // Set to true to write test FITS files locally for inspection
    const WRITE_LOCAL: bool = true;

    /// Generate test output path based on whether we're writing locally
    fn get_test_path(test_name: &str) -> (PathBuf, Option<NamedTempFile>) {
        if WRITE_LOCAL {
            // Create local test output directory
            let dir = PathBuf::from("test_output/fits");
            fs::create_dir_all(&dir).expect("Failed to create test output directory");

            // Generate filename with .fits extension
            let filename = format!("{test_name}.fits");
            let path = dir.join(filename);

            println!("Writing test FITS to: {}", path.display());
            (path, None)
        } else {
            // Use temp file
            let temp_file = NamedTempFile::new().unwrap();
            let path = temp_file.path().to_path_buf();
            (path, Some(temp_file))
        }
    }

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
    fn test_fits_roundtrip_f64() {
        // Create test data
        let array1 = Array2::from_elem((3, 4), 1.5);
        let mut array2 = Array2::zeros((2, 2));
        array2[[0, 0]] = 2.5;
        array2[[1, 1]] = 3.7;

        let mut data = HashMap::new();
        data.insert("IMAGE1".to_string(), FitsDataType::Float64(array1));
        data.insert("IMAGE2".to_string(), FitsDataType::Float64(array2));

        // Write and read back
        let (path, _temp) = get_test_path("roundtrip_f64_multi_hdu");

        write_typed_fits(&data, &path).unwrap();
        let read_data = read_fits_to_hashmap(&path).unwrap();

        // Verify we got the same data back
        assert_eq!(read_data.len(), 2);
        assert!(read_data.contains_key("IMAGE1"));
        assert!(read_data.contains_key("IMAGE2"));

        let read_array1 = &read_data["IMAGE1"];
        let read_array2 = &read_data["IMAGE2"];

        assert_eq!(read_array1.dim(), (3, 4));
        assert_eq!(read_array2.dim(), (2, 2));

        // Check values (allowing for floating point precision)
        // After roundtrip (write flip + read flip), positions should be restored to original
        assert_relative_eq!(read_array1[[0, 0]], 1.5, epsilon = 1e-10);
        assert_relative_eq!(read_array2[[0, 0]], 2.5, epsilon = 1e-10);
        assert_relative_eq!(read_array2[[1, 1]], 3.7, epsilon = 1e-10);
    }

    #[test]
    fn test_typed_fits_uint16() {
        let mut data = HashMap::new();
        let mut array = Array2::<u16>::zeros((3, 3));
        array[[0, 0]] = 100;
        array[[1, 1]] = 1000;
        array[[2, 2]] = 65535;

        data.insert("UINT16_IMAGE".to_string(), FitsDataType::UInt16(array));

        let (path, _temp) = get_test_path("uint16_diagonal_values");

        write_typed_fits(&data, &path).unwrap();

        // Verify file was created
        assert!(path.exists());
    }

    #[test]
    fn test_typed_fits_multiple_types() {
        let mut data = HashMap::new();

        // Add different data types
        let u8_array = Array2::<u8>::from_elem((2, 2), 128);
        data.insert("UINT8".to_string(), FitsDataType::UInt8(u8_array));

        let i32_array = Array2::<i32>::from_elem((3, 3), -42);
        data.insert("INT32".to_string(), FitsDataType::Int32(i32_array));

        let f32_array = Array2::<f32>::from_elem((4, 4), 3.14);
        data.insert("FLOAT32".to_string(), FitsDataType::Float32(f32_array));

        let (path, _temp) = get_test_path("mixed_types_u8_i32_f32");

        write_typed_fits(&data, &path).unwrap();

        // Verify file was created
        assert!(path.exists());
    }

    #[test]
    fn test_non_square_array_export() {
        let mut data = HashMap::new();

        // Test with various aspect ratios
        let wide = Array2::<f64>::from_elem((100, 200), 1.0);
        let tall = Array2::<f64>::from_elem((200, 100), 2.0);
        let square = Array2::<f64>::from_elem((150, 150), 3.0);

        data.insert("WIDE".to_string(), FitsDataType::Float64(wide));
        data.insert("TALL".to_string(), FitsDataType::Float64(tall));
        data.insert("SQUARE".to_string(), FitsDataType::Float64(square));

        let (path, _temp) = get_test_path("aspect_ratios_wide_tall_square");

        write_typed_fits(&data, &path).unwrap();

        // Read back and verify dimensions are preserved
        let read_data = read_fits_to_hashmap(&path).unwrap();

        assert_eq!(read_data["WIDE"].dim(), (100, 200));
        assert_eq!(read_data["TALL"].dim(), (200, 100));
        assert_eq!(read_data["SQUARE"].dim(), (150, 150));
    }

    #[test]
    fn test_extreme_aspect_ratios() {
        let mut data = HashMap::new();

        // Very wide array (like a panorama)
        let panorama = Array2::<u16>::from_elem((50, 1000), 512);
        data.insert("PANORAMA".to_string(), FitsDataType::UInt16(panorama));

        // Very tall array (like a column)
        let column = Array2::<u16>::from_elem((1000, 50), 1024);
        data.insert("COLUMN".to_string(), FitsDataType::UInt16(column));

        // Single row
        let row = Array2::<f32>::from_elem((1, 500), 0.5);
        data.insert("ROW".to_string(), FitsDataType::Float32(row));

        // Single column
        let col = Array2::<f32>::from_elem((500, 1), 0.25);
        data.insert("COL".to_string(), FitsDataType::Float32(col));

        let (path, _temp) = get_test_path("extreme_ratios_panorama_column_1d");

        write_typed_fits(&data, &path).unwrap();

        // Verify file was created successfully
        assert!(path.exists());
    }

    #[test]
    fn test_gradient_and_noise_patterns() {
        let mut data = HashMap::new();

        // Horizontal gradient (0 to 65535)
        let h_gradient = generate_horizontal_gradient::<u16>(256, 100, 0, 65535);
        data.insert(
            "HORIZONTAL_GRADIENT".to_string(),
            FitsDataType::UInt16(h_gradient),
        );

        // Vertical gradient
        let v_gradient = generate_vertical_gradient::<u16>(100, 256, 0, 65535);
        data.insert(
            "VERTICAL_GRADIENT".to_string(),
            FitsDataType::UInt16(v_gradient),
        );

        // Checkerboard pattern using shared function
        let checkerboard = generate_checkerboard::<u8>(8, 8, 16, 0, 255, false);
        data.insert(
            "CHECKERBOARD_16PX".to_string(),
            FitsDataType::UInt8(checkerboard),
        );

        // Gaussian-like blob in center
        let gaussian = generate_gaussian_blob(64, 10.0, 1.0);
        data.insert("GAUSSIAN_BLOB".to_string(), FitsDataType::Float32(gaussian));

        let (path, _temp) = get_test_path("test_patterns_gradient_checker_gaussian");

        write_typed_fits(&data, &path).unwrap();

        // Verify file was created
        assert!(path.exists());
    }

    #[test]
    fn test_checkerboard_11x13_blocks() {
        let mut data = HashMap::new();

        // Create 11x13 grid of 8x8 pixel blocks using shared function
        let checkerboard = generate_checkerboard::<u8>(11, 13, 8, 0, 255, true);

        data.insert(
            "CHECKERBOARD_8PX_11X13".to_string(),
            FitsDataType::UInt8(checkerboard.clone()),
        );

        let (path, _temp) = get_test_path("checkerboard_8px_blocks_11x13_grid");

        write_typed_fits(&data, &path).unwrap();

        // Verify file was created
        assert!(path.exists());

        // If writing locally, print info
        if WRITE_LOCAL {
            let (height, width) = checkerboard.dim();
            println!("Created {width}x{height} checkerboard with 11x13 blocks of 8 pixels each");
        }
    }

    #[test]
    fn test_checkerboard_square() {
        let mut data = HashMap::new();

        // Create square 10x10 grid of 10x10 pixel blocks
        // Total image size: 100x100 pixels
        let checkerboard = generate_checkerboard::<u8>(10, 10, 10, 0, 255, true);

        data.insert(
            "CHECKERBOARD_10PX_10X10".to_string(),
            FitsDataType::UInt8(checkerboard.clone()),
        );

        let (path, _temp) = get_test_path("checkerboard_10px_blocks_square_10x10");

        write_typed_fits(&data, &path).unwrap();

        // Verify file was created
        assert!(path.exists());

        // If writing locally, print info
        if WRITE_LOCAL {
            let (height, width) = checkerboard.dim();
            println!(
                "Created {width}x{height} square checkerboard with 10x10 blocks of 10 pixels each"
            );
        }
    }

    #[test]
    fn test_vertical_gradient_dark_bottom_bright_top() {
        let mut data = HashMap::new();

        // Create vertical gradient - dark at bottom (0), bright at top (65535)
        // This tests FITS orientation is correct
        let v_gradient = generate_vertical_gradient::<u16>(256, 256, 0, 65535);

        data.insert(
            "VERTICAL_GRADIENT".to_string(),
            FitsDataType::UInt16(v_gradient),
        );

        let (path, _temp) = get_test_path("vertical_gradient_dark_bottom_bright_top");

        write_typed_fits(&data, &path).unwrap();

        // Verify file was created
        assert!(path.exists());

        if WRITE_LOCAL {
            println!("Created 256x256 vertical gradient: dark bottom (0) to bright top (65535)");
        }
    }

    #[test]
    fn test_horizontal_gradient_dark_left_bright_right() {
        let mut data = HashMap::new();

        // Create horizontal gradient - dark at left (0), bright at right (65535)
        let h_gradient = generate_horizontal_gradient::<u16>(256, 256, 0, 65535);

        data.insert(
            "HORIZONTAL_GRADIENT".to_string(),
            FitsDataType::UInt16(h_gradient),
        );

        let (path, _temp) = get_test_path("horizontal_gradient_dark_left_bright_right");

        write_typed_fits(&data, &path).unwrap();

        // Verify file was created
        assert!(path.exists());

        if WRITE_LOCAL {
            println!("Created 256x256 horizontal gradient: dark left (0) to bright right (65535)");
        }
    }

    #[test]
    fn test_combined_gradients() {
        let mut data = HashMap::new();

        // Vertical gradient: dark bottom to bright top
        let v_grad = generate_vertical_gradient::<f32>(128, 256, 0.0, 1.0);
        data.insert(
            "VERT_DARK_BOT_BRIGHT_TOP".to_string(),
            FitsDataType::Float32(v_grad),
        );

        // Horizontal gradient: dark left to bright right
        let h_grad = generate_horizontal_gradient::<f32>(256, 128, 0.0, 1.0);
        data.insert(
            "HORIZ_DARK_LEFT_BRIGHT_RIGHT".to_string(),
            FitsDataType::Float32(h_grad),
        );

        // Inverted vertical: bright bottom to dark top
        let v_grad_inv = generate_vertical_gradient::<f32>(128, 256, 1.0, 0.0);
        data.insert(
            "VERT_BRIGHT_BOT_DARK_TOP".to_string(),
            FitsDataType::Float32(v_grad_inv),
        );

        // Inverted horizontal: bright left to dark right
        let h_grad_inv = generate_horizontal_gradient::<f32>(256, 128, 1.0, 0.0);
        data.insert(
            "HORIZ_BRIGHT_LEFT_DARK_RIGHT".to_string(),
            FitsDataType::Float32(h_grad_inv),
        );

        let (path, _temp) = get_test_path("gradients_all_orientations_labeled");

        write_typed_fits(&data, &path).unwrap();

        // Verify file was created
        assert!(path.exists());

        if WRITE_LOCAL {
            println!("Created multi-HDU FITS with all gradient orientations clearly labeled");
        }
    }
}

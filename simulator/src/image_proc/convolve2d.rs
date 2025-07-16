//! 2D convolution implementation for astronomical image processing.
//!
//! This module provides efficient 2D convolution operations essential for
//! astronomical image analysis, including point spread function (PSF) modeling,
//! noise filtering, and feature detection. Supports both valid and same-size
//! convolution modes with optimized Gaussian kernel generation.
//!
//! # Key Features
//!
//! - **Convolution modes**: Valid (no padding) and Same (zero-padded) modes
//! - **Gaussian kernels**: Normalized Gaussian PSF generation for any size/sigma
//! - **Performance**: Optimized inner loops for astronomical image sizes
//! - **Flexibility**: Generic kernel support for custom filter operations
//!
//! # Common Applications
//!
//! - **PSF convolution**: Apply telescope point spread functions to star fields
//! - **Image smoothing**: Reduce noise while preserving astronomical features
//! - **Feature detection**: Edge detection and structure analysis filters
//! - **Template matching**: Cross-correlation for object identification
//!
//! # Usage
//!
//! Create test images, generate Gaussian kernels, and apply PSF convolution
//! using the convolve2d function with appropriate boundary handling modes.

use ndarray::{Array2, ArrayView2};

/// Configuration options for 2D convolution operations.
///
/// Controls how convolution handles image boundaries and edge effects.
/// Different modes are optimized for different astronomical applications.
///
/// # Usage
///
/// Configure boundary handling for convolution operations. Use Same mode
/// for PSF modeling and Valid mode for feature detection.
#[derive(Debug, Clone, Copy)]
pub struct ConvolveOptions {
    /// Boundary handling mode for convolution operation
    pub mode: ConvolveMode,
}

/// Boundary handling modes for convolution operations.
///
/// Defines how the convolution handles pixels near image edges where the
/// kernel extends beyond the image boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvolveMode {
    /// Only compute output where input and kernel fully overlap.
    ///
    /// Output size: (input_rows - kernel_rows + 1, input_cols - kernel_cols + 1)
    /// Use for: Feature detection, template matching where edge artifacts are unwanted
    Valid,
    /// Use zero-padding to maintain input image size.
    ///
    /// Output size: same as input image
    /// Use for: PSF convolution, image filtering where size preservation is needed
    Same,
}

impl Default for ConvolveOptions {
    fn default() -> Self {
        Self {
            mode: ConvolveMode::Valid,
        }
    }
}

/// Perform 2D convolution of an image with a kernel.
///
/// Implements standard 2D discrete convolution optimized for astronomical
/// image processing. Supports both valid and same-size output modes with
/// appropriate boundary handling.
///
/// # Mathematical Definition
/// For each output pixel (i,j):
/// ```text
/// output[i,j] = Σ(m,n) image[i+m, j+n] * kernel[m, n]
/// ```
/// where the sum is over the valid kernel indices.
///
/// # Arguments
/// * `image` - Input image as a 2D array view (typically f64 for astronomical data)
/// * `kernel` - Convolution kernel/filter as a 2D array view
/// * `options` - Convolution configuration (None uses Valid mode)
///
/// # Returns
/// Convolved image as a new 2D Array. Size depends on the specified mode:
/// - Valid: (img_rows - ker_rows + 1, img_cols - ker_cols + 1)
/// - Same: (img_rows, img_cols)
///
/// # Performance Notes
/// - Time complexity: O(img_rows × img_cols × ker_rows × ker_cols)
/// - Memory: Allocates new array for output
/// - For large kernels, consider separable filters when possible
///
/// # Usage
///
/// Perform 2D convolution with various kernel types and boundary handling modes.
/// Use with gaussian_kernel for PSF modeling and custom kernels for feature detection.
pub fn convolve2d(
    image: &ArrayView2<f64>,
    kernel: &ArrayView2<f64>,
    options: Option<ConvolveOptions>,
) -> Array2<f64> {
    let options = options.unwrap_or_default();

    let (img_rows, img_cols) = image.dim();
    let (ker_rows, ker_cols) = kernel.dim();

    // Calculate output dimensions based on mode
    let (out_rows, out_cols) = match options.mode {
        ConvolveMode::Valid => (
            img_rows.saturating_sub(ker_rows) + 1,
            img_cols.saturating_sub(ker_cols) + 1,
        ),
        ConvolveMode::Same => (img_rows, img_cols),
    };

    // Return empty array if any dimension is zero
    if out_rows == 0 || out_cols == 0 {
        return Array2::zeros((0, 0));
    }

    let mut output = Array2::zeros((out_rows, out_cols));

    match options.mode {
        ConvolveMode::Valid => {
            // Perform convolution where kernel fully overlaps with image
            for i in 0..out_rows {
                for j in 0..out_cols {
                    let mut sum = 0.0;

                    // Apply kernel
                    for ki in 0..ker_rows {
                        for kj in 0..ker_cols {
                            sum += image[[i + ki, j + kj]] * kernel[[ki, kj]];
                        }
                    }

                    output[[i, j]] = sum;
                }
            }
        }
        ConvolveMode::Same => {
            // Calculate padding
            let pad_rows = ker_rows / 2;
            let pad_cols = ker_cols / 2;

            // Perform convolution with zero-padding
            for i in 0..out_rows {
                for j in 0..out_cols {
                    let mut sum = 0.0;

                    // Apply kernel
                    for ki in 0..ker_rows {
                        for kj in 0..ker_cols {
                            let img_row = i as isize + ki as isize - pad_rows as isize;
                            let img_col = j as isize + kj as isize - pad_cols as isize;

                            // Check if within image bounds
                            if img_row >= 0
                                && img_row < img_rows as isize
                                && img_col >= 0
                                && img_col < img_cols as isize
                            {
                                sum +=
                                    image[[img_row as usize, img_col as usize]] * kernel[[ki, kj]];
                            }
                        }
                    }

                    output[[i, j]] = sum;
                }
            }
        }
    }

    output
}

/// Create a normalized 2D Gaussian kernel for convolution.
///
/// Generates a discrete approximation to a 2D Gaussian function, commonly
/// used for image smoothing and PSF modeling in astronomical applications.
/// The kernel is automatically normalized so that all elements sum to 1.0.
///
/// # Mathematical Definition
/// Each kernel element follows:
/// ```text
/// G(x,y) = exp(-(x² + y²) / (2σ²))
/// ```
/// where (x,y) are distances from the kernel center.
///
/// # Arguments
/// * `size` - Kernel dimensions (size × size). Must be odd for proper centering.
/// * `sigma` - Standard deviation of the Gaussian distribution in pixels
///
/// # Returns
/// Normalized 2D Gaussian kernel where all elements sum to 1.0
///
/// # Panics
/// Panics if `size` is even (kernel must have a well-defined center)
///
/// # Relationship to FWHM
/// For astronomical PSF modeling:
/// ```text
/// FWHM = 2.355 × σ
/// σ = FWHM / 2.355
/// ```
///
/// # Kernel Size Guidelines
/// - Use size ≥ 6σ to capture 99.7% of the Gaussian
/// - Minimum recommended size: 2×ceil(3σ) + 1
/// - For computational efficiency, don't make size unnecessarily large
///
/// # Usage
///
/// Create normalized Gaussian kernels for PSF modeling and image smoothing.
/// Convert FWHM to sigma using the relationship: sigma = FWHM / 2.355.
pub fn gaussian_kernel(size: usize, sigma: f64) -> Array2<f64> {
    assert!(size % 2 == 1, "Kernel size must be odd");

    let mut kernel = Array2::zeros((size, size));
    let center = size as isize / 2;

    let mut sum = 0.0;

    for i in 0..size {
        for j in 0..size {
            let x = j as isize - center;
            let y = i as isize - center;

            let value = (-((x * x + y * y) as f64) / (2.0 * sigma * sigma)).exp();
            kernel[[i, j]] = value;
            sum += value;
        }
    }

    // Normalize the kernel
    if sum > 0.0 {
        kernel.mapv_inplace(|x| x / sum);
    }

    kernel
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2, Axis};

    #[test]
    fn test_convolve2d_valid() {
        let image = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let kernel = array![[1.0, 0.0], [0.0, 1.0]];

        let expected = array![[1.0 + 5.0, 2.0 + 6.0], [4.0 + 8.0, 5.0 + 9.0]];

        let result = convolve2d(
            &image.view(),
            &kernel.view(),
            Some(ConvolveOptions {
                mode: ConvolveMode::Valid,
            }),
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_convolve2d_same() {
        let image = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let kernel = array![[0.25, 0.25], [0.25, 0.25]]; // Simple 2x2 averaging kernel

        let options = ConvolveOptions {
            mode: ConvolveMode::Same,
        };

        let result = convolve2d(&image.view(), &kernel.view(), Some(options));

        // Check dimensions match the input
        assert_eq!(result.dim(), image.dim());

        // For a 3x3 image with 2x2 kernel in SAME mode, corners should involve fewer elements
        // Top-left corner only has one value contributing (itself)
        assert!((result[[0, 0]] - 0.25 * image[[0, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_kernel() {
        let kernel = gaussian_kernel(3, 1.0);

        // Sum should be close to 1.0
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Center should have highest value
        assert!(kernel[[1, 1]] > kernel[[0, 0]]);
        assert!(kernel[[1, 1]] > kernel[[0, 1]]);
        assert!(kernel[[1, 1]] > kernel[[0, 2]]);
        assert!(kernel[[1, 1]] > kernel[[1, 0]]);
        assert!(kernel[[1, 1]] > kernel[[1, 2]]);
        assert!(kernel[[1, 1]] > kernel[[2, 0]]);
        assert!(kernel[[1, 1]] > kernel[[2, 1]]);
        assert!(kernel[[1, 1]] > kernel[[2, 2]]);

        // Check that all values are non-negative
        for &value in kernel.iter() {
            assert!(value >= 0.0, "Kernel value should be non-negative");
        }
    }

    #[test]
    fn test_convolve2d_modes_comparison() {
        // Create a test input array with a simple pattern
        let mut input = Array2::zeros((10, 10));
        for i in 0..10 {
            for j in 0..10 {
                input[[i, j]] = (i * 10 + j) as f64;
            }
        }

        // Create a Gaussian kernel
        let kernel_size = 3;
        let sigma = 1.0;
        let kernel = gaussian_kernel(kernel_size, sigma);

        // Test Same mode
        let options_same = ConvolveOptions {
            mode: ConvolveMode::Same,
        };
        let result_same = convolve2d(&input.view(), &kernel.view(), Some(options_same));

        // Test Valid mode
        let options_valid = ConvolveOptions {
            mode: ConvolveMode::Valid,
        };
        let result_valid = convolve2d(&input.view(), &kernel.view(), Some(options_valid));

        // Check dimensions
        assert_eq!(result_same.dim(), input.dim());
        assert_eq!(
            result_valid.dim(),
            (
                input.dim().0 - kernel.dim().0 + 1,
                input.dim().1 - kernel.dim().1 + 1
            )
        );

        // Compare edge behavior - first row of result_same should be different from input
        // due to convolution with the Gaussian kernel
        let first_row_input = input.index_axis(Axis(0), 0);
        let first_row_same = result_same.index_axis(Axis(0), 0);

        // Check that values are different (convolution has had an effect)
        let mut all_same = true;
        for i in 0..first_row_input.len() {
            if (first_row_input[i] - first_row_same[i]).abs() > 1e-10 {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "Convolution with Same mode should change values");

        // Test with larger kernel
        let large_kernel = gaussian_kernel(5, 2.0);
        let options = ConvolveOptions {
            mode: ConvolveMode::Same,
        };
        let smoothed = convolve2d(&input.view(), &large_kernel.view(), Some(options));

        // Check that smoothing has an effect - variance should be lower
        let input_variance = calculate_variance(&input);
        let smoothed_variance = calculate_variance(&smoothed);
        assert!(
            smoothed_variance < input_variance,
            "Smoothing should reduce variance"
        );
    }

    // Helper function to calculate variance of a 2D array
    fn calculate_variance(arr: &Array2<f64>) -> f64 {
        let mean = arr.mean().unwrap();
        let mut sum_squared_diff = 0.0;
        let (rows, cols) = arr.dim();

        for i in 0..rows {
            for j in 0..cols {
                sum_squared_diff += (arr[[i, j]] - mean).powi(2);
            }
        }

        sum_squared_diff / (rows * cols) as f64
    }
}

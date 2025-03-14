//! 2D convolution implementation for image processing
//!
//! This module provides functions for performing 2D convolution operations,
//! which are essential for image filtering, blurring, and feature detection.

use ndarray::{Array2, ArrayView2};

/// Options for controlling the convolution operation
#[derive(Debug, Clone, Copy)]
pub struct ConvolveOptions {
    /// How to handle edges (currently only 'valid' is supported)
    pub mode: ConvolveMode,
}

/// Mode for handling edges in convolution
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvolveMode {
    /// Only compute output where input and kernel fully overlap
    Valid,
    /// Use zero-padding to maintain input size
    Same,
}

impl Default for ConvolveOptions {
    fn default() -> Self {
        Self {
            mode: ConvolveMode::Valid,
        }
    }
}

/// Perform 2D convolution of an image with a kernel
///
/// # Arguments
/// * `image` - Input image as a 2D array
/// * `kernel` - Convolution kernel
/// * `options` - Optional configuration for the convolution
///
/// # Returns
/// * Result of the convolution as a 2D array
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

/// Create a Gaussian kernel with specified size and sigma
///
/// # Arguments
/// * `size` - Size of the kernel (must be odd)
/// * `sigma` - Standard deviation of the Gaussian
///
/// # Returns
/// * Gaussian kernel as a 2D array
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
    use ndarray::array;

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
    }
}

//! Noise quantification and estimation utilities for astronomical image processing.
//!
//! Provides methods for analyzing and estimating noise levels in images:
//! - Chen et al. 2015 statistical noise estimation method
//! - Patch-based noise analysis
//! - Statistical noise characterization

use ndarray::{Array2, ArrayView2, Axis};

/// Transform image to patches for noise estimation
///
/// Converts an image into overlapping patches for statistical analysis.
/// Implementation of patch extraction from Chen et al. 2015 ICCV paper.
///
/// # Arguments
/// * `image` - 2D array representing the image
/// * `patch_size` - Size of square patches to extract
/// * `stride` - Step size between patches (typically 3 for noise estimation)
///
/// # Returns
/// 3D array where first dimension is flattened patch, second is patch index
fn im2patch(image: &ArrayView2<f64>, patch_size: usize, stride: usize) -> Array2<f64> {
    let (height, width) = image.dim();

    // Calculate number of patches in each dimension
    let num_h = ((height - patch_size) / stride) + 1;
    let num_w = ((width - patch_size) / stride) + 1;
    let num_patches = num_h * num_w;
    let patch_elements = patch_size * patch_size;

    // Create output array: each column is a flattened patch
    let mut patches = Array2::<f64>::zeros((patch_elements, num_patches));

    let mut patch_idx = 0;
    for i in (0..height.saturating_sub(patch_size - 1)).step_by(stride) {
        for j in (0..width.saturating_sub(patch_size - 1)).step_by(stride) {
            // Extract patch and flatten it
            let mut flat_idx = 0;
            for pi in 0..patch_size {
                for pj in 0..patch_size {
                    patches[[flat_idx, patch_idx]] = image[[i + pi, j + pj]];
                    flat_idx += 1;
                }
            }
            patch_idx += 1;
        }
    }

    patches
}

/// Estimate noise level using Chen et al. 2015 method
///
/// Implements statistical noise estimation from:
/// "An Efficient Statistical Method for Image Noise Level Estimation"
/// Chen, Zhu, Heng - ICCV 2015
///
/// # Algorithm
/// 1. Extract overlapping patches from image
/// 2. Compute patch covariance matrix
/// 3. Find eigenvalues of covariance
/// 4. Select noise level using median eigenvalue criterion
///
/// # Arguments
/// * `image` - 2D array of pixel values (any scale)
/// * `patch_size` - Size of patches for analysis (default: 8)
///
/// # Returns
/// Estimated noise standard deviation in same units as input
///
pub fn estimate_noise_level(image: &ArrayView2<f64>, patch_size: usize) -> f64 {
    // Extract patches with stride of 3 (as in original paper)
    let patches = im2patch(image, patch_size, 3);
    let (d, num_patches) = patches.dim();

    // Compute mean of each patch (column-wise mean)
    let mu = patches.mean_axis(Axis(1)).unwrap();

    // Center the patches (subtract mean from each column)
    let mut x = patches.clone();
    for i in 0..num_patches {
        for j in 0..d {
            x[[j, i]] -= mu[j];
        }
    }

    // Compute covariance matrix: (1/N) * X * X^T
    let sigma_x = x.dot(&x.t()) / num_patches as f64;

    // Compute eigenvalues using nalgebra for eigendecomposition
    use nalgebra::{DMatrix, SymmetricEigen};

    // Convert ndarray to nalgebra matrix
    let na_matrix = DMatrix::from_fn(d, d, |i, j| sigma_x[[i, j]]);

    // Compute eigenvalues
    let eigen = SymmetricEigen::new(na_matrix);
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Find noise level using tail eigenvalue selection
    for i in (0..d).rev() {
        let tau = if i > 0 {
            eigenvalues[0..i].iter().sum::<f64>() / i as f64
        } else {
            continue;
        };

        let num_greater = eigenvalues[0..i].iter().filter(|&&v| v > tau).count();
        let num_less = eigenvalues[0..i].iter().filter(|&&v| v < tau).count();

        if num_greater == num_less {
            return tau.sqrt();
        }
    }

    // Fallback: use median of smaller eigenvalues
    let mid = d / 2;
    (eigenvalues[0..mid].iter().sum::<f64>() / mid as f64).sqrt()
}

/// Estimate background level using median of downsampled image pixels
///
/// Computes median pixel value for robust background estimation.
/// Supports optional downsampling for faster computation on large images.
///
/// # Algorithm
/// 1. Sample pixels uniformly (every `downsample` pixels in both dimensions)
/// 2. Sort sampled values
/// 3. Return median value
///
/// # Arguments
/// * `image` - 2D array of pixel values
/// * `downsample` - Sampling stride (1 = no downsampling, 2 = every other pixel, etc.)
///
/// # Returns
/// Median pixel value in same units as input
///
/// # Performance
/// With downsample=N, samples ~1/N² pixels, making this O(W*H/N²) where W,H are image dimensions.
/// For 9576x6388 image with downsample=10, processes ~610k pixels instead of 61M pixels.
pub fn estimate_background(image: &ArrayView2<f64>, downsample: usize) -> f64 {
    let (height, width) = image.dim();
    let downsample = downsample.max(1); // Ensure at least 1

    // Sample pixels uniformly with stride
    let mut sampled: Vec<f64> = Vec::new();
    for i in (0..height).step_by(downsample) {
        for j in (0..width).step_by(downsample) {
            sampled.push(image[[i, j]]);
        }
    }

    // Sort and find median
    sampled.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sampled[sampled.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_proc::noise::generate::simple_normal_array;
    use approx::assert_relative_eq;

    #[test]
    fn test_im2patch_basic() {
        // Create simple 4x4 test image
        let image = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);

        // Extract 2x2 patches with stride 1
        let patches = im2patch(&image.view(), 2, 1);

        // Should have 9 patches (3x3 grid) with 4 elements each
        assert_eq!(patches.dim(), (4, 9));

        // Check first patch (top-left)
        assert_eq!(patches[[0, 0]], 0.0);
        assert_eq!(patches[[1, 0]], 1.0);
        assert_eq!(patches[[2, 0]], 4.0);
        assert_eq!(patches[[3, 0]], 5.0);
    }

    #[test]
    fn test_noise_estimation_on_pure_noise() {
        // Create image with known Gaussian noise
        let noise_std = 10.0;
        let image = simple_normal_array((100, 100), 128.0, noise_std, 42);

        // Estimate noise level
        let estimated_noise = estimate_noise_level(&image.view(), 8);

        // Should be close to the true noise level
        assert_relative_eq!(estimated_noise, noise_std, epsilon = 2.0);
    }

    #[test]
    fn test_noise_estimation_with_signal() {
        // Create base image with gradient
        let mut image =
            Array2::from_shape_fn((100, 100), |(i, j)| (i as f64 * 0.5 + j as f64 * 0.5));

        // Add known noise
        let noise_std = 5.0;
        let noise = simple_normal_array((100, 100), 0.0, noise_std, 123);
        image = image + noise;

        // Estimate noise level
        let estimated_noise = estimate_noise_level(&image.view(), 8);

        // Should estimate the noise component reasonably well
        assert_relative_eq!(estimated_noise, noise_std, epsilon = 2.0);
    }

    #[test]
    fn test_noise_estimation_small_image() {
        // Test with minimum size image
        let image = simple_normal_array((16, 16), 100.0, 3.0, 789);

        // Should handle small images gracefully
        let estimated_noise = estimate_noise_level(&image.view(), 4);

        // Just check it doesn't panic and returns positive value
        assert!(estimated_noise > 0.0);
    }

    #[test]
    fn test_doctest_example() {
        // Test the example from the estimate_noise_level function documentation
        use ndarray::Array2;

        // Create an image with some structure (not just zeros)
        let mut noisy_image = Array2::zeros((100, 100));
        // Add some noise to make the test more realistic
        let noise = simple_normal_array((100, 100), 0.0, 1.0, 42);
        noisy_image = noisy_image + noise;

        let noise_level = estimate_noise_level(&noisy_image.view(), 8);

        // Should return a positive noise estimate
        assert!(noise_level > 0.0);
        assert!(noise_level < 10.0); // Should be reasonable for our test case
    }

    #[test]
    fn test_estimate_background_uniform() {
        // Test with uniform image - should return the constant value
        let image = Array2::from_elem((100, 100), 150.0);

        let background = estimate_background(&image.view(), 1);
        assert_relative_eq!(background, 150.0, epsilon = 1e-10);

        // Should work with any downsample factor
        let background_10 = estimate_background(&image.view(), 10);
        assert_relative_eq!(background_10, 150.0, epsilon = 1e-10);
    }

    #[test]
    fn test_estimate_background_with_noise() {
        // Test with noisy background - median should be close to mean
        let background_level = 150.0;
        let noise_std = 5.0;
        let image = simple_normal_array((100, 100), background_level, noise_std, 42);

        let estimated_bg = estimate_background(&image.view(), 1);

        // Median should be close to the true background
        assert_relative_eq!(estimated_bg, background_level, epsilon = 10.0);
    }

    #[test]
    fn test_estimate_background_robust_to_outliers() {
        // Test that median is robust to bright stars (outliers)
        let mut image = Array2::from_elem((100, 100), 150.0);

        // Add some bright "stars" (outliers)
        image[[10, 10]] = 1000.0;
        image[[20, 20]] = 2000.0;
        image[[30, 30]] = 1500.0;
        image[[40, 40]] = 3000.0;

        let background = estimate_background(&image.view(), 1);

        // Median should still be close to 150 despite outliers
        assert_relative_eq!(background, 150.0, epsilon = 1.0);
    }

    #[test]
    fn test_estimate_background_downsample_factor() {
        // Test that different downsample factors give similar results
        let background_level = 150.0;
        let noise_std = 3.0;
        let image = simple_normal_array((1000, 1000), background_level, noise_std, 123);

        let bg_1 = estimate_background(&image.view(), 1);
        let bg_10 = estimate_background(&image.view(), 10);
        let bg_100 = estimate_background(&image.view(), 100);

        // All should be close to the true background
        assert_relative_eq!(bg_1, background_level, epsilon = 5.0);
        assert_relative_eq!(bg_10, background_level, epsilon = 5.0);
        assert_relative_eq!(bg_100, background_level, epsilon = 10.0);

        // Should be roughly similar to each other
        assert!((bg_1 - bg_10).abs() < 10.0);
        assert!((bg_10 - bg_100).abs() < 15.0);
    }

    #[test]
    fn test_estimate_background_small_image() {
        // Test with small image
        let image = Array2::from_elem((10, 10), 100.0);

        let background = estimate_background(&image.view(), 1);
        assert_relative_eq!(background, 100.0, epsilon = 1e-10);

        // With large downsample on small image
        let background_large = estimate_background(&image.view(), 5);
        assert_relative_eq!(background_large, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_estimate_background_gradient() {
        // Test with gradient - median should be near middle value
        let image = Array2::from_shape_fn((100, 100), |(i, j)| 100.0 + (i as f64 + j as f64) / 2.0);

        let background = estimate_background(&image.view(), 1);

        // Median should be somewhere in the middle of the range [100, 200]
        assert!(background > 120.0);
        assert!(background < 180.0);
    }

    #[test]
    fn test_estimate_background_zero_downsample() {
        // Test that downsample factor is clamped to at least 1
        let image = Array2::from_elem((100, 100), 150.0);

        // Even with downsample=0, should work (internally clamped to 1)
        let background = estimate_background(&image.view(), 0);
        assert_relative_eq!(background, 150.0, epsilon = 1e-10);
    }
}

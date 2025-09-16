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
/// # Example
/// ```ignore
/// use ndarray::Array2;
/// use shared::image_proc::noise::quantify::estimate_noise_level;
///
/// let noisy_image = Array2::zeros((100, 100));
/// let noise_level = estimate_noise_level(&noisy_image.view(), 8);
/// ```
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
}

//! Implementation of thresholding algorithms for star detection
//!
//! This module provides functions for image segmentation using Otsu's method
//! and connected components labeling.

use ndarray::{Array2, ArrayView2};

/// Computes Otsu's threshold for a grayscale image
///
/// # Arguments
/// * `image` - Grayscale image as 2D array
///
/// # Returns
/// * Threshold value
pub fn otsu_threshold(image: &ArrayView2<f64>) -> f64 {
    // Convert to histogram
    let mut histogram = vec![0; 256];
    let total_pixels = image.len() as f64;

    // Populate histogram
    for &pixel in image.iter() {
        let bin = (pixel.clamp(0.0, 1.0) * 255.0) as usize;
        histogram[bin] += 1;
    }

    // Calculate threshold using Otsu's method
    let mut sum = 0.0;
    for (i, &count) in histogram.iter().enumerate().take(256) {
        sum += i as f64 * count as f64;
    }

    let mut sum_b = 0.0;
    let mut weight_b = 0.0;
    let mut weight_f;

    let mut max_variance = 0.0;
    let mut threshold = 0.0;

    for (i, &count) in histogram.iter().enumerate().take(256) {
        weight_b += count as f64;
        if weight_b.abs() < f64::EPSILON {
            continue;
        }

        weight_f = total_pixels - weight_b;
        if weight_f.abs() < f64::EPSILON {
            break;
        }

        sum_b += (i as f64) * (count as f64);
        let mean_b = sum_b / weight_b;
        let mean_f = (sum - sum_b) / weight_f;

        let variance = weight_b * weight_f * (mean_b - mean_f).powi(2);

        if variance > max_variance {
            max_variance = variance;
            threshold = i as f64;
        }
    }

    // Normalize threshold back to [0,1] range
    threshold / 255.0
}

/// Apply threshold to an image
///
/// # Arguments
/// * `image` - Input grayscale image
/// * `threshold` - Threshold value
///
/// # Returns
/// * Binary image where 1.0 represents pixels above threshold
pub fn apply_threshold(image: &ArrayView2<f64>, threshold: f64) -> Array2<f64> {
    let shape = image.dim();
    let mut binary = Array2::zeros(shape);

    for ((i, j), &pixel) in image.indexed_iter() {
        binary[[i, j]] = if pixel >= threshold { 1.0 } else { 0.0 };
    }

    binary
}

/// Connected component labeling for binary images
///
/// # Arguments
/// * `binary_image` - Binary image where non-zero values represent objects
///
/// # Returns
/// * Labeled image where each connected component has a unique label
pub fn connected_components(binary_image: &ArrayView2<f64>) -> Array2<usize> {
    let (height, width) = binary_image.dim();
    let mut labels = Array2::zeros((height, width));
    let mut label_count = 0;

    // First pass: assign initial labels
    for i in 0..height {
        for j in 0..width {
            if binary_image[[i, j]] > 0.0 {
                // Check neighbors (4-connectivity)
                let mut neighbors = Vec::new();

                if i > 0 && labels[[i - 1, j]] > 0 {
                    neighbors.push(labels[[i - 1, j]]);
                }

                if j > 0 && labels[[i, j - 1]] > 0 {
                    neighbors.push(labels[[i, j - 1]]);
                }

                if neighbors.is_empty() {
                    // New label
                    label_count += 1;
                    labels[[i, j]] = label_count;
                } else {
                    // Use minimum of neighbor labels
                    labels[[i, j]] = *neighbors.iter().min().unwrap();
                }
            }
        }
    }

    labels
}

/// Calculate bounding boxes for labeled objects
///
/// # Arguments
/// * `labeled_image` - Image with labeled connected components
///
/// # Returns
/// * Vector of bounding boxes (min_row, min_col, max_row, max_col) for each label
pub fn get_bounding_boxes(labeled_image: &ArrayView2<usize>) -> Vec<(usize, usize, usize, usize)> {
    let max_label = labeled_image.iter().copied().max().unwrap_or(0);
    let mut bboxes = vec![(usize::MAX, usize::MAX, 0, 0); max_label + 1];

    // Skip label 0 (background) and initialize all bboxes
    for bbox in bboxes.iter_mut().skip(1).take(max_label) {
        *bbox = (usize::MAX, usize::MAX, 0, 0);
    }

    for ((row, col), &label) in labeled_image.indexed_iter() {
        if label > 0 {
            let (min_row, min_col, max_row, max_col) = bboxes[label];
            bboxes[label] = (
                min_row.min(row),
                min_col.min(col),
                max_row.max(row),
                max_col.max(col),
            );
        }
    }

    // Remove background (label 0)
    bboxes.remove(0);

    bboxes
}

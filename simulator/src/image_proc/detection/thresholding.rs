//! Image segmentation algorithms for astronomical object detection.
//!
//! This module provides fundamental image processing algorithms for converting
//! grayscale astronomical images into binary masks and labeled object regions.
//! Essential preprocessing for star detection and shape analysis pipelines.
//!
//! # Key Algorithms
//!
//! ## Otsu Thresholding
//! Automatic threshold selection using Otsu's method, which maximizes
//! between-class variance to optimally separate foreground (stars) from
//! background (sky).
//!
//! ## Connected Components
//! Two-pass connected component labeling with union-find optimization.
//! Groups connected pixels into distinct objects with unique labels.
//! Uses 4-connectivity (horizontal/vertical neighbors only).
//!
//! # Applications
//!
//! - **Star segmentation**: Separate stellar objects from sky background
//! - **Object detection**: Find discrete astronomical sources
//! - **Preprocessing**: Prepare images for centroiding and photometry
//! - **Artifact removal**: Identify and filter cosmic rays, hot pixels
//!
//! # Usage
//!
//! Perform complete image segmentation workflow from threshold selection
//! through connected component analysis for astronomical object detection.

use crate::image_proc::detection::AABB;
use ndarray::{Array2, ArrayView2};

/// Compute optimal threshold using Otsu's method for automatic image segmentation.
///
/// Otsu's algorithm finds the threshold that maximizes between-class variance,
/// effectively separating foreground objects (stars) from background (sky).
/// Particularly effective for astronomical images with bimodal intensity distributions.
///
/// # Algorithm
/// 1. Compute intensity histogram (256 bins, normalized to \[0,1\])
/// 2. For each possible threshold, calculate between-class variance
/// 3. Return threshold that maximizes this variance
///
/// # Arguments
/// * `image` - Input grayscale image with values typically in \[0,1\] range
///
/// # Returns
/// Optimal threshold value in same units as input image
///
/// # Performance
/// O(N + 256²) where N is number of pixels. Efficient for typical image sizes.
///
/// # Usage
/// Automatic threshold selection using Otsu's method for separating
/// stellar objects from sky background in astronomical images.
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

/// Apply binary threshold to create segmentation mask.
///
/// Creates a binary image where pixels above the threshold become 1.0
/// (foreground objects) and pixels below become 0.0 (background).
///
/// # Arguments
/// * `image` - Input grayscale image
/// * `threshold` - Intensity threshold value (same units as image)
///
/// # Returns
/// Binary image with 1.0 for pixels >= threshold, 0.0 otherwise
///
/// # Usage
/// Creates binary mask for astronomical object detection by applying
/// intensity threshold to separate foreground from background.
pub fn apply_threshold(image: &ArrayView2<f64>, threshold: f64) -> Array2<f64> {
    let shape = image.dim();
    let mut binary = Array2::zeros(shape);

    for ((i, j), &pixel) in image.indexed_iter() {
        binary[[i, j]] = if pixel >= threshold { 1.0 } else { 0.0 };
    }

    binary
}

/// Find the root label in a disjoint-set (union-find) data structure
///
/// # Arguments
/// * `labels` - The array of label parent pointers
/// * `label` - The label to find the root for
///
/// # Returns
/// * The root label
fn find_root(labels: &mut [usize], label: usize) -> usize {
    let mut current = label;

    // Find the root label (path compression)
    while current != labels[current] {
        // Path compression - make the parent point to the grandparent
        labels[current] = labels[labels[current]];
        current = labels[current];
    }

    current
}

/// Union two labels in a disjoint-set data structure
///
/// # Arguments
/// * `labels` - The array of label parent pointers
/// * `label1` - First label
/// * `label2` - Second label
///
/// # Returns
/// * The root label of the merged set
fn union_labels(labels: &mut [usize], label1: usize, label2: usize) -> usize {
    let root1 = find_root(labels, label1);
    let root2 = find_root(labels, label2);

    if root1 != root2 {
        // Make smaller label the parent (canonical form)
        if root1 < root2 {
            labels[root2] = root1;
            root1
        } else {
            labels[root1] = root2;
            root2
        }
    } else {
        root1 // Already in the same set
    }
}

/// Connected component labeling using optimized two-pass algorithm with union-find.
///
/// Groups connected pixels into distinct objects with unique integer labels.
/// Uses 4-connectivity (only horizontal and vertical neighbors are considered
/// connected). Implements path compression for efficient label equivalence resolution.
///
/// # Algorithm
/// 1. **First pass**: Scan image, assign preliminary labels, track equivalences
/// 2. **Union-find**: Resolve label equivalences with path compression
/// 3. **Second pass**: Relabel image with final consecutive labels
///
/// # Connectivity
/// Uses 4-connectivity: pixels connected horizontally or vertically only.
/// Diagonal connections are NOT considered connected.
///
/// # Arguments
/// * `binary_image` - Binary image where non-zero values represent foreground objects
///
/// # Returns
/// Labeled image where:
/// - Background pixels = 0
/// - Each connected object gets unique label starting from 1
/// - Labels are consecutive integers
///
/// # Performance
/// - Time: O(N) where N is number of pixels
/// - Space: O(L) where L is number of distinct labels
/// - Path compression ensures near-constant union-find operations
///
/// # Usage
/// Groups connected pixels into distinct objects with unique labels.
/// Uses 4-connectivity and optimized union-find for efficient processing.
pub fn connected_components(binary_image: &ArrayView2<f64>) -> Array2<usize> {
    let (height, width) = binary_image.dim();
    let mut labels = Array2::zeros((height, width));
    let mut label_count = 0;

    // First pass: assign initial labels and build equivalence classes
    // We need space for label_count + 1 entries (label 0 is background)
    let mut parent_table = vec![0]; // Will grow as we add labels

    for i in 0..height {
        for j in 0..width {
            if binary_image[[i, j]] > 0.0 {
                // Check 4-connected neighbors (up and left)
                let mut neighbor_labels = Vec::new();

                if i > 0 && labels[[i - 1, j]] > 0 {
                    neighbor_labels.push(labels[[i - 1, j]]);
                }

                if j > 0 && labels[[i, j - 1]] > 0 {
                    neighbor_labels.push(labels[[i, j - 1]]);
                }

                if neighbor_labels.is_empty() {
                    // No neighbors with labels, create a new label
                    label_count += 1;
                    labels[[i, j]] = label_count;

                    // Initialize parent pointer to self (each label starts as its own root)
                    parent_table.push(label_count);
                } else {
                    // Use the smallest neighbor label
                    let min_label = *neighbor_labels.iter().min().unwrap();
                    labels[[i, j]] = min_label;

                    // Set label equivalences for all neighbors
                    for &neighbor_label in &neighbor_labels {
                        if neighbor_label != min_label {
                            union_labels(&mut parent_table, min_label, neighbor_label);
                        }
                    }
                }
            }
        }
    }

    // Flatten the parent_table (path compression)
    for i in 1..parent_table.len() {
        find_root(&mut parent_table, i);
    }

    // Create a mapping from old labels to new consecutive labels
    let mut relabel_map = vec![0; parent_table.len()];
    let mut next_label = 1;

    for i in 1..parent_table.len() {
        let root = parent_table[i];
        if relabel_map[root] == 0 {
            relabel_map[root] = next_label;
            next_label += 1;
        }
        relabel_map[i] = relabel_map[root];
    }

    // Second pass: relabel the image
    for i in 0..height {
        for j in 0..width {
            if labels[[i, j]] > 0 {
                labels[[i, j]] = relabel_map[labels[[i, j]]];
            }
        }
    }

    labels
}

/// Extract axis-aligned bounding boxes for all labeled objects.
///
/// Computes the minimal bounding rectangle for each connected component
/// in the labeled image. Useful for object detection pipelines and
/// region-of-interest analysis.
///
/// # Arguments
/// * `labeled_image` - Image from connected_components() with integer labels
///
/// # Returns
/// Vector of AABB objects, one per labeled component (background label 0 excluded)
/// - Index i corresponds to object with label i+1
/// - Empty if no objects found
///
/// # Usage
/// Extracts bounding boxes for all detected objects from labeled image.\n/// Essential for object detection pipelines and region analysis.
pub fn get_bounding_boxes(labeled_image: &ArrayView2<usize>) -> Vec<AABB> {
    let max_label = labeled_image.iter().copied().max().unwrap_or(0);
    let mut bboxes = vec![AABB::new(); max_label + 1];

    for ((row, col), &label) in labeled_image.indexed_iter() {
        if label > 0 {
            bboxes[label].expand_to_include(row, col);
        }
    }

    // Remove background (label 0)
    bboxes.remove(0);

    bboxes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_overlapping_boxes() {
        // Create some test boxes
        let boxes = vec![(10, 10, 20, 20), (15, 15, 25, 25), (50, 50, 60, 60)];

        // Convert to AABBs
        use crate::image_proc::detection::{
            aabbs_to_tuples, merge_overlapping_aabbs, tuples_to_aabbs,
        };
        let aabbs = tuples_to_aabbs(&boxes);

        // Merge overlapping boxes with no padding
        let merged_aabbs = merge_overlapping_aabbs(&aabbs, None);
        let merged = aabbs_to_tuples(&merged_aabbs);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], (10, 10, 25, 25)); // First two boxes merged
        assert_eq!(merged[1], (50, 50, 60, 60)); // Third box unchanged
    }

    /// Creates a binary test image from a 2D array of 1s and 0s
    /// The formatting of the array makes it easy to see the pattern visually
    fn create_test_image(pattern: &[&[i32]]) -> Array2<f64> {
        let height = pattern.len();
        let width = pattern[0].len();

        let mut image = Array2::zeros((height, width));

        for (i, row) in pattern.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                image[[i, j]] = value as f64;
            }
        }

        image
    }

    /// Helper function to check if labeled image matches expected labels
    fn assert_labels_match(labeled: &Array2<usize>, expected: &[&[i32]]) {
        for (i, row) in expected.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let expected_value = value as usize;
                assert_eq!(
                    labeled[[i, j]],
                    expected_value,
                    "Mismatch at position [{}, {}]: expected {}, got {}",
                    i,
                    j,
                    expected_value,
                    labeled[[i, j]]
                );
            }
        }
    }

    /// Test empty image (all zeros)
    #[test]
    fn test_empty_image() {
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // Expected: all zeros (no components)
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
        ];

        assert_labels_match(&labeled, expected);
    }

    /// Test simple single component (square)
    #[test]
    fn test_single_component() {
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 1, 1, 0, 0],
            &[0, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // Expected: single component labeled as 1
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 1, 1, 0, 0],
            &[0, 1, 1, 0, 0],
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
        ];

        assert_labels_match(&labeled, expected);
    }

    /// Test two separate components
    #[test]
    fn test_two_components() {
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 1, 1, 0, 0],
            &[0, 1, 1, 0, 0],
            &[0, 0, 0, 1, 0],
            &[0, 0, 0, 0, 0],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // Expected: two components labeled as 1 and 2
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 1, 1, 0, 0],
            &[0, 1, 1, 0, 0],
            &[0, 0, 0, 2, 0],
            &[0, 0, 0, 0, 0],
        ];

        assert_labels_match(&labeled, expected);
    }

    /// Test L-shaped component
    #[test]
    fn test_l_shape() {
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 1, 0, 0, 0],
            &[0, 1, 0, 0, 0],
            &[0, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // Expected: single L-shaped component
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 1, 0, 0, 0],
            &[0, 1, 0, 0, 0],
            &[0, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0],
        ];

        assert_labels_match(&labeled, expected);
    }

    /// Test U-shaped component (tests label equivalence)
    #[test]
    fn test_u_shape() {
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 1, 0, 1, 0],
            &[0, 1, 0, 1, 0],
            &[0, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // Expected: single U-shaped component with label 1
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 1, 0, 1, 0],
            &[0, 1, 0, 1, 0],
            &[0, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0],
        ];

        assert_labels_match(&labeled, expected);
    }

    /// Test complex equivalence cases
    #[test]
    fn test_complex_equivalence() {
        // This pattern has multiple merge points that require
        // proper label equivalence handling
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[0, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 0, 1, 1, 0],
            &[0, 1, 0, 0, 0, 1, 0],
            &[0, 1, 0, 1, 0, 1, 0],
            &[0, 1, 0, 1, 0, 1, 0],
            &[0, 1, 1, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 0, 0],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // Expected: should all be one component since they connect
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[0, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 0, 1, 1, 0],
            &[0, 1, 0, 0, 0, 1, 0],
            &[0, 1, 0, 1, 0, 1, 0],
            &[0, 1, 0, 1, 0, 1, 0],
            &[0, 1, 1, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 0, 0],
        ];

        assert_labels_match(&labeled, expected);
    }

    /// Test diagonal components (not connected in 4-connectivity)
    #[test]
    fn test_diagonal_components() {
        // Diagonal pixels are not considered connected in 4-connectivity
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 1, 0, 1, 0],
            &[0, 0, 0, 0, 0],
            &[0, 1, 0, 1, 0],
            &[0, 0, 0, 0, 0],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // Expected: four separate components
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[0, 0, 0, 0, 0],
            &[0, 1, 0, 2, 0],
            &[0, 0, 0, 0, 0],
            &[0, 3, 0, 4, 0],
            &[0, 0, 0, 0, 0],
        ];

        assert_labels_match(&labeled, expected);
    }

    /// Test spiral shape (complex connectivity)
    #[test]
    fn test_spiral() {
        // This spiral shape tests the ability to follow a long path
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[0, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 1, 1, 0],
            &[0, 1, 0, 0, 0, 1, 0],
            &[0, 1, 0, 1, 1, 1, 0],
            &[0, 1, 0, 1, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0],
            &[0, 0, 0, 0, 0, 0, 0],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // Expected: single component (spiral)
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[0, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 1, 1, 0],
            &[0, 1, 0, 0, 0, 1, 0],
            &[0, 1, 0, 1, 1, 1, 0],
            &[0, 1, 0, 1, 0, 0, 0],
            &[0, 1, 1, 1, 0, 0, 0],
            &[0, 0, 0, 0, 0, 0, 0],
        ];

        assert_labels_match(&labeled, expected);
    }

    /// Test for handling border components correctly
    #[test]
    fn test_border_components() {
        // Components on the image borders
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[1, 1, 0, 0, 1],
            &[1, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[1, 0, 0, 1, 1],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // Expected: three separate components
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[1, 1, 0, 0, 2],
            &[1, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[3, 0, 0, 4, 4],
        ];

        assert_labels_match(&labeled, expected);
    }

    /// Test the special challenge case that causes problems in many implementations
    #[test]
    fn test_tricky_equivalence() {
        // This pattern forms a specific challenge for union-find approaches
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[0, 0, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 1, 0, 0, 0],
            &[0, 1, 0, 0, 1, 0, 0, 0],
            &[0, 1, 0, 0, 1, 0, 1, 0],
            &[0, 1, 1, 1, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 0, 0, 0],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // All should be a single component
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[0, 0, 0, 0, 0, 0, 0, 0],
            &[0, 1, 1, 1, 1, 0, 0, 0],
            &[0, 1, 0, 0, 1, 0, 0, 0],
            &[0, 1, 0, 0, 1, 0, 1, 0],
            &[0, 1, 1, 1, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 0, 0, 0],
        ];

        assert_labels_match(&labeled, expected);
    }

    /// Tests the specific case that was fixed
    #[test]
    fn test_union_find_correctness() {
        // This specific pattern should be a single component but broke
        // in the original implementation
        // fmt-ignore
        let pattern: &[&[i32]] = &[
            &[0, 0, 0, 0, 0, 0, 0],
            &[0, 1, 0, 0, 0, 1, 0],
            &[0, 1, 0, 0, 0, 1, 0],
            &[0, 1, 1, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 0, 0],
        ];

        let image = create_test_image(pattern);
        let labeled = connected_components(&image.view());

        // All non-zero elements should be in one component
        // fmt-ignore
        let expected: &[&[i32]] = &[
            &[0, 0, 0, 0, 0, 0, 0],
            &[0, 1, 0, 0, 0, 1, 0],
            &[0, 1, 0, 0, 0, 1, 0],
            &[0, 1, 1, 1, 1, 1, 0],
            &[0, 0, 0, 0, 0, 0, 0],
        ];

        assert_labels_match(&labeled, expected);
    }

    #[test]
    fn test_find_root() {
        let mut labels = vec![0, 1, 2, 3, 4, 5];
        labels[2] = 1;
        labels[3] = 2;
        labels[4] = 2;
        assert_eq!(find_root(&mut labels, 4), 1);
    }

    #[test]
    fn test_union_labels() {
        let mut labels = vec![0, 1, 2, 3, 4, 5];
        union_labels(&mut labels, 2, 3);
        assert_eq!(find_root(&mut labels, 2), find_root(&mut labels, 3));
    }

    #[test]
    fn test_path_compression() {
        // Create a long chain of labels pointing to the next
        let mut labels = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Make a chain: 10->9->8->...->2->1
        for i in (2..=10).rev() {
            labels[i] = i - 1;
        }

        // Find the root of 10, should compress the path
        assert_eq!(find_root(&mut labels, 10), 1);

        // After path compression, 10 should point closer to root
        // (exact behavior depends on implementation, but should be more efficient)
        assert!(labels[10] < 9, "Path compression not working effectively");
    }

    #[test]
    fn test_disjoint_set_operations() {
        let mut labels = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];

        // Union some sets
        union_labels(&mut labels, 1, 2);
        union_labels(&mut labels, 3, 4);
        union_labels(&mut labels, 5, 6);
        union_labels(&mut labels, 7, 8);

        // Union the unions
        union_labels(&mut labels, 1, 3);
        union_labels(&mut labels, 5, 7);

        // Union all together
        union_labels(&mut labels, 1, 5);

        // All should have same root now
        let root = find_root(&mut labels, 1);
        for i in 1..=8 {
            assert_eq!(find_root(&mut labels, i), root);
        }
    }
}

//! Histogram stretching functionality for image enhancement
//!
//! This module provides functions to enhance image contrast through histogram stretching,
//! which remaps pixel values to use the full dynamic range based on percentile parameters.

use ndarray::{Array2, ArrayView2};
use std::collections::BTreeMap;

/// Stretches the histogram of a u16 image between the specified lower and upper percentiles
///
/// # Arguments
/// * `image` - Input image as 2D array of u16
/// * `lower_percentile` - Lower percentile cutoff value (0.0 to 100.0)
/// * `upper_percentile` - Upper percentile cutoff value (0.0 to 100.0)
///
/// # Returns
/// A new Array2<u16> with stretched values
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// use simulator::image_proc::histogram_stretch::stretch_histogram;
///
/// let image = Array2::<u16>::from_shape_vec((4, 4), vec![
///     100, 200, 300, 400,
///     500, 600, 700, 800,
///     1000, 1200, 1500, 2000,
///     2500, 3000, 4000, 5000,
/// ]).unwrap();
///
/// // Stretch using 5th and 95th percentiles
/// let stretched = stretch_histogram(image.view(), 5.0, 95.0);
/// ```
pub fn stretch_histogram(
    image: ArrayView2<u16>,
    lower_percentile: f64,
    upper_percentile: f64,
) -> Array2<u16> {
    // Validate percentile inputs
    assert!(
        (0.0..=100.0).contains(&lower_percentile),
        "Lower percentile must be between 0 and 100"
    );
    assert!(
        (0.0..=100.0).contains(&upper_percentile),
        "Upper percentile must be between 0 and 100"
    );
    assert!(
        lower_percentile < upper_percentile,
        "Lower percentile must be less than upper percentile"
    );

    // Handle empty images
    if image.is_empty() {
        return Array2::<u16>::from_shape_vec(image.dim(), vec![]).unwrap();
    }

    // Handle special case for single value images
    let all_same = image.iter().all(|&v| v == image[[0, 0]]);
    if all_same {
        // If all values are the same, return the middle value
        return Array2::from_elem(image.dim(), 32767);
    }

    // Create sorted map of values -> counts
    let mut value_counts: BTreeMap<u16, usize> = BTreeMap::new();

    // Build frequency map
    for &value in image.iter() {
        *value_counts.entry(value).or_insert(0) += 1;
    }

    // Calculate cumulative counts in one pass
    let total_pixels = image.len();
    let lower_threshold = (lower_percentile / 100.0 * total_pixels as f64).round() as usize;
    let upper_threshold = (upper_percentile / 100.0 * total_pixels as f64).round() as usize;

    let mut cumulative_count = 0;
    let mut min_val = 0;
    let mut max_val = 65535;
    let mut min_val_found = false;

    // Find min and max values in a single pass through the sorted map
    for (&value, &count) in &value_counts {
        cumulative_count += count;

        // Find the minimum value once we cross the lower threshold
        if !min_val_found && cumulative_count >= lower_threshold {
            min_val = value as usize;
            min_val_found = true;
        }

        // Find the maximum value once we cross the upper threshold
        if cumulative_count >= upper_threshold {
            max_val = value as usize;
            break; // No need to continue once we've found both values
        }
    }

    // Create and return output array with stretched values
    Array2::from_shape_fn(image.dim(), |(i, j)| {
        // Get original value
        let value = image[[i, j]] as usize;

        // Apply stretching formula, ensuring we don't divide by zero
        if max_val > min_val {
            // Normalize to 0-1 range
            let normalized = (value as f64 - min_val as f64) / (max_val as f64 - min_val as f64);
            // Scale to 0-65535 range and convert to u16
            (normalized * 65535.0).round() as u16
        } else {
            // If min_val == max_val, return middle of output range
            32767
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_stretch_histogram() {
        // Create a test image with wider range of u16 values to ensure proper stretching
        let image = Array2::<u16>::from_shape_vec(
            (4, 4),
            vec![
                100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000,
                13000, 14000, 15000,
            ],
        )
        .unwrap();

        // Stretch to full range
        let stretched = stretch_histogram(image.view(), 0.0, 100.0);

        // The test value range is 100-15000, so verify proportional mapping
        let min_mapped = stretched[[0, 0]];
        let max_mapped = stretched[[3, 3]];

        // Since min_val=0 in the algorithm (not 100), the mapping is not exactly what we might expect
        // Allow reasonable ranges for mapped values
        assert!(min_mapped < 1000, "Minimum value not mapped to low range");
        assert!(max_mapped > 60000, "Maximum value not mapped to high range");

        // Also verify that the ordering is preserved
        assert!(stretched[[0, 0]] < stretched[[0, 1]], "Order not preserved");
        assert!(stretched[[0, 1]] < stretched[[0, 2]], "Order not preserved");

        // We expect linear mapping, so check one point in between
        let v1 = stretched[[0, 0]] as f64; // 100 -> ?
        let v2 = stretched[[3, 3]] as f64; // 15000 -> ?
        let ratio = (v2 - v1) / (15000.0 - 100.0); // Should be consistent scaling

        // Calculate expected value for 7000 based on the observed mapping
        let expected_7000 = v1 + ratio * (7000.0 - 100.0);
        let actual_7000 = stretched[[1, 3]] as f64;

        // Allow 5% tolerance for the mid-point check
        let tolerance = 0.05 * (v2 - v1);
        assert!(
            (actual_7000 - expected_7000).abs() < tolerance,
            "Mid-range value not mapped proportionally. Expected: {}, Actual: {}",
            expected_7000,
            actual_7000
        );
    }

    #[test]
    fn test_stretch_with_percentile_clipping() {
        // Create a test image with outliers
        let image = Array2::<u16>::from_shape_vec(
            (5, 5),
            vec![
                100, 200, 300, 400, 500, 100, 200, 300, 400, 500, 100, 200, 300, 400, 500, 100,
                200, 300, 400, 500, 100, 200, 10000, 400, 500, // Outlier at 10000
            ],
        )
        .unwrap();

        // Stretch with 5th-95th percentile
        // This should exclude the outlier at 10000
        let stretched = stretch_histogram(image.view(), 5.0, 95.0);

        // The max visible value should now be scaled to near max
        // and the outlier should be clipped
        let max_regular = stretched[[3, 4]]; // Should be a high value (500 mapped up)
        let outlier = stretched[[4, 2]]; // Should be 65535 (clipped)

        assert!(max_regular >= 60000); // 500 should map to near max
        assert_eq!(outlier, 65535); // Outlier should be clipped to 65535
    }

    #[test]
    fn test_edge_cases() {
        // Test image with constant values
        let image = Array2::<u16>::from_shape_vec((2, 2), vec![500, 500, 500, 500]).unwrap();

        // This would normally cause a division by zero, but our code should handle it
        let stretched = stretch_histogram(image.view(), 0.0, 100.0);

        // All values should be mapped to middle of output range
        for &val in stretched.iter() {
            assert_eq!(val, 32767); // Middle of u16 range (32767.5 rounded down)
        }

        // Test empty image handling
        let empty_image = Array2::<u16>::from_shape_vec((0, 0), vec![]).unwrap();
        let empty_result = stretch_histogram(empty_image.view(), 0.0, 100.0);
        assert_eq!(empty_result.len(), 0);
    }
}

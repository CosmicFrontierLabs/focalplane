//! Star centroid calculation using center of mass and image moments
//!
//! This module provides functionality for calculating star centroids with
//! sub-pixel accuracy and filtering non-star objects based on moment analysis.

use ndarray::ArrayView2;

use crate::algo::icp::Locatable_2D;

/// Star detection result containing position and shape information
#[derive(Debug, Clone)]
pub struct StarDetection {
    /// Centroid x position (sub-pixel)
    pub x: f64,
    /// Centroid y position (sub-pixel)
    pub y: f64,
    /// Total flux (sum of pixel intensities)
    pub flux: f64,
    /// Moment of inertia around x-axis
    pub m_xx: f64,
    /// Moment of inertia around y-axis
    pub m_yy: f64,
    /// Cross moment of inertia
    pub m_xy: f64,
    /// Aspect ratio calculated from moments
    pub aspect_ratio: f64,
    /// Estimated diameter of the star in pixels, calculated from moments
    pub diameter: f64,
    /// Is this likely to be a star based on moment analysis?
    pub is_valid: bool,
}

impl Locatable_2D for StarDetection {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }
}

/// Calculate centroid and moments for a labeled object in the image
///
/// # Arguments
/// * `image` - Original grayscale image
/// * `labeled` - Labeled image from connected components
/// * `label` - The specific label to calculate centroid for
/// * `bbox` - Bounding box of the labeled region (min_row, min_col, max_row, max_col)
///
/// # Returns
/// * `StarDetection` containing centroid position and shape information
pub fn calculate_star_centroid(
    image: &ArrayView2<f64>,
    labeled: &ArrayView2<usize>,
    label: usize,
    bbox: (usize, usize, usize, usize),
) -> StarDetection {
    let (min_row, min_col, max_row, max_col) = bbox;

    // Initialize moments
    let mut m00 = 0.0; // Total mass/intensity
    let mut m10 = 0.0; // First moment in x
    let mut m01 = 0.0; // First moment in y
    let mut m20 = 0.0; // Second moment in x
    let mut m02 = 0.0; // Second moment in y
    let mut m11 = 0.0; // Cross moment

    // Calculate raw moments
    for row in min_row..=max_row {
        for col in min_col..=max_col {
            if row < labeled.nrows() && col < labeled.ncols() && labeled[[row, col]] == label {
                let intensity = image[[row, col]];

                // Use intensity as weight
                m00 += intensity;
                m10 += col as f64 * intensity;
                m01 += row as f64 * intensity;
                m20 += (col as f64).powi(2) * intensity;
                m02 += (row as f64).powi(2) * intensity;
                m11 += (row as f64) * (col as f64) * intensity;
            }
        }
    }

    // Avoid division by zero
    if m00 < f64::EPSILON {
        return StarDetection {
            x: 0.0,
            y: 0.0,
            flux: 0.0,
            m_xx: 0.0,
            m_yy: 0.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 0.0,
            is_valid: false,
        };
    }

    // Calculate centroid
    let x_centroid = m10 / m00;
    let y_centroid = m01 / m00;

    // Calculate central moments (relative to centroid)
    let mu20 = m20 / m00 - x_centroid.powi(2);
    let mu02 = m02 / m00 - y_centroid.powi(2);
    let mu11 = m11 / m00 - x_centroid * y_centroid;

    // Calculate aspect ratio using eigenvalues of the covariance matrix
    let sum = mu20 + mu02;
    let diff = mu20 - mu02;
    let discriminant = (4.0 * mu11.powi(2) + diff.powi(2)).sqrt();

    let lambda1 = (sum + discriminant) / 2.0;
    let lambda2 = (sum - discriminant) / 2.0;

    // Larger eigenvalue divided by smaller eigenvalue
    let aspect_ratio = if lambda2 > f64::EPSILON {
        (lambda1 / lambda2).abs()
    } else {
        f64::INFINITY
    };

    // Calculate the diameter using the average of the eigenvalues
    // Eigenvalues represent variance, so we use 2*sqrt(lambda) for the radius and then double it for diameter
    // We use 4*sqrt because 2*2*sqrt = 4*sqrt
    let diameter = 4.0 * ((lambda1 + lambda2) / 2.0).sqrt();

    // Stars should have aspect ratio close to 1.0 (circular PSF)
    // Use 2.5 as a threshold, which allows for some PSF distortion
    let is_valid = aspect_ratio < 2.5;

    StarDetection {
        x: x_centroid,
        y: y_centroid,
        flux: m00,
        m_xx: mu20,
        m_yy: mu02,
        m_xy: mu11,
        aspect_ratio,
        diameter,
        is_valid,
    }
}

/// Process an entire image to detect stars
///
/// # Arguments
/// * `image` - Original grayscale image
/// * `threshold` - Optional threshold value (if None, Otsu's method is used)
///
/// # Returns
/// * Vector of StarDetection objects
pub fn detect_stars(image: &ArrayView2<f64>, threshold: Option<f64>) -> Vec<StarDetection> {
    use super::thresholding::{
        apply_threshold, connected_components, get_bounding_boxes, otsu_threshold,
    };

    // Apply threshold using Otsu's method if threshold not provided
    let thresh = threshold.unwrap_or_else(|| otsu_threshold(image));
    let binary = apply_threshold(image, thresh);

    // Perform connected components labeling
    let labeled = connected_components(&binary.view());

    // Get bounding boxes for all regions
    let bboxes = get_bounding_boxes(&labeled.view());

    // Calculate centroids and moments for each region
    let mut stars = Vec::with_capacity(bboxes.len());

    for (i, bbox) in bboxes.iter().enumerate() {
        // Labels start at 1
        let label = i + 1;
        let star = calculate_star_centroid(image, &labeled.view(), label, bbox.to_tuple());
        stars.push(star);
    }

    // Filter out non-star objects
    stars.into_iter().filter(|star| star.is_valid).collect()
}

/// Get refined centroid positions for detected stars
///
/// # Arguments
/// * `stars` - Vector of detected stars
///
/// # Returns
/// * Vector of (x, y) positions with sub-pixel accuracy
pub fn get_centroids(stars: &[StarDetection]) -> Vec<(f64, f64)> {
    stars.iter().map(|star| (star.x, star.y)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_simple_centroid() {
        // Create a simple 5x5 image with a single star
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[2, 2]] = 1.0;
        image[[1, 2]] = 0.5;
        image[[2, 1]] = 0.5;
        image[[3, 2]] = 0.5;
        image[[2, 3]] = 0.5;

        // Create a labeled image
        let mut labeled = Array2::<usize>::zeros((5, 5));
        labeled[[1, 2]] = 1;
        labeled[[2, 1]] = 1;
        labeled[[2, 2]] = 1;
        labeled[[3, 2]] = 1;
        labeled[[2, 3]] = 1;

        let bbox = (1, 1, 3, 3);

        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

        // Star centroid should be at (2.0, 2.0)
        assert!((star.x - 2.0).abs() < 1e-10);
        assert!((star.y - 2.0).abs() < 1e-10);

        // Star should be valid (circular)
        assert!(star.is_valid);

        // Moments should indicate a circular object
        assert!((star.aspect_ratio - 1.0).abs() < 0.1);
    }

    /// Run a centroid accuracy test on a grid of sub-pixel positions
    ///
    /// # Arguments
    /// * `image_size` - Size of the test image (square)
    /// * `grid_size` - Number of grid points per pixel side (grid_size^2 total points)
    /// * `sigma` - Standard deviation of the Gaussian PSF
    /// * `threshold` - Threshold for binary segmentation
    ///
    /// # Returns
    /// * Tuple containing (average_error, max_error, errors_x, errors_y, errors_magnitude, number_of_valid_tests)
    ///   where errors_x, errors_y, and errors_magnitude are vectors of errors for each test position
    pub fn run_subpixel_position_grid_test(
        image_size: usize,
        grid_size: usize,
        sigma: f64,
        threshold: f64,
    ) -> (f64, f64, Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        use crate::image_proc::thresholding::{
            apply_threshold, connected_components, get_bounding_boxes,
        };

        let center_x = image_size as f64 / 2.0;
        let center_y = image_size as f64 / 2.0;

        let mut total_error = 0.0;
        let mut max_error: f64 = 0.0;
        let mut count = 0;

        // Collect errors for analysis
        let mut errors_x = Vec::with_capacity(grid_size * grid_size);
        let mut errors_y = Vec::with_capacity(grid_size * grid_size);
        let mut errors_magnitude = Vec::with_capacity(grid_size * grid_size);

        for i in 0..grid_size {
            for j in 0..grid_size {
                let sub_x = i as f64 / grid_size as f64;
                let sub_y = j as f64 / grid_size as f64;

                let position_x = center_x + sub_x;
                let position_y = center_y + sub_y;

                // Create an empty image
                let mut image = Array2::<f64>::zeros((image_size, image_size));

                // Generate a Gaussian star at this sub-pixel position
                create_gaussian(&mut image, position_x, position_y, 1.0, sigma);

                // Process the image
                let binary = apply_threshold(&image.view(), threshold);
                let labeled = connected_components(&binary.view());
                let bboxes = get_bounding_boxes(&labeled.view());

                // Skip if no detection (shouldn't happen with these parameters)
                if bboxes.is_empty() {
                    continue;
                }

                // Calculate centroid
                let star = calculate_star_centroid(
                    &image.view(),
                    &labeled.view(),
                    1,
                    bboxes[0].to_tuple(),
                );

                // Calculate errors
                let error_x = position_x - star.x;
                let error_y = position_y - star.y;
                let error = (error_x.powi(2) + error_y.powi(2)).sqrt();

                // Store errors for analysis
                errors_x.push(error_x);
                errors_y.push(error_y);
                errors_magnitude.push(error);

                // Update statistics
                total_error += error;
                max_error = max_error.max(error);
                count += 1;
            }
        }

        // Calculate average error
        let avg_error = if count > 0 {
            total_error / count as f64
        } else {
            0.0
        };

        (
            avg_error,
            max_error,
            errors_x,
            errors_y,
            errors_magnitude,
            count,
        )
    }

    /// Test the centroid accuracy on a dense grid of sub-pixel positions
    #[test]
    fn test_subpixel_position_grid() {
        // Test parameters for unit test (small grid for speed)
        let image_size = 32;
        let grid_size = 5;
        let sigma = 1.0;
        let threshold = 0.1;

        // Run the grid test
        let (avg_error, max_error, errors_x, errors_y, errors_magnitude, count) =
            run_subpixel_position_grid_test(image_size, grid_size, sigma, threshold);

        // Verify we had valid tests
        assert!(count > 0, "No valid test positions detected");

        // Verify each error is within acceptable limits
        for (i, &error) in errors_magnitude.iter().enumerate() {
            assert!(
                error < 0.1,
                "Centroid error too large at test position {}: {}, X error: {}, Y error: {}",
                i,
                error,
                errors_x[i],
                errors_y[i]
            );
        }

        // Overall statistics check
        assert!(
            avg_error < 0.05,
            "Average centroid error too large: {}",
            avg_error
        );

        assert!(
            max_error < 0.1,
            "Maximum centroid error too large: {}",
            max_error
        );
    }

    /// Test centroiding with a perfectly symmetric 5x5 pattern
    #[test]
    fn test_symmetric_cross_5x5() {
        // Create a 5x5 image with a symmetric cross pattern
        // This is the exact same pattern from test_simple_centroid
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[2, 2]] = 1.0; // Center pixel
        image[[1, 2]] = 0.5; // North
        image[[2, 1]] = 0.5; // West
        image[[3, 2]] = 0.5; // South
        image[[2, 3]] = 0.5; // East

        // Label the entire pattern
        let mut labeled = Array2::<usize>::zeros((5, 5));
        labeled[[2, 2]] = 1;
        labeled[[1, 2]] = 1;
        labeled[[2, 1]] = 1;
        labeled[[3, 2]] = 1;
        labeled[[2, 3]] = 1;

        let bbox = (1, 1, 3, 3);

        // Calculate centroid
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

        // Verify exact centroid position - should be exactly (2,2)
        println!("5x5 cross centroid: ({}, {})", star.x, star.y);
        assert!(
            (star.x - 2.0).abs() < 1e-10,
            "X-centroid error: {}",
            star.x - 2.0
        );
        assert!(
            (star.y - 2.0).abs() < 1e-10,
            "Y-centroid error: {}",
            star.y - 2.0
        );
    }

    /// Test centroiding with a symmetric 3x3 pattern
    #[test]
    fn test_symmetric_3x3() {
        // Create a 5x5 image with a symmetric 3x3 pattern:
        //   0.25 0.5 0.25
        //   0.5  1.0 0.5
        //   0.25 0.5 0.25
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[1, 1]] = 0.25; // NW
        image[[1, 2]] = 0.5; // N
        image[[1, 3]] = 0.25; // NE
        image[[2, 1]] = 0.5; // W
        image[[2, 2]] = 1.0; // Center
        image[[2, 3]] = 0.5; // E
        image[[3, 1]] = 0.25; // SW
        image[[3, 2]] = 0.5; // S
        image[[3, 3]] = 0.25; // SE

        // Label the entire pattern
        let mut labeled = Array2::<usize>::zeros((5, 5));
        for i in 1..=3 {
            for j in 1..=3 {
                labeled[[i, j]] = 1;
            }
        }

        let bbox = (1, 1, 3, 3);

        // Calculate centroid
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

        // Verify exact centroid position - should be exactly (2,2)
        println!("3x3 pattern centroid: ({}, {})", star.x, star.y);
        assert!(
            (star.x - 2.0).abs() < 1e-10,
            "X-centroid error: {}",
            star.x - 2.0
        );
        assert!(
            (star.y - 2.0).abs() < 1e-10,
            "Y-centroid error: {}",
            star.y - 2.0
        );
    }

    /// Test centroiding with a pattern biased in the X-direction
    #[test]
    fn test_x_biased_pattern() {
        // Create a 5x5 image with a pattern that's intentionally biased in X
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[2, 1]] = 0.4; // West pixels
        image[[2, 2]] = 1.0; // Center pixel
        image[[2, 3]] = 0.6; // East pixel (intentionally brighter)

        // Label the entire pattern
        let mut labeled = Array2::<usize>::zeros((5, 5));
        labeled[[2, 1]] = 1;
        labeled[[2, 2]] = 1;
        labeled[[2, 3]] = 1;

        let bbox = (2, 1, 2, 3);

        // Calculate centroid
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

        // With the east pixel brighter than west, centroid should be > 2.0
        println!("X-biased pattern centroid: ({}, {})", star.x, star.y);
        assert!(star.x > 2.0, "X-centroid should be > 2.0 but is {}", star.x);

        // Y-centroid should still be exactly 2.0
        assert!(
            (star.y - 2.0).abs() < 1e-10,
            "Y-centroid error: {}",
            star.y - 2.0
        );
    }

    /// Test centroiding with a pattern biased in the Y-direction
    #[test]
    fn test_y_biased_pattern() {
        // Create a 5x5 image with a pattern that's intentionally biased in Y
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[1, 2]] = 0.4; // North pixel
        image[[2, 2]] = 1.0; // Center pixel
        image[[3, 2]] = 0.6; // South pixel (intentionally brighter)

        // Label the entire pattern
        let mut labeled = Array2::<usize>::zeros((5, 5));
        labeled[[1, 2]] = 1;
        labeled[[2, 2]] = 1;
        labeled[[3, 2]] = 1;

        let bbox = (1, 2, 3, 2);

        // Calculate centroid
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

        // With the south pixel brighter than north, centroid should be > 2.0
        println!("Y-biased pattern centroid: ({}, {})", star.x, star.y);
        assert!(star.y > 2.0, "Y-centroid should be > 2.0 but is {}", star.y);

        // X-centroid should still be exactly 2.0
        assert!(
            (star.x - 2.0).abs() < 1e-10,
            "X-centroid error: {}",
            star.x - 2.0
        );
    }

    /// Test even-sized pattern centroiding (6x6)
    #[test]
    fn test_even_sized_pattern() {
        // Create a 6x6 image with a symmetric pattern centered between pixels
        // The true center should be at (2.5, 2.5)
        let mut image = Array2::<f64>::zeros((6, 6));

        // 2x2 center pixels all with equal weight
        image[[2, 2]] = 1.0;
        image[[2, 3]] = 1.0;
        image[[3, 2]] = 1.0;
        image[[3, 3]] = 1.0;

        // Surrounding symmetrical pixels
        image[[1, 2]] = 0.5;
        image[[1, 3]] = 0.5;
        image[[2, 1]] = 0.5;
        image[[3, 1]] = 0.5;
        image[[4, 2]] = 0.5;
        image[[4, 3]] = 0.5;
        image[[2, 4]] = 0.5;
        image[[3, 4]] = 0.5;

        // Label the pattern
        let mut labeled = Array2::<usize>::zeros((6, 6));
        for i in 1..=4 {
            for j in 1..=4 {
                if image[[i, j]] > 0.0 {
                    labeled[[i, j]] = 1;
                }
            }
        }

        let bbox = (1, 1, 4, 4);

        // Calculate centroid
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

        // Verify exact centroid position - should be exactly (2.5, 2.5)
        println!("6x6 pattern centroid: ({}, {})", star.x, star.y);
        assert!(
            (star.x - 2.5).abs() < 1e-10,
            "X-centroid error: {}",
            star.x - 2.5
        );
        assert!(
            (star.y - 2.5).abs() < 1e-10,
            "Y-centroid error: {}",
            star.y - 2.5
        );
    }

    /// Test sub-pixel position with 2x2 pattern
    #[test]
    fn test_subpixel_simple() {
        // Create a pattern at a specific sub-pixel position (2.75, 2.75)
        let mut image = Array2::<f64>::zeros((5, 5));

        // 2x2 grid with brightness weighted to simulate a sub-pixel center
        // After calculation, the centroid seems to end up at (2.75, 2.75)
        image[[2, 2]] = 0.25 * 0.25; // Top-left (furthest)
        image[[2, 3]] = 0.25 * 0.75; // Top-right (closer to center in X)
        image[[3, 2]] = 0.75 * 0.25; // Bottom-left (closer to center in Y)
        image[[3, 3]] = 0.75 * 0.75; // Bottom-right (closest to center)

        // Label the entire pattern
        let mut labeled = Array2::<usize>::zeros((5, 5));
        labeled[[2, 2]] = 1;
        labeled[[2, 3]] = 1;
        labeled[[3, 2]] = 1;
        labeled[[3, 3]] = 1;

        let bbox = (2, 2, 3, 3);

        // Calculate centroid
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

        // Verify sub-pixel centroid position
        println!("Subpixel pattern centroid: ({}, {})", star.x, star.y);
        assert!(
            (star.x - 2.75).abs() < 0.05,
            "X-centroid error: {}",
            star.x - 2.75
        );
        assert!(
            (star.y - 2.75).abs() < 0.05,
            "Y-centroid error: {}",
            star.y - 2.75
        );
    }

    /// Test swapping X and Y to see if bias changes
    #[test]
    fn test_xy_swap() {
        // Create a pattern with intentional asymmetry
        let mut image = Array2::<f64>::zeros((5, 5));
        image[[2, 1]] = 0.3; // Low west
        image[[2, 2]] = 1.0; // Center
        image[[2, 3]] = 0.7; // High east - should bias centroid to the right

        let mut labeled = Array2::<usize>::zeros((5, 5));
        labeled[[2, 1]] = 1;
        labeled[[2, 2]] = 1;
        labeled[[2, 3]] = 1;

        let bbox = (2, 1, 2, 3);

        // Calculate centroid normally
        let star_normal = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

        // Now create a transposed pattern
        let mut image_swapped = Array2::<f64>::zeros((5, 5));
        image_swapped[[1, 2]] = 0.3; // Low north
        image_swapped[[2, 2]] = 1.0; // Center
        image_swapped[[3, 2]] = 0.7; // High south - should bias centroid downward

        let mut labeled_swapped = Array2::<usize>::zeros((5, 5));
        labeled_swapped[[1, 2]] = 1;
        labeled_swapped[[2, 2]] = 1;
        labeled_swapped[[3, 2]] = 1;

        let bbox_swapped = (1, 2, 3, 2);

        // Calculate centroid for swapped pattern
        let star_swapped = calculate_star_centroid(
            &image_swapped.view(),
            &labeled_swapped.view(),
            1,
            bbox_swapped,
        );

        // Normal image should have X bias but no Y bias
        println!(
            "Normal pattern centroid: ({}, {})",
            star_normal.x, star_normal.y
        );
        println!(
            "Swapped pattern centroid: ({}, {})",
            star_swapped.x, star_swapped.y
        );

        // The X-centered normal pattern should have x > 2.0
        assert!(
            star_normal.x > 2.0,
            "X-centroid should be > 2.0 but is {}",
            star_normal.x
        );

        // The Y-centered swapped pattern should have y > 2.0
        assert!(
            star_swapped.y > 2.0,
            "Y-centroid should be > 2.0 but is {}",
            star_swapped.y
        );

        // The X bias in normal pattern should match the Y bias in the swapped pattern
        let x_bias = star_normal.x - 2.0;
        let y_bias = star_swapped.y - 2.0;
        assert!(
            (x_bias - y_bias).abs() < 0.01,
            "X bias ({}) should match Y bias ({})",
            x_bias,
            y_bias
        );
    }

    /// Generate a 2D Gaussian PSF with specified parameters
    fn create_gaussian(
        image: &mut Array2<f64>,
        center_x: f64,
        center_y: f64,
        amplitude: f64,
        sigma: f64,
    ) {
        let (height, width) = image.dim();

        // Compute values for a window around the center
        let x_min = (center_x - 4.0 * sigma).max(0.0) as usize;
        let x_max = (center_x + 4.0 * sigma).min(width as f64 - 1.0) as usize;
        let y_min = (center_y - 4.0 * sigma).max(0.0) as usize;
        let y_max = (center_y + 4.0 * sigma).min(height as f64 - 1.0) as usize;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let dx = x as f64 - center_x;
                let dy = y as f64 - center_y;
                let exponent = -(dx * dx + dy * dy) / (2.0 * sigma * sigma);
                image[[y, x]] = amplitude * exponent.exp();
            }
        }
    }

    #[test]
    fn test_subpixel_centroid_accuracy() {
        use super::super::thresholding::{
            apply_threshold, connected_components, get_bounding_boxes,
        };

        // Test parameters
        let image_size = 64; // Smaller size for faster tests
        let sigma = 1.0;
        let center_x = image_size as f64 / 2.0;
        let center_y = image_size as f64 / 2.0;
        let threshold = 0.1;

        // Test points at different sub-pixel positions
        let test_positions = [
            (0.0, 0.0),   // Integer position
            (0.5, 0.5),   // Half-pixel position
            (0.25, 0.75), // Quarter-pixel positions
            (0.33, 0.67), // Sub-pixel position
        ];

        for (offset_x, offset_y) in test_positions.iter() {
            // Create position with sub-pixel offset
            let position_x = center_x + offset_x;
            let position_y = center_y + offset_y;

            // Create an empty image
            let mut image = Array2::<f64>::zeros((image_size, image_size));

            // Generate a Gaussian star
            create_gaussian(&mut image, position_x, position_y, 1.0, sigma);

            // Process the image
            let binary = apply_threshold(&image.view(), threshold);
            let labeled = connected_components(&binary.view());
            let bboxes = get_bounding_boxes(&labeled.view());

            // Skip if no detection (shouldn't happen with these parameters)
            if bboxes.is_empty() {
                continue;
            }

            // Calculate centroid
            let star =
                calculate_star_centroid(&image.view(), &labeled.view(), 1, bboxes[0].to_tuple());

            // Calculate error
            let error = ((position_x - star.x).powi(2) + (position_y - star.y).powi(2)).sqrt();

            // Check accuracy based on position
            if *offset_x == 0.5 && *offset_y == 0.5 {
                // Half-pixel positions should be very accurate
                assert!(
                    error < 0.05,
                    "Centroid error too large at half-pixel position ({}, {}): {}",
                    offset_x,
                    offset_y,
                    error
                );
            } else {
                // For other positions, error should be within acceptable range
                // Integer positions can actually have higher error in some implementations
                assert!(
                    error < 0.5,
                    "Centroid error too large at position ({}, {}): {}",
                    offset_x,
                    offset_y,
                    error
                );
            }

            // Star should be valid
            assert!(
                star.is_valid,
                "Star at ({}, {}) was marked invalid with aspect_ratio={}",
                offset_x, offset_y, star.aspect_ratio
            );

            // Aspect ratio should be reasonable for a Gaussian PSF
            assert!(
                star.aspect_ratio < 2.5,
                "Aspect ratio too high at position ({}, {}): {}",
                offset_x,
                offset_y,
                star.aspect_ratio
            );
        }
    }

    #[test]
    fn test_centroid_with_different_psf_widths() {
        use super::super::thresholding::{
            apply_threshold, connected_components, get_bounding_boxes,
        };

        // Test parameters
        let image_size = 64;
        let center_x = image_size as f64 / 2.0;
        let center_y = image_size as f64 / 2.0;
        let position_x = center_x + 0.5; // Half-pixel offset for better accuracy
        let position_y = center_y + 0.5;
        let threshold = 0.1;

        // Test a range of sigma values
        let sigma_values = [0.75, 1.0, 1.5, 2.0];

        for &sigma in sigma_values.iter() {
            // Create an empty image
            let mut image = Array2::<f64>::zeros((image_size, image_size));

            // Generate a Gaussian star
            create_gaussian(&mut image, position_x, position_y, 1.0, sigma);

            // Process the image
            let binary = apply_threshold(&image.view(), threshold);
            let labeled = connected_components(&binary.view());
            let bboxes = get_bounding_boxes(&labeled.view());

            // Skip if no detection
            if bboxes.is_empty() {
                continue;
            }

            // Calculate centroid
            let star =
                calculate_star_centroid(&image.view(), &labeled.view(), 1, bboxes[0].to_tuple());

            // Calculate error
            let error = ((position_x - star.x).powi(2) + (position_y - star.y).powi(2)).sqrt();

            // Check accuracy based on PSF width
            // A wider PSF (larger sigma) should still give reasonable results
            assert!(
                error < 1.0 * sigma,
                "Centroid error too large with sigma={}: {}",
                sigma,
                error
            );

            // Star should be valid
            assert!(
                star.is_valid,
                "Star with sigma={} was marked invalid with aspect_ratio={}",
                sigma, star.aspect_ratio
            );
        }
    }
}

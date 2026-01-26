//! Naive star detection using center-of-mass centroiding and image moments.
//!
//! This module implements a simple but effective star detection algorithm based on
//! threshold segmentation followed by precise centroiding using image moments.
//! Provides sub-pixel accuracy and basic shape filtering to reject artifacts.
//!
//! # Algorithm Overview
//!
//! 1. **Threshold segmentation**: Binary threshold to identify bright regions
//! 2. **Connected components**: Group adjacent pixels into objects
//! 3. **Moment calculation**: Compute 0th, 1st, and 2nd order moments
//! 4. **Centroiding**: Calculate weighted center-of-mass positions
//! 5. **Shape analysis**: Use eigenvalues for aspect ratio and validity checking
//!
//! # Key Features
//!
//! - **Sub-pixel precision**: Moment-based centroids accurate to ~0.05 pixels
//! - **Shape filtering**: Aspect ratio analysis to reject elongated artifacts
//! - **Fast processing**: Simple operations suitable for real-time analysis
//! - **Robust moments**: Handles various PSF sizes and brightness levels
//! - **Otsu thresholding**: Automatic threshold selection when not specified
//!
//! # Usage
//!
//! Use detect_stars() for direct detection on f64 images.

use ndarray::{Array2, ArrayView2};
#[cfg(test)]
use std::collections::HashSet;

use crate::image_proc::centroid::{compute_centroid_from_mask, SpotShape};
use meter_math::Locatable2d;
use starfield::image::starfinders::StellarSource;

/// Star detection result with sub-pixel position and shape characterization.
///
/// Contains all information needed to characterize a detected stellar object,
/// including precise centroid position, flux measurement, and shape parameters
/// for quality assessment.
///
/// # Shape Analysis
///
/// The aspect ratio and diameter are computed from the eigenvalues of the
/// second moment matrix (covariance matrix), providing robust estimates of
/// object size and elongation.
///
/// # Validity Filtering
///
/// Objects are marked as valid stars if their aspect ratio is less than 2.5,
/// which helps reject cosmic rays, hot pixels, and other elongated artifacts.
///
/// # Usage
/// Contains detection results with sub-pixel centroid coordinates, total flux,
/// and shape parameters for quality assessment and filtering.
#[derive(Debug, Clone)]
pub struct StarDetection {
    /// Unique identifier for this detection (assigned sequentially)
    pub id: usize,
    /// Centroid x-coordinate with sub-pixel precision
    pub x: f64,
    /// Centroid y-coordinate with sub-pixel precision  
    pub y: f64,
    /// Total flux (sum of all pixel intensities in the object)
    pub flux: f64,
    /// Second central moment μ₂₀ (variance in x-direction)
    pub m_xx: f64,
    /// Second central moment μ₀₂ (variance in y-direction)
    pub m_yy: f64,
    /// Second central moment μ₁₁ (covariance between x and y)
    pub m_xy: f64,
    /// Aspect ratio (λ₁/λ₂) from eigenvalues of moment matrix
    pub aspect_ratio: f64,
    /// Estimated object diameter in pixels (4√(λ₁+λ₂)/2)
    pub diameter: f64,
}

impl StarDetection {
    /// Check if detection is likely to be a valid star based on aspect ratio
    /// Stars should have aspect ratio close to 1.0 (circular PSF)
    /// Returns true if aspect_ratio < 2.5, which allows for some PSF distortion
    pub fn is_valid(&self) -> bool {
        self.aspect_ratio < 2.5
    }

    /// Extract shape characterization without position.
    ///
    /// Use this when passing shape data through a pipeline where position
    /// is tracked separately.
    pub fn to_shape(&self) -> SpotShape {
        SpotShape {
            flux: self.flux,
            m_xx: self.m_xx,
            m_yy: self.m_yy,
            m_xy: self.m_xy,
            aspect_ratio: self.aspect_ratio,
            diameter: self.diameter,
        }
    }
}

impl Locatable2d for StarDetection {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }
}

impl StellarSource for StarDetection {
    fn id(&self) -> usize {
        self.id
    }

    fn get_centroid(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    fn flux(&self) -> f64 {
        self.flux
    }
}

/// Debug function to validate centroid calculation arguments
#[inline]
fn validate_centroid_args(
    image: &ArrayView2<f64>,
    labeled: &ArrayView2<usize>,
    bbox: (usize, usize, usize, usize),
) {
    let (min_row, min_col, max_row, max_col) = bbox;

    debug_assert!(
        min_row < image.nrows(),
        "bbox min_row {} is out of image bounds (nrows: {})",
        min_row,
        image.nrows()
    );
    debug_assert!(
        max_row < image.nrows(),
        "bbox max_row {} is out of image bounds (nrows: {})",
        max_row,
        image.nrows()
    );
    debug_assert!(
        min_col < image.ncols(),
        "bbox min_col {} is out of image bounds (ncols: {})",
        min_col,
        image.ncols()
    );
    debug_assert!(
        max_col < image.ncols(),
        "bbox max_col {} is out of image bounds (ncols: {})",
        max_col,
        image.ncols()
    );
    debug_assert!(
        min_row <= max_row,
        "bbox min_row {min_row} must be <= max_row {max_row}"
    );
    debug_assert!(
        min_col <= max_col,
        "bbox min_col {min_col} must be <= max_col {max_col}"
    );
    debug_assert_eq!(
        image.shape(),
        labeled.shape(),
        "image shape {:?} does not match labeled shape {:?}",
        image.shape(),
        labeled.shape()
    );
}

/// Calculate precise centroid and shape moments for a labeled object.
///
/// Computes weighted center-of-mass and second-order moments for accurate
/// sub-pixel centroid determination and shape characterization. Uses intensity
/// weighting to handle realistic PSF profiles.
///
/// # Mathematical Background
///
/// Computes raw moments:
/// - m₀₀ = Σ I(x,y) (total intensity)
/// - m₁₀ = Σ x·I(x,y), m₀₁ = Σ y·I(x,y) (first moments)
/// - m₂₀ = Σ x²·I(x,y), m₀₂ = Σ y²·I(x,y), m₁₁ = Σ xy·I(x,y) (second moments)
///
/// Then calculates central moments relative to centroid and eigenvalues
/// of the covariance matrix for shape analysis.
///
/// # Arguments
/// * `image` - Original grayscale image with intensity values
/// * `labeled` - Connected component labels from segmentation
/// * `label` - Specific label ID to process (labels start from 1)
/// * `bbox` - Bounding box (min_row, min_col, max_row, max_col) for efficiency
/// * `id` - Unique identifier to assign to this detection
///
/// # Returns
/// Complete StarDetection with centroid, flux, moments, and validity assessment
pub fn calculate_star_centroid(
    image: &ArrayView2<f64>,
    labeled: &ArrayView2<usize>,
    label: usize,
    bbox: (usize, usize, usize, usize),
    id: usize,
) -> StarDetection {
    // Validate arguments in debug builds
    validate_centroid_args(image, labeled, bbox);

    let (min_row, min_col, max_row, max_col) = bbox;

    // Extract sub-image and create mask
    let height = max_row - min_row + 1;
    let width = max_col - min_col + 1;

    // Create views for the AABB region
    let sub_image = image.slice(ndarray::s![min_row..=max_row, min_col..=max_col]);

    // Create binary mask for the label
    let mask = Array2::from_shape_fn((height, width), |(row, col)| {
        labeled[[min_row + row, min_col + col]] == label
    });

    // Compute centroid using the new function
    let centroid_result = compute_centroid_from_mask(&sub_image, &mask.view());

    // Convert relative coordinates to absolute image coordinates
    StarDetection {
        id,
        x: centroid_result.x + min_col as f64,
        y: centroid_result.y + min_row as f64,
        flux: centroid_result.flux,
        m_xx: centroid_result.m_xx,
        m_yy: centroid_result.m_yy,
        m_xy: centroid_result.m_xy,
        aspect_ratio: centroid_result.aspect_ratio,
        diameter: centroid_result.diameter,
    }
}

/// Detect stars in an image using threshold segmentation and centroiding.
///
/// Complete star detection pipeline that segments bright objects, performs
/// connected component analysis, and calculates precise centroids with
/// shape filtering to reject non-stellar objects.
///
/// # Algorithm Steps
/// 1. Apply threshold (Otsu automatic if not specified)
/// 2. Find connected components in binary image
/// 3. Calculate bounding boxes for efficiency
/// 4. Compute moments and centroids for each component
/// 5. Filter results based on aspect ratio (< 2.5)
///
/// # Arguments
/// * `image` - Input astronomical image as f64 array
/// * `threshold` - Optional intensity threshold (None = Otsu automatic)
///
/// # Returns
/// Vector of valid StarDetection objects with sub-pixel centroids
///
/// # Usage
/// Core detection function using threshold segmentation and moment analysis.
/// Returns StarDetection objects with sub-pixel centroid precision.
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
        let id = i; // Use index as ID
        let star = calculate_star_centroid(image, &labeled.view(), label, bbox.to_tuple(), id);
        stars.push(star);
    }

    // Filter out non-star objects
    stars.into_iter().filter(|star| star.is_valid()).collect()
}

/// Extract centroid positions from star detections.
///
/// Convenience function to get just the (x, y) coordinates from
/// StarDetection objects for algorithms that only need positions.
///
/// # Arguments
/// * `stars` - Vector of detected stars with complete information
///
/// # Returns
/// Vector of (x, y) centroid positions with sub-pixel precision
///
/// # Usage
/// Extracts position coordinates from star detections for algorithms
/// that only need centroid locations without shape information.
pub fn get_centroids(stars: &[StarDetection]) -> Vec<(f64, f64)> {
    stars.iter().map(|star| (star.x, star.y)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
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

        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox, 0);

        // Star centroid should be at (2.0, 2.0)
        assert_relative_eq!(star.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(star.y, 2.0, epsilon = 1e-10);

        // Star should be valid (circular)
        assert!(star.is_valid());

        // Moments should indicate a circular object
        assert_relative_eq!(star.aspect_ratio, 1.0, epsilon = 0.1);
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
        use crate::image_proc::detection::thresholding::{
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
                    0,
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
            "Average centroid error too large: {avg_error}"
        );

        assert!(
            max_error < 0.1,
            "Maximum centroid error too large: {max_error}"
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
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox, 0);

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
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox, 0);

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
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox, 0);

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
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox, 0);

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
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox, 0);

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
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox, 0);

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
        let star_normal = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox, 0);

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
            0,
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
            "X bias ({x_bias}) should match Y bias ({y_bias})"
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
        use crate::image_proc::detection::thresholding::{
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
                calculate_star_centroid(&image.view(), &labeled.view(), 1, bboxes[0].to_tuple(), 0);

            // Calculate error
            let error = ((position_x - star.x).powi(2) + (position_y - star.y).powi(2)).sqrt();

            // Check accuracy based on position
            if *offset_x == 0.5 && *offset_y == 0.5 {
                // Half-pixel positions should be very accurate
                assert!(
                    error < 0.05,
                    "Centroid error too large at half-pixel position ({offset_x}, {offset_y}): {error}"
                );
            } else {
                // For other positions, error should be within acceptable range
                // Integer positions can actually have higher error in some implementations
                assert!(
                    error < 0.5,
                    "Centroid error too large at position ({offset_x}, {offset_y}): {error}"
                );
            }

            // Star should be valid
            assert!(
                star.is_valid(),
                "Star at ({}, {}) was marked invalid with aspect_ratio={}",
                offset_x,
                offset_y,
                star.aspect_ratio
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
        use crate::image_proc::detection::thresholding::{
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
                calculate_star_centroid(&image.view(), &labeled.view(), 1, bboxes[0].to_tuple(), 0);

            // Calculate error
            let error = ((position_x - star.x).powi(2) + (position_y - star.y).powi(2)).sqrt();

            // Check accuracy based on PSF width
            // A wider PSF (larger sigma) should still give reasonable results
            assert!(
                error < 1.0 * sigma,
                "Centroid error too large with sigma={sigma}: {error}"
            );

            // Star should be valid
            assert!(
                star.is_valid(),
                "Star with sigma={} was marked invalid with aspect_ratio={}",
                sigma,
                star.aspect_ratio
            );
        }
    }

    #[test]
    fn test_unique_ids() {
        // Create an image with multiple detectable star patterns
        let mut image = Array2::<f64>::zeros((15, 15));

        // Add three separate star patterns with Gaussian-like profiles
        // Star 1: centered at (3, 3)
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let x = 3 + dx;
                let y = 3 + dy;
                if x >= 0 && y >= 0 && x < 15 && y < 15 {
                    let distance = ((dx * dx + dy * dy) as f64).sqrt();
                    image[[y as usize, x as usize]] = 10.0 * (-distance / 0.5).exp();
                }
            }
        }

        // Star 2: centered at (3, 11)
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let x = 11 + dx;
                let y = 3 + dy;
                if x >= 0 && y >= 0 && x < 15 && y < 15 {
                    let distance = ((dx * dx + dy * dy) as f64).sqrt();
                    image[[y as usize, x as usize]] = 10.0 * (-distance / 0.5).exp();
                }
            }
        }

        // Star 3: centered at (11, 3)
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let x = 3 + dx;
                let y = 11 + dy;
                if x >= 0 && y >= 0 && x < 15 && y < 15 {
                    let distance = ((dx * dx + dy * dy) as f64).sqrt();
                    image[[y as usize, x as usize]] = 10.0 * (-distance / 0.5).exp();
                }
            }
        }

        // Detect stars using the main function with low threshold
        let stars = detect_stars(&image.view(), Some(1.0));

        // Verify that all IDs are unique
        let mut ids = HashSet::new();
        for star in &stars {
            assert!(ids.insert(star.id), "Duplicate ID found: {}", star.id);
        }

        // Verify we have at least the expected number of stars
        assert!(
            stars.len() >= 3,
            "Expected at least 3 stars, found {}",
            stars.len()
        );

        // Verify IDs are consecutive starting from 0 (for the first N stars)
        let mut sorted_ids: Vec<usize> = stars.iter().map(|s| s.id).collect();
        sorted_ids.sort();
        for (i, &id) in sorted_ids.iter().enumerate() {
            assert_eq!(id, i, "Expected ID {i}, found {id} at position {i}");
        }
    }
}

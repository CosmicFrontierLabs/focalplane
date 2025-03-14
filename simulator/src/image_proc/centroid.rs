//! Star centroid calculation using center of mass and image moments
//!
//! This module provides functionality for calculating star centroids with
//! sub-pixel accuracy and filtering non-star objects based on moment analysis.

use ndarray::ArrayView2;

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
        let star = calculate_star_centroid(image, &labeled.view(), label, *bbox);
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
}

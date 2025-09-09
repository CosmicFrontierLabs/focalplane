//! Centroid calculation methods for astronomical image processing
//!
//! This module provides core centroiding algorithms for computing precise
//! sub-pixel positions of stellar objects from image data and masks.

use ndarray::ArrayView2;

/// Result from centroid calculation containing position and shape properties
///
/// Contains the computed centroid position relative to the input sub-image,
/// along with flux measurements and shape characterization parameters.
#[derive(Debug, Clone)]
pub struct CentroidResult {
    /// Centroid x-coordinate relative to sub-image origin
    pub x: f64,
    /// Centroid y-coordinate relative to sub-image origin
    pub y: f64,
    /// Total flux (sum of all pixel intensities in mask)
    pub flux: f64,
    /// Second central moment μ₂₀ (variance in x-direction)
    pub m_xx: f64,
    /// Second central moment μ₀₂ (variance in y-direction)
    pub m_yy: f64,
    /// Second central moment μ₁₁ (covariance between x and y)
    pub m_xy: f64,
    /// Aspect ratio (λ₁/λ₂) from eigenvalues of moment matrix
    pub aspect_ratio: f64,
    /// Estimated object diameter in pixels
    pub diameter: f64,
}

/// Calculate centroid and shape moments from image data and binary mask
///
/// Computes intensity-weighted center-of-mass and second-order moments
/// for accurate sub-pixel position determination and shape characterization.
///
/// # Arguments
///
/// * `image` - Sub-image containing the object (AABB size)
/// * `mask` - Binary mask (same size as image) with 1.0 where pixels belong to object
///
/// # Returns
///
/// CentroidResult with position relative to sub-image origin and shape parameters
pub fn compute_centroid_from_mask(
    image: &ArrayView2<f64>,
    mask: &ArrayView2<f64>,
) -> CentroidResult {
    assert_eq!(
        image.shape(),
        mask.shape(),
        "Image and mask must have same dimensions"
    );

    // Initialize moments
    let mut m00 = 0.0; // Total mass/intensity
    let mut m10 = 0.0; // First moment in x
    let mut m01 = 0.0; // First moment in y
    let mut m20 = 0.0; // Second moment in x
    let mut m02 = 0.0; // Second moment in y
    let mut m11 = 0.0; // Cross moment

    // Calculate raw moments
    for ((row, col), &mask_val) in mask.indexed_iter() {
        if mask_val > 0.5 {
            // Binary threshold for mask
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

    // Handle empty or zero-flux objects
    if m00 < f64::EPSILON {
        return CentroidResult {
            x: 0.0,
            y: 0.0,
            flux: 0.0,
            m_xx: 0.0,
            m_yy: 0.0,
            m_xy: 0.0,
            aspect_ratio: f64::INFINITY,
            diameter: 0.0,
        };
    }

    // Calculate centroid (relative to sub-image origin)
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

    // Calculate diameter using average of eigenvalues
    let diameter = 4.0 * ((lambda1 + lambda2) / 2.0).sqrt();

    CentroidResult {
        x: x_centroid,
        y: y_centroid,
        flux: m00,
        m_xx: mu20,
        m_yy: mu02,
        m_xy: mu11,
        aspect_ratio,
        diameter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_centroid_single_pixel() {
        let image = Array2::from_elem((3, 3), 0.0);
        let mut mask = Array2::from_elem((3, 3), 0.0);
        let mut image_mut = image.clone();

        image_mut[[1, 1]] = 100.0;
        mask[[1, 1]] = 1.0;

        let result = compute_centroid_from_mask(&image_mut.view(), &mask.view());

        assert!((result.x - 1.0).abs() < 1e-10);
        assert!((result.y - 1.0).abs() < 1e-10);
        assert!((result.flux - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_symmetric_pattern() {
        let mut image = Array2::from_elem((5, 5), 0.0);
        let mut mask = Array2::from_elem((5, 5), 0.0);

        // Create symmetric cross pattern
        image[[2, 1]] = 50.0;
        mask[[2, 1]] = 1.0;
        image[[2, 2]] = 100.0;
        mask[[2, 2]] = 1.0;
        image[[2, 3]] = 50.0;
        mask[[2, 3]] = 1.0;
        image[[1, 2]] = 50.0;
        mask[[1, 2]] = 1.0;
        image[[3, 2]] = 50.0;
        mask[[3, 2]] = 1.0;

        let result = compute_centroid_from_mask(&image.view(), &mask.view());

        assert!((result.x - 2.0).abs() < 1e-10);
        assert!((result.y - 2.0).abs() < 1e-10);
        assert!((result.flux - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_moments_and_aspect_ratio() {
        let mut image = Array2::from_elem((7, 7), 0.0);
        let mut mask = Array2::from_elem((7, 7), 0.0);

        // Create elongated horizontal pattern (should have aspect ratio > 1)
        // Pattern centered at (3, 3) with more spread in x than y
        image[[3, 1]] = 25.0;
        mask[[3, 1]] = 1.0;
        image[[3, 2]] = 50.0;
        mask[[3, 2]] = 1.0;
        image[[3, 3]] = 100.0;
        mask[[3, 3]] = 1.0;
        image[[3, 4]] = 50.0;
        mask[[3, 4]] = 1.0;
        image[[3, 5]] = 25.0;
        mask[[3, 5]] = 1.0;

        // Add slight vertical component to make it 2D
        image[[2, 3]] = 25.0;
        mask[[2, 3]] = 1.0;
        image[[4, 3]] = 25.0;
        mask[[4, 3]] = 1.0;

        let result = compute_centroid_from_mask(&image.view(), &mask.view());

        // Check centroid is at center
        assert!(
            (result.x - 3.0).abs() < 1e-10,
            "x centroid should be 3.0, got {}",
            result.x
        );
        assert!(
            (result.y - 3.0).abs() < 1e-10,
            "y centroid should be 3.0, got {}",
            result.y
        );
        assert!(
            (result.flux - 300.0).abs() < 1e-10,
            "flux should be 300.0, got {}",
            result.flux
        );

        // Check that m_xx > m_yy (more spread in x direction)
        assert!(
            result.m_xx > result.m_yy,
            "m_xx ({}) should be > m_yy ({}) for horizontal pattern",
            result.m_xx,
            result.m_yy
        );

        // Check aspect ratio > 1 for elongated pattern
        assert!(
            result.aspect_ratio > 1.5,
            "aspect ratio should be > 1.5 for elongated pattern, got {}",
            result.aspect_ratio
        );
        assert!(
            result.aspect_ratio < 10.0,
            "aspect ratio should be reasonable (< 10), got {}",
            result.aspect_ratio
        );

        // Check m_xy is near zero for aligned pattern
        assert!(
            result.m_xy.abs() < 0.1,
            "m_xy should be near zero for axis-aligned pattern, got {}",
            result.m_xy
        );
    }

    #[test]
    fn test_circular_pattern_aspect_ratio() {
        let mut image = Array2::from_elem((7, 7), 0.0);
        let mut mask = Array2::from_elem((7, 7), 0.0);

        // Create roughly circular Gaussian-like pattern
        // Should have aspect ratio close to 1.0
        let center = 3;
        let sigma = 1.0;

        for i in 1..6 {
            for j in 1..6 {
                let dist_sq =
                    ((i as f64 - center as f64).powi(2) + (j as f64 - center as f64).powi(2));
                let intensity = 100.0 * (-dist_sq / (2.0 * sigma * sigma)).exp();
                if intensity > 10.0 {
                    // Threshold to create mask
                    image[[i, j]] = intensity;
                    mask[[i, j]] = 1.0;
                }
            }
        }

        let result = compute_centroid_from_mask(&image.view(), &mask.view());

        // For circular pattern, aspect ratio should be close to 1.0
        assert!(
            result.aspect_ratio > 0.9 && result.aspect_ratio < 1.1,
            "aspect ratio should be close to 1.0 for circular pattern, got {}",
            result.aspect_ratio
        );

        // Moments should be roughly equal for circular pattern
        let moment_ratio = result.m_xx / result.m_yy;
        assert!(
            moment_ratio > 0.8 && moment_ratio < 1.2,
            "m_xx/m_yy should be close to 1.0 for circular pattern, got {}",
            moment_ratio
        );
    }

    #[test]
    fn test_diagonal_pattern_moments() {
        let mut image = Array2::from_elem((7, 7), 0.0);
        let mut mask = Array2::from_elem((7, 7), 0.0);

        // Create diagonal pattern - should have non-zero m_xy
        for i in 1..6 {
            image[[i, i]] = 100.0;
            mask[[i, i]] = 1.0;
        }

        let result = compute_centroid_from_mask(&image.view(), &mask.view());

        // For diagonal pattern, m_xy should be positive and significant
        assert!(
            result.m_xy > 0.5,
            "m_xy should be positive for diagonal pattern, got {}",
            result.m_xy
        );

        // Moments in x and y should be equal for 45-degree diagonal
        let moment_ratio = result.m_xx / result.m_yy;
        assert!(
            (moment_ratio - 1.0).abs() < 0.1,
            "m_xx/m_yy should be close to 1.0 for diagonal pattern, got {}",
            moment_ratio
        );
    }
}

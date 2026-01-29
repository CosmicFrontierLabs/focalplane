//! Centroid calculation methods for astronomical image processing
//!
//! This module provides core centroiding algorithms for computing precise
//! sub-pixel positions of stellar objects from image data and masks.

use ndarray::ArrayView2;

use serde::{Deserialize, Serialize};

/// Maximum intensity value for 16-bit unsigned images (2^16 - 1)
pub const SATURATION_16BIT: f64 = 65535.0;

/// Spot shape characterization without position.
///
/// Contains flux, shape moments, and size measurements extracted from a centroid
/// calculation. Used for transmitting shape data separately from position
/// (e.g., in tracking messages where frame-relative position is stored separately).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotShape {
    /// Total flux (sum of all pixel intensities)
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

/// Result from centroid calculation containing position and shape properties
///
/// Contains the computed centroid position relative to the input sub-image,
/// along with flux measurements and shape characterization parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Saturation tracking: (cutoff value, count of pixels above cutoff)
    pub n_saturated: (f64, u32),
}

impl CentroidResult {
    /// Extract shape characterization without position.
    ///
    /// Use this when passing shape data through a pipeline where position
    /// is tracked separately (e.g., frame-relative coordinates).
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

/// Calculate centroid and shape moments from image data and binary mask
///
/// Computes intensity-weighted center-of-mass and second-order moments
/// for accurate sub-pixel position determination and shape characterization.
/// Uses a default saturation cutoff of 65535.0 (16-bit max).
///
/// # Arguments
///
/// * `image` - Sub-image containing the object (AABB size)
/// * `mask` - Binary mask (same size as image) with true where pixels belong to object
///
/// # Returns
///
/// CentroidResult with position relative to sub-image origin and shape parameters
pub fn compute_centroid_from_mask(
    image: &ArrayView2<f64>,
    mask: &ArrayView2<bool>,
) -> CentroidResult {
    compute_centroid_from_mask_with_saturation(image, mask, SATURATION_16BIT)
}

/// Calculate centroid and shape moments with custom saturation threshold
///
/// Computes intensity-weighted center-of-mass and second-order moments
/// for accurate sub-pixel position determination and shape characterization,
/// while counting pixels above the specified saturation threshold.
///
/// # Arguments
///
/// * `image` - Sub-image containing the object (AABB size)
/// * `mask` - Binary mask (same size as image) with true where pixels belong to object
/// * `saturation_cutoff` - Intensity threshold for counting saturated pixels
///
/// # Returns
///
/// CentroidResult with position, shape parameters, and saturation statistics
pub fn compute_centroid_from_mask_with_saturation(
    image: &ArrayView2<f64>,
    mask: &ArrayView2<bool>,
    saturation_cutoff: f64,
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
    let mut n_saturated: u32 = 0; // Count of saturated pixels

    // Calculate raw moments
    for ((row, col), &mask_val) in mask.indexed_iter() {
        if mask_val {
            let intensity = image[[row, col]];

            // Check for saturation
            if intensity > saturation_cutoff {
                n_saturated += 1;
            }

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
            n_saturated: (saturation_cutoff, n_saturated),
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
        n_saturated: (saturation_cutoff, n_saturated),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{abs_diff_eq, assert_relative_eq};
    use ndarray::Array2;

    #[test]
    fn test_centroid_single_pixel() {
        let image = Array2::from_elem((3, 3), 0.0);
        let mut mask = Array2::from_elem((3, 3), false);
        let mut image_mut = image.clone();

        image_mut[[1, 1]] = 100.0;
        mask[[1, 1]] = true;

        let result = compute_centroid_from_mask(&image_mut.view(), &mask.view());

        assert_relative_eq!(result.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.flux, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_centroid_symmetric_pattern() {
        let mut image = Array2::from_elem((5, 5), 0.0);
        let mut mask = Array2::from_elem((5, 5), false);

        // Create symmetric cross pattern
        image[[2, 1]] = 50.0;
        mask[[2, 1]] = true;
        image[[2, 2]] = 100.0;
        mask[[2, 2]] = true;
        image[[2, 3]] = 50.0;
        mask[[2, 3]] = true;
        image[[1, 2]] = 50.0;
        mask[[1, 2]] = true;
        image[[3, 2]] = 50.0;
        mask[[3, 2]] = true;

        let result = compute_centroid_from_mask(&image.view(), &mask.view());

        assert_relative_eq!(result.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.flux, 300.0, epsilon = 1e-10);
    }

    #[test]
    fn test_moments_and_aspect_ratio() {
        let mut image = Array2::from_elem((7, 7), 0.0);
        let mut mask = Array2::from_elem((7, 7), false);

        // Create elongated horizontal pattern (should have aspect ratio > 1)
        // Pattern centered at (3, 3) with more spread in x than y
        image[[3, 1]] = 25.0;
        mask[[3, 1]] = true;
        image[[3, 2]] = 50.0;
        mask[[3, 2]] = true;
        image[[3, 3]] = 100.0;
        mask[[3, 3]] = true;
        image[[3, 4]] = 50.0;
        mask[[3, 4]] = true;
        image[[3, 5]] = 25.0;
        mask[[3, 5]] = true;

        // Add slight vertical component to make it 2D
        image[[2, 3]] = 25.0;
        mask[[2, 3]] = true;
        image[[4, 3]] = 25.0;
        mask[[4, 3]] = true;

        let result = compute_centroid_from_mask(&image.view(), &mask.view());

        // Check centroid is at center
        assert!(
            abs_diff_eq!(result.x, 3.0, epsilon = 1e-10),
            "x centroid should be 3.0, got {}",
            result.x
        );
        assert!(
            abs_diff_eq!(result.y, 3.0, epsilon = 1e-10),
            "y centroid should be 3.0, got {}",
            result.y
        );
        assert!(
            abs_diff_eq!(result.flux, 300.0, epsilon = 1e-10),
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
            abs_diff_eq!(result.m_xy, 0.0, epsilon = 0.1),
            "m_xy should be near zero for axis-aligned pattern, got {}",
            result.m_xy
        );
    }

    #[test]
    fn test_circular_pattern_aspect_ratio() {
        let mut image = Array2::from_elem((7, 7), 0.0);
        let mut mask = Array2::from_elem((7, 7), false);

        // Create roughly circular Gaussian-like pattern
        // Should have aspect ratio close to 1.0
        let center = 3;
        let sigma = 1.0;

        for i in 1..6 {
            for j in 1..6 {
                let dist_sq =
                    (i as f64 - center as f64).powi(2) + (j as f64 - center as f64).powi(2);
                let intensity = 100.0 * (-dist_sq / (2.0 * sigma * sigma)).exp();
                if intensity > 10.0 {
                    // Threshold to create mask
                    image[[i, j]] = intensity;
                    mask[[i, j]] = true;
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
            "m_xx/m_yy should be close to 1.0 for circular pattern, got {moment_ratio}"
        );
    }

    #[test]
    fn test_diagonal_pattern_moments() {
        let mut image = Array2::from_elem((7, 7), 0.0);
        let mut mask = Array2::from_elem((7, 7), false);

        // Create diagonal pattern - should have non-zero m_xy
        for i in 1..6 {
            image[[i, i]] = 100.0;
            mask[[i, i]] = true;
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
            abs_diff_eq!(moment_ratio, 1.0, epsilon = 0.1),
            "m_xx/m_yy should be close to 1.0 for diagonal pattern, got {moment_ratio}"
        );
    }

    #[test]
    fn test_saturation_counting() {
        let mut image = Array2::from_elem((5, 5), 0.0);
        let mut mask = Array2::from_elem((5, 5), false);

        // Create pattern with mix of saturated and non-saturated pixels
        image[[2, 1]] = 50.0; // Below cutoff
        mask[[2, 1]] = true;
        image[[2, 2]] = 100.0; // At cutoff
        mask[[2, 2]] = true;
        image[[2, 3]] = 150.0; // Above cutoff
        mask[[2, 3]] = true;
        image[[1, 2]] = 200.0; // Above cutoff
        mask[[1, 2]] = true;
        image[[3, 2]] = 75.0; // Below cutoff
        mask[[3, 2]] = true;

        // Test with cutoff at 100.0
        let result = compute_centroid_from_mask_with_saturation(&image.view(), &mask.view(), 100.0);

        // Should have 2 saturated pixels (150.0 and 200.0)
        assert_eq!(result.n_saturated.0, 100.0, "Cutoff should be 100.0");
        assert_eq!(result.n_saturated.1, 2, "Should have 2 saturated pixels");
        assert!(
            abs_diff_eq!(result.flux, 575.0, epsilon = 1e-10),
            "Total flux should be 575.0"
        );

        // Test with higher cutoff
        let result_high =
            compute_centroid_from_mask_with_saturation(&image.view(), &mask.view(), 180.0);
        assert_eq!(result_high.n_saturated.0, 180.0, "Cutoff should be 180.0");
        assert_eq!(
            result_high.n_saturated.1, 1,
            "Should have 1 saturated pixel"
        );

        // Test with low cutoff (all saturated)
        let result_low =
            compute_centroid_from_mask_with_saturation(&image.view(), &mask.view(), 40.0);
        assert_eq!(result_low.n_saturated.0, 40.0, "Cutoff should be 40.0");
        assert_eq!(
            result_low.n_saturated.1, 5,
            "All 5 pixels should be saturated"
        );
    }

    #[test]
    fn test_default_saturation_cutoff() {
        let mut image = Array2::from_elem((3, 3), 0.0);
        let mut mask = Array2::from_elem((3, 3), false);

        // Create a very bright pixel
        image[[1, 1]] = 70000.0; // Above default 16-bit max (65535)
        mask[[1, 1]] = true;
        image[[1, 2]] = 30000.0; // Below default cutoff
        mask[[1, 2]] = true;

        // Using default function should use SATURATION_16BIT as cutoff
        let result = compute_centroid_from_mask(&image.view(), &mask.view());

        assert_eq!(
            result.n_saturated.0, SATURATION_16BIT,
            "Default cutoff should be SATURATION_16BIT"
        );
        assert_eq!(
            result.n_saturated.1, 1,
            "Should have 1 saturated pixel above 65535"
        );
    }

    #[test]
    fn benchmark_centroid_tracking_parameters() {
        use std::time::Instant;

        // Typical tracking parameters from monocle FGS
        const ROI_SIZE: usize = 32;
        const FWHM: f64 = 4.0;
        const CENTROID_RADIUS_MULTIPLIER: f64 = 3.0;
        const ITERATIONS: usize = 10000;

        // Calculate typical mask radius
        let mask_radius = FWHM * CENTROID_RADIUS_MULTIPLIER;

        // Create realistic synthetic star data for ROI
        let mut image = Array2::from_elem((ROI_SIZE, ROI_SIZE), 100.0); // Background
        let mut mask = Array2::from_elem((ROI_SIZE, ROI_SIZE), false);

        // Generate Gaussian star profile centered in ROI
        let center_x = ROI_SIZE as f64 / 2.0;
        let center_y = ROI_SIZE as f64 / 2.0;
        let peak_intensity = 8000.0;
        let sigma = FWHM / 2.355; // Convert FWHM to Gaussian sigma

        for row in 0..ROI_SIZE {
            for col in 0..ROI_SIZE {
                let dx = col as f64 - center_x;
                let dy = row as f64 - center_y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Add Gaussian intensity
                let gauss_intensity =
                    peak_intensity * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                image[[row, col]] += gauss_intensity;

                // Create circular mask based on centroid radius
                if dist <= mask_radius {
                    mask[[row, col]] = true;
                }
            }
        }

        // Warmup iterations
        for _ in 0..100 {
            let _ = compute_centroid_from_mask(&image.view(), &mask.view());
        }

        // Benchmark iterations with detailed timing
        let mut timings = Vec::with_capacity(ITERATIONS);

        for _ in 0..ITERATIONS {
            let start = Instant::now();
            let _result = compute_centroid_from_mask(&image.view(), &mask.view());
            let duration = start.elapsed();
            timings.push(duration);
        }

        // Calculate statistics
        timings.sort();
        let total_nanos: u128 = timings.iter().map(|d| d.as_nanos()).sum();
        let mean_nanos = total_nanos / ITERATIONS as u128;
        let median_nanos = timings[ITERATIONS / 2].as_nanos();
        let p95_nanos = timings[(ITERATIONS as f64 * 0.95) as usize].as_nanos();
        let p99_nanos = timings[(ITERATIONS as f64 * 0.99) as usize].as_nanos();
        let min_nanos = timings[0].as_nanos();
        let max_nanos = timings[ITERATIONS - 1].as_nanos();

        // Print results
        println!("\n========== CENTROID TIMING BENCHMARK ==========");
        println!("Configuration:");
        println!("  ROI Size: {}x{} pixels", ROI_SIZE, ROI_SIZE);
        println!("  FWHM: {:.1} pixels", FWHM);
        println!(
            "  Centroid Radius Multiplier: {:.1}x",
            CENTROID_RADIUS_MULTIPLIER
        );
        println!("  Mask Radius: {:.1} pixels", mask_radius);
        println!("  Iterations: {}", ITERATIONS);
        println!("\nTiming Results:");
        println!("  Mean:   {:>8.2} µs", mean_nanos as f64 / 1000.0);
        println!("  Median: {:>8.2} µs", median_nanos as f64 / 1000.0);
        println!("  Min:    {:>8.2} µs", min_nanos as f64 / 1000.0);
        println!("  Max:    {:>8.2} µs", max_nanos as f64 / 1000.0);
        println!("  P95:    {:>8.2} µs", p95_nanos as f64 / 1000.0);
        println!("  P99:    {:>8.2} µs", p99_nanos as f64 / 1000.0);
        println!("===============================================\n");

        // Verify result is reasonable
        let final_result = compute_centroid_from_mask(&image.view(), &mask.view());
        assert!(
            abs_diff_eq!(final_result.x, center_x, epsilon = 0.1),
            "Centroid x should be near center"
        );
        assert!(
            abs_diff_eq!(final_result.y, center_y, epsilon = 0.1),
            "Centroid y should be near center"
        );
        assert!(
            final_result.flux > peak_intensity,
            "Flux should be significant"
        );
    }
}

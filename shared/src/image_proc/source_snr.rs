//! Signal-to-noise ratio (SNR) calculations for detected sources.
//!
//! This module provides aperture photometry-based SNR calculations for astronomical
//! point sources detected in images. The SNR estimation uses a circular aperture
//! centered on the source and a background annulus to estimate local background
//! and noise characteristics.
//!
//! # Methodology
//!
//! The SNR calculation follows standard aperture photometry practices:
//!
//! 1. **Signal Measurement**: Sum of all pixel values within the aperture radius minus background contribution
//! 2. **Background Estimation**: Median of pixels in annulus (2-3× aperture radius)
//! 3. **Noise Estimation**: Median Absolute Deviation (MAD) in annulus, converted to RMS
//! 4. **SNR Calculation**: (aperture_sum - background_contribution) / noise_rms
//!
//! The use of MAD for noise estimation provides robustness against outliers from
//! cosmic rays or neighboring sources in the background annulus.

use ndarray::ArrayView2;
use thiserror::Error;

use super::aperture_photometry::collect_aperture_pixels;
use super::detection::StarDetection;
use meter_math::stats::median;

/// Errors from SNR calculations.
#[derive(Error, Debug)]
pub enum SnrError {
    /// Aperture contains no pixels at the given position.
    #[error("aperture contains no pixels at position ({x:.1}, {y:.1}) with radius {radius:.1}")]
    EmptyAperture {
        /// X coordinate of the source.
        x: f64,
        /// Y coordinate of the source.
        y: f64,
        /// Aperture radius used.
        radius: f64,
    },

    /// Insufficient background pixels for noise estimation.
    #[error(
        "insufficient background pixels ({count}) at position ({x:.1}, {y:.1}), need at least 10"
    )]
    InsufficientBackground {
        /// Number of background pixels found.
        count: usize,
        /// X coordinate of the source.
        x: f64,
        /// Y coordinate of the source.
        y: f64,
    },

    /// Statistical computation failed.
    #[error("stats computation failed: {0}")]
    StatsError(String),
}

/// Calculate signal-to-noise ratio at a specific position using aperture photometry.
///
/// This is the core SNR calculation function that takes coordinates directly.
/// Use this when you have position coordinates but no `StarDetection` struct.
///
/// # Arguments
///
/// * `x` - X coordinate (column) of the source centroid
/// * `y` - Y coordinate (row) of the source centroid
/// * `image` - The image array as f64 pixel values
/// * `aperture_radius` - Radius in pixels for the measurement aperture
/// * `background_inner_radius` - Inner radius of background annulus in pixels
/// * `background_outer_radius` - Outer radius of background annulus in pixels
///
/// # Returns
///
/// Result containing the signal-to-noise ratio as a positive f64 value.
///
/// # Errors
///
/// Returns [`SnrError`] if:
/// - Aperture contains no pixels ([`SnrError::EmptyAperture`])
/// - Background annulus contains fewer than 10 pixels ([`SnrError::InsufficientBackground`])
/// - Statistical computation fails ([`SnrError::StatsError`])
pub fn calculate_snr_at_position(
    x: f64,
    y: f64,
    image: &ArrayView2<f64>,
    aperture_radius: f64,
    background_inner_radius: f64,
    background_outer_radius: f64,
) -> Result<f64, SnrError> {
    let (aperture_pixels, background_pixels) = collect_aperture_pixels(
        image,
        x,
        y,
        aperture_radius,
        background_inner_radius,
        background_outer_radius,
    );

    if aperture_pixels.is_empty() {
        return Err(SnrError::EmptyAperture {
            x,
            y,
            radius: aperture_radius,
        });
    }

    if background_pixels.len() < 10 {
        return Err(SnrError::InsufficientBackground {
            count: background_pixels.len(),
            x,
            y,
        });
    }

    // Estimate local background level using median of background annulus pixels
    let background_median =
        median(&background_pixels).map_err(|e| SnrError::StatsError(e.to_string()))?;

    // Calculate signal: total flux in aperture minus background contribution
    let aperture_sum: f64 = aperture_pixels.iter().sum();
    let background_contribution = background_median * aperture_pixels.len() as f64;
    let signal = aperture_sum - background_contribution;

    // Calculate noise using Median Absolute Deviation (MAD) for robustness
    let deviations: Vec<f64> = background_pixels
        .iter()
        .map(|&val| (val - background_median).abs())
        .collect();

    let mad = median(&deviations).map_err(|e| SnrError::StatsError(e.to_string()))?;

    // Convert MAD to RMS noise estimate (σ ≈ 1.4826 × MAD for Gaussian noise)
    let noise_rms = 1.4826 * mad;

    if noise_rms <= 0.0 {
        log::warn!(
            "Zero or negative noise estimate (MAD={mad:.3}, RMS={noise_rms:.3}) at position ({x:.1}, {y:.1}), returning f64::MAX"
        );
        return Ok(f64::MAX);
    }

    Ok(signal / noise_rms)
}

/// Calculate signal-to-noise ratio for a detected source using aperture photometry.
///
/// This function performs aperture photometry to estimate the SNR of a point source.
/// It uses a circular aperture centered on the detection position to measure the signal,
/// and a background annulus to estimate local background and noise.
///
/// # Algorithm Details
///
/// - **Aperture**: Circular region of radius `aperture_radius` centered on detection
/// - **Background annulus**: Between `background_inner_radius` and `background_outer_radius`
/// - **Signal**: Sum of all aperture pixels minus background contribution (median × n_pixels)
/// - **Noise**: RMS estimated from MAD (σ ≈ 1.4826 × MAD for Gaussian noise)
///
/// # Arguments
///
/// * `detection` - The detected source with centroid position
/// * `image` - The image array as f64 pixel values
/// * `aperture_radius` - Radius in pixels for the measurement aperture
/// * `background_inner_radius` - Inner radius of background annulus in pixels
/// * `background_outer_radius` - Outer radius of background annulus in pixels
///
/// # Returns
///
/// Result containing the signal-to-noise ratio as a positive f64 value.
///
/// # Errors
///
/// Returns [`SnrError`] if:
/// - Aperture contains no pixels ([`SnrError::EmptyAperture`])
/// - Background annulus contains fewer than 10 pixels ([`SnrError::InsufficientBackground`])
/// - Statistical computation fails ([`SnrError::StatsError`])
///
/// # Performance Notes
///
/// The function allocates two vectors to collect aperture and background pixels.
/// For typical aperture radii (2-5 pixels), this involves analyzing 50-300 pixels.
pub fn calculate_snr(
    detection: &StarDetection,
    image: &ArrayView2<f64>,
    aperture_radius: f64,
    background_inner_radius: f64,
    background_outer_radius: f64,
) -> Result<f64, SnrError> {
    calculate_snr_at_position(
        detection.x,
        detection.y,
        image,
        aperture_radius,
        background_inner_radius,
        background_outer_radius,
    )
}

/// Filter a detection by minimum SNR threshold.
///
/// Convenience function that calculates SNR and compares against a threshold.
/// Returns true if the detection meets or exceeds the minimum SNR requirement.
/// Returns false if SNR calculation fails or SNR is below threshold.
///
/// # Arguments
///
/// * `detection` - The detected source to evaluate
/// * `image` - The image array as f64 pixel values
/// * `min_snr` - Minimum acceptable signal-to-noise ratio
/// * `aperture_radius` - Radius in pixels for the measurement aperture
/// * `background_inner_radius` - Inner radius of background annulus in pixels
/// * `background_outer_radius` - Outer radius of background annulus in pixels
///
/// # Returns
///
/// `true` if SNR >= min_snr, `false` if SNR < min_snr or calculation fails
///
/// # Example Use Cases
///
/// - Guide star selection: Require SNR > 10 for reliable centroiding
/// - Photometry quality control: Require SNR > 5 for trustworthy measurements
/// - Source validation: Filter out marginal detections with SNR < 3
pub fn filter_by_snr(
    detection: &StarDetection,
    image: &ArrayView2<f64>,
    min_snr: f64,
    aperture_radius: f64,
    background_inner_radius: f64,
    background_outer_radius: f64,
) -> bool {
    calculate_snr(
        detection,
        image,
        aperture_radius,
        background_inner_radius,
        background_outer_radius,
    )
    .map(|snr| snr >= min_snr)
    .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    fn make_test_detection(x: f64, y: f64, flux: f64) -> StarDetection {
        StarDetection {
            id: 0,
            x,
            y,
            flux,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 2.0,
        }
    }

    #[test]
    fn test_snr_bright_star_on_flat_background() {
        let mut image = Array2::<f64>::zeros((20, 20));

        for i in 0..20 {
            for j in 0..20 {
                let noise = ((i * 3 + j * 7) % 5) as f64 - 2.0;
                image[[i, j]] = 100.0 + noise;
            }
        }

        image[[10, 10]] = 1000.0;
        image[[9, 10]] = 800.0;
        image[[11, 10]] = 800.0;
        image[[10, 9]] = 800.0;
        image[[10, 11]] = 800.0;

        let detection = make_test_detection(10.0, 10.0, 4200.0);
        let snr = calculate_snr(&detection, &image.view(), 2.0, 4.0, 6.0)
            .expect("SNR calculation should succeed");

        assert!(snr > 10.0, "Expected high SNR for bright star, got {}", snr);
    }

    #[test]
    fn test_snr_zero_for_empty_aperture() {
        let mut image = Array2::<f64>::zeros((10, 10));
        for i in 0..10 {
            for j in 0..10 {
                image[[i, j]] = 1.0 + ((i + j) % 3) as f64 * 0.1;
            }
        }

        let detection = make_test_detection(5.0, 5.0, 0.0);

        let snr = calculate_snr(&detection, &image.view(), 2.0, 4.0, 6.0)
            .expect("SNR calculation should succeed");
        assert!(
            snr < 5.0,
            "SNR should be low for detection with no signal above background"
        );
    }

    #[test]
    fn test_snr_edge_case_detection() {
        let mut image = Array2::<f64>::zeros((20, 20));
        for i in 0..20 {
            for j in 0..20 {
                let noise = ((i * 5 + j * 3) % 7) as f64 - 3.0;
                image[[i, j]] = 50.0 + noise;
            }
        }

        image[[2, 2]] = 500.0;

        let detection = make_test_detection(2.0, 2.0, 500.0);
        let snr = calculate_snr(&detection, &image.view(), 1.5, 3.0, 4.5)
            .expect("SNR calculation should succeed");

        assert!(snr > 0.0, "SNR should be positive for detection near edge");
    }

    #[test]
    fn test_snr_with_background_gradient() {
        let mut image = Array2::<f64>::zeros((30, 30));

        for i in 0..30 {
            for j in 0..30 {
                image[[i, j]] = 100.0 + (i as f64) * 2.0;
            }
        }

        image[[15, 15]] = 1500.0;
        image[[14, 15]] = 1000.0;
        image[[16, 15]] = 1000.0;
        image[[15, 14]] = 1000.0;
        image[[15, 16]] = 1000.0;

        let detection = make_test_detection(15.0, 15.0, 5500.0);
        let snr = calculate_snr(&detection, &image.view(), 2.0, 4.0, 6.0)
            .expect("SNR calculation should succeed");

        assert!(
            snr > 5.0,
            "SNR should be reasonable even with background gradient, got {}",
            snr
        );
    }

    #[test]
    fn test_snr_max_for_noiseless_signal() {
        let mut image = Array2::<f64>::zeros((20, 20));

        for i in 0..20 {
            for j in 0..20 {
                image[[i, j]] = 100.0;
            }
        }

        image[[10, 10]] = 200.0;

        let detection = make_test_detection(10.0, 10.0, 200.0);
        let snr = calculate_snr(&detection, &image.view(), 1.5, 3.0, 4.5)
            .expect("SNR calculation should succeed");

        assert_eq!(
            snr,
            f64::MAX,
            "SNR should be MAX for perfect signal with zero noise"
        );
    }

    #[test]
    fn test_snr_with_noisy_background() {
        let mut image = Array2::<f64>::zeros((30, 30));

        for i in 0..30 {
            for j in 0..30 {
                let noise = ((i * 7 + j * 13) % 20) as f64 - 10.0;
                image[[i, j]] = 100.0 + noise;
            }
        }

        image[[15, 15]] = 1000.0;
        image[[14, 15]] = 800.0;
        image[[16, 15]] = 800.0;
        image[[15, 14]] = 800.0;
        image[[15, 16]] = 800.0;

        let detection = make_test_detection(15.0, 15.0, 4200.0);
        let snr = calculate_snr(&detection, &image.view(), 2.0, 4.0, 6.0)
            .expect("SNR calculation should succeed");

        assert!(
            snr > 5.0 && snr < 1000.0,
            "SNR should be reasonable for noisy background, got {}",
            snr
        );
    }

    #[test]
    fn test_snr_with_subpixel_centroid() {
        let mut image = Array2::<f64>::zeros((20, 20));

        for i in 0..20 {
            for j in 0..20 {
                let noise = ((i * 11 + j * 13) % 6) as f64 - 3.0;
                image[[i, j]] = 100.0 + noise;
            }
        }

        image[[10, 10]] = 1000.0;
        image[[10, 11]] = 900.0;
        image[[11, 10]] = 900.0;

        let detection = make_test_detection(10.5, 10.3, 2800.0);
        let snr = calculate_snr(&detection, &image.view(), 2.0, 4.0, 6.0)
            .expect("SNR calculation should succeed");

        assert!(
            snr > 8.0,
            "SNR should work correctly with sub-pixel centroids, got {}",
            snr
        );
    }

    #[test]
    fn test_filter_by_snr_passes_high_snr() {
        let mut image = Array2::<f64>::zeros((20, 20));

        for i in 0..20 {
            for j in 0..20 {
                let noise = ((i * 7 + j * 11) % 5) as f64 - 2.0;
                image[[i, j]] = 100.0 + noise;
            }
        }

        image[[10, 10]] = 1000.0;

        let detection = make_test_detection(10.0, 10.0, 1000.0);

        assert!(
            filter_by_snr(&detection, &image.view(), 5.0, 2.0, 4.0, 6.0),
            "High SNR detection should pass filter"
        );
    }

    #[test]
    fn test_filter_by_snr_rejects_low_snr() {
        let mut image = Array2::<f64>::zeros((30, 30));

        for i in 0..30 {
            for j in 0..30 {
                let noise = ((i * 7 + j * 13) % 20) as f64 - 10.0;
                image[[i, j]] = 100.0 + noise;
            }
        }

        image[[15, 15]] = 110.0;

        let detection = make_test_detection(15.0, 15.0, 110.0);

        assert!(
            !filter_by_snr(&detection, &image.view(), 50.0, 2.0, 4.0, 6.0),
            "Low SNR detection should fail filter"
        );
    }

    #[test]
    fn test_snr_aperture_size_effect() {
        let mut image = Array2::<f64>::zeros((40, 40));

        for i in 0..40 {
            for j in 0..40 {
                let noise = ((i * 11 + j * 7) % 15) as f64 - 7.0;
                image[[i, j]] = 100.0 + noise;
            }
        }

        image[[20, 20]] = 2000.0;
        for di in -2..=2 {
            for dj in -2..=2 {
                if di != 0 || dj != 0 {
                    image[[(20 + di) as usize, (20 + dj) as usize]] = 1000.0;
                }
            }
        }

        let detection = make_test_detection(20.0, 20.0, 10000.0);

        let snr_small = calculate_snr(&detection, &image.view(), 1.5, 3.0, 4.5)
            .expect("SNR calculation should succeed");
        let snr_large = calculate_snr(&detection, &image.view(), 4.0, 8.0, 12.0)
            .expect("SNR calculation should succeed");

        assert!(
            snr_small > 0.0 && snr_large > 0.0,
            "Both aperture sizes should give valid SNR"
        );
        assert_ne!(
            snr_small, snr_large,
            "Different aperture sizes should give different SNR values"
        );
    }

    #[test]
    fn test_snr_with_insufficient_background_pixels() {
        let mut image = Array2::<f64>::zeros((5, 5));

        for i in 0..5 {
            for j in 0..5 {
                image[[i, j]] = 100.0;
            }
        }
        image[[2, 2]] = 1000.0;

        let detection = make_test_detection(2.0, 2.0, 1000.0);
        let result = calculate_snr(&detection, &image.view(), 2.0, 4.0, 6.0);

        assert!(
            result.is_err(),
            "SNR calculation should fail when insufficient background pixels available"
        );
        assert!(
            matches!(result.unwrap_err(), SnrError::InsufficientBackground { .. }),
            "Error should be InsufficientBackground variant"
        );
    }

    #[test]
    fn test_snr_multiple_aperture_radii() {
        let mut image = Array2::<f64>::zeros((40, 40));

        for i in 0..40 {
            for j in 0..40 {
                image[[i, j]] = 100.0 + ((i + j) % 5) as f64;
            }
        }

        image[[20, 20]] = 2000.0;
        image[[19, 20]] = 1500.0;
        image[[21, 20]] = 1500.0;
        image[[20, 19]] = 1500.0;
        image[[20, 21]] = 1500.0;

        let detection = make_test_detection(20.0, 20.0, 8000.0);

        let radii = vec![1.5, 2.0, 2.5, 3.0, 3.5];
        for &radius in &radii {
            let inner = radius * 2.0;
            let outer = radius * 3.0;
            let snr = calculate_snr(&detection, &image.view(), radius, inner, outer)
                .expect("SNR calculation should succeed");
            assert!(
                snr > 0.0,
                "SNR should be positive for aperture radius {}",
                radius
            );
        }
    }

    #[test]
    fn test_snr_with_known_signal_and_noise() {
        use crate::image_proc::noise::generate::simple_normal_array;

        // Test with controlled synthetic data to verify SNR calculation accuracy
        // Parameters:
        // - Background level: 1000.0 ADU
        // - Gaussian noise sigma: 10.0 ADU (clamped positive)
        // - Gaussian spot: amplitude 500.0 ADU, sigma 1.5 pixels at center (50, 50)

        const IMAGE_SIZE: usize = 100;
        const BACKGROUND: f64 = 1000.0;
        const NOISE_SIGMA: f64 = 10.0;
        const SPOT_AMPLITUDE: f64 = 500.0;
        const SPOT_SIGMA: f64 = 1.5;
        const CENTER_X: f64 = 50.0;
        const CENTER_Y: f64 = 50.0;
        const RNG_SEED: u64 = 42;

        // Generate background with properly seeded Gaussian noise
        let noise = simple_normal_array((IMAGE_SIZE, IMAGE_SIZE), 0.0, NOISE_SIGMA, RNG_SEED);
        let mut image = Array2::<f64>::zeros((IMAGE_SIZE, IMAGE_SIZE));

        // Add background with Gaussian noise (clamped positive)
        for i in 0..IMAGE_SIZE {
            for j in 0..IMAGE_SIZE {
                image[[i, j]] = (BACKGROUND + noise[[i, j]]).max(0.0); // Clamp positive
            }
        }

        // Add Gaussian spot centered at (CENTER_X, CENTER_Y)
        for i in 0..IMAGE_SIZE {
            for j in 0..IMAGE_SIZE {
                let dx = j as f64 - CENTER_X;
                let dy = i as f64 - CENTER_Y;
                let r_squared = dx * dx + dy * dy;
                let gaussian =
                    SPOT_AMPLITUDE * (-r_squared / (2.0 * SPOT_SIGMA * SPOT_SIGMA)).exp();
                image[[i, j]] += gaussian;
            }
        }

        let detection = make_test_detection(CENTER_X, CENTER_Y, SPOT_AMPLITUDE);

        // Use aperture radius = 3*sigma to capture ~99% of Gaussian flux
        let aperture_radius = 3.0 * SPOT_SIGMA;
        let background_inner = aperture_radius * 2.0;
        let background_outer = aperture_radius * 3.0;

        let snr = calculate_snr(
            &detection,
            &image.view(),
            aperture_radius,
            background_inner,
            background_outer,
        )
        .expect("SNR calculation should succeed");

        // Expected SNR calculation:
        // Signal = integrated Gaussian flux within aperture
        // For a 2D Gaussian, total flux = 2π * amplitude * sigma^2
        // Within radius r, fraction of flux ≈ 1 - exp(-r²/(2σ²))
        let total_gaussian_flux =
            2.0 * std::f64::consts::PI * SPOT_AMPLITUDE * SPOT_SIGMA * SPOT_SIGMA;
        let aperture_fraction =
            1.0 - (-aperture_radius * aperture_radius / (2.0 * SPOT_SIGMA * SPOT_SIGMA)).exp();
        let expected_signal = total_gaussian_flux * aperture_fraction;

        // Noise = background noise sigma (MAD converts back to sigma via 1.4826 factor)
        let expected_noise = NOISE_SIGMA;

        let expected_snr = expected_signal / expected_noise;

        // Allow 5% relative tolerance due to:
        // - Discrete pixel sampling of continuous Gaussian
        // - MAD estimation variance on finite sample
        // - Noise pattern approximation
        assert_relative_eq!(snr, expected_snr, max_relative = 0.05);
    }
}

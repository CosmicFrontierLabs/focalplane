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
//! 1. **Signal Measurement**: Peak pixel value within the aperture radius
//! 2. **Background Estimation**: Median of pixels in annulus (2-3× aperture radius)
//! 3. **Noise Estimation**: Median Absolute Deviation (MAD) in annulus, converted to RMS
//! 4. **SNR Calculation**: (signal - background) / noise_rms
//!
//! The use of MAD for noise estimation provides robustness against outliers from
//! cosmic rays or neighboring sources in the background annulus.

use ndarray::ArrayView2;

use super::detection::StarDetection;

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
/// - **Signal**: Peak pixel value in aperture minus background median
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
/// Returns an error string if:
/// - Aperture contains no pixels
/// - Background annulus contains fewer than 10 pixels
/// - Noise estimate is zero or negative (indicates no variation in background)
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
) -> Result<f64, String> {
    let (height, width) = image.dim();

    let x_center = detection.x.round() as isize;
    let y_center = detection.y.round() as isize;

    let x_min = (x_center - background_outer_radius.ceil() as isize).max(0) as usize;
    let x_max =
        ((x_center + background_outer_radius.ceil() as isize + 1).min(width as isize)) as usize;
    let y_min = (y_center - background_outer_radius.ceil() as isize).max(0) as usize;
    let y_max =
        ((y_center + background_outer_radius.ceil() as isize + 1).min(height as isize)) as usize;

    let mut background_pixels = Vec::new();
    let mut aperture_pixels = Vec::new();

    for y in y_min..y_max {
        for x in x_min..x_max {
            let dx = x as f64 - detection.x;
            let dy = y as f64 - detection.y;
            let distance = (dx * dx + dy * dy).sqrt();

            if distance <= aperture_radius {
                aperture_pixels.push(image[[y, x]]);
            } else if distance >= background_inner_radius && distance <= background_outer_radius {
                background_pixels.push(image[[y, x]]);
            }
        }
    }

    if aperture_pixels.is_empty() {
        return Err(format!(
            "Aperture contains no pixels at position ({:.1}, {:.1}) with radius {:.1}",
            detection.x, detection.y, aperture_radius
        ));
    }

    if background_pixels.len() < 10 {
        return Err(format!(
            "Insufficient background pixels ({}) at position ({:.1}, {:.1}), need at least 10",
            background_pixels.len(),
            detection.x,
            detection.y
        ));
    }

    background_pixels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let background_median = if background_pixels.len() % 2 == 0 {
        let mid = background_pixels.len() / 2;
        (background_pixels[mid - 1] + background_pixels[mid]) / 2.0
    } else {
        background_pixels[background_pixels.len() / 2]
    };

    let peak_value = aperture_pixels.iter().fold(0.0, |max, &val| val.max(max));
    let signal = peak_value - background_median;

    let mut deviations: Vec<f64> = background_pixels
        .iter()
        .map(|&val| (val - background_median).abs())
        .collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mad = if deviations.len() % 2 == 0 {
        let mid = deviations.len() / 2;
        (deviations[mid - 1] + deviations[mid]) / 2.0
    } else {
        deviations[deviations.len() / 2]
    };

    let noise_rms = 1.4826 * mad;

    if noise_rms <= 0.0 {
        log::warn!(
            "Zero or negative noise estimate (MAD={:.3}, RMS={:.3}) at position ({:.1}, {:.1}), returning f64::MAX",
            mad, noise_rms, detection.x, detection.y
        );
        return Ok(f64::MAX);
    }

    Ok(signal / noise_rms)
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
            snr > 5.0 && snr < 200.0,
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
            result
                .unwrap_err()
                .contains("Insufficient background pixels"),
            "Error should mention insufficient background pixels"
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
}

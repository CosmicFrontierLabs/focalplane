//! Guide star filtering and selection criteria for FGS
//!
//! This module provides filtering functions to select suitable guide stars
//! from detected stellar sources based on SNR, saturation, and other criteria.

use ndarray::ArrayView2;
use shared::image_proc::detection::StarDetection;

/// Filter stars by signal-to-noise ratio
///
/// Calculates SNR using aperture photometry with background annulus estimation.
/// Returns true if the star's SNR exceeds the minimum threshold.
pub fn filter_by_snr(
    detection: &StarDetection,
    image: &ArrayView2<f64>,
    min_snr: f64,
    aperture_radius: f64,
) -> bool {
    let snr = calculate_snr(detection, image, aperture_radius);
    snr >= min_snr
}

/// Calculate signal-to-noise ratio for a detected star
///
/// Uses aperture photometry with a background annulus to estimate SNR.
/// SNR = (peak - background) / noise_rms
pub fn calculate_snr(
    detection: &StarDetection,
    image: &ArrayView2<f64>,
    aperture_radius: f64,
) -> f64 {
    // Define annulus for background estimation
    let background_inner_radius = 2.0 * aperture_radius;
    let background_outer_radius = 3.0 * aperture_radius;

    let (height, width) = image.dim();

    // Calculate bounding box
    let x_center = detection.x.round() as isize;
    let y_center = detection.y.round() as isize;

    let x_min = (x_center - background_outer_radius.ceil() as isize).max(0) as usize;
    let x_max =
        ((x_center + background_outer_radius.ceil() as isize + 1).min(width as isize)) as usize;
    let y_min = (y_center - background_outer_radius.ceil() as isize).max(0) as usize;
    let y_max =
        ((y_center + background_outer_radius.ceil() as isize + 1).min(height as isize)) as usize;

    // Collect pixels
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

    if aperture_pixels.is_empty() || background_pixels.len() < 10 {
        return 0.0;
    }

    // Estimate background using median
    background_pixels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let background_median = if background_pixels.len() % 2 == 0 {
        let mid = background_pixels.len() / 2;
        (background_pixels[mid - 1] + background_pixels[mid]) / 2.0
    } else {
        background_pixels[background_pixels.len() / 2]
    };

    // Find peak in aperture
    let peak_value = aperture_pixels.iter().fold(0.0, |max, &val| val.max(max));
    let signal = peak_value - background_median;

    // Estimate noise using MAD
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

    // Convert MAD to RMS (sigma ≈ 1.4826 * MAD for Gaussian)
    let noise_rms = 1.4826 * mad;

    if noise_rms <= 0.0 {
        return f64::MAX;
    }

    signal / noise_rms
}

/// Filter stars that have saturated pixels
///
/// Returns false if any pixels within search_radius of the star center are saturated.
pub fn filter_by_saturation(
    detection: &StarDetection,
    image: &ArrayView2<u16>,
    saturation_level: u16,
    search_radius: usize,
) -> bool {
    let (height, width) = image.dim();

    let x_center = detection.x.round() as isize;
    let y_center = detection.y.round() as isize;

    let x_min = (x_center - search_radius as isize).max(0) as usize;
    let x_max = ((x_center + search_radius as isize + 1).min(width as isize)) as usize;
    let y_min = (y_center - search_radius as isize).max(0) as usize;
    let y_max = ((y_center + search_radius as isize + 1).min(height as isize)) as usize;

    for y in y_min..y_max {
        for x in x_min..x_max {
            let dx = x as f64 - detection.x;
            let dy = y as f64 - detection.y;
            let distance_squared = dx * dx + dy * dy;

            if distance_squared <= (search_radius as f64 * search_radius as f64)
                && image[[y, x]] >= saturation_level
            {
                return false;
            }
        }
    }

    true
}

/// Check if a star is isolated from other bright sources
///
/// Returns true if no other stars are within isolation_radius.
pub fn is_isolated(
    detection: &StarDetection,
    all_detections: &[StarDetection],
    isolation_radius: f64,
) -> bool {
    for other in all_detections {
        if other.id == detection.id {
            continue;
        }

        let dx = detection.x - other.x;
        let dy = detection.y - other.y;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < isolation_radius {
            return false;
        }
    }

    true
}

/// Calculate distance from image center
pub fn distance_from_center(detection: &StarDetection, image_shape: (usize, usize)) -> f64 {
    let (height, width) = image_shape;
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    let dx = detection.x - center_x;
    let dy = detection.y - center_y;

    (dx * dx + dy * dy).sqrt()
}

/// Calculate a composite quality score for guide star selection
///
/// Combines multiple metrics into a single score (0.0 to 1.0).
pub fn calculate_quality_score(
    snr: f64,
    distance_from_center: f64,
    aspect_ratio: f64,
    image_diagonal: f64,
    weights: (f64, f64, f64), // (snr_weight, position_weight, psf_weight)
) -> f64 {
    let (snr_weight, position_weight, psf_weight) = weights;

    // Normalize SNR (sigmoid function, centered at SNR=20)
    let snr_score = 1.0 / (1.0 + (-0.1 * (snr - 20.0)).exp());

    // Normalize position (prefer center)
    let position_score = 1.0 - (distance_from_center / (image_diagonal / 2.0)).min(1.0);

    // PSF quality based on aspect ratio (1.0 is perfect circle)
    let psf_score = 1.0 / aspect_ratio.max(1.0);

    snr_weight * snr_score + position_weight * position_score + psf_weight * psf_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_snr_calculation() {
        // Create test image with a star
        let mut image = Array2::<f64>::zeros((20, 20));

        // Add background
        for i in 0..20 {
            for j in 0..20 {
                image[[i, j]] = 100.0;
            }
        }

        // Add star at center
        image[[10, 10]] = 1000.0;
        image[[9, 10]] = 800.0;
        image[[11, 10]] = 800.0;
        image[[10, 9]] = 800.0;
        image[[10, 11]] = 800.0;

        let detection = StarDetection {
            id: 0,
            x: 10.0,
            y: 10.0,
            flux: 4200.0,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 2.0,
        };

        let snr = calculate_snr(&detection, &image.view(), 2.0);
        assert!(snr > 10.0);
    }

    #[test]
    fn test_saturation_filter() {
        let mut image = Array2::<u16>::zeros((10, 10));
        image[[5, 5]] = 65535;

        let detection = StarDetection {
            id: 0,
            x: 5.0,
            y: 5.0,
            flux: 65535.0,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 2.0,
        };

        assert!(!filter_by_saturation(&detection, &image.view(), 65535, 2));
    }

    #[test]
    fn test_isolation_check() {
        let star1 = StarDetection {
            id: 0,
            x: 10.0,
            y: 10.0,
            flux: 100.0,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 2.0,
        };

        let star2 = StarDetection {
            id: 1,
            x: 15.0,
            y: 10.0,
            flux: 100.0,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 2.0,
        };

        let detections = vec![star1.clone(), star2];

        assert!(!is_isolated(&star1, &detections, 10.0));
        assert!(is_isolated(&star1, &detections, 4.0));
    }
}

//! Guide star filtering and selection criteria for FGS
//!
//! This module provides filtering functions to select suitable guide stars
//! from detected stellar sources based on SNR, saturation, and other criteria.

use ndarray::ArrayView2;
use shared::image_proc::detection::StarDetection;

pub use shared::image_proc::source_snr::{calculate_snr, filter_by_snr};

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

/// Check if a star is sufficiently far from image edges
///
/// Returns true if the star is at least min_edge_distance pixels from all edges.
pub fn is_within_edge_distance(
    detection: &StarDetection,
    image_shape: (usize, usize),
    min_edge_distance: f64,
) -> bool {
    let (height, width) = image_shape;
    detection.x > min_edge_distance
        && detection.y > min_edge_distance
        && detection.x < (width as f64 - min_edge_distance)
        && detection.y < (height as f64 - min_edge_distance)
}

/// Check if a star has saturated pixels
///
/// Returns false if any pixels within search_radius of the star center exceed saturation_threshold.
/// Works with f64 images (e.g., averaged frames).
pub fn has_no_saturation(
    detection: &StarDetection,
    image: &ArrayView2<f64>,
    saturation_threshold: f64,
    search_radius: f64,
) -> bool {
    let (height, width) = image.dim();

    let x_center = detection.x.round() as isize;
    let y_center = detection.y.round() as isize;

    let search_radius_int = search_radius.ceil() as isize;
    let x_min = (x_center - search_radius_int).max(0) as usize;
    let x_max = ((x_center + search_radius_int + 1).min(width as isize)) as usize;
    let y_min = (y_center - search_radius_int).max(0) as usize;
    let y_max = ((y_center + search_radius_int + 1).min(height as isize)) as usize;

    for y in y_min..y_max {
        for x in x_min..x_max {
            let dx = x as f64 - detection.x;
            let dy = y as f64 - detection.y;
            let distance = (dx * dx + dy * dy).sqrt();

            if distance <= search_radius && image[[y, x]] >= saturation_threshold {
                return false;
            }
        }
    }

    true
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
    fn test_saturation_filter_u16() {
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

    #[test]
    fn test_edge_distance_filter() {
        let image_shape = (100, 100);

        let star_center = StarDetection {
            id: 0,
            x: 50.0,
            y: 50.0,
            flux: 100.0,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 2.0,
        };

        let star_too_close_left = StarDetection {
            id: 1,
            x: 5.0,
            y: 50.0,
            flux: 100.0,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 2.0,
        };

        let star_too_close_bottom = StarDetection {
            id: 2,
            x: 50.0,
            y: 95.0,
            flux: 100.0,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 2.0,
        };

        assert!(is_within_edge_distance(&star_center, image_shape, 10.0));
        assert!(!is_within_edge_distance(
            &star_too_close_left,
            image_shape,
            10.0
        ));
        assert!(!is_within_edge_distance(
            &star_too_close_bottom,
            image_shape,
            10.0
        ));

        assert!(!is_within_edge_distance(
            &star_too_close_left,
            image_shape,
            5.0
        ));
        assert!(is_within_edge_distance(
            &star_too_close_left,
            image_shape,
            4.0
        ));
    }

    #[test]
    fn test_saturation_filter_f64() {
        let mut image = Array2::<f64>::zeros((20, 20));

        // Add background
        for i in 0..20 {
            for j in 0..20 {
                image[[i, j]] = 100.0;
            }
        }

        // Add a bright star at (10, 10) with a saturated pixel
        image[[10, 10]] = 5000.0; // Saturated
        image[[9, 10]] = 800.0;
        image[[11, 10]] = 800.0;

        let detection_saturated = StarDetection {
            id: 0,
            x: 10.0,
            y: 10.0,
            flux: 5000.0,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 2.0,
        };

        let detection_not_saturated = StarDetection {
            id: 1,
            x: 5.0,
            y: 5.0,
            flux: 800.0,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 2.0,
        };

        // Test with saturation threshold at 4000
        assert!(!has_no_saturation(
            &detection_saturated,
            &image.view(),
            4000.0,
            2.0
        ));
        assert!(has_no_saturation(
            &detection_not_saturated,
            &image.view(),
            4000.0,
            2.0
        ));

        // Test with higher threshold - should pass
        assert!(has_no_saturation(
            &detection_saturated,
            &image.view(),
            6000.0,
            2.0
        ));
    }
}

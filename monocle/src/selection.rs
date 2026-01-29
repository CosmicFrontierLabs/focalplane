use ndarray::ArrayView2;
use shared::image_proc::detection::{detect_stars, StarDetection};
use shared::image_proc::downsample_f64;
use shared::image_proc::noise::quantify::estimate_noise_level;

use crate::config::FgsConfig;
use crate::filters;
use crate::GuideStar;

#[derive(Debug, Clone)]
pub struct StarDetectionStats {
    pub count: usize,
    pub brightest_flux: f64,
    pub dimmest_flux: f64,
    pub mean_flux: f64,
}

impl StarDetectionStats {
    pub fn log(&self, note: &str) {
        log::info!(
            "{}: count={}, brightest={:.0}, dimmest={:.0}, mean={:.0}",
            note,
            self.count,
            self.brightest_flux,
            self.dimmest_flux,
            self.mean_flux
        );
    }
}

pub fn calculate_detection_stats(detections: &[StarDetection]) -> Option<StarDetectionStats> {
    if detections.is_empty() {
        return None;
    }

    let count = detections.len();
    let brightest_flux = detections
        .iter()
        .map(|s| s.flux)
        .max_by(|a, b| {
            a.partial_cmp(b)
                .expect("flux values should be valid numbers for comparison")
        })
        .expect("detections should have at least one element after empty check");
    let dimmest_flux = detections
        .iter()
        .map(|s| s.flux)
        .min_by(|a, b| {
            a.partial_cmp(b)
                .expect("flux values should be valid numbers for comparison")
        })
        .expect("detections should have at least one element after empty check");
    let mean_flux = detections.iter().map(|s| s.flux).sum::<f64>() / count as f64;

    Some(StarDetectionStats {
        count,
        brightest_flux,
        dimmest_flux,
        mean_flux,
    })
}

/// Detect stars and select the best guide star from the averaged frame.
///
/// # Arguments
/// * `averaged_frame` - Averaged acquisition frames
/// * `config` - FGS configuration
///
/// # Returns
/// * `Ok((guide_star, all_detections))` - Selected guide star (if any) and all detected stars
/// * `Err(message)` - If detection fails
pub fn detect_and_select_guides(
    averaged_frame: ArrayView2<f64>,
    config: &FgsConfig,
) -> Result<(Option<GuideStar>, Vec<StarDetection>), String> {
    use shared::image_proc::noise::quantify::estimate_background;
    use std::time::Instant;

    let calibration_start = Instant::now();

    // Calculate downsample factor to get ~100 samples for fast background estimation
    // For image of size (H, W), downsample n gives ~(H*W)/n² samples
    // We want (H*W)/n² > 100, so n < sqrt((H*W)/100)
    let (height, width) = averaged_frame.dim();
    let total_pixels = (height * width) as f64;
    let downsample = ((total_pixels / 100.0).sqrt() as usize).max(1);

    let t0 = Instant::now();
    let background_level = estimate_background(&averaged_frame, downsample);
    let background_elapsed = t0.elapsed();

    let t0 = Instant::now();
    // Downsample image for faster noise estimation (noise is a global sensor property)
    let downsampled = downsample_f64(&averaged_frame, config.noise_estimation_downsample);
    let noise_level = estimate_noise_level(&downsampled.view(), 8);
    let noise_elapsed = t0.elapsed();

    let detection_threshold =
        background_level + (config.filters.detection_threshold_sigma * noise_level);

    log::info!(
        "Background: {background_level:.2} ({:.1}ms), Noise: {noise_level:.2} ({:.1}ms), using {}-sigma threshold: {detection_threshold:.2}",
        background_elapsed.as_secs_f64() * 1000.0,
        noise_elapsed.as_secs_f64() * 1000.0,
        config.filters.detection_threshold_sigma
    );

    let t0 = Instant::now();
    let detections = detect_stars(&averaged_frame, Some(detection_threshold));
    let detection_elapsed = t0.elapsed();
    log::info!(
        "Star detection: found {} candidates in {:.1}ms",
        detections.len(),
        detection_elapsed.as_secs_f64() * 1000.0
    );

    if let Some(stats) = calculate_detection_stats(&detections) {
        stats.log("Initial detections");
    }

    // Apply bad pixel filter
    let t0 = Instant::now();
    let after_bad_pixel_filter: Vec<StarDetection> = detections
        .iter()
        .filter(|star| {
            let passes = filters::is_far_from_bad_pixels(
                star,
                &config.filters.bad_pixel_map,
                config.filters.minimum_bad_pixel_distance,
            );
            if !passes {
                log::warn!(
                    "Star rejected by bad pixel filter: position=({:.1}, {:.1}), diameter={:.2}, min_bad_pixel_distance={:.1}, flux={:.0}",
                    star.x, star.y, star.diameter, config.filters.minimum_bad_pixel_distance, star.flux
                );
            }
            passes
        })
        .cloned()
        .collect();
    let bad_pixel_elapsed = t0.elapsed();

    if let Some(stats) = calculate_detection_stats(&after_bad_pixel_filter) {
        stats.log(&format!(
            "After bad pixel filter ({:.1}ms)",
            bad_pixel_elapsed.as_secs_f64() * 1000.0
        ));
    } else {
        log::warn!(
            "All stars filtered out by bad pixel filter ({:.1}ms)",
            bad_pixel_elapsed.as_secs_f64() * 1000.0
        );
    }

    // Apply diameter filter
    let t0 = Instant::now();
    let (diameter_min, diameter_max) = config.filters.diameter_range;
    let after_diameter_filter: Vec<StarDetection> = after_bad_pixel_filter
        .iter()
        .filter(|star| {
            let passes = star.diameter > diameter_min && star.diameter < diameter_max;
            if !passes {
                log::warn!(
                    "Star rejected by diameter filter: diameter={:.2} (range: {:.2}-{:.2}), flux={:.0}, position=({:.1}, {:.1})",
                    star.diameter, diameter_min, diameter_max, star.flux, star.x, star.y
                );
            }
            passes
        })
        .cloned()
        .collect();
    let diameter_elapsed = t0.elapsed();

    if let Some(stats) = calculate_detection_stats(&after_diameter_filter) {
        stats.log(&format!(
            "After diameter filter ({:.1}ms)",
            diameter_elapsed.as_secs_f64() * 1000.0
        ));
    } else {
        log::warn!(
            "All stars filtered out by diameter filter ({:.1}ms)",
            diameter_elapsed.as_secs_f64() * 1000.0
        );
    }

    // Apply aspect ratio filter
    let t0 = Instant::now();
    let after_aspect_filter: Vec<StarDetection> = after_diameter_filter
        .iter()
        .filter(|star| {
            let passes = star.aspect_ratio < config.filters.aspect_ratio_max;
            if !passes {
                log::warn!(
                    "Star rejected by aspect ratio filter: aspect_ratio={:.2} (max: {:.2}), diameter={:.2}, flux={:.0}, position=({:.1}, {:.1})",
                    star.aspect_ratio, config.filters.aspect_ratio_max, star.diameter, star.flux, star.x, star.y
                );
            }
            passes
        })
        .cloned()
        .collect();
    let aspect_elapsed = t0.elapsed();

    if let Some(stats) = calculate_detection_stats(&after_aspect_filter) {
        stats.log(&format!(
            "After aspect ratio filter ({:.1}ms)",
            aspect_elapsed.as_secs_f64() * 1000.0
        ));
    } else {
        log::warn!(
            "All stars filtered out by aspect ratio filter ({:.1}ms)",
            aspect_elapsed.as_secs_f64() * 1000.0
        );
    }

    // Apply edge distance filter
    let t0 = Instant::now();
    let image_shape = averaged_frame.dim();
    let after_edge_filter: Vec<StarDetection> = after_aspect_filter
        .iter()
        .filter(|star| {
            let passes = filters::is_within_edge_distance(
                star,
                image_shape,
                config.filters.minimum_edge_distance,
            );
            if !passes {
                log::warn!(
                    "Star rejected by edge distance filter: position=({:.1}, {:.1}), diameter={:.2}, min_edge_distance={:.1}, image_size=({}, {}), flux={:.0}",
                    star.x, star.y, star.diameter, config.filters.minimum_edge_distance, image_shape.0, image_shape.1, star.flux
                );
            }
            passes
        })
        .cloned()
        .collect();
    let edge_elapsed = t0.elapsed();

    if let Some(stats) = calculate_detection_stats(&after_edge_filter) {
        stats.log(&format!(
            "After edge distance filter ({:.1}ms)",
            edge_elapsed.as_secs_f64() * 1000.0
        ));
    } else {
        log::warn!(
            "All stars filtered out by edge distance filter ({:.1}ms)",
            edge_elapsed.as_secs_f64() * 1000.0
        );
    }

    // Apply saturation filter
    let t0 = Instant::now();
    let saturation_input_count = after_edge_filter.len();
    let after_saturation_filter: Vec<StarDetection> = after_edge_filter
        .iter()
        .filter(|star| {
            let passes = filters::has_no_saturation(
                star,
                &averaged_frame,
                config.filters.saturation_value,
                config.filters.saturation_search_radius,
            );
            if !passes {
                log::warn!(
                    "Star rejected by saturation filter: position=({:.1}, {:.1}), diameter={:.2}, saturation_value={:.0}, search_radius={:.1}, flux={:.0}",
                    star.x, star.y, star.diameter, config.filters.saturation_value, config.filters.saturation_search_radius, star.flux
                );
            }
            passes
        })
        .cloned()
        .collect();
    let saturation_elapsed = t0.elapsed();

    if let Some(stats) = calculate_detection_stats(&after_saturation_filter) {
        stats.log(&format!(
            "After saturation filter ({:.1}ms for {} stars, {:.2}ms/star)",
            saturation_elapsed.as_secs_f64() * 1000.0,
            saturation_input_count,
            if saturation_input_count > 0 {
                saturation_elapsed.as_secs_f64() * 1000.0 / saturation_input_count as f64
            } else {
                0.0
            }
        ));
    } else {
        log::warn!(
            "All stars filtered out by saturation filter ({:.1}ms)",
            saturation_elapsed.as_secs_f64() * 1000.0
        );
    }

    // Apply SNR filter
    let t0 = Instant::now();
    let snr_input_count = after_saturation_filter.len();
    let after_snr_filter: Vec<StarDetection> = after_saturation_filter
        .iter()
        .filter(|star| {
            let aperture = star.diameter / 2.0;
            let passes = filters::filter_by_snr(
                star,
                &averaged_frame,
                config.filters.snr_min,
                aperture,
                aperture * 2.0,
                aperture * 3.0,
            );
            if !passes {
                log::warn!(
                    "Star rejected by SNR filter: position=({:.1}, {:.1}), diameter={:.2}, min_snr={:.1}, flux={:.0}",
                    star.x, star.y, star.diameter, config.filters.snr_min, star.flux
                );
            }
            passes
        })
        .cloned()
        .collect();
    let snr_elapsed = t0.elapsed();

    if let Some(stats) = calculate_detection_stats(&after_snr_filter) {
        stats.log(&format!(
            "After SNR filter ({:.1}ms for {} stars, {:.2}ms/star)",
            snr_elapsed.as_secs_f64() * 1000.0,
            snr_input_count,
            if snr_input_count > 0 {
                snr_elapsed.as_secs_f64() * 1000.0 / snr_input_count as f64
            } else {
                0.0
            }
        ));
    } else {
        log::warn!(
            "All stars filtered out by SNR filter ({:.1}ms)",
            snr_elapsed.as_secs_f64() * 1000.0
        );
    }

    // Sort by flux descending
    let mut candidates = after_snr_filter;
    candidates.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .expect("flux values should be valid numbers for comparison")
    });

    if let Some(stats) = calculate_detection_stats(&candidates) {
        stats.log("Final candidates (sorted by flux)");
    }

    let guide_star = candidates.into_iter().next().and_then(|star| {
        let aperture = star.diameter / 2.0;
        let snr = filters::calculate_snr(
            &star,
            &averaged_frame,
            aperture,
            aperture * 2.0,
            aperture * 3.0,
        )
        .map_err(|e| {
            log::warn!("Failed to calculate SNR for selected guide star: {e}");
            e
        })
        .ok()?;

        Some(GuideStar {
            id: 0,
            x: star.x,
            y: star.y,
            snr,
            shape: star.to_shape(),
        })
    });

    let mut detected_stars = detections;
    detected_stars.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .expect("flux values should be valid numbers for comparison")
    });

    let calibration_elapsed = calibration_start.elapsed();
    log::info!(
        "Total calibration time: {:.1}ms (background: {:.1}ms, noise: {:.1}ms, detection: {:.1}ms, filters: bad_pixel={:.1}ms diameter={:.1}ms aspect={:.1}ms edge={:.1}ms saturation={:.1}ms snr={:.1}ms)",
        calibration_elapsed.as_secs_f64() * 1000.0,
        background_elapsed.as_secs_f64() * 1000.0,
        noise_elapsed.as_secs_f64() * 1000.0,
        detection_elapsed.as_secs_f64() * 1000.0,
        bad_pixel_elapsed.as_secs_f64() * 1000.0,
        diameter_elapsed.as_secs_f64() * 1000.0,
        aspect_elapsed.as_secs_f64() * 1000.0,
        edge_elapsed.as_secs_f64() * 1000.0,
        saturation_elapsed.as_secs_f64() * 1000.0,
        snr_elapsed.as_secs_f64() * 1000.0,
    );

    log::info!(
        "Selected {} guide star for tracking",
        if guide_star.is_some() { 1 } else { 0 }
    );

    Ok((guide_star, detected_stars))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detection(id: usize, x: f64, y: f64, flux: f64) -> StarDetection {
        StarDetection {
            id,
            x,
            y,
            flux,
            m_xx: 1.0,
            m_yy: 1.0,
            m_xy: 0.0,
            aspect_ratio: 1.0,
            diameter: 4.0,
        }
    }

    #[test]
    fn test_calculate_detection_stats_empty() {
        let detections: Vec<StarDetection> = vec![];
        assert!(calculate_detection_stats(&detections).is_none());
    }

    #[test]
    fn test_calculate_detection_stats_single() {
        let detections = vec![make_detection(0, 50.0, 50.0, 1000.0)];
        let stats = calculate_detection_stats(&detections).unwrap();
        assert_eq!(stats.count, 1);
        assert_eq!(stats.brightest_flux, 1000.0);
        assert_eq!(stats.dimmest_flux, 1000.0);
        assert_eq!(stats.mean_flux, 1000.0);
    }

    #[test]
    fn test_calculate_detection_stats_multiple() {
        let detections = vec![
            make_detection(0, 10.0, 10.0, 100.0),
            make_detection(1, 20.0, 20.0, 200.0),
            make_detection(2, 30.0, 30.0, 300.0),
        ];
        let stats = calculate_detection_stats(&detections).unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.brightest_flux, 300.0);
        assert_eq!(stats.dimmest_flux, 100.0);
        assert_eq!(stats.mean_flux, 200.0);
    }

    #[test]
    fn test_calculate_detection_stats_mean_accuracy() {
        let detections = vec![
            make_detection(0, 0.0, 0.0, 10.0),
            make_detection(1, 0.0, 0.0, 20.0),
            make_detection(2, 0.0, 0.0, 30.0),
            make_detection(3, 0.0, 0.0, 40.0),
        ];
        let stats = calculate_detection_stats(&detections).unwrap();
        assert_eq!(stats.count, 4);
        assert_eq!(stats.mean_flux, 25.0); // (10+20+30+40)/4
    }

    #[test]
    fn test_star_detection_stats_log() {
        // Just verify log doesn't panic - this is a basic smoke test
        let stats = StarDetectionStats {
            count: 5,
            brightest_flux: 5000.0,
            dimmest_flux: 100.0,
            mean_flux: 1500.0,
        };
        // This should not panic
        stats.log("Test note");
    }

    #[test]
    fn test_star_detection_stats_clone() {
        let stats = StarDetectionStats {
            count: 3,
            brightest_flux: 3000.0,
            dimmest_flux: 500.0,
            mean_flux: 1250.0,
        };
        let cloned = stats.clone();
        assert_eq!(stats.count, cloned.count);
        assert_eq!(stats.brightest_flux, cloned.brightest_flux);
        assert_eq!(stats.dimmest_flux, cloned.dimmest_flux);
        assert_eq!(stats.mean_flux, cloned.mean_flux);
    }

    #[test]
    fn test_star_detection_stats_debug() {
        let stats = StarDetectionStats {
            count: 10,
            brightest_flux: 10000.0,
            dimmest_flux: 200.0,
            mean_flux: 2500.0,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("StarDetectionStats"));
        assert!(debug_str.contains("10"));
    }
}

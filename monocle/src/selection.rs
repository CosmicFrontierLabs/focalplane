use ndarray::ArrayView2;
use shared::image_proc::detection::{detect_stars, StarDetection};
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

pub fn detect_and_select_guides(
    averaged_frame: ArrayView2<f64>,
    config: &FgsConfig,
) -> Result<(Option<GuideStar>, Vec<StarDetection>), String> {
    use shared::image_proc::noise::quantify::estimate_background;

    // Calculate downsample factor to get ~100 samples for fast background estimation
    // For image of size (H, W), downsample n gives ~(H*W)/n² samples
    // We want (H*W)/n² > 100, so n < sqrt((H*W)/100)
    let (height, width) = averaged_frame.dim();
    let total_pixels = (height * width) as f64;
    let downsample = ((total_pixels / 100.0).sqrt() as usize).max(1);

    let background_level = estimate_background(&averaged_frame, downsample);
    let noise_level = estimate_noise_level(&averaged_frame, 8);

    let detection_threshold =
        background_level + (config.filters.detection_threshold_sigma * noise_level);

    log::info!(
        "Background: {background_level:.2}, Noise: {noise_level:.2}, using {}-sigma threshold: {detection_threshold:.2}",
        config.filters.detection_threshold_sigma
    );

    let detections = detect_stars(&averaged_frame, Some(detection_threshold));

    if let Some(stats) = calculate_detection_stats(&detections) {
        stats.log("Initial detections");
    }

    // Apply bad pixel filter
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

    if let Some(stats) = calculate_detection_stats(&after_bad_pixel_filter) {
        stats.log("After bad pixel filter");
    } else {
        log::warn!("All stars filtered out by bad pixel filter");
    }

    // Apply diameter filter
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

    if let Some(stats) = calculate_detection_stats(&after_diameter_filter) {
        stats.log("After diameter filter");
    } else {
        log::warn!("All stars filtered out by diameter filter");
    }

    // Apply aspect ratio filter
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

    if let Some(stats) = calculate_detection_stats(&after_aspect_filter) {
        stats.log("After aspect ratio filter");
    } else {
        log::warn!("All stars filtered out by aspect ratio filter");
    }

    // Apply edge distance filter
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

    if let Some(stats) = calculate_detection_stats(&after_edge_filter) {
        stats.log("After edge distance filter");
    } else {
        log::warn!("All stars filtered out by edge distance filter");
    }

    // Apply saturation filter
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

    if let Some(stats) = calculate_detection_stats(&after_saturation_filter) {
        stats.log("After saturation filter");
    } else {
        log::warn!("All stars filtered out by saturation filter");
    }

    // Apply SNR filter
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

    if let Some(stats) = calculate_detection_stats(&after_snr_filter) {
        stats.log("After SNR filter");
    } else {
        log::warn!("All stars filtered out by SNR filter");
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

    let (image_height, image_width) = averaged_frame.dim();

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

        // Compute aligned ROI centered on star position
        let roi = config
            .compute_aligned_roi(star.x, star.y, image_width, image_height)
            .or_else(|| {
                log::warn!(
                    "Could not compute aligned ROI for star at ({:.1}, {:.1})",
                    star.x,
                    star.y
                );
                None
            })?;

        Some(GuideStar {
            id: 0,
            x: star.x,
            y: star.y,
            snr,
            roi,
            shape: star.to_shape(),
        })
    });

    let mut detected_stars = detections;
    detected_stars.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .expect("flux values should be valid numbers for comparison")
    });

    log::info!(
        "Selected {} guide star for tracking",
        if guide_star.is_some() { 1 } else { 0 }
    );

    Ok((guide_star, detected_stars))
}

use ndarray::ArrayView2;
use shared::image_proc::detection::aabb::AABB;
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
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let dimmest_flux = detections
        .iter()
        .map(|s| s.flux)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
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
    let noise_level = estimate_noise_level(&averaged_frame, 8);

    let detection_threshold = config.filters.detection_threshold_sigma * noise_level;

    log::info!(
        "Estimated noise level: {noise_level:.2}, using {}-sigma threshold: {detection_threshold:.2}",
        config.filters.detection_threshold_sigma
    );

    let detections = detect_stars(&averaged_frame, Some(detection_threshold));

    if let Some(stats) = calculate_detection_stats(&detections) {
        stats.log("Initial detections");
    }

    // Apply diameter filter
    let (diameter_min, diameter_max) = config.filters.diameter_range;
    let after_diameter_filter: Vec<StarDetection> = detections
        .iter()
        .filter(|star| star.diameter > diameter_min && star.diameter < diameter_max)
        .cloned()
        .collect();

    if let Some(stats) = calculate_detection_stats(&after_diameter_filter) {
        stats.log("After diameter filter");
    }

    // Apply aspect ratio filter
    let after_aspect_filter: Vec<StarDetection> = after_diameter_filter
        .iter()
        .filter(|star| star.aspect_ratio < config.filters.aspect_ratio_max)
        .cloned()
        .collect();

    if let Some(stats) = calculate_detection_stats(&after_aspect_filter) {
        stats.log("After aspect ratio filter");
    }

    // Apply edge distance filter
    let image_shape = averaged_frame.dim();
    let after_edge_filter: Vec<StarDetection> = after_aspect_filter
        .iter()
        .filter(|star| {
            filters::is_within_edge_distance(
                star,
                image_shape,
                config.filters.minimum_edge_distance,
            )
        })
        .cloned()
        .collect();

    if let Some(stats) = calculate_detection_stats(&after_edge_filter) {
        stats.log("After edge distance filter");
    }

    // Apply bad pixel filter
    let after_bad_pixel_filter: Vec<StarDetection> = after_edge_filter
        .iter()
        .filter(|star| {
            filters::is_far_from_bad_pixels(
                star,
                &config.filters.bad_pixel_map,
                config.filters.minimum_bad_pixel_distance,
            )
        })
        .cloned()
        .collect();

    if let Some(stats) = calculate_detection_stats(&after_bad_pixel_filter) {
        stats.log("After bad pixel filter");
    }

    // Apply saturation filter
    let after_saturation_filter: Vec<StarDetection> = after_bad_pixel_filter
        .iter()
        .filter(|star| {
            filters::has_no_saturation(
                star,
                &averaged_frame,
                config.filters.saturation_value,
                config.filters.saturation_search_radius,
            )
        })
        .cloned()
        .collect();

    if let Some(stats) = calculate_detection_stats(&after_saturation_filter) {
        stats.log("After saturation filter");
    }

    // Apply SNR filter
    let after_snr_filter: Vec<StarDetection> = after_saturation_filter
        .iter()
        .filter(|star| {
            let aperture = star.diameter / 2.0;
            filters::filter_by_snr(
                star,
                &averaged_frame,
                config.filters.snr_min,
                aperture,
                aperture * 2.0,
                aperture * 3.0,
            )
        })
        .cloned()
        .collect();

    if let Some(stats) = calculate_detection_stats(&after_snr_filter) {
        stats.log("After SNR filter");
    }

    // Sort by flux descending
    let mut candidates = after_snr_filter;
    candidates.sort_by(|a, b| b.flux.partial_cmp(&a.flux).unwrap());

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
            flux: star.flux,
            snr,
            roi: AABB::from_coords(
                (star.y as i32 - config.roi_size as i32 / 2).max(0) as usize,
                (star.x as i32 - config.roi_size as i32 / 2).max(0) as usize,
                ((star.y as i32 + config.roi_size as i32 / 2)
                    .min(averaged_frame.shape()[0] as i32 - 1)) as usize,
                ((star.x as i32 + config.roi_size as i32 / 2)
                    .min(averaged_frame.shape()[1] as i32 - 1)) as usize,
            ),
            diameter: star.diameter,
        })
    });

    let mut detected_stars = detections;
    detected_stars.sort_by(|a, b| b.flux.partial_cmp(&a.flux).unwrap());

    log::info!(
        "Selected {} guide star for tracking",
        if guide_star.is_some() { 1 } else { 0 }
    );

    Ok((guide_star, detected_stars))
}

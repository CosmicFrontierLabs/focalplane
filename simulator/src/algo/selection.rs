//! Guide star selection algorithms for tracking
//!
//! This module provides algorithms for selecting and ranking stellar sources
//! for use as guide stars in tracking systems. The selection criteria consider
//! factors like brightness, position, PSF quality, and tracking suitability.

use crate::hardware::SatelliteConfig;
use ndarray::ArrayView2;
use shared::image_proc::detection::StarDetection;

/// Quality metrics for a potential guide star
#[derive(Debug, Clone)]
pub struct GuideStarQuality {
    /// Star detection data
    pub detection: StarDetection,

    /// Overall quality score (0.0 to 1.0, higher is better)
    pub quality_score: f64,

    /// Signal-to-noise ratio
    pub snr: f64,

    /// Distance from image center (pixels)
    pub distance_from_center: f64,

    /// PSF shape quality metric (0.0 to 1.0, 1.0 = perfect circular PSF)
    pub psf_quality: f64,

    /// Saturation fraction (0.0 = no saturation, 1.0 = fully saturated)
    pub saturation_fraction: f64,

    /// Whether the star is isolated (no bright neighbors)
    pub is_isolated: bool,
}

/// Guide star selector that ranks stellar sources for tracking
pub struct GuideStarSelector {
    /// Satellite configuration (telescope + sensor)
    #[allow(dead_code)]
    satellite: SatelliteConfig,

    /// Minimum SNR threshold for guide stars
    #[allow(dead_code)]
    min_snr: f64,

    /// Maximum allowed saturation fraction (0.0 to 1.0)
    #[allow(dead_code)]
    max_saturation: f64,

    /// Minimum isolation radius in pixels (no bright stars within this radius)
    #[allow(dead_code)]
    isolation_radius: f64,

    /// Weight for SNR in quality score calculation
    snr_weight: f64,

    /// Weight for position (center preference) in quality score calculation  
    position_weight: f64,

    /// Weight for PSF quality in quality score calculation
    psf_weight: f64,
}

impl GuideStarSelector {
    /// Create a new guide star selector with default parameters
    pub fn new(satellite: SatelliteConfig) -> Self {
        Self {
            satellite,
            min_snr: 10.0,
            max_saturation: 0.1,
            isolation_radius: 20.0,
            snr_weight: 0.4,
            position_weight: 0.3,
            psf_weight: 0.3,
        }
    }

    /// Create a guide star selector with custom parameters
    pub fn with_parameters(
        satellite: SatelliteConfig,
        min_snr: f64,
        max_saturation: f64,
        isolation_radius: f64,
    ) -> Self {
        Self {
            satellite,
            min_snr,
            max_saturation,
            isolation_radius,
            snr_weight: 0.4,
            position_weight: 0.3,
            psf_weight: 0.3,
        }
    }

    /// Set the quality score weights (must sum to 1.0)
    pub fn set_weights(&mut self, snr_weight: f64, position_weight: f64, psf_weight: f64) {
        assert!(
            (snr_weight + position_weight + psf_weight - 1.0).abs() < 1e-6,
            "Weights must sum to 1.0"
        );
        self.snr_weight = snr_weight;
        self.position_weight = position_weight;
        self.psf_weight = psf_weight;
    }

    /// Select and rank guide stars from a list of detected stars
    ///
    /// Returns a vector of guide star candidates sorted by quality score (best first)
    pub fn select_guide_stars(
        &self,
        _detections: &[StarDetection],
        _image: &ArrayView2<u16>,
    ) -> Vec<GuideStarQuality> {
        // TODO: Implement guide star selection logic
        // 1. Calculate quality metrics for each source
        // 2. Filter by minimum requirements
        // 3. Calculate composite quality scores
        // 4. Sort by quality score

        todo!("Implement guide star selection")
    }

    /// Check if a detected star has sufficient signal-to-noise ratio
    ///
    /// Returns false if the SNR is below the minimum threshold (default 10.0).
    /// SNR is calculated as peak signal above background divided by noise RMS.
    #[allow(dead_code)]
    fn filter_snr(&self, detection: &StarDetection, image: &ArrayView2<u16>) -> bool {
        // Calculate SNR for this detection
        let snr = self.calculate_snr_for_detection(detection, image);

        // Must exceed minimum SNR threshold
        snr >= self.min_snr
    }

    /// Calculate the signal-to-noise ratio for a detected star
    ///
    /// SNR = (peak - background) / noise_rms
    /// where noise includes shot noise, read noise, and dark current
    #[allow(dead_code)]
    fn calculate_snr_for_detection(
        &self,
        detection: &StarDetection,
        image: &ArrayView2<u16>,
    ) -> f64 {
        // Define aperture for SNR calculation (use 2x FWHM radius)
        let fwhm_pixels = self.satellite.airy_disk_fwhm_sampled().fwhm();
        let aperture_radius = fwhm_pixels; // 1x FWHM radius for aperture
        let background_inner_radius = 2.0 * fwhm_pixels;
        let background_outer_radius = 3.0 * fwhm_pixels;

        // Get image dimensions
        let (height, width) = image.dim();

        // Calculate bounding box for background estimation
        let x_center = detection.x.round() as isize;
        let y_center = detection.y.round() as isize;

        let x_min = (x_center - background_outer_radius.ceil() as isize).max(0) as usize;
        let x_max =
            ((x_center + background_outer_radius.ceil() as isize + 1).min(width as isize)) as usize;
        let y_min = (y_center - background_outer_radius.ceil() as isize).max(0) as usize;
        let y_max = ((y_center + background_outer_radius.ceil() as isize + 1).min(height as isize))
            as usize;

        // Collect background pixels (annulus between inner and outer radius)
        let mut background_pixels = Vec::new();
        let mut aperture_pixels = Vec::new();

        for y in y_min..y_max {
            for x in x_min..x_max {
                let dx = x as f64 - detection.x;
                let dy = y as f64 - detection.y;
                let distance = (dx * dx + dy * dy).sqrt();

                if distance <= aperture_radius {
                    // Pixel is in the aperture
                    aperture_pixels.push(image[[y, x]] as f64);
                } else if distance >= background_inner_radius && distance <= background_outer_radius
                {
                    // Pixel is in the background annulus
                    background_pixels.push(image[[y, x]] as f64);
                }
            }
        }

        // If we don't have enough pixels, return 0 SNR (will fail filter)
        if aperture_pixels.is_empty() || background_pixels.len() < 10 {
            return 0.0;
        }

        // Estimate background using median (robust to outliers)
        background_pixels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let background_median = if background_pixels.len() % 2 == 0 {
            let mid = background_pixels.len() / 2;
            (background_pixels[mid - 1] + background_pixels[mid]) / 2.0
        } else {
            background_pixels[background_pixels.len() / 2]
        };

        // Find peak pixel value in aperture
        let peak_value = aperture_pixels.iter().fold(0.0, |max, &val| val.max(max));

        // Calculate signal (peak above background)
        let signal = peak_value - background_median;

        // Estimate noise from background pixels (using MAD - Median Absolute Deviation)
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

        // Convert MAD to standard deviation estimate (for Gaussian noise, sigma ≈ 1.4826 * MAD)
        let noise_rms = 1.4826 * mad;

        // Avoid division by zero
        if noise_rms <= 0.0 {
            return f64::MAX; // If no noise, SNR is effectively infinite
        }

        // Calculate and return SNR
        signal / noise_rms
    }

    /// Calculate the signal-to-noise ratio for a detected star
    #[allow(dead_code)]
    fn calculate_snr(&self, _detection: &StarDetection, _image: &ArrayView2<u16>) -> f64 {
        // TODO: Calculate SNR from image data around source position
        // Consider peak pixel value, background noise, and sensor characteristics

        todo!("Implement SNR calculation")
    }

    /// Calculate the distance of a detection from the image center
    #[allow(dead_code)]
    fn distance_from_center(&self, _detection: &StarDetection, _image: &ArrayView2<u16>) -> f64 {
        // TODO: Calculate Euclidean distance from source to image center

        todo!("Implement distance calculation")
    }

    /// Evaluate the PSF quality of a detected star
    #[allow(dead_code)]
    fn evaluate_psf_quality(&self, _detection: &StarDetection, _image: &ArrayView2<u16>) -> f64 {
        // TODO: Analyze PSF shape metrics
        // - Circularity/ellipticity
        // - FWHM consistency
        // - Centroid stability
        // Return value between 0.0 and 1.0

        todo!("Implement PSF quality evaluation")
    }

    /// Check if a detected star has saturated pixels near its center
    ///
    /// Returns false if any pixels within ±2 pixels of the star center are saturated
    #[allow(dead_code)]
    fn filter_saturation(&self, detection: &StarDetection, image: &ArrayView2<u16>) -> bool {
        // Get sensor's maximum value (saturation level) from bit depth
        let saturation_level = (1u32 << self.satellite.sensor.bit_depth) - 1;

        // Define search radius around star center
        let search_radius = 2;

        // Get image dimensions
        let (height, width) = image.dim();

        // Calculate bounding box around star center
        let x_center = detection.x.round() as isize;
        let y_center = detection.y.round() as isize;

        let x_min = (x_center - search_radius).max(0) as usize;
        let x_max = ((x_center + search_radius + 1).min(width as isize)) as usize;
        let y_min = (y_center - search_radius).max(0) as usize;
        let y_max = ((y_center + search_radius + 1).min(height as isize)) as usize;

        // Check all pixels within the search radius
        for y in y_min..y_max {
            for x in x_min..x_max {
                // Check if pixel is within circular radius (not just square)
                let dx = x as f64 - detection.x;
                let dy = y as f64 - detection.y;
                let distance_squared = dx * dx + dy * dy;

                if distance_squared <= (search_radius as f64 * search_radius as f64) {
                    // Check if pixel is saturated
                    if image[[y, x]] >= saturation_level as u16 {
                        return false; // Found saturated pixel, star is not suitable
                    }
                }
            }
        }

        true // No saturated pixels found, star is suitable
    }

    /// Check if a detected star is too close to image edges
    ///
    /// Returns false if the star center is within the minimum distance from any edge.
    /// This prevents selection of stars that might drift out of frame during tracking.
    #[allow(dead_code)]
    fn filter_edge_distance(&self, detection: &StarDetection, image: &ArrayView2<u16>) -> bool {
        // TODO: This distance should be guided by expected open-loop stability
        // For now, hardcode to 5 pixels
        // In the future, could calculate based on:
        // - Expected drift rate (arcsec/s)
        // - Tracking update rate (Hz)
        // - Pixel scale (arcsec/pixel)
        // - Safety margin
        let min_edge_distance = 5.0;

        let (height, width) = image.dim();

        // Check distance from each edge
        let distance_from_left = detection.x;
        let distance_from_right = width as f64 - detection.x;
        let distance_from_top = detection.y;
        let distance_from_bottom = height as f64 - detection.y;

        // Star must be at least min_edge_distance from all edges
        distance_from_left >= min_edge_distance
            && distance_from_right >= min_edge_distance
            && distance_from_top >= min_edge_distance
            && distance_from_bottom >= min_edge_distance
    }

    /// Calculate the saturation fraction for a detected star
    #[allow(dead_code)]
    fn calculate_saturation(&self, _detection: &StarDetection, _image: &ArrayView2<u16>) -> f64 {
        // TODO: Determine what fraction of pixels in the PSF are saturated
        // Use sensor saturation level from satellite config

        todo!("Implement saturation calculation")
    }

    /// Check if a star is isolated from other detections
    ///
    /// Returns false if any other detection is within 3x the FWHM distance.
    /// This ensures the star's PSF doesn't significantly overlap with neighbors.
    #[allow(dead_code)]
    fn filter_isolation(
        &self,
        detection: &StarDetection,
        all_detections: &[StarDetection],
    ) -> bool {
        // Get the FWHM in pixels for this telescope/sensor combination
        let fwhm_pixels = self.satellite.airy_disk_fwhm_sampled().fwhm();

        // Minimum separation is 3x the FWHM
        let min_separation = 3.0 * fwhm_pixels;
        let min_separation_squared = min_separation * min_separation;

        // Check distance to all other detections
        for other in all_detections {
            // Skip self-comparison
            if other.id == detection.id {
                continue;
            }

            // Calculate squared distance (avoid sqrt for performance)
            let dx = detection.x - other.x;
            let dy = detection.y - other.y;
            let distance_squared = dx * dx + dy * dy;

            // If any star is too close, this star is not isolated
            if distance_squared < min_separation_squared {
                return false;
            }
        }

        true // No nearby detections found, star is isolated
    }

    /// Check if a star is isolated from other bright detections
    #[allow(dead_code)]
    fn check_isolation(
        &self,
        _detection: &StarDetection,
        _all_detections: &[StarDetection],
        _image: &ArrayView2<u16>,
    ) -> bool {
        // TODO: Check if there are any bright sources within isolation_radius
        // Consider both detected sources and potential undetected bright pixels

        todo!("Implement isolation check")
    }

    /// Calculate the composite quality score for a guide star candidate
    #[allow(dead_code)]
    fn calculate_quality_score(
        &self,
        _snr: f64,
        _distance_from_center: f64,
        _psf_quality: f64,
        _saturation_fraction: f64,
    ) -> f64 {
        // TODO: Combine metrics with weights to produce final score
        // Normalize each component appropriately
        // Handle edge cases (saturation penalty, etc.)

        todo!("Implement quality score calculation")
    }

    /// Filter detections to only include viable guide star candidates
    #[allow(dead_code)]
    fn filter_viable_candidates(
        &self,
        _detections: &[StarDetection],
        _image: &ArrayView2<u16>,
    ) -> Vec<StarDetection> {
        // TODO: Apply hard filters:
        // - Minimum SNR
        // - Maximum saturation
        // - Must be fully within image bounds
        // - Must have valid PSF

        todo!("Implement candidate filtering")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{sensor::models::GSENSE4040BSI, telescope::models::IDEAL_50CM};
    use crate::units::{Temperature, TemperatureExt};
    use ndarray::Array2;

    #[test]
    fn test_guide_star_selector_creation() {
        // TODO: Add tests for selector creation and configuration
    }

    #[test]
    fn test_weight_validation() {
        // TODO: Test that weights must sum to 1.0
    }

    #[test]
    fn test_guide_star_selection() {
        // TODO: Test full selection pipeline with mock data
    }

    #[test]
    fn test_filter_edge_distance() {
        // Create a test satellite configuration
        let satellite = SatelliteConfig::new(
            IDEAL_50CM.clone(),
            GSENSE4040BSI.clone(),
            Temperature::from_celsius(-10.0),
        );

        let selector = GuideStarSelector::new(satellite);

        // Create test image
        let image = Array2::<u16>::zeros((100, 100));

        // Test 1: Star in center - should pass
        let center_detection = StarDetection {
            id: 1,
            x: 50.0,
            y: 50.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        assert!(selector.filter_edge_distance(&center_detection, &image.view()));

        // Test 2: Star too close to left edge (x = 4.5) - should fail
        let left_edge_detection = StarDetection {
            id: 2,
            x: 4.5,
            y: 50.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        assert!(!selector.filter_edge_distance(&left_edge_detection, &image.view()));

        // Test 3: Star exactly at minimum distance from left edge (x = 5.0) - should pass
        let left_ok_detection = StarDetection {
            id: 3,
            x: 5.0,
            y: 50.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        assert!(selector.filter_edge_distance(&left_ok_detection, &image.view()));

        // Test 4: Star too close to right edge (x = 95.5) - should fail
        let right_edge_detection = StarDetection {
            id: 4,
            x: 95.5,
            y: 50.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        assert!(!selector.filter_edge_distance(&right_edge_detection, &image.view()));

        // Test 5: Star too close to top edge (y = 3.0) - should fail
        let top_edge_detection = StarDetection {
            id: 5,
            x: 50.0,
            y: 3.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        assert!(!selector.filter_edge_distance(&top_edge_detection, &image.view()));

        // Test 6: Star too close to bottom edge (y = 96.0) - should fail
        let bottom_edge_detection = StarDetection {
            id: 6,
            x: 50.0,
            y: 96.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        assert!(!selector.filter_edge_distance(&bottom_edge_detection, &image.view()));

        // Test 7: Star in corner but still within bounds (x = 5.0, y = 5.0) - should pass
        let corner_ok_detection = StarDetection {
            id: 7,
            x: 5.0,
            y: 5.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        assert!(selector.filter_edge_distance(&corner_ok_detection, &image.view()));

        // Test 8: Star in corner too close (x = 4.0, y = 4.0) - should fail
        let corner_bad_detection = StarDetection {
            id: 8,
            x: 4.0,
            y: 4.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        assert!(!selector.filter_edge_distance(&corner_bad_detection, &image.view()));
    }

    #[test]
    fn test_filter_isolation() {
        // Create a test satellite configuration
        let satellite = SatelliteConfig::new(
            IDEAL_50CM.clone(),
            GSENSE4040BSI.clone(),
            Temperature::from_celsius(-10.0),
        );

        let selector = GuideStarSelector::new(satellite.clone());

        // Get FWHM for calculations
        let fwhm_pixels = satellite.airy_disk_fwhm_sampled().fwhm();
        let min_separation = 3.0 * fwhm_pixels;

        // Create main detection at center
        let main_detection = StarDetection {
            id: 1,
            x: 50.0,
            y: 50.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };

        // Test 1: No other detections - should pass
        let detections = vec![main_detection.clone()];
        assert!(selector.filter_isolation(&main_detection, &detections));

        // Test 2: One far detection - should pass
        let far_detection = StarDetection {
            id: 2,
            x: 50.0 + min_separation + 1.0,
            y: 50.0,
            flux: 8000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        let detections = vec![main_detection.clone(), far_detection];
        assert!(selector.filter_isolation(&main_detection, &detections));

        // Test 3: One detection too close - should fail
        let close_detection = StarDetection {
            id: 3,
            x: 50.0 + min_separation - 0.5,
            y: 50.0,
            flux: 8000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        let detections = vec![main_detection.clone(), close_detection];
        assert!(!selector.filter_isolation(&main_detection, &detections));

        // Test 4: Detection exactly at minimum distance - should pass
        // Add small epsilon to avoid floating point precision issues
        let edge_detection = StarDetection {
            id: 4,
            x: 50.0 + min_separation + 0.001,
            y: 50.0,
            flux: 8000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        let detections = vec![main_detection.clone(), edge_detection];
        assert!(selector.filter_isolation(&main_detection, &detections));

        // Test 5: Multiple detections, one too close - should fail
        let detection_far1 = StarDetection {
            id: 5,
            x: 20.0,
            y: 20.0,
            flux: 5000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        let detection_far2 = StarDetection {
            id: 6,
            x: 80.0,
            y: 80.0,
            flux: 5000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        let detection_close = StarDetection {
            id: 7,
            x: 50.0 + min_separation * 0.5,
            y: 50.0,
            flux: 3000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        let detections = vec![
            main_detection.clone(),
            detection_far1,
            detection_far2,
            detection_close,
        ];
        assert!(!selector.filter_isolation(&main_detection, &detections));

        // Test 6: Diagonal distance check (should use Euclidean distance)
        let diagonal_distance = min_separation / std::f64::consts::SQRT_2;
        let diagonal_close = StarDetection {
            id: 8,
            x: 50.0 + diagonal_distance - 0.1,
            y: 50.0 + diagonal_distance - 0.1,
            flux: 7000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        let detections = vec![main_detection.clone(), diagonal_close];
        assert!(!selector.filter_isolation(&main_detection, &detections));

        // Test 7: Same diagonal but far enough - should pass
        let diagonal_far = StarDetection {
            id: 9,
            x: 50.0 + diagonal_distance + 1.0,
            y: 50.0 + diagonal_distance + 1.0,
            flux: 7000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };
        let detections = vec![main_detection.clone(), diagonal_far];
        assert!(selector.filter_isolation(&main_detection, &detections));
    }

    #[test]
    fn test_filter_snr() {
        // Create a test satellite configuration
        let satellite = SatelliteConfig::new(
            IDEAL_50CM.clone(),
            GSENSE4040BSI.clone(),
            Temperature::from_celsius(-10.0),
        );

        let selector = GuideStarSelector::new(satellite.clone());
        let fwhm_pixels = satellite.airy_disk_fwhm_sampled().fwhm();

        // Create test image with known background level
        let mut image = Array2::<u16>::zeros((100, 100));
        let background_level = 100u16;
        image.fill(background_level);

        // Add some noise to background
        for y in 0..100 {
            for x in 0..100 {
                // Add small random-like variation (deterministic for test)
                let noise = ((x + y) % 10) as u16;
                image[[y, x]] = background_level + noise;
            }
        }

        // Test 1: High SNR star - should pass
        let high_snr_detection = StarDetection {
            id: 1,
            x: 50.0,
            y: 50.0,
            flux: 50000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: fwhm_pixels,
        };

        // Add bright star to image (high signal)
        let peak_signal = 500u16;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let x = (50 + dx) as usize;
                let y = (50 + dy) as usize;
                let distance = ((dx * dx + dy * dy) as f64).sqrt();
                if distance <= 2.0 {
                    // Gaussian-like profile
                    let intensity = peak_signal - (distance * 100.0) as u16;
                    image[[y, x]] = background_level + intensity;
                }
            }
        }
        image[[50, 50]] = background_level + peak_signal; // Peak at center

        assert!(selector.filter_snr(&high_snr_detection, &image.view()));

        // Test 2: Low SNR star - should fail
        let low_snr_detection = StarDetection {
            id: 2,
            x: 25.0,
            y: 25.0,
            flux: 500.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: fwhm_pixels,
        };

        // Add faint star (low signal, SNR < 10)
        let faint_peak = 20u16; // Much fainter to ensure SNR < 10
        image[[25, 25]] = background_level + faint_peak;
        image[[24, 25]] = background_level + faint_peak / 4;
        image[[26, 25]] = background_level + faint_peak / 4;
        image[[25, 24]] = background_level + faint_peak / 4;
        image[[25, 26]] = background_level + faint_peak / 4;

        assert!(!selector.filter_snr(&low_snr_detection, &image.view()));

        // Test 3: Custom SNR threshold
        let mut custom_selector = GuideStarSelector::with_parameters(
            satellite.clone(),
            5.0, // Lower SNR threshold
            0.1,
            20.0,
        );

        // The low SNR star should now pass with lower threshold
        assert!(custom_selector.filter_snr(&low_snr_detection, &image.view()));

        // Test 4: Edge star with partial aperture - should handle gracefully
        let edge_detection = StarDetection {
            id: 3,
            x: 2.0,
            y: 2.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: fwhm_pixels,
        };

        // Add bright edge star
        image[[2, 2]] = background_level + 300;
        image[[2, 3]] = background_level + 200;
        image[[3, 2]] = background_level + 200;

        // Should handle edge case without crashing
        let snr = selector.calculate_snr_for_detection(&edge_detection, &image.view());
        assert!(snr >= 0.0); // Should return valid SNR even at edge

        // Test 5: No signal (star at background level) - should fail
        let no_signal_detection = StarDetection {
            id: 4,
            x: 75.0,
            y: 75.0,
            flux: 100.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: fwhm_pixels,
        };

        // Don't add any signal above background
        assert!(!selector.filter_snr(&no_signal_detection, &image.view()));
    }

    #[test]
    fn test_filter_saturation() {
        // Create a test satellite configuration
        let satellite = SatelliteConfig::new(
            IDEAL_50CM.clone(),
            GSENSE4040BSI.clone(),
            Temperature::from_celsius(-10.0),
        );

        let selector = GuideStarSelector::new(satellite.clone());

        // Create test image with known saturation
        let mut image = Array2::<u16>::zeros((50, 50));
        let max_value = ((1u32 << satellite.sensor.bit_depth) - 1) as u16;

        // Create a star detection at position (25, 25)
        let detection = StarDetection {
            id: 1,
            x: 25.0,
            y: 25.0,
            flux: 10000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };

        // Test 1: No saturation - should pass
        image[[25, 25]] = max_value - 100; // Peak below saturation
        image[[24, 25]] = max_value / 2; // Nearby pixels not saturated
        image[[26, 25]] = max_value / 2;
        image[[25, 24]] = max_value / 2;
        image[[25, 26]] = max_value / 2;

        assert!(selector.filter_saturation(&detection, &image.view()));

        // Test 2: Saturated pixel at center - should fail
        image[[25, 25]] = max_value;
        assert!(!selector.filter_saturation(&detection, &image.view()));

        // Test 3: Saturated pixel within 2 pixel radius - should fail
        image[[25, 25]] = max_value - 100; // Center not saturated
        image[[27, 25]] = max_value; // But pixel at distance 2 is saturated
        assert!(!selector.filter_saturation(&detection, &image.view()));

        // Test 4: Saturated pixel just outside 2 pixel radius - should pass
        image[[27, 25]] = max_value / 2; // Clear the previous saturation
        image[[28, 25]] = max_value; // Pixel at distance 3 is saturated
        assert!(selector.filter_saturation(&detection, &image.view()));

        // Test 5: Star near edge of image - should handle bounds correctly
        let edge_detection = StarDetection {
            id: 2,
            x: 1.0,
            y: 1.0,
            flux: 5000.0,
            m_xx: 1.0,
            m_xy: 0.0,
            m_yy: 1.0,
            aspect_ratio: 1.0,
            diameter: 3.0,
        };

        // Clear previous saturation
        image.fill(max_value / 2);
        assert!(selector.filter_saturation(&edge_detection, &image.view()));

        // Add saturation at the edge star position
        image[[1, 1]] = max_value;
        assert!(!selector.filter_saturation(&edge_detection, &image.view()));
    }
}

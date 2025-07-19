//! Single star detection experiments module
//!
//! This module provides functionality for running single star detection experiments
//! to test detection and centroiding accuracy across different:
//! - Sensor models
//! - Star magnitudes
//! - PSF (Point Spread Function) sizes
//!
//! It conducts experiments by:
//! 1. Simulating stars with known positions in images
//! 2. Adding realistic sensor noise based on exposure parameters
//! 3. Running detection algorithms
//! 4. Measuring detection rates and centroid accuracy

use crate::hardware::SatelliteConfig;
use crate::image_proc::detection::{detect_stars_unified, StarDetection, StarFinder};
use crate::image_proc::render::StarInFrame;
use crate::photometry::zodical::SolarAngularCoordinates;
use crate::star_data_to_fluxes;
use crate::Scene;
use core::f64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use starfield::catalogs::StarData;
use starfield::Equatorial;
use std::f64::consts::PI;
use std::time::Duration;

/// Parameters for a single experiment
#[derive(Clone)]
pub struct ExperimentParams {
    /// Domain size (image dimensions)
    pub domain: usize,
    /// Satellite configuration (telescope, sensor, temperature, wavelength)
    pub satellite: SatelliteConfig,
    /// Exposure duration
    pub exposure: Duration,
    /// Star magnitude
    pub mag: f64,
    /// Noise floor multiplier for detection threshold
    pub noise_floor_multiplier: f64,
    /// Indices for result matrix (disk_idx, exposure_idx, mag_idx)
    pub indices: (usize, usize, usize),
    /// Number of times to run the experiment
    pub experiment_count: u32,
    /// Solar coordinates for zodiacal background
    pub coordinates: SolarAngularCoordinates,
    /// Star detection algorithm to use
    pub star_finder: StarFinder,
    /// Random seed for reproducible experiments
    pub seed: u64,
}

impl ExperimentParams {
    /// Creates a StarInFrame at the given position with flux calculated from magnitude
    fn star_at_pos(&self, xpos: f64, ypos: f64) -> StarInFrame {
        // Create dummy star data (position doesn't matter for this test)
        let star_data = StarData {
            id: 0,
            position: Equatorial::from_degrees(0.0, 0.0),
            magnitude: self.mag,
            b_v: None,
        };

        StarInFrame {
            star: star_data,
            x: xpos,
            y: ypos,
            spot: star_data_to_fluxes(&star_data, &self.satellite),
        }
    }
}

pub struct ExperimentResults {
    pub params: ExperimentParams,
    pub xy_err: Vec<(f64, f64)>,
    pub spurious_detections: usize,
    pub multiple_detections: usize,
    pub no_detections: usize,
}

impl ExperimentResults {
    /// Calculate the RMS error from collected x/y errors
    pub fn rms_error(&self) -> f64 {
        let mut sum = 0.0;

        for &(x, y) in &self.xy_err {
            sum += (x * x + y * y).sqrt();
        }

        sum /= self.xy_err.len() as f64;
        sum
    }

    /// Get the detection rate based on number of accumulated errors vs experiments
    pub fn detection_rate(&self) -> f64 {
        let total_experiments = self.params.experiment_count;
        if total_experiments == 0 {
            return f64::NAN;
        }
        self.xy_err.len() as f64 / total_experiments as f64
    }

    pub fn rms_error_radians(&self) -> f64 {
        let pix_err = self.rms_error();
        let err_m = self.params.satellite.sensor.pixel_size_um * pix_err / 1_000_000.0;
        (err_m / self.params.satellite.telescope.focal_length_m).tan()
    }

    pub fn rms_err_mas(&self) -> f64 {
        let _pix_err = self.rms_error();
        self.rms_error_radians() * (648_000_000.0 / PI)
    }

    pub fn false_positive_rate(&self) -> f64 {
        let total_experiments = self.params.experiment_count;
        (self.spurious_detections + self.multiple_detections) as f64 / total_experiments as f64
    }

    pub fn spurious_rate(&self) -> f64 {
        let total_experiments = self.params.experiment_count;
        self.spurious_detections as f64 / total_experiments as f64
    }
}

/// Run a single experiment and return the error (distance) or NaN if not detected, and detection rate
pub fn run_single_experiment(params: &ExperimentParams) -> ExperimentResults {
    let half_d = params.domain as f64 / 2.0;

    // Create seeded RNG for reproducible results
    let mut rng = StdRng::seed_from_u64(params.seed);

    // Run the experiment multiple times and average the results
    let mut xy_err: Vec<(f64, f64)> = Vec::new();
    let mut spurious_detections = 0;
    let mut multiple_detections = 0;
    let mut no_detections = 0;

    for _ in 0..params.experiment_count {
        // Generate random position in the center area
        let xpos = rng.gen::<f64>() * half_d + half_d;
        let ypos = rng.gen::<f64>() * half_d + half_d;

        // Create star at the position with correct flux
        let star = params.star_at_pos(xpos, ypos);

        // Create scene with single star
        let scene = Scene::from_stars(
            params.satellite.clone(),
            vec![star],
            Equatorial::from_degrees(0.0, 0.0), // Dummy pointing (not used for pre-positioned stars)
            params.coordinates,
        );

        // Render the scene with a unique seed for this iteration
        let render_seed = rng.gen::<u64>();
        let render_result = scene.render_with_seed(&params.exposure, Some(render_seed));

        // Use consistent background RMS calculation
        let background_rms = render_result.background_rms();
        // Run star detection algorithm
        let detected_stars: Vec<StarDetection> = match detect_stars_unified(
            render_result.quantized_image.view(),
            params.star_finder,
            &params.satellite.airy_disk_fwhm_sampled(),
            background_rms,
            params.noise_floor_multiplier,
        ) {
            Ok(stars) => stars
                .into_iter()
                .map(|star| {
                    // Convert boxed StellarSource to StarDetection-like structure
                    {
                        let (x, y) = star.get_centroid();
                        StarDetection {
                            id: 0,
                            x,
                            y,
                            flux: star.flux(),
                            m_xx: 1.0,
                            m_yy: 1.0,
                            m_xy: 0.0,
                            aspect_ratio: 1.0,
                            diameter: 2.0,
                            is_valid: true,
                        }
                    }
                })
                .collect(),
            Err(e) => {
                log::warn!("Star detection failed: {}", e);
                continue;
            }
        };

        // Calculate detection error if star was found
        let mut best: Option<(f64, f64)> = None;
        let mut best_err = f64::INFINITY;

        if detected_stars.is_empty() {
            // No stars detected
            no_detections += 1;
            continue;
        } else if detected_stars.len() > 1 {
            // Multiple stars detected
            multiple_detections += 1;
            log::debug!(
                "{} stars detected in frame (expected 1)",
                detected_stars.len()
            );
            continue;
        }

        for detection in &detected_stars {
            let x_diff = detection.x - xpos;
            let y_diff = detection.y - ypos;
            let err = (x_diff * x_diff + y_diff * y_diff).sqrt();

            // Detect spurious detections (mostly) and skip them
            let airy_disk = params.satellite.airy_disk_fwhm_sampled();
            if err > airy_disk.first_zero().max(1.0) * 3.0 {
                spurious_detections += 1;
                log::debug!("Spurious detection: {} pixels from true position", err);
                break; // Break out of detection loop for this experiment
            }

            if err < best_err {
                best_err = err;
                best = Some((x_diff, y_diff));
            }
        }

        if let Some((x_diff, y_diff)) = best {
            xy_err.push((x_diff, y_diff));
        }
    }

    ExperimentResults {
        params: params.clone(),
        xy_err,
        spurious_detections,
        multiple_detections,
        no_detections,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::sensor::models::{ALL_SENSORS, GSENSE4040BSI, HWK4123};
    use crate::hardware::telescope::models::DEMO_50CM;
    use crate::photometry::zodical::SolarAngularCoordinates;
    use std::time::Duration;

    #[test]
    fn test_single_star_detection_all_sensors() {
        // Test parameters
        let domain = 128;
        let bright_magnitude = 8.0; // Bright star that should be easily detected
        let exposure = Duration::from_millis(100);
        let noise_floor_multiplier = 5.0;
        let experiment_count = 5; // Run multiple times to ensure consistency

        // Use the 50cm demo telescope from models
        use crate::hardware::telescope::models::DEMO_50CM;
        let telescope = DEMO_50CM.clone();

        // Solar coordinates (minimum zodiacal background)
        let coordinates = SolarAngularCoordinates::zodiacal_minimum();

        // Test all sensors
        for sensor_model in ALL_SENSORS.iter() {
            let sized_sensor = sensor_model.with_dimensions(domain, domain);
            let satellite = SatelliteConfig::new(
                telescope.clone(),
                sized_sensor,
                0.0,   // 0°C
                550.0, // 550nm wavelength
            );

            // Adjust for reasonable PSF sampling
            let adjusted_satellite = satellite.with_fwhm_sampling(1.65);

            let params = ExperimentParams {
                domain,
                satellite: adjusted_satellite,
                exposure,
                mag: bright_magnitude,
                noise_floor_multiplier,
                indices: (0, 0, 0), // Not used in single test
                experiment_count,
                coordinates,
                star_finder: StarFinder::Naive,
                seed: 42, // Fixed seed for reproducible tests
            };

            let results = run_single_experiment(&params);

            // Verify detection rate is high for bright star
            let detection_rate = results.detection_rate();
            assert!(
                detection_rate > 0.9,
                "Sensor {} should detect bright star (mag {}) with >90% success rate, got {}%",
                sensor_model.name,
                bright_magnitude,
                detection_rate * 100.0
            );

            // Verify position accuracy is reasonable
            if !results.xy_err.is_empty() {
                let rms_error_pixels = results.rms_error();
                assert!(
                    rms_error_pixels < 0.5,
                    "Sensor {} centroid error should be <0.5 pixels for bright star, got {} pixels",
                    sensor_model.name,
                    rms_error_pixels
                );
            }

            // Verify low false positive rate
            let spurious_rate = results.spurious_rate();
            assert!(
                spurious_rate < 0.1,
                "Sensor {} spurious detection rate should be <10%, got {}%",
                sensor_model.name,
                spurious_rate * 100.0
            );

            println!(
                "Sensor {} - Detection: {:.1}%, RMS Error: {:.3} pixels, Spurious: {:.1}%",
                sensor_model.name,
                detection_rate * 100.0,
                results.rms_error(),
                spurious_rate * 100.0
            );
        }
    }

    #[test]
    fn test_faint_star_detection() {
        // Test with a faint star to verify detection limits
        let domain = 128;
        let faint_magnitude = 16.0; // Faint star
        let exposure = Duration::from_secs(1); // Longer exposure

        let telescope = DEMO_50CM.clone();
        let sensor = HWK4123.with_dimensions(domain, domain);
        let satellite = SatelliteConfig::new(telescope, sensor.clone(), 0.0, 550.0);
        let adjusted_satellite = satellite.with_fwhm_sampling(2.0);

        let params = ExperimentParams {
            domain,
            satellite: adjusted_satellite,
            exposure,
            mag: faint_magnitude,
            noise_floor_multiplier: 3.0, // More sensitive threshold
            indices: (0, 0, 0),
            experiment_count: 5,
            coordinates: SolarAngularCoordinates::zodiacal_minimum(),
            star_finder: StarFinder::Naive,
            seed: 12345, // Fixed seed
        };

        let results = run_single_experiment(&params);
        let detection_rate = results.detection_rate();

        println!(
            "Faint star (mag {}) detection rate with {}: {:.1}%",
            faint_magnitude,
            sensor.name,
            detection_rate * 100.0
        );

        // We expect some detections but not 100% for faint stars
        assert!(
            detection_rate > 0.0,
            "Should detect some faint stars with long exposure"
        );
    }

    #[test]
    fn test_seed_reproducibility() {
        // Test that using the same seed produces identical results
        let domain = 64;
        let telescope = DEMO_50CM.clone();
        let sensor = GSENSE4040BSI.with_dimensions(domain, domain);
        let satellite = SatelliteConfig::new(telescope, sensor, 0.0, 550.0);
        let adjusted_satellite = satellite.with_fwhm_sampling(2.0);

        let params = ExperimentParams {
            domain,
            satellite: adjusted_satellite,
            exposure: Duration::from_millis(100),
            mag: 10.0,
            noise_floor_multiplier: 3.0,
            indices: (0, 0, 0),
            experiment_count: 10,
            coordinates: SolarAngularCoordinates::zodiacal_minimum(),
            star_finder: StarFinder::Naive,
            seed: 12345, // Fixed seed
        };

        // Run experiment twice with same seed
        let result1 = run_single_experiment(&params);
        let result2 = run_single_experiment(&params);

        println!(
            "First run - Detection rate: {:.1}%, Detections: {}",
            result1.detection_rate() * 100.0,
            result1.xy_err.len()
        );
        println!(
            "Second run - Detection rate: {:.1}%, Detections: {}",
            result2.detection_rate() * 100.0,
            result2.xy_err.len()
        );

        // Results should be identical
        assert_eq!(
            result1.xy_err.len(),
            result2.xy_err.len(),
            "Same seed should produce same number of detections"
        );
        assert_eq!(
            result1.spurious_detections, result2.spurious_detections,
            "Same seed should produce same spurious detections"
        );
        assert_eq!(
            result1.multiple_detections, result2.multiple_detections,
            "Same seed should produce same multiple detections"
        );
        assert_eq!(
            result1.no_detections, result2.no_detections,
            "Same seed should produce same no-detection count"
        );

        // Check position errors are identical
        for (i, ((x1, y1), (x2, y2))) in result1.xy_err.iter().zip(&result2.xy_err).enumerate() {
            assert!(
                (x1 - x2).abs() < 1e-10 && (y1 - y2).abs() < 1e-10,
                "Detection {i} positions should be identical: ({x1:.6}, {y1:.6}) vs ({x2:.6}, {y2:.6})"
            );
        }
    }
}

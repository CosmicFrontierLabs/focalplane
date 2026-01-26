//! Scene running and experiment execution module
//!
//! This module provides functionality for running imaging experiments with star field simulation,
//! including star detection, ICP matching, and CSV result logging.

use crate::hardware::SatelliteConfig;
use crate::image_proc::render::{RenderingResult, StarInFrame};
use crate::photometry::zodiacal::SolarAngularCoordinates;
use crate::scene::Scene;
use crate::{
    units::{Length, LengthExt, TemperatureExt},
    SensorConfig,
};
use core::f64;
use image::DynamicImage;
use log::{debug, info, warn};
use meter_math::icp::{icp_match_indices, Locatable2d};
use shared::algo::MinMaxScan;
use shared::frame_writer::{FrameFormat, FrameWriterHandle};
use shared::image_proc::airy::PixelScaledAiryDisk;
use shared::image_proc::detection::{detect_stars_unified, StarFinder};
use shared::image_proc::histogram_stretch::sigma_stretch;
use shared::image_proc::image::array2_to_gray_image;
use shared::image_proc::{draw_stars_with_x_markers, stretch_histogram, u16_to_u8_scaled};
use shared::viz::histogram::{Histogram, HistogramConfig, Scale};
use starfield::catalogs::{StarCatalog, StarData};
use starfield::image::starfinders::StellarSource;
use starfield::Equatorial;

/// Wrapper to implement Locatable2d for dyn StellarSource
struct LocatableStellarSource<'a>(&'a dyn StellarSource);

impl Locatable2d for LocatableStellarSource<'_> {
    fn x(&self) -> f64 {
        self.0.get_centroid().0
    }

    fn y(&self) -> f64 {
        self.0.get_centroid().1
    }
}
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Thread-safe CSV writer for experiment results
#[derive(Debug)]
pub struct CsvWriter {
    file: Arc<Mutex<File>>,
}

impl CsvWriter {
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        // Write header
        let headers = vec![
            "experiment_num",
            "trial_num",
            "ra",
            "dec",
            "focal_length_m",
            "sensor",
            "exposure_ms",
            "star_count",
            "brightest_mag",
            "faintest_mag",
            "pixel_error",
            "translation_x",
            "translation_y",
            "rotation_deg",
            "brightest_star_pixel_error",
            "brightest_star_dx",
            "brightest_star_dy",
        ];
        writeln!(file, "{}", headers.join(","))?;
        Ok(CsvWriter {
            file: Arc::new(Mutex::new(file)),
        })
    }

    pub fn write_result(
        &self,
        result: &ExperimentResult,
        aperture_m: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = self.file.lock().unwrap();
        let focal_length_m = result.f_number * aperture_m;

        // Write one row per exposure
        for (duration, exposure_result) in result.exposure_results.iter() {
            let exposure_ms = duration.as_millis();
            writeln!(
                file,
                "{},{},{:.6},{:.6},{:.2},{},{},{},{:.2},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                result.experiment_num,
                result.trial_num,
                result.coordinates.ra_degrees(),
                result.coordinates.dec_degrees(),
                focal_length_m,
                result.sensor_name,
                exposure_ms,
                exposure_result.detected_count,
                exposure_result.brightest_magnitude,
                exposure_result.faintest_magnitude,
                exposure_result.alignment_error,
                exposure_result.translation_x,
                exposure_result.translation_y,
                exposure_result.rotation_deg,
                exposure_result.brightest_star_pixel_error,
                exposure_result.brightest_star_dx,
                exposure_result.brightest_star_dy
            )?;
        }
        Ok(())
    }
}

/// Common arguments for experiments
#[derive(Debug)]
pub struct ExperimentCommonArgs {
    pub exposures: Vec<Duration>, // Multiple exposure durations
    pub coordinates: SolarAngularCoordinates,
    pub noise_multiple: f64,
    pub output_dir: String,
    pub save_images: bool,
    pub icp_max_iterations: usize,
    pub icp_convergence_threshold: f64,
    pub star_finder: StarFinder,
    pub csv_writer: Arc<CsvWriter>, // Thread-safe CSV writer
    pub aperture_m: f64,            // Needed for focal length calculation
}

impl Clone for ExperimentCommonArgs {
    fn clone(&self) -> Self {
        Self {
            exposures: self.exposures.clone(),
            coordinates: self.coordinates,
            noise_multiple: self.noise_multiple,
            output_dir: self.output_dir.clone(),
            save_images: self.save_images,
            icp_max_iterations: self.icp_max_iterations,
            icp_convergence_threshold: self.icp_convergence_threshold,
            star_finder: self.star_finder,
            csv_writer: Arc::clone(&self.csv_writer),
            aperture_m: self.aperture_m,
        }
    }
}

/// Parameters for a single experiment (one sky pointing with all satellites)
#[derive(Debug, Clone)]
pub struct ExperimentParams {
    pub experiment_num: u32,
    pub trial_num: u32, // Trial number for this pointing (0-based)
    pub rng_seed: u64,  // Unique RNG seed for this trial
    pub ra_dec: Equatorial,
    pub satellites: Vec<SatelliteConfig>,
    pub f_numbers: Vec<f64>, // Multiple f-numbers to test
    pub common_args: ExperimentCommonArgs,
}

/// Results from a single exposure test
#[derive(Debug, Clone)]
pub struct ExposureResult {
    pub detected_count: usize,
    pub brightest_magnitude: f64,
    pub faintest_magnitude: f64,
    pub alignment_error: f64, // Legacy: magnitude of translation (kept for compatibility)
    pub translation_x: f64,   // X component of frame misalignment (pixels)
    pub translation_y: f64,   // Y component of frame misalignment (pixels)
    pub rotation_deg: f64,    // Rotational misalignment (degrees)
    pub brightest_star_pixel_error: f64, // Distance from brightest star to its ICP-aligned position
    pub brightest_star_dx: f64, // X component of brightest star error (pixels)
    pub brightest_star_dy: f64, // Y component of brightest star error (pixels)
}

/// Results from all exposures for one experiment/sensor combination
#[derive(Debug)]
pub struct ExperimentResult {
    pub experiment_num: u32,
    pub trial_num: u32, // Trial number for this pointing
    pub sensor_name: String,
    pub f_number: f64, // F-number used for this result
    pub coordinates: Equatorial,
    pub exposure_results: HashMap<Duration, ExposureResult>, // Key is exposure duration
    pub duration: Duration,                                  // Time taken for this experiment
}

/// Prints histogram of star magnitudes
///
/// # Arguments
/// * `stars` - Vector of stars to analyze
pub fn print_am_hist(stars: &[StarData]) {
    // Print histogram of star magnitudes
    if stars.is_empty() {
        warn!("No stars available to create histogram");
    } else {
        debug!("Creating histogram of star magnitudes...");
        debug!("Note that these stats include stars in the sensor circumcircle");
        let star_magnitudes: Vec<f64> = stars.iter().map(|star| star.magnitude).collect();

        // Create a magnitude histogram using the new specialized function
        // This automatically creates bins centered on integer magnitudes with 1.0 width
        let mag_hist = shared::viz::histogram::create_magnitude_histogram(
            &star_magnitudes,
            Some(format!("Star Magnitude Histogram ({} stars)", stars.len())),
            false, // Use linear scale
        )
        .expect("Failed to create magnitude histogram");

        // Print the histogram
        debug!(
            "\n{}",
            mag_hist.format().expect("Failed to format histogram")
        );
    }
}

/// Runs a single imaging experiment with specified parameters
///
/// Renders star field for one sky pointing across all sensors, detects stars, and optionally saves output images
///
/// # Arguments
/// * `params` - Complete experiment parameters (one sky pointing, all sensors)
/// * `catalog` - Star catalog for field selection
/// * `max_fov` - Maximum field of view to use for star selection
///
/// # Returns
/// * `Vec<ExperimentResult>` containing detection results for each sensor
pub fn run_experiment<T: StarCatalog>(
    params: &ExperimentParams,
    catalog: &T,
    max_fov: f64,
    frame_writer: &FrameWriterHandle,
) -> Vec<ExperimentResult> {
    let experiment_start = Instant::now();
    let output_path = Path::new(&params.common_args.output_dir);

    debug!(
        "Running experiment {} (trial {})...",
        params.experiment_num, params.trial_num
    );
    debug!(
        "  RA: {:.2}, Dec: {:.2}",
        params.ra_dec.ra_degrees(),
        params.ra_dec.dec_degrees()
    );
    debug!("  RNG Seed: {}", params.rng_seed);
    debug!("  Temperature/Wavelength: per-satellite configuration");
    debug!("  Exposures: {:?}", params.common_args.exposures);
    debug!("  F-numbers: {:?}", params.f_numbers);
    debug!("  Noise Multiple: {}", params.common_args.noise_multiple);
    debug!("  Output Dir: {}", params.common_args.output_dir);
    debug!("  Save Images: {}", params.common_args.save_images);

    // Compute stars once for this sky pointing using max FOV
    let stars = catalog.stars_in_field(
        params.ra_dec.ra_degrees(),
        params.ra_dec.dec_degrees(),
        max_fov,
    );
    print_am_hist(&stars);

    let mut results = Vec::new();

    // Run experiment for each f-number
    for f_number in params.f_numbers.iter() {
        // Run for each satellite at this f-number
        for satellite in params.satellites.iter() {
            debug!(
                "Running experiment for satellite: {} at f/{:.1} (T: {:.1}°C, λ: {:.0}nm)",
                satellite.sensor.name,
                f_number,
                satellite.temperature.as_celsius(),
                satellite.telescope.corrected_to.as_nanometers()
            );

            let mut exposure_results = HashMap::new();

            // Create a modified satellite config with the new f-number
            let focal_length =
                Length::from_meters(satellite.telescope.aperture.as_meters() * f_number);
            let modified_telescope = satellite.telescope.with_focal_length(focal_length);
            let modified_satellite = SatelliteConfig::new(
                modified_telescope,
                satellite.sensor.clone(),
                satellite.temperature,
            );

            // Create scene with star projection for this satellite and f-number
            let scene = Scene::from_catalog(
                modified_satellite.clone(),
                stars.clone(),
                params.ra_dec,
                params.common_args.coordinates,
            );

            // Loop through each exposure duration
            for exposure_duration in params.common_args.exposures.iter() {
                debug!(
                    "  Testing exposure duration: {:.1}s",
                    exposure_duration.as_secs_f64()
                );

                // Log rendering start
                info!(
                    "Rendering: Exp {} | Sensor {} | RA {:.2}° Dec {:.2}° | Exposure {:.3}s",
                    params.experiment_num,
                    satellite.sensor.name,
                    params.ra_dec.ra_degrees(),
                    params.ra_dec.dec_degrees(),
                    exposure_duration.as_secs_f64()
                );

                // Render the scene with unique seed for this trial
                let render_result =
                    scene.render_with_seed(exposure_duration, Some(params.rng_seed));

                let exposure_ms = exposure_duration.as_millis();
                let prefix = format!(
                    "{:04}_{}_{:05}ms",
                    params.experiment_num,
                    satellite.sensor.name.replace(" ", "_"),
                    exposure_ms
                );

                // Calculate background RMS using the new method
                let background_rms = render_result.background_rms();
                let airy_disk_pixels = satellite.airy_disk_fwhm_sampled().fwhm();
                let detection_sigma = params.common_args.noise_multiple;

                // Do the star detection
                let scaled_airy_disk = PixelScaledAiryDisk::with_fwhm(
                    airy_disk_pixels,
                    satellite.telescope.corrected_to,
                );
                let detected_stars: Vec<Box<dyn StellarSource>> = match detect_stars_unified(
                    render_result.quantized_image.view(),
                    params.common_args.star_finder,
                    &scaled_airy_disk,
                    background_rms,
                    detection_sigma,
                ) {
                    Ok(stars) => stars,
                    Err(e) => {
                        log::warn!("Star detection failed: {e}");
                        vec![]
                    }
                };

                // Save images if enabled
                if params.common_args.save_images {
                    save_image_outputs(
                        &render_result,
                        &satellite.sensor,
                        &detected_stars,
                        output_path,
                        &prefix,
                        frame_writer,
                    );
                }

                // Now we take our detected stars and match them against the sources
                // Get projected stars from scene instead of render_result
                let projected_stars: Vec<StarInFrame> = scene.stars.clone();
                // Wrap detected stars for ICP matching
                let wrapped_detected: Vec<LocatableStellarSource> = detected_stars
                    .iter()
                    .map(|s| LocatableStellarSource(s.as_ref()))
                    .collect();
                let result = match icp_match_indices(
                    &wrapped_detected,
                    &projected_stars,
                    params.common_args.icp_max_iterations,
                    params.common_args.icp_convergence_threshold,
                ) {
                    Ok((match_indices, icp_result)) => {
                        // Debug statistics output
                        debug_stats(
                            &render_result,
                            &detected_stars,
                            &projected_stars,
                            &match_indices,
                        )
                        .unwrap();

                        let magnitudes: Vec<f64> = match_indices
                            .iter()
                            .map(|(_, tgt_idx)| projected_stars[*tgt_idx].star.magnitude)
                            .collect();

                        let mag_scan = MinMaxScan::new(&magnitudes);
                        let (brightest_mag, faintest_mag) =
                            mag_scan.min_max().unwrap_or((f64::NAN, f64::NAN));

                        info!(
                            "Detected {} stars. Faintest magnitude: {:.2}",
                            magnitudes.len(),
                            faintest_mag
                        );

                        debug!("ICP match results:");
                        debug!("\tMatched stars: {}", match_indices.len());
                        debug!("\tConverged in {} iterations", icp_result.iterations);
                        debug!("\tTranslation: {:?}", icp_result.translation);
                        debug!("\tRotation: {:?}", icp_result.rotation);
                        debug!("\tScale: {:?}", icp_result.rotation_quat);

                        let alignment_error =
                            icp_result.translation.map(|disp| disp * disp).sum().sqrt();

                        // Extract translation components
                        let translation_x = icp_result.translation.x;
                        let translation_y = icp_result.translation.y;

                        // Calculate rotation angle from rotation matrix
                        // For 2D rotation matrix [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
                        // We can extract θ from atan2(sin(θ), cos(θ))
                        let rotation_rad =
                            icp_result.rotation[(1, 0)].atan2(icp_result.rotation[(0, 0)]);
                        let rotation_deg = rotation_rad.to_degrees();

                        // Find the brightest star (lowest magnitude) and calculate its pixel error
                        let (brightest_star_pixel_error, brightest_star_dx, brightest_star_dy) =
                            if !match_indices.is_empty() {
                                match_indices
                                    .iter()
                                    .min_by(|(_, a_idx), (_, b_idx)| {
                                        projected_stars[*a_idx]
                                            .star
                                            .magnitude
                                            .partial_cmp(&projected_stars[*b_idx].star.magnitude)
                                            .unwrap()
                                    })
                                    .map(|(src_idx, tgt_idx)| {
                                        let detected = &detected_stars[*src_idx];
                                        let catalog = &projected_stars[*tgt_idx];
                                        let (det_x, det_y) = detected.get_centroid();
                                        let dx = det_x - catalog.x();
                                        let dy = det_y - catalog.y();
                                        let error = (dx * dx + dy * dy).sqrt();
                                        (error, dx, dy)
                                    })
                                    .unwrap_or((f64::NAN, f64::NAN, f64::NAN))
                            } else {
                                (f64::NAN, f64::NAN, f64::NAN)
                            };

                        ExposureResult {
                            detected_count: magnitudes.len(),
                            brightest_magnitude: brightest_mag,
                            faintest_magnitude: faintest_mag,
                            alignment_error,
                            translation_x,
                            translation_y,
                            rotation_deg,
                            brightest_star_pixel_error,
                            brightest_star_dx,
                            brightest_star_dy,
                        }
                    }
                    Err(e) => {
                        warn!(
                            "ICP matching failed for satellite {}: {}",
                            satellite.description(),
                            e
                        );

                        ExposureResult {
                            detected_count: 0,
                            brightest_magnitude: f64::NAN,
                            faintest_magnitude: f64::NAN,
                            alignment_error: f64::NAN,
                            translation_x: f64::NAN,
                            translation_y: f64::NAN,
                            rotation_deg: f64::NAN,
                            brightest_star_pixel_error: f64::NAN,
                            brightest_star_dx: f64::NAN,
                            brightest_star_dy: f64::NAN,
                        }
                    }
                };

                // Store result for this exposure duration
                exposure_results.insert(*exposure_duration, result);
            } // End exposure loop

            // Create experiment result for this satellite with all exposure results
            let experiment_result = ExperimentResult {
                experiment_num: params.experiment_num,
                trial_num: params.trial_num,
                sensor_name: satellite.sensor.name.clone(),
                f_number: *f_number,
                coordinates: params.ra_dec,
                exposure_results,
                duration: experiment_start.elapsed(),
            };

            // Write result immediately to CSV
            if let Err(e) = params
                .common_args
                .csv_writer
                .write_result(&experiment_result, params.common_args.aperture_m)
            {
                warn!("Failed to write result to CSV: {e}");
            }

            results.push(experiment_result);
        } // End satellite loop
    } // End f-number loop

    results
}

/// Saves multiple image outputs from rendered star field data
///
/// This function creates and saves several image formats from the rendered image data:
/// 1. Regular PNG - Direct conversion of sensor data to 8-bit image format
/// 2. Histogram stretched PNG - Enhanced contrast version for better visibility of dim objects
/// 3. Overlay PNG - Detected stars marked with X markers and flux values
///
/// # Arguments
/// * `render_result` - Complete results from star field rendering, containing image data and metadata
/// * `sensor` - Sensor configuration used for image scaling
/// * `detected_stars` - Vector of detected star objects with position and flux information
/// * `output_path` - Directory where output files will be saved
/// * `prefix` - Filename prefix for all output files (typically includes experiment number and sensor name)
///
/// # Example Filenames
/// * `{prefix}_regular.png` - Direct visualization
/// * `{prefix}_stretched.png` - Histogram-stretched for better visibility
/// * `{prefix}_overlay.png` - Regular image with X markers at detected star positions
pub fn save_image_outputs(
    render_result: &RenderingResult,
    sensor: &SensorConfig,
    detected_stars: &[Box<dyn StellarSource>],
    output_path: &Path,
    prefix: &str,
    frame_writer: &FrameWriterHandle,
) {
    // Convert u16 image to u8 for saving (normalize by max bit depth value)
    let max_bit_value = (1 << (sensor.bit_depth as u32)) - 1;
    let u8_image = u16_to_u8_scaled(&render_result.quantized_image, max_bit_value);

    // Save the raw image
    let regular_path = output_path.join(format!("{prefix}_regular.png"));
    if let Err(e) = frame_writer.write_u8_frame(&u8_image, regular_path.clone(), FrameFormat::Png) {
        warn!(
            "Failed to async save image {}: {}",
            regular_path.display(),
            e
        );
    }

    // Create and save histogram stretched version
    let stretched_image = stretch_histogram(render_result.quantized_image.view(), 0.0, 50.0);

    // Convert stretched u16 image to u8 using auto-scaling for best contrast
    let img_flt = stretched_image.mapv(|x| x as f64);
    let normed = sigma_stretch(&img_flt, 5.0, Some(5));
    let u8_stretched = normed.mapv(|x| (x * 255.0).round() as u8);
    let stretched_path = output_path.join(format!("{prefix}_stretched.png"));

    if let Err(e) =
        frame_writer.write_u8_frame(&u8_stretched, stretched_path.clone(), FrameFormat::Png)
    {
        warn!(
            "Failed to async save stretched image {}: {}",
            stretched_path.display(),
            e
        );
    }

    // Use light blue (135, 206, 250) for X markers
    let vis_image = array2_to_gray_image(&u8_image);
    let dyn_image = DynamicImage::ImageLuma8(vis_image);

    // Mutate the detected stars into the shape needed for rendering
    let mut label_map = HashMap::new();

    detected_stars.iter().for_each(|detect| {
        let (x, y) = detect.get_centroid();
        label_map.insert(format!("{:.1}", detect.flux()), (y, x, 10.0));
    });

    let x_markers_image = draw_stars_with_x_markers(
        &dyn_image,
        &label_map,
        (135, 206, 250), // Light blue color
        1.0,             // Arm length factor (1.0 = full diameter)
    );

    let overlay_path = output_path.join(format!("{prefix}_overlay.png"));

    if let Err(e) =
        frame_writer.write_dynamic_image(x_markers_image, overlay_path.clone(), FrameFormat::Png)
    {
        warn!(
            "Failed to async save overlay image {}: {}",
            overlay_path.display(),
            e
        );
    }

    // Export FITS files
    // Split into two files for async writing
    let raw_fits_path = output_path.join(format!("{prefix}_data_raw.fits"));
    let electron_fits_path = output_path.join(format!("{prefix}_data_electron.fits"));

    if let Err(e) = frame_writer.write_frame(
        &render_result.quantized_image,
        raw_fits_path.clone(),
        FrameFormat::Fits,
    ) {
        warn!(
            "Failed to async save raw FITS {}: {}",
            raw_fits_path.display(),
            e
        );
    }
    if let Err(e) = frame_writer.write_f64_frame(
        &render_result.mean_electron_image(),
        electron_fits_path.clone(),
        FrameFormat::Fits,
    ) {
        warn!(
            "Failed to async save electron FITS {}: {}",
            electron_fits_path.display(),
            e
        );
    }
}

/// Display debug statistics including electron counts, noise, match distances and histogram
pub fn debug_stats(
    render_result: &RenderingResult,
    detected_stars: &[Box<dyn StellarSource>],
    projected_stars: &[StarInFrame],
    match_indices: &[(usize, usize)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Guard expensive debug calls for performance
    if !log::log_enabled!(log::Level::Debug) {
        return Ok(());
    }

    // Print some statistics about the rendered image
    debug!(
        "Total electrons in image: {:.2}e-",
        render_result.mean_electron_image().sum()
    );
    debug!(
        "Total noise in image: {:.2}e-",
        (&render_result.zodiacal_image + &render_result.sensor_noise_image).sum()
    );

    // Print ICP match distances
    for (src_idx, tgt_idx) in match_indices.iter() {
        let dete = &detected_stars[*src_idx];
        let star = &projected_stars[*tgt_idx];
        let (dete_x, dete_y) = dete.get_centroid();
        let distance = ((dete_x - star.x()).powf(2.0) + (dete_y - star.y()).powf(2.0)).sqrt();
        debug!("Matched star with distance {distance:.2}");
        debug!(
            "\tDetected X/Y: ({:.2}, {:.2}), Source X/Y: ({:.2}, {:.2})",
            dete_x,
            dete_y,
            star.x(),
            star.y()
        );
        debug!(
            "\tDetected flux: {:.2}, Source magnitude: {:.2}",
            dete.flux(),
            star.star.magnitude
        );
    }
    // Get statistics for binning (gross)
    let num_bins = 25;
    let electron_image = render_result.mean_electron_image();
    let electron_values: Vec<f64> = electron_image.iter().copied().collect();
    let electron_scan = MinMaxScan::new(&electron_values);
    let (min_val, max_val) = electron_scan.min_max().unwrap_or((0.0, 1.0));

    // Skip if all values are the same
    if (max_val - min_val).abs() < 1e-10 {
        debug!("  All pixel values are approximately: {min_val:.2}");
        return Ok(());
    }

    // Calculate some basic stats for display
    let total_pixels = electron_image.len();

    let mut full_hist = Histogram::new_equal_bins(min_val..max_val, num_bins)?;
    full_hist.add_all(electron_image.iter().copied());

    let full_config = HistogramConfig {
        title: Some(format!(
            "Electron Count Histogram (Full Range: {min_val:.2} - {max_val:.2}e-)"
        )),
        width: 80,
        height: None,
        bar_char: '#',
        show_percentage: true,
        show_counts: true,
        scale: Scale::Linear,
        show_empty_bins: true,
        max_bar_width: 50,
    };

    // Display basic statistics
    debug!("\nElectron Count Statistics:");
    debug!("  Total Pixels: {total_pixels}");
    debug!("  Min Value: {min_val:.2} electrons");
    debug!("  Max Value: {max_val:.2} electrons");

    // Print the histograms
    debug!("\n{}", full_hist.with_config(full_config).format()?);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::sensor::models::HWK4123;
    use crate::hardware::telescope::TelescopeConfig;
    use crate::units::{Length, LengthExt, Temperature, TemperatureExt};
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_csv_writer_creation_and_header() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("test.csv");

        let _writer = CsvWriter::new(csv_path.to_str().unwrap()).unwrap();

        // Check file was created with header
        let contents = fs::read_to_string(&csv_path).unwrap();
        assert!(contents.contains("experiment_num"));
        assert!(contents.contains("star_count"));
        assert!(contents.contains("brightest_mag"));
        assert!(contents.contains("translation_x"));
        assert!(contents.contains("rotation_deg"));
    }

    #[test]
    fn test_csv_writer_write_result() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("test_results.csv");

        let writer = Arc::new(CsvWriter::new(csv_path.to_str().unwrap()).unwrap());

        // Create a test result
        let mut exposure_results = HashMap::new();
        exposure_results.insert(
            Duration::from_millis(100),
            ExposureResult {
                detected_count: 5,
                brightest_magnitude: 3.5,
                faintest_magnitude: 8.2,
                alignment_error: 0.1,
                translation_x: 0.05,
                translation_y: -0.03,
                rotation_deg: 0.002,
                brightest_star_pixel_error: 0.15,
                brightest_star_dx: 0.1,
                brightest_star_dy: -0.05,
            },
        );

        let result = ExperimentResult {
            experiment_num: 1,
            trial_num: 0,
            sensor_name: "Test Sensor".to_string(),
            f_number: 10.0,
            coordinates: Equatorial::from_degrees(45.0, 30.0),
            exposure_results,
            duration: Duration::from_secs(2),
        };

        writer.write_result(&result, 0.2).unwrap();

        // Check CSV contents
        let contents = fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2); // Header + 1 data row
        assert!(lines[1].contains("1,0")); // experiment_num, trial_num
        assert!(lines[1].contains("Test Sensor"));
        assert!(lines[1].contains("100")); // exposure_ms
    }

    #[test]
    #[ignore] // See TODO.md: Simulator - Scene Processing - Fix catalog mock
    fn test_run_experiment_basic() {
        // Create minimal test setup
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("experiment.csv");

        // Create a simple telescope for testing
        let telescope = TelescopeConfig::new(
            "Test Telescope".to_string(),
            Length::from_meters(0.2), // 200mm aperture
            Length::from_meters(2.0), // 2m focal length (f/10)
            0.9,                      // transmission efficiency
        );
        let sensor = HWK4123.clone();
        let satellite =
            SatelliteConfig::new(telescope.clone(), sensor, Temperature::from_celsius(20.0));

        let common_args = ExperimentCommonArgs {
            exposures: vec![Duration::from_millis(100)],
            coordinates: crate::photometry::zodiacal::SolarAngularCoordinates::new(90.0, 30.0)
                .unwrap(),
            noise_multiple: 3.0,
            output_dir: temp_dir.path().to_str().unwrap().to_string(),
            save_images: false, // Don't save images in test
            icp_max_iterations: 10,
            icp_convergence_threshold: 0.001,
            star_finder: StarFinder::Naive,
            csv_writer: Arc::new(CsvWriter::new(csv_path.to_str().unwrap()).unwrap()),
            aperture_m: telescope.aperture.as_meters(),
        };

        let _params = ExperimentParams {
            experiment_num: 0,
            trial_num: 0,
            rng_seed: 42,
            ra_dec: Equatorial::from_degrees(56.75, 24.12), // Pleiades
            satellites: vec![satellite],
            f_numbers: vec![10.0],
            common_args,
        };

        // See TODO.md: Simulator - Scene Processing - Fix catalog mock
        // let catalog = TestCatalog { stars: vec![] };

        // Run experiment (disabled until catalog mock is fixed)
        // let results = run_experiment(&params, &catalog, 1.0);

        // Basic assertions
        // assert_eq!(results.len(), 1); // One satellite
        // assert_eq!(results[0].experiment_num, 0);
        // assert_eq!(results[0].f_number, 10.0);
        // assert!(!results[0].exposure_results.is_empty());

        // Check CSV was written
        // let csv_contents = fs::read_to_string(&csv_path).unwrap();
        // assert!(csv_contents.contains("experiment_num"));
        // assert!(csv_contents.lines().count() >= 2); // At least header + 1 row
    }
}

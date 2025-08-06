//! Telescope/scope view with star field simulation and rendering
//!
//! This example provides a comprehensive telescope simulation by combining
//! functionality from star_simulation.rs and star_field_renderer.rs, with additional features:
//!
//! 1. Loading stars from any catalog (binary, Hipparcos, or synthetic)
//! 2. Computing flux values based on telescope/sensor parameters
//! 3. Applying PSF (Point Spread Function) for realistic star appearance
//! 4. Precise sub-pixel star positioning for accurate centroids
//! 5. Adding realistic noise tied to exposure time and sensor characteristics
//! 6. Multiple output formats:
//!    - Raw image (normalized 8-bit PNG)
//!    - Histogram stretched image for better visibility (8-bit PNG)
//!    - Text file with detailed star list (positions, magnitudes, pixel coordinates)
//!
//! Usage:
//! ```
//! cargo run --release --bin sensor_shootout -- [OPTIONS]
//! ```
//!
//! See --help for detailed options.

use chrono::Local;
use clap::Parser;
use core::f64;
use image::DynamicImage;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{debug, info, warn};
use rayon::prelude::*;
use simulator::algo::{
    icp::{icp_match_objects, Locatable2d},
    MinMaxScan,
};
use simulator::hardware::sensor::models as sensor_models;
use simulator::hardware::SatelliteConfig;
use simulator::image_proc::airy::PixelScaledAiryDisk;
use simulator::image_proc::detection::{detect_stars_unified, StarFinder};
use simulator::image_proc::histogram_stretch::sigma_stretch;
use simulator::image_proc::image::array2_to_gray_image;
use simulator::image_proc::io::{write_typed_fits, FitsDataType};
use simulator::image_proc::render::{RenderingResult, StarInFrame};
use simulator::image_proc::{
    draw_stars_with_x_markers, save_u8_image, stretch_histogram, u16_to_u8_scaled, StarDetection,
};
use simulator::photometry::zodical::SolarAngularCoordinates;
use simulator::scene::Scene;
use simulator::shared_args::{RangeArg, SharedSimulationArgs};
use simulator::{star_math::field_diameter, SensorConfig};
use starfield::catalogs::{StarCatalog, StarData};
use starfield::framelib::random::RandomEquatorial;
use starfield::Equatorial;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use viz::histogram::{Histogram, HistogramConfig, Scale};

/// Thread-safe CSV writer
#[derive(Debug)]
struct CsvWriter {
    file: Arc<Mutex<File>>,
}

impl CsvWriter {
    fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        // Write header
        writeln!(file, "experiment_num,ra,dec,focal_length_m,sensor,exposure_ms,star_count,brightest_mag,faintest_mag,pixel_error,brightest_star_pixel_error")?;
        Ok(CsvWriter {
            file: Arc::new(Mutex::new(file)),
        })
    }

    fn write_result(
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
                "{},{:.6},{:.6},{:.2},{},{},{},{:.2},{:.2},{:.4},{:.4}",
                result.experiment_num,
                result.coordinates.ra_degrees(),
                result.coordinates.dec_degrees(),
                focal_length_m,
                result.sensor_name,
                exposure_ms,
                exposure_result.detected_count,
                exposure_result.brightest_magnitude,
                exposure_result.faintest_magnitude,
                exposure_result.alignment_error,
                exposure_result.brightest_star_pixel_error
            )?;
        }
        Ok(())
    }
}

/// Common arguments for experiments
#[derive(Debug)]
struct ExperimentCommonArgs {
    exposures: Vec<Duration>, // Multiple exposure durations
    coordinates: SolarAngularCoordinates,
    noise_multiple: f64,
    output_dir: String,
    save_images: bool,
    icp_max_iterations: usize,
    icp_convergence_threshold: f64,
    star_finder: StarFinder,
    csv_writer: Arc<CsvWriter>, // Thread-safe CSV writer
    aperture_m: f64,            // Needed for focal length calculation
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
struct ExperimentParams {
    experiment_num: u32,
    ra_dec: Equatorial,
    satellites: Vec<SatelliteConfig>,
    f_numbers: Vec<f64>, // Multiple f-numbers to test
    common_args: ExperimentCommonArgs,
}

/// Results from a single exposure test
#[derive(Debug, Clone)]
struct ExposureResult {
    detected_count: usize,
    brightest_magnitude: f64,
    faintest_magnitude: f64,
    alignment_error: f64,
    brightest_star_pixel_error: f64, // Distance from brightest star to its ICP-aligned position
}

/// Results from all exposures for one experiment/sensor combination
#[derive(Debug)]
struct ExperimentResult {
    experiment_num: u32,
    sensor_name: String,
    f_number: f64, // F-number used for this result
    coordinates: Equatorial,
    exposure_results: HashMap<Duration, ExposureResult>, // Key is exposure duration
    duration: Duration,                                  // Time taken for this experiment
}

/// Parse coordinates string in format "ra,dec" (degrees)
fn parse_ra_dec_coordinates(s: &str) -> Result<Equatorial, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err("Coordinates must be in format 'ra,dec' (degrees)".to_string());
    }

    let ra = parts[0]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid RA value".to_string())?;
    let dec = parts[1]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid Dec value".to_string())?;

    if !(0.0..360.0).contains(&ra) {
        return Err("RA must be in range [0, 360) degrees".to_string());
    }
    if !(-90.0..=90.0).contains(&dec) {
        return Err("Dec must be in range [-90, 90] degrees".to_string());
    }

    Ok(Equatorial::from_degrees(ra, dec))
}

/// Command line arguments for telescope view simulation
#[derive(Parser, Debug)]
#[command(
    name = "Telescope View Simulator",
    about = "Simulates telescope view with star field rendering",
    long_about = None
)]
struct Args {
    #[command(flatten)]
    shared: SharedSimulationArgs,

    /// Number of experiments to run
    #[arg(long, default_value_t = 100)]
    experiments: u32,

    /// Single-shot debug mode: specify RA,Dec coordinates in degrees (format: "ra,dec")
    /// Example: "56.75,24.12" points to the Pleiades cluster for easy visual comparison
    /// When specified, runs simulation only at this position for all sensors instead of random sampling
    #[arg(long, value_parser = parse_ra_dec_coordinates)]
    single_shot_debug: Option<Equatorial>,

    /// Output directory for experiment images and data
    #[arg(long, default_value = "experiment_output")]
    output_dir: String,

    /// Output CSV file for experiment log
    #[arg(long, default_value = "experiment_log.csv")]
    output_csv: String,

    /// Save image outputs (PNG, FITS files). Disabling this speeds up experiments significantly
    #[arg(long, action)]
    no_save_images: bool,

    /// Run experiments serially instead of in parallel
    #[arg(long, default_value_t = false)]
    serial: bool,

    /// Match FWHM sampling across all sensors using this value (pixels per FWHM).
    /// If not specified, uses the selected telescope (from --telescope) with each sensor
    #[arg(long)]
    match_pixel_sampling: Option<f64>,

    /// Exposure duration range in milliseconds (start:stop:step format)
    /// Example: --exposure_range_ms 10000:120000:10000 for 10s to 120s in 10s steps
    #[arg(long, default_value = "50:1000:50")]
    exposure_range_ms: RangeArg,

    /// Maximum iterations for ICP star matching algorithm
    #[arg(long, default_value_t = 20)]
    icp_max_iterations: usize,

    /// Convergence threshold for ICP star matching algorithm
    #[arg(long, default_value_t = 0.05)]
    icp_convergence_threshold: f64,

    /// Star detection algorithm to use (dao, iraf, naive)
    #[arg(long, default_value = "naive")]
    star_finder: StarFinder,

    /// F-number range to test (format: "start:stop:step" or "start:stop:count#")
    /// Examples: "8:16:2" tests f/8, f/10, f/12, f/14, f/16
    ///          "8:16:5#" tests 5 evenly spaced values from f/8 to f/16
    #[arg(long)]
    f_number_range: Option<RangeArg>,
}

/// Prints histogram of star magnitudes
///
/// # Arguments
/// * `stars` - Vector of stars to analyze
fn print_am_hist(stars: &[StarData]) {
    // Print histogram of star magnitudes
    if stars.is_empty() {
        warn!("No stars available to create histogram");
    } else {
        debug!("Creating histogram of star magnitudes...");
        debug!("Note that these stats include stars in the sensor circumcircle");
        let star_magnitudes: Vec<f64> = stars.iter().map(|star| star.magnitude).collect();

        // Create a magnitude histogram using the new specialized function
        // This automatically creates bins centered on integer magnitudes with 1.0 width
        let mag_hist = viz::histogram::create_magnitude_histogram(
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
fn run_experiment<T: StarCatalog>(
    params: &ExperimentParams,
    catalog: &T,
    max_fov: f64,
) -> Vec<ExperimentResult> {
    let experiment_start = Instant::now();
    let output_path = Path::new(&params.common_args.output_dir);

    debug!("Running experiment {}...", params.experiment_num);
    debug!(
        "  RA: {:.2}, Dec: {:.2}",
        params.ra_dec.ra_degrees(),
        params.ra_dec.dec_degrees()
    );
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
                satellite.sensor.name, f_number, satellite.temperature_c, satellite.wavelength_nm
            );

            let mut exposure_results = HashMap::new();

            // Create a modified satellite config with the new f-number
            let focal_length = satellite.telescope.aperture_m * f_number;
            let modified_telescope = satellite.telescope.with_focal_length(focal_length);
            let modified_satellite = SatelliteConfig::new(
                modified_telescope,
                satellite.sensor.clone(),
                satellite.temperature_c,
                satellite.wavelength_nm,
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

                // Render the scene
                let render_result = scene.render(exposure_duration);

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
                let scaled_airy_disk =
                    PixelScaledAiryDisk::with_fwhm(airy_disk_pixels, satellite.wavelength_nm);
                let detected_stars = match detect_stars_unified(
                    render_result.quantized_image.view(),
                    params.common_args.star_finder,
                    &scaled_airy_disk,
                    background_rms,
                    detection_sigma,
                ) {
                    // TODO: Is there a way to make StellarSource impl Locatable2d?
                    // This transform seems needless
                    Ok(stars) => stars
                        .into_iter()
                        .map(|star| {
                            // Convert boxed StellarSource to StarDetection-like structure
                            {
                                let (x, y) = star.get_centroid();
                                simulator::image_proc::StarDetection {
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
                        vec![] // Return empty vector instead of early return
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
                    );
                }

                // Now we take our detected stars and match them against the sources
                // Get projected stars from scene instead of render_result
                let projected_stars: Vec<StarInFrame> = scene.stars.clone();
                let result = match icp_match_objects::<StarDetection, StarInFrame>(
                    &detected_stars,
                    &projected_stars,
                    params.common_args.icp_max_iterations,
                    params.common_args.icp_convergence_threshold,
                ) {
                    Ok((matches, icp_result)) => {
                        // Debug statistics output
                        debug_stats(&render_result, &matches).unwrap();

                        let magnitudes: Vec<f64> =
                            matches.iter().map(|(_, s)| s.star.magnitude).collect();

                        let mag_scan = MinMaxScan::new(&magnitudes);
                        let (brightest_mag, faintest_mag) =
                            mag_scan.min_max().unwrap_or((f64::NAN, f64::NAN));

                        info!(
                            "Detected {} stars. Faintest magnitude: {:.2}",
                            magnitudes.len(),
                            faintest_mag
                        );

                        debug!("ICP match results:");
                        debug!("\tMatched stars: {}", matches.len());
                        debug!("\tTranslation: {:?}", icp_result.translation);
                        debug!("\tRotation: {:?}", icp_result.rotation);
                        debug!("\tScale: {:?}", icp_result.rotation_quat);

                        let alignment_error =
                            icp_result.translation.map(|disp| disp * disp).sum().sqrt();

                        // Find the brightest star (lowest magnitude) and calculate its pixel error
                        let brightest_star_pixel_error = if !matches.is_empty() {
                            matches
                                .iter()
                                .min_by(|(_, a), (_, b)| {
                                    a.star.magnitude.partial_cmp(&b.star.magnitude).unwrap()
                                })
                                .map(|(detected, catalog)| {
                                    let dx = detected.x() - catalog.x();
                                    let dy = detected.y() - catalog.y();
                                    (dx * dx + dy * dy).sqrt()
                                })
                                .unwrap_or(f64::NAN)
                        } else {
                            f64::NAN
                        };

                        ExposureResult {
                            detected_count: magnitudes.len(),
                            brightest_magnitude: brightest_mag,
                            faintest_magnitude: faintest_mag,
                            alignment_error,
                            brightest_star_pixel_error,
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
                            brightest_star_pixel_error: f64::NAN,
                        }
                    }
                };

                // Store result for this exposure duration
                exposure_results.insert(*exposure_duration, result);
            } // End exposure loop

            // Create experiment result for this satellite with all exposure results
            let experiment_result = ExperimentResult {
                experiment_num: params.experiment_num,
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
                warn!("Failed to write result to CSV: {}", e);
            }

            results.push(experiment_result);
        } // End satellite loop
    } // End f-number loop

    results
}

/// Main function for telescope view simulation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging from environment variables
    env_logger::init();

    // Parse command line arguments
    let args = Args::parse();

    // Start wallclock timer
    let wallclock_start = Instant::now();

    // Parse exposure durations from range in milliseconds
    let exposure_durations_vec = args.exposure_range_ms.to_duration_vec_ms();
    let exposure_durations: Vec<f64> = exposure_durations_vec
        .iter()
        .map(|d| d.as_secs_f64())
        .collect();
    info!(
        "Using {} exposure durations: {:?}s",
        exposure_durations.len(),
        exposure_durations
    );

    // Use only IMX455 and HWK4123 sensors
    let selected_sensors = [&*sensor_models::IMX455, &*sensor_models::HWK4123];

    // Get telescope from shared args
    let selected_telescope = args.shared.telescope.to_config();
    info!(
        "Using telescope: {} (aperture: {:.2}m, f/{:.1})",
        selected_telescope.name,
        selected_telescope.aperture_m,
        selected_telescope.f_number()
    );

    // Parse f-number range
    let f_numbers: Vec<f64> = if let Some(f_range) = args.f_number_range {
        let (start, stop, step) = f_range.as_tuple();
        let mut values = Vec::new();
        let mut current = start;
        while current <= stop {
            values.push(current);
            current += step;
        }
        values
    } else {
        // Default to current telescope f-number
        vec![selected_telescope.f_number()]
    };

    info!("Using f-numbers: {:?}", f_numbers);

    // Use base telescope for satellite configurations (will be modified per f-number in experiments)
    let base_telescope = selected_telescope.clone();

    // Create satellite configurations directly from sensors and base telescope
    let satellites: Vec<SatelliteConfig> = if let Some(pixel_sampling) = args.match_pixel_sampling {
        // Use with_fwhm_sampling() to match pixel sampling across all sensors
        selected_sensors
            .iter()
            .map(|sensor| {
                // Create base satellite with (potentially modified) telescope
                let base_satellite = SatelliteConfig::new(
                    base_telescope.clone(),
                    (*sensor).clone(),
                    args.shared.temperature,
                    args.shared.wavelength,
                );

                // Adjust to match the desired pixel sampling
                base_satellite.with_fwhm_sampling(pixel_sampling)
            })
            .collect()
    } else {
        // Use base telescope with each sensor (no resampling)
        selected_sensors
            .iter()
            .map(|sensor| {
                SatelliteConfig::new(
                    base_telescope.clone(),
                    (*sensor).clone(),
                    args.shared.temperature,
                    args.shared.wavelength,
                )
            })
            .collect()
    };

    // Compute the maximal FOV using smallest f-number (largest FOV)
    let min_f_number = f_numbers.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut max_fov = 0.0;

    for satellite in satellites.iter() {
        // Create telescope with smallest f-number for FOV calculation
        let focal_length = satellite.telescope.aperture_m * min_f_number;
        let telescope_min_f = satellite.telescope.with_focal_length(focal_length);
        let fov_deg = field_diameter(&telescope_min_f, &satellite.sensor);

        info!(
            "Satellite: {}, FOV at f/{:.1}: {:.4}°",
            satellite.sensor.name, min_f_number, fov_deg
        );
        if fov_deg > max_fov {
            max_fov = fov_deg;
        }
    }

    // Load the catalog....this requires some serious RAM
    let catalog = args.shared.load_catalog().expect("Could not load catalog?");
    info!("Loaded catalog with {} stars", catalog.len());

    // Generate timestamp for file/directory naming
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();

    // Add timestamp to output directory
    let output_dir_with_timestamp = format!("{}_{}", args.output_dir, timestamp);

    // Add timestamp to CSV filename only if default is being used (no override specified)
    let output_csv_path = if args.output_csv == "experiment_log.csv" {
        format!("experiment_log_{timestamp}.csv")
    } else {
        args.output_csv.clone()
    };

    // Ensure the output directory exists
    let output_path = Path::new(&output_dir_with_timestamp);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path).expect("Failed to create output directory");
    }

    // Create CSV writer
    let csv_writer = Arc::new(CsvWriter::new(&output_csv_path)?);

    // Create common experiment arguments
    let common_args = ExperimentCommonArgs {
        exposures: exposure_durations_vec.clone(),
        coordinates: args.shared.coordinates,
        noise_multiple: args.shared.noise_multiple,
        output_dir: output_dir_with_timestamp.clone(),
        save_images: !args.no_save_images,
        icp_max_iterations: args.icp_max_iterations,
        icp_convergence_threshold: args.icp_convergence_threshold,
        star_finder: args.star_finder,
        csv_writer,
        aperture_m: base_telescope.aperture_m,
    };

    // Build all experiment parameters upfront
    info!("Setting up experiments...");
    let mut all_experiments = Vec::new();

    // Create randomizer with fixed seed for reproducible results
    let mut randomizer = RandomEquatorial::with_seed(42);

    for i in 0..args.experiments {
        // Generate coordinates based on mode
        let ra_dec = if let Some(fixed_coords) = args.single_shot_debug {
            // Use fixed coordinates for single-shot debug mode
            fixed_coords
        } else {
            // Generate random RA/Dec coordinates for normal operation
            randomizer.next().unwrap()
        };

        // Create one experiment param per sky pointing (stars computed in run_experiment)
        let params = ExperimentParams {
            experiment_num: i,
            ra_dec,
            satellites: satellites.clone(),
            f_numbers: f_numbers.clone(),
            common_args: common_args.clone(),
        };
        all_experiments.push(params);

        if args.single_shot_debug.is_some() {
            // If in single-shot debug mode, only run one set of experiments
            break;
        }
    }

    info!(
        "Running {} experiments (sky pointings) {}...",
        all_experiments.len(),
        if args.serial {
            "serially"
        } else {
            "in parallel"
        }
    );

    // Setup progress tracking
    let multi_progress = MultiProgress::new();
    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ ");

    // Create progress bar
    let pb = multi_progress.add(ProgressBar::new(all_experiments.len() as u64));
    pb.set_style(progress_style);
    pb.set_message("Running experiments");

    // Run experiments with progress tracking (parallel or serial based on flag)
    let experiment_results: Vec<Vec<ExperimentResult>> = if args.serial {
        all_experiments
            .iter()
            .map(|params| run_experiment(params, &catalog, max_fov))
            .inspect(|_| {
                pb.inc(1);
            })
            .collect()
    } else {
        all_experiments
            .par_iter()
            .map(|params| run_experiment(params, &catalog, max_fov))
            .inspect(|_| {
                pb.inc(1);
            })
            .collect()
    };

    // Flatten results from Vec<Vec<ExperimentResult>> to Vec<ExperimentResult>
    let flattened_results: Vec<ExperimentResult> =
        experiment_results.into_iter().flatten().collect();

    pb.finish_with_message("Experiments complete!");

    // Process results
    info!("Processing results...");

    // Results are written on-the-fly during experiments
    info!("Results written to CSV file: {}", output_csv_path);

    // Calculate and report timing statistics
    let wallclock_duration = wallclock_start.elapsed();

    // Calculate total compute time (sum of all experiment durations)
    let total_compute_time: Duration = flattened_results.iter().map(|r| r.duration).sum();

    // Calculate average experiment time
    let avg_experiment_time = if flattened_results.is_empty() {
        Duration::from_secs(0)
    } else {
        total_compute_time / flattened_results.len() as u32
    };

    // Print timing report
    info!("==================== TIMING REPORT ====================");
    info!("Total experiments run: {}", all_experiments.len());
    info!("Total sensor configurations: {}", satellites.len());
    info!(
        "Total experiment-sensor combinations: {}",
        flattened_results.len()
    );
    info!(
        "Execution mode: {}",
        if args.serial { "Serial" } else { "Parallel" }
    );
    info!("");
    info!("Wallclock time: {:.2}s", wallclock_duration.as_secs_f64());
    info!(
        "Total compute time: {:.2}s",
        total_compute_time.as_secs_f64()
    );
    info!(
        "Average time per experiment: {:.2}s",
        avg_experiment_time.as_secs_f64()
    );

    if !args.serial {
        let speedup = total_compute_time.as_secs_f64() / wallclock_duration.as_secs_f64();
        info!("Parallel speedup: {:.2}x", speedup);
        info!(
            "Effective CPU utilization: {:.1}%",
            speedup * 100.0 / num_cpus::get() as f64
        );
    }
    info!("======================================================");

    Ok(())
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
fn save_image_outputs(
    render_result: &RenderingResult,
    sensor: &SensorConfig,
    detected_stars: &[StarDetection],
    output_path: &Path,
    prefix: &str,
) {
    // Convert u16 image to u8 for saving (normalize by max bit depth value)
    let max_bit_value = (1 << (sensor.bit_depth as u32)) - 1;
    let u8_image = u16_to_u8_scaled(&render_result.quantized_image, max_bit_value);

    // Save the raw image
    let regular_path = output_path.join(format!("{prefix}_regular.png"));
    save_u8_image(&u8_image, &regular_path).expect("Failed to save image");

    // Create and save histogram stretched version
    let stretched_image = stretch_histogram(render_result.quantized_image.view(), 0.0, 50.0);

    // Convert stretched u16 image to u8 using auto-scaling for best contrast
    let img_flt = stretched_image.mapv(|x| x as f64);
    let normed = sigma_stretch(&img_flt, 5.0, Some(5));
    let u8_stretched = normed.mapv(|x| (x * 255.0).round() as u8);
    let stretched_path = output_path.join(format!("{prefix}_stretched.png"));
    save_u8_image(&u8_stretched, &stretched_path).expect("Failed to save stretched image");

    // Use light blue (135, 206, 250) for X markers
    let vis_image = array2_to_gray_image(&u8_image);
    let dyn_image = DynamicImage::ImageLuma8(vis_image);

    // Mutate the detected stars into the shape needed for rendering
    let mut label_map = HashMap::new();

    detected_stars.iter().for_each(|detect| {
        label_map.insert(format!("{:.1}", detect.flux), (detect.y, detect.x, 10.0));
    });

    let x_markers_image = draw_stars_with_x_markers(
        &dyn_image,
        &label_map,
        (135, 206, 250), // Light blue color
        1.0,             // Arm length factor (1.0 = full diameter)
    );

    let overlay_path = output_path.join(format!("{prefix}_overlay.png"));
    x_markers_image
        .save(&overlay_path)
        .expect("Failed to save image with X markers");

    // Export FITS files for both regular image and electron image
    let mut fits_data = HashMap::new();

    // Keep u16 image in native format (no upcasting)
    fits_data.insert(
        "IMAGE".to_string(),
        FitsDataType::UInt16(render_result.quantized_image.clone()),
    );

    // Add electron image as f64
    fits_data.insert(
        "ELECTRON_IMAGE".to_string(),
        FitsDataType::Float64(render_result.mean_electron_image()),
    );

    // Save FITS file with both datasets
    let fits_path = output_path.join(format!("{prefix}_data.fits"));
    write_typed_fits(&fits_data, &fits_path).expect("Failed to save FITS file");
}

/// Display debug statistics including electron counts, noise, match distances and histogram
fn debug_stats(
    render_result: &RenderingResult,
    matches: &[(StarDetection, StarInFrame)],
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
    for (dete, star) in matches.iter() {
        let distance = ((dete.x() - star.x()).powf(2.0) + (dete.y() - star.y()).powf(2.0)).sqrt();
        debug!("Matched star with distance {:.2}", distance);
        debug!(
            "\tDetected X/Y: ({:.2}, {:.2}), Source X/Y: ({:.2}, {:.2})",
            dete.x(),
            dete.y(),
            star.x(),
            star.y()
        );
        debug!(
            "\tDetected flux: {:.2}, Source magnitude: {:.2}",
            dete.flux, star.star.magnitude
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
        debug!("  All pixel values are approximately: {:.2}", min_val);
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
    debug!("  Total Pixels: {}", total_pixels);
    debug!("  Min Value: {:.2} electrons", min_val);
    debug!("  Max Value: {:.2} electrons", max_val);

    // Print the histograms
    debug!("\n{}", full_hist.with_config(full_config).format()?);

    Ok(())
}

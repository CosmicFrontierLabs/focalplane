//! Sensor floor estimation for star detection and centroiding
//!
//! This tool simulates star detection and centroiding accuracy across different:
//! - Sensor models
//! - Star magnitudes
//! - PSF (Point Spread Function) sizes
//!
//! It conducts experiments by:
//! 1. Simulating stars with known positions in images
//! 2. Adding realistic sensor noise based on exposure parameters
//! 3. Running detection algorithms
//! 4. Measuring detection rates and centroid accuracy
//!
//! Results are presented as matrices showing detection rates and position errors
//! across different star magnitudes and optical parameters.
//!
//! Usage:
//! ```
//! cargo run --bin sensor_floor_est -- [OPTIONS]
//! ```
//!
//! See --help for detailed options.

use clap::Parser;
use core::f64;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::{Array1, Array3};
use rayon::prelude::*;
use simulator::hardware::sensor::models::ALL_SENSORS;
use simulator::hardware::SatelliteConfig;
use simulator::image_proc::detection::{detect_stars_unified, StarDetection, StarFinder};
use simulator::image_proc::render::StarInFrame;
use simulator::photometry::zodical::SolarAngularCoordinates;
use simulator::shared_args::{RangeArg, SharedSimulationArgs};
use simulator::star_data_to_electrons;
use simulator::Scene;
use starfield::catalogs::StarData;
use starfield::Equatorial;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

/// Command line arguments for sensor floor estimation
#[derive(Parser, Debug)]
#[command(
    name = "Sensor Floor Estimator",
    about = "Simulates star detection and centroiding accuracy across different sensors",
    long_about = None
)]
struct Args {
    #[command(flatten)]
    shared: SharedSimulationArgs,

    /// Number of experiments to run per configuration
    #[arg(long, default_value_t = 1000)]
    experiments: u32,

    /// Output CSV file path
    #[arg(long, default_value = "sensor_floor_results.csv")]
    output_csv: String,

    /// Run experiments serially instead of in parallel
    #[arg(long, default_value_t = false)]
    serial: bool,

    /// Domain size for test images (width and height in pixels)
    #[arg(long, default_value_t = 128)]
    domain: usize,

    /// Test multiple exposure durations instead of just the shared exposure setting
    #[arg(long, default_value_t = true)]
    test_exposures: bool,

    /// PSF disk size range in Airy disk FWHM units (format: start:stop:step)
    #[arg(long, default_value = "1.5:2.1:0.1")]
    disks: RangeArg,

    /// Star magnitude range (format: start:stop:step)
    #[arg(long, default_value = "12.0:18.1:0.1")]
    mags: RangeArg,

    /// Exposure duration range in milliseconds (format: start:stop:step)
    #[arg(long, default_value = "100:1050:100")]
    exposures: RangeArg,

    /// Star detection algorithm to use (dao, iraf, naive)
    #[arg(long, default_value = "naive")]
    star_finder: StarFinder,
}

/// Parameters for a single experiment
#[derive(Clone)]
struct ExperimentParams {
    /// Domain size (image dimensions)
    domain: usize,
    /// Satellite configuration (telescope, sensor, temperature, wavelength)
    satellite: SatelliteConfig,
    /// Exposure duration
    exposure: Duration,
    /// Star magnitude
    mag: f64,
    /// Noise floor multiplier for detection threshold
    noise_floor_multiplier: f64,
    /// Indices for result matrix (disk_idx, exposure_idx, mag_idx)
    indices: (usize, usize, usize),
    /// Number of times to run the experiment
    experiment_count: u32,
    /// Solar coordinates for zodiacal background
    coordinates: SolarAngularCoordinates,
    /// Star detection algorithm to use
    star_finder: StarFinder,
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

        // Calculate electrons based on magnitude
        let flux = star_data_to_electrons(
            &star_data,
            &self.exposure,
            &self.satellite.telescope,
            &self.satellite.sensor,
        );

        StarInFrame {
            star: star_data,
            x: xpos,
            y: ypos,
            flux,
        }
    }
}

struct ExperimentResults {
    pub params: ExperimentParams,
    xy_err: Vec<(f64, f64)>,
    spurious_detections: usize,
    multiple_detections: usize,
    no_detections: usize,
}

impl ExperimentResults {
    /// Calculate the RMS error from collected x/y errors
    fn rms_error(&self) -> f64 {
        let mut sum = 0.0;

        for &(x, y) in &self.xy_err {
            sum += (x * x + y * y).sqrt();
        }

        sum /= self.xy_err.len() as f64;
        sum
    }

    /// Get the detection rate based on number of accumulated errors vs experiments
    fn detection_rate(&self) -> f64 {
        let total_experiments = self.params.experiment_count;
        if total_experiments == 0 {
            return f64::NAN;
        }
        self.xy_err.len() as f64 / total_experiments as f64
    }

    fn rms_error_radians(&self) -> f64 {
        let pix_err = self.rms_error();
        let err_m = self.params.satellite.sensor.pixel_size_um * pix_err / 1_000_000.0;
        (err_m / self.params.satellite.telescope.focal_length_m).tan()
    }

    fn rms_err_mas(&self) -> f64 {
        let _pix_err = self.rms_error();
        self.rms_error_radians() * (648_000_000.0 / PI)
    }

    fn false_positive_rate(&self) -> f64 {
        let total_experiments = self.params.experiment_count;
        (self.spurious_detections + self.multiple_detections) as f64 / total_experiments as f64
    }

    fn spurious_rate(&self) -> f64 {
        let total_experiments = self.params.experiment_count;
        self.spurious_detections as f64 / total_experiments as f64
    }
}

/// Run a single experiment and return the error (distance) or NaN if not detected, and detection rate
fn run_single_experiment(params: &ExperimentParams) -> ExperimentResults {
    let half_d = params.domain as f64 / 2.0;

    // Run the experiment multiple times and average the results
    let mut xy_err: Vec<(f64, f64)> = Vec::new();
    let mut spurious_detections = 0;
    let mut multiple_detections = 0;
    let mut no_detections = 0;

    for _ in 0..params.experiment_count {
        // Generate random position in the center area
        let xpos = rand::random::<f64>() * half_d + half_d;
        let ypos = rand::random::<f64>() * half_d + half_d;

        // Create star at the position with correct flux
        let star = params.star_at_pos(xpos, ypos);

        // Create scene with single star
        let scene = Scene::from_stars(
            params.satellite.clone(),
            vec![star],
            Equatorial::from_degrees(0.0, 0.0), // Dummy pointing (not used for pre-positioned stars)
            params.exposure,
            params.coordinates,
        );

        // Render the scene
        let render_result = scene.render();

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

/// Main function for sensor floor estimation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging from environment variables
    env_logger::init();

    // Parse command line arguments
    let args = Args::parse();

    // Use domain size from CLI arguments
    let domain = args.domain;

    // Create satellite configurations with sensors sized to domain
    let telescope_config = args.shared.telescope.to_config().clone();
    let all_satellites: Vec<SatelliteConfig> = ALL_SENSORS
        .iter()
        .map(|sensor| {
            let sized_sensor = sensor.with_dimensions(args.domain, args.domain);
            SatelliteConfig::new(
                telescope_config.clone(),
                sized_sensor,
                args.shared.temperature,
                args.shared.wavelength,
            )
        })
        .collect();

    // PSF disk sizes to test (in Airy disk FWHM units) - from CLI args
    let (disk_start, disk_stop, disk_step) = args.disks.as_tuple();
    let disks = Array1::range(disk_start, disk_stop, disk_step);

    // Star magnitudes to test - from CLI args
    let (mag_start, mag_stop, mag_step) = args.mags.as_tuple();
    let mags = Array1::range(mag_start, mag_stop, mag_step);

    // Exposure durations to test
    let exposures = if args.test_exposures {
        // Generate exposure durations from CLI range (in milliseconds)
        let (exp_start, exp_stop, exp_step) = args.exposures.as_tuple();
        let exp_range = Array1::range(exp_start, exp_stop, exp_step);
        exp_range
            .iter()
            .map(|&ms| Duration::from_millis(ms as u64))
            .collect()
    } else {
        vec![args.shared.exposure.0]
    };

    println!("Setting up experiments...");

    // Store results for each sensor (3D: disk, exposure, mag)
    let mut sensor_results: HashMap<String, Array3<f64>> = HashMap::new();
    let mut sensor_pixel_results: HashMap<String, Array3<f64>> = HashMap::new();

    // Build a vector of all experiments to run
    let mut all_experiments = Vec::new();

    // For each satellite, initialize the results arrays (3D: disk, exposure, mag)
    for satellite in &all_satellites {
        let sensor_results_array =
            Array3::<f64>::from_elem((disks.len(), exposures.len(), mags.len()), f64::NAN);
        sensor_results.insert(satellite.sensor.name.clone(), sensor_results_array);

        let sensor_pixel_results_array =
            Array3::<f64>::from_elem((disks.len(), exposures.len(), mags.len()), f64::NAN);
        sensor_pixel_results.insert(satellite.sensor.name.clone(), sensor_pixel_results_array);

        println!(
            "Preparing experiments for satellite: {}",
            satellite.sensor.name
        );

        // Create a vector of experiment parameters for this satellite
        for (disk_idx, disk) in disks.iter().enumerate() {
            for (exposure_idx, exposure) in exposures.iter().enumerate() {
                for (mag_idx, mag) in mags.iter().enumerate() {
                    // Adjust satellite to have the desired FWHM sampling
                    let adjusted_satellite = satellite.with_fwhm_sampling(*disk);

                    // Create experiment parameters
                    let params = ExperimentParams {
                        domain,
                        satellite: adjusted_satellite,
                        exposure: *exposure,
                        mag: *mag,
                        noise_floor_multiplier: args.shared.noise_multiple,
                        indices: (disk_idx, exposure_idx, mag_idx),
                        experiment_count: args.experiments,
                        coordinates: args.shared.coordinates,
                        star_finder: args.star_finder,
                    };

                    all_experiments.push(params);
                }
            }
        }
    }

    println!(
        "Running {} experiments {}...",
        all_experiments.len(),
        if args.serial {
            "serially"
        } else {
            "in parallel"
        }
    );

    // Create another hashmap to store detection rates (3D: disk, exposure, mag)
    let mut detection_rates: HashMap<String, Array3<f64>> = HashMap::new();

    // Create hashmap to store spurious detection rates
    let mut spurious_rates: HashMap<String, Array3<f64>> = HashMap::new();

    // Initialize detection rates arrays
    for satellite in &all_satellites {
        let detection_rate_array =
            Array3::<f64>::from_elem((disks.len(), exposures.len(), mags.len()), 0.0);
        detection_rates.insert(satellite.sensor.name.clone(), detection_rate_array);

        let spurious_rate_array =
            Array3::<f64>::from_elem((disks.len(), exposures.len(), mags.len()), 0.0);
        spurious_rates.insert(satellite.sensor.name.clone(), spurious_rate_array);
    }

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
    let results: Vec<_> = if args.serial {
        all_experiments
            .iter()
            .map(|params| {
                pb.inc(1);
                run_single_experiment(params)
            })
            .collect()
    } else {
        all_experiments
            .par_iter()
            .map(|params| {
                pb.inc(1);
                run_single_experiment(params)
            })
            .collect()
    };

    pb.finish_with_message("Experiments complete!");

    // Process results
    println!("Processing results...");
    for result in results {
        let sensor_name = result.params.satellite.sensor.name.clone();
        let detection_rate = result.detection_rate();
        let (disk_idx, exposure_idx, mag_idx) = result.params.indices;

        // Log false positive statistics
        if result.spurious_detections > 0 || result.multiple_detections > 0 {
            log::info!("Algorithm: {:?}, Sensor: {}, Spurious: {}, Multiple: {}, No Detection: {}, False Positive Rate: {:.1}%",
                      result.params.star_finder,
                      sensor_name,
                      result.spurious_detections,
                      result.multiple_detections,
                      result.no_detections,
                      result.false_positive_rate() * 100.0);
        }

        if let Some(sensor_array) = sensor_results.get_mut(&sensor_name) {
            sensor_array[[disk_idx, exposure_idx, mag_idx]] = result.rms_err_mas();
        }

        if let Some(sensor_pixel_array) = sensor_pixel_results.get_mut(&sensor_name) {
            sensor_pixel_array[[disk_idx, exposure_idx, mag_idx]] = result.rms_error();
        }

        if let Some(detection_array) = detection_rates.get_mut(&sensor_name) {
            detection_array[[disk_idx, exposure_idx, mag_idx]] = detection_rate;
        }

        if let Some(spurious_array) = spurious_rates.get_mut(&sensor_name) {
            spurious_array[[disk_idx, exposure_idx, mag_idx]] = result.spurious_rate();
        }
    }

    // Print summary of results
    println!("\n===== Results Summary =====");

    // Open CSV file for writing
    let csv_path = Path::new(&args.output_csv);
    let mut csv_file = File::create(csv_path)
        .unwrap_or_else(|_| panic!("Failed to create CSV file: {}", args.output_csv));

    println!("Writing results to CSV file: {}", args.output_csv);

    // Write CSV header
    writeln!(csv_file, "Sensor Floor Estimation Results").unwrap();
    writeln!(csv_file, "Parameters:").unwrap();
    writeln!(csv_file, "Exposure: {} seconds", args.shared.exposure).unwrap();
    writeln!(
        csv_file,
        "Noise Floor Multiplier: {}",
        args.shared.noise_multiple
    )
    .unwrap();
    writeln!(csv_file, "Wavelength: {} nm", args.shared.wavelength).unwrap();
    writeln!(csv_file, "Telescope: {}", args.shared.telescope).unwrap();
    writeln!(
        csv_file,
        "Aperture diameter: {} m",
        telescope_config.aperture_m
    )
    .unwrap();
    writeln!(
        csv_file,
        "Experiments per configuration: {}",
        args.experiments
    )
    .unwrap();
    writeln!(csv_file, "PSF Disk Range (FWHM units): {}", args.disks).unwrap();
    writeln!(csv_file, "Star Magnitude Range: {}", args.mags).unwrap();
    writeln!(csv_file, "Exposure Range (ms): {}", args.exposures).unwrap();
    writeln!(
        csv_file,
        "Domain Size: {}x{} pixels",
        args.domain, args.domain
    )
    .unwrap();
    writeln!(csv_file).unwrap();

    // Sort the sensor names for consistent output
    let mut sensors_ordered: Vec<_> = all_satellites
        .iter()
        .map(|satellite| satellite.sensor.name.clone())
        .collect();

    sensors_ordered.sort();

    for sensor_name in sensors_ordered {
        let results = sensor_results.get(&sensor_name).unwrap();
        let pixel_results = sensor_pixel_results.get(&sensor_name).unwrap();
        let detection_rate_array = detection_rates.get(&sensor_name).unwrap();
        let spurious_rate_array = spurious_rates.get(&sensor_name).unwrap();

        println!("\n==== Sensor: {} ====", sensor_name);

        // Write sensor name to CSV
        writeln!(csv_file, "SENSOR: {}", sensor_name).unwrap();
        writeln!(csv_file).unwrap();

        // Detection Rate Matrix - Console output (simplified, just show first exposure)
        println!(
            "  Detection Rate Matrix (% of attempts that detected a star, first exposure only):"
        );

        // Print header row with magnitude values
        print!("  Q\\Mag |");
        for mag in &mags {
            print!(" {:.2} |", mag);
        }
        println!();

        // Print separator
        print!("  ---------|");
        for _ in &mags {
            print!("------|");
        }
        println!();

        // Print first exposure only for console
        for (disk_idx, disk) in disks.iter().enumerate() {
            print!("  {:.1}      |", disk);
            for mag_idx in 0..mags.len() {
                let rate = detection_rate_array[[disk_idx, 0, mag_idx]]; // First exposure only
                print!(" {:3.0}% |", rate * 100.0);
            }
            println!();
        }

        // CSV Output - Detection Rate Matrix with disk and exposure unrolled
        writeln!(csv_file, "Detection Rate Matrix (%)").unwrap();

        // CSV header: Disk, Exposure, then magnitude columns
        write!(csv_file, "Q_Value,Exposure_ms,").unwrap();
        for mag in &mags {
            write!(csv_file, "{:.2},", mag).unwrap();
        }
        writeln!(csv_file).unwrap();

        // Write all combinations of exposure and disk (exposure grouped together)
        for (exposure_idx, exposure) in exposures.iter().enumerate() {
            for (disk_idx, disk) in disks.iter().enumerate() {
                write!(csv_file, "{:.2},{},", disk, exposure.as_millis()).unwrap();
                for mag_idx in 0..mags.len() {
                    let rate = detection_rate_array[[disk_idx, exposure_idx, mag_idx]];
                    write!(csv_file, "{:.1},", rate * 100.0).unwrap();
                }
                writeln!(csv_file).unwrap();
            }
        }
        writeln!(csv_file).unwrap();

        // Mean Position Error Matrix (mas)
        writeln!(csv_file, "Mean Position Error Matrix (milliarcseconds)").unwrap();

        // CSV header: Disk, Exposure, then magnitude columns
        write!(csv_file, "Q_Value,Exposure_ms,").unwrap();
        for mag in &mags {
            write!(csv_file, "{:.2},", mag).unwrap();
        }
        writeln!(csv_file).unwrap();

        // Write all combinations of exposure and disk (exposure grouped together)
        for (exposure_idx, exposure) in exposures.iter().enumerate() {
            for (disk_idx, disk) in disks.iter().enumerate() {
                write!(csv_file, "{:.2},{},", disk, exposure.as_millis()).unwrap();
                for mag_idx in 0..mags.len() {
                    let err = results[[disk_idx, exposure_idx, mag_idx]];
                    if err.is_nan() {
                        write!(csv_file, ",").unwrap(); // Empty cell for NaN
                    } else {
                        write!(csv_file, "{:.2},", err).unwrap();
                    }
                }
                writeln!(csv_file).unwrap();
            }
        }
        writeln!(csv_file).unwrap();

        // RMS Position Error Matrix (pixels)
        writeln!(csv_file, "RMS Position Error Matrix (pixels)").unwrap();

        // CSV header: Disk, Exposure, then magnitude columns
        write!(csv_file, "Q_Value,Exposure_ms,").unwrap();
        for mag in &mags {
            write!(csv_file, "{:.2},", mag).unwrap();
        }
        writeln!(csv_file).unwrap();

        // Write all combinations of exposure and disk (exposure grouped together)
        for (exposure_idx, exposure) in exposures.iter().enumerate() {
            for (disk_idx, disk) in disks.iter().enumerate() {
                write!(csv_file, "{:.2},{},", disk, exposure.as_millis()).unwrap();
                for mag_idx in 0..mags.len() {
                    let err = pixel_results[[disk_idx, exposure_idx, mag_idx]];
                    if err.is_nan() {
                        write!(csv_file, ",").unwrap(); // Empty cell for NaN
                    } else {
                        write!(csv_file, "{:.2},", err).unwrap();
                    }
                }
                writeln!(csv_file).unwrap();
            }
        }
        writeln!(csv_file).unwrap();

        // Spurious Detection Rate Matrix (%)
        writeln!(csv_file, "Spurious Detection Rate Matrix (%)").unwrap();

        // CSV header: Disk, Exposure, then magnitude columns
        write!(csv_file, "Q_Value,Exposure_ms,").unwrap();
        for mag in &mags {
            write!(csv_file, "{:.2},", mag).unwrap();
        }
        writeln!(csv_file).unwrap();

        // Write all combinations of exposure and disk (exposure grouped together)
        for (exposure_idx, exposure) in exposures.iter().enumerate() {
            for (disk_idx, disk) in disks.iter().enumerate() {
                write!(csv_file, "{:.2},{},", disk, exposure.as_millis()).unwrap();
                for mag_idx in 0..mags.len() {
                    let rate = spurious_rate_array[[disk_idx, exposure_idx, mag_idx]];
                    write!(csv_file, "{:.2},", rate * 100.0).unwrap();
                }
                writeln!(csv_file).unwrap();
            }
        }
        writeln!(csv_file).unwrap();

        // Add separator between sensors
        writeln!(csv_file).unwrap();
        writeln!(csv_file, "-----------------------------------------------").unwrap();
        writeln!(csv_file).unwrap();

        println!();
    }

    println!("CSV output completed: {}", args.output_csv);

    Ok(())
}

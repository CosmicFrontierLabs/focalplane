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
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use simulator::hardware::sensor::models as sensor_models;
use simulator::hardware::telescope::models::DEMO_50CM;
use simulator::hardware::telescope::TelescopeConfig;
use simulator::image_proc::generate_sensor_noise;
use simulator::image_proc::render::{
    add_stars_to_image, approx_airy_pixels, quantize_image, StarInFrame,
};
use simulator::image_proc::segment::do_detections;
use simulator::photometry::{zodical::SolarAngularCoordinates, ZodicalLight};
use simulator::shared_args::SharedSimulationArgs;
use simulator::{magnitude_to_electrons, SensorConfig};
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

    /// Detection threshold multiplier above noise floor
    #[arg(long, default_value_t = 5.0)]
    noise_floor: f64,

    /// Number of experiments to run per configuration
    #[arg(long, default_value_t = 1000)]
    experiments: u32,

    /// Output CSV file path
    #[arg(long, default_value = "sensor_floor_results.csv")]
    output_csv: String,

    /// Run experiments serially instead of in parallel
    #[arg(long, default_value_t = false)]
    serial: bool,
}

/// Parameters for a single experiment
#[derive(Clone)]
struct ExperimentParams {
    /// Domain size (image dimensions)
    domain: usize,
    /// Sensor configuration
    sensor: SensorConfig,
    /// Telescope configuration
    telescope: TelescopeConfig,
    /// Exposure duration
    exposure: Duration,
    /// Star magnitude
    mag: f64,
    /// PSF size in pixels
    psf_pix: f64,
    /// Noise floor multiplier for detection threshold
    noise_floor_multiplier: f64,
    /// Indices for result matrix (disk_idx, mag_idx)
    indices: (usize, usize),
    /// Number of times to run the experiment
    experiment_count: u32,
    /// Sensor temperature in degrees Celsius
    temperature: f64,
    /// Solar elongation for zodiacal background
    elongation: f64,
    /// Ecliptic latitude for zodiacal background
    latitude: f64,
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
        let flux = magnitude_to_electrons(self.mag, &self.exposure, &self.telescope, &self.sensor);

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
}

impl ExperimentResults {
    /// Calculate the RMS error from collected x/y errors
    fn rms_error(&self) -> f64 {
        let mut sum = 0.0;

        for &(x, y) in &self.xy_err {
            sum += x * x + y * y;
        }

        sum /= self.xy_err.len() as f64;
        sum.sqrt()
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
        let err_m = self.params.sensor.pixel_size_um * pix_err / 1_000_000.0;
        (err_m / self.params.telescope.focal_length_m).tan()
    }

    fn rms_err_mas(&self) -> f64 {
        let _pix_err = self.rms_error();
        self.rms_error_radians() * (648_000_000.0 / PI)
    }
}

/// Run a single experiment and return the error (distance) or NaN if not detected, and detection rate
fn run_single_experiment(params: &ExperimentParams) -> ExperimentResults {
    let half_d = params.domain as f64 / 2.0;

    // Run the experiment multiple times and average the results
    let mut xy_err: Vec<(f64, f64)> = Vec::new();

    for _ in 0..params.experiment_count {
        // Generate random position in the center area
        let xpos = rand::random::<f64>() * half_d + half_d;
        let ypos = rand::random::<f64>() * half_d + half_d;

        // Create star at the position with correct flux
        let star = params.star_at_pos(xpos, ypos);

        // Create electron image and add star
        let mut e_image: Array2<f64> = Array2::zeros((params.domain, params.domain));
        add_stars_to_image(&mut e_image, &vec![star], params.psf_pix);

        // Generate and add noise to image
        let sensor_noise =
            generate_sensor_noise(&params.sensor, &params.exposure, params.temperature, None);

        // Background light sources - using coordinates from CLI arguments
        let coords = SolarAngularCoordinates::new(params.elongation, params.latitude)
            .expect("Invalid coordinates");
        let z_light = ZodicalLight::new();
        let zodical = z_light.generate_zodical_background(
            &params.sensor,
            &params.telescope,
            &params.exposure,
            &coords,
        );

        let noise = &sensor_noise + &zodical;
        let total_e_image = &e_image + &noise;

        // Quantize to digital numbers
        let quantized = quantize_image(&total_e_image, &params.sensor);

        // Calculate detection threshold based on noise floor
        let quantized_noise = quantize_image(&noise, &params.sensor);
        let noise_mean = quantized_noise.map(|&x| x as f64).mean().unwrap();
        let cutoff_value = noise_mean * params.noise_floor_multiplier;

        // Run star detection algorithm
        let detected_stars = do_detections(&quantized, None, Some(cutoff_value));

        // Calculate detection error if star was found
        if detected_stars.len() == 1 {
            let detected_star = &detected_stars.first().unwrap();

            let x_diff = detected_star.x - xpos;
            let y_diff = detected_star.y - ypos;
            let err = (x_diff * x_diff + y_diff * y_diff).sqrt();

            // Detect spurious detections (mostly) and skip them
            if err > params.psf_pix * 2.0 {
                println!("Spurious detection: {} pixels", err);
                continue;
            }

            xy_err.push((x_diff, y_diff));
        }
    }

    ExperimentResults {
        params: params.clone(),
        xy_err,
    }
}

/// Main function for sensor floor estimation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Get coordinates from shared args
    let (elongation, latitude) = args.shared.coordinates;

    // Set domain size for our test images
    let domain = 256_usize;

    // Define all sensors to test with same dimensions
    let all_sensors = vec![
        sensor_models::GSENSE4040BSI.with_dimensions(domain as u32, domain as u32),
        sensor_models::GSENSE6510BSI.with_dimensions(domain as u32, domain as u32),
        sensor_models::HWK4123.with_dimensions(domain as u32, domain as u32),
        sensor_models::IMX455.with_dimensions(domain as u32, domain as u32),
    ];

    let exposure = Duration::from_secs_f64(args.shared.exposure);
    let telescope = DEMO_50CM.clone();

    // PSF disk sizes to test (in Airy disk units) - 2 to 8 in steps of 0.25
    let disks = Array1::range(1.0, 6.25, 0.25);

    // Star magnitudes to test - 15 to 20 in steps of 0.25
    let mags = Array1::range(12.0, 19.25, 0.25);

    println!("Setting up experiments...");

    // Store results for each sensor
    let mut sensor_results: HashMap<String, Array2<f64>> = HashMap::new();

    // Build a vector of all experiments to run
    let mut all_experiments = Vec::new();

    // For each sensor, initialize the results array
    for sensor in &all_sensors {
        let sensor_results_array = Array2::<f64>::from_elem((disks.len(), mags.len()), f64::NAN);
        sensor_results.insert(sensor.name.clone(), sensor_results_array);

        println!("Preparing experiments for sensor: {}", sensor.name);

        // Create a vector of experiment parameters for this sensor
        for (disk_idx, disk) in disks.iter().enumerate() {
            // Calculate PSF size in pixels for this disk configuration
            let psf_pix = approx_airy_pixels(&telescope, sensor, args.shared.wavelength) * disk;

            for (mag_idx, mag) in mags.iter().enumerate() {
                // Create experiment parameters
                let params = ExperimentParams {
                    domain,
                    sensor: sensor.clone(),
                    telescope: telescope.clone(),
                    exposure,
                    mag: *mag,
                    psf_pix,
                    noise_floor_multiplier: args.noise_floor,
                    indices: (disk_idx, mag_idx),
                    experiment_count: args.experiments,
                    temperature: args.shared.temperature,
                    elongation,
                    latitude,
                };

                all_experiments.push(params);
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

    // Create another hashmap to store detection rates
    let mut detection_rates: HashMap<String, Array2<f64>> = HashMap::new();

    // Initialize detection rates arrays
    for sensor in &all_sensors {
        let detection_rate_array = Array2::<f64>::from_elem((disks.len(), mags.len()), 0.0);
        detection_rates.insert(sensor.name.clone(), detection_rate_array);
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
        let sensor_name = result.params.sensor.name.clone();
        let detection_rate = result.detection_rate();
        let (disk_idx, mag_idx) = result.params.indices;

        if let Some(sensor_array) = sensor_results.get_mut(&sensor_name) {
            sensor_array[[disk_idx, mag_idx]] = result.rms_err_mas();
        }

        if let Some(detection_array) = detection_rates.get_mut(&sensor_name) {
            detection_array[[disk_idx, mag_idx]] = detection_rate;
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
    writeln!(csv_file, "Noise Floor Multiplier: {}", args.noise_floor).unwrap();
    writeln!(csv_file, "Wavelength: {} nm", args.shared.wavelength).unwrap();
    writeln!(csv_file, "Aperture diameter {} m", telescope.aperture_m).unwrap();
    writeln!(
        csv_file,
        "Experiments per configuration: {}",
        args.experiments
    )
    .unwrap();
    writeln!(csv_file).unwrap();

    // Sort the sensor names for consistent output
    let mut sensors_ordered: Vec<_> = all_sensors
        .iter()
        .map(|sensor| sensor.name.clone())
        .collect();

    sensors_ordered.sort();

    for sensor_name in sensors_ordered {
        let results = sensor_results.get(&sensor_name).unwrap();
        let detection_rate_array = detection_rates.get(&sensor_name).unwrap();

        println!("\n==== Sensor: {} ====", sensor_name);

        // Write sensor name to CSV
        writeln!(csv_file, "SENSOR: {}", sensor_name).unwrap();
        writeln!(csv_file).unwrap();

        println!("  Detection Rate Matrix (% of attempts that detected a star):");

        // Write detection rate matrix title to CSV
        writeln!(csv_file, "Detection Rate Matrix (%)").unwrap();

        // CSV header row with magnitude values
        write!(csv_file, "Disk\\Mag,").unwrap();
        for mag in &mags {
            write!(csv_file, "{:.2},", mag).unwrap();
        }
        writeln!(csv_file).unwrap();

        // Print header row with magnitude values
        print!("  Disk\\Mag |");
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

        // Print and write detection rate matrix
        for (disk_idx, disk) in disks.iter().enumerate() {
            // CSV row
            write!(csv_file, "{:.2},", disk).unwrap();

            // Console row
            print!("  {:.1}      |", disk);

            for mag_idx in 0..mags.len() {
                let rate = detection_rate_array[[disk_idx, mag_idx]];

                // CSV cell
                write!(csv_file, "{:.1},", rate * 100.0).unwrap();

                // Console cell
                print!(" {:3.0}% |", rate * 100.0);
            }
            writeln!(csv_file).unwrap();
            println!();
        }

        // Add blank line after matrix
        writeln!(csv_file).unwrap();

        println!("\n  Mean Position Error Matrix (mas, NaN = no detection):");

        // Write error matrix title to CSV
        writeln!(csv_file, "Mean Position Error Matrix (mas)").unwrap();

        // CSV header row with magnitude values
        write!(csv_file, "Disk\\Mag,").unwrap();
        for mag in &mags {
            write!(csv_file, "{:.2},", mag).unwrap();
        }
        writeln!(csv_file).unwrap();

        // Print header row with magnitude values
        print!("  Disk\\Mag |");
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

        // Print and write error matrix
        for (disk_idx, disk) in disks.iter().enumerate() {
            // CSV row
            write!(csv_file, "{:.2},", disk).unwrap();

            // Console row
            print!("  {:.1}      |", disk);

            for mag_idx in 0..mags.len() {
                let err = results[[disk_idx, mag_idx]];

                // CSV cell
                if err.is_nan() {
                    write!(csv_file, ",").unwrap(); // Empty cell for NaN
                } else {
                    write!(csv_file, "{:.2},", err).unwrap();
                }

                // Console cell
                if err.is_nan() {
                    print!("  --- |");
                } else {
                    print!(" {:.2} |", err);
                }
            }
            writeln!(csv_file).unwrap();
            println!();
        }

        // Add blank line after matrix
        writeln!(csv_file).unwrap();

        println!("\n  Summary by Disk Size:");

        // Write summary title to CSV
        writeln!(csv_file, "Summary by Disk Size").unwrap();
        writeln!(csv_file, "Disk,Mean Error (pixels),Detection Rate (%)").unwrap();

        for (disk_idx, disk) in disks.iter().enumerate() {
            let mut error_sum = 0.0;
            let mut error_count = 0;
            let mut detection_sum = 0.0;

            for mag_idx in 0..mags.len() {
                let err = results[[disk_idx, mag_idx]];
                let rate = detection_rate_array[[disk_idx, mag_idx]];

                detection_sum += rate;

                if !err.is_nan() {
                    error_sum += err;
                    error_count += 1;
                }
            }

            let mean_err = if error_count > 0 {
                error_sum / error_count as f64
            } else {
                f64::NAN
            };

            let avg_detection_rate = if !mags.is_empty() {
                detection_sum / mags.len() as f64
            } else {
                0.0
            };

            // Write summary row to CSV
            if mean_err.is_nan() {
                writeln!(csv_file, "{:.2},,{:.1}", disk, avg_detection_rate * 100.0).unwrap();
            } else {
                writeln!(
                    csv_file,
                    "{:.2},{:.4},{:.1}",
                    disk,
                    mean_err,
                    avg_detection_rate * 100.0
                )
                .unwrap();
            }

            println!(
                "    Disk {:.1}: Avg Error {:.4} pixels, Avg Detection Rate {:.1}%",
                disk,
                mean_err,
                avg_detection_rate * 100.0
            );
        }

        // Add separator between sensors
        writeln!(csv_file).unwrap();
        writeln!(csv_file, "-----------------------------------------------").unwrap();
        writeln!(csv_file).unwrap();

        println!();
    }

    println!("CSV output completed: {}", args.output_csv);

    Ok(())
}

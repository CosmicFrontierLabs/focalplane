//! Matrix experiments for single star detection across sensors
//!
//! This tool runs comprehensive experiments testing star detection and centroiding
//! accuracy across different sensors, magnitudes, and PSF sizes.

use clap::Parser;
use core::f64;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::{Array1, Array3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use simulator::hardware::sensor::models::ALL_SENSORS;
use simulator::hardware::SatelliteConfig;
use simulator::shared_args::{RangeArg, SharedSimulationArgs};
use simulator::sims::single_detection::{run_single_experiment, ExperimentParams};
use simulator::units::{LengthExt, Temperature, TemperatureExt, Wavelength};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Command line arguments for sensor floor estimation
#[derive(Parser, Debug)]
#[command(
    name = "Single Detection Matrix",
    about = "Runs matrix experiments for single star detection across sensors",
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
    star_finder: simulator::image_proc::detection::StarFinder,

    /// Random seed for reproducible experiments
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Main function for sensor floor estimation matrix experiments
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
                Temperature::from_celsius(args.shared.temperature),
                Wavelength::from_nanometers(args.shared.wavelength),
            )
        })
        .collect();

    // PSF disk sizes to test (in Airy disk FWHM units) - from CLI args
    let (disk_start, disk_stop, disk_step) = args.disks.as_tuple();
    let disks = Array1::range(disk_start, disk_stop, disk_step);

    // Star magnitudes to test - from CLI args
    let (mag_start, mag_stop, mag_step) = args.mags.as_tuple();
    let mags = Array1::range(mag_start, mag_stop, mag_step);

    // Exposure durations to test from CLI range (in milliseconds)
    let exposures = args.exposures.to_duration_vec_ms();

    println!("Setting up experiments...");

    // Initialize master RNG with CLI seed
    let mut master_rng = StdRng::seed_from_u64(args.seed);
    println!("Using random seed: {}", args.seed);

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

                    // Create experiment parameters with unique seed
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
                        seed: master_rng.gen(), // Generate unique seed for this experiment
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
        telescope_config.aperture.as_meters()
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

        println!("\n==== Sensor: {sensor_name} ====");

        // Write sensor name to CSV
        writeln!(csv_file, "SENSOR: {sensor_name}").unwrap();
        writeln!(csv_file).unwrap();

        // Detection Rate Matrix - Console output (simplified, just show first exposure)
        println!(
            "  Detection Rate Matrix (% of attempts that detected a star, first exposure only):"
        );

        // Print header row with magnitude values
        print!("  Q\\Mag |");
        for mag in &mags {
            print!(" {mag:.2} |");
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
            print!("  {disk:.1}      |");
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
            write!(csv_file, "{mag:.2},").unwrap();
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
            write!(csv_file, "{mag:.2},").unwrap();
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
                        write!(csv_file, "{err:.4},").unwrap();
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
            write!(csv_file, "{mag:.2},").unwrap();
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
                        write!(csv_file, "{err:.4},").unwrap();
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
            write!(csv_file, "{mag:.2},").unwrap();
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

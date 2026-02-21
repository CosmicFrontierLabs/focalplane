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
use shared::range_arg::RangeArg;
use shared::units::{LengthExt, Temperature, TemperatureExt};
use simulator::hardware::sensor::models::ALL_SENSORS;
use simulator::hardware::SatelliteConfig;
use simulator::shared_args::SharedSimulationArgs;
use simulator::sims::single_detection::{run_single_experiment, ExperimentParams};
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
    star_finder: shared::image_proc::detection::StarFinder,

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
            )
        })
        .collect();

    // PSF disk sizes to test (in Airy disk FWHM units) - from CLI args
    let disks = Array1::from_vec(args.disks.to_vec().expect("Invalid disk range"));

    // Star magnitudes to test - from CLI args
    let mags = Array1::from_vec(args.mags.to_vec().expect("Invalid magnitude range"));

    // Exposure durations to test from CLI range (in milliseconds)
    let exposures: Vec<std::time::Duration> = args
        .exposures
        .to_vec()
        .expect("Invalid exposure range")
        .iter()
        .map(|&ms| std::time::Duration::from_millis(ms as u64))
        .collect();

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
                        seed: master_rng.random(), // Generate unique seed for this experiment
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
        .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
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
    let mut csv_file = File::create(csv_path)?;

    println!("Writing results to CSV file: {}", args.output_csv);

    // Write CSV header
    writeln!(csv_file, "Sensor Floor Estimation Results")?;
    writeln!(csv_file, "Parameters:")?;
    writeln!(csv_file, "Exposure: {} seconds", args.shared.exposure)?;
    writeln!(
        csv_file,
        "Noise Floor Multiplier: {}",
        args.shared.noise_multiple
    )?;
    writeln!(csv_file, "Telescope: {}", args.shared.telescope)?;
    writeln!(
        csv_file,
        "Aperture diameter: {} m",
        telescope_config.aperture.as_meters()
    )?;
    writeln!(
        csv_file,
        "Experiments per configuration: {}",
        args.experiments
    )?;
    writeln!(csv_file, "PSF Disk Range (FWHM units): {}", args.disks)?;
    writeln!(csv_file, "Star Magnitude Range: {}", args.mags)?;
    writeln!(csv_file, "Exposure Range (ms): {}", args.exposures)?;
    writeln!(
        csv_file,
        "Domain Size: {}x{} pixels",
        args.domain, args.domain
    )?;
    writeln!(csv_file)?;

    // Sort the sensor names for consistent output
    let mut sensors_ordered: Vec<_> = all_satellites
        .iter()
        .map(|satellite| satellite.sensor.name.clone())
        .collect();

    sensors_ordered.sort();

    for sensor_name in sensors_ordered {
        let results = &sensor_results[&sensor_name];
        let pixel_results = &sensor_pixel_results[&sensor_name];
        let detection_rate_array = &detection_rates[&sensor_name];
        let spurious_rate_array = &spurious_rates[&sensor_name];

        println!("\n==== Sensor: {sensor_name} ====");

        // Write sensor name to CSV
        writeln!(csv_file, "SENSOR: {sensor_name}")?;
        writeln!(csv_file)?;

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
        writeln!(csv_file, "Detection Rate Matrix (%)")?;

        // CSV header: Disk, Exposure, then magnitude columns
        write!(csv_file, "Q_Value,Exposure_ms,")?;
        for mag in &mags {
            write!(csv_file, "{mag:.2},")?;
        }
        writeln!(csv_file)?;

        // Write all combinations of exposure and disk (exposure grouped together)
        for (exposure_idx, exposure) in exposures.iter().enumerate() {
            for (disk_idx, disk) in disks.iter().enumerate() {
                write!(csv_file, "{:.2},{},", disk, exposure.as_millis())?;
                for mag_idx in 0..mags.len() {
                    let rate = detection_rate_array[[disk_idx, exposure_idx, mag_idx]];
                    write!(csv_file, "{:.1},", rate * 100.0)?;
                }
                writeln!(csv_file)?;
            }
        }
        writeln!(csv_file)?;

        // Mean Position Error Matrix (mas)
        writeln!(csv_file, "Mean Position Error Matrix (milliarcseconds)")?;

        // CSV header: Disk, Exposure, then magnitude columns
        write!(csv_file, "Q_Value,Exposure_ms,")?;
        for mag in &mags {
            write!(csv_file, "{mag:.2},")?;
        }
        writeln!(csv_file)?;

        // Write all combinations of exposure and disk (exposure grouped together)
        for (exposure_idx, exposure) in exposures.iter().enumerate() {
            for (disk_idx, disk) in disks.iter().enumerate() {
                write!(csv_file, "{:.2},{},", disk, exposure.as_millis())?;
                for mag_idx in 0..mags.len() {
                    let err = results[[disk_idx, exposure_idx, mag_idx]];
                    if err.is_nan() {
                        write!(csv_file, ",")?; // Empty cell for NaN
                    } else {
                        write!(csv_file, "{err:.4},")?;
                    }
                }
                writeln!(csv_file)?;
            }
        }
        writeln!(csv_file)?;

        // RMS Position Error Matrix (pixels)
        writeln!(csv_file, "RMS Position Error Matrix (pixels)")?;

        // CSV header: Disk, Exposure, then magnitude columns
        write!(csv_file, "Q_Value,Exposure_ms,")?;
        for mag in &mags {
            write!(csv_file, "{mag:.2},")?;
        }
        writeln!(csv_file)?;

        // Write all combinations of exposure and disk (exposure grouped together)
        for (exposure_idx, exposure) in exposures.iter().enumerate() {
            for (disk_idx, disk) in disks.iter().enumerate() {
                write!(csv_file, "{:.2},{},", disk, exposure.as_millis())?;
                for mag_idx in 0..mags.len() {
                    let err = pixel_results[[disk_idx, exposure_idx, mag_idx]];
                    if err.is_nan() {
                        write!(csv_file, ",")?; // Empty cell for NaN
                    } else {
                        write!(csv_file, "{err:.4},")?;
                    }
                }
                writeln!(csv_file)?;
            }
        }
        writeln!(csv_file)?;

        // Spurious Detection Rate Matrix (%)
        writeln!(csv_file, "Spurious Detection Rate Matrix (%)")?;

        // CSV header: Disk, Exposure, then magnitude columns
        write!(csv_file, "Q_Value,Exposure_ms,")?;
        for mag in &mags {
            write!(csv_file, "{mag:.2},")?;
        }
        writeln!(csv_file)?;

        // Write all combinations of exposure and disk (exposure grouped together)
        for (exposure_idx, exposure) in exposures.iter().enumerate() {
            for (disk_idx, disk) in disks.iter().enumerate() {
                write!(csv_file, "{:.2},{},", disk, exposure.as_millis())?;
                for mag_idx in 0..mags.len() {
                    let rate = spurious_rate_array[[disk_idx, exposure_idx, mag_idx]];
                    write!(csv_file, "{:.2},", rate * 100.0)?;
                }
                writeln!(csv_file)?;
            }
        }
        writeln!(csv_file)?;

        // Add separator between sensors
        writeln!(csv_file)?;
        writeln!(csv_file, "-----------------------------------------------")?;
        writeln!(csv_file)?;

        println!();
    }

    println!("CSV output completed: {}", args.output_csv);

    Ok(())
}

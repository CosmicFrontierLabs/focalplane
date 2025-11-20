//! Telescope/scope view with star field simulation and rendering
//!
//! Command-line interface for running sensor imaging experiments using the scene_runner module.
//!
//! Usage:
//! ```
//! cargo run --release --bin sensor_shootout -- [OPTIONS]
//! ```
//!
//! See --help for detailed options.

use chrono::Local;
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::info;
use rayon::prelude::*;
use shared::frame_writer::FrameWriterHandle;
use shared::image_proc::detection::StarFinder;
use shared::range_arg::RangeArg;
use simulator::hardware::sensor::models as sensor_models;
use simulator::hardware::SatelliteConfig;
use simulator::shared_args::SharedSimulationArgs;
use simulator::sims::scene_runner::{
    run_experiment, CsvWriter, ExperimentCommonArgs, ExperimentParams, ExperimentResult,
};
use simulator::{
    star_math::field_diameter,
    units::{AngleExt, Length, LengthExt, Temperature, TemperatureExt},
};
use starfield::framelib::random::RandomEquatorial;
use starfield::Equatorial;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default filename for experiment results CSV
/// Will be appended with timestamp in format: experiment_log_YYYYMMDD_HHMMSS.csv
const DEFAULT_CSV_FILENAME: &str = "experiment_log_YYYYMMDD_HHMMSS.csv";

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
    #[arg(long, default_value = DEFAULT_CSV_FILENAME)]
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
    #[arg(long, default_value_t = 40)]
    icp_max_iterations: usize,

    /// Convergence threshold for ICP star matching algorithm
    #[arg(long, default_value_t = 0.00001)]
    icp_convergence_threshold: f64,

    /// Star detection algorithm to use (dao, iraf, naive)
    #[arg(long, default_value = "naive")]
    star_finder: StarFinder,

    /// F-number range to test (format: "start:stop:step" or "start:stop:count#")
    /// Examples: "8:16:2" tests f/8, f/10, f/12, f/14, f/16
    ///          "8:16:5#" tests 5 evenly spaced values from f/8 to f/16
    #[arg(long)]
    f_number_range: Option<RangeArg>,

    /// Number of trials to run for each pointing (each trial uses different noise seed)
    #[arg(long, default_value_t = 1)]
    trials: u32,
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
    let exposure_durations_vec: Vec<Duration> = args
        .exposure_range_ms
        .to_vec()
        .expect("Invalid exposure range")
        .iter()
        .map(|&ms| Duration::from_millis(ms as u64))
        .collect();
    let exposure_durations: Vec<f64> = exposure_durations_vec
        .iter()
        .map(|d| d.as_secs_f64())
        .collect();
    info!(
        "Using {} exposure durations: {:?}s",
        exposure_durations.len(),
        exposure_durations
    );

    // Use only HWK4123 sensor for testing
    let selected_sensors = [&*sensor_models::HWK4123];

    // Get telescope from shared args
    let selected_telescope = args.shared.telescope.to_config();
    info!(
        "Using telescope: {} (aperture: {:.2}m, f/{:.1})",
        selected_telescope.name,
        selected_telescope.aperture.as_meters(),
        selected_telescope.f_number()
    );

    // Parse f-number range
    let f_numbers: Vec<f64> = if let Some(f_range) = args.f_number_range {
        f_range.to_vec().expect("Invalid f-number range")
    } else {
        // Default to current telescope f-number
        vec![selected_telescope.f_number()]
    };

    info!("Using f-numbers: {f_numbers:?}");

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
                    Temperature::from_celsius(args.shared.temperature),
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
                    Temperature::from_celsius(args.shared.temperature),
                )
            })
            .collect()
    };

    // Compute the maximal FOV using smallest f-number (largest FOV)
    let min_f_number = f_numbers.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut max_fov = 0.0;

    for satellite in satellites.iter() {
        // Create telescope with smallest f-number for FOV calculation
        let focal_length =
            Length::from_meters(satellite.telescope.aperture.as_meters() * min_f_number);
        let telescope_min_f = satellite.telescope.with_focal_length(focal_length);
        let fov_angle = field_diameter(&telescope_min_f, &satellite.sensor);
        let fov_deg = fov_angle.as_degrees();

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
    let output_csv_path = if args.output_csv == DEFAULT_CSV_FILENAME {
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
        aperture_m: base_telescope.aperture.as_meters(),
    };

    // Initialize frame writer if saving images
    let num_workers = num_cpus::get().max(2);
    let buffer_size = num_workers * 4;
    info!("Initializing frame writer with {num_workers} workers");
    let frame_writer =
        FrameWriterHandle::new(num_workers, buffer_size).expect("Failed to create frame writer");

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

        // Create multiple trials for each pointing with different RNG seeds
        for trial in 0..args.trials {
            // Generate unique seed combining experiment number and trial number
            // This ensures reproducibility while giving different noise for each trial
            let rng_seed = ((i as u64) << 32) | (trial as u64);

            let params = ExperimentParams {
                experiment_num: i,
                trial_num: trial,
                rng_seed,
                ra_dec,
                satellites: satellites.clone(),
                f_numbers: f_numbers.clone(),
                common_args: common_args.clone(),
            };
            all_experiments.push(params);
        }

        if args.single_shot_debug.is_some() {
            // If in single-shot debug mode, only run one set of experiments
            break;
        }
    }

    info!(
        "Running {} total experiments ({} pointings × {} trials) {}...",
        all_experiments.len(),
        args.experiments,
        args.trials,
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
            .map(|params| run_experiment(params, &catalog, max_fov, &frame_writer))
            .inspect(|_| {
                pb.inc(1);
            })
            .collect()
    } else {
        all_experiments
            .par_iter()
            .map(|params| run_experiment(params, &catalog, max_fov, &frame_writer))
            .inspect(|_| {
                pb.inc(1);
            })
            .collect()
    };

    // Flatten results from Vec<Vec<ExperimentResult>> to Vec<ExperimentResult>
    let flattened_results: Vec<ExperimentResult> =
        experiment_results.into_iter().flatten().collect();

    pb.finish_with_message("Experiments complete!");

    info!("Waiting for frame writer to finish...");
    frame_writer.wait_for_completion();

    // Process results
    info!("Processing results...");

    // Results are written on-the-fly during experiments
    info!("Results written to CSV file: {output_csv_path}");

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
    info!("  Unique pointings: {}", args.experiments);
    info!("  Trials per pointing: {}", args.trials);
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
        info!("Parallel speedup: {speedup:.2}x");
        info!(
            "Effective CPU utilization: {:.1}%",
            speedup * 100.0 / num_cpus::get() as f64
        );
    }
    info!("======================================================");

    Ok(())
}

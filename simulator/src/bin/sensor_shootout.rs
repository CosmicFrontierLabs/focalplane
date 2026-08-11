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
    name = "Sensor Shootout",
    about = "Compare sensor performance across sky pointings and exposure settings",
    long_about = "Sensor performance comparison tool for evaluating detection accuracy.\n\n\
        This tool simulates telescope imaging across multiple sky pointings, exposure \
        durations, and optical configurations. For each experiment it:\n  \
        1. Renders a synthetic star field from the Tycho-2 catalog\n  \
        2. Applies sensor noise model (read noise, dark current, photon noise)\n  \
        3. Runs star detection algorithm\n  \
        4. Matches detected stars against known catalog positions via ICP\n  \
        5. Records detection count, positional errors, and timing to CSV\n\n\
        Use this to evaluate sensor configurations, detection algorithms, and \
        exposure strategies before deploying to actual hardware."
)]
struct Args {
    #[command(flatten)]
    shared: SharedSimulationArgs,

    #[arg(
        long,
        default_value_t = 100,
        help = "Number of experiments (sky pointings) to run",
        long_help = "Number of random sky pointings to simulate. Each pointing generates \
            a unique star field. More experiments provide better statistical coverage \
            of detection performance across varying star densities and magnitudes. \
            Use --single-shot-debug for deterministic testing at a fixed position."
    )]
    experiments: u32,

    #[arg(
        long,
        value_parser = parse_ra_dec_coordinates,
        help = "Fixed RA,Dec coordinates for debugging (format: 'ra,dec' in degrees)",
        long_help = "Run simulation at a single fixed sky position instead of random \
            sampling. Useful for debugging or comparing sensors at a known location. \
            Format: 'ra,dec' in degrees, e.g., '56.75,24.12' for the Pleiades cluster. \
            RA must be in [0, 360), Dec must be in [-90, 90]. When specified, only \
            one pointing is simulated regardless of --experiments value."
    )]
    single_shot_debug: Option<Equatorial>,

    #[arg(
        long,
        default_value = "experiment_output",
        help = "Directory for output images and data",
        long_help = "Base directory for experiment outputs. A timestamp suffix is \
            automatically appended (e.g., experiment_output_20240115_143022/). \
            Contains PNG images of rendered frames and FITS files if --save-images \
            is enabled. The directory is created if it doesn't exist."
    )]
    output_dir: String,

    #[arg(
        long,
        default_value = DEFAULT_CSV_FILENAME,
        help = "CSV file for experiment results",
        long_help = "Path to CSV file for logging experiment results. If using the \
            default filename, a timestamp is automatically inserted. Each row contains: \
            experiment number, sensor name, RA/Dec, exposure, stars detected, \
            catalog matches, ICP error, read noise, dark current, and timing. \
            File is written incrementally during execution."
    )]
    output_csv: String,

    #[arg(
        long,
        action,
        help = "Disable saving PNG/FITS images (faster execution)",
        long_help = "Skip saving rendered images to disk. Significantly speeds up \
            experiments when only CSV metrics are needed. Detection and matching \
            still run normally; only disk I/O is skipped. Recommended for large \
            parameter sweeps where images aren't needed for inspection."
    )]
    no_save_images: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Run experiments serially instead of in parallel",
        long_help = "Execute experiments one at a time instead of using all CPU cores. \
            Useful for debugging (predictable ordering, cleaner logs) or when memory \
            is constrained. Parallel mode uses Rayon with work-stealing for optimal \
            CPU utilization. Serial mode shows accurate per-experiment timing."
    )]
    serial: bool,

    #[arg(
        long,
        help = "Target pixels per FWHM for consistent sampling across sensors",
        long_help = "Adjust telescope focal length to achieve this many pixels per \
            PSF full-width at half-maximum for each sensor. This normalizes sampling \
            across sensors with different pixel sizes, enabling fair comparison. \
            If not specified, uses the telescope's native focal length with each \
            sensor. Typical values: 2.0-5.0 pixels/FWHM (Nyquist is ~2.0).",
        value_name = "PIXELS"
    )]
    match_pixel_sampling: Option<f64>,

    #[arg(
        long,
        default_value = "50:1000:50",
        help = "Exposure range in milliseconds (start:stop:step)",
        long_help = "Range of exposure durations to test, in milliseconds. Format is \
            'start:stop:step', e.g., '50:1000:50' tests 50ms, 100ms, ..., 1000ms. \
            Each experiment runs at all exposure values. Longer exposures collect \
            more photons but increase dark current contribution. For bright stars, \
            shorter exposures may suffice; for faint detection limits, try longer \
            exposures up to several seconds.",
        value_name = "RANGE"
    )]
    exposure_range_ms: RangeArg,

    #[arg(
        long,
        default_value_t = 40,
        help = "Maximum iterations for ICP star matching",
        long_help = "Maximum number of iterations for the Iterative Closest Point (ICP) \
            algorithm that matches detected stars to catalog positions. ICP refines \
            the transformation between pixel coordinates and celestial coordinates. \
            More iterations improve accuracy but increase computation time. The \
            algorithm terminates early if convergence threshold is met."
    )]
    icp_max_iterations: usize,

    #[arg(
        long,
        default_value_t = 0.00001,
        help = "Convergence threshold for ICP matching",
        long_help = "ICP terminates when the change in mean squared error between \
            iterations drops below this threshold. Smaller values produce more \
            precise alignment but require more iterations. The threshold is in \
            units of squared pixels. Typical range: 1e-6 to 1e-4."
    )]
    icp_convergence_threshold: f64,

    #[arg(
        long,
        default_value = "naive",
        help = "Star detection algorithm (dao, iraf, naive)",
        long_help = "Star detection algorithm to use:\n  \
            - naive: Fast centroid-based detection, good for bright isolated stars\n  \
            - iraf: IRAF-style detection with iterative background fitting\n  \
            - dao: DAOPHOT-style PSF fitting, best for crowded fields\n\n\
            Performance/accuracy trade-offs vary by star density and SNR. \
            'naive' is fastest and sufficient for most guide star applications."
    )]
    star_finder: StarFinder,

    #[arg(
        long,
        help = "F-number range to test (format: 'start:stop:step' or 'start:stop:count#')",
        long_help = "Range of f-numbers (focal ratio) to test. Varying f-number changes \
            both field-of-view and image scale. Format options:\n  \
            - 'start:stop:step': e.g., '8:16:2' tests f/8, f/10, f/12, f/14, f/16\n  \
            - 'start:stop:count#': e.g., '8:16:5#' tests 5 evenly spaced values\n\n\
            Smaller f-numbers give wider FOV but larger PSF (in pixels). \
            If not specified, uses the telescope's native f-number only.",
        value_name = "RANGE"
    )]
    f_number_range: Option<RangeArg>,

    #[arg(
        long,
        default_value_t = 1,
        help = "Number of trials per pointing (different noise seeds)",
        long_help = "Number of times to repeat each experiment with different random \
            noise seeds. Multiple trials at the same pointing provide statistics on \
            detection variability due to noise. Each trial uses a deterministic seed \
            derived from experiment and trial number for reproducibility. \
            Typical values: 1 for quick surveys, 5-10 for noise characterization."
    )]
    trials: u32,

    #[arg(
        long,
        default_value_t = 42,
        help = "Random seed for sky pointing generation"
    )]
    seed: u64,
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
        .to_vec()?
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

    let selected_sensors = [&*sensor_models::IMX455];

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
        f_range.to_vec()?
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
    let catalog = args.shared.load_catalog()?;
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
        std::fs::create_dir_all(output_path)?;
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
    let frame_writer = FrameWriterHandle::new(num_workers, buffer_size)?;

    // Build all experiment parameters upfront
    info!("Setting up experiments...");
    let mut all_experiments = Vec::new();

    let mut randomizer = RandomEquatorial::with_seed(args.seed);

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
        .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
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

//! FGS (Fine Guidance System) performance shootout
//!
//! Tests FGS tracking performance across various telescope and exposure configurations.
//! Evaluates tracking accuracy after lock acquisition over multiple time points.
//!
//! Usage:
//! ```
//! cargo run --release --bin fgs_shootout -- [OPTIONS]
//! ```

use chrono::Local;
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{debug, LevelFilter};
use monocle::{
    config::FgsConfig,
    state::{FgsEvent, FgsState},
    FineGuidanceSystem,
};
use monocle_harness::{motion_profiles::StaticPointing, simulator_camera::SimulatorCamera};
use rayon::prelude::*;
use shared::camera_interface::CameraInterface;
use shared::range_arg::RangeArg;
use shared::units::Temperature;
use shared::units::{Length, LengthExt, TemperatureExt};
use simulator::{
    hardware::SatelliteConfig,
    shared_args::{SensorModel, TelescopeModel},
};
use starfield::catalogs::binary_catalog::BinaryCatalog;
use starfield::framelib::random::RandomEquatorial;
use starfield::Equatorial;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Default filename for experiment results CSV
const DEFAULT_CSV_FILENAME: &str = "fgs_shootout_YYYYMMDD_HHMMSS.csv";

/// Default number of tracked points after lock
const DEFAULT_TRACKED_POINTS: usize = 10;

/// Command line arguments for FGS shootout
#[derive(Parser, Debug)]
#[command(
    name = "FGS Performance Shootout",
    about = "Tests FGS tracking performance across telescope and exposure configurations",
    long_about = None
)]
struct Args {
    /// Number of random sky pointings to test
    #[arg(short = 'n', long, default_value_t = 100)]
    num_pointings: usize,

    /// Sensor model to use (gsense4040bsi, gsense6510bsi, hwk4123, imx455)
    #[arg(long, default_value_t = SensorModel::Hwk4123)]
    sensor: SensorModel,

    /// Telescope model to use
    #[arg(long, default_value_t = TelescopeModel::CosmicFrontierJbt50cm)]
    telescope: TelescopeModel,

    /// F-number range to test (start:stop:step)
    /// Example: --f-number-range 8:16:2 tests f/8, f/10, f/12, f/14, f/16
    /// If not specified, uses the telescope's default f-number
    #[arg(long)]
    f_number_range: Option<RangeArg>,

    /// Exposure time range in milliseconds (start:stop:step)
    /// Example: --exposure-range-ms 100:1000:100
    #[arg(long, default_value = "100:1000:200")]
    exposure_range_ms: RangeArg,

    /// Number of time points to track after lock acquisition
    #[arg(long, default_value_t = DEFAULT_TRACKED_POINTS)]
    tracked_points: usize,

    /// Output CSV file for experiment results
    #[arg(short, long, default_value = DEFAULT_CSV_FILENAME)]
    output_csv: String,

    /// Path to star catalog file
    #[arg(long, default_value = "gaia_mag16_multi.bin")]
    catalog: PathBuf,

    /// Run experiments serially instead of in parallel
    #[arg(long)]
    serial: bool,

    /// Number of acquisition frames for FGS
    #[arg(long, default_value_t = 3)]
    acquisition_frames: usize,

    /// Minimum SNR for guide star selection
    #[arg(long, default_value_t = 10.0)]
    min_snr: f64,

    /// ROI size in pixels
    #[arg(long, default_value_t = 32)]
    roi_size: usize,

    /// Centroid radius multiplier (times FWHM)
    #[arg(long, default_value_t = 5.0)]
    centroid_multiplier: f64,

    /// Random seed for reproducibility
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Temperature in Celsius for sensor simulation
    #[arg(long, default_value_t = -10.0)]
    temperature: f64,

    /// Number of threads for parallel execution (0 = use all available)
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

/// Parameters for a single FGS experiment
#[derive(Debug, Clone)]
struct ExperimentParams {
    pointing: Equatorial,
    satellite_config: SatelliteConfig,
    exposure_ms: f64,
    experiment_id: usize,
}

/// FGS configuration parameters
#[derive(Debug, Clone)]
struct FgsParams {
    acquisition_frames: usize,
    min_snr: f64,
    roi_size: usize,
    centroid_radius_multiplier: f64,
}

/// Results from a single FGS tracking point
#[derive(Debug, Clone)]
struct TrackingPoint {
    time_s: f64,
    x_error_pixels: f64,
    y_error_pixels: f64,
    x_actual: f64,
    y_actual: f64,
    x_estimated: f64,
    y_estimated: f64,
    magnitude: f64,
    flux: f64,
}

/// Results from a single FGS experiment
#[derive(Debug, Clone)]
struct ExperimentResult {
    // Original experiment parameters
    params: ExperimentParams,

    // Acquisition results
    lock_acquired: bool,
    num_guide_stars: usize,
    guide_star_magnitude: f64, // Gaia magnitude of tracked star (NaN if unknown)

    // Tracking results (empty if no lock)
    tracking_points: Vec<TrackingPoint>,

    // Summary statistics
    mean_x_error: f64,
    mean_y_error: f64,
    std_x_error: f64,
    std_y_error: f64,
}

impl ExperimentResult {
    /// Write CSV header
    fn write_csv_header(file: &mut File) -> std::io::Result<()> {
        writeln!(
            file,
            "experiment_id,pointing_ra_deg,pointing_dec_deg,f_number,exposure_ms,\
            sensor,telescope,temperature_c,\
            lock_acquired,num_guide_stars,guide_star_magnitude,\
            mean_x_error_pixels,mean_y_error_pixels,std_x_error_pixels,std_y_error_pixels,\
            num_tracked_points"
        )
    }

    /// Write result as CSV row
    fn write_csv_row(&self, file: &mut File) -> std::io::Result<()> {
        writeln!(
            file,
            "{},{:.6},{:.6},{:.1},{:.1},{},{},{:.1},{},{},{:.2},{:.6},{:.6},{:.6},{:.6},{}",
            self.params.experiment_id,
            self.params.pointing.ra.to_degrees(),
            self.params.pointing.dec.to_degrees(),
            self.params.satellite_config.telescope.f_number(),
            self.params.exposure_ms,
            self.params.satellite_config.sensor.name,
            self.params.satellite_config.telescope.name,
            self.params.satellite_config.temperature.as_celsius(),
            self.lock_acquired,
            self.num_guide_stars,
            self.guide_star_magnitude,
            self.mean_x_error,
            self.mean_y_error,
            self.std_x_error,
            self.std_y_error,
            self.tracking_points.len()
        )
    }

    /// Write detailed tracking data to separate CSV
    fn write_tracking_csv(&self, file: &mut File) -> std::io::Result<()> {
        for point in &self.tracking_points {
            writeln!(
                file,
                "{},{:.6},{:.6},{:.1},{:.1},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2},{:.2}",
                self.params.experiment_id,
                self.params.pointing.ra.to_degrees(),
                self.params.pointing.dec.to_degrees(),
                self.params.satellite_config.telescope.f_number(),
                self.params.exposure_ms,
                point.time_s,
                point.x_actual,
                point.y_actual,
                point.x_estimated,
                point.y_estimated,
                point.x_error_pixels,
                point.y_error_pixels,
                point.magnitude,
                point.flux,
            )?;
        }
        Ok(())
    }
}

/// Find the closest rendered star to a given pixel position
fn find_closest_rendered_star(
    rendered_stars: &[simulator::image_proc::render::StarInFrame],
    x: f64,
    y: f64,
    max_distance: f64,
) -> Option<&simulator::image_proc::render::StarInFrame> {
    let mut min_distance = f64::MAX;
    let mut closest_star = None;

    for star in rendered_stars {
        let dx = star.x - x;
        let dy = star.y - y;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < min_distance && distance <= max_distance {
            min_distance = distance;
            closest_star = Some(star);
        }
    }

    closest_star
}

/// Run a single FGS experiment
fn run_single_experiment(
    params: ExperimentParams,
    catalog: Arc<BinaryCatalog>,
    fgs_params: &FgsParams,
    tracked_points: usize,
) -> ExperimentResult {
    // Use the satellite configuration from params
    let satellite = params.satellite_config.clone();

    // Compute FWHM from satellite optics
    let pix_fwhm = satellite.airy_disk_pixel_space().fwhm();

    // Create simulator camera with catalog (Arc is cheap to clone, catalog itself is shared)
    let mut camera = SimulatorCamera::new(
        satellite,
        catalog,
        StaticPointing::from_equatorial_boxed(params.pointing),
    );

    // Set exposure time
    let exposure = Duration::from_millis(params.exposure_ms as u64);
    camera
        .set_exposure(exposure)
        .expect("Failed to set exposure");

    // Capture a frame to get rendered stars before passing camera to FGS
    camera
        .capture_frame()
        .expect("Failed to capture initial frame");
    let rendered_stars = camera.get_last_rendered_stars().to_vec();

    // Create FGS config with computed FWHM and camera saturation value
    let fgs_config = FgsConfig {
        acquisition_frames: fgs_params.acquisition_frames,
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: fgs_params.min_snr,
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: camera.saturation_value() * 0.95,
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
            bad_pixel_map: shared::bad_pixel_map::BadPixelMap::empty(),
            minimum_bad_pixel_distance: 5.0,
        },
        roi_size: fgs_params.roi_size,
        max_reacquisition_attempts: 5,
        centroid_radius_multiplier: fgs_params.centroid_radius_multiplier,
        fwhm: pix_fwhm,
        snr_dropout_threshold: 3.0,
    };

    // Create FGS
    let mut fgs = FineGuidanceSystem::new(fgs_config);

    // Start FGS
    let (_update, _settings) = fgs
        .process_event(FgsEvent::StartFgs)
        .expect("Failed to start FGS");

    // Acquisition phase
    for _ in 0..fgs_params.acquisition_frames {
        let (frame, metadata) = camera.capture_frame().expect("Failed to capture frame");
        let (_update, _settings) = fgs
            .process_frame(frame.view(), metadata.timestamp)
            .expect("Failed to process acquisition frame");
    }

    // Calibration frame
    let (frame, metadata) = camera.capture_frame().expect("Failed to capture frame");
    let (_update, _settings) = fgs
        .process_frame(frame.view(), metadata.timestamp)
        .expect("Failed to process calibration frame");

    // Check if we achieved lock and get guide star magnitude
    let (lock_acquired, num_guide_stars, guide_star_magnitude) = match fgs.state() {
        FgsState::Tracking { .. } => {
            // Get guide star position and find matching rendered star
            let magnitude = if let Some(guide_star) = fgs.guide_star() {
                find_closest_rendered_star(&rendered_stars, guide_star.x, guide_star.y, 10.0)
                    .map(|s| s.star.magnitude)
                    .unwrap_or(f64::NAN)
            } else {
                f64::NAN
            };
            (true, 1, magnitude)
        }
        _ => (false, 0, f64::NAN),
    };

    debug!(
        "Experiment {}: Pointing ({:.2}, {:.2})°, f/{:.1}, {}ms exposure - Lock: {}, Stars: {}, Mag: {}",
        params.experiment_id,
        params.pointing.ra.to_degrees(),
        params.pointing.dec.to_degrees(),
        params.satellite_config.telescope.f_number(),
        params.exposure_ms,
        lock_acquired,
        num_guide_stars,
        if guide_star_magnitude.is_nan() { "N/A".to_string() } else { format!("{guide_star_magnitude:.2}") }
    );

    // Tracking phase
    let mut tracking_points = Vec::new();

    if !lock_acquired {
        return ExperimentResult {
            params,
            lock_acquired: false,
            num_guide_stars: 0,
            guide_star_magnitude: f64::NAN,
            tracking_points: Vec::new(),
            mean_x_error: f64::NAN,
            mean_y_error: f64::NAN,
            std_x_error: f64::NAN,
            std_y_error: f64::NAN,
        };
    }

    // Get actual star position and magnitude from rendered stars for ground truth
    let guide_star = fgs
        .guide_star()
        .expect("Guide star should exist in tracking state");
    let (actual_x, actual_y, magnitude) = if let Some(star) =
        find_closest_rendered_star(&rendered_stars, guide_star.x, guide_star.y, 10.0)
    {
        (star.x, star.y, star.star.magnitude)
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };

    for _ in 0..tracked_points {
        let (frame, metadata) = camera.capture_frame().expect("Failed to capture frame");
        if let Ok((Some(update), _settings)) = fgs.process_frame(frame.view(), metadata.timestamp) {
            let point = TrackingPoint {
                time_s: update.timestamp.to_duration().as_secs_f64(),
                x_actual: actual_x,
                y_actual: actual_y,
                x_estimated: update.x,
                y_estimated: update.y,
                x_error_pixels: update.x - actual_x,
                y_error_pixels: update.y - actual_y,
                magnitude,
                flux: update.flux,
            };

            tracking_points.push(point);
        }
    }

    // Calculate statistics
    let (mean_x_error, mean_y_error, std_x_error, std_y_error) = if !tracking_points.is_empty() {
        let x_errors: Vec<f64> = tracking_points.iter().map(|p| p.x_error_pixels).collect();
        let y_errors: Vec<f64> = tracking_points.iter().map(|p| p.y_error_pixels).collect();

        let mean_x = x_errors.iter().sum::<f64>() / x_errors.len() as f64;
        let mean_y = y_errors.iter().sum::<f64>() / y_errors.len() as f64;

        let var_x =
            x_errors.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>() / x_errors.len() as f64;
        let var_y =
            y_errors.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>() / y_errors.len() as f64;

        let std_x = var_x.sqrt();
        let std_y = var_y.sqrt();

        (mean_x, mean_y, std_x, std_y)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    ExperimentResult {
        params,
        lock_acquired,
        num_guide_stars,
        guide_star_magnitude,
        tracking_points,
        mean_x_error,
        mean_y_error,
        std_x_error,
        std_y_error,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse arguments first
    let args = Args::parse();

    // Generate timestamp and create output directory
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = PathBuf::from(format!("fgs_output/experiment_{timestamp}"));
    create_dir_all(&output_dir)?;

    // Initialize logging to file in output directory
    let log_path = output_dir.join("output.log");
    let log_file = File::create(&log_path)?;
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .init();

    log::info!(
        "FGS Shootout starting - output directory: {}",
        output_dir.display()
    );

    println!("FGS Performance Shootout");
    println!("========================");
    println!("Output directory: {}", output_dir.display());
    println!("Number of pointings: {}", args.num_pointings);
    println!("F-number range: {:?}", args.f_number_range);
    println!("Exposure range (ms): {}", args.exposure_range_ms);
    println!("Tracked points per lock: {}", args.tracked_points);
    println!("Catalog: {}", args.catalog.display());
    println!("Parallel execution: {}", !args.serial);

    // Set thread pool size if specified
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
    }

    // Load star catalog
    println!("\nLoading catalog...");
    let catalog = Arc::new(BinaryCatalog::load(&args.catalog)?);
    println!("Loaded {} stars", catalog.len());

    // Get base satellite configuration
    let base_sensor = args.sensor.to_config();
    let base_telescope = args.telescope.to_config();

    // Generate parameter combinations
    let f_numbers = if let Some(ref f_range) = args.f_number_range {
        f_range.to_vec()?
    } else {
        vec![base_telescope.f_number()]
    };
    let exposures_ms = args.exposure_range_ms.to_vec()?;
    // Generate random pointings using RandomEquatorial with seed for reproducibility
    let mut randomizer = RandomEquatorial::with_seed(args.seed);
    let pointings: Vec<Equatorial> = (0..args.num_pointings)
        .map(|_| randomizer.next().unwrap())
        .collect();

    println!("\nParameter space:");
    println!("  Sensor: {}", base_sensor.name);
    println!("  Telescope: {}", base_telescope.name);
    println!("  Temperature: {:.1}°C", args.temperature);
    println!("  F-numbers: {f_numbers:?}");
    println!("  Exposures (ms): {exposures_ms:?}");
    println!(
        "  Total experiments: {}",
        f_numbers.len() * exposures_ms.len() * pointings.len()
    );

    // Generate all experiment parameters with satellite configurations
    let mut all_params = Vec::new();
    let mut experiment_id = 0;

    for pointing in &pointings {
        for &f_number in &f_numbers {
            // Create telescope with modified f-number
            let focal_length = Length::from_meters(base_telescope.aperture.as_meters() * f_number);
            let telescope = base_telescope.clone().with_focal_length(focal_length);

            // Create satellite configuration
            let satellite_config = SatelliteConfig::new(
                telescope,
                base_sensor.clone(),
                Temperature::from_celsius(args.temperature),
            );

            for &exposure_ms in &exposures_ms {
                all_params.push(ExperimentParams {
                    pointing: *pointing,
                    satellite_config: satellite_config.clone(),
                    exposure_ms,
                    experiment_id,
                });
                experiment_id += 1;
            }
        }
    }

    // Setup progress tracking
    let multi_progress = MultiProgress::new();
    let main_progress = multi_progress.add(ProgressBar::new(all_params.len() as u64));
    main_progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    main_progress.set_message("Running experiments...");

    // Shared results vector
    let results = Arc::new(Mutex::new(Vec::new()));

    // Create FGS parameters
    let fgs_params = FgsParams {
        acquisition_frames: args.acquisition_frames,
        min_snr: args.min_snr,
        roi_size: args.roi_size,
        centroid_radius_multiplier: args.centroid_multiplier,
    };

    // Run experiments
    let experiment_start = Instant::now();

    if args.serial {
        // Serial execution
        for params in all_params {
            let result =
                run_single_experiment(params, catalog.clone(), &fgs_params, args.tracked_points);

            results.lock().unwrap().push(result);
            main_progress.inc(1);
        }
    } else {
        // Parallel execution
        all_params.into_par_iter().for_each(|params| {
            let result =
                run_single_experiment(params, catalog.clone(), &fgs_params, args.tracked_points);

            results.lock().unwrap().push(result);
            main_progress.inc(1);
        });
    }

    main_progress.finish_with_message("Experiments complete!");

    let total_time = experiment_start.elapsed();

    // Sort results by experiment_id for consistent output
    let mut final_results = results.lock().unwrap().clone();
    final_results.sort_by_key(|r| r.params.experiment_id);

    // Calculate statistics
    let total_experiments = final_results.len();
    let successful_locks = final_results.iter().filter(|r| r.lock_acquired).count();
    let lock_rate = successful_locks as f64 / total_experiments as f64 * 100.0;

    println!("\n=== Results Summary ===");
    println!("Total experiments: {total_experiments}");
    println!("Successful locks: {successful_locks} ({lock_rate:.1}%)");
    println!("Total time: {:.2}s", total_time.as_secs_f64());
    println!(
        "Time per experiment: {:.3}s",
        total_time.as_secs_f64() / total_experiments as f64
    );

    // Write results to CSV in output directory
    let results_path = output_dir.join("results.csv");
    println!("\nWriting results to: {}", results_path.display());

    let mut output_file = File::create(&results_path)?;
    ExperimentResult::write_csv_header(&mut output_file)?;

    for result in &final_results {
        result.write_csv_row(&mut output_file)?;
    }

    // Write detailed tracking data
    let tracking_path = output_dir.join("tracking.csv");
    let mut tracking_file = File::create(&tracking_path)?;

    writeln!(
        tracking_file,
        "experiment_id,pointing_ra_deg,pointing_dec_deg,f_number,exposure_ms,\
        time_s,x_actual,y_actual,x_estimated,y_estimated,x_error_pixels,y_error_pixels,magnitude,flux"
    )?;

    for result in &final_results {
        if result.lock_acquired {
            result.write_tracking_csv(&mut tracking_file)?;
        }
    }

    println!("Tracking details written to: {}", tracking_path.display());

    // Print some statistics by configuration
    println!("\n=== Lock Rate by F-number ===");
    for f_num in &f_numbers {
        let f_results: Vec<_> = final_results
            .iter()
            .filter(|r| (r.params.satellite_config.telescope.f_number() - f_num).abs() < 0.01)
            .collect();
        let f_locks = f_results.iter().filter(|r| r.lock_acquired).count();
        println!(
            "f/{:.1}: {}/{} ({:.1}%)",
            f_num,
            f_locks,
            f_results.len(),
            f_locks as f64 / f_results.len() as f64 * 100.0
        );
    }

    println!("\n=== Lock Rate by Exposure ===");
    for exp in &exposures_ms {
        let exp_results: Vec<_> = final_results
            .iter()
            .filter(|r| (r.params.exposure_ms - exp).abs() < 0.01)
            .collect();
        let exp_locks = exp_results.iter().filter(|r| r.lock_acquired).count();
        println!(
            "{:.0}ms: {}/{} ({:.1}%)",
            exp,
            exp_locks,
            exp_results.len(),
            exp_locks as f64 / exp_results.len() as f64 * 100.0
        );
    }

    println!("\n✅ FGS shootout complete!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::image_proc::airy::PixelScaledAiryDisk;
    use shared::units::LengthExt;
    use simulator::image_proc::render::StarInFrame;
    use simulator::photometry::photoconversion::{SourceFlux, SpotFlux};
    use simulator::units::Length;
    use starfield::catalogs::StarData;

    fn create_test_star(x: f64, y: f64, magnitude: f64) -> StarInFrame {
        // Create minimal spot flux for testing (values don't matter for position tests)
        let wavelength = Length::from_meters(550e-9);
        let disk = PixelScaledAiryDisk::with_fwhm(2.0, wavelength);
        let spot_flux = SpotFlux { disk, flux: 1.0 };
        let source_flux = SourceFlux {
            photons: spot_flux.clone(),
            electrons: spot_flux,
        };

        StarInFrame {
            x,
            y,
            spot: source_flux,
            star: StarData {
                id: 1,
                position: Equatorial::from_degrees(0.0, 0.0),
                magnitude,
                b_v: None,
            },
        }
    }

    #[test]
    fn test_find_closest_star_within_distance() {
        let stars = vec![
            create_test_star(10.0, 10.0, 5.0),
            create_test_star(20.0, 20.0, 6.0),
        ];

        let result = find_closest_rendered_star(&stars, 11.0, 11.0, 5.0);
        assert!(result.is_some());
        let star = result.unwrap();
        assert_eq!(star.x, 10.0);
        assert_eq!(star.y, 10.0);
    }

    #[test]
    fn test_find_closest_star_outside_distance() {
        let stars = vec![create_test_star(10.0, 10.0, 5.0)];

        let result = find_closest_rendered_star(&stars, 20.0, 20.0, 5.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_closest_star_multiple_stars() {
        let stars = vec![
            create_test_star(10.0, 10.0, 5.0),
            create_test_star(12.0, 12.0, 6.0),
            create_test_star(8.0, 8.0, 7.0),
        ];

        let result = find_closest_rendered_star(&stars, 11.0, 11.0, 10.0);
        assert!(result.is_some());
        let star = result.unwrap();
        assert_eq!(star.x, 10.0);
        assert_eq!(star.y, 10.0);
    }

    #[test]
    fn test_find_closest_star_empty_list() {
        let stars: Vec<StarInFrame> = vec![];

        let result = find_closest_rendered_star(&stars, 10.0, 10.0, 5.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_closest_star_exact_match() {
        let stars = vec![create_test_star(10.0, 10.0, 5.0)];

        let result = find_closest_rendered_star(&stars, 10.0, 10.0, 1.0);
        assert!(result.is_some());
        let star = result.unwrap();
        assert_eq!(star.x, 10.0);
        assert_eq!(star.y, 10.0);
    }

    #[test]
    fn test_find_closest_star_at_boundary() {
        let stars = vec![create_test_star(10.0, 10.0, 5.0)];

        // Exactly at max distance (5.0 pixels away)
        let result = find_closest_rendered_star(&stars, 13.0, 14.0, 5.0);
        assert!(result.is_some());

        // Just over max distance
        let result = find_closest_rendered_star(&stars, 13.0, 14.0, 4.9);
        assert!(result.is_none());
    }
}

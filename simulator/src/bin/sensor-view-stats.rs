//! Sensor view statistics tool for catalog star counting
//!
//! This tool analyzes how many stars from a catalog are visible above a magnitude
//! threshold for a single sensor configuration across random sky pointings.
//!
//! It loads a single SatelliteConfig using shared_args, generates random RA/Dec coordinates,
//! and efficiently processes all stars once using the same projection mechanisms as render.rs
//! to determine which stars land on the sensor plane.
//!
//! Key features:
//! 1. Single SatelliteConfig built from telescope and sensor selection
//! 2. Generate configurable number of random RA/Dec sky pointings
//! 3. Single-pass iteration over all catalog stars for all pointings
//! 4. Uses StarProjector for efficient star-to-pixel coordinate transformation
//! 5. Count stars above specified magnitude threshold on sensor plane
//! 6. Output comprehensive CSV statistics
//!
//! Usage:
//! ```
//! cargo run --bin sensor-view-stats -- [OPTIONS]
//! ```
//!
//! See --help for detailed options.

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::prelude::*;
use shared::star_projector::StarProjector;
use shared::units::{Angle, AngleExt, LengthExt, Temperature, TemperatureExt};
use simulator::hardware::SatelliteConfig;
use simulator::shared_args::{parse_additional_stars, SensorModel, TelescopeModel};
use simulator::star_math::field_diameter;
use starfield::catalogs::minimal_catalog::MinimalCatalog;
use starfield::catalogs::StarPosition;
use starfield::framelib::random::RandomEquatorial;
use starfield::Equatorial;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Configuration for one projector (pointing only, single satellite)
struct ProjectorConfig {
    /// Experiment number for tracking
    pointing_num: u32,
    /// Projector for this pointing (exposes center coordinates)
    projector: StarProjector,
}

/// Accumulated results for one projector configuration
#[derive(Debug)]
struct ProjectorResult {
    /// Index into projector_configs array
    config_index: usize,
    /// Total stars that projected onto sensor
    total_stars_on_sensor: usize,
    /// List of star magnitudes (sorted brightest first)
    star_magnitudes: Vec<f64>,
}

/// Final results for one sky pointing
#[derive(Debug)]
struct SkyPointingResult {
    /// Experiment number
    pointing_num: u32,
    /// Sky coordinates
    coordinates: Equatorial,
    /// Total stars on sensor plane
    total_stars_on_sensor: usize,
    /// List of star magnitudes (sorted brightest first)
    star_magnitudes: Vec<f64>,
}

/// Command line arguments for sensor view statistics
#[derive(Parser, Debug)]
#[command(
    name = "Sensor View Statistics",
    about = "Analyzes catalog star visibility for a sensor across random sky pointings",
    long_about = None
)]
struct Args {
    /// Path to binary star catalog
    #[arg(long, default_value = "gaia_mag16_multi.bin")]
    catalog: PathBuf,

    /// Sensor model to use
    #[arg(long, default_value_t = SensorModel::Hwk4123)]
    sensor: SensorModel,

    /// Telescope model to use
    #[arg(long, default_value_t = TelescopeModel::CosmicFrontierJbt50cm)]
    telescope: TelescopeModel,

    /// Number of random sky pointings to analyze
    #[arg(long, default_value_t = 100)]
    pointings: u32,

    /// Output CSV file for results
    #[arg(long, default_value = "sensor_view_stats.csv")]
    output_csv: String,

    /// Run experiments serially instead of in parallel
    #[arg(long, default_value_t = false)]
    serial: bool,

    /// Random seed for reproducible sky pointings
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Efficiently process all stars once for all projector configurations
fn process_all_stars_efficiently(
    catalog: &MinimalCatalog,
    projector_configs: &[ProjectorConfig],
) -> Vec<ProjectorResult> {
    let num_configs = projector_configs.len();
    let num_stars = catalog.len();
    info!("Processing {num_stars} stars against {num_configs} projector configurations...");

    // Initialize thread-safe results for each projector config
    let results: Vec<Mutex<ProjectorResult>> = (0..projector_configs.len())
        .map(|i| {
            Mutex::new(ProjectorResult {
                config_index: i,
                total_stars_on_sensor: 0,
                star_magnitudes: Vec::new(),
            })
        })
        .collect();

    // Setup progress bar for star processing
    let pb = ProgressBar::new(num_stars as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) ETA: {eta}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    pb.set_message(format!("Processing {num_stars} stars in parallel"));

    // Process all stars in parallel
    catalog.stars().par_iter().for_each(|star| {
        let star_radec = Equatorial::from_degrees(star.ra(), star.dec());

        // Test this star against all projector configurations
        for (config_idx, config) in projector_configs.iter().enumerate() {
            // Use project() which does bounds checking automatically
            if config.projector.project(&star_radec).is_some() {
                // Star is on sensor plane
                let mut result = results[config_idx].lock().unwrap();
                result.total_stars_on_sensor += 1;
                result.star_magnitudes.push(star.magnitude);
            }
        }

        // Update progress bar
        pb.inc(1);
    });

    pb.finish_with_message(format!("Processed {num_stars} stars in parallel"));

    // Convert back to regular Vec and sort magnitudes (brightest first)
    let mut final_results: Vec<ProjectorResult> = results
        .into_iter()
        .map(|mutex_result| mutex_result.into_inner().unwrap())
        .collect();

    for result in &mut final_results {
        result
            .star_magnitudes
            .sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    final_results
}

/// Convert projector results to the final organized format
fn organize_results(
    projector_results: Vec<ProjectorResult>,
    projector_configs: &[ProjectorConfig],
) -> Vec<SkyPointingResult> {
    let mut sky_pointing_results: Vec<SkyPointingResult> = projector_results
        .into_iter()
        .map(|result| {
            let config = &projector_configs[result.config_index];
            SkyPointingResult {
                pointing_num: config.pointing_num,
                coordinates: config.projector.center,
                total_stars_on_sensor: result.total_stars_on_sensor,
                star_magnitudes: result.star_magnitudes,
            }
        })
        .collect();

    // Sort by pointing number for consistent output
    sky_pointing_results.sort_by_key(|r| r.pointing_num);
    sky_pointing_results
}

/// Write results to CSV file
fn write_results_to_csv(
    results: &[SkyPointingResult],
    args: &Args,
    satellite: &SatelliteConfig,
    fov_deg: Angle,
) -> Result<(), Box<dyn std::error::Error>> {
    let csv_path = Path::new(&args.output_csv);
    let mut csv_file = File::create(csv_path)
        .unwrap_or_else(|_| panic!("Failed to create CSV file: {}", args.output_csv));

    info!("Writing results to CSV file: {}", args.output_csv);

    // Write CSV header with configuration information
    writeln!(csv_file, "Sensor View Statistics Results")?;
    writeln!(csv_file, "Configuration Parameters:")?;
    writeln!(csv_file, "Number of sky pointings: {}", args.pointings)?;
    writeln!(csv_file, "Random seed: {}", args.seed)?;
    writeln!(csv_file, "Star catalog: {}", args.catalog.display())?;
    writeln!(csv_file)?;

    // Write satellite information
    writeln!(csv_file, "Satellite Configuration:")?;
    writeln!(
        csv_file,
        "Sensor: {} - FOV: {:.4}° - Focal Length: {:.2}m - Aperture: {:.2}m",
        satellite.sensor.name,
        fov_deg.as_degrees(),
        satellite.telescope.focal_length.as_meters(),
        satellite.telescope.aperture.as_meters()
    )?;
    writeln!(csv_file)?;

    // Write detailed data header
    writeln!(csv_file, "Detailed Results:")?;
    writeln!(
        csv_file,
        "pointing_num,ra_degrees,dec_degrees,total_stars_on_sensor,star_magnitudes"
    )?;

    // Write detailed results
    for result in results {
        let magnitudes_str = result
            .star_magnitudes
            .iter()
            .map(|m| format!("{m:.2}"))
            .collect::<Vec<_>>()
            .join(";");

        writeln!(
            csv_file,
            "{},{:.6},{:.6},{},\"{}\"",
            result.pointing_num,
            result.coordinates.ra_degrees(),
            result.coordinates.dec_degrees(),
            result.total_stars_on_sensor,
            magnitudes_str
        )?;
    }

    writeln!(csv_file)?;

    // Write summary statistics
    writeln!(csv_file, "Summary Statistics:")?;
    writeln!(
        csv_file,
        "mean_total_stars,std_total_stars,min_total_stars,max_total_stars"
    )?;

    let total_stars: Vec<usize> = results.iter().map(|r| r.total_stars_on_sensor).collect();

    let mean_total = total_stars.iter().sum::<usize>() as f64 / total_stars.len() as f64;

    let var_total = total_stars
        .iter()
        .map(|&x| (x as f64 - mean_total).powi(2))
        .sum::<f64>()
        / total_stars.len() as f64;
    let std_total = var_total.sqrt();

    let min_total = *total_stars.iter().min().unwrap();
    let max_total = *total_stars.iter().max().unwrap();

    writeln!(
        csv_file,
        "{mean_total:.2},{std_total:.2},{min_total},{max_total}"
    )?;

    Ok(())
}

/// Main function for sensor view statistics
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging from environment variables
    env_logger::init();

    // Parse command line arguments
    let args = Args::parse();

    // Create single satellite configuration (temperature irrelevant for star projection)
    let satellite = SatelliteConfig::new(
        args.telescope.to_config().clone(),
        args.sensor.to_config().clone(),
        Temperature::from_celsius(0.0),
    );

    let fov_deg = field_diameter(&satellite.telescope, &satellite.sensor);
    info!(
        "Satellite: {}, FOV: {:.4}°, Aperture: {:.2}m, Focal Length: {:.2}m",
        satellite.sensor.name,
        fov_deg.as_degrees(),
        satellite.telescope.aperture.as_meters(),
        satellite.telescope.focal_length.as_meters()
    );

    // Load the catalog and merge with additional bright stars
    info!(
        "Loading binary star catalog from: {}",
        args.catalog.display()
    );
    let catalog = MinimalCatalog::load(&args.catalog)?;

    let additional_stars = parse_additional_stars()?;
    info!(
        "Parsed {} additional bright stars from embedded CSV",
        additional_stars.len()
    );

    let mut all_stars = catalog.stars().to_vec();
    all_stars.extend(additional_stars);

    let updated_description = format!(
        "{} + {} additional bright stars",
        catalog.description(),
        all_stars.len() - catalog.len()
    );
    let catalog = MinimalCatalog::from_stars(all_stars, &updated_description);

    info!("Setting up {} random sky pointings...", args.pointings);

    // Create randomizer with specified seed for reproducible results
    let mut randomizer = RandomEquatorial::with_seed(args.seed);

    // Build all projector configurations (one per pointing)
    let mut projector_configs = Vec::new();

    let fov_rad = fov_deg;
    let (sensor_width, sensor_height) = satellite.sensor.dimensions.get_pixel_width_height();
    let radians_per_pixel = fov_rad.as_radians() / sensor_width.max(sensor_height) as f64;

    for i in 0..args.pointings {
        let coordinates = randomizer.next().unwrap();

        let projector =
            StarProjector::new(&coordinates, radians_per_pixel, sensor_width, sensor_height);

        projector_configs.push(ProjectorConfig {
            pointing_num: i,
            projector,
        });
    }

    info!(
        "Created {} projector configurations ({} pointings)",
        projector_configs.len(),
        args.pointings,
    );

    // Process all stars efficiently in one pass
    info!("Processing all stars in single pass...");
    let projector_results = process_all_stars_efficiently(&catalog, &projector_configs);

    // Organize results by pointing
    let sky_pointing_results = organize_results(projector_results, &projector_configs);

    // Write results to CSV
    write_results_to_csv(&sky_pointing_results, &args, &satellite, fov_deg)?;

    // Print summary to console
    info!("Analysis completed successfully!");
    info!("Results written to: {}", args.output_csv);

    // Print quick summary statistics
    info!("\nQuick Summary:");
    let total_stars: usize = sky_pointing_results
        .iter()
        .map(|r| r.total_stars_on_sensor)
        .sum();
    let mean_stars = total_stars as f64 / sky_pointing_results.len() as f64;
    info!(
        "  {}: {:.1} stars/pointing",
        satellite.sensor.name, mean_stars
    );

    Ok(())
}

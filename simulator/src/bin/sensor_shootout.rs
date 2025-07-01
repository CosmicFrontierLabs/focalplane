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
//! cargo run --bin sensor_shootout -- [OPTIONS]
//! ```
//!
//! See --help for detailed options.

use clap::Parser;
use core::f64;
use image::DynamicImage;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{debug, info, warn};
use rayon::prelude::*;
use simulator::algo::icp::{icp_match_objects, Locatable2d};
use simulator::hardware::sensor::models as sensor_models;
use simulator::hardware::telescope::models::DEMO_50CM;
use simulator::hardware::SatelliteConfig;
use simulator::image_proc::detection::{detect_stars_unified, StarFinder};
use simulator::image_proc::histogram_stretch::sigma_stretch;
use simulator::image_proc::image::array2_to_gray_image;
use simulator::image_proc::io::write_hashmap_to_fits;
use simulator::image_proc::render::{render_star_field, RenderingResult, StarInFrame};
use simulator::image_proc::{
    draw_stars_with_x_markers, save_u8_image, stretch_histogram, u16_to_u8_scaled, StarDetection,
};
use simulator::photometry::zodical::SolarAngularCoordinates;
use simulator::shared_args::SharedSimulationArgs;
use simulator::star_math::EquatorialRandomizer;
use simulator::{field_diameter, SensorConfig};
use starfield::catalogs::{StarCatalog, StarData};
use starfield::Equatorial;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::time::Duration;
use viz::histogram::{Histogram, HistogramConfig, Scale};

/// Common arguments for experiments
#[derive(Debug, Clone)]
struct ExperimentCommonArgs {
    exposure: Duration,
    coordinates: SolarAngularCoordinates,
    noise_multiple: f64,
    output_dir: String,
    save_images: bool,
    icp_max_iterations: usize,
    icp_convergence_threshold: f64,
    star_finder: StarFinder,
}

/// Parameters for a single experiment (one sky pointing with all satellites)
#[derive(Debug, Clone)]
struct ExperimentParams {
    experiment_num: u32,
    ra_dec: Equatorial,
    satellites: Vec<SatelliteConfig>,
    common_args: ExperimentCommonArgs,
}

/// Results from a single experiment run
#[derive(Debug)]
struct ExperimentResult {
    experiment_num: u32,
    sensor_name: String,
    coordinates: Equatorial,
    detected_magnitudes: Vec<f64>,
    icp_rms_error: f64,
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

    /// Match FWHM sampling across all sensors using this value (pixels per FWHM). If not specified, uses base DEMO_50CM telescope with each sensor
    #[arg(long)]
    match_pixel_sampling: Option<f64>,

    /// Maximum iterations for ICP star matching algorithm
    #[arg(long, default_value_t = 20)]
    icp_max_iterations: usize,

    /// Convergence threshold for ICP star matching algorithm
    #[arg(long, default_value_t = 0.25)]
    icp_convergence_threshold: f64,

    /// Star detection algorithm to use (dao, iraf, naive)
    #[arg(long, default_value = "naive")]
    star_finder: StarFinder,
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
/// * Vec<ExperimentResult> containing detection results for each sensor
fn run_experiment<T: StarCatalog>(
    params: &ExperimentParams,
    catalog: &T,
    max_fov: f64,
) -> Vec<ExperimentResult> {
    let output_path = Path::new(&params.common_args.output_dir);

    debug!("Running experiment {}...", params.experiment_num);
    debug!(
        "  RA: {:.2}, Dec: {:.2}",
        params.ra_dec.ra_degrees(),
        params.ra_dec.dec_degrees()
    );
    debug!("  Temperature/Wavelength: per-satellite configuration");
    debug!("  Exposure: {:?}", params.common_args.exposure);
    debug!("  Noise Multiple: {}", params.common_args.noise_multiple);
    debug!("  Output Dir: {}", params.common_args.output_dir);
    debug!("  Save Images: {}", params.common_args.save_images);

    // Compute stars once for this sky pointing
    let stars = catalog.stars_in_field(
        params.ra_dec.ra_degrees(),
        params.ra_dec.dec_degrees(),
        max_fov,
    );
    print_am_hist(&stars);

    let star_refs: Vec<&StarData> = stars.iter().collect();

    let mut results = Vec::new();

    // Run experiment for each satellite at this sky pointing
    for satellite in params.satellites.iter() {
        debug!(
            "Running experiment for satellite: {} (T: {:.1}°C, λ: {:.0}nm)",
            satellite.sensor.name, satellite.temperature_c, satellite.wavelength_nm
        );
        // Render the star field for this satellite
        let render_result = render_star_field(
            &star_refs,
            &params.ra_dec,
            satellite,
            &params.common_args.exposure,
            &params.common_args.coordinates,
        );

        let prefix = format!(
            "{}_{}_",
            params.experiment_num,
            satellite.sensor.name.replace(" ", "_"),
        );

        // Only pick stuff that is noise_multiple above the noise floor
        let mean_noise_elec = render_result
            .noise_image
            .mean()
            .expect("Can't take image mean");
        let background_rms = mean_noise_elec * satellite.sensor.dn_per_electron;
        let airy_disk_pixels = satellite.airy_disk_fwhm_sampled().fwhm();
        let detection_sigma = params.common_args.noise_multiple;

        // Do the star detection
        let detected_stars = match detect_stars_unified(
            render_result.image.view(),
            params.common_args.star_finder,
            airy_disk_pixels,
            background_rms,
            detection_sigma,
        ) {
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
        let result = match icp_match_objects::<StarDetection, StarInFrame>(
            &detected_stars,
            &render_result.rendered_stars,
            params.common_args.icp_max_iterations,
            params.common_args.icp_convergence_threshold,
        ) {
            Ok((matches, icp_result)) => {
                // Debug statistics output
                debug_stats(&render_result, &matches).unwrap();

                let mut magnitudes: Vec<f64> =
                    matches.iter().map(|(_, s)| s.star.magnitude).collect();
                magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

                info!(
                    "Detected {} stars. Faintest magnitude: {:.2}",
                    magnitudes.len(),
                    magnitudes.last().unwrap_or(&f64::NAN)
                );

                ExperimentResult {
                    experiment_num: params.experiment_num,
                    sensor_name: satellite.sensor.name.clone(),
                    coordinates: params.ra_dec,
                    detected_magnitudes: magnitudes,
                    icp_rms_error: icp_result.mean_squared_error.sqrt(),
                }
            }
            Err(e) => {
                warn!(
                    "ICP matching failed for satellite {}: {}",
                    satellite.sensor.name, e
                );

                ExperimentResult {
                    experiment_num: params.experiment_num,
                    sensor_name: satellite.sensor.name.clone(),
                    coordinates: params.ra_dec,
                    detected_magnitudes: Vec::new(),
                    icp_rms_error: f64::NAN,
                }
            }
        };

        results.push(result);
    }
    results
}

/// Writes experiment results to CSV file with comprehensive header information
///
/// # Arguments
/// * `results` - Vector of experiment results to write
/// * `args` - Command line arguments containing configuration
/// * `satellites` - Vector of satellite configurations used  
/// * `all_experiments` - Vector of all experiment parameters (for count)
///
/// # Returns
/// * Result indicating success or failure of file writing
fn write_results_to_csv(
    results: &[ExperimentResult],
    args: &Args,
    satellites: &[SatelliteConfig],
    all_experiments: &[ExperimentParams],
) -> Result<(), Box<dyn std::error::Error>> {
    let log_file_path = Path::new(&args.output_csv);

    // Create CSV file and write header information
    let mut csv_file = std::fs::File::create(log_file_path)
        .unwrap_or_else(|_| panic!("Failed to create CSV file: {}", args.output_csv));

    info!("Writing results to CSV file: {}", args.output_csv);

    // Write CSV header with simulation configuration
    writeln!(csv_file, "Sensor Shootout Experiment Results")?;
    writeln!(csv_file, "Configuration Parameters:")?;
    writeln!(csv_file, "Exposure: {} seconds", args.shared.exposure)?;
    writeln!(csv_file, "Wavelength: per-satellite configuration")?;
    writeln!(csv_file, "Sensor Temperature: per-satellite configuration")?;
    writeln!(
        csv_file,
        "Noise Floor Multiplier: {}",
        args.shared.noise_multiple
    )?;
    writeln!(
        csv_file,
        "Solar Elongation: {:.2}°",
        args.shared.coordinates.elongation()
    )?;
    writeln!(
        csv_file,
        "Ecliptic Latitude: {:.2}°",
        args.shared.coordinates.latitude()
    )?;
    writeln!(csv_file, "Total Experiments: {}", args.experiments)?;
    writeln!(
        csv_file,
        "Total Runs: {} (experiments × satellites)",
        all_experiments.len() * satellites.len()
    )?;
    writeln!(
        csv_file,
        "Execution Mode: {}",
        if args.serial { "Serial" } else { "Parallel" }
    )?;
    writeln!(csv_file, "Save Images: {}", args.no_save_images)?;

    if let Some(pixel_sampling) = args.match_pixel_sampling {
        writeln!(
            csv_file,
            "FWHM Sampling Mode: Matched at {:.3} pixels per FWHM",
            pixel_sampling
        )?;
    } else {
        writeln!(
            csv_file,
            "FWHM Sampling Mode: Individual telescopes (DEMO_50CM base with each sensor)"
        )?;
    }

    if let Some(coords) = args.single_shot_debug {
        writeln!(
            csv_file,
            "Single Shot Debug: RA={:.2}°, Dec={:.2}°",
            coords.ra_degrees(),
            coords.dec_degrees()
        )?;
    } else {
        writeln!(
            csv_file,
            "Single Shot Debug: Disabled (random sky positions)"
        )?;
    }

    writeln!(csv_file, "Star Catalog: {}", args.shared.catalog.display())?;
    writeln!(csv_file, "Output Directory: {}", args.output_dir)?;
    writeln!(csv_file)?;

    // Write satellite information
    writeln!(csv_file, "Satellite Configurations:")?;
    for (i, satellite) in satellites.iter().enumerate() {
        let fov_deg = field_diameter(&satellite.telescope, &satellite.sensor);
        writeln!(
            csv_file,
            "Satellite {}: {} - Circumcircle FOV: {:.4}° - Focal Length: {:.8}m - Aperture: {:.2}m - Temp: {:.1}°C - Wavelength: {:.0}nm",
            i + 1,
            satellite.sensor.name,
            fov_deg,
            satellite.telescope.focal_length_m,
            satellite.telescope.aperture_m,
            satellite.temperature_c,
            satellite.wavelength_nm
        )?;
    }
    writeln!(csv_file)?;

    // Write CSV data header
    writeln!(csv_file, "Experiment Data:")?;
    writeln!(csv_file, "experiment_num,sensor_name,ra_degrees,dec_degrees,detected_count,icp_rms_error,detected_magnitudes")?;

    // Write experiment results
    for result in results {
        let log_entry = format!(
            "{}, {}, {:.2}, {:.2}, {}, {:.4}, {}\n",
            result.experiment_num,
            result.sensor_name,
            result.coordinates.ra_degrees(),
            result.coordinates.dec_degrees(),
            result.detected_magnitudes.len(),
            result.icp_rms_error,
            result
                .detected_magnitudes
                .iter()
                .map(|m| format!("{:.2}", m))
                .collect::<Vec<_>>()
                .join(", "),
        );

        csv_file.write_all(log_entry.as_bytes())?;
    }

    Ok(())
}

/// Main function for telescope view simulation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging from environment variables
    env_logger::init();

    // Parse command line arguments
    let args = Args::parse();

    let all_sensors = &*sensor_models::ALL_SENSORS;

    // Create satellite configurations directly from sensors and base telescope
    let satellites: Vec<SatelliteConfig> = if let Some(pixel_sampling) = args.match_pixel_sampling {
        // Use with_fwhm_sampling() to match pixel sampling across all sensors
        all_sensors
            .iter()
            .map(|sensor| {
                // Create base satellite with DEMO_50CM telescope
                let base_satellite = SatelliteConfig::new(
                    DEMO_50CM.clone(),
                    sensor.clone(),
                    args.shared.temperature,
                    args.shared.wavelength,
                );

                // Adjust to match the desired pixel sampling
                base_satellite.with_fwhm_sampling(pixel_sampling)
            })
            .collect()
    } else {
        // Use base DEMO_50CM telescope with each sensor (no resampling)
        all_sensors
            .iter()
            .map(|sensor| {
                SatelliteConfig::new(
                    DEMO_50CM.clone(),
                    sensor.clone(),
                    args.shared.temperature,
                    args.shared.wavelength,
                )
            })
            .collect()
    };

    // Compute the maximal FOV of all satellites:
    let mut max_fov = 0.0;
    for satellite in satellites.iter() {
        let fov_deg = field_diameter(&satellite.telescope, &satellite.sensor);
        info!("Satellite: {}, FOV: {:.4}°", satellite.sensor.name, fov_deg);
        if fov_deg > max_fov {
            max_fov = fov_deg;
        }
    }

    // Load the catalog....this requires some serious RAM
    let catalog = args.shared.load_catalog().expect("Could not load catalog?");
    info!("Loaded catalog with {} stars", catalog.len());

    // Ensure the output directory exists
    let output_path = Path::new(&args.output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path).expect("Failed to create output directory");
    }

    // Create common experiment arguments
    let common_args = ExperimentCommonArgs {
        exposure: args.shared.exposure.0,
        coordinates: args.shared.coordinates,
        noise_multiple: args.shared.noise_multiple,
        output_dir: args.output_dir.clone(),
        save_images: !args.no_save_images,
        icp_max_iterations: args.icp_max_iterations,
        icp_convergence_threshold: args.icp_convergence_threshold,
        star_finder: args.star_finder,
    };

    // Build all experiment parameters upfront
    info!("Setting up experiments...");
    let mut all_experiments = Vec::new();

    // Create randomizer with fixed seed for reproducible results
    let mut randomizer = EquatorialRandomizer::new(42);

    for i in 0..args.experiments {
        // Generate coordinates based on mode
        let ra_dec = if let Some(fixed_coords) = args.single_shot_debug {
            // Use fixed coordinates for single-shot debug mode
            fixed_coords
        } else {
            // Generate random RA/Dec coordinates for normal operation
            randomizer.generate()
        };

        // Create one experiment param per sky pointing (stars computed in run_experiment)
        let params = ExperimentParams {
            experiment_num: i,
            ra_dec,
            satellites: satellites.clone(),
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

    // Write results to CSV file
    write_results_to_csv(&flattened_results, &args, &satellites, &all_experiments)?;

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
    let u8_image = u16_to_u8_scaled(&render_result.image, max_bit_value);

    // Save the raw image
    let regular_path = output_path.join(format!("{}_regular.png", prefix));
    save_u8_image(&u8_image, &regular_path).expect("Failed to save image");

    // Create and save histogram stretched version
    let stretched_image = stretch_histogram(render_result.image.view(), 0.0, 50.0);

    // Convert stretched u16 image to u8 using auto-scaling for best contrast
    let img_flt = stretched_image.mapv(|x| x as f64);
    let normed = sigma_stretch(&img_flt, 5.0, Some(5));
    let u8_stretched = normed.mapv(|x| (x * 255.0).round() as u8);
    let stretched_path = output_path.join(format!("{}_stretched.png", prefix));
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

    let overlay_path = output_path.join(format!("{}_overlay.png", prefix));
    x_markers_image
        .save(&overlay_path)
        .expect("Failed to save image with X markers");

    // Export FITS files for both regular image and electron image
    let mut fits_data = HashMap::new();

    // Convert regular u16 image to f64 for FITS export
    let image_f64 = render_result.image.mapv(|x| x as f64);
    fits_data.insert("IMAGE".to_string(), image_f64);

    // Add electron image (already f64)
    fits_data.insert(
        "ELECTRON_IMAGE".to_string(),
        render_result.electron_image.clone(),
    );

    // Save FITS file with both datasets
    let fits_path = output_path.join(format!("{}_data.fits", prefix));
    write_hashmap_to_fits(&fits_data, &fits_path).expect("Failed to save FITS file");
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
        render_result.electron_image.sum()
    );
    debug!(
        "Total noise in image: {:.2}e-",
        render_result.noise_image.sum()
    );

    // Print ICP match distances
    for (dete, star) in matches.iter() {
        let distance = ((dete.x() - star.x()).powf(2.0) + (dete.y() - star.y()).powf(2.0)).sqrt();
        debug!(
            "Matched star {:?} to source {:?} with distance {:.2}",
            dete, star, distance
        );
    }
    // Get statistics for binning (gross)
    let num_bins = 25;
    let min_val = render_result
        .electron_image
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = render_result
        .electron_image
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Skip if all values are the same
    if (max_val - min_val).abs() < 1e-10 {
        debug!("  All pixel values are approximately: {:.2}", min_val);
        return Ok(());
    }

    // Calculate some basic stats for display
    let total_pixels = render_result.electron_image.len();

    let mut full_hist = Histogram::new_equal_bins(min_val..max_val, num_bins)?;
    full_hist.add_all(render_result.electron_image.iter().copied());

    let full_config = HistogramConfig {
        title: Some(format!(
            "Electron Count Histogram (Full Range: {:.2} - {:.2}e-)",
            min_val, max_val
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

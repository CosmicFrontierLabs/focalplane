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
//! cargo run --example scope_view -- [OPTIONS]
//! ```
//!
//! See --help for detailed options.

use clap::Parser;
use core::f64;
use image::DynamicImage;
use ndarray::Array2;
use simulator::algo::icp::{icp_match_objects, Locatable2d};
use simulator::hardware::sensor::models as sensor_models;
use simulator::hardware::telescope::{models as telescope_models, TelescopeConfig};
use simulator::image_proc::histogram_stretch::sigma_stretch;
use simulator::image_proc::image::array2_to_gray_image;
use simulator::image_proc::render::{render_star_field, RenderingResult, StarInFrame};
use simulator::image_proc::segment::do_detections;
use simulator::image_proc::{
    draw_stars_with_x_markers, save_u8_image, stretch_histogram, u16_to_u8_scaled, StarDetection,
};
use simulator::{field_diameter, SensorConfig};
use starfield::catalogs::{BinaryCatalog, StarCatalog, StarData};
use starfield::RaDec;
use std::collections::HashMap;
use std::io::Write;
use std::iter::zip;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;
use viz::histogram::{Histogram, HistogramConfig, Scale};

/// Command line arguments for telescope view simulation
#[derive(Parser, Debug)]
#[command(
    name = "Telescope View Simulator",
    about = "Simulates telescope view with star field rendering",
    long_about = None
)]
struct Args {
    /// Path to binary star catalog
    #[arg(long, default_value = "gaia_mag16_multi.bin")]
    catalog: PathBuf,

    /// Right ascension of field center in degrees
    #[arg(long, default_value_t = 100.0)]
    ra: f64,

    /// Declination of field center in degrees
    #[arg(long, default_value_t = 10.0)]
    dec: f64,

    /// Exposure time in seconds
    #[arg(long, default_value_t = 1.0)]
    exposure: f64,

    /// Wavelength in nanometers
    #[arg(long, default_value_t = 550.0)]
    wavelength: f64,

    /// Sensor model (GSENSE4040BSI, GSENSE6510BSI, HWK4123, IMX455)
    #[arg(long, default_value = "GSENSE4040BSI")]
    sensor: String,

    /// Enable debug output like electron histograms
    #[arg(long, default_value_t = false)]
    debug: bool,
}

fn build_optics_for_sensor(sensor: &SensorConfig, wavelength_nm: f64) -> TelescopeConfig {
    // Make a pretend telescope with focal length driven to make 4pix/airy disk
    let target_airy_disk_um = sensor.pixel_size_um * 4.0; // 4 pixels per Airy disk
    let aperture = telescope_models::DEMO_50CM.aperture_m;
    let wavelength_m = wavelength_nm * 1e-9; // Convert nm to m
    let focal_length_m = (target_airy_disk_um * aperture) / (1e6 * 1.22 * wavelength_m);

    let name = format!("{}-{:.2}", telescope_models::DEMO_50CM.name, focal_length_m);
    TelescopeConfig::new(
        &name,
        aperture,
        focal_length_m, // Default focal length, will be overridden by aperture
        telescope_models::DEMO_50CM.light_efficiency, // Light efficiency
    )
}

fn print_am_hist(stars: &Vec<StarData>) {
    // Print histogram of star magnitudes
    if stars.is_empty() {
        println!("No stars available to create histogram");
    } else {
        println!("Creating histogram of star magnitudes...");
        println!("Note that these stats include stars in the sensor circumcircle");
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
        println!(
            "\n{}",
            mag_hist.format().expect("Failed to format histogram")
        );
    }
}

fn run_experiment(
    sensor: &SensorConfig,
    telescope: &TelescopeConfig,
    ra_dec: RaDec,
    stars: &Vec<StarData>,
    exposure: Duration,
    experiment_num: u32,
    debug: bool,
) {
    let output_path = Path::new("experiment_output");

    // Ensure the output directory exists
    if !output_path.exists() {
        std::fs::create_dir_all(output_path).expect("Failed to create output directory");
    }

    // Convert Vec<StarData> to Vec<&StarData> for filtering
    let star_refs: Vec<&StarData> = stars.iter().collect();

    // Render the star field
    let render_result = render_star_field(
        &star_refs, &ra_dec, &telescope, &sensor, &exposure,
        550.0, // TODO(meawoppl) make this a parameter
    );

    if debug {
        // Print some statistics about the rendered image
        println!(
            "Total electrons in image: {:.2}e-",
            render_result.electron_image.sum()
        );
        println!(
            "Total noise in image: {:.2}e-",
            render_result.noise_image.sum()
        );
        println!(
            "{} pixels were clipped to maximum well depth of {:.2} electrons",
            render_result.n_clipped, sensor.max_well_depth_e
        );

        display_electron_histogram(&render_result.electron_image, 25).unwrap();
    }

    let prefix = format!("{}_{}_", experiment_num, sensor.name.replace(" ", "_"),);

    // Only pick stuff that is 5x above the noise floor
    let mean_noise_elec = render_result
        .noise_image
        .mean()
        .expect("Can't take image mean");
    let cutoff_value = mean_noise_elec * sensor.dn_per_electron * 5.0; // 5x above noise floor

    // Do the star detection
    let detected_stars = do_detections(
        &render_result.image,
        None, // Can tweak this setting later
        Some(cutoff_value),
    );

    // Use threaded version of save_image_outputs
    save_image_outputs_threaded(
        &render_result,
        sensor,
        &detected_stars,
        &output_path,
        &prefix,
    );

    // Now we take our detected stars and match them against the sources
    let matches = icp_match_objects::<StarDetection, StarInFrame>(
        &detected_stars,
        &render_result.rendered_stars,
        20,
        0.25,
    );

    if debug {
        for (dete, star) in matches.iter() {
            let distance =
                ((dete.x() - star.x()).powf(2.0) + (dete.y() - star.y()).powf(2.0)).sqrt();
            println!(
                "Matched star {:?} to source {:?} with distance {:.2}",
                dete, star, distance
            );
        }
    }

    let mut magnitudes: Vec<f64> = matches.iter().map(|(_, s)| s.star.magnitude).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "Detected {} stars. Faintest magnitude: {:.2}",
        magnitudes.len(),
        magnitudes.last().unwrap_or(&f64::NAN)
    );

    // Append a row to a file with the provided formatted string
    let log_file_path = output_path.join("experiment_log.txt");
    let log_entry = format!(
        "{}, {}, {:.2}, {:.2}, {}, {}\n",
        experiment_num,
        sensor.name,
        ra_dec.ra_degrees(),
        ra_dec.dec_degrees(),
        magnitudes.len(),
        magnitudes
            .iter()
            .map(|m| format!("{:.2}", m))
            .collect::<Vec<_>>()
            .join(", "),
    );

    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_path)
        .and_then(|mut file| file.write_all(log_entry.as_bytes()))
        .expect("Failed to write to log file");
}

/// Main function for telescope view simulation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    let all_sensors = vec![
        sensor_models::GSENSE4040BSI.clone(),
        sensor_models::GSENSE6510BSI.clone(),
        sensor_models::HWK4123.clone(),
        sensor_models::IMX455.clone(),
    ];

    let all_scopes = all_sensors
        .iter()
        .map(|s| build_optics_for_sensor(s, args.wavelength))
        .collect::<Vec<_>>();

    // Compute the maximal FOV of all sensors:
    let mut max_fov = 0.0;
    for (sensor, scope) in zip(all_sensors.iter(), all_scopes.iter()) {
        let fov_deg = field_diameter(scope, sensor);
        println!("Sensor: {}, FOV: {:.4}°", sensor.name, fov_deg);
        if fov_deg > max_fov {
            max_fov = fov_deg;
        }
    }

    // Load the catalog....this requires some serious RAM
    println!("Loading catalog from: {}", args.catalog.display());
    let catalog = BinaryCatalog::load(args.catalog).expect("Could not load catalog?");
    println!("Loaded catalog with {} stars", catalog.len());

    for i in 0..100 {
        // Generate a random  RaDec Pair:
        let ra = rand::random::<f64>() * 360.0; // Random RA in degrees
        let dec = rand::random::<f64>() * 0.0; // Random Dec in degrees
        let ra_dec = RaDec::from_degrees(ra, dec);

        // Siphon out the stars for the maximum FOV
        let stars = catalog.stars_in_field(ra_dec.ra_degrees(), ra_dec.dec_degrees(), max_fov);

        print_am_hist(&stars);

        for (sensor, scope) in zip(all_sensors.iter(), all_scopes.iter()) {
            println!(
                "Running experiment {} with sensor: {}, telescope: {}",
                i, sensor.name, scope.name
            );
            run_experiment(
                sensor,
                scope,
                ra_dec,
                &stars,
                Duration::from_secs_f64(args.exposure),
                i,
                args.debug,
            );
        }
    }

    Ok(())
}

/// Saves multiple image outputs from rendered star field data in a background thread
///
/// This function creates a new thread to perform image saving operations, allowing the main
/// thread to continue processing without waiting for I/O operations to complete.
///
/// # Arguments
/// * `render_result` - Complete results from star field rendering
/// * `sensor` - Sensor configuration used for image scaling
/// * `detected_stars` - Vector of detected star objects
/// * `output_path` - Directory where output files will be saved
/// * `prefix` - Filename prefix for all output files
fn save_image_outputs_threaded(
    render_result: &RenderingResult,
    sensor: &SensorConfig,
    detected_stars: &Vec<StarDetection>,
    output_path: &Path,
    prefix: &str,
) {
    thread::spawn({
        let render_result_clone = render_result.clone();
        let sensor_clone = sensor.clone();
        let detected_stars_clone = detected_stars.clone();
        let output_path = output_path.to_path_buf();
        let prefix = prefix.to_string();

        move || {
            save_image_outputs(
                &render_result_clone,
                &sensor_clone,
                &detected_stars_clone,
                &output_path,
                &prefix,
            );
        }
    });
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
    detected_stars: &Vec<StarDetection>,
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
}

/// Display a histogram of electron counts in the image
fn display_electron_histogram(
    image: &Array2<f64>,
    num_bins: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get statistics for binning (gross)
    let min_val = image.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = image.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Skip if all values are the same
    if (max_val - min_val).abs() < 1e-10 {
        println!("  All pixel values are approximately: {:.2}", min_val);
        return Ok(());
    }

    // Calculate some basic stats for display
    let total_pixels = image.len();

    let mut full_hist = Histogram::new_equal_bins(min_val..max_val, num_bins)?;
    full_hist.add_all(image.iter().copied());

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
    println!("\nElectron Count Statistics:");
    println!("  Total Pixels: {}", total_pixels);
    println!("  Min Value: {:.2} electrons", min_val);
    println!("  Max Value: {:.2} electrons", max_val);

    // Print the histograms
    println!("\n{}", full_hist.with_config(full_config).format()?);

    Ok(())
}

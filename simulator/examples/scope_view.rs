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
use image::DynamicImage;
use ndarray::Array2;
use simulator::algo::icp::{icp_match_objects, Locatable_2D};
use simulator::hardware::sensor::models as sensor_models;
use simulator::hardware::telescope::{models as telescope_models, TelescopeConfig};
use simulator::image_proc::histogram_stretch::stretch_histogram;
use simulator::image_proc::image::array2_to_gray_image;
use simulator::image_proc::render::render_star_field;
use simulator::image_proc::segment::do_detections;
use simulator::image_proc::{
    draw_stars_with_x_markers, save_u8_image, u16_to_u8_auto_scale, u16_to_u8_scaled,
};
use simulator::star_math::save_star_list;
use simulator::{field_diameter, SensorConfig};
use starfield::catalogs::StarData;
use starfield::{RaDec, StarfieldError};
use std::collections::HashMap;
use std::io::Write;
use std::iter::zip;
use std::path::{Path, PathBuf};
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

    /// Output image path
    #[arg(long, default_value = "scope_view.png")]
    output: String,

    /// Histogram stretched output path
    #[arg(long, default_value = "scope_view_stretched.png")]
    output_stretched: String,

    /// Star list output path
    #[arg(long, default_value = "star_list.txt")]
    star_list: String,

    /// Sensor model (GSENSE4040BSI, GSENSE6510BSI, HWK4123, IMX455)
    #[arg(long, default_value = "GSENSE4040BSI")]
    sensor: String,
}

fn stars_from_args(
    catalog: &PathBuf,
    ra_dec: RaDec,
    fov_deg: f64,
) -> Result<Vec<StarData>, StarfieldError> {
    // Determine catalog source

    // Get stars from selected catalog
    starfield::catalogs::get_stars_in_window(
        starfield::catalogs::CatalogSource::Binary(catalog.clone()),
        ra_dec,
        fov_deg * 2.0, // TODO(meawoppl) fov in catalog source is radius vs diameter?
    )
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
) {
    let output_path = Path::new("experiment_output");

    // Ensure the output directory exists
    if !output_path.exists() {
        std::fs::create_dir_all(output_path).expect("Failed to create output directory");
    }

    let fov_deg = field_diameter(telescope, sensor);

    // Convert Vec<StarData> to Vec<&StarData> for filtering
    let star_refs: Vec<&StarData> = stars.iter().collect();

    // Render the star field
    println!("Rendering star field...");
    let render_result = render_star_field(
        &star_refs,
        ra_dec.ra_degrees(),
        ra_dec.dec_degrees(),
        fov_deg,
        &telescope,
        &sensor,
        &exposure,
        550.0, // TODO(meawoppl) make this a parameter
    );

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

    // TODO(meawoppl) - export this again
    // Filter stars visible in the field for star list output
    // Write star list to file
    // save_star_list(
    //     &star_refs,
    //     &RaDec::from_degrees(args.ra, args.dec),
    //     fov_deg,
    //     &telescope.name,
    //     &sensor.name,
    //     sensor.width_px as usize,
    //     sensor.height_px as usize,
    //     &args.star_list,
    // )?;
    // println!("Star list saved to: {}", args.star_list);

    // Convert u16 image to u8 for saving (normalize by max bit depth value)
    let max_bit_value = (1 << (sensor.bit_depth as u32)) - 1;
    let u8_image = u16_to_u8_scaled(&render_result.image, max_bit_value);

    let prefix = format!("{}_{}_", experiment_num, sensor.name.replace(" ", "_"),);
    // Save the raw image
    let regular_path = output_path.join(format!("{}_regular.png", prefix));
    save_u8_image(&u8_image, &regular_path).expect("Failed to save image");
    println!("Image saved to: {:?}", regular_path);

    // Create and save histogram stretched version
    println!("Creating histogram stretched image...");
    let stretched_image = stretch_histogram(render_result.image.view(), 0.0, 100.0);

    // Convert stretched u16 image to u8 using auto-scaling for best contrast
    let stretched_path = output_path.join(format!("{}_stretched.png", prefix));
    let u8_stretched = u16_to_u8_auto_scale(&stretched_image);
    save_u8_image(&u8_stretched, &stretched_path).expect("Failed to save stretched image");
    println!("Stretched image saved to: {:?}", stretched_path);

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
    println!("Detected {} stars in the image", detected_stars.len());

    // Use light blue (135, 206, 250) for X markers
    let vis_image = array2_to_gray_image(&u8_image);
    let dyn_image = DynamicImage::ImageLuma8(vis_image);

    // Mutate the detected stars into the shape needed by rendering
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

    // Now we take our detected stars and match them against the sources
    let matches = icp_match_objects(&detected_stars, &render_result.rendered_stars, 20, 0.25);
    for (dete, star) in matches.iter() {
        let distance = f64::sqrt(dete.x() - star.x()).powi(2) + (dete.y() - star.y()).powi(2);
        println!(
            "Matched star {:?} to source {:?} with distance {:.2}",
            dete, star, distance
        );
    }

    let mut magnitudes: Vec<f64> = matches.iter().map(|(_, s)| s.star.magnitude).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

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

    // // Select telescope and sensor models
    // let telescope = match args.telescope.as_str() {
    //     "50cm" => telescope_models::DEMO_50CM.clone(),
    //     "1m" => telescope_models::FINAL_1M.clone(),
    //     "small" => telescope_models::SMALL_50MM.clone(),
    //     _ => return Err(format!("Unknown telescope model: {}", args.telescope).into()),
    // };

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

    for i in 0..100 {
        // Generate a random  RaDec Pair:
        let ra = rand::random::<f64>() * 360.0; // Random RA in degrees
        let dec = rand::random::<f64>() * 0.0; // Random Dec in degrees
        let ra_dec = RaDec::from_degrees(ra, dec);

        // Siphon out the stars for the maximum FOV
        let stars = stars_from_args(&args.catalog, ra_dec, max_fov)
            .expect("Failed to load stars from catalog");

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
            );
        }
    }

    Ok(())
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

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
use ndarray::Array2;
use simulator::hardware::sensor::{models as sensor_models, SensorConfig};
use simulator::hardware::telescope::{models as telescope_models, TelescopeConfig};
use simulator::image_proc::electron::{add_stars_to_image, StarInFrame};
use simulator::image_proc::histogram_stretch::stretch_histogram;
use simulator::image_proc::{save_u8_image, u16_to_u8_auto_scale, u16_to_u8_scaled};
use simulator::star_math::{equatorial_to_pixel, save_star_list};
use simulator::{field_diameter, magnitude_to_electrons};
use starfield::catalogs::StarData;
use starfield::RaDec;
use std::path::PathBuf;
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
    catalog: Option<PathBuf>,

    /// Right ascension of field center in degrees
    #[arg(long, default_value_t = 100.0)]
    ra: f64,

    /// Declination of field center in degrees
    #[arg(long, default_value_t = 10.0)]
    dec: f64,

    /// Number of synthetic stars if no catalog
    #[arg(long, default_value_t = 100)]
    stars: usize,

    /// Random seed for synthetic stars
    #[arg(long, default_value_t = 42)]
    seed: u64,

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

    /// Telescope model (50cm, 1m, 1m Final)
    #[arg(long, default_value = "1m")]
    telescope: String,

    /// Sensor model (GSENSE4040BSI, GSENSE6510BSI, HWK4123, IMX455)
    #[arg(long, default_value = "GSENSE4040BSI")]
    sensor: String,

    /// Use synthetic stars even if catalog is provided
    #[arg(long)]
    synthetic: bool,
}

/// Main function for telescope view simulation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Select telescope and sensor models
    let telescope = match args.telescope.as_str() {
        "50cm" => telescope_models::DEMO_50CM.clone(),
        "1m" => telescope_models::FINAL_1M.clone(),
        "small" => telescope_models::SMALL_50MM.clone(),
        _ => return Err(format!("Unknown telescope model: {}", args.telescope).into()),
    };

    let sensor = match args.sensor.as_str() {
        "GSENSE4040BSI" => sensor_models::GSENSE4040BSI.clone(),
        "GSENSE6510BSI" => sensor_models::GSENSE6510BSI.clone(),
        "HWK4123" => sensor_models::HWK4123.clone(),
        "IMX455" => sensor_models::IMX455.clone(),
        _ => return Err(format!("Unknown sensor model: {}", args.sensor).into()),
    };

    // Get field diameter (either from parameters or calculate from hardware)
    let fov_deg = field_diameter(&telescope, &sensor);

    // Show parameters
    println!("Telescope View Simulator");
    println!("=======================");
    println!("RA: {:.4}°", args.ra);
    println!("Dec: {:.4}°", args.dec);
    println!("FOV: {:.4}°", fov_deg);
    println!("Exposure: {}s", args.exposure);
    println!("Wavelength: {}nm", args.wavelength);
    println!(
        "Telescope: {} (aperture: {}m, focal length: {}m)",
        telescope.name, telescope.aperture_m, telescope.focal_length_m
    );
    println!(
        "Sensor: {} ({}×{} pixels, {}μm/px)",
        sensor.name, sensor.width_px, sensor.height_px, sensor.pixel_size_um
    );

    // Use this to print some things nicely
    approx_airy_pixels(&telescope, &sensor, 550.0);

    // Output paths
    println!("Outputs:");
    println!("Output image: {}", args.output);
    println!("Stretched image: {}", args.output_stretched);
    println!("Star list: {}", args.star_list);

    // Determine catalog source
    let catalog_source = if args.synthetic {
        starfield::catalogs::CatalogSource::Random {
            count: args.stars,
            seed: args.seed,
        }
    } else if let Some(path) = args.catalog {
        starfield::catalogs::CatalogSource::Binary(path)
    } else {
        // Default to Hipparcos if present, otherwise synthetic
        if PathBuf::from("hip_main.dat").exists() {
            starfield::catalogs::CatalogSource::Hipparcos
        } else {
            println!("No catalog specified, using synthetic stars");
            starfield::catalogs::CatalogSource::Random {
                count: args.stars,
                seed: args.seed,
            }
        }
    };

    // Get stars from selected catalog
    let stars = starfield::catalogs::get_stars_in_window(
        catalog_source,
        RaDec::from_degrees(args.ra, args.dec),
        fov_deg * 2.0, // TODO(meawoppl) fov in catalog source is radius vs diameter?
    )?;

    // Convert Vec<StarData> to Vec<&StarData> for filtering
    let star_refs: Vec<&StarData> = stars.iter().collect();

    // Print histogram of star magnitudes
    if star_refs.is_empty() {
        println!("No stars available to create histogram");
    } else {
        println!("Creating histogram of star magnitudes...");
        println!("Note that these stats include stars in the sensor circumcircle");
        let star_magnitudes: Vec<f64> = star_refs.iter().map(|star| star.magnitude).collect();

        // Create a magnitude histogram using the new specialized function
        // This automatically creates bins centered on integer magnitudes with 1.0 width
        let mag_hist = viz::histogram::create_magnitude_histogram(
            &star_magnitudes,
            Some(format!(
                "Star Magnitude Histogram ({} stars)",
                star_refs.len()
            )),
            false, // Use linear scale
        )
        .expect("Failed to create magnitude histogram");

        // Print the histogram
        println!(
            "\n{}",
            mag_hist.format().expect("Failed to format histogram")
        );
    }

    let duration = Duration::from_secs_f64(args.exposure);

    // Render the star field
    println!("Rendering star field...");
    let image = render_star_field(
        &star_refs,
        args.ra,
        args.dec,
        fov_deg,
        &telescope,
        &sensor,
        &duration,
        args.wavelength,
    );

    // Filter stars visible in the field for star list output
    // Write star list to file
    save_star_list(
        &star_refs,
        &RaDec::from_degrees(args.ra, args.dec),
        fov_deg,
        &telescope.name,
        &sensor.name,
        sensor.width_px as usize,
        sensor.height_px as usize,
        &args.star_list,
    )?;
    println!("Star list saved to: {}", args.star_list);

    // Convert u16 image to u8 for saving (normalize by max bit depth value)
    let max_bit_value = (1 << sensor.bit_depth) - 1;
    let u8_image = u16_to_u8_scaled(&image, max_bit_value);

    // Save the raw image
    save_u8_image(&u8_image, &args.output)?;
    println!("Image saved to: {}", args.output);

    // Create and save histogram stretched version
    println!("Creating histogram stretched image...");
    let stretched_image = stretch_histogram(image.view(), 25.0, 75.0);

    // Convert stretched u16 image to u8 using auto-scaling for best contrast
    let u8_stretched = u16_to_u8_auto_scale(&stretched_image);
    save_u8_image(&u8_stretched, &args.output_stretched)?;
    println!("Stretched image saved to: {}", args.output_stretched);

    Ok(())
}

/// Create gaussian PSF kernel based on telescope properties
fn approx_airy_pixels(
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    wavelength_nm: f64,
) -> f64 {
    // Calculate PSF size based on Airy disk
    let airy_radius_um = telescope.airy_disk_radius_um(wavelength_nm);
    let airy_radius_px = airy_radius_um / sensor.pixel_size_um;

    println!(
        "Airy disk radius: {:.2} um, {:.2} px",
        airy_radius_um, airy_radius_px
    );

    // Create a Gaussian approximation of the Airy disk
    // Using sigma ≈ radius/1.22 to approximate Airy disk with Gaussian
    airy_radius_px / 1.22
}

/// Render a simulated star field based on star data and telescope parameters
fn render_star_field(
    stars: &Vec<&StarData>,
    ra_deg: f64,
    dec_deg: f64,
    fov_deg: f64,
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    exposure: &Duration,
    wavelength_nm: f64,
) -> Array2<u16> {
    // Create image array dimensions
    let image_width = sensor.width_px as usize;
    let image_height = sensor.height_px as usize;

    // Create PSF kernel for the given wavelength
    let psf = approx_airy_pixels(telescope, sensor, wavelength_nm);

    println!("Rendering stars...");

    // Create a star field image (in electron counts)
    let mut image = Array2::zeros((image_height, image_width));
    let mut to_render: Vec<StarInFrame> = Vec::new();

    // Add stars with sub-pixel precision
    for &star in stars {
        // Convert position to pixel coordinates (sub-pixel precision)
        let star_radec = RaDec::from_degrees(star.ra_deg(), star.dec_deg());
        let center_radec = RaDec::from_degrees(ra_deg, dec_deg);
        let (x, y) = equatorial_to_pixel(
            &star_radec,
            &center_radec,
            fov_deg,
            image_width,
            image_height,
        );

        // Skip if outside image bounds
        if x < 0.0 || y < 0.0 || x >= image_width as f64 || y >= image_height as f64 {
            println!(
                "Star {} at RA: {}, DEC: {} is outside image bounds (x: {}, y: {})",
                star.id,
                star.ra_deg(),
                star.dec_deg(),
                x,
                y
            );
            continue;
        }

        // Calculate photon flux using telescope model
        let electrons = magnitude_to_electrons(star.magnitude, exposure, telescope, sensor);

        // Add star to image with PSF
        to_render.push(StarInFrame {
            x,
            y,
            flux: electrons,
        });
    }

    to_render.sort_by(|a, b| a.flux.partial_cmp(&b.flux).unwrap());

    add_stars_to_image(&mut image, to_render, psf);

    // Report the number of electrons before noise
    let total_electrons: f64 = image.sum();
    println!(
        "Total electrons in image before noise: {:.2}e-",
        total_electrons
    );

    // Add noise
    println!("Adding noise...");

    // Generate sensor noise (read noise and dark current)
    let noise = simulator::image_proc::generate_sensor_noise(
        &sensor, &exposure, None, // Use random noise
    );

    // Print the noise statistics
    let total_noise = noise.sum();
    println!("Total noise in image: {:.2}e-", total_noise);

    println!("Approximate SNR: {:.2}", total_electrons / total_noise);

    let mut max_well_clipped = 0;
    // Add sensor noise to the image
    for ((i, j), &noise_val) in noise.indexed_iter() {
        image[[i, j]] += noise_val;

        // Clip possibly negative values
        if image[[i, j]] < 0.0 {
            image[[i, j]] = 0.0; // Clip negative values to zero
        }

        // Clip to maximum well depth
        if image[[i, j]] > sensor.max_well_depth_e {
            image[[i, j]] = sensor.max_well_depth_e; // Clip to max well depth
            max_well_clipped += 1; // Count how many pixels were clipped
        }
    }
    // The image now contains electron counts

    // Print how many pixels were clipped to max well depth
    println!(
        "{} pixels were clipped to maximum well depth of {:.2} electrons",
        max_well_clipped, sensor.max_well_depth_e
    );

    display_electron_histogram(&image, 25).unwrap();

    // Get the DN per electron conversion factor from the sensor
    // Calculate max DN value based on sensor bit depth (saturate at sensor's max value)
    let max_dn = ((1 << sensor.bit_depth) - 1) as f64;

    // Combine conversion, clipping and rounding in a single mapv operation:
    // 1. Convert from electrons to DN
    // 2. Clip to valid range (0 to max DN for the sensor bit depth)
    // 3. Round to nearest integer
    // 4. Convert to u16
    let quantized = image.mapv(|x| {
        let dn = x * sensor.dn_per_electron();
        let clipped = dn.clamp(0.0, max_dn as f64);
        clipped.round() as u16
    });

    quantized
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
    let background_threshold = min_val + (max_val - min_val) * 0.01; // Arbitrary threshold for "background"
    let background_pixels = image.iter().filter(|&&x| x <= background_threshold).count();
    let background_percentage = (background_pixels as f64 / total_pixels as f64) * 100.0;

    // Create enhanced range to make the histogram more informative
    // We'll use a log scale for better visualization of the dynamic range

    // Create a new histogram with viz crate (create multiple histograms for different views)

    // 1. Full range histogram
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
    println!(
        "  Dynamic Range: {:.2}:1",
        if min_val > 0.0 {
            max_val / min_val
        } else {
            f64::INFINITY
        }
    );
    println!(
        "  Background Pixels: {:.1}% ({} pixels)",
        background_percentage, background_pixels
    );

    // Print the histograms
    println!("\n{}", full_hist.with_config(full_config).format()?);

    Ok(())
}

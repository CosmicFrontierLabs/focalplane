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
use simulator::image_proc::convolve2d::gaussian_kernel;
use simulator::image_proc::histogram_stretch::stretch_histogram;
use simulator::image_proc::{save_u8_image, u16_to_u8_auto_scale, u16_to_u8_scaled};
use simulator::{field_diameter, magnitude_to_photon_flux};
use starfield::catalogs::StarData;
use starfield::RaDec;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
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
    #[arg(long)]
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
    #[arg(long, default_value = "1m Final")]
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

    // Output paths
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
        fov_deg,
    )?;

    // Filter to the ones within the rectangle
    let filtered_stars: Vec<&StarData> = stars
        .iter()
        .filter(|star| {
            let (x, y) = equatorial_to_pixel(
                star.ra_deg(),
                star.dec_deg(),
                args.ra,
                args.dec,
                fov_deg,
                sensor.width_px as usize,
                sensor.height_px as usize,
            );
            x >= 0.0 && y >= 0.0 && x < sensor.width_px as f64 && y < sensor.height_px as f64
        })
        .collect();
    println!(
        "Filtered to {} stars in field of view rectangle",
        filtered_stars.len()
    );

    // Render the star field
    println!("Rendering star field...");
    let image = render_star_field(
        &filtered_stars,
        args.ra,
        args.dec,
        fov_deg,
        &telescope,
        &sensor,
        args.exposure,
        args.wavelength,
    );

    // Filter stars visible in the field for star list output
    // Write star list to file
    save_star_list(
        &filtered_stars,
        args.ra,
        args.dec,
        fov_deg,
        &telescope,
        &sensor,
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

/// Create PSF kernel based on telescope properties
fn create_psf_kernel(
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    wavelength_nm: f64,
) -> Array2<f64> {
    // Calculate PSF size based on Airy disk
    let airy_radius_um = telescope.airy_disk_radius_um(wavelength_nm);
    let airy_radius_px = airy_radius_um / sensor.pixel_size_um;

    println!(
        "Airy disk radius: {:.2} um, {:.2} px",
        airy_radius_um, airy_radius_px
    );

    // Create a Gaussian approximation of the Airy disk
    // Using sigma ≈ radius/1.22 to approximate Airy disk with Gaussian
    let sigma = airy_radius_px / 1.22;

    // Create kernel with size = 6*sigma (covers >99.999% of PSF)
    let kernel_size = (6.0 * sigma).ceil() as usize;
    let kernel_size = if kernel_size % 2 == 0 {
        kernel_size + 1 // Ensure odd size for centered kernel
    } else {
        kernel_size
    };

    // Create the Gaussian kernel
    gaussian_kernel(kernel_size, sigma)
}

/// Convert equatorial coordinates to pixel coordinates with sub-pixel precision
fn equatorial_to_pixel(
    ra: f64,
    dec: f64,
    center_ra: f64,
    center_dec: f64,
    fov_deg: f64,
    image_width: usize,
    image_height: usize,
) -> (f64, f64) {
    // Convert to radians
    let ra_rad = ra.to_radians();
    let dec_rad = dec.to_radians();
    let center_ra_rad = center_ra.to_radians();
    let center_dec_rad = center_dec.to_radians();

    // Calculate projection factors
    let x_factor = ra_rad - center_ra_rad;
    let y_factor = dec_rad - center_dec_rad;

    // Scale to pixel coordinates
    let fov_rad = fov_deg.to_radians();
    let x_pixels_per_rad = image_width as f64 / fov_rad;
    let y_pixels_per_rad = image_height as f64 / fov_rad;

    // Convert to pixel coordinates (center of image is at center_ra, center_dec)
    let x = (x_factor * x_pixels_per_rad) + (image_width as f64 / 2.0);
    let y = (y_factor * y_pixels_per_rad) + (image_height as f64 / 2.0);

    (x, y)
}

/// Add a star to the image at specified position with sub-pixel precision
fn add_star_to_image(image: &mut Array2<f64>, x: f64, y: f64, flux: f64, psf: &Array2<f64>) {
    let (image_height, image_width) = image.dim();
    let (kernel_height, kernel_width) = psf.dim();

    // Calculate integer parts of position
    let x_int = x.floor() as isize;
    let y_int = y.floor() as isize;

    // Calculate kernel center
    let k_center_y = kernel_height as isize / 2;
    let k_center_x = kernel_width as isize / 2;

    // Calculate bounds for the kernel application
    let min_y = 0.max(y_int - k_center_y);
    let min_x = 0.max(x_int - k_center_x);
    let max_y = (y_int + k_center_y + 1).min(image_height as isize);
    let max_x = (x_int + k_center_x + 1).min(image_width as isize);

    // Apply PSF * flux to the image with sub-pixel offset
    for iy in min_y..max_y {
        for ix in min_x..max_x {
            // Calculate the kernel position adjusted for sub-pixel offset
            let ky = (iy as f64 - y + k_center_y as f64).round() as isize;
            let kx = (ix as f64 - x + k_center_x as f64).round() as isize;

            if ky >= 0 && kx >= 0 && ky < kernel_height as isize && kx < kernel_width as isize {
                // Apply the flux with corresponding kernel weight
                if iy >= 0 && ix >= 0 && iy < image_height as isize && ix < image_width as isize {
                    image[[iy as usize, ix as usize]] += flux * psf[[ky as usize, kx as usize]];
                }
            }
        }
    }
}

/// Render a simulated star field based on star data and telescope parameters
fn render_star_field(
    stars: &Vec<&StarData>,
    ra_deg: f64,
    dec_deg: f64,
    fov_deg: f64,
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    exposure_time: f64,
    wavelength_nm: f64,
) -> Array2<u16> {
    // Create image array dimensions
    let image_width = sensor.width_px as usize;
    let image_height = sensor.height_px as usize;

    // Create PSF kernel for the given wavelength
    let psf = create_psf_kernel(telescope, sensor, wavelength_nm);

    // Note: QE and effective area are already accounted for in the magnitude_to_photon_flux function

    println!("Rendering stars...");

    // Create a star field image (in electron counts)
    let mut image = Array2::zeros((image_height, image_width));

    // Add stars with sub-pixel precision
    for &star in stars {
        // Convert position to pixel coordinates (sub-pixel precision)
        let (x, y) = equatorial_to_pixel(
            star.ra_deg(),
            star.dec_deg(),
            ra_deg,
            dec_deg,
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
        let flux = magnitude_to_photon_flux(
            star.magnitude,
            exposure_time,
            telescope,
            sensor,
            wavelength_nm,
        );

        // Get quantum efficiency at the specified wavelength
        let qe = sensor.qe_at_wavelength(wavelength_nm as u32); // [dimensionless] (0-1 value)

        // Calculate total photons detected by sensor
        let electron_flux = flux * qe; // [photons]

        // Add star to image with PSF
        add_star_to_image(&mut image, x, y, electron_flux, &psf);
    }

    // Report the number of electrons before noise
    let total_electrons: f64 = image.sum();
    println!(
        "Total electrons in image before noise: {:.2}e-",
        total_electrons
    );

    // Add noise
    println!("Adding noise...");

    // Generate sensor noise (read noise and dark current)
    let (height, width) = image.dim();
    let noise = simulator::image_proc::generate_sensor_noise(
        (height, width),
        sensor.read_noise_e,
        sensor.dark_current_e_p_s,
        exposure_time,
        None, // Use random noise
    );

    // Print the noise statistics
    let total_noise = noise.sum();
    println!("Total noise in image: {:.2}e-", total_noise);

    // Add sensor noise to the image
    for ((i, j), &noise_val) in noise.indexed_iter() {
        image[[i, j]] += noise_val;
    }

    display_electron_histogram(&image, 1000).unwrap();

    // The image now contains electron counts

    // Get the DN per electron conversion factor from the sensor
    let electrons_to_dn = sensor.dn_per_electron_guesstimate();

    // Calculate max DN value based on sensor bit depth (saturate at sensor's max value)
    let max_dn = (1 << sensor.bit_depth) - 1;

    // Combine conversion, clipping and rounding in a single mapv operation:
    // 1. Convert from electrons to DN
    // 2. Clip to valid range (0 to max DN for the sensor bit depth)
    // 3. Round to nearest integer
    // 4. Convert to u16
    let quantized = image.mapv(|x| {
        let dn = x * electrons_to_dn;
        let clipped = dn.max(0.0).min(max_dn as f64);
        clipped.round() as u16
    });

    quantized
}

/// Display a histogram of electron counts in the image
fn display_electron_histogram(
    image: &Array2<f64>,
    num_bins: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get statistics for binning
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
        show_empty_bins: false,
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

/// Save a text file with visible star information
fn save_star_list(
    stars: &[&StarData],
    ra_deg: f64,
    dec_deg: f64,
    fov_deg: f64,
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;

    // Write header
    writeln!(
        file,
        "# Star List for field centered at RA={:.4}°, Dec={:.4}°, FOV={:.4}°",
        ra_deg, dec_deg, fov_deg
    )?;
    writeln!(
        file,
        "# Telescope: {}, Sensor: {}",
        telescope.name, sensor.name
    )?;
    writeln!(file, "# Total stars in field: {}", stars.len())?;
    writeln!(file, "#")?;
    writeln!(file, "# ID, RA(°), Dec(°), Magnitude, B-V, X(px), Y(px)")?;

    // Calculate image dimensions
    let image_width = sensor.width_px as usize;
    let image_height = sensor.height_px as usize;

    // Get stars sorted by magnitude (brightest first)
    let mut sorted_stars = stars.to_vec();
    sorted_stars.sort_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap());

    // Write star data
    for star in sorted_stars {
        // Calculate pixel coordinates
        let (x, y) = equatorial_to_pixel(
            star.ra_deg(),
            star.dec_deg(),
            ra_deg,
            dec_deg,
            fov_deg,
            image_width,
            image_height,
        );

        // Format B-V value or "N/A" if None
        let b_v_str = if let Some(b_v) = star.b_v {
            format!("{:.2}", b_v)
        } else {
            "N/A".to_string()
        };

        writeln!(
            file,
            "{}, {:.6}, {:.6}, {:.2}, {}, {:.2}, {:.2}",
            star.id,
            star.ra_deg(),
            star.dec_deg(),
            star.magnitude,
            b_v_str,
            x,
            y
        )?;
    }

    Ok(())
}

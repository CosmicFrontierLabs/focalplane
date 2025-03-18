//! Enhanced star field renderer for telescope simulations
//!
//! This utility simulates how a telescope would observe a star field by:
//! 1. Filtering stars to those within the field of view
//! 2. Converting star magnitudes to photons, then to electrons on sensor
//! 3. Applying appropriate PSF (Point Spread Function) based on telescope parameters
//! 4. Rendering both the full field of view and the sensor boundaries
//! 5. Saving the complete visualization to disk

use image::{DynamicImage, Rgb, RgbImage};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use simulator::hardware::sensor::{models as sensor_models, SensorConfig};
use simulator::hardware::star_projection;
use simulator::hardware::telescope::{models as telescope_models, TelescopeConfig};
use simulator::image_proc::convolve2d::{
    convolve2d, gaussian_kernel, ConvolveMode, ConvolveOptions,
};
use usvg::fontdb;

/// Main function to render an enhanced simulated star field
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Default values
    let mut ra_deg = 100.0;
    let mut dec_deg = 45.0;
    let mut fov_deg = 1.0;
    let mut num_stars = 100;
    let mut seed = 42u64;
    let mut exposure_seconds = 1.0;
    let mut output_path = "test_output/enhanced_star_field.png";
    let mut field_expansion_factor = 2.0; // Factor to expand beyond sensor FOV

    // Parse arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ra" => {
                if i + 1 < args.len() {
                    ra_deg = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --ra".into());
                }
            }
            "--dec" => {
                if i + 1 < args.len() {
                    dec_deg = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --dec".into());
                }
            }
            "--fov" => {
                if i + 1 < args.len() {
                    fov_deg = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --fov".into());
                }
            }
            "--stars" => {
                if i + 1 < args.len() {
                    num_stars = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --stars".into());
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    seed = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --seed".into());
                }
            }
            "--exposure" => {
                if i + 1 < args.len() {
                    exposure_seconds = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --exposure".into());
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = &args[i + 1];
                    i += 2;
                } else {
                    return Err("Missing value for --output".into());
                }
            }
            "--expand" => {
                if i + 1 < args.len() {
                    field_expansion_factor = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --expand".into());
                }
            }
            _ => {
                println!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    // Show parameters and help
    println!("Enhanced Star Field Renderer");
    println!("===========================");
    println!("RA: {:.4}°", ra_deg);
    println!("Dec: {:.4}°", dec_deg);
    println!("FOV: {:.4}°", fov_deg);
    println!("Stars: {}", num_stars);
    println!("Seed: {}", seed);
    println!("Exposure: {:.2} seconds", exposure_seconds);
    println!("Field expansion factor: {}", field_expansion_factor);
    println!("Output: {}", output_path);

    // Set up telescope and sensor configurations
    let telescope = telescope_models::FINAL_1M.clone();
    let sensor = sensor_models::GSENSE4040BSI.clone();

    println!("Telescope: {}", telescope.name);
    println!("Sensor: {}", sensor.name);

    // Generate synthetic stars
    println!("Generating synthetic stars...");
    // Use expanded FOV to include stars outside sensor FOV but within the larger visualization
    let expanded_fov = fov_deg * field_expansion_factor;
    let stars = generate_synthetic_stars(num_stars, ra_deg, dec_deg, expanded_fov, seed);
    println!("Generated {} synthetic stars", stars.len());

    // Render the enhanced star field
    println!("Rendering enhanced star field visualization...");
    let visualization = render_enhanced_star_field(
        &stars,
        ra_deg,
        dec_deg,
        fov_deg,
        expanded_fov,
        &telescope,
        &sensor,
        exposure_seconds,
    )?;

    // Save the image
    visualization.save(output_path)?;
    println!("Enhanced visualization saved to: {}", output_path);

    Ok(())
}

/// Generate synthetic stars for testing
fn generate_synthetic_stars(
    count: usize,
    center_ra: f64,
    center_dec: f64,
    fov_deg: f64,
    seed: u64,
) -> Vec<SyntheticStar> {
    use rand::distributions::{Distribution, Uniform};

    // Create a seeded RNG for reproducible results
    let mut rng = StdRng::seed_from_u64(seed);
    let mut stars = Vec::with_capacity(count);

    // Half FOV in degrees
    let half_fov = fov_deg / 2.0;

    // Distributions for random star positions and magnitudes
    let ra_dist = Uniform::from(center_ra - half_fov..center_ra + half_fov);
    let dec_dist = Uniform::from(center_dec - half_fov..center_dec + half_fov);

    // For realistic magnitude distribution, we use exponential distribution
    // For every step in magnitude, there are ~1.5x more stars (50% increase)
    let min_mag = 3.0; // Brightest stars (lower magnitude = brighter)
    let max_mag = 8.0; // Dimmest stars

    // We'll generate random values and transform them to follow stellar magnitude distribution
    let uniform = Uniform::from(0.0..1.0);

    for id in 1..=count {
        let ra = ra_dist.sample(&mut rng);
        let dec = dec_dist.sample(&mut rng);

        // Generate magnitude using the exponential distribution
        // Start with uniform random value
        let u = uniform.sample(&mut rng);

        // Transform uniform distribution to exponential distribution
        // Using the fact that for every magnitude step, we have 1.5× more stars
        // This means number of stars grows as 1.5^magnitude
        let log_base: f64 = 1.5; // 50% more stars per magnitude
        let exp_range = log_base.powf(max_mag - min_mag) - 1.0;
        let t: f64 = u * exp_range + 1.0; // Transform to [1, 1.5^range]

        // Convert back to magnitude scale
        let magnitude = min_mag + t.log(log_base).clamp(0.0, max_mag - min_mag);

        stars.push(SyntheticStar {
            id: id as u64,
            ra,
            dec,
            magnitude,
        });
    }

    stars
}

/// Synthetic star data for testing without a real catalog
#[derive(Debug, Clone)]
struct SyntheticStar {
    /// Star identifier
    pub id: u64,
    /// Right ascension in degrees
    pub ra: f64,
    /// Declination in degrees
    pub dec: f64,
    /// Apparent magnitude
    pub magnitude: f64,
}

impl starfield::catalogs::StarPosition for SyntheticStar {
    fn ra(&self) -> f64 {
        self.ra
    }

    fn dec(&self) -> f64 {
        self.dec
    }
}

/// Convert photons to electrons based on quantum efficiency
fn photons_to_electrons(photons: f64, qe: f64) -> f64 {
    // Apply quantum efficiency to convert photons to electrons
    // In reality, this is a stochastic process, but we approximate with expectation value
    photons * qe
}

/// Create a PSF kernel based on telescope properties
fn create_psf_kernel(
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    wavelength_nm: f64,
) -> Array2<f64> {
    // Calculate PSF size
    let airy_radius_um = telescope.airy_disk_radius_um(wavelength_nm);
    let airy_radius_px = airy_radius_um / sensor.pixel_size_um;

    // Create a Gaussian approximation of the Airy disk
    // Using sigma ≈ radius/1.22 to approximate the Airy disk with a Gaussian
    let sigma = airy_radius_px / 1.22;

    // Create a kernel with size = 3*sigma (covers >99% of the PSF)
    let kernel_size = (3.0 * sigma).ceil() as usize;
    let kernel_size = if kernel_size % 2 == 0 {
        kernel_size + 1
    } else {
        kernel_size
    };

    // Create the Gaussian kernel
    gaussian_kernel(kernel_size, sigma)
}

/// Add noise to the image (simulating sensor noise)
fn add_noise(image: &mut Array2<f64>, read_noise: f64, dark_current: f64, exposure_time: f64) {
    let mut rng = thread_rng();

    // Calculate dark current noise for the given exposure time
    let dark_noise = dark_current * exposure_time;

    for v in image.iter_mut() {
        // Add Poisson noise to simulate photon counting (shot noise)
        if *v > 0.0 {
            // Approximate Poisson with Gaussian for large values
            let photon_noise = (*v).sqrt();
            *v += rng.gen_range(-photon_noise..photon_noise);
        }

        // Add read noise
        *v += rng.gen_range(-read_noise..read_noise);

        // Add dark current noise
        *v += rng.gen_range(0.0..dark_noise * 2.0) - dark_noise;

        // Ensure no negative values (electrons cannot be negative)
        *v = v.max(0.0);
    }
}

/// Convert equatorial coordinates to pixel coordinates
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

    // Convert to pixel coordinates with center of image as origin
    let x = (x_factor * x_pixels_per_rad) + (image_width as f64 / 2.0);
    let y = (y_factor * y_pixels_per_rad) + (image_height as f64 / 2.0);

    (x, y)
}

/// Draw colored overlay elements on an image using SVG
fn draw_svg_overlay(
    base_image: &DynamicImage,
    sensor_bounds: (f64, f64, f64, f64), // (min_x, min_y, width, height)
    field_circle_center: (f64, f64),
    field_circle_radius: f64,
) -> DynamicImage {
    let width = base_image.width();
    let height = base_image.height();

    // Create SVG overlay
    let mut svg_data = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
        "#,
        width, height
    );

    // Draw the circular field of view (sensor's full FOV)
    svg_data.push_str(&format!(
        r#"<circle cx="{}" cy="{}" r="{}" fill="none" stroke="cyan" stroke-width="2" stroke-dasharray="5,5"/>"#,
        field_circle_center.0, field_circle_center.1, field_circle_radius
    ));

    // Draw the sensor boundary rectangle
    svg_data.push_str(&format!(
        r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="magenta" stroke-width="2"/>"#,
        sensor_bounds.0, sensor_bounds.1, sensor_bounds.2, sensor_bounds.3
    ));

    // Add explanatory text labels
    svg_data.push_str(&format!(
        r#"<text x="{}" y="{}" font-family="monospace" font-size="14" fill="cyan">Sensor FOV</text>"#,
        field_circle_center.0, field_circle_center.1 - field_circle_radius - 10.0
    ));

    svg_data.push_str(&format!(
        r#"<text x="{}" y="{}" font-family="monospace" font-size="14" fill="magenta">Sensor Bounds</text>"#,
        sensor_bounds.0, sensor_bounds.1 - 10.0
    ));

    svg_data.push_str("</svg>");

    // Render the SVG overlay onto the base image
    overlay_to_image(base_image, &svg_data)
}

/// Convert an SVG overlay to an image
fn overlay_to_image(image: &DynamicImage, svg_data: &str) -> DynamicImage {
    // Create font database with system fonts
    let mut fontdb = fontdb::Database::new();
    fontdb.load_system_fonts();

    // Create options with the font database
    let mut options = usvg::Options::default();
    options.fontdb = std::sync::Arc::new(fontdb);
    options.font_family = "DejaVu Sans".to_string(); // Widely available on Linux

    // Parse SVG with our custom options
    let svg_tree = match usvg::Tree::from_str(svg_data, &options) {
        Ok(tree) => tree,
        Err(e) => {
            println!("Failed to parse SVG: {:?}", e);
            panic!("SVG parsing failed");
        }
    };

    // Create a pixel buffer for the overlay
    let mut pixmap = match tiny_skia::Pixmap::new(image.width(), image.height()) {
        Some(p) => p,
        None => {
            println!("Failed to create pixel buffer");
            panic!("Pixmap creation failed");
        }
    };

    // Render SVG to the pixel buffer
    resvg::render(
        &svg_tree,
        tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );

    // Convert original image to RGB format
    let rgb_image: RgbImage = image.to_rgb8();
    let mut output_buffer = RgbImage::new(image.width(), image.height());

    // Combine original image with overlay
    for (x, y, pixel) in output_buffer.enumerate_pixels_mut() {
        let source_pixel = rgb_image.get_pixel(x, y);

        // Get SVG pixel at this position
        if let Some(overlay_pixel) = pixmap.pixel(x, y) {
            // Only apply overlay if it's not fully transparent
            if overlay_pixel.alpha() > 0 {
                *pixel = Rgb([
                    blend_channel(source_pixel[0], overlay_pixel.red(), overlay_pixel.alpha()),
                    blend_channel(
                        source_pixel[1],
                        overlay_pixel.green(),
                        overlay_pixel.alpha(),
                    ),
                    blend_channel(source_pixel[2], overlay_pixel.blue(), overlay_pixel.alpha()),
                ]);
            } else {
                *pixel = *source_pixel;
            }
        } else {
            *pixel = *source_pixel;
        }
    }

    DynamicImage::ImageRgb8(output_buffer)
}

// Helper function to blend color channels based on alpha
fn blend_channel(base: u8, overlay: u8, alpha: u8) -> u8 {
    let base_f = base as f32;
    let overlay_f = overlay as f32;
    let alpha_f = alpha as f32 / 255.0;

    // Alpha blending formula
    (base_f * (1.0 - alpha_f) + overlay_f * alpha_f).round() as u8
}

/// Render an enhanced star field visualization
fn render_enhanced_star_field(
    stars: &[SyntheticStar],
    ra_deg: f64,
    dec_deg: f64,
    sensor_fov_deg: f64,
    visualization_fov_deg: f64,
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    exposure_time: f64,
) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    // Filter stars in the expanded field of view
    println!("Filtering stars in expanded field of view...");
    let visible_stars =
        star_projection::filter_stars_in_field(stars, ra_deg, dec_deg, visualization_fov_deg);
    println!("Found {} stars in expanded field", visible_stars.len());

    // Set up visualization dimensions
    let visualization_width = 1200; // Fixed size for visualization
    let visualization_height = 1200;

    // Create a larger visualization array for the expanded field
    let mut visualization_array = Array2::zeros((visualization_height, visualization_width));

    // Calculate sensor boundaries relative to visualization
    let scaling_factor = visualization_fov_deg / sensor_fov_deg;
    let sensor_width_px = (sensor.width_px as f64 / scaling_factor) as usize;
    let sensor_height_px = (sensor.height_px as f64 / scaling_factor) as usize;

    // Center sensor in visualization
    let sensor_min_x = (visualization_width - sensor_width_px) / 2;
    let sensor_min_y = (visualization_height - sensor_height_px) / 2;
    let sensor_max_x = sensor_min_x + sensor_width_px;
    let sensor_max_y = sensor_min_y + sensor_height_px;

    // Create a separate array just for the sensor image
    let mut sensor_array = Array2::zeros((sensor.height_px as usize, sensor.width_px as usize));

    // Create PSF kernel for typical visible light wavelength (550nm = green light)
    let wavelength_nm = 550.0;
    let psf = create_psf_kernel(telescope, sensor, wavelength_nm);

    // Get quantum efficiency for this wavelength
    let qe = sensor.qe_at_wavelength(wavelength_nm as u32);

    // Calculate telescope effective area
    let effective_area = telescope.effective_collecting_area_m2();
    println!(
        "Telescope effective collecting area: {:.4} m²",
        effective_area
    );

    println!("Rendering stars across full field and sensor...");

    // Process each visible star
    for star in visible_stars {
        // Convert position to visualization pixel coordinates
        let (viz_x, viz_y) = equatorial_to_pixel(
            star.ra,
            star.dec,
            ra_deg,
            dec_deg,
            visualization_fov_deg,
            visualization_width,
            visualization_height,
        );

        // Skip stars outside the visualization bounds
        if viz_x < 0.0
            || viz_x >= visualization_width as f64
            || viz_y < 0.0
            || viz_y >= visualization_height as f64
        {
            continue;
        }

        // Convert magnitude to photon flux
        let photon_flux = star_projection::magnitude_to_photon_flux(
            star.magnitude,
            exposure_time,
            telescope,
            sensor,
            wavelength_nm,
        );

        // Convert photons to electrons
        let electron_flux = photons_to_electrons(photon_flux, qe);

        // Add star to visualization array (integer coordinates)
        let viz_x_int = viz_x.round() as usize;
        let viz_y_int = viz_y.round() as usize;

        if viz_x_int < visualization_width && viz_y_int < visualization_height {
            visualization_array[[viz_y_int, viz_x_int]] += electron_flux;
        }

        // Check if star is within sensor boundaries
        if viz_x >= sensor_min_x as f64
            && viz_x < sensor_max_x as f64
            && viz_y >= sensor_min_y as f64
            && viz_y < sensor_max_y as f64
        {
            // Convert to sensor coordinates
            let sensor_x = (viz_x - sensor_min_x as f64) * scaling_factor;
            let sensor_y = (viz_y - sensor_min_y as f64) * scaling_factor;

            // If within sensor bounds, add to sensor array
            let sensor_x_int = sensor_x.round() as usize;
            let sensor_y_int = sensor_y.round() as usize;

            if sensor_x_int < sensor.width_px as usize && sensor_y_int < sensor.height_px as usize {
                sensor_array[[sensor_y_int, sensor_x_int]] += electron_flux;
            }
        }
    }

    // Apply PSF and noise to sensor image
    println!("Applying PSF to sensor image...");
    let options = ConvolveOptions {
        mode: ConvolveMode::Same,
    };

    // Convolve sensor image with PSF
    let mut sensor_image = convolve2d(&sensor_array.view(), &psf.view(), Some(options));

    // Add noise to sensor image
    println!("Adding sensor noise...");
    add_noise(
        &mut sensor_image,
        sensor.read_noise_e,
        sensor.dark_current_e_p_s,
        exposure_time,
    );

    // Also apply a simple PSF to visualization image (less detailed than sensor)
    println!("Applying simplified PSF to visualization image...");
    // Use a simplified PSF for visualization (faster)
    let viz_psf_size = 5; // smaller kernel for visualization
    let viz_psf = gaussian_kernel(viz_psf_size, 1.5);

    let viz_options = ConvolveOptions {
        mode: ConvolveMode::Same,
    };

    let viz_image = convolve2d(
        &visualization_array.view(),
        &viz_psf.view(),
        Some(viz_options),
    );

    // Convert the visualization array to an image
    println!("Converting to RGB visualization...");

    // Find max values for normalization
    let viz_max = viz_image.iter().fold(0.0, |max, &x| f64::max(max, x));
    let sensor_max = sensor_image.iter().fold(0.0, |max, &x| f64::max(max, x));

    // Create color image for main visualization
    let mut rgb_viz = RgbImage::new(visualization_width as u32, visualization_height as u32);

    // Fill visualization with bluish stars on black background
    for (x, y, pixel) in rgb_viz.enumerate_pixels_mut() {
        let normalized_value = viz_image[[y as usize, x as usize]] / viz_max;

        // Use white-blue color gradient for stars
        let blue_value = (normalized_value * 255.0) as u8;
        let white_value = ((normalized_value.powf(1.5)) * 255.0) as u8;

        *pixel = Rgb([white_value, white_value, blue_value.max(white_value)]);
    }

    // Convert sensor image to grayscale
    let mut sensor_rgb = RgbImage::new(sensor.width_px, sensor.height_px);

    // Fill sensor image with grayscale values
    for y in 0..sensor.height_px {
        for x in 0..sensor.width_px {
            let normalized_value = sensor_image[[y as usize, x as usize]] / sensor_max;
            let gray_value = (normalized_value * 255.0) as u8;
            sensor_rgb.put_pixel(x, y, Rgb([gray_value, gray_value, gray_value]));
        }
    }

    // Convert to DynamicImage
    let viz_dynamic = DynamicImage::ImageRgb8(rgb_viz);

    // Insert sensor image into visualization at the correct position
    let mut viz_canvas = viz_dynamic.to_rgb8();

    // Copy sensor image onto visualization (with a scaling factor to make it visible)
    let sensor_scale = 0.25; // Scale down sensor image to fit in visualization
    let scaled_sensor_width = (sensor.width_px as f64 * sensor_scale) as u32;
    let scaled_sensor_height = (sensor.height_px as f64 * sensor_scale) as u32;

    // Position at bottom-right corner with padding
    let sensor_thumbnail_x = viz_dynamic.width() - scaled_sensor_width - 20;
    let sensor_thumbnail_y = viz_dynamic.height() - scaled_sensor_height - 20;

    // Create a scaled version of the sensor image
    let scaled_sensor = image::imageops::resize(
        &sensor_rgb,
        scaled_sensor_width,
        scaled_sensor_height,
        image::imageops::FilterType::Triangle,
    );

    // Overlay the sensor thumbnail onto visualization
    image::imageops::overlay(
        &mut viz_canvas,
        &scaled_sensor,
        sensor_thumbnail_x as i64,
        sensor_thumbnail_y as i64,
    );

    // Draw sensor FOV circle and sensor boundaries
    let viz_dynamic_updated = DynamicImage::ImageRgb8(viz_canvas);

    // Calculate sensor field circle parameters
    let sensor_fov_radius = (visualization_width.min(visualization_height) as f64 / 2.0)
        * (sensor_fov_deg / visualization_fov_deg);

    // Add overlays
    let result = draw_svg_overlay(
        &viz_dynamic_updated,
        (
            sensor_min_x as f64,
            sensor_min_y as f64,
            sensor_width_px as f64,
            sensor_height_px as f64,
        ),
        (
            visualization_width as f64 / 2.0,
            visualization_height as f64 / 2.0,
        ),
        sensor_fov_radius,
    );

    Ok(result)
}

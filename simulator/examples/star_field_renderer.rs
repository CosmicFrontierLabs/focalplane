//! Star field renderer for simulating telescope images
//!
//! This utility simulates how a telescope would observe a star field by:
//! 1. Filtering stars to those within the field of view
//! 2. Converting star magnitudes to flux values
//! 3. Applying appropriate PSF (Point Spread Function) based on telescope parameters
//! 4. Rendering stars to a final image
//! 5. Generating a histogram of star magnitudes in the field

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use simulator::hardware::sensor::{models as sensor_models, SensorConfig};
use simulator::hardware::telescope::{models as telescope_models, TelescopeConfig};
use simulator::image_proc::convolve2d::{
    convolve2d, gaussian_kernel, ConvolveMode, ConvolveOptions,
};
use viz::histogram::{Histogram, HistogramConfig, Scale};

/// Main function to render a simulated star field
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Default values
    let mut ra_deg = 100.0;
    let mut dec_deg = 45.0;
    let mut fov_deg = 1.0;
    let mut num_stars = 100;
    let mut seed = 42u64;
    let mut output_path = "star_field.png";

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
            "--output" => {
                if i + 1 < args.len() {
                    output_path = &args[i + 1];
                    i += 2;
                } else {
                    return Err("Missing value for --output".into());
                }
            }
            _ => {
                println!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    // Show parameters and help
    println!("Star Field Renderer");
    println!("===================");
    println!("RA: {:.4}°", ra_deg);
    println!("Dec: {:.4}°", dec_deg);
    println!("FOV: {:.4}°", fov_deg);
    println!("Stars: {}", num_stars);
    println!("Seed: {}", seed);
    println!("Output: {}", output_path);

    // Set up telescope and sensor configurations
    let telescope = telescope_models::FINAL_1M.clone();
    let sensor = sensor_models::GSENSE4040BSI.clone();

    println!("Telescope: {}", telescope.name);
    println!("Sensor: {}", sensor.name);

    // Generate synthetic stars
    println!("Generating synthetic stars...");
    let stars = generate_synthetic_stars(num_stars, ra_deg, dec_deg, fov_deg, seed);
    println!("Generated {} synthetic stars", stars.len());

    // Render the star field
    println!("Rendering star field...");
    let image = render_star_field(&stars, ra_deg, dec_deg, fov_deg, &telescope, &sensor);

    // Save the image
    save_image(&image, output_path)?;
    println!("Image saved to: {}", output_path);

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
    // Use exponential sampling to generate realistic stellar magnitudes

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

/// Filter stars to those visible in the field of view
fn filter_stars_in_fov(
    stars: &[SyntheticStar],
    center_ra: f64,
    center_dec: f64,
    fov_deg: f64,
) -> Vec<&SyntheticStar> {
    // Convert FOV to radians for calculations
    let fov_rad = fov_deg.to_radians();
    let center_ra_rad = center_ra.to_radians();
    let center_dec_rad = center_dec.to_radians();

    // Filter stars that fall within the FOV
    stars
        .iter()
        .filter(|star| {
            let star_ra_rad = star.ra.to_radians();
            let star_dec_rad = star.dec.to_radians();

            // Calculate angular distance between star and center
            // Using the haversine formula for great-circle distance
            let d_ra = star_ra_rad - center_ra_rad;
            let d_dec = star_dec_rad - center_dec_rad;

            let a = (d_dec / 2.0).sin().powi(2)
                + center_dec_rad.cos() * star_dec_rad.cos() * (d_ra / 2.0).sin().powi(2);
            let angular_distance = 2.0 * a.sqrt().asin();

            // Check if star is within FOV/2 (circular FOV)
            angular_distance <= fov_rad / 2.0
        })
        .collect()
}

/// Convert star magnitude to flux
fn magnitude_to_flux(magnitude: f64) -> f64 {
    // Pogson equation: flux = 10^(10 - 0.4 * magnitude)
    10.0_f64.powf(10.0 - 0.4 * magnitude)
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

/// Convert equatorial coordinates to pixel coordinates
fn equatorial_to_pixel(
    ra: f64,
    dec: f64,
    center_ra: f64,
    center_dec: f64,
    fov_deg: f64,
    image_width: usize,
    image_height: usize,
) -> (usize, usize) {
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

    // Convert to pixel coordinates
    let x = ((x_factor * x_pixels_per_rad) + (image_width as f64 / 2.0)).round() as i64;
    let y = ((y_factor * y_pixels_per_rad) + (image_height as f64 / 2.0)).round() as i64;

    // Clamp to image bounds
    let x = x.clamp(0, image_width as i64 - 1) as usize;
    let y = y.clamp(0, image_height as i64 - 1) as usize;

    (x, y)
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

/// Add a star to the image at specified position with specified flux
fn add_star_to_image(image: &mut Array2<f64>, x: usize, y: usize, flux: f64, psf: &Array2<f64>) {
    let (image_height, image_width) = image.dim();
    let (kernel_height, kernel_width) = psf.dim();

    // Calculate kernel center
    let k_center_y = kernel_height / 2;
    let k_center_x = kernel_width / 2;

    // Calculate bounds for the kernel application
    let min_y = y.saturating_sub(k_center_y);
    let min_x = x.saturating_sub(k_center_x);
    let max_y = (y + k_center_y + 1).min(image_height);
    let max_x = (x + k_center_x + 1).min(image_width);

    // Apply PSF * flux to the image
    for iy in min_y..max_y {
        for ix in min_x..max_x {
            let ky = iy as isize - y as isize + k_center_y as isize;
            let kx = ix as isize - x as isize + k_center_x as isize;

            if ky >= 0 && kx >= 0 && ky < kernel_height as isize && kx < kernel_width as isize {
                image[[iy, ix]] += flux * psf[[ky as usize, kx as usize]];
            }
        }
    }
}

/// Add noise to the image
fn add_noise(image: &mut Array2<f64>, read_noise: f64, dark_current: f64, exposure_time: f64) {
    let mut rng = thread_rng();

    // Calculate dark current noise for the given exposure time
    let dark_noise = dark_current * exposure_time;

    for v in image.iter_mut() {
        // Add Poisson noise to simulate photon counting
        if *v > 0.0 {
            // Approximate Poisson with Gaussian for large values
            let photon_noise = (*v).sqrt();
            *v += rng.gen_range(-photon_noise..photon_noise);
        }

        // Add read noise
        *v += rng.gen_range(-read_noise..read_noise);

        // Add dark current noise
        *v += rng.gen_range(0.0..dark_noise * 2.0) - dark_noise;

        // Ensure no negative values
        *v = v.max(0.0);
    }
}

/// Render a simulated star field based on synthetic stars and telescope parameters
fn render_star_field(
    stars: &[SyntheticStar],
    ra_deg: f64,
    dec_deg: f64,
    fov_deg: f64,
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
) -> Array2<f64> {
    // Filter to stars in field of view
    println!("Filtering stars in field of view...");
    let visible_stars = filter_stars_in_fov(stars, ra_deg, dec_deg, fov_deg);
    println!("Found {} stars in field of view", visible_stars.len());

    // Create image array dimensions
    let image_width = sensor.width_px as usize;
    let image_height = sensor.height_px as usize;

    // Create PSF kernel for typical visible light wavelength
    let psf = create_psf_kernel(telescope, sensor, 550.0); // 550nm green light

    // Get average QE (quantum efficiency) for visible spectrum
    let qe = sensor.qe_at_wavelength(550);

    // Simulate exposure time (seconds)
    let exposure_time = 1.0;

    // Calculate telescope effective area
    let effective_area = telescope.effective_collecting_area_m2();

    println!("Rendering stars...");

    // Create a star field without PSF application
    let mut star_points = Array2::zeros((image_height, image_width));

    // Add stars as point sources
    for star in visible_stars {
        // Convert position to pixel coordinates
        let (x, y) = equatorial_to_pixel(
            star.ra,
            star.dec,
            ra_deg,
            dec_deg,
            fov_deg,
            image_width,
            image_height,
        );

        // Convert magnitude to flux
        let base_flux = magnitude_to_flux(star.magnitude);

        // Scale flux based on telescope and sensor properties
        // This is a simplified model - real implementation would be more complex
        let scaled_flux = base_flux * effective_area * qe * exposure_time;

        // Add point source
        star_points[[y, x]] += scaled_flux;
    }

    // Apply PSF to the entire image at once using convolution
    println!("Applying PSF via convolution...");
    let options = ConvolveOptions {
        mode: ConvolveMode::Same,
    };

    // Convolve the point sources with the PSF
    let mut image = convolve2d(&star_points.view(), &psf.view(), Some(options));

    // Add noise
    println!("Adding noise...");
    add_noise(
        &mut image,
        sensor.read_noise_e,
        sensor.dark_current_e_p_s,
        exposure_time,
    );

    // Normalize image (for visualization)
    let max_value = image
        .iter()
        .fold(0.0, |max, &x| if x > max { x } else { max });
    if max_value > 0.0 {
        for v in image.iter_mut() {
            *v /= max_value;
        }
    }

    image
}

/// Save the rendered image to a file
fn save_image(image: &Array2<f64>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use image::{ImageBuffer, Luma};

    let (height, width) = image.dim();

    // Create an 8-bit grayscale image
    let mut img_buffer = ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
        let value = (image[[y as usize, x as usize]] * 255.0)
            .min(255.0)
            .max(0.0) as u8;
        *pixel = Luma([value]);
    }

    img_buffer.save(path)?;

    Ok(())
}

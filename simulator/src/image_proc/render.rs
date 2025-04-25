use std::time::Duration;

use ndarray::Array2;
use starfield::{catalogs::StarData, RaDec};

use crate::{
    algo::icp::Locatable2d, field_diameter, magnitude_to_electrons, star_math::equatorial_to_pixel,
    SensorConfig, TelescopeConfig,
};

use super::generate_sensor_noise;

#[derive(Clone, Debug)]
pub struct StarInFrame {
    pub x: f64,
    pub y: f64,
    pub flux: f64,
    pub star: StarData,
}

// Implement locatable so we can ICP them together with the segmentation results
impl Locatable2d for StarInFrame {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }
}

#[derive(Clone)]
pub struct RenderingResult {
    /// The final rendered image AU quantized etc (u16)
    pub image: Array2<u16>,

    /// The approximate number of electrons on the sensor including noise
    /// and max well depth effects.
    pub electron_image: Array2<f64>,

    /// The noise image (in electron counts)
    /// This includes read noise, dark current
    pub noise_image: Array2<f64>,

    /// The stars that were rendered in the image (not clipped)
    pub rendered_stars: Vec<StarInFrame>,

    /// Number of pixels that were clipped to maximum well depth
    /// This is useful for understanding saturation effects
    pub n_clipped: u32,
}

/// Create gaussian PSF kernel based on telescope properties
pub fn approx_airy_pixels(
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    wavelength_nm: f64,
) -> f64 {
    // Calculate PSF size based on Airy disk
    let airy_radius_um = telescope.airy_disk_radius_um(wavelength_nm);
    let airy_radius_px = airy_radius_um / sensor.pixel_size_um;

    // Create a Gaussian approximation of the Airy disk
    // Using sigma ≈ radius/1.22 to approximate Airy disk with Gaussian
    airy_radius_px / 1.22
}

/// Renders a simulated star field based on catalog data and optical system parameters.
///
/// This function simulates how stars would appear through a telescope onto a digital sensor.
/// It performs the following steps:
/// - Converts star equatorial coordinates (RA/Dec) to pixel coordinates
/// - Calculates the expected electron count for each star based on magnitude, exposure and system parameters
/// - Applies a Gaussian approximation of Airy disk as point spread function (PSF)
/// - Adds realistic sensor noise (read noise and dark current)
/// - Applies sensor physical limitations (well depth saturation)
/// - Converts electron counts to digital numbers (DN)
///
/// # Arguments
/// * `stars` - Reference to a vector of StarData pointers containing catalog information
/// * `center` - RaDec coordinates of field center
/// * `telescope` - Reference to telescope configuration
/// * `sensor` - Reference to sensor configuration
/// * `exposure` - Reference to exposure duration
/// * `wavelength_nm` - Wavelength in nanometers to use for PSF calculation
///
/// # Returns
/// * `RenderingResult` - Contains the rendered image, electron counts, noise, star positions, and saturation info
pub fn render_star_field(
    stars: &Vec<&StarData>,
    center: &RaDec,
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    exposure: &Duration,
    wavelength_nm: f64,
) -> RenderingResult {
    // Create image array dimensions
    let image_width = sensor.width_px as usize;
    let image_height = sensor.height_px as usize;

    // Create a star field image (in electron counts)
    let mut image = Array2::zeros((image_height, image_width));
    let mut to_render: Vec<StarInFrame> = Vec::new();

    let mut xy_mag = Vec::with_capacity(stars.len());

    let psf_pix = approx_airy_pixels(telescope, sensor, wavelength_nm);

    // Calculate field of view from telescope and sensor
    let fov_deg = field_diameter(telescope, sensor);

    // Padded bounds check
    let padding = psf_pix * 2.0;

    // Add stars with sub-pixel precision
    for &star in stars {
        // Convert position to pixel coordinates (sub-pixel precision)
        let star_radec = RaDec::from_degrees(star.ra_deg(), star.dec_deg());
        let (x, y) = equatorial_to_pixel(&star_radec, center, fov_deg, image_width, image_height);

        // Check if star is within the image bounds
        if x < -padding
            || y < -padding
            || x >= image_width as f64 + padding
            || y >= image_height as f64 + padding
        {
            continue; // Skip stars outside the image
        }

        // Post transfrom/selection
        xy_mag.push((x, y, star.clone()));

        // Calculate photon flux using telescope model
        let electrons = magnitude_to_electrons(star.magnitude, exposure, telescope, sensor);

        // Add star to image with PSF
        to_render.push(StarInFrame {
            x,
            y,
            flux: electrons,
            star: star.clone(),
        });
    }

    to_render.sort_by(|a, b| a.flux.partial_cmp(&b.flux).unwrap());

    // Create PSF kernel for the given wavelength
    add_stars_to_image(&mut image, &to_render, psf_pix);

    // Generate sensor noise (read noise and dark current)
    let noise = generate_sensor_noise(
        &sensor, &exposure, None, // Use random noise
    );

    let mut max_well_clipped = 0;
    // Add sensor noise to the image
    for ((i, j), &noise_val) in noise.indexed_iter() {
        image[[i, j]] += noise_val;

        // Clip to maximum well depth
        if image[[i, j]] > sensor.max_well_depth_e {
            image[[i, j]] = sensor.max_well_depth_e; // Clip to max well depth
            max_well_clipped += 1; // Count how many pixels were clipped
        }
    }

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

    RenderingResult {
        image: quantized,
        electron_image: image,
        noise_image: noise,
        n_clipped: max_well_clipped,
        rendered_stars: to_render,
    }
}

/// Adds stars to an image by approximating a Gaussian point spread function (PSF).
///
/// This function takes a mutable reference to an image and adds the flux contribution
/// of each star to the appropriate pixels based on a Gaussian PSF with the specified sigma.
///
/// # Arguments
/// * `image` - A mutable reference to the 2D array representing the image
/// * `stars` - A vector of StarInFrame objects containing position and flux information
/// * `sigma_pix` - The standard deviation of the Gaussian PSF in pixels
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// use simulator::image_proc::render::{add_stars_to_image, StarInFrame};
/// use starfield::catalogs::StarData;
/// use starfield::RaDec;
/// let star_data = StarData {
///     id: 0,
///     magnitude: 10.0,
///     position: RaDec::from_degrees(0.0, 0.0),
///     b_v: None,
/// };
/// let mut image = Array2::zeros((100, 100));
/// let stars = vec![StarInFrame { x: 50.0, y: 50.0, flux: 1000.0, star: star_data }];
/// add_stars_to_image(&mut image, &stars, 2.0);
/// ```
pub fn add_stars_to_image(image: &mut Array2<f64>, stars: &Vec<StarInFrame>, sigma_pix: f64) {
    // 4 std's is a good approximation for the PSF
    let max_pix_dist = (sigma_pix.max(1.0) * 4.0).ceil() as i32;

    let (width, height) = image.dim();
    let c = sigma_pix * sigma_pix * 2.0;

    let pre_term = 1.0 / (2.0 * sigma_pix * sigma_pix * std::f64::consts::PI);

    // Calculate the contribution of all stars to this pixel
    for star in stars {
        // Calculate distance from star to pixel
        let xc = star.x.round() as i32;
        let yc = star.y.round() as i32;

        for y in (xc - max_pix_dist)..=(xc + max_pix_dist) {
            for x in (yc - max_pix_dist)..=(yc + max_pix_dist) {
                // Bounds check x/y - Skip out of bounds pixels
                if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                    continue;
                }

                let dx = star.x - y as f64;
                let dy = star.y - x as f64;
                let distance_squared = dx * dx + dy * dy;
                // Update pixel value with total flux
                let contribution = star.flux * pre_term * (-distance_squared / c).exp();

                image[[x as usize, y as usize]] += contribution;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::Rng;

    use super::*;

    fn test_star_data() -> StarData {
        StarData {
            id: 0,
            magnitude: 10.0,
            position: RaDec::from_degrees(0.0, 0.0),
            b_v: None,
        }
    }

    #[test]
    fn test_add_star_total_flux() {
        let mut image = Array2::zeros((50, 50));
        let sigma_pix = 2.0;
        let total_flux = 1000.0;

        let stars = vec![StarInFrame {
            x: 25.0,
            y: 25.0,
            flux: total_flux,
            star: test_star_data(),
        }];

        add_stars_to_image(&mut image, &stars, sigma_pix);

        let added_flux = image.sum();
        assert_relative_eq!(added_flux, total_flux, epsilon = 0.1);
    }

    #[test]
    fn test_add_star_oob() {
        let mut image = Array2::zeros((50, 50));
        let sigma_pix = 2.0;
        let total_flux = 1000.0;

        let stars = vec![StarInFrame {
            x: 60.0,
            y: 60.0,
            flux: total_flux,
            star: test_star_data(),
        }];

        add_stars_to_image(&mut image, &stars, sigma_pix);

        let added_flux = image.sum();
        assert_relative_eq!(added_flux, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_add_star_edge() {
        let mut image = Array2::zeros((50, 50));
        let sigma_pix = 2.0;
        let total_flux = 1000.0;

        let stars = vec![StarInFrame {
            x: 0.5,
            y: 10.0,
            flux: total_flux,
            star: test_star_data(),
        }];

        add_stars_to_image(&mut image, &stars, sigma_pix);

        let added_flux = image.sum();

        // TODO(meawoppl) - tighten up image edge conventions pix vs. edge centered etc
        // Right now pixel coords are edge/corner centered, but flus is kinda not intuitive that way
        assert!(
            added_flux > 100.0,
            "Added flux is out of expected range: {}",
            added_flux
        );
    }

    #[test]
    fn test_add_four_stars_corners() {
        let mut image = Array2::zeros((50, 50));
        let sigma_pix = 2.0;
        let total_flux = 250.0;

        let stars = vec![
            StarInFrame {
                x: 0.0,
                y: 0.0,
                flux: total_flux,
                star: test_star_data(),
            },
            StarInFrame {
                x: 0.0,
                y: 50.0,
                flux: total_flux,
                star: test_star_data(),
            },
            StarInFrame {
                x: 50.0,
                y: 0.0,
                flux: total_flux,
                star: test_star_data(),
            },
            StarInFrame {
                x: 50.0,
                y: 50.0,
                flux: total_flux,
                star: test_star_data(),
            },
        ];

        add_stars_to_image(&mut image, &stars, sigma_pix);

        let added_flux = image.sum();

        // Each of these should fall 3/4 off the image, resulting one flux worth
        assert_relative_eq!(added_flux, total_flux, epsilon = 1.0)
    }

    #[test]
    fn test_fuzz() {
        let mut image = Array2::zeros((50, 50));
        let sigma_pix = 2.0;
        let total_flux = 100.0;

        let mut rng = rand::thread_rng();

        let mut stars = Vec::new();
        for _ in 0..100 {
            stars.push(StarInFrame {
                x: rng.gen_range(-50.0..150.0),
                y: rng.gen_range(-50.0..150.0),
                flux: total_flux,
                star: test_star_data(),
            });
        }

        add_stars_to_image(&mut image, &stars, sigma_pix);

        let added_flux = image.sum();

        // Very loose bounds, but should catch egregious errors
        assert!(added_flux > 0.0);
    }
}

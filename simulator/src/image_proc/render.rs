use std::time::Duration;

use ndarray::Array2;
use starfield::{catalogs::StarData, Equatorial};
use viz::histogram::Scale;

use crate::{
    algo::icp::Locatable2d,
    field_diameter,
    image_proc::airy::ScaledAiryDisk,
    photometry::{zodical::SolarAngularCoordinates, ZodicalLight},
    star_data_to_electrons,
    star_math::StarProjector,
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
}

/// Create gaussian PSF kernel based on telescope properties
pub fn approx_airy_pixels(
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    wavelength_nm: f64,
) -> ScaledAiryDisk {
    // Calculate PSF size based on Airy disk
    let airy_radius_um = telescope.airy_disk_radius_um(wavelength_nm);
    let airy_radius_px = airy_radius_um / sensor.pixel_size_um;

    // Create a Gaussian approximation of the Airy disk
    // Using sigma ≈ radius/1.22 to approximate Airy disk with Gaussian
    ScaledAiryDisk::with_radius_scale(airy_radius_px / 1.22)
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
/// * `center` - Equatorial coordinates of field center
/// * `telescope` - Reference to telescope configuration
/// * `sensor` - Reference to sensor configuration
/// * `exposure` - Reference to exposure duration
/// * `wavelength_nm` - Wavelength in nanometers to use for PSF calculation
/// * `temp_c` - Sensor temperature in degrees Celsius for dark current calculation
/// * `zodiacal_coords` - Solar angular coordinates for zodiacal light calculation
///
/// # Returns
/// * `RenderingResult` - Contains the rendered image, electron counts, noise, star positions, and saturation info
pub fn render_star_field(
    stars: &Vec<&StarData>,
    center: &Equatorial,
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    exposure: &Duration,
    wavelength_nm: f64,
    temp_c: f64,
    zodiacal_coords: &SolarAngularCoordinates,
) -> RenderingResult {
    // Create image array dimensions
    let image_width = sensor.width_px as usize;
    let image_height = sensor.height_px as usize;

    // Create a star field image (in electron counts)
    let mut image = Array2::zeros((image_height, image_width));
    let mut to_render: Vec<StarInFrame> = Vec::new();

    let mut xy_mag = Vec::with_capacity(stars.len());

    let airy_pix = approx_airy_pixels(telescope, sensor, wavelength_nm);

    // Calculate field of view from telescope and sensor
    let fov_deg = field_diameter(telescope, sensor);

    // Create star projector for coordinate transformation
    let fov_rad = fov_deg.to_radians();
    let radians_per_pixel = fov_rad / image_width.max(image_height) as f64;
    let projector =
        StarProjector::new(center, radians_per_pixel, sensor.width_px, sensor.height_px);

    // Padded bounds check
    let padding = airy_pix.first_zero() * 2.0;

    // Add stars with sub-pixel precision
    for &star in stars {
        // Convert position to pixel coordinates (sub-pixel precision)
        let star_radec = Equatorial::from_degrees(star.ra_deg(), star.dec_deg());
        let (x, y) = match projector.project_unbounded(&star_radec) {
            Some(coords) => coords,
            None => continue, // Skip stars behind the camera
        };

        // Check if star is within the image bounds
        if x < -padding
            || y < -padding
            || x >= image_width as f64 + padding
            || y >= image_height as f64 + padding
        {
            continue; // Skip stars outside the image
        }

        // Post transfrom/selection
        xy_mag.push((x, y, *star));

        // Calculate photon flux using telescope model
        let electrons = star_data_to_electrons(star, exposure, telescope, sensor);

        // Add star to image with PSF
        to_render.push(StarInFrame {
            x,
            y,
            flux: electrons,
            star: *star,
        });
    }

    to_render.sort_by(|a, b| a.flux.partial_cmp(&b.flux).unwrap());

    // Create PSF kernel for the given wavelength
    add_stars_to_image(&mut image, &to_render, airy_pix);

    // Generate sensor noise (read noise and dark current)
    let sensor_noise = generate_sensor_noise(
        sensor, exposure, temp_c, None, // Use specified temperature and random noise
    );

    let z_light = ZodicalLight::new();
    let zodiacal_noise =
        z_light.generate_zodical_background(sensor, telescope, exposure, zodiacal_coords);

    image += &sensor_noise;

    let quantized = quantize_image(&image, sensor);

    let noise_image = &sensor_noise + &zodiacal_noise;

    RenderingResult {
        image: quantized,
        electron_image: image,
        noise_image,
        rendered_stars: to_render,
    }
}

pub fn quantize_image(electron_img: &Array2<f64>, sensor: &SensorConfig) -> Array2<u16> {
    // Get the DN per electron conversion factor from the sensor
    // Calculate max DN value based on sensor bit depth (saturate at sensor's max value)
    let max_dn = ((1 << sensor.bit_depth) - 1) as f64;

    // Combine conversion, clipping and rounding in a single mapv operation:
    // 1. Convert from electrons to DN
    // 2. Clip to valid range (0 to max DN for the sensor bit depth)
    // 3. Round to nearest integer
    // 4. Convert to u16
    electron_img.mapv(|total_e| {
        let clipped_e = total_e.clamp(0.0, sensor.max_well_depth_e);
        let dn = clipped_e * sensor.dn_per_electron();
        let clipped = dn.clamp(0.0, max_dn);
        clipped.round() as u16
    })
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
/// use starfield::Equatorial;
/// use simulator::image_proc::airy::ScaledAiryDisk;
/// let star_data = StarData {
///     id: 0,
///     magnitude: 10.0,
///     position: Equatorial::from_degrees(0.0, 0.0),
///     b_v: None,
/// };
/// let mut image = Array2::zeros((100, 100));
/// let stars = vec![StarInFrame { x: 50.0, y: 50.0, flux: 1000.0, star: star_data }];
/// let airy_pix = ScaledAiryDisk::with_fwhm(2.0);
/// add_stars_to_image(&mut image, &stars, airy_pix);
/// ```
pub fn add_stars_to_image(
    image: &mut Array2<f64>,
    stars: &Vec<StarInFrame>,
    airy_pix: ScaledAiryDisk,
) {
    // 2x the first 0 should cover 99.99999% of flux or so
    let max_pix_dist = (airy_pix.first_zero().max(1.0) * 4.0).ceil() as i32;
    let (width, height) = image.dim();

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
                let radius = (dx * dx + dy * dy).sqrt();
                // Update pixel value with total flux
                let contribution = star.flux * airy_pix.gaussian_approximation_normalized(radius);

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
    use crate::hardware::{dark_current::DarkCurrentEstimator, sensor::create_flat_qe};

    fn test_star_data() -> StarData {
        StarData {
            id: 0,
            magnitude: 10.0,
            position: Equatorial::from_degrees(0.0, 0.0),
            b_v: None,
        }
    }

    fn create_test_sensor(
        bit_depth: u8,
        dn_per_electron: f64,
        max_well_depth_e: f64,
    ) -> SensorConfig {
        let qe = create_flat_qe(0.5);
        SensorConfig::new(
            "Test",
            qe,
            1024,
            1024,
            5.5,
            2.0,
            DarkCurrentEstimator::new(0.01, 20.0),
            bit_depth,
            dn_per_electron,
            max_well_depth_e,
            30.0,
        )
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

        let airy_pix = ScaledAiryDisk::with_fwhm(sigma_pix);

        add_stars_to_image(&mut image, &stars, airy_pix);

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
        let airy_pix = ScaledAiryDisk::with_radius_scale(sigma_pix);

        add_stars_to_image(&mut image, &stars, airy_pix);

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

        let airy_pix = ScaledAiryDisk::with_radius_scale(sigma_pix);

        add_stars_to_image(&mut image, &stars, airy_pix);

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
        let airy_pix = ScaledAiryDisk::with_fwhm(sigma_pix);

        add_stars_to_image(&mut image, &stars, airy_pix);

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

        let airy_pix = ScaledAiryDisk::with_radius_scale(sigma_pix);

        add_stars_to_image(&mut image, &stars, airy_pix);

        let added_flux = image.sum();

        // Very loose bounds, but should catch egregious errors
        assert!(added_flux > 0.0);
    }

    #[test]
    fn test_quantize_image_basic() {
        let sensor = create_test_sensor(8, 1.0, 1000.0);
        let mut electron_img = Array2::<f64>::zeros((3, 3));

        // Set known electron values
        electron_img[[0, 0]] = 0.0; // Min value - should be 0 DN
        electron_img[[0, 1]] = 100.0; // Middle value - should be 100 DN
        electron_img[[0, 2]] = 200.0; // Higher value - should be 200 DN

        let quantized = quantize_image(&electron_img, &sensor);

        assert_eq!(quantized[[0, 0]], 0);
        assert_eq!(quantized[[0, 1]], 100);
        assert_eq!(quantized[[0, 2]], 200);
    }

    #[test]
    fn test_quantize_image_saturation() {
        let sensor = create_test_sensor(8, 1.0, 200.0);
        let mut electron_img = Array2::<f64>::zeros((3, 3));

        // Set known electron values
        electron_img[[0, 0]] = 100.0; // Within well depth
        electron_img[[0, 1]] = 200.0; // At well depth limit
        electron_img[[0, 2]] = 300.0; // Exceeds well depth - should be clamped

        let quantized = quantize_image(&electron_img, &sensor);

        assert_eq!(quantized[[0, 0]], 100);
        assert_eq!(quantized[[0, 1]], 200);
        assert_eq!(quantized[[0, 2]], 200); // Clamped to well depth
    }

    #[test]
    fn test_quantize_image_bit_depth() {
        // Test with 10-bit sensor
        let sensor = create_test_sensor(10, 1.0, 1000.0);
        let mut electron_img = Array2::<f64>::zeros((3, 3));

        // Set known electron values
        electron_img[[0, 0]] = 0.0;
        electron_img[[0, 1]] = 500.0;
        electron_img[[0, 2]] = 1500.0; // Above max well depth

        let quantized = quantize_image(&electron_img, &sensor);

        // Max value for 10-bit is 1023
        assert_eq!(quantized[[0, 0]], 0);
        assert_eq!(quantized[[0, 1]], 500);
        assert_eq!(quantized[[0, 2]], 1000); // Clamped to well depth
    }

    #[test]
    fn test_quantize_image_dn_conversion() {
        // Test with different DN per electron values
        let sensor = create_test_sensor(12, 0.5, 1000.0);
        let mut electron_img = Array2::<f64>::zeros((3, 3));

        // Set known electron values
        electron_img[[0, 0]] = 10.0; // 10 * 0.5 = 5 DN
        electron_img[[0, 1]] = 100.0; // 100 * 0.5 = 50 DN

        let quantized = quantize_image(&electron_img, &sensor);

        assert_eq!(quantized[[0, 0]], 5);
        assert_eq!(quantized[[0, 1]], 50);
    }

    #[test]
    fn test_quantize_image_rounding() {
        // Test rounding behavior
        let sensor = create_test_sensor(12, 0.3, 1000.0);
        let mut electron_img = Array2::<f64>::zeros((3, 3));

        // Set electron values that will require rounding
        electron_img[[0, 0]] = 10.0; // 10 * 0.3 = 3.0 DN
        electron_img[[0, 1]] = 15.0; // 15 * 0.3 = 4.5 DN -> should round to 5
        electron_img[[0, 2]] = 16.0; // 16 * 0.3 = 4.8 DN -> should round to 5

        let quantized = quantize_image(&electron_img, &sensor);

        assert_eq!(quantized[[0, 0]], 3);
        assert_eq!(quantized[[0, 1]], 5); // Rounded up
        assert_eq!(quantized[[0, 2]], 5); // Rounded up
    }

    #[test]
    fn test_quantize_image_negative_values() {
        // Test handling of negative electron values (shouldn't happen in practice but test defense)
        let sensor = create_test_sensor(8, 1.0, 1000.0);
        let mut electron_img = Array2::<f64>::zeros((3, 3));

        // Set some negative values
        electron_img[[0, 0]] = -10.0; // Should be clamped to 0

        let quantized = quantize_image(&electron_img, &sensor);

        assert_eq!(quantized[[0, 0]], 0);
    }
}

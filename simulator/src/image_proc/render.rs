use std::time::Duration;

use ndarray::Array2;
use starfield::{catalogs::StarData, Equatorial};

use crate::{
    algo::icp::Locatable2d,
    hardware::SatelliteConfig,
    photometry::{photoconversion::SourceFlux, zodical::SolarAngularCoordinates, ZodicalLight},
    star_math::{field_diameter, star_data_to_fluxes, StarProjector},
    SensorConfig,
};

use super::{generate_sensor_noise, noise::apply_poisson_photon_noise};

#[derive(Clone, Debug)]
pub struct StarInFrame {
    pub x: f64,
    pub y: f64,
    pub spot: SourceFlux,
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
    /// The final quantized image (u16)
    pub quantized_image: Array2<u16>,

    /// Star contribution to the image (in electron counts)
    pub star_image: Array2<f64>,

    /// Zodiacal light background (in electron counts)
    pub zodiacal_image: Array2<f64>,

    /// Sensor noise including read noise and dark current (in electron counts)
    pub sensor_noise_image: Array2<f64>,

    /// The stars that were rendered in the image (not clipped)
    pub rendered_stars: Vec<StarInFrame>,

    /// Sensor configuration for lazy computation
    pub sensor_config: SensorConfig,
}

impl RenderingResult {
    /// Lazily compute the total electron image from all components
    pub fn mean_electron_image(&self) -> Array2<f64> {
        &self.star_image + &self.zodiacal_image + &self.sensor_noise_image
    }

    /// Lazily compute quantized image from total electron image  
    pub fn compute_quantized_image(&self) -> Array2<u16> {
        let electron_img = self.mean_electron_image();
        quantize_image(&electron_img, &self.sensor_config)
    }

    /// Calculate the background RMS noise in DN units
    ///
    /// This combines sensor noise (read noise + dark current) and zodiacal light,
    /// converts from electrons to DN units using the sensor's DN per electron factor.
    ///
    /// # Returns
    /// The RMS (root mean square) noise level in DN units
    pub fn background_rms(&self) -> f64 {
        // Combine noise sources (sensor noise + zodiacal background)
        let noise_electrons = &self.sensor_noise_image + &self.zodiacal_image;

        // Calculate RMS (root mean square) of the noise
        let noise_rms_electrons = noise_electrons
            .mapv(|x| x * x)
            .mean()
            .expect("Failed to calculate mean of noise array - array is empty")
            .sqrt();

        // Convert from electrons to DN units
        noise_rms_electrons * self.sensor_config.dn_per_electron()
    }
}

/// Star field renderer that handles exposure scaling and noise generation
///
/// This struct maintains the configuration needed to render star fields at different
/// exposure times, handling the scaling of flux and generation of random noise components.
#[derive(Clone)]
pub struct Renderer {
    /// Satellite configuration (telescope + sensor + environment)
    pub satellite_config: SatelliteConfig,

    /// Base star image for 1 second exposure (in electrons)
    pub base_star_image: Array2<f64>,

    /// Stars that were rendered in the base image (not clipped)
    pub rendered_stars: Vec<StarInFrame>,
}

/// Output from a rendering operation with cleanly separated components
#[derive(Clone)]
pub struct RenderedImage {
    /// Star contribution scaled for exposure duration (in electrons)
    pub star_image: Array2<f64>,

    /// Zodiacal light background for exposure duration (in electrons)  
    pub zodiacal_image: Array2<f64>,

    /// Sensor noise for exposure duration (in electrons)
    pub sensor_noise_image: Array2<f64>,

    /// Final quantized image (u16)
    pub quantized_image: Array2<u16>,

    /// Stars that were rendered in the image (not clipped)
    pub rendered_stars: Vec<StarInFrame>,
}

impl Renderer {
    /// Create a new renderer from a star catalog and satellite configuration
    ///
    /// This method projects stars and creates a base 1-second exposure image
    pub fn from_catalog(
        stars: &Vec<&StarData>,
        center: &Equatorial,
        satellite_config: SatelliteConfig,
    ) -> Self {
        let airy_pix = satellite_config.airy_disk_pixel_space();
        let padding = airy_pix.first_zero() * 2.0;

        // Project stars to pixel coordinates with 1-second flux
        let rendered_stars = project_stars_to_pixels(stars, center, &satellite_config, padding);

        // Create base star image for 1 second exposure
        let base_star_image = add_stars_to_image(
            satellite_config.sensor.width_px,
            satellite_config.sensor.height_px,
            &rendered_stars,
            &Duration::from_secs(1),
            satellite_config.telescope.collecting_area_cm2(),
        );

        Self {
            satellite_config,
            base_star_image,
            rendered_stars,
        }
    }

    /// Create a new renderer from pre-computed stars with known pixel positions.
    ///
    /// This constructor allows direct creation of a renderer from stars that have
    /// already been positioned in pixel coordinates, bypassing the catalog projection
    /// step. Useful for controlled experiments and testing scenarios.
    ///
    /// # Arguments
    /// * `stars` - Pre-computed stars with pixel positions and 1-second fluxes
    /// * `satellite_config` - Satellite configuration for rendering parameters
    ///
    /// # Returns
    /// A new Renderer with the base star image pre-computed
    ///
    pub fn from_stars(stars: &Vec<StarInFrame>, satellite_config: SatelliteConfig) -> Self {
        // Create base star image for 1 second exposure
        let base_star_image = add_stars_to_image(
            satellite_config.sensor.width_px,
            satellite_config.sensor.height_px,
            stars,
            &Duration::from_secs(1),
            satellite_config.telescope.collecting_area_cm2(),
        );

        Self {
            satellite_config,
            base_star_image,
            rendered_stars: stars.clone(),
        }
    }

    /// Render an image with specified exposure duration and zodiacal coordinates
    ///
    /// This method scales the base star image and generates fresh noise components
    pub fn render(
        &self,
        exposure: &Duration,
        zodiacal_coords: &SolarAngularCoordinates,
    ) -> RenderedImage {
        self.render_with_options(exposure, zodiacal_coords, true, None)
    }

    /// Render an image with a specific RNG seed for reproducible results
    pub fn render_with_seed(
        &self,
        exposure: &Duration,
        zodiacal_coords: &SolarAngularCoordinates,
        seed: Option<u64>,
    ) -> RenderedImage {
        self.render_with_options(exposure, zodiacal_coords, true, seed)
    }

    /// Render an image with options for Poisson noise and RNG seed
    ///
    /// # Arguments
    /// * `exposure` - Exposure duration
    /// * `zodiacal_coords` - Solar angular coordinates for zodiacal light
    /// * `apply_poisson` - Whether to apply Poisson arrival statistics to star photons
    /// * `rng_seed` - Optional seed for random number generation
    pub fn render_with_options(
        &self,
        exposure: &Duration,
        zodiacal_coords: &SolarAngularCoordinates,
        apply_poisson: bool,
        rng_seed: Option<u64>,
    ) -> RenderedImage {
        let exposure_factor = exposure.as_secs_f64();

        // Scale star image by exposure duration
        let scaled_star_image = &self.base_star_image * exposure_factor;

        // Apply Poisson arrival time statistics to star photons if requested
        let star_image = if apply_poisson {
            apply_poisson_photon_noise(&scaled_star_image, rng_seed)
        } else {
            scaled_star_image
        };

        // Generate sensor noise
        let sensor_noise_image = generate_sensor_noise(
            &self.satellite_config.sensor,
            exposure,
            self.satellite_config.temperature_c,
            rng_seed,
        );

        // Generate zodiacal background
        let z_light = ZodicalLight::new();
        let zodiacal_image =
            z_light.generate_zodical_background(&self.satellite_config, exposure, zodiacal_coords);

        // Compute final electron and quantized images
        let total_electrons = &star_image + &zodiacal_image + &sensor_noise_image;
        let quantized_image = quantize_image(&total_electrons, &self.satellite_config.sensor);

        RenderedImage {
            star_image,
            zodiacal_image,
            sensor_noise_image,
            quantized_image,
            rendered_stars: self.rendered_stars.clone(),
        }
    }
}

impl RenderedImage {
    /// Compute the total electron image from all components
    pub fn mean_electron_image(&self) -> Array2<f64> {
        &self.star_image + &self.zodiacal_image + &self.sensor_noise_image
    }
}

/// Projects stars from catalog coordinates to pixel coordinates with flux calculation
///
/// This function handles the complete star screening and projection pipeline:
/// - Creates coordinate projector for celestial-to-pixel transformation
/// - Projects each star and filters out-of-bounds stars
/// - Calculates expected electron flux for each visible star
/// - Returns list of StarInFrame objects ready for rendering
///
/// # Arguments
///
/// * `stars` - Vector of star catalog entries to project
/// * `center` - Central pointing direction in equatorial coordinates
/// * `satellite` - Satellite configuration for projection parameters
/// * `exposure` - Exposure duration for flux calculation
/// * `padding` - Padding around image edges for PSF bleeding
///
/// # Returns
///
/// Vector of `StarInFrame` objects with pixel coordinates and flux values
pub fn project_stars_to_pixels(
    stars: &Vec<&StarData>,
    center: &Equatorial,
    satellite: &SatelliteConfig,
    padding: f64,
) -> Vec<StarInFrame> {
    let image_width = satellite.sensor.width_px;
    let image_height = satellite.sensor.height_px;

    // Calculate field of view from telescope and sensor
    let fov_deg = field_diameter(&satellite.telescope, &satellite.sensor);

    // Create star projector for coordinate transformation
    let fov_rad = fov_deg.to_radians();
    let radians_per_pixel = fov_rad / image_width.max(image_height) as f64;
    let projector = StarProjector::new(
        center,
        radians_per_pixel,
        satellite.sensor.width_px,
        satellite.sensor.height_px,
    );

    let mut projected_stars = Vec::new();

    // Project each star to pixel coordinates
    for &star in stars {
        // Convert position to pixel coordinates (sub-pixel precision)
        let (x, y) = match projector.project_unbounded(&star.position) {
            Some(coords) => coords,
            None => continue, // Skip stars behind the camera
        };

        // Check if star is within the image bounds (with padding for PSF)
        if x < -padding
            || y < -padding
            || x >= image_width as f64 + padding
            || y >= image_height as f64 + padding
        {
            continue; // Skip stars outside the image
        }

        // Create StarInFrame with projected coordinates and flux
        projected_stars.push(StarInFrame {
            x,
            y,
            spot: star_data_to_fluxes(star, satellite),
            star: *star,
        });
    }

    // Sort by flux for consistent rendering (float addition isn't associative)
    projected_stars.sort_by(|a, b| {
        a.spot
            .electrons
            .flux
            .partial_cmp(&b.spot.electrons.flux)
            .unwrap()
    });

    projected_stars
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

/// Creates an image with stars using Simpson's rule integration for the Airy disk PSF.
///
/// This function creates a new image and adds the flux contribution
/// of each star to the appropriate pixels using 3x3 Simpson's rule integration
/// for accurate flux assignment based on the Airy disk point spread function.
///
/// # Arguments
/// * `width` - Width of the image in pixels
/// * `height` - Height of the image in pixels
/// * `stars` - A vector of StarInFrame objects containing position and flux information
/// * `airy_pix` - The scaled Airy disk for PSF calculation
///
/// # Returns
/// * `Array2<f64>` - A new 2D array representing the image with stars
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// use simulator::image_proc::render::{add_stars_to_image, StarInFrame};
/// use simulator::photometry::photoconversion::{SourceFlux, SpotFlux};
/// use simulator::image_proc::airy::PixelScaledAiryDisk;
/// use starfield::catalogs::StarData;
/// use starfield::Equatorial;
/// use std::time::Duration;
///
/// let star_data = StarData {
///     id: 0,
///     magnitude: 10.0,
///     position: Equatorial::from_degrees(0.0, 0.0),
///     b_v: None,
/// };
/// let disk = PixelScaledAiryDisk::with_fwhm(2.0, 550.0);
/// let source_flux = SourceFlux {
///     photons: SpotFlux { disk: disk.clone(), flux: 1000.0 },
///     electrons: SpotFlux { disk: disk.clone(), flux: 1000.0 },
/// };
/// let stars = vec![StarInFrame { x: 50.0, y: 50.0, spot: source_flux, star: star_data }];
/// let image = add_stars_to_image(100, 100, &stars, &Duration::from_secs(1), 1.0);
/// ```
pub fn add_stars_to_image(
    width: usize,
    height: usize,
    stars: &Vec<StarInFrame>,
    exposure: &Duration,
    aperture_cm2: f64,
) -> Array2<f64> {
    // Create new image array with specified dimensions
    let mut image = Array2::zeros((height, width));

    // Calculate the contribution of all stars to this pixel
    for star in stars {
        // 2x the first 0 should cover 99.99999% of flux or so
        let max_pix_dist: i32 =
            (star.spot.electrons.disk.first_zero().max(1.0) * 2.0).ceil() as i32;

        // Calculate pixel range to check based on star position
        let xc = star.x.round() as i32;
        let yc = star.y.round() as i32;

        for x in (xc - max_pix_dist)..=(xc + max_pix_dist) {
            for y in (yc - max_pix_dist)..=(yc + max_pix_dist) {
                // Bounds check x/y - Skip out of bounds pixels
                if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                    continue;
                }

                // Calculate pixel center position relative to star
                let x_pixel = x as f64 - star.x;
                let y_pixel = y as f64 - star.y;

                let flux = &star.spot.electrons;

                // Use Simpson's rule integration for accurate flux calculation
                let contribution = flux.disk.pixel_flux_simpson(
                    x_pixel,
                    y_pixel,
                    flux.integrated_over(exposure, aperture_cm2),
                );

                image[[y as usize, x as usize]] += contribution;
            }
        }
    }

    image
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::hardware::{
        dark_current::DarkCurrentEstimator, read_noise::ReadNoiseEstimator, sensor::create_flat_qe,
    };
    use crate::image_proc::airy::PixelScaledAiryDisk;
    use crate::photometry::photoconversion::SpotFlux;

    fn test_star_data() -> StarData {
        StarData {
            id: 0,
            magnitude: 10.0,
            position: Equatorial::from_degrees(0.0, 0.0),
            b_v: None,
        }
    }

    fn create_star_in_frame(x: f64, y: f64, sigma: f64, flux: f64) -> StarInFrame {
        let disk = PixelScaledAiryDisk::with_fwhm(sigma, 550.0);

        StarInFrame {
            x,
            y,
            spot: SourceFlux {
                photons: SpotFlux {
                    disk: disk.clone(),
                    flux: flux,
                },
                electrons: SpotFlux {
                    disk: disk.clone(),
                    flux: flux,
                },
            },
            star: test_star_data(),
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
            ReadNoiseEstimator::constant(2.0),
            DarkCurrentEstimator::from_reference_point(0.01, 20.0),
            bit_depth,
            dn_per_electron,
            max_well_depth_e,
            30.0,
        )
    }

    #[test]
    fn test_add_star_total_flux() {
        for sigma_pix in vec![2.0, 4.0, 8.0] {
            let total_flux = 1000.0;

            let stars = vec![create_star_in_frame(25.0, 25.0, sigma_pix, total_flux)];
            let image = add_stars_to_image(50, 50, &stars, &Duration::from_secs(1), 1.0);
            let added_flux = image.sum();
            println!("Sigma: {}, Added Flux: {}", sigma_pix, added_flux);
            // assert_relative_eq!(added_flux, total_flux, epsilon = 1.0);
        }
    }

    #[test]
    fn test_add_star_oob() {
        let sigma_pix = 2.0;
        let total_flux = 1000.0;

        let stars = vec![create_star_in_frame(60.0, 60.0, sigma_pix, total_flux)];
        let image = add_stars_to_image(50, 50, &stars, &Duration::from_secs(1), 1.0);
        let added_flux = image.sum();
        // With Simpson's rule, some flux can still be captured even when star center is outside
        // Expect very small flux contribution
        assert!(
            added_flux < 100.0,
            "Out of bounds star flux too high: {}",
            added_flux
        );
    }

    #[test]
    fn test_add_star_edge() {
        let sigma_pix = 4.0;
        let total_flux = 1000.0;

        let stars = vec![create_star_in_frame(-0.5, 25.0, sigma_pix, total_flux)];
        let image = add_stars_to_image(50, 50, &stars, &Duration::from_secs(1), 1.0);
        let added_flux = image.sum();
        assert_relative_eq!(added_flux * 2.0, total_flux, epsilon = total_flux * 0.01);
    }

    #[test]
    fn test_add_four_stars_corners() {
        let sigma_pix = 2.0;
        let total_flux = 250.0;

        let stars = vec![
            create_star_in_frame(-0.5, -0.5, sigma_pix, total_flux),
            create_star_in_frame(-0.5, 49.5, sigma_pix, total_flux),
            create_star_in_frame(49.5, -0.5, sigma_pix, total_flux),
            create_star_in_frame(49.5, 49.5, sigma_pix, total_flux),
        ];

        let image = add_stars_to_image(50, 50, &stars, &Duration::from_secs(1), 1.0);
        let added_flux = image.sum();

        // The total flux should be about 1 star flux value, because we see 1/4 of each star
        assert_relative_eq!(added_flux, total_flux, epsilon = total_flux * 0.01);
    }

    #[test]
    fn test_fuzz() {
        let sigma_pix = 2.0;
        let total_flux = 100.0;

        let mut rng = StdRng::seed_from_u64(42);

        let mut stars = Vec::new();
        for _ in 0..100 {
            let x = rng.gen_range(-50.0..150.0);
            let y = rng.gen_range(-50.0..150.0);
            stars.push(create_star_in_frame(x, y, sigma_pix, total_flux));
        }

        let image = add_stars_to_image(50, 50, &stars, &Duration::from_secs(1), 1.0);
        let added_flux = image.sum();

        // Very loose bounds, but should catch egregious errors
        assert!(added_flux > 0.0);
    }

    #[test]
    fn test_fuzz_aspected() {
        let sigma_pix = 2.0;
        let total_flux = 100.0;

        let mut rng = StdRng::seed_from_u64(42);

        let mut stars = Vec::new();
        for _ in 0..1000 {
            let x = rng.gen_range(-50.0..150.0);
            let y = rng.gen_range(-50.0..150.0);
            stars.push(create_star_in_frame(x, y, sigma_pix, total_flux));
        }

        let image = add_stars_to_image(23, 57, &stars, &Duration::from_secs(1), 1.0);
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

    #[test]
    fn test_background_rms_zero_noise() {
        // Test with zero noise - should return 0
        let sensor = create_test_sensor(12, 1.0, 1000.0);

        let result = RenderingResult {
            quantized_image: Array2::zeros((10, 10)),
            star_image: Array2::zeros((10, 10)),
            zodiacal_image: Array2::zeros((10, 10)),
            sensor_noise_image: Array2::zeros((10, 10)),
            rendered_stars: vec![],
            sensor_config: sensor,
        };

        assert_eq!(result.background_rms(), 0.0);
    }

    #[test]
    fn test_background_rms_uniform_noise() {
        // Test with uniform noise values
        let sensor = create_test_sensor(12, 1.0, 1000.0);

        let result = RenderingResult {
            quantized_image: Array2::zeros((10, 10)),
            star_image: Array2::zeros((10, 10)),
            zodiacal_image: Array2::from_elem((10, 10), 5.0), // 5 electrons per pixel
            sensor_noise_image: Array2::from_elem((10, 10), 3.0), // 3 electrons per pixel
            rendered_stars: vec![],
            sensor_config: sensor,
        };

        // Total noise per pixel: 5 + 3 = 8 electrons
        // RMS of uniform distribution is just the value itself
        // With dn_per_electron = 1.0, expect 8.0 DN
        assert_relative_eq!(result.background_rms(), 8.0, epsilon = 1e-6);
    }

    #[test]
    fn test_background_rms_with_dn_conversion() {
        // Test DN conversion factor
        let sensor = create_test_sensor(12, 0.5, 1000.0); // 0.5 DN per electron

        let result = RenderingResult {
            quantized_image: Array2::zeros((10, 10)),
            star_image: Array2::zeros((10, 10)),
            zodiacal_image: Array2::from_elem((10, 10), 4.0),
            sensor_noise_image: Array2::from_elem((10, 10), 4.0),
            rendered_stars: vec![],
            sensor_config: sensor,
        };

        // Total noise: 8 electrons per pixel
        // With dn_per_electron = 0.5, expect 4.0 DN
        assert_relative_eq!(result.background_rms(), 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_background_rms_varying_noise() {
        // Test with varying noise levels
        let sensor = create_test_sensor(12, 1.0, 1000.0);

        let mut zodiacal = Array2::zeros((2, 2));
        let mut sensor_noise = Array2::zeros((2, 2));

        // Create a simple pattern: total noise of 0, 2, 4, 6 electrons
        zodiacal[[0, 0]] = 0.0;
        sensor_noise[[0, 0]] = 0.0; // Total: 0

        zodiacal[[0, 1]] = 1.0;
        sensor_noise[[0, 1]] = 1.0; // Total: 2

        zodiacal[[1, 0]] = 2.0;
        sensor_noise[[1, 0]] = 2.0; // Total: 4

        zodiacal[[1, 1]] = 3.0;
        sensor_noise[[1, 1]] = 3.0; // Total: 6

        let result = RenderingResult {
            quantized_image: Array2::zeros((2, 2)),
            star_image: Array2::zeros((2, 2)),
            zodiacal_image: zodiacal,
            sensor_noise_image: sensor_noise,
            rendered_stars: vec![],
            sensor_config: sensor,
        };

        // RMS = sqrt(mean(x^2)) = sqrt((0^2 + 2^2 + 4^2 + 6^2) / 4)
        //     = sqrt((0 + 4 + 16 + 36) / 4) = sqrt(56/4) = sqrt(14)
        let expected_rms = (14.0_f64).sqrt();
        assert_relative_eq!(result.background_rms(), expected_rms, epsilon = 1e-6);
    }

    #[test]
    fn test_background_rms_realistic_values() {
        // Test with realistic noise values from a sensor
        let sensor = create_test_sensor(12, 0.25, 50000.0); // 0.25 DN/e-, 50k well depth

        // Simulate realistic noise levels in electrons
        let mut rng = StdRng::seed_from_u64(42);
        let mut zodiacal = Array2::zeros((100, 100));
        let mut sensor_noise = Array2::zeros((100, 100));

        // Add some Gaussian-like noise (simplified)
        for i in 0..100 {
            for j in 0..100 {
                // Zodiacal: ~10 electrons with some variation
                zodiacal[[i, j]] = 10.0 + rng.gen_range(-2.0..2.0);
                // Sensor noise: ~5 electrons with some variation
                sensor_noise[[i, j]] = 5.0 + rng.gen_range(-1.0..1.0);
            }
        }

        let result = RenderingResult {
            quantized_image: Array2::zeros((100, 100)),
            star_image: Array2::zeros((100, 100)),
            zodiacal_image: zodiacal,
            sensor_noise_image: sensor_noise,
            rendered_stars: vec![],
            sensor_config: sensor,
        };

        let rms = result.background_rms();

        // With mean noise ~15 electrons and dn_per_electron = 0.25
        // Expected RMS should be around 15 * 0.25 = 3.75 DN
        // Allow for some variation due to randomness
        assert!(rms > 3.0 && rms < 4.5, "RMS {} not in expected range", rms);
    }
}

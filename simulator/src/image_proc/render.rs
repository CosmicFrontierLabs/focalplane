use std::time::Duration;

use ndarray::Array2;
use starfield::{catalogs::StarData, Equatorial};

use crate::{
    algo::icp::Locatable2d,
    hardware::SatelliteConfig,
    photometry::{photoconversion::SourceFlux, zodical::SolarAngularCoordinates, ZodicalLight},
    star_math::{field_diameter, star_data_to_fluxes, StarProjector},
    units::{AngleExt, Area},
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
        let noise_rms_electrons = noise_electrons.std(0.0);

        // Convert from electrons to DN units
        noise_rms_electrons * self.sensor_config.dn_per_electron()
    }
}

/// Star field renderer that caches star projections for efficient multi-exposure rendering.
///
/// The Renderer pre-computes star positions and a base 1-second exposure image,
/// enabling efficient generation of images at different exposure times without
/// re-calculating star projections or PSF convolutions. Only the exposure scaling
/// and noise generation are performed per render call.
///
/// # Inner Workings
///
/// ## One-Time Initialization (Expensive Operations)
/// 1. **Star Projection**: Convert celestial coordinates (RA/Dec) to pixel coordinates
/// 2. **Flux Calculation**: Compute photon flux for each star based on magnitude and spectrum
/// 3. **PSF Rendering**: Apply Airy disk point spread function to each star
/// 4. **Base Image Creation**: Render all stars into a single 1-second exposure image
///
/// These expensive operations involve:
/// - Spherical trigonometry for coordinate transformation
/// - Simpson's rule integration over stellar spectra and quantum efficiency curves
/// - Bessel function evaluation for realistic Airy disk patterns
/// - Sub-pixel positioning with analytical PSF integration
///
/// ## Per-Exposure Rendering (Fast Operations)
/// 1. **Linear Scaling**: Multiply base image by `exposure_time / 1_second`
/// 2. **Zodiacal Light**: Add background light scaled to exposure duration
/// 3. **Noise Generation**: Apply Poisson noise to photon counts, add read noise and dark current
/// 4. **Quantization**: Convert from electrons to ADU using sensor gain
///
/// The key insight is that photon arrival is a linear process - doubling exposure time
/// doubles the collected photons. This allows us to render once and scale many times.
///
/// # Architecture
/// - **One-time setup**: Star projection and base image creation (O(N) for N stars)
/// - **Per-exposure**: Linear scaling and noise generation (O(pixels))
/// - **Memory efficient**: Single base image reused for all exposures
/// - **Thread safe**: Immutable base data allows concurrent rendering
///
/// # Performance Benefits
/// When rendering multiple exposures of the same field:
/// - 10-100x faster than re-projecting stars each time
/// - Consistent star positions across exposure series
/// - Minimal memory overhead (one base image)
/// - Enables Monte Carlo analysis with fixed star fields
///
/// # Usage Pattern
/// ```ignore
/// // Create renderer once
/// let renderer = Renderer::from_catalog(&stars, &center, satellite_config);
///
/// // Render many exposures efficiently
/// for exposure_time in exposure_times {
///     let image = renderer.render(&exposure_time, &zodiacal_coords);
///     // Process image...
/// }
/// ```
#[derive(Clone)]
pub struct Renderer {
    /// Satellite configuration (telescope + sensor + environment)
    pub satellite_config: SatelliteConfig,

    /// Base star image for 1 second exposure (in electrons)
    pub base_star_image: Array2<f64>,

    /// Stars that were rendered in the base image (not clipped)
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

        // NOTE(meawopppl) - Should be erf() based cutoff here
        let padding = airy_pix.first_zero() * 2.0;

        // Project stars to pixel coordinates with 1-second flux
        let rendered_stars = project_stars_to_pixels(stars, center, &satellite_config, padding);

        // Create base star image for 1 second exposure
        let (width, height) = satellite_config.sensor.dimensions.get_pixel_width_height();
        let base_star_image = add_stars_to_image(
            width,
            height,
            &rendered_stars,
            &Duration::from_secs(1),
            satellite_config.telescope.clear_aperture_area(),
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
        let (width, height) = satellite_config.sensor.dimensions.get_pixel_width_height();
        let base_star_image = add_stars_to_image(
            width,
            height,
            stars,
            &Duration::from_secs(1),
            satellite_config.telescope.clear_aperture_area(),
        );

        Self {
            satellite_config,
            base_star_image,
            rendered_stars: stars.clone(),
        }
    }

    /// Render an image with specified exposure duration and zodiacal coordinates.
    ///
    /// This method efficiently generates images by:
    /// 1. Scaling the pre-computed base star image by exposure time (O(pixels))
    /// 2. Generating fresh noise components (read noise, dark current, Poisson)
    /// 3. Computing zodiacal background based on solar coordinates
    /// 4. Combining all components and quantizing to detector units
    ///
    /// # Performance
    /// Since star positions and PSF convolutions are pre-computed in the base image,
    /// this method is extremely fast - typically 10-100x faster than re-projecting
    /// stars for each exposure. Perfect for time series or Monte Carlo simulations.
    ///
    /// # Arguments
    /// * `exposure` - Integration time for the image
    /// * `zodiacal_coords` - Solar position for background light calculation
    ///
    /// # Returns
    /// Complete `RenderingResult` with separate star, background, and noise components
    pub fn render(
        &self,
        exposure: &Duration,
        zodiacal_coords: &SolarAngularCoordinates,
    ) -> RenderingResult {
        self.render_with_options(exposure, zodiacal_coords, true, None)
    }

    /// Render an image with a specific RNG seed for reproducible results
    pub fn render_with_seed(
        &self,
        exposure: &Duration,
        zodiacal_coords: &SolarAngularCoordinates,
        seed: Option<u64>,
    ) -> RenderingResult {
        self.render_with_options(exposure, zodiacal_coords, true, seed)
    }

    /// Render an image with full control over noise generation and reproducibility.
    ///
    /// This is the core rendering method that all other render methods delegate to.
    /// It leverages the pre-computed base star image to efficiently generate realistic
    /// astronomical images with proper noise characteristics.
    ///
    /// # Rendering Pipeline
    /// 1. **Star scaling**: Base image multiplied by exposure time (linear photon accumulation)
    /// 2. **Poisson sampling**: Optional photon arrival statistics for star light
    /// 3. **Sensor noise**: Read noise + dark current based on temperature and exposure
    /// 4. **Zodiacal light**: Sky background from solar system dust
    /// 5. **Quantization**: Convert electrons to ADU with realistic bit depth
    ///
    /// # Arguments
    /// * `exposure` - Exposure duration
    /// * `zodiacal_coords` - Solar angular coordinates for zodiacal light
    /// * `apply_poisson` - Whether to apply Poisson arrival statistics to star photons
    /// * `rng_seed` - Optional seed for random number generation
    ///
    /// # Noise Models
    /// - **Poisson noise**: Shot noise on photon arrivals (stars and background)
    /// - **Read noise**: Gaussian electronic noise from sensor readout
    /// - **Dark current**: Temperature-dependent thermal electrons
    /// - **Quantization**: ADC discretization effects
    pub fn render_with_options(
        &self,
        exposure: &Duration,
        zodiacal_coords: &SolarAngularCoordinates,
        apply_poisson: bool,
        rng_seed: Option<u64>,
    ) -> RenderingResult {
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
            self.satellite_config.temperature,
            rng_seed,
        );

        // Generate zodiacal background
        let z_light = ZodicalLight::new();

        let zodiacal_mean =
            z_light.generate_zodical_background(&self.satellite_config, exposure, zodiacal_coords);

        let zodiacal_image = if apply_poisson {
            let new_seed = rng_seed.map(|val| val + 1);
            apply_poisson_photon_noise(&zodiacal_mean, new_seed)
        } else {
            zodiacal_mean
        };

        // Compute final electron and quantized images
        let total_electrons = &star_image + &zodiacal_image + &sensor_noise_image;
        let quantized_image = quantize_image(&total_electrons, &self.satellite_config.sensor);

        RenderingResult {
            quantized_image,
            star_image,
            zodiacal_image,
            sensor_noise_image,
            rendered_stars: self.rendered_stars.clone(),
            sensor_config: self.satellite_config.sensor.clone(),
        }
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
    let (image_width, image_height) = satellite.sensor.dimensions.get_pixel_width_height();

    // Calculate field of view from telescope and sensor
    let fov_angle = field_diameter(&satellite.telescope, &satellite.sensor);

    // Create star projector for coordinate transformation
    let fov_rad = fov_angle.as_radians();
    let radians_per_pixel = fov_rad / image_width.max(image_height) as f64;
    let projector = StarProjector::new(center, radians_per_pixel, image_width, image_height);

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
/// use simulator::units::{Area, AreaExt, LengthExt, Wavelength};
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
/// let disk = PixelScaledAiryDisk::with_fwhm(2.0, Wavelength::from_nanometers(550.0));
/// let source_flux = SourceFlux {
///     photons: SpotFlux { disk: disk.clone(), flux: 1000.0 },
///     electrons: SpotFlux { disk: disk.clone(), flux: 1000.0 },
/// };
/// let stars = vec![StarInFrame { x: 50.0, y: 50.0, spot: source_flux, star: star_data }];
/// let image = add_stars_to_image(100, 100, &stars, &Duration::from_secs(1), Area::from_square_centimeters(1.0));
/// ```
pub fn add_stars_to_image(
    width: usize,
    height: usize,
    stars: &Vec<StarInFrame>,
    exposure: &Duration,
    aperture: Area,
) -> Array2<f64> {
    // Create new image array with specified dimensions
    let mut image = Array2::zeros((height, width));

    // Calculate the contribution of all stars to this pixel
    for star in stars {
        // NOTE(meawopppl) - Should be erf() based cutoff here.
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
                    flux.integrated_over(exposure, aperture),
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
        dark_current::DarkCurrentEstimator,
        read_noise::ReadNoiseEstimator,
        sensor::{create_flat_qe, SensorGeometry},
    };
    use crate::image_proc::airy::PixelScaledAiryDisk;
    use crate::image_proc::noise::simple_normal_array;
    use crate::photometry::photoconversion::SpotFlux;
    use crate::units::{Area, AreaExt, Length, LengthExt, Temperature, TemperatureExt};

    fn test_star_data() -> StarData {
        StarData {
            id: 0,
            magnitude: 10.0,
            position: Equatorial::from_degrees(0.0, 0.0),
            b_v: None,
        }
    }

    fn create_star_in_frame(x: f64, y: f64, sigma: f64, flux: f64) -> StarInFrame {
        let disk =
            PixelScaledAiryDisk::with_fwhm(sigma, crate::units::Wavelength::from_nanometers(550.0));

        StarInFrame {
            x,
            y,
            spot: SourceFlux {
                photons: SpotFlux { disk, flux },
                electrons: SpotFlux { disk, flux },
            },
            star: test_star_data(),
        }
    }

    fn create_test_sensor(
        bit_depth: u8,
        dn_per_electron: f64,
        max_well_depth_e: f64,
    ) -> SensorConfig {
        let geometry = SensorGeometry::of_width_height(1024, 1024, Length::from_micrometers(5.5));
        SensorConfig::new(
            "Test",
            create_flat_qe(0.5),
            geometry,
            ReadNoiseEstimator::constant(2.0),
            DarkCurrentEstimator::from_reference_point(0.01, Temperature::from_celsius(20.0)),
            bit_depth,
            dn_per_electron,
            max_well_depth_e,
            30.0,
        )
    }

    #[test]
    fn test_add_stars_centering_is_very_correct() {
        let psf =
            PixelScaledAiryDisk::with_fwhm(2.5, crate::units::Wavelength::from_nanometers(550.0));
        let source_flux = SourceFlux {
            photons: SpotFlux {
                disk: psf,
                flux: 10000.0,
            },
            electrons: SpotFlux {
                disk: psf,
                flux: 10000.0,
            },
        };

        let mut seeded_rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let x_loc = seeded_rng.gen_range(5.0..45.0);
            let y_loc = seeded_rng.gen_range(5.0..45.0);

            let stars = vec![StarInFrame {
                x: x_loc,
                y: y_loc,
                spot: source_flux.clone(),
                star: test_star_data(),
            }];

            let image = add_stars_to_image(
                50,
                50,
                &stars,
                &Duration::from_secs(1),
                Area::from_square_centimeters(1.0),
            );
            // Compute the weighted center of mass
            let mut total_flux = 0.0;
            let mut x_cm = 0.0;
            let mut y_cm = 0.0;
            for ((y, x), &value) in image.indexed_iter() {
                total_flux += value;
                x_cm += x as f64 * value;
                y_cm += y as f64 * value;
            }
            x_cm /= total_flux;
            y_cm /= total_flux;

            assert_relative_eq!(x_cm, x_loc, epsilon = 1e-4);
            assert_relative_eq!(y_cm, y_loc, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_add_star_total_flux() {
        for sigma_pix in [2.0, 4.0, 8.0] {
            let total_flux = 1000.0;

            let stars = vec![create_star_in_frame(25.0, 25.0, sigma_pix, total_flux)];
            let image = add_stars_to_image(
                50,
                50,
                &stars,
                &Duration::from_secs(1),
                Area::from_square_centimeters(1.0),
            );
            let added_flux = image.sum();
            println!("Sigma: {sigma_pix}, Added Flux: {added_flux}");
            assert_relative_eq!(added_flux, total_flux, epsilon = 1e-6 * total_flux);
        }
    }

    #[test]
    fn test_add_star_oob() {
        let sigma_pix = 2.0;
        let total_flux = 1000.0;

        let stars = vec![create_star_in_frame(60.0, 60.0, sigma_pix, total_flux)];
        let image = add_stars_to_image(
            50,
            50,
            &stars,
            &Duration::from_secs(1),
            Area::from_square_centimeters(1.0),
        );
        let added_flux = image.sum();
        // With Simpson's rule, some flux can still be captured even when star center is outside
        // Expect very small flux contribution
        assert!(
            added_flux < 100.0,
            "Out of bounds star flux too high: {added_flux}"
        );
    }

    #[test]
    fn test_add_star_edge() {
        let sigma_pix = 4.0;
        let total_flux = 1000.0;

        let stars = vec![create_star_in_frame(-0.5, 25.0, sigma_pix, total_flux)];
        let image = add_stars_to_image(
            50,
            50,
            &stars,
            &Duration::from_secs(1),
            Area::from_square_centimeters(1.0),
        );
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

        let image = add_stars_to_image(
            50,
            50,
            &stars,
            &Duration::from_secs(1),
            Area::from_square_centimeters(1.0),
        );
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

        let image = add_stars_to_image(
            50,
            50,
            &stars,
            &Duration::from_secs(1),
            Area::from_square_centimeters(1.0),
        );
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

        let image = add_stars_to_image(
            23,
            57,
            &stars,
            &Duration::from_secs(1),
            Area::from_square_centimeters(1.0),
        );
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
    fn test_background_rms_constant_noise() {
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

        assert_eq!(result.background_rms(), 0.0);
    }

    #[test]
    fn test_background_rms_normal() {
        for std_dev in [4.0, 8.0, 10.0] {
            // Test DN conversion factor
            let sensor = create_test_sensor(12, 1.0, 10000.0); // 0.5 DN per electron

            let result = RenderingResult {
                quantized_image: Array2::zeros((100, 100)),
                star_image: Array2::zeros((100, 100)),
                zodiacal_image: Array2::zeros((100, 100)),
                sensor_noise_image: simple_normal_array((100, 100), 1000.0, std_dev, 33),
                rendered_stars: vec![],
                sensor_config: sensor,
            };

            println!("std_dev: {std_dev}. Result: {:?}", result.background_rms());
            assert_relative_eq!(result.background_rms(), std_dev, epsilon = 0.1);
        }
    }

    #[test]
    fn test_background_rms_with_dn_conversion() {
        let std_dev = 5.0;
        for dn in [4.0, 8.0, 10.0] {
            // Test DN conversion factor
            let sensor = create_test_sensor(12, dn, 10000.0); // 0.5 DN per electron

            let result = RenderingResult {
                quantized_image: Array2::zeros((100, 100)),
                star_image: Array2::zeros((100, 100)),
                zodiacal_image: Array2::zeros((100, 100)),
                sensor_noise_image: simple_normal_array((100, 100), 1000.0, std_dev, 33),
                rendered_stars: vec![],
                sensor_config: sensor,
            };

            println!(
                "Testing with DN conversion: {dn}, std_dev: {std_dev}. Result: {:?}",
                result.background_rms()
            );
            assert_relative_eq!(result.background_rms() / dn, std_dev, epsilon = 0.1);
        }
    }

    #[test]
    fn test_background_rms_realistic_values() {
        // Test with realistic noise values from a sensor
        let sensor = create_test_sensor(12, 1.0, 50000.0); // 0.25 DN/e-, 50k well depth

        let zodical_std = 1.0;
        let sensor_std = 5.0;

        let result = RenderingResult {
            quantized_image: Array2::zeros((100, 100)),
            star_image: Array2::zeros((100, 100)),
            zodiacal_image: simple_normal_array((100, 100), 100.0, zodical_std, 7),
            sensor_noise_image: simple_normal_array((100, 100), 100.0, sensor_std, 8),
            rendered_stars: vec![],
            sensor_config: sensor,
        };

        let rms = result.background_rms();
        assert_relative_eq!(
            rms,
            (sensor_std.powf(2.0) + zodical_std.powf(2.0)).sqrt(),
            epsilon = 0.1
        );
    }
}

//! Common utilities for monocle tests

use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Parameters for synthetic star generation
#[derive(Debug, Clone)]
pub struct StarParams {
    pub x: f64,
    pub y: f64,
    pub peak_flux: f64,
    pub fwhm: f64, // Full Width Half Maximum in pixels
}

impl StarParams {
    /// Create a star with default FWHM
    pub fn new(x: f64, y: f64, peak_flux: f64) -> Self {
        Self {
            x,
            y,
            peak_flux,
            fwhm: 4.0, // Default ~4 pixel FWHM
        }
    }

    /// Create a star with specific FWHM
    pub fn with_fwhm(x: f64, y: f64, peak_flux: f64, fwhm: f64) -> Self {
        Self {
            x,
            y,
            peak_flux,
            fwhm,
        }
    }
}

/// Configuration for synthetic image generation
#[derive(Debug, Clone)]
pub struct SyntheticImageConfig {
    pub width: usize,
    pub height: usize,
    pub read_noise_std: f64,
    pub include_photon_noise: bool,
    pub seed: u64,
}

impl Default for SyntheticImageConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            read_noise_std: 5.0,
            include_photon_noise: true,
            seed: 42,
        }
    }
}

/// Create a synthetic star image with configurable parameters
///
/// NOTE: This is NOT intended to be a realistic star field representation.
/// It only exists to ensure the algorithm coarsely works without direct
/// dependencies on the simulator. Performance and tuning tests with realistic
/// star fields are in the monocle_harness crate.
pub fn create_synthetic_star_image(
    config: &SyntheticImageConfig,
    stars: &[StarParams],
) -> Array2<u16> {
    let mut image = Array2::<f64>::zeros((config.height, config.width));

    // Add stars with Gaussian PSFs
    for star in stars {
        // Convert FWHM to sigma: FWHM = 2.355 * sigma
        let sigma = star.fwhm / 2.355;
        let sigma2 = sigma * sigma;

        // Calculate extent (3 sigma radius)
        let radius = (3.0 * sigma).ceil() as i32;

        // Add star to image
        let y_min = ((star.y as i32 - radius).max(0)) as usize;
        let y_max = ((star.y as i32 + radius + 1).min(config.height as i32)) as usize;
        let x_min = ((star.x as i32 - radius).max(0)) as usize;
        let x_max = ((star.x as i32 + radius + 1).min(config.width as i32)) as usize;

        for y in y_min..y_max {
            for x in x_min..x_max {
                let dx = x as f64 - star.x;
                let dy = y as f64 - star.y;
                let r2 = dx * dx + dy * dy;

                // Gaussian PSF
                let intensity = star.peak_flux * (-r2 / (2.0 * sigma2)).exp();
                image[[y, x]] += intensity;
            }
        }
    }

    // Add noise
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    for pixel in image.iter_mut() {
        // Photon noise (Poisson, approximated as Gaussian)
        if config.include_photon_noise && *pixel > 0.0 {
            let photon_noise_std = (*pixel).sqrt();
            let photon_noise = rng.gen_range(-3.0..3.0) * photon_noise_std / 3.0;
            *pixel += photon_noise;
        }

        // Read noise (Gaussian)
        if config.read_noise_std > 0.0 {
            let read_noise = rng.gen_range(-3.0..3.0) * config.read_noise_std / 3.0;
            *pixel += read_noise;
        }

        // Ensure non-negative
        if *pixel < 0.0 {
            *pixel = 0.0;
        }
    }

    // Convert to u16
    image.mapv(|v| v.round().min(65535.0) as u16)
}

/// Create a simple test frame with minimal noise
pub fn create_simple_test_frame() -> Array2<u16> {
    let config = SyntheticImageConfig {
        width: 256,
        height: 256,
        read_noise_std: 2.0,
        include_photon_noise: false,
        seed: 12345,
    };

    let stars = vec![
        StarParams::new(128.0, 128.0, 5000.0), // Bright star in center
    ];

    create_synthetic_star_image(&config, &stars)
}

/// Create a frame with multiple guide stars
pub fn create_multi_star_frame() -> Array2<u16> {
    let config = SyntheticImageConfig {
        width: 512,
        height: 512,
        read_noise_std: 5.0,
        include_photon_noise: true,
        seed: 54321,
    };

    let stars = vec![
        StarParams::new(100.0, 100.0, 10000.0), // Bright star
        StarParams::new(400.0, 100.0, 8000.0),  // Medium star
        StarParams::new(250.0, 400.0, 6000.0),  // Dimmer star
    ];

    create_synthetic_star_image(&config, &stars)
}

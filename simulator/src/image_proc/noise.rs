//! Noise simulation module for sensor modeling
//!
//! This module provides functions for generating realistic sensor noise
//! for astronomical images.

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};

/// Generate sensor noise based on sensor characteristics and exposure settings
///
/// # Arguments
/// * `image_shape` - The shape of the image array as (height, width)
/// * `read_noise` - Read noise in electrons (e-)
/// * `dark_current` - Dark current in electrons per pixel per second (e-/p/s)
/// * `exposure_time` - Exposure time in seconds (s)
/// * `rng_option` - Optional random number generator for reproducible noise
///
/// # Returns
/// * `Array2<f64>` - An array of noise values in electrons to be added to the image
///
/// The function generates two common types of image noise:
/// 1. Read noise - applies to all pixels
/// 2. Dark current noise - proportional to exposure time
///
/// Note: Photon shot noise should be applied separately based on signal levels.
pub fn generate_sensor_noise(
    image_shape: (usize, usize),
    read_noise: f64,
    dark_current: f64,
    exposure_time: f64,
    mut rng_option: Option<StdRng>,
) -> Array2<f64> {
    let (height, width) = image_shape;
    let mut noise_image: Array2<f64> = Array2::zeros((height, width));

    // Calculate dark current noise for the given exposure time
    let dark_noise = dark_current * exposure_time;

    // Get or create new RNG
    let mut local_rng;
    let rng = match &mut rng_option {
        Some(r) => r,
        None => {
            // Create a new StdRng seeded from thread RNG if none provided
            local_rng = StdRng::from_rng(thread_rng()).unwrap();
            &mut local_rng
        }
    };

    for value in noise_image.iter_mut() {
        // Add read noise (uniform distribution - simplified model)
        *value += rng.gen_range(-read_noise..read_noise);

        // Add dark current noise
        *value += rng.gen_range(0.0..dark_noise * 2.0) - dark_noise;

        // Ensure noise doesn't go negative (can't have negative electrons)
        *value = value.max(0.0);
    }

    noise_image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_sensor_noise_dimensions() {
        // Test that the generated noise has the correct dimensions
        let shape = (10, 20);
        let read_noise = 2.0;
        let dark_current = 0.1;
        let exposure_time = 1.0;

        let noise = generate_sensor_noise(shape, read_noise, dark_current, exposure_time, None);

        // Check dimensions
        assert_eq!(noise.dim(), shape);

        // Check that all values are positive or zero (noise is clamped to ≥0)
        for value in noise.iter() {
            assert!(*value >= 0.0);
        }
    }

    #[test]
    fn test_generate_sensor_noise_deterministic() {
        // Test that with the same RNG seed, we get the same noise pattern
        let shape = (5, 5);
        let read_noise = 2.0;
        let dark_current = 0.1;
        let exposure_time = 1.0;

        // Create seeded RNGs with same seed
        let seed = 42u64; // Answer to the Ultimate Question...
        let rng1 = StdRng::seed_from_u64(seed);
        let rng2 = StdRng::seed_from_u64(seed);

        // Generate noise with both RNGs
        let noise1 =
            generate_sensor_noise(shape, read_noise, dark_current, exposure_time, Some(rng1));
        let noise2 =
            generate_sensor_noise(shape, read_noise, dark_current, exposure_time, Some(rng2));

        // Check that the noise patterns are identical
        for (v1, v2) in noise1.iter().zip(noise2.iter()) {
            assert_eq!(*v1, *v2);
        }

        // Check that a different seed produces different noise
        let rng3 = StdRng::seed_from_u64(seed + 1);
        let noise3 =
            generate_sensor_noise(shape, read_noise, dark_current, exposure_time, Some(rng3));

        // At least one value should be different
        let mut any_different = false;
        for (v1, v3) in noise1.iter().zip(noise3.iter()) {
            if (v1 - v3).abs() > 1e-10 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Noise patterns with different seeds should differ"
        );
    }
}

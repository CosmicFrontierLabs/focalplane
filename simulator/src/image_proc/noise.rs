//! Optimized noise simulation module for sensor modeling
//!
//! This module provides optimized functions for generating realistic sensor noise
//! for astronomical images, focusing on performance improvements.

use std::time::Duration;

use ndarray::Array2;
use rand::{thread_rng, RngCore};
use rand_distr::{Distribution, Normal, Poisson};

use crate::algo::process_array_in_parallel_chunks;
use crate::SensorConfig;

/// Generates a plausible noise field for a sensor with given parameters, optimized for performance.
///
/// This optimized version uses several techniques to improve performance:
/// - Uses Rayon for parallel processing of pixel rows
/// - Pre-computes distributions outside of pixel loops
/// - Optimizes memory access patterns
/// - Uses SIMD-friendly operations where possible
///
/// # Arguments
/// * `sensor` - Configuration of the sensor
/// * `exposure_time` - Exposure time as Duration
/// * `rng_seed` - Optional seed for random number generator
///
/// # Returns
/// * An ndarray::Array2<f64> containing the noise values for each pixel
pub fn generate_sensor_noise(
    sensor: &SensorConfig,
    exposure_time: &Duration,
    rng_seed: Option<u64>,
) -> Array2<f64> {
    // Create a random number generator from the supplied seed
    let rng_seed = rng_seed.unwrap_or(thread_rng().next_u64());

    // Calculate expected dark current electrons during exposure
    let dark_electrons_mean = sensor.dark_current_e_p_s * exposure_time.as_secs_f64();

    // Create dimensions for output
    let height = sensor.height_px as usize;
    let width = sensor.width_px as usize;

    // Choose appropriate noise model based on dark current magnitude
    if dark_electrons_mean < 0.1 {
        // For very low dark current, use Gaussian approximation
        generate_gaussian_noise(
            width,
            height,
            sensor.read_noise_e,
            dark_electrons_mean,
            rng_seed,
        )
    } else {
        // For higher dark current, use Poisson distribution
        generate_poisson_noise(
            width,
            height,
            sensor.read_noise_e,
            dark_electrons_mean,
            rng_seed,
        )
    }
}

/// Generate noise using Gaussian approximation for both read noise and dark current
fn generate_gaussian_noise(
    width: usize,
    height: usize,
    read_noise: f64,
    dark_current_mean: f64,
    rng_seed: u64,
) -> Array2<f64> {
    // Create the array with zeros
    let noise_field = Array2::<f64>::zeros((height, width));

    // Process the array in parallel chunks with our helper function
    process_array_in_parallel_chunks(
        noise_field,
        rng_seed,
        Some(64), // Process 64 rows at a time
        |chunk, rng| {
            // Create distributions
            let read_noise_dist = Normal::new(read_noise, read_noise.sqrt()).unwrap();
            let dark_noise_dist = Normal::new(0.0, dark_current_mean.sqrt()).unwrap();

            // Fill the chunk with noise values
            chunk.iter_mut().for_each(|pixel| {
                let dark_noise = dark_noise_dist.sample(rng).max(0.0);
                let read_noise_value = read_noise_dist.sample(rng).max(0.0);
                *pixel = dark_noise + read_noise_value;
            });
        },
    )
}

/// Generate noise using Poisson distribution for dark current and Gaussian for read noise
fn generate_poisson_noise(
    width: usize,
    height: usize,
    read_noise: f64,
    dark_current_mean: f64,
    rng_seed: u64,
) -> Array2<f64> {
    // Create the array with zeros
    let noise_field = Array2::<f64>::zeros((height, width));

    // Process the array in parallel chunks with our helper function
    process_array_in_parallel_chunks(
        noise_field,
        rng_seed,
        Some(64), // Process 64 rows at a time
        |chunk, rng| {
            // Create distributions
            let read_noise_dist = Normal::new(read_noise, read_noise.sqrt()).unwrap();
            let dark_poisson = Poisson::new(dark_current_mean).unwrap();

            // Fill the chunk with noise values
            chunk.iter_mut().for_each(|pixel| {
                let dark_noise = dark_poisson.sample(rng);
                let read_noise_value = read_noise_dist.sample(rng).max(0.0);
                *pixel = dark_noise + read_noise_value;
            });
        },
    )
}

/// Generate noise for a sensor with precomputed parameters
///
/// This version avoids repeated distribution creation and is optimized
/// for repeated calls with the same sensor parameters.
pub fn generate_noise_with_precomputed_params(
    width: usize,
    height: usize,
    read_noise: f64,
    dark_current_mean: f64,
    rng_seed: Option<u64>,
) -> Array2<f64> {
    let seed = rng_seed.unwrap_or(thread_rng().next_u64());

    if dark_current_mean < 0.1 {
        generate_gaussian_noise(width, height, read_noise, dark_current_mean, seed)
    } else {
        generate_poisson_noise(width, height, read_noise, dark_current_mean, seed)
    }
}

/// Estimates the noise floor for a given sensor and exposure time.
///
/// Uses a smaller sensor size for faster estimation while maintaining accuracy.
///
/// # Arguments
/// * `sensor` - Configuration of the sensor
/// * `exposure_time` - Exposure time as Duration
/// * `rng_seed` - Optional seed for random number generator
///
/// # Returns
/// * The estimated noise floor value (mean noise)
pub fn est_noise_floor(
    sensor: &SensorConfig,
    exposure_time: &Duration,
    rng_seed: Option<u64>,
) -> f64 {
    // Create a smaller sensor with the same noise characteristics for faster estimation
    let width = 64;
    let height = 64;

    // Calculate dark current mean directly
    let dark_electrons_mean = sensor.dark_current_e_p_s * exposure_time.as_secs_f64();

    // Generate the noise field using the optimized function
    let noise_field = generate_noise_with_precomputed_params(
        width,
        height,
        sensor.read_noise_e,
        dark_electrons_mean,
        rng_seed,
    );

    // Return the mean of the noise field as the estimated noise floor
    noise_field
        .mean()
        .expect("Failed to calculate mean noise value?")
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{photometry::Band, QuantumEfficiency};

    use super::*;

    fn make_tiny_test_sensor(
        size: (usize, usize),
        dark_current: f64,
        read_noise: f64,
    ) -> SensorConfig {
        let band = Band::new(300.0, 700.0);
        let qe = QuantumEfficiency::from_notch(&band, 1.0).unwrap();

        SensorConfig::new(
            "Test Sensor",
            qe,
            size.1 as u32,
            size.0 as u32,
            5.0,
            read_noise,
            dark_current,
            8,
            1.0,
            1.0,
        )
    }

    #[test]
    fn test_est_noise_floor() {
        // Test that the estimated noise floor is close to the expected value
        let shape = (100, 100);
        let read_noise = 5.0;
        let dark_current = 10.0;
        let exposure_time = Duration::from_secs(1);

        let sensor = make_tiny_test_sensor(shape, dark_current, read_noise);

        let noise_floor = est_noise_floor(&sensor, &exposure_time, Some(42));

        // Calculate the expected noise floor (mean)
        let expected_noise_floor = read_noise + dark_current * exposure_time.as_secs_f64();

        // Assert that the estimated noise floor is close to the expected value
        assert_relative_eq!(noise_floor, expected_noise_floor, epsilon = 0.05);
    }

    #[test]
    fn test_generate_sensor_noise_dimensions() {
        // Test that the generated noise has the correct dimensions
        let shape = (10, 20);
        let read_noise = 2.0;
        let dark_current = 0.1;

        let sensor = make_tiny_test_sensor(shape, dark_current, read_noise);
        let exposure_time = Duration::from_secs(1);

        let noise = generate_sensor_noise(&sensor, &exposure_time, None);

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
        let exposure_time = Duration::from_secs_f64(1.0);

        let sensor = make_tiny_test_sensor(shape, dark_current, read_noise);

        // Generate noise with the same seed
        let noise1 = generate_sensor_noise(&sensor, &exposure_time, Some(5));
        let noise2 = generate_sensor_noise(&sensor, &exposure_time, Some(5));

        // Check that the noise patterns are identical
        for (v1, v2) in noise1.iter().zip(noise2.iter()) {
            assert_eq!(*v1, *v2);
        }

        // Check that a different seed produces different noise
        let noise3 = generate_sensor_noise(&sensor, &exposure_time, Some(6));

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

    #[test]
    fn test_generate_sensor_noise_zero_exposure() {
        // Test that with zero exposure time, the noise is only read noise
        let shape = (100, 100); // Use a larger shape for better statistics
        let read_noise = 5.0;
        let dark_current = 10.0;
        let exposure_time = Duration::from_secs_f64(0.0);

        // Create a sensor with the specified parameters
        let sensor = make_tiny_test_sensor(shape, dark_current, read_noise);

        let noise = generate_sensor_noise(&sensor, &exposure_time, Some(3));

        // Calculate mean and standard deviation of the noise
        let mean = noise.mean().unwrap();

        // With zero exposure, the mean should be approximately the read noise
        assert_relative_eq!(mean, read_noise, epsilon = 0.1);
    }

    #[test]
    fn test_generate_sensor_noise_grows_linear() {
        // Test that with zero exposure time, the noise is only read noise
        let shape = (100, 100); // Use a larger shape for better statistics
        let read_noise = 5.0;
        let dark_current = 10.0;

        // Create a sensor with the specified parameters
        let sensor = make_tiny_test_sensor(shape, dark_current, read_noise);

        let mean_0 = generate_sensor_noise(&sensor, &Duration::from_secs(0), Some(7))
            .mean()
            .unwrap();
        let mean_1 = generate_sensor_noise(&sensor, &Duration::from_secs(10), Some(8))
            .mean()
            .unwrap();
        let mean_2 = generate_sensor_noise(&sensor, &Duration::from_secs(20), Some(9))
            .mean()
            .unwrap();

        //
        assert_relative_eq!(mean_1 - mean_0, 100.0, epsilon = 1.0);
        assert_relative_eq!(mean_2 - mean_0, 200.0, epsilon = 1.0);
    }

    #[test]
    fn test_generate_sensor_noise_always_positive() {
        // Test that noise values are always positive, even with zero exposure time
        let shape = (50, 50);
        let read_noise = 5.0;
        let dark_current = 10.0;

        let sensor = make_tiny_test_sensor(shape, dark_current, read_noise);

        // Test with zero exposure time
        let zero_exposure = Duration::from_secs(0);
        let noise_zero = generate_sensor_noise(&sensor, &zero_exposure, Some(42));

        // Check all values are positive
        for value in noise_zero.iter() {
            assert!(
                *value >= 0.0,
                "Noise should never be negative with zero exposure"
            );
        }

        // Test with non-zero exposure time
        let long_exposure = Duration::from_secs(5);
        let noise_long = generate_sensor_noise(&sensor, &long_exposure, Some(43));

        // Check all values are positive
        for value in noise_long.iter() {
            assert!(
                *value >= 0.0,
                "Noise should never be negative with non-zero exposure"
            );
        }
    }

    #[test]
    fn test_generate_gaussian_noise_deterministic() {
        const TEST_WIDTH: usize = 16;
        const TEST_HEIGHT: usize = 16;
        const TEST_READ_NOISE: f64 = 5.0;
        const LOW_DARK_CURRENT: f64 = 0.05; // Should trigger Gaussian path
        const TEST_SEED: u64 = 42;

        let noise1 = generate_gaussian_noise(
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_READ_NOISE,
            LOW_DARK_CURRENT,
            TEST_SEED,
        );
        let noise2 = generate_gaussian_noise(
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_READ_NOISE,
            LOW_DARK_CURRENT,
            TEST_SEED,
        );
        assert_eq!(
            noise1, noise2,
            "generate_gaussian_noise should produce identical results for the same seed"
        );
    }

    #[test]
    fn test_generate_poisson_noise_deterministic() {
        const TEST_WIDTH: usize = 16;
        const TEST_HEIGHT: usize = 16;
        const TEST_READ_NOISE: f64 = 5.0;
        const HIGH_DARK_CURRENT: f64 = 1.5; // Should trigger Poisson path
        const TEST_SEED: u64 = 42;

        let noise1 = generate_poisson_noise(
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_READ_NOISE,
            HIGH_DARK_CURRENT,
            TEST_SEED,
        );
        let noise2 = generate_poisson_noise(
            TEST_WIDTH,
            TEST_HEIGHT,
            TEST_READ_NOISE,
            HIGH_DARK_CURRENT,
            TEST_SEED,
        );
        assert_eq!(
            noise1, noise2,
            "generate_poisson_noise should produce identical results for the same seed"
        );
    }
}

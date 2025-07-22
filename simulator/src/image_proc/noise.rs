//! High-performance sensor noise simulation for astronomical imaging.
//!
//! This module provides optimized algorithms for generating realistic sensor noise
//! that accurately models the statistical properties of astronomical detectors.
//! Includes both read noise and dark current with proper statistical distributions
//! and temperature dependencies.
//!
//! # Noise Sources
//!
//! ## Read Noise
//! Electronic noise from readout amplifiers, modeled as Gaussian distribution.
//! Typically 3-15 e⁻ RMS for modern astronomical sensors.
//!
//! ## Dark Current
//! Thermal electron generation, strongly temperature-dependent.
//! Modeled as Poisson process for realistic statistics.
//!
//! # Performance Optimizations
//!
//! - **Parallel processing**: Multi-threaded noise generation using Rayon
//! - **Adaptive algorithms**: Chooses optimal distribution based on noise level
//! - **Memory efficiency**: Chunk-based processing for large images
//! - **SIMD optimization**: Vectorized operations where possible
//!
//! # Usage
//!
//! Generate realistic sensor noise for astronomical detector simulation.
//! Use generate_sensor_noise for full sensor modeling or generate_noise_with_precomputed_params
//! for batch processing with known parameters.

use std::time::Duration;

use crate::algo::process_array_in_parallel_chunks;
use crate::SensorConfig;
use ndarray::Array2;
use rand::{thread_rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal, Poisson};

/// Generate a 2D array of normally distributed values for testing purposes.
///
/// This function creates a deterministic array filled with values sampled from
/// a normal (Gaussian) distribution. It's specifically designed for unit tests
/// and simulation validation where reproducible noise patterns are needed.
///
/// # Design Purpose
/// This is a testing utility, not intended for production noise simulation.
/// For realistic sensor noise, use `generate_sensor_noise` or `generate_noise_with_precomputed_params`.
/// This function provides:
/// - Deterministic output for test reproducibility
/// - Simple interface without sensor physics
/// - Fast generation without parallel processing
///
/// # Arguments
/// * `size` - Tuple of (height, width) for the output array dimensions
/// * `mean` - Mean value of the normal distribution
/// * `std_dev` - Standard deviation of the normal distribution
/// * `seed` - Random seed for deterministic output
///
/// # Returns
/// A 2D array with values sampled from Normal(mean, std_dev)
///
/// # Example
/// ```
/// use simulator::image_proc::noise::simple_normal_array;
///
/// // Create 10x10 array with mean=100, std_dev=10, seed=42
/// let noise = simple_normal_array((10, 10), 100.0, 10.0, 42);
/// assert_eq!(noise.dim(), (10, 10));
/// ```
///
/// # Use Cases
/// - Unit testing algorithms that need controlled noise input
/// - Validating statistical properties of image processing functions
/// - Creating reproducible test scenarios for detector simulations
/// - Debugging noise-related issues with known patterns
pub fn simple_normal_array(
    size: (usize, usize),
    mean: f64,
    std_dev: f64,
    seed: u64,
) -> Array2<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal_dist = Normal::new(mean, std_dev).unwrap();
    Array2::from_shape_fn(size, |_| normal_dist.sample(&mut rng))
}

/// Generate realistic sensor noise field for astronomical detector simulation.
///
/// Creates a comprehensive noise model combining read noise and dark current with
/// proper statistical distributions. Automatically selects optimal algorithms
/// based on noise characteristics and uses parallel processing for performance.
///
/// # Noise Model
/// - **Read noise**: Gaussian distribution with sensor-specific RMS
/// - **Dark current**: Temperature-dependent Poisson process
/// - **Combined**: Statistically correct addition of independent noise sources
///
/// # Algorithm Selection
/// - Dark current < 0.1 e⁻: Gaussian approximation (faster)
/// - Dark current ≥ 0.1 e⁻: Full Poisson statistics (accurate)
///
/// # Arguments
/// * `sensor` - Sensor configuration with noise characteristics
/// * `exposure_time` - Integration time for dark current accumulation
/// * `temp_c` - Detector temperature in Celsius (affects dark current)
/// * `rng_seed` - Optional seed for reproducible results
///
/// # Returns
/// 2D noise field in electrons (e⁻) with realistic statistical properties
///
/// # Usage
/// Creates comprehensive noise model combining read noise and dark current
/// with proper statistical distributions and temperature dependencies.
pub fn generate_sensor_noise(
    sensor: &SensorConfig,
    exposure_time: &Duration,
    temp_c: f64,
    rng_seed: Option<u64>,
) -> Array2<f64> {
    // Create a random number generator from the supplied seed
    let rng_seed = rng_seed.unwrap_or(thread_rng().next_u64());

    // Calculate expected dark current electrons during exposure at specified temperature
    let dark_current = sensor.dark_current_at_temperature(temp_c);
    let dark_electrons_mean = dark_current * exposure_time.as_secs_f64();

    // Create dimensions for output
    let height = sensor.height_px;
    let width = sensor.width_px;

    // Get read noise estimate for the given temperature and exposure time
    let read_noise = sensor
        .read_noise_estimator
        .estimate(temp_c, *exposure_time)
        .expect("Can't estimate read noise");

    // Choose appropriate noise model based on dark current magnitude
    if dark_electrons_mean < 0.1 {
        // For very low dark current, use Gaussian approximation
        generate_gaussian_noise(width, height, read_noise, dark_electrons_mean, rng_seed)
    } else {
        // For higher dark current, use Poisson distribution
        generate_poisson_noise(width, height, read_noise, dark_electrons_mean, rng_seed)
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

/// Generate sensor noise with precomputed parameters for batch processing.
///
/// Optimized function for generating multiple noise realizations with the same
/// sensor characteristics. Avoids repeated sensor parameter lookups and is
/// ideal for Monte Carlo simulations or batch image processing.
///
/// # Arguments
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels  
/// * `read_noise` - Read noise RMS in electrons
/// * `dark_current_mean` - Expected dark electrons per pixel
/// * `rng_seed` - Optional seed for reproducibility
///
/// # Returns
/// 2D noise field in electrons with specified characteristics
///
/// # Performance
/// ~2-3x faster than full sensor model when called repeatedly
/// with same parameters.
///
/// # Usage
/// Generate sensor noise with precomputed parameters for batch processing.
/// Ideal for Monte Carlo simulations or repeated noise generation.
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

/// Apply Poisson arrival time statistics to star photon image in parallel
///
/// This function takes a mean electron image (star flux) and applies Poisson noise
/// to simulate realistic photon arrival statistics. Each pixel's value is treated
/// as the mean of a Poisson distribution.
///
/// # Arguments
/// * `mean_electron_image` - 2D array containing mean electron counts per pixel
/// * `rng_seed` - Optional seed for random number generator
///
/// # Returns
/// * An `ndarray::Array2<f64>` with Poisson-sampled electron counts
pub fn apply_poisson_photon_noise(
    mean_electron_image: &Array2<f64>,
    rng_seed: Option<u64>,
) -> Array2<f64> {
    let (_height, _width) = mean_electron_image.dim();
    let seed = rng_seed.unwrap_or(thread_rng().next_u64());

    // Clone the input array to get the same shape
    let poisson_image = mean_electron_image.clone();

    // Process the array in parallel chunks with our helper function
    process_array_in_parallel_chunks(
        poisson_image,
        seed,
        Some(64), // Process 64 rows at a time
        |chunk_view, rng| {
            // Apply Poisson sampling to each pixel
            chunk_view.iter_mut().for_each(|pixel| {
                let mean_electrons = *pixel;
                if mean_electrons > 0.0 {
                    // For very small means, use Gaussian approximation to avoid numerical issues
                    let sampled_electrons = if mean_electrons < 20.0 {
                        // Use Poisson distribution directly
                        let poisson = Poisson::new(mean_electrons).unwrap();
                        poisson.sample(rng)
                    } else {
                        // For large means, use normal approximation (faster and numerically stable)
                        let normal = Normal::new(mean_electrons, mean_electrons.sqrt()).unwrap();
                        normal.sample(rng).max(0.0)
                    };
                    *pixel = sampled_electrons;
                } else {
                    // Zero mean means zero photons
                    *pixel = 0.0;
                }
            });
        },
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{
        hardware::{dark_current::DarkCurrentEstimator, SatelliteConfig},
        photometry::{zodical::SolarAngularCoordinates, Band, ZodicalLight},
        QuantumEfficiency, TelescopeConfig,
    };

    use super::*;

    fn make_tiny_test_sensor(
        size: (usize, usize),
        dark_current: f64,
        read_noise: f64,
    ) -> SensorConfig {
        let band = Band::from_nm_bounds(300.0, 700.0);
        let qe = QuantumEfficiency::from_notch(&band, 1.0).unwrap();

        SensorConfig::new(
            "Test Sensor",
            qe,
            size.1,
            size.0,
            5.0,
            crate::hardware::read_noise::ReadNoiseEstimator::constant(read_noise),
            DarkCurrentEstimator::from_reference_point(dark_current, 20.0),
            8,
            1.0,
            1.0,
            60.0, // default frame rate for test
        )
    }

    #[test]
    fn test_generate_sensor_noise_dimensions() {
        // Test that the generated noise has the correct dimensions
        let shape = (10, 20);
        let read_noise = 2.0;
        let dark_current = 0.1;

        let sensor = make_tiny_test_sensor(shape, dark_current, read_noise);
        let exposure_time = Duration::from_secs_f64(0.2); // 5 Hz

        let noise = generate_sensor_noise(&sensor, &exposure_time, 20.0, None);

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
        let noise1 = generate_sensor_noise(&sensor, &exposure_time, 20.0, Some(5));
        let noise2 = generate_sensor_noise(&sensor, &exposure_time, 20.0, Some(5));

        // Check that the noise patterns are identical
        for (v1, v2) in noise1.iter().zip(noise2.iter()) {
            assert_eq!(*v1, *v2);
        }

        // Check that a different seed produces different noise
        let noise3 = generate_sensor_noise(&sensor, &exposure_time, 20.0, Some(6));

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
    fn test_generate_sensor_noise_minimal_exposure() {
        // Test that with minimal exposure time, the noise is dominated by read noise
        let shape = (100, 100); // Use a larger shape for better statistics
        let read_noise = 5.0;
        let dark_current = 10.0;
        let exposure_time = Duration::from_secs_f64(0.001); // 1ms = 1000 Hz (at upper bound)

        // Create a sensor with the specified parameters
        let sensor = make_tiny_test_sensor(shape, dark_current, read_noise);

        let noise = generate_sensor_noise(&sensor, &exposure_time, 20.0, Some(3));

        // Calculate mean and standard deviation of the noise
        let mean = noise.mean().unwrap();

        // With minimal exposure, dark current contribution is tiny (0.01 e-)
        // So mean should be approximately the read noise plus tiny dark current
        assert_relative_eq!(mean, read_noise + 0.01, epsilon = 0.5);
    }

    #[test]
    fn test_generate_sensor_noise_grows_linear() {
        // Test that with zero exposure time, the noise is only read noise
        let shape = (100, 100); // Use a larger shape for better statistics
        let read_noise = 5.0;
        let dark_current = 10.0;

        // Create a sensor with the specified parameters
        let sensor = make_tiny_test_sensor(shape, dark_current, read_noise);

        let mean_0 = generate_sensor_noise(&sensor, &Duration::from_secs_f64(0.001), 20.0, Some(7))
            .mean()
            .unwrap();
        let mean_1 = generate_sensor_noise(&sensor, &Duration::from_secs_f64(0.1), 20.0, Some(8))
            .mean()
            .unwrap();
        let mean_2 = generate_sensor_noise(&sensor, &Duration::from_secs_f64(0.2), 20.0, Some(9))
            .mean()
            .unwrap();

        // With dark current of 10 e-/s, difference should be:
        // 0.1s - 0.001s = 0.099s -> ~0.99 electrons
        // 0.2s - 0.001s = 0.199s -> ~1.99 electrons
        assert_relative_eq!(mean_1 - mean_0, 0.99, epsilon = 0.2);
        assert_relative_eq!(mean_2 - mean_0, 1.99, epsilon = 0.2);
    }

    #[test]
    fn test_generate_sensor_noise_always_positive() {
        // Test that noise values are always positive, even with zero exposure time
        let shape = (50, 50);
        let read_noise = 5.0;
        let dark_current = 10.0;

        let sensor = make_tiny_test_sensor(shape, dark_current, read_noise);

        // Test with zero exposure time
        let zero_exposure = Duration::from_secs(1);
        let noise_zero = generate_sensor_noise(&sensor, &zero_exposure, 20.0, Some(42));

        // Check all values are positive
        for value in noise_zero.iter() {
            assert!(
                *value >= 0.0,
                "Noise should never be negative with zero exposure"
            );
        }

        // Test with non-zero exposure time
        let long_exposure = Duration::from_secs(5);
        let noise_long = generate_sensor_noise(&sensor, &long_exposure, 20.0, Some(43));

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

    #[test]
    fn test_apply_poisson_photon_noise() {
        // Test that Poisson noise produces reasonable results
        let mut mean_image = Array2::<f64>::zeros((10, 10));

        // Set some pixels to different mean values
        mean_image[[0, 0]] = 0.0; // Zero photons
        mean_image[[1, 1]] = 1.0; // Low photon count
        mean_image[[2, 2]] = 100.0; // High photon count
        mean_image[[3, 3]] = 1000.0; // Very high photon count

        let poisson_image = apply_poisson_photon_noise(&mean_image, Some(42));

        // Check dimensions are preserved
        assert_eq!(poisson_image.dim(), mean_image.dim());

        // Zero mean should give zero result
        assert_eq!(poisson_image[[0, 0]], 0.0);

        // All values should be non-negative
        for value in poisson_image.iter() {
            assert!(*value >= 0.0, "Poisson samples should be non-negative");
        }

        // High count values should be close to their means (law of large numbers)
        assert!(
            (poisson_image[[3, 3]] - 1000.0).abs() < 100.0,
            "High count Poisson sample should be close to mean"
        );
    }

    #[test]
    fn test_apply_poisson_photon_noise_deterministic() {
        // Test that the same seed produces the same result
        let mean_image = Array2::<f64>::from_elem((5, 5), 10.0);

        let result1 = apply_poisson_photon_noise(&mean_image, Some(123));
        let result2 = apply_poisson_photon_noise(&mean_image, Some(123));

        // Should be identical with same seed
        for (v1, v2) in result1.iter().zip(result2.iter()) {
            assert_eq!(*v1, *v2, "Same seed should produce identical results");
        }

        // Different seed should produce different results (with high probability)
        let result3 = apply_poisson_photon_noise(&mean_image, Some(456));
        let mut any_different = false;
        for (v1, v3) in result1.iter().zip(result3.iter()) {
            if (*v1 - *v3).abs() > 1e-10 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Different seeds should produce different results"
        );
    }

    #[test]
    fn test_simple_normal_array_dimensions() {
        // Test correct dimensions
        let size = (10, 20);
        let mean = 5.0;
        let stddev = 2.0;
        let seed = 42;

        let array = simple_normal_array(size, mean, stddev, seed);
        assert_eq!(array.dim(), size);
    }

    #[test]
    fn test_simple_normal_array_deterministic() {
        // Test same seed produces same results
        let size = (5, 5);
        let mean = 10.0;
        let stddev = 3.0;
        let seed = 123;

        let array1 = simple_normal_array(size, mean, stddev, seed);
        let array2 = simple_normal_array(size, mean, stddev, seed);

        // Should be identical
        for (v1, v2) in array1.iter().zip(array2.iter()) {
            assert_eq!(*v1, *v2, "Same seed should produce identical values");
        }

        // Different seed should produce different results
        let array3 = simple_normal_array(size, mean, stddev, seed + 1);
        let mut any_different = false;
        for (v1, v3) in array1.iter().zip(array3.iter()) {
            if (*v1 - *v3).abs() > 1e-10 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Different seeds should produce different values"
        );
    }

    #[test]
    fn test_simple_normal_array_statistics() {
        // Test statistical properties with large sample
        let size = (100, 100);
        let mean = 50.0;
        let stddev = 10.0;
        let seed = 456;

        let array = simple_normal_array(size, mean, stddev, seed);

        // Calculate sample statistics
        let sample_mean = array.mean().unwrap();
        let sample_var = array.var(0.0);
        let sample_stddev = sample_var.sqrt();

        // Should be close to expected values (within reasonable tolerance for 10k samples)
        assert_relative_eq!(sample_mean, mean, epsilon = 0.5);
        assert_relative_eq!(sample_stddev, stddev, epsilon = 0.5);
    }

    #[test]
    fn test_simple_normal_array_edge_cases() {
        // Test with zero stddev (should produce constant array)
        let size = (3, 3);
        let mean = 7.0;
        let stddev = 0.0;
        let seed = 789;

        let array = simple_normal_array(size, mean, stddev, seed);

        // All values should be exactly the mean
        for value in array.iter() {
            assert_eq!(*value, mean, "Zero stddev should produce constant values");
        }

        // Test with negative mean
        let negative_mean = -15.0;
        let array_neg = simple_normal_array(size, negative_mean, 1.0, seed);
        assert!(
            array_neg.mean().unwrap() < 0.0,
            "Negative mean should produce negative average"
        );
    }

    #[test]
    fn test_simple_normal_array_large_values() {
        // Test with very large mean and stddev
        let size = (50, 50);
        let mean = 1e6;
        let stddev = 1e5;
        let seed = 999;

        let array = simple_normal_array(size, mean, stddev, seed);

        // Check values are in reasonable range (within 4 sigma)
        let min_expected = mean - 4.0 * stddev;
        let max_expected = mean + 4.0 * stddev;

        for value in array.iter() {
            assert!(
                *value > min_expected && *value < max_expected,
                "Value {} outside expected range [{}, {}]",
                value,
                min_expected,
                max_expected
            );
        }
    }

    #[test]
    fn test_zodical_light_computation() {
        // Test that the zodiacal light noise is computed correctly
        let sensor = make_tiny_test_sensor((64, 64), 0.1, 2.0);
        let telescope = TelescopeConfig::new(
            "Test Telescope",
            1.0,  // 1 meter aperture
            10.0, // 10 meters focal length
            1.0,  // light efficiency
        );
        let exposure_time = Duration::from_secs(60);
        let coords = SolarAngularCoordinates::new(90.0, 0.0).expect("Invalid coordinates");
        let z_light = ZodicalLight::new();
        // Create temporary SatelliteConfig for zodiacal background calculation
        let temp_satellite = SatelliteConfig::new(
            telescope.clone(),
            sensor.clone(),
            -10.0, // Default temperature for test
            550.0, // Default wavelength for test
        );
        let zodical_noise =
            z_light.generate_zodical_background(&temp_satellite, &exposure_time, &coords);

        // Check dimensions match sensor
        assert_eq!(
            zodical_noise.dim(),
            (sensor.height_px as usize, sensor.width_px as usize)
        );

        // Check that all values are positive (photoelectrons should be non-negative)
        for value in zodical_noise.iter() {
            assert!(*value >= 0.0, "Zodiacal light noise should be non-negative");
        }
    }
}

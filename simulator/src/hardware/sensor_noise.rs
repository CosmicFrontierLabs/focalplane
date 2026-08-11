//! Sensor noise generation for astronomical detectors.
//!
//! This module provides realistic noise modeling for CCD and CMOS sensors,
//! including read noise, dark current, and their temperature dependencies.

use std::time::Duration;

use crate::hardware::sensor::SensorConfig;
use ndarray::Array2;
use rand::{rng, RngCore};
use shared::image_proc::noise::generate_noise_with_precomputed_params;
use shared::units::{Temperature, TemperatureExt};

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
/// * `temperature` - Detector temperature (affects dark current)
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
    temperature: Temperature,
    rng_seed: Option<u64>,
) -> Array2<f64> {
    // Create a random number generator from the supplied seed
    let rng_seed = rng_seed.unwrap_or(rng().next_u64());

    // Calculate expected dark current electrons during exposure at specified temperature
    let dark_current = sensor.dark_current_at_temperature(temperature);
    let dark_electrons_mean = dark_current * exposure_time.as_secs_f64();

    // Create dimensions for output
    let (width, height) = sensor.dimensions.get_pixel_width_height();

    // Get read noise estimate for the given temperature and exposure time
    let read_noise = sensor
        .read_noise_estimator
        .estimate(temperature.as_celsius(), *exposure_time)
        .expect("Can't estimate read noise");

    // Use the shared function for actual noise generation
    generate_noise_with_precomputed_params(
        width,
        height,
        read_noise,
        dark_electrons_mean,
        Some(rng_seed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::sensor::models::IMX455;
    use shared_wasm::StatsScan;

    #[test]
    fn test_generate_sensor_noise_dimensions() {
        let sensor = IMX455.clone().with_dimensions(100, 80);
        let exposure = Duration::from_millis(100);
        let temperature = Temperature::from_celsius(-10.0);

        let noise = generate_sensor_noise(&sensor, &exposure, temperature, Some(42));

        assert_eq!(noise.shape(), &[80, 100]);
    }

    #[test]
    fn test_generate_sensor_noise_reproducible_with_seed() {
        let sensor = IMX455.clone().with_dimensions(64, 64);
        let exposure = Duration::from_millis(50);
        let temperature = Temperature::from_celsius(-20.0);

        let noise1 = generate_sensor_noise(&sensor, &exposure, temperature, Some(12345));
        let noise2 = generate_sensor_noise(&sensor, &exposure, temperature, Some(12345));

        assert_eq!(noise1, noise2);
    }

    #[test]
    fn test_generate_sensor_noise_different_seeds_differ() {
        let sensor = IMX455.clone().with_dimensions(64, 64);
        let exposure = Duration::from_millis(50);
        let temperature = Temperature::from_celsius(-20.0);

        let noise1 = generate_sensor_noise(&sensor, &exposure, temperature, Some(111));
        let noise2 = generate_sensor_noise(&sensor, &exposure, temperature, Some(222));

        assert_ne!(noise1, noise2);
    }

    #[test]
    fn test_generate_sensor_noise_statistical_properties() {
        let sensor = IMX455.clone().with_dimensions(256, 256);
        let exposure = Duration::from_millis(100);
        let temperature = Temperature::from_celsius(-10.0);

        let noise = generate_sensor_noise(&sensor, &exposure, temperature, Some(99));
        let noise_vec: Vec<f64> = noise.iter().cloned().collect();

        let stats = StatsScan::new(&noise_vec);
        let mean = stats.mean().unwrap();
        let std_dev = stats.std_dev(&noise_vec).unwrap();

        // Mean includes dark current offset, so just check it's finite and reasonable
        assert!(
            mean.is_finite() && mean.abs() < 100.0,
            "Mean should be finite and reasonable, got {}",
            mean
        );

        // Standard deviation should be positive and reasonable for sensor noise
        assert!(
            std_dev > 0.5 && std_dev < 50.0,
            "Std dev should be reasonable, got {}",
            std_dev
        );
    }

    #[test]
    fn test_generate_sensor_noise_temperature_affects_noise() {
        let sensor = IMX455.clone().with_dimensions(128, 128);
        let exposure = Duration::from_secs(1);

        // Cold sensor should have less dark current noise
        // Use temperatures within the sensor's valid range (-20 to +20)
        let cold_temp = Temperature::from_celsius(-20.0);
        let warm_temp = Temperature::from_celsius(20.0);

        let cold_noise = generate_sensor_noise(&sensor, &exposure, cold_temp, Some(42));
        let warm_noise = generate_sensor_noise(&sensor, &exposure, warm_temp, Some(42));

        let cold_vec: Vec<f64> = cold_noise.iter().cloned().collect();
        let warm_vec: Vec<f64> = warm_noise.iter().cloned().collect();

        let cold_stats = StatsScan::new(&cold_vec);
        let warm_stats = StatsScan::new(&warm_vec);

        let cold_variance = cold_stats.variance(&cold_vec).unwrap();
        let warm_variance = warm_stats.variance(&warm_vec).unwrap();

        // Warm sensor should have higher variance due to dark current
        assert!(
            warm_variance > cold_variance,
            "Warm sensor variance ({}) should exceed cold sensor variance ({})",
            warm_variance,
            cold_variance
        );
    }

    #[test]
    fn test_generate_sensor_noise_exposure_affects_dark_current() {
        let sensor = IMX455.clone().with_dimensions(128, 128);
        let temperature = Temperature::from_celsius(0.0);

        let short_exposure = Duration::from_millis(10);
        let long_exposure = Duration::from_secs(10);

        let short_noise = generate_sensor_noise(&sensor, &short_exposure, temperature, Some(42));
        let long_noise = generate_sensor_noise(&sensor, &long_exposure, temperature, Some(42));

        let short_vec: Vec<f64> = short_noise.iter().cloned().collect();
        let long_vec: Vec<f64> = long_noise.iter().cloned().collect();

        let short_stats = StatsScan::new(&short_vec);
        let long_stats = StatsScan::new(&long_vec);

        let short_variance = short_stats.variance(&short_vec).unwrap();
        let long_variance = long_stats.variance(&long_vec).unwrap();

        // Longer exposure should accumulate more dark current noise
        assert!(
            long_variance > short_variance,
            "Long exposure variance ({}) should exceed short exposure variance ({})",
            long_variance,
            short_variance
        );
    }
}

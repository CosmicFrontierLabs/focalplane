//! Sensor noise generation for astronomical detectors.
//!
//! This module provides realistic noise modeling for CCD and CMOS sensors,
//! including read noise, dark current, and their temperature dependencies.

use std::time::Duration;

use crate::hardware::sensor::SensorConfig;
use crate::units::{Temperature, TemperatureExt};
use ndarray::Array2;
use rand::{thread_rng, RngCore};
use shared::image_proc::noise::generate_noise_with_precomputed_params;

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
    let rng_seed = rng_seed.unwrap_or(thread_rng().next_u64());

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

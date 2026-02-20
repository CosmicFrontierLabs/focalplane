//! Sensor noise analyzer using renderer
//!
//! This tool generates empty sensor images with zodiacal background
//! and analyzes the noise statistics directly from the rendered images.
//! It helps characterize sensor noise properties for different configurations.
//!
//! Usage:
//! ```
//! cargo run --release --bin sensor_noise_renderer -- [OPTIONS]
//! ```

use clap::Parser;
use ndarray::Array2;
use simulator::hardware::{
    sensor::models as sensor_models, SatelliteConfig, SensorConfig, TelescopeConfig,
};
use simulator::image_proc::render::{Renderer, StarInFrame};
use simulator::photometry::zodiacal::SolarAngularCoordinates;
use simulator::units::{Length, LengthExt, Temperature, TemperatureExt};
use std::time::Duration;

/// Command-line arguments for noise analysis
#[derive(Parser, Debug)]
#[clap(author, version, about = "Analyze sensor noise using rendered images")]
struct Args {
    /// Sensor model to use (imx455, gsense4040, gsense6510, hwk4123, all)
    #[clap(short = 's', long, default_value = "imx455")]
    sensor: String,

    /// Exposure time in milliseconds
    #[clap(short = 'e', long, default_value = "1000")]
    exposure_ms: u64,

    /// Temperature in Celsius
    #[clap(short = 't', long, default_value = "-20")]
    temperature_c: f64,

    /// Number of trials to average
    #[clap(short = 'n', long, default_value = "10")]
    num_trials: usize,

    /// Image size (will create NxN patch)
    #[clap(short = 'i', long, default_value = "256")]
    image_size: usize,

    /// Random seed for reproducibility (optional)
    #[clap(long)]
    seed: Option<u64>,

    /// Solar elongation in degrees (for zodiacal light)
    #[clap(long)]
    solar_elongation_deg: Option<f64>,

    /// Ecliptic latitude in degrees (for zodiacal light)
    #[clap(long)]
    ecliptic_latitude_deg: Option<f64>,

    /// Use zodiacal minimum coordinates (overrides solar coords if set)
    #[clap(long)]
    zodiacal_minimum: bool,

    /// Verbose output
    #[clap(short = 'v', long)]
    verbose: bool,
}

/// Create a minimal telescope configuration for noise analysis
fn create_minimal_telescope() -> TelescopeConfig {
    // Use the new constructor with flat QE curve
    TelescopeConfig::new(
        "MinimalTelescope",
        Length::from_meters(0.1), // 100mm aperture
        Length::from_meters(0.5), // 500mm focal length
        0.9,                      // 90% light efficiency
    )
}

/// Generate an empty sensor image using the renderer (includes zodiacal background)
fn generate_rendered_image(
    satellite_config: &SatelliteConfig,
    exposure: Duration,
    size: usize,
    zodiacal_coords: &SolarAngularCoordinates,
    seed: Option<u64>,
) -> Array2<f64> {
    // Create empty star list (no stars)
    let empty_stars: Vec<StarInFrame> = Vec::new();

    // Create a custom sensor config with the desired image size
    let custom_sensor = satellite_config.sensor.with_dimensions(size, size);

    // Create custom satellite config with modified sensor
    let custom_satellite = SatelliteConfig::new(
        satellite_config.telescope.clone(),
        custom_sensor,
        satellite_config.temperature,
    );

    // Create renderer with empty star field
    let renderer = Renderer::from_stars(&empty_stars, custom_satellite);

    // Render image with zodiacal background and sensor noise
    let result = renderer.render_with_seed(&exposure, zodiacal_coords, seed);

    // Return the mean electron image (before quantization)
    result.mean_electron_image()
}

/// Calculate noise statistics from rendered images
fn analyze_sensor_noise(
    satellite_config: &SatelliteConfig,
    zodiacal_coords: &SolarAngularCoordinates,
    args: &Args,
) -> Vec<f64> {
    let exposure = Duration::from_millis(args.exposure_ms);

    let mut noise_values = Vec::new();

    for trial in 0..args.num_trials {
        // Use seed if provided, otherwise use trial number for reproducibility
        let seed = args.seed.map(|s| s + trial as u64);

        // Generate rendered image with noise and zodiacal background
        let rendered_image = generate_rendered_image(
            satellite_config,
            exposure,
            args.image_size,
            zodiacal_coords,
            seed,
        );

        // Calculate standard deviation as noise measure
        let mean = rendered_image.mean().unwrap();
        let variance = rendered_image
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / rendered_image.len() as f64;
        let std_dev = variance.sqrt();

        noise_values.push(std_dev);

        if args.verbose {
            println!(
                "  Trial {}: Mean = {:.2} e-, Std Dev = {:.2} e-",
                trial + 1,
                mean,
                std_dev
            );
        }
    }

    noise_values
}

fn main() {
    let args = Args::parse();

    // Create telescope config (minimal, just for completeness)
    let telescope = create_minimal_telescope();

    // Determine zodiacal coordinates
    let zodiacal_coords = if args.zodiacal_minimum {
        SolarAngularCoordinates::zodiacal_minimum()
    } else if let (Some(elongation), Some(latitude)) =
        (args.solar_elongation_deg, args.ecliptic_latitude_deg)
    {
        SolarAngularCoordinates::new(elongation, latitude).unwrap_or_else(|_| {
            eprintln!("Invalid zodiacal coordinates, using minimum");
            SolarAngularCoordinates::zodiacal_minimum()
        })
    } else {
        // Default to zodiacal minimum
        SolarAngularCoordinates::zodiacal_minimum()
    };

    // Process requested sensors
    let sensor_configs: Vec<(&str, SensorConfig)> = if args.sensor == "all" {
        vec![
            ("IMX455", sensor_models::IMX455.clone()),
            ("GSENSE4040BSI", sensor_models::GSENSE4040BSI.clone()),
            ("GSENSE6510BSI", sensor_models::GSENSE6510BSI.clone()),
            ("HWK4123", sensor_models::HWK4123.clone()),
        ]
    } else {
        let sensor_config = match args.sensor.to_lowercase().as_str() {
            "imx455" => sensor_models::IMX455.clone(),
            "gsense4040bsi" | "gsense4040" => sensor_models::GSENSE4040BSI.clone(),
            "gsense6510bsi" | "gsense6510" => sensor_models::GSENSE6510BSI.clone(),
            "hwk4123" | "hwk" => sensor_models::HWK4123.clone(),
            _ => {
                eprintln!("Unknown sensor: {}. Using IMX455.", args.sensor);
                sensor_models::IMX455.clone()
            }
        };
        vec![(args.sensor.as_str(), sensor_config)]
    };

    let temperature = Temperature::from_celsius(args.temperature_c);

    println!("Sensor Noise Analysis (Renderer-based)");
    println!("=======================================");
    println!("Exposure: {} ms", args.exposure_ms);
    println!("Temperature: {:.1}°C", args.temperature_c);
    println!("Image size: {}x{} pixels", args.image_size, args.image_size);
    println!("Trials: {}", args.num_trials);
    println!(
        "Zodiacal coords: elongation={:.1}°, latitude={:.1}°",
        zodiacal_coords.elongation(),
        zodiacal_coords.latitude()
    );
    println!();

    // Process each sensor
    for (sensor_name, sensor_config) in sensor_configs {
        println!("Sensor: {sensor_name}");
        println!("-------------------");

        // Create satellite configuration
        let satellite_config =
            SatelliteConfig::new(telescope.clone(), sensor_config.clone(), temperature);

        // Get expected noise from sensor model
        // Assume 10 FPS for read noise estimation
        let frame_duration = Duration::from_millis(100);
        let read_noise = sensor_config
            .read_noise_estimator
            .estimate(args.temperature_c, frame_duration)
            .unwrap_or(1.0);
        let dark_current = sensor_config.dark_current_at_temperature(temperature);
        let exposure_s = args.exposure_ms as f64 / 1000.0;
        let dark_electrons = dark_current * exposure_s;

        let total_expected_noise = (read_noise.powi(2) + dark_electrons).sqrt();

        println!("Expected noise components:");
        println!("  Read noise: {read_noise:.2} e-");
        println!(
            "  Dark current: {:.4} e-/s at {:.1}°C",
            dark_current, args.temperature_c
        );
        println!(
            "  Dark electrons: {:.2} e- ({} ms exposure)",
            dark_electrons, args.exposure_ms
        );
        println!("  Total expected (without zodiacal): {total_expected_noise:.2} e- RMS");
        println!();

        // Analyze noise from rendered images
        let noise_measurements = analyze_sensor_noise(&satellite_config, &zodiacal_coords, &args);

        // Calculate statistics
        let mean_noise = noise_measurements.iter().sum::<f64>() / noise_measurements.len() as f64;
        let std_dev = if noise_measurements.len() > 1 {
            let variance = noise_measurements
                .iter()
                .map(|x| (x - mean_noise).powi(2))
                .sum::<f64>()
                / (noise_measurements.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        println!("Measured noise from rendered images:");
        println!("  Mean: {mean_noise:.2} e- RMS");
        println!("  Std dev: {std_dev:.2} e-");
        println!(
            "  Ratio to expected (without zodiacal): {:.2}%",
            (mean_noise / total_expected_noise) * 100.0
        );
        println!();
    }
}

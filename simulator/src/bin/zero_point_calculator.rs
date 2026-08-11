//! Zero-point magnitude calculator for telescope/sensor combinations
//!
//! This tool calculates the zero-point magnitude for different telescope and sensor
//! combinations by finding the stellar magnitude that produces exactly 1 photoelectron
//! per second. Uses binary search between magnitude 14 and 30.
//!
//! The zero-point magnitude is a fundamental calibration parameter that relates
//! instrumental measurements (counts/second) to astronomical magnitudes.

use clap::Parser;
use simulator::photometry::{
    photoconversion::photo_electrons,
    stellar::BlackbodyStellarSpectrum,
    zodiacal::{SolarAngularCoordinates, ZodiacalLight},
};
use simulator::shared_args::{SensorModel, TelescopeModel};
use simulator::{
    hardware::{
        satellite::SatelliteConfig,
        sensor::models::ALL_SENSORS,
        telescope::models::{
            COSMIC_FRONTIER_JBT_1M, COSMIC_FRONTIER_JBT_50CM, COSMIC_FRONTIER_JBT_MAX, IDEAL_100CM,
            IDEAL_50CM, OFFICINA_STELLARE_WEASEL, OPTECH_LINA_LS35, OPTECH_LINA_LS50, SMALL_50MM,
            WEASEL,
        },
        TelescopeConfig,
    },
    star_math::DEFAULT_BV,
    units::{LengthExt, Temperature, TemperatureExt},
};
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "Zero-Point Calculator",
    about = "Calculates zero-point magnitude (mag at 1 e⁻/s) for telescope/sensor combinations",
    long_about = None
)]
struct Args {
    /// Specific telescope model to analyze (if not specified, analyzes all)
    #[arg(long)]
    telescope: Option<TelescopeModel>,

    /// Specific sensor model to analyze (if not specified, analyzes all)
    #[arg(long)]
    sensor: Option<SensorModel>,

    /// B-V color index for the stellar spectrum (default: 0.0, Vega-like)
    #[arg(long, default_value_t = DEFAULT_BV)]
    bv_color: f64,

    /// Convergence tolerance in magnitudes
    #[arg(long, default_value_t = 0.001)]
    tolerance: f64,

    /// Maximum iterations for binary search
    #[arg(long, default_value_t = 50)]
    max_iterations: usize,

    /// Show detailed calculation steps
    #[arg(long)]
    verbose: bool,

    /// Calculate limiting magnitude instead of zero-point
    #[arg(long)]
    limiting_magnitude: bool,

    /// Exposure time in seconds for limiting magnitude calculation
    #[arg(long, default_value_t = 100.0)]
    exposure_time: f64,

    /// Sensor temperature in Celsius for limiting magnitude calculation
    #[arg(long, default_value_t = -20.0)]
    temperature_c: f64,
}

/// Calculate limiting magnitude where signal equals noise floor
///
/// Finds the magnitude at which a star's flux onto a pixel center equals
/// the combined noise from read noise, dark current, and zodiacal light.
fn find_limiting_magnitude(
    satellite: &SatelliteConfig,
    exposure_time: Duration,
    temperature: Temperature,
    bv_color: f64,
    tolerance: f64,
    max_iterations: usize,
    verbose: bool,
) -> Result<f64, String> {
    // Binary search bounds (magnitudes)
    let mut mag_low = 10.0;
    let mut mag_high = 35.0;

    // Get telescope and sensor properties
    let aperture_area = satellite.telescope.clear_aperture_area();

    // Calculate noise floor components
    let exposure_s = exposure_time.as_secs_f64();

    // Read noise (electrons)
    let read_noise = satellite
        .sensor
        .read_noise_estimator
        .estimate(temperature.as_celsius(), exposure_time)
        .map_err(|e| format!("Failed to estimate read noise: {e}"))?;

    // Dark current (electrons/pixel/s)
    let dark_current = satellite.sensor.dark_current_at_temperature(temperature);
    let dark_electrons = dark_current * exposure_s;

    // Zodiacal light at minimum (best case - ecliptic pole)
    let zodiacal = ZodiacalLight::default();
    let min_zodiacal_coords = SolarAngularCoordinates::zodiacal_minimum();

    // Generate zodiacal background for entire sensor
    let zodiacal_background =
        zodiacal.generate_zodiacal_background(satellite, &exposure_time, &min_zodiacal_coords);

    // Get per-pixel zodiacal electrons (uniform across sensor)
    let zodiacal_electrons = zodiacal_background[[0, 0]]; // All pixels have same value

    // Total noise floor (RMS)
    let noise_floor = (read_noise * read_noise + dark_electrons + zodiacal_electrons).sqrt();

    if verbose {
        println!("\nNoise floor calculation:");
        println!("  Read noise: {read_noise:.2} e⁻");
        println!(
            "  Dark current: {dark_current:.3} e⁻/px/s × {exposure_s} s = {dark_electrons:.2} e⁻"
        );
        println!("  Zodiacal light: {zodiacal_electrons:.2} e⁻");
        println!("  Total noise floor: {noise_floor:.2} e⁻ RMS");
        println!();
    }

    let mut iteration = 0;

    while iteration < max_iterations && (mag_high - mag_low) > tolerance {
        let mag_mid = (mag_low + mag_high) / 2.0;

        // Create stellar spectrum at this magnitude
        let spectrum = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(bv_color, mag_mid);

        // Calculate photoelectrons for this exposure
        let star_electrons = photo_electrons(
            &spectrum,
            &satellite.combined_qe,
            aperture_area,
            &exposure_time,
        );

        if verbose {
            println!(
                "  Iteration {}: mag = {:.3}, star e⁻ = {:.3e}, noise = {:.3e}",
                iteration + 1,
                mag_mid,
                star_electrons,
                noise_floor
            );
        }

        // Compare star signal to noise floor
        if star_electrons > noise_floor {
            // Too bright, increase magnitude (dimmer)
            mag_low = mag_mid;
        } else {
            // Too dim, decrease magnitude (brighter)
            mag_high = mag_mid;
        }

        iteration += 1;
    }

    if iteration >= max_iterations {
        return Err(format!(
            "Failed to converge after {max_iterations} iterations"
        ));
    }

    // Return the midpoint as the limiting magnitude
    Ok((mag_low + mag_high) / 2.0)
}

/// Calculate zero-point magnitude using binary search
///
/// Finds the magnitude at which the given satellite configuration
/// produces exactly 1 photoelectron per second.
fn find_zero_point(
    satellite: &SatelliteConfig,
    bv_color: f64,
    tolerance: f64,
    max_iterations: usize,
    verbose: bool,
) -> Result<f64, String> {
    // Binary search bounds (magnitudes)
    let mut mag_low = 5.0;
    let mut mag_high = 30.0;

    // Use telescope's clear aperture area method which accounts for obscuration
    let aperture_area = satellite.telescope.clear_aperture_area();

    // 1 second exposure
    let exposure = Duration::from_secs(1);

    let mut iteration = 0;

    while iteration < max_iterations && (mag_high - mag_low) > tolerance {
        let mag_mid = (mag_low + mag_high) / 2.0;

        // Create stellar spectrum at this magnitude
        let spectrum = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(bv_color, mag_mid);

        // Calculate photoelectrons per second using combined QE from satellite
        let electrons_per_s =
            photo_electrons(&spectrum, &satellite.combined_qe, aperture_area, &exposure);

        if verbose {
            println!(
                "  Iteration {}: mag = {:.3}, e⁻/s = {:.3e}",
                iteration + 1,
                mag_mid,
                electrons_per_s
            );
        }

        // Adjust search bounds based on result
        if electrons_per_s > 1.0 {
            // Too bright, increase magnitude (dimmer)
            mag_low = mag_mid;
        } else {
            // Too dim, decrease magnitude (brighter)
            mag_high = mag_mid;
        }

        iteration += 1;
    }

    if iteration >= max_iterations {
        return Err(format!(
            "Failed to converge after {max_iterations} iterations"
        ));
    }

    // Return the midpoint as the zero-point magnitude
    Ok((mag_low + mag_high) / 2.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.limiting_magnitude {
        println!("Limiting Magnitude Calculator");
        println!("=============================");
        println!();
        println!("Finding faintest detectable star (signal = noise floor)");
        println!("Exposure Time: {} s", args.exposure_time);
        println!("Temperature: {:.1} °C", args.temperature_c);
        println!("B-V Color Index: {:.2}", args.bv_color);
        println!("Convergence Tolerance: {:.3} mag", args.tolerance);
        println!();
    } else {
        println!("Zero-Point Magnitude Calculator");
        println!("===============================");
        println!();
        println!("Finding magnitude at which telescope/sensor produces 1 e⁻/s");
        println!("B-V Color Index: {:.2}", args.bv_color);
        println!("Convergence Tolerance: {:.3} mag", args.tolerance);
        println!();
    }

    // Use temperature from args for limiting magnitude, default for zero-point
    let temperature = if args.limiting_magnitude {
        Temperature::from_celsius(args.temperature_c)
    } else {
        Temperature::from_celsius(-10.0)
    };

    // Determine which telescopes to analyze
    let telescopes: Vec<(&str, &TelescopeConfig)> = match args.telescope {
        Some(ref model) => match model {
            TelescopeModel::Small50mm => vec![("Small 50mm", &SMALL_50MM)],
            TelescopeModel::Demo50cm => vec![("Ideal 50cm", &IDEAL_50CM)],
            TelescopeModel::Final1m => vec![("Ideal 100cm", &IDEAL_100CM)],
            TelescopeModel::Weasel => vec![("Weasel", &WEASEL)],
            TelescopeModel::OfficinaStelareWeasel => {
                vec![("Officina Stellare Weasel", &OFFICINA_STELLARE_WEASEL)]
            }
            TelescopeModel::OptechLinaLs50 => vec![("Optech/Lina LS50", &OPTECH_LINA_LS50)],
            TelescopeModel::OptechLinaLs35 => vec![("Optech/Lina LS35", &OPTECH_LINA_LS35)],
            TelescopeModel::CosmicFrontierJbt50cm => {
                vec![("Cosmic Frontier JBT .5m", &COSMIC_FRONTIER_JBT_50CM)]
            }
            TelescopeModel::CosmicFrontierJbtMax => {
                vec![("Cosmic Frontier JBT MAX", &COSMIC_FRONTIER_JBT_MAX)]
            }
            TelescopeModel::CosmicFrontierJbt1m => {
                vec![("Cosmic Frontier JBT 1.0m", &COSMIC_FRONTIER_JBT_1M)]
            }
        },
        None => vec![
            ("Small 50mm", &SMALL_50MM),
            ("Ideal 50cm", &IDEAL_50CM),
            ("Ideal 100cm", &IDEAL_100CM),
            ("Weasel", &WEASEL),
            ("Officina Stellare Weasel", &OFFICINA_STELLARE_WEASEL),
            ("Optech/Lina LS50", &OPTECH_LINA_LS50),
            ("Optech/Lina LS35", &OPTECH_LINA_LS35),
            ("Cosmic Frontier JBT .5m", &COSMIC_FRONTIER_JBT_50CM),
            ("Cosmic Frontier JBT MAX", &COSMIC_FRONTIER_JBT_MAX),
            ("Cosmic Frontier JBT 1.0m", &COSMIC_FRONTIER_JBT_1M),
        ],
    };

    // Determine which sensors to analyze
    let sensors: Vec<&simulator::hardware::sensor::SensorConfig> = match args.sensor {
        Some(ref model) => vec![model.to_config()],
        None => ALL_SENSORS.iter().collect(),
    };

    // Print results header
    if args.limiting_magnitude {
        println!(
            "{:<15} {:<20} {:<15} {:<15}",
            "Telescope", "Sensor", "Limiting Mag", "Aperture (m)"
        );
    } else {
        println!(
            "{:<15} {:<20} {:<15} {:<15}",
            "Telescope", "Sensor", "Zero-Point (mag)", "Aperture (m)"
        );
    }
    println!("{:-<65}", "");

    // Calculate for each combination
    for (telescope_name, telescope) in &telescopes {
        for sensor in &sensors {
            // Create SatelliteConfig for this combination
            let satellite =
                SatelliteConfig::new((*telescope).clone(), (*sensor).clone(), temperature);

            if args.verbose {
                println!("\nCalculating for {} + {}:", telescope_name, sensor.name);
            }

            let result = if args.limiting_magnitude {
                find_limiting_magnitude(
                    &satellite,
                    Duration::from_secs_f64(args.exposure_time),
                    temperature,
                    args.bv_color,
                    args.tolerance,
                    args.max_iterations,
                    args.verbose,
                )
            } else {
                find_zero_point(
                    &satellite,
                    args.bv_color,
                    args.tolerance,
                    args.max_iterations,
                    args.verbose,
                )
            };

            match result {
                Ok(magnitude) => {
                    println!(
                        "{:<15} {:<20} {:<15.3} {:<15.2}",
                        telescope_name,
                        sensor.name,
                        magnitude,
                        telescope.aperture.as_meters()
                    );
                }
                Err(e) => {
                    eprintln!(
                        "Error calculating for {} + {}: {}",
                        telescope_name, sensor.name, e
                    );
                }
            }
        }
    }

    println!();
    if args.limiting_magnitude {
        println!("Note: Limiting magnitude is where star flux equals noise floor");
        println!("      (read noise + dark current + zodiacal light at ecliptic pole)");
    } else {
        println!("Note: Zero-point is the stellar magnitude that produces 1 photoelectron/second");
        println!("      for the given telescope/sensor combination.");
    }

    Ok(())
}

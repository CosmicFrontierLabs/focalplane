//! Motion simulation for space telescope tracking and attitude control
//!
//! This simulator models spacecraft motion, attitude dynamics, and star tracking
//! for space telescope applications. It provides functionality for:
//!
//! 1. Spacecraft attitude propagation and control
//! 2. Star tracker simulation with realistic noise and errors
//! 3. Pointing stability analysis
//! 4. Tracking performance evaluation
//!
//! Usage:
//! ```
//! cargo run --bin motion_simulator -- [OPTIONS]
//! ```
//!
//! See --help for detailed options.

use clap::{Parser, ValueEnum};
use log::debug;
use shared::units::{LengthExt, Temperature, TemperatureExt, Wavelength};
use simulator::hardware::telescope::{models, TelescopeConfig};
use simulator::shared_args::{DurationArg, SensorModel, SharedSimulationArgs};

/// Available telescope models for selection
#[derive(Debug, Clone, ValueEnum)]
enum TelescopeModel {
    /// Small 50mm telescope (f/10)
    Small50mm,
    /// 50cm Demo telescope (f/20) - Default
    Demo50cm,
    /// 1m Final telescope (f/10)
    Final1m,
    /// Weasel telescope (f/7.3)
    Weasel,
}

impl std::fmt::Display for TelescopeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TelescopeModel::Small50mm => write!(f, "small50mm"),
            TelescopeModel::Demo50cm => write!(f, "demo50cm"),
            TelescopeModel::Final1m => write!(f, "final1m"),
            TelescopeModel::Weasel => write!(f, "weasel"),
        }
    }
}

impl TelescopeModel {
    /// Get the corresponding TelescopeConfig for the selected model
    fn to_config(&self) -> &'static TelescopeConfig {
        match self {
            TelescopeModel::Small50mm => &models::SMALL_50MM,
            TelescopeModel::Demo50cm => &models::IDEAL_50CM,
            TelescopeModel::Final1m => &models::IDEAL_100CM,
            TelescopeModel::Weasel => &models::WEASEL,
        }
    }
}

/// Command line arguments for motion simulation
#[derive(Parser, Debug)]
#[command(
    name = "Motion Simulator",
    about = "Simulates spacecraft motion and star tracking dynamics",
    long_about = "Motion simulation for space telescope tracking and attitude control.\n\n\
        This simulator models spacecraft motion, attitude dynamics, and star tracking \
        for space telescope applications. It provides functionality for:\n  \
        - Spacecraft attitude propagation and control\n  \
        - Star tracker simulation with realistic noise and errors\n  \
        - Pointing stability analysis\n  \
        - Tracking performance evaluation\n\n\
        Note: This is currently a stub with planned future implementation."
)]
struct Args {
    #[command(flatten)]
    shared: SharedSimulationArgs,

    #[arg(
        long,
        default_value_t = TelescopeModel::Demo50cm,
        help = "Telescope model to use for simulation",
        long_help = "Telescope optical configuration for the simulation. Available models:\n  \
            - small50mm: Small 50mm aperture telescope (f/10)\n  \
            - demo50cm: 50cm demo telescope (f/20) - default\n  \
            - final1m: 1m final telescope design (f/10)\n  \
            - weasel: Weasel telescope (f/7.3)"
    )]
    telescope: TelescopeModel,

    #[arg(
        long,
        default_value_t = SensorModel::Gsense6510bsi,
        help = "Sensor model to use for simulation",
        long_help = "Camera sensor model defining pixel size, read noise, dark current, \
            and other characteristics. The sensor choice affects detection limits and \
            noise floor calculations."
    )]
    sensor: SensorModel,

    #[arg(
        long,
        default_value = "60s",
        help = "Simulation duration (e.g., \"60s\", \"1.5m\", \"2000ms\")",
        long_help = "Total duration to run the motion simulation. Accepts human-readable \
            duration strings like \"60s\" (60 seconds), \"1.5m\" (1.5 minutes), or \
            \"2000ms\" (2000 milliseconds)."
    )]
    duration: DurationArg,

    #[arg(
        long,
        default_value = "100ms",
        help = "Simulation time step (e.g., \"100ms\", \"0.1s\")",
        long_help = "Time step for the simulation integration. Smaller steps provide \
            higher accuracy but increase computation time. Accepts human-readable \
            duration strings."
    )]
    timestep: DurationArg,

    #[arg(
        long,
        default_value = "motion_results.csv",
        help = "Output CSV file for motion data",
        long_help = "Path to the output CSV file containing motion simulation results. \
            The file will contain timestamped attitude and position data."
    )]
    output_csv: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging from environment variables
    env_logger::init();

    let args = Args::parse();

    let telescope = args.telescope.to_config();
    let sensor = args.sensor.to_config();

    println!("Motion Simulator");
    println!("================");
    println!("Shared parameters:");
    println!("  Exposure: {}", args.shared.exposure);
    println!("  Temperature: {} °C", args.shared.temperature);
    println!(
        "  Coordinates: {:.1}°,{:.1}°",
        args.shared.coordinates.elongation(),
        args.shared.coordinates.latitude()
    );
    println!("  Catalog: {}", args.shared.catalog.display());
    println!();
    println!("Telescope configuration:");
    println!("  Model: {:?}", args.telescope);
    println!("  Name: {}", telescope.name);
    println!("  Aperture: {:.1} m", telescope.aperture.as_meters());
    println!(
        "  Focal length: {:.1} m",
        telescope.focal_length.as_meters()
    );
    println!(
        "  Light efficiency: {:.1}%",
        telescope
            .quantum_efficiency
            .at(Wavelength::from_nanometers(550.0))
            * 100.0
    );
    println!();
    println!("Sensor configuration:");
    println!("  Model: {:?}", args.sensor);
    println!("  Name: {}", sensor.name);
    println!("  Dimensions: {}", sensor.dimensions);
    // Get read noise estimate at operating temperature with exposure time
    let read_noise = sensor
        .read_noise_estimator
        .estimate(
            Temperature::from_celsius(args.shared.temperature).as_celsius(),
            args.shared.exposure.0,
        )
        .unwrap_or(0.0);
    println!(
        "  Read noise: {:.1} e⁻ @ {}°C",
        read_noise, args.shared.temperature
    );
    println!(
        "  Dark current: {:.3} e⁻/px/s @ {}°C",
        sensor.dark_current_at_temperature(Temperature::from_celsius(args.shared.temperature)),
        args.shared.temperature
    );
    println!("  Max frame rate: {:.1} fps", sensor.max_frame_rate_fps);
    println!();
    println!("Motion parameters:");
    println!("  Duration: {}", args.duration);
    println!("  Timestep: {}", args.timestep);
    println!("  Output file: {}", args.output_csv);

    let timestep = args.timestep.0;
    let duration = args.duration.0;
    let total_steps = (duration.as_millis() / timestep.as_millis()) as usize;

    println!("\nSimulation will run for {total_steps} steps");
    println!("Each step represents {}", args.timestep);

    // Example of loading catalog using shared helper function
    debug!("Attempting to load catalog for demonstration...");
    match args.shared.load_catalog() {
        Ok(catalog) => {
            debug!("Successfully loaded catalog with {} stars", catalog.len());
        }
        Err(e) => {
            debug!("Note: Could not load catalog ({e})");
            debug!("This is not required for motion simulation");
        }
    }

    println!("\n[STUB] Motion simulation not yet implemented");
    println!("This is a placeholder for future development");

    Ok(())
}

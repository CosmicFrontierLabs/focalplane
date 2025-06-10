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
use simulator::hardware::telescope::{models, TelescopeConfig};
use simulator::shared_args::SharedSimulationArgs;
use std::time::Duration;

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
            TelescopeModel::Small50mm => write!(f, "small-50mm"),
            TelescopeModel::Demo50cm => write!(f, "demo-50cm"),
            TelescopeModel::Final1m => write!(f, "final-1m"),
            TelescopeModel::Weasel => write!(f, "weasel"),
        }
    }
}

impl TelescopeModel {
    /// Get the corresponding TelescopeConfig for the selected model
    fn to_config(&self) -> &'static TelescopeConfig {
        match self {
            TelescopeModel::Small50mm => &models::SMALL_50MM,
            TelescopeModel::Demo50cm => &models::DEMO_50CM,
            TelescopeModel::Final1m => &models::FINAL_1M,
            TelescopeModel::Weasel => &models::WEASEL,
        }
    }
}

/// Command line arguments for motion simulation
#[derive(Parser, Debug)]
#[command(
    name = "Motion Simulator",
    about = "Simulates spacecraft motion and star tracking dynamics",
    long_about = None
)]
struct Args {
    #[command(flatten)]
    shared: SharedSimulationArgs,

    /// Telescope model to use for simulation
    #[arg(long, default_value_t = TelescopeModel::Demo50cm)]
    telescope: TelescopeModel,

    /// Simulation duration in seconds
    #[arg(long, default_value_t = 60.0)]
    duration: f64,

    /// Simulation time step in milliseconds
    #[arg(long, default_value_t = 100.0)]
    timestep_ms: f64,

    /// Output CSV file for motion data
    #[arg(long, default_value = "motion_results.csv")]
    output_csv: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let telescope = args.telescope.to_config();

    println!("Motion Simulator");
    println!("================");
    println!("Shared parameters:");
    println!("  Exposure: {} seconds", args.shared.exposure);
    println!("  Wavelength: {} nm", args.shared.wavelength);
    println!("  Temperature: {} °C", args.shared.temperature);
    println!("  Debug mode: {}", args.shared.debug);
    println!("  Coordinates: {:?}", args.shared.coordinates);
    println!();
    println!("Telescope configuration:");
    println!("  Model: {:?}", args.telescope);
    println!("  Name: {}", telescope.name);
    println!("  Aperture: {:.1} m", telescope.aperture_m);
    println!("  Focal length: {:.1} m", telescope.focal_length_m);
    println!(
        "  Light efficiency: {:.1}%",
        telescope.light_efficiency * 100.0
    );
    println!();
    println!("Motion parameters:");
    println!("  Duration: {} seconds", args.duration);
    println!("  Timestep: {} ms", args.timestep_ms);
    println!("  Output file: {}", args.output_csv);

    if args.shared.debug {
        println!("\nDebug mode enabled - additional diagnostics will be shown");
    }

    let timestep = Duration::from_millis(args.timestep_ms as u64);
    let total_steps = (args.duration * 1000.0 / args.timestep_ms) as usize;

    println!("\nSimulation will run for {} steps", total_steps);
    println!("Each step represents {:.1} ms", args.timestep_ms);

    // TODO: Implement actual motion simulation
    println!("\n[STUB] Motion simulation not yet implemented");
    println!("This is a placeholder for future development");

    Ok(())
}

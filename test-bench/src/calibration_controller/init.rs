//! Shared initialization logic for calibration modes.
//!
//! Sets up pattern client, tracking collector, and discovers sensor info.

use std::time::Duration;

use shared::tracking_collector::TrackingCollector;
use test_bench_shared::CalibrateServerClient;

use super::communication::discover_sensor_info;
use super::grid::generate_centered_grid;
use super::Args;

/// Initialized resources for calibration.
pub struct CalibrationContext {
    /// HTTP client for pattern commands
    pub pattern_client: CalibrateServerClient,
    /// Tracking message collector
    pub tracking_collector: TrackingCollector,
    /// Detected or default sensor width
    pub sensor_width: u32,
    /// Detected or default sensor height
    pub sensor_height: u32,
    /// Grid positions in display coordinates
    pub positions: Vec<(f64, f64)>,
}

/// Initialize all connections and discover sensor info.
///
/// Returns a CalibrationContext with all resources ready to use.
pub async fn initialize(
    args: &Args,
    verbose: bool,
) -> Result<CalibrationContext, Box<dyn std::error::Error + Send + Sync>> {
    if verbose {
        println!("HTTP endpoint: {}", args.http_endpoint);
        println!("Tracking endpoint: {}", args.tracking_endpoint);
        println!(
            "Grid: {}x{} with {:.0}px spacing",
            args.grid_size, args.grid_size, args.grid_spacing
        );
        println!("Display: {}x{}", args.display_width, args.display_height);
        println!();
    }

    // Create pattern client
    let pattern_client = CalibrateServerClient::new(&args.http_endpoint);

    // Create tracking collector (waits for connection to establish)
    if verbose {
        println!("Connecting to tracking endpoint...");
    }
    let tracking_collector = TrackingCollector::connect(&args.tracking_endpoint)?;
    if verbose {
        println!("Connected to tracking endpoint");
    }

    // Enable RemoteControlled mode on the display server
    if verbose {
        println!("Enabling RemoteControlled mode on display server...");
    } else {
        eprintln!("Enabling RemoteControlled mode...");
    }
    pattern_client
        .enable_remote_mode()
        .await
        .map_err(|e| format!("Failed to enable RemoteControlled mode: {e}"))?;
    if verbose {
        println!("RemoteControlled mode enabled");
    }

    // Try to auto-discover sensor info from tracking messages
    if verbose {
        println!("Waiting for sensor info from tracker...");
    }
    let (sensor_width, sensor_height) =
        if let Some(info) = discover_sensor_info(&tracking_collector, Duration::from_secs(2)) {
            let msg = format!(
                "Auto-discovered sensor: {}x{} ({}, {:.2}um pitch)",
                info.width, info.height, info.name, info.pixel_pitch_um
            );
            if verbose {
                println!("{msg}");
            } else {
                eprintln!("{msg}");
            }
            (info.width, info.height)
        } else {
            let msg = format!(
                "No sensor info received, using CLI defaults: {}x{}",
                args.sensor_width, args.sensor_height
            );
            if verbose {
                println!("{msg}");
            } else {
                eprintln!("{msg}");
            }
            (args.sensor_width, args.sensor_height)
        };

    if verbose {
        println!();
    }

    // Generate grid positions
    let positions = generate_centered_grid(
        args.grid_size,
        args.grid_spacing,
        args.display_width,
        args.display_height,
    );

    if verbose {
        println!("Generated {} calibration positions", positions.len());
        println!();
    }

    Ok(CalibrationContext {
        pattern_client,
        tracking_collector,
        sensor_width,
        sensor_height,
        positions,
    })
}

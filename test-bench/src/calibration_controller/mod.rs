//! Calibration controller module for remote optical calibration.
//!
//! This module provides tools for driving optical calibration by:
//! - Commanding display patterns via REST API
//! - Collecting tracking feedback from the FGS
//! - Estimating display-to-sensor affine transforms
//!
//! ## Modes
//!
//! - **Single-pass**: Run through each grid position once, estimate transform
//! - **Live mode**: Continuously cycle through positions with real-time TUI visualization

mod communication;
mod grid;
mod init;
pub mod live_mode;
mod render_tui;
pub mod single_pass;
mod types;

pub use grid::generate_centered_grid;
pub use init::{initialize, CalibrationContext};
pub use types::{App, CalibrationPoint, Measurement, MeasurementBuffer};

use clap::Parser;

/// Desktop calibration controller for remote optical calibration.
#[derive(Parser, Debug, Clone)]
#[command(name = "calibration_controller")]
#[command(
    about = "Drives optical calibration by commanding display patterns and collecting tracking feedback"
)]
pub struct Args {
    /// HTTP endpoint for calibrate_serve pattern API
    #[arg(long, default_value = "http://test-bench-pi.tail944341.ts.net:3001")]
    pub http_endpoint: String,

    /// ZMQ endpoint to receive tracking messages (SUB socket connects here)
    #[arg(long, default_value = "tcp://orin-416.tail944341.ts.net:5555")]
    pub tracking_endpoint: String,

    /// Grid size (NxN points)
    #[arg(long, default_value = "5")]
    pub grid_size: usize,

    /// Grid spacing in display pixels
    #[arg(long, default_value = "200.0")]
    pub grid_spacing: f64,

    /// Display width in pixels
    #[arg(long, default_value = "2560")]
    pub display_width: u32,

    /// Display height in pixels
    #[arg(long, default_value = "2560")]
    pub display_height: u32,

    /// Spot FWHM in pixels
    #[arg(long, default_value = "5.0")]
    pub spot_fwhm: f64,

    /// Spot intensity (0.0 to 1.0)
    #[arg(long, default_value = "1.0")]
    pub spot_intensity: f64,

    /// Settle time after moving spot (seconds)
    #[arg(long, default_value = "1.5")]
    pub settle_secs: f64,

    /// Number of measurements to average per position
    #[arg(long, default_value = "30")]
    pub measurements_per_position: usize,

    /// Timeout for waiting for measurements (seconds)
    #[arg(long, default_value = "30.0")]
    pub timeout_secs: f64,

    /// Sensor width in pixels (for visualization)
    #[arg(long, default_value = "9576")]
    pub sensor_width: u32,

    /// Sensor height in pixels (for visualization)
    #[arg(long, default_value = "6388")]
    pub sensor_height: u32,

    /// Enable live mode for continuous updating
    #[arg(long)]
    pub live: bool,

    /// Number of measurements to keep in rolling history per position
    #[arg(long, default_value = "50")]
    pub history: usize,
}

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
    about = "Drives optical calibration by commanding display patterns and collecting tracking feedback",
    long_about = "Desktop calibration controller for remote optical calibration.\n\n\
        This tool drives optical calibration by commanding display patterns on calibrate_serve \
        and collecting tracking feedback from fgs_server. It estimates the affine transform \
        between display coordinates and sensor coordinates.\n\n\
        Modes:\n  \
        - Single-pass (default): Run through each grid position once and compute transform\n  \
        - Live mode (--live): Continuously cycle through positions with real-time TUI visualization\n\n\
        The controller communicates via:\n  \
        - HTTP REST API to calibrate_serve for pattern commands\n  \
        - ZMQ SUB socket to receive tracking messages from fgs_server"
)]
pub struct Args {
    #[arg(
        long,
        default_value = "http://cfl-test-bench.tail944341.ts.net:3001",
        help = "HTTP endpoint for calibrate_serve pattern API",
        long_help = "Base URL for the calibrate_serve HTTP REST API. Pattern configuration \
            requests are sent to <endpoint>/config. Default points to cfl-test-bench Tailscale hostname."
    )]
    pub http_endpoint: String,

    #[arg(
        long,
        default_value = "tcp://orin-005.tail944341.ts.net:5555",
        help = "ZMQ endpoint to receive tracking messages",
        long_help = "ZeroMQ SUB socket endpoint for receiving TrackingMessage updates from \
            fgs_server. The controller subscribes to all messages on this endpoint. \
            Default points to orin-005 (Neutralino) Tailscale hostname on port 5555."
    )]
    pub tracking_endpoint: String,

    #[arg(
        long,
        default_value = "5",
        help = "Grid size (NxN points)",
        long_help = "Number of calibration points per axis, creating an NxN grid pattern. \
            Larger grids provide more points for transform estimation but take longer. \
            A 5x5 grid (25 points) is usually sufficient."
    )]
    pub grid_size: usize,

    #[arg(
        long,
        default_value = "200.0",
        help = "Grid spacing in display pixels",
        long_help = "Distance between adjacent grid points in display pixel coordinates. \
            Larger spacing covers more of the display but may result in fewer points \
            visible on the sensor."
    )]
    pub grid_spacing: f64,

    #[arg(
        long,
        default_value = "2560",
        help = "Display width in pixels",
        long_help = "Width of the calibration display in pixels. Used to center the grid \
            pattern. Default: 2560 (OLED display resolution)."
    )]
    pub display_width: u32,

    #[arg(
        long,
        default_value = "2560",
        help = "Display height in pixels",
        long_help = "Height of the calibration display in pixels. Used to center the grid \
            pattern. Default: 2560 (OLED display resolution)."
    )]
    pub display_height: u32,

    #[arg(
        long,
        default_value = "5.0",
        help = "Spot FWHM in display pixels",
        long_help = "Full width at half maximum of the Gaussian spot in display pixels. \
            Larger spots are easier to detect but may reduce position accuracy. \
            Should be matched to the optical system's PSF."
    )]
    pub spot_fwhm: f64,

    #[arg(
        long,
        default_value = "1.0",
        help = "Spot intensity (0.0 to 1.0)",
        long_help = "Peak intensity of the calibration spot, from 0.0 (black) to 1.0 (full white). \
            Lower intensities can be used to avoid sensor saturation."
    )]
    pub spot_intensity: f64,

    #[arg(
        long,
        default_value = "1.5",
        help = "Settle time after moving spot (seconds)",
        long_help = "Time to wait in seconds after commanding a new spot position before \
            collecting measurements. Allows for display update latency and any mechanical \
            settling in the optical path."
    )]
    pub settle_secs: f64,

    #[arg(
        long,
        default_value = "30",
        help = "Number of measurements to average per position",
        long_help = "Number of tracking measurements to collect and average at each grid \
            position. More measurements improve precision but increase calibration time."
    )]
    pub measurements_per_position: usize,

    #[arg(
        long,
        default_value = "30.0",
        help = "Timeout for waiting for measurements (seconds)",
        long_help = "Maximum time to wait in seconds for tracking measurements at each \
            position. If the timeout is reached before collecting enough measurements, \
            the position is skipped."
    )]
    pub timeout_secs: f64,

    #[arg(
        long,
        default_value = "9576",
        help = "Sensor width in pixels (for visualization)",
        long_help = "Width of the camera sensor in pixels. Used for TUI visualization to \
            scale the display. Default: 9576 (GSense6510 sensor width)."
    )]
    pub sensor_width: u32,

    #[arg(
        long,
        default_value = "6388",
        help = "Sensor height in pixels (for visualization)",
        long_help = "Height of the camera sensor in pixels. Used for TUI visualization to \
            scale the display. Default: 6388 (GSense6510 sensor height)."
    )]
    pub sensor_height: u32,

    #[arg(
        long,
        help = "Enable live mode for continuous updating",
        long_help = "Run in live mode with a terminal UI that continuously cycles through \
            grid positions and displays real-time tracking feedback. Useful for system \
            alignment and debugging. Press 'q' to quit."
    )]
    pub live: bool,

    #[arg(
        long,
        default_value = "50",
        help = "Number of measurements to keep in rolling history",
        long_help = "Number of recent measurements to retain per grid position for rolling \
            statistics in live mode. Larger values provide smoother statistics but use \
            more memory."
    )]
    pub history: usize,
}

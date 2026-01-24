//! Desktop calibration controller for remote optical calibration.
//!
//! Sends PatternCommand to calibrate_serve via REST API,
//! receives TrackingMessage from fgs_server via SSE,
//! and estimates the display-to-sensor affine transform.
//!
//! Supports live mode for real-time adjustment and visualization.

use clap::Parser;
use test_bench::calibration_controller::{live_mode, single_pass, Args};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    if args.live {
        live_mode::run(&args).await
    } else {
        single_pass::run(&args).await
    }
}

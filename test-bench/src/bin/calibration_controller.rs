//! Desktop calibration controller for remote optical calibration.
//!
//! Sends PatternCommand to calibrate_serve via ZMQ REQ/REP,
//! receives TrackingMessage from cam_track via ZMQ SUB,
//! and estimates the display-to-sensor affine transform.
//!
//! Supports live mode for real-time adjustment and visualization.

use clap::Parser;
use test_bench::calibration_controller::{live_mode, single_pass, Args};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.live {
        live_mode::run(&args)
    } else {
        single_pass::run(&args)
    }
}

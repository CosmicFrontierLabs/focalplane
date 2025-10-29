//! Unified camera HTTP server.
//!
//! Supports different camera types via feature flags:
//! - Mock camera: Always available
//! - PlayerOne cameras: Requires "playerone" feature
//! - NSV455 cameras: Requires "nsv455" feature
//!
//! IMPORTANT: playerone and nsv455 features are mutually exclusive due to USB conflicts.

use clap::Parser;
use test_bench::camera_init::{initialize_camera, CameraArgs};
use test_bench::camera_server::CommonServerArgs;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "HTTP server for astronomy cameras")]
struct Args {
    #[command(flatten)]
    camera: CameraArgs,

    #[command(flatten)]
    server: CommonServerArgs,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Initializing camera...");
    let camera = initialize_camera(&args.camera)?;

    info!("Starting camera server...");
    test_bench::camera_server::run_server(camera, args.server).await
}

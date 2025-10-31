//! Unified camera HTTP server for all camera types.

use clap::Parser;
use test_bench::camera_init::{initialize_camera, CameraArgs};
use test_bench::camera_server::CommonServerArgs;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "HTTP server for all camera types")]
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

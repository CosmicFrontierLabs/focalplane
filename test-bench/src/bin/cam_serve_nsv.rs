//! NSV455 astronomy camera HTTP server.
//!
//! IMPORTANT: This binary CANNOT be combined with POA cameras in the same executable.
//! The v4l2 libraries (v4l, v4l2-sys-mit) enumerate and claim USB video devices at
//! program initialization, which conflicts with the PlayerOne SDK's USB device access.

use clap::Parser;
use nsv455::camera::nsv455_camera::NSV455Camera;
use shared::camera_interface::CameraInterface;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use test_bench::camera_server::{analysis_loop, capture_loop, AppState, FrameStats};
use tokio::sync::{Mutex, RwLock};
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "HTTP server for NSV455 astronomy camera")]
struct Args {
    #[arg(short = 'p', long, default_value = "3000")]
    port: u16,

    #[arg(short = 'b', long, default_value = "0.0.0.0")]
    bind_address: String,

    #[arg(short = 'd', long, default_value = "/dev/video0")]
    device_path: String,

    #[arg(long, default_value = "1024")]
    width: usize,

    #[arg(long, default_value = "1024")]
    height: usize,

    #[arg(short = 'e', long, default_value = "10")]
    exposure_ms: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!(
        "Initializing NSV455 camera at {} ({}x{})",
        args.device_path, args.width, args.height
    );
    let mut camera = NSV455Camera::new(
        args.device_path.clone(),
        args.width as u32,
        args.height as u32,
    )
    .map_err(|e| anyhow::anyhow!("Failed to initialize NSV455 camera: {e}"))?;

    let exposure = Duration::from_millis(args.exposure_ms);
    camera
        .set_exposure(exposure)
        .map_err(|e| anyhow::anyhow!("Failed to set exposure: {e}"))?;
    info!("Set camera exposure to {}ms", args.exposure_ms);

    let bit_depth = camera.get_bit_depth();
    info!("Camera bit depth: {}", bit_depth);

    let state = Arc::new(AppState {
        camera: Arc::new(Mutex::new(camera)),
        stats: Arc::new(Mutex::new(FrameStats::default())),
        latest_frame: Arc::new(RwLock::new(None)),
        latest_annotated: Arc::new(RwLock::new(None)),
        latest_overlay_svg: Arc::new(RwLock::new(None)),
        annotated_notify: Arc::new(tokio::sync::Notify::new()),
        zoom_region: Arc::new(RwLock::new(None)),
        zoom_notify: Arc::new(tokio::sync::Notify::new()),
        bit_depth,
    });

    info!("Starting background capture loop...");
    let capture_state = state.clone();
    tokio::spawn(async move {
        capture_loop(capture_state).await;
    });

    info!("Starting background analysis loop...");
    let analysis_state = state.clone();
    tokio::spawn(async move {
        analysis_loop(analysis_state).await;
    });

    let app = test_bench::camera_server::create_router(state);

    let addr: SocketAddr = format!("{}:{}", args.bind_address, args.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid bind address: {e}"))?;

    info!("Starting server on http://{}", addr);
    info!("Access camera status at http://{}", addr);
    info!("JPEG endpoint: http://{}/jpeg", addr);
    info!("Raw data endpoint: http://{}/raw", addr);
    info!("Stats endpoint: http://{}/stats", addr);
    info!("Annotated endpoint (JPEG): http://{}/annotated", addr);
    info!(
        "Annotated endpoint (uncompressed GRAY8): http://{}/annotated_raw",
        addr
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .map_err(|e| anyhow::anyhow!("Server error: {e}"))?;

    Ok(())
}

use clap::{Parser, ValueEnum};
use shared::camera_interface::mock::MockCameraInterface;
use shared::camera_interface::{CameraConfig, CameraInterface};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use test_bench::camera_server::{analysis_loop, capture_loop, AppState, FrameStats};
use test_bench::display_patterns::apriltag;
use test_bench::poa::camera::PlayerOneCamera;
use tokio::sync::{Mutex, RwLock};
use tracing::info;

#[derive(Debug, Clone, ValueEnum)]
enum CameraType {
    Poa,
    Nsv,
    Mock,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "HTTP server for astronomy cameras")]
struct Args {
    #[arg(short = 'p', long, default_value = "3000")]
    port: u16,

    #[arg(short = 'i', long, default_value = "0")]
    camera_id: i32,

    #[arg(short = 'b', long, default_value = "0.0.0.0")]
    bind_address: String,

    #[arg(short = 'c', long, default_value = "poa")]
    camera: CameraType,

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

    let mut camera: Box<dyn CameraInterface> = match args.camera {
        CameraType::Mock => {
            info!("Generating AprilTag calibration pattern...");
            info!("  Target size: {}x{}", args.width, args.height);

            let apriltag_frame = apriltag::generate_as_array(args.width, args.height)?;
            let inverted_frame = apriltag_frame.mapv(|v| 65535 - v);

            let config = CameraConfig {
                width: args.width,
                height: args.height,
                exposure: Duration::from_millis(100),
                bit_depth: 16,
            };

            Box::new(MockCameraInterface::new(config, inverted_frame))
        }
        CameraType::Poa => {
            info!("Initializing PlayerOne camera with ID {}", args.camera_id);
            let camera = PlayerOneCamera::new(args.camera_id)
                .map_err(|e| anyhow::anyhow!("Failed to initialize camera: {e}"))?;
            Box::new(camera)
        }
        CameraType::Nsv => {
            todo!("NSV455 camera support not yet implemented")
        }
    };

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
        annotated_notify: Arc::new(tokio::sync::Notify::new()),
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
    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Server error: {e}"))?;

    Ok(())
}

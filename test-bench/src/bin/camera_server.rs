use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use test_bench::camera_server::{AppState, FrameStats};
use test_bench::playerone_camera::PlayerOneCamera;
use tokio::sync::Mutex;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "HTTP server for PlayerOne astronomy cameras")]
struct Args {
    #[arg(short = 'p', long, default_value = "3000")]
    port: u16,

    #[arg(short = 'c', long, default_value = "0")]
    camera_id: i32,

    #[arg(short = 'b', long, default_value = "0.0.0.0")]
    bind_address: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Initializing PlayerOne camera with ID {}", args.camera_id);
    let camera = PlayerOneCamera::new(args.camera_id)
        .map_err(|e| anyhow::anyhow!("Failed to initialize camera: {e}"))?;

    let state = Arc::new(AppState {
        camera: Arc::new(Mutex::new(camera)),
        stats: Arc::new(Mutex::new(FrameStats::default())),
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

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Server error: {e}"))?;

    Ok(())
}

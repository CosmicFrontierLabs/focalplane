use clap::Parser;
use flight_software::camera::v4l2_utils::get_available_resolutions;
use flight_software::hil::{show_camera_info, AppState, FrameStats};
use flight_software::v4l2_capture::{CameraConfig, CaptureSession};
use log::{error, info};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "mjpeg_server")]
#[command(about = "Camera status server with V4L2", long_about = None)]
struct Args {
    /// Resolution index (use --list to see available)
    #[arg(short, long, default_value_t = 0)]
    resolution: usize,

    /// List available resolutions and exit
    #[arg(long)]
    list: bool,

    /// Padding size in pixels
    #[arg(short, long, default_value_t = 0)]
    padding: u32,

    /// Port to listen on
    #[arg(long, default_value_t = 9999)]
    port: u16,

    /// Video device path
    #[arg(long, default_value = "/dev/video0")]
    device: String,

    /// Timeout in seconds (0 = run forever)
    #[arg(long, default_value_t = 0)]
    timeout: u64,

    /// Show camera info and exit
    #[arg(long)]
    info: bool,
}

#[tokio::main]
async fn main() {
    // Initialize logging
    env_logger::init();

    let args = Args::parse();
    info!("=== MJPEG Server Starting ===");

    // If --info flag is set, show camera info and exit
    if args.info {
        match show_camera_info(&args.device) {
            Ok(_) => std::process::exit(0),
            Err(e) => {
                error!("Failed to query camera info: {e}");
                std::process::exit(1);
            }
        }
    }

    // Get available resolutions
    let resolutions = match get_available_resolutions(&args.device) {
        Ok(r) => r,
        Err(e) => {
            error!("Failed to query resolutions: {e}");
            std::process::exit(1);
        }
    };

    // If --list flag is set, show resolutions and exit
    if args.list {
        info!("=== Available Resolutions ===");
        for (i, res) in resolutions.iter().enumerate() {
            match res.fps {
                Some(fps) => info!("[{}] {}x{} @ {:.1} fps", i, res.width, res.height, fps),
                None => info!("[{}] {}x{} @ unknown fps", i, res.width, res.height),
            }
        }
        std::process::exit(0);
    }

    // Get selected resolution
    if args.resolution >= resolutions.len() {
        error!(
            "Invalid resolution index {}. Use --list to see available resolutions.",
            args.resolution
        );
        std::process::exit(1);
    }

    let selected = &resolutions[args.resolution];

    info!("Camera Server Configuration:");
    let fps_str = selected
        .fps
        .map_or("unknown fps".to_string(), |f| format!("{f:.1} fps"));
    info!(
        "  Resolution [{}]: {}x{} @ {}",
        args.resolution, selected.width, selected.height, fps_str
    );
    info!("  Padding: {} pixels", args.padding);
    info!("  Device: {}", args.device);
    info!("  Port: {}", args.port);
    if args.timeout > 0 {
        info!("  Timeout: {} seconds", args.timeout);
    }

    // Initialize V4L2 capture session
    let camera_config = CameraConfig {
        device_path: args.device.clone(),
        width: selected.width,
        height: selected.height,
        ..Default::default()
    };

    info!("Creating capture session with config:");
    info!("  Device: {}", camera_config.device_path);
    info!(
        "  Resolution: {}x{}",
        camera_config.width, camera_config.height
    );
    info!("  Gain: {}", camera_config.gain);
    info!("  Exposure: {}", camera_config.exposure);

    let mut session = match CaptureSession::new(&camera_config) {
        Ok(s) => {
            info!("Capture session created successfully");
            s
        }
        Err(e) => {
            error!("Failed to initialize camera: {e}");
            std::process::exit(1);
        }
    };

    info!("Starting camera stream...");
    if let Err(e) = session.start_stream() {
        error!("Failed to start camera stream: {e}");
        std::process::exit(1);
    }

    info!("Camera stream started successfully!");

    let state = Arc::new(AppState {
        width: selected.width,
        height: selected.height,
        padding: args.padding,
        session: Arc::new(Mutex::new(session)),
        device_path: args.device.clone(),
        stats: Arc::new(Mutex::new(FrameStats::default())),
    });

    let app = flight_software::hil::create_router(state);

    let addr = format!("0.0.0.0:{}", args.port);
    info!("Starting camera status server on http://{addr}/");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind port");

    if args.timeout > 0 {
        tokio::select! {
            _ = axum::serve(listener, app) => {
                info!("Server stopped");
            }
            _ = tokio::time::sleep(Duration::from_secs(args.timeout)) => {
                info!("Server timeout after {} seconds", args.timeout);
            }
        }
    } else {
        axum::serve(listener, app).await.expect("Server failed");
    }

    info!("Server stopped.");
}

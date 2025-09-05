use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{Html, Response},
    routing::get,
    Router,
};
use clap::Parser;
use flight_software::camera::neutralino_imx455::read_sensor_temperatures;
use flight_software::camera::v4l2_utils::{
    collect_camera_metadata, get_available_resolutions, query_menu_item,
};
use flight_software::v4l2_capture::{CameraConfig, CaptureSession};
use image::{ImageBuffer, Luma};
use log::{debug, error, info, warn};
use std::os::unix::io::AsRawFd;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Duration;
use v4l::prelude::*;
use v4l::video::Capture;

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

struct FrameStats {
    total_frames: u64,
    last_frame_time: std::time::Instant,
    fps_samples: Vec<f32>,
    last_histogram: Vec<u32>,
    fpga_temp_celsius: Option<f32>,
    pcb_temp_celsius: Option<f32>,
    last_temp_update: std::time::Instant,
    last_timestamp_sec: i64,
    last_timestamp_usec: i64,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            total_frames: 0,
            last_frame_time: std::time::Instant::now(),
            fps_samples: Vec::with_capacity(10),
            last_histogram: vec![0; 256],
            fpga_temp_celsius: None,
            pcb_temp_celsius: None,
            last_temp_update: std::time::Instant::now(),
            last_timestamp_sec: 0,
            last_timestamp_usec: 0,
        }
    }
}

struct AppState {
    width: u32,
    height: u32,
    #[allow(dead_code)]
    padding: u32,
    session: Arc<Mutex<CaptureSession<'static>>>,
    device_path: String,
    stats: Arc<Mutex<FrameStats>>,
}

fn show_camera_info(device_path: &str) -> anyhow::Result<()> {
    let device = Device::with_path(device_path)?;

    info!("=== Camera Information ===");
    info!("Device: {device_path}");

    // Query device capabilities
    let caps = device.query_caps()?;
    info!("Driver: {}", caps.driver);
    info!("Card: {}", caps.card);
    info!("Bus: {}", caps.bus);

    // List supported formats
    info!("=== Supported Formats ===");
    let formats = device.enum_formats()?;

    for fmt in formats {
        let fourcc_bytes = fmt.fourcc.repr;
        let fourcc_str = std::str::from_utf8(&fourcc_bytes).unwrap_or("????");
        info!("Format: {} ({})", fourcc_str, fmt.description);
        debug!("  FourCC bytes: {fourcc_bytes:?}");

        // List frame sizes for this format
        if let Ok(framesizes) = device.enum_framesizes(fmt.fourcc) {
            info!("  Supported resolutions:");
            for (i, size) in framesizes.into_iter().enumerate() {
                match size.size {
                    v4l::framesize::FrameSizeEnum::Discrete(discrete) => {
                        info!("    {}. {}x{}", i + 1, discrete.width, discrete.height);

                        // Try to get frame intervals for this resolution
                        if let Ok(intervals) =
                            device.enum_frameintervals(fmt.fourcc, discrete.width, discrete.height)
                        {
                            for interval in intervals {
                                match interval.interval {
                                    v4l::frameinterval::FrameIntervalEnum::Discrete(disc) => {
                                        let fps = disc.denominator as f64 / disc.numerator as f64;
                                        info!("       - {fps:.2} fps");
                                    }
                                    v4l::frameinterval::FrameIntervalEnum::Stepwise(step) => {
                                        let min_fps =
                                            step.min.denominator as f64 / step.min.numerator as f64;
                                        let max_fps =
                                            step.max.denominator as f64 / step.max.numerator as f64;
                                        info!("       - {min_fps:.2} to {max_fps:.2} fps");
                                    }
                                }
                            }
                        }
                    }
                    v4l::framesize::FrameSizeEnum::Stepwise(stepwise) => {
                        info!(
                            "    Stepwise from {}x{} to {}x{} (step: {}x{})",
                            stepwise.min_width,
                            stepwise.min_height,
                            stepwise.max_width,
                            stepwise.max_height,
                            stepwise.step_width,
                            stepwise.step_height
                        );
                    }
                }
            }
        }
    }

    // List camera controls
    info!("=== Camera Controls ===");
    if let Ok(controls) = device.query_controls() {
        for ctrl in controls {
            info!(
                "  {}: {} (ID: {})",
                ctrl.name,
                match ctrl.typ {
                    v4l::control::Type::Integer => "Integer",
                    v4l::control::Type::Boolean => "Boolean",
                    v4l::control::Type::Menu => "Menu",
                    v4l::control::Type::Button => "Button",
                    v4l::control::Type::Integer64 => "Integer64",
                    v4l::control::Type::String => "String",
                    _ => "Unknown",
                },
                ctrl.id
            );

            // Show range for integer controls
            if matches!(
                ctrl.typ,
                v4l::control::Type::Integer | v4l::control::Type::Integer64
            ) {
                info!(
                    "    Range: {} to {} (step: {})",
                    ctrl.minimum, ctrl.maximum, ctrl.step
                );
                info!("    Default: {}", ctrl.default);
            }

            // Show menu items for menu controls
            if ctrl.typ == v4l::control::Type::Menu {
                info!("    Menu items:");
                // Open device separately for raw fd access
                if let Ok(fd) = std::fs::File::open(device_path) {
                    let raw_fd = fd.as_raw_fd();
                    let mut index = ctrl.minimum as u32;
                    let max = ctrl.maximum as u32;
                    while index <= max {
                        if let Some(menu_name) = query_menu_item(raw_fd, ctrl.id, index) {
                            info!("      [{index}] {menu_name}");
                        } else {
                            // Skip invalid indices silently
                        }
                        index += 1;
                    }
                } else {
                    warn!("    (Could not open device for menu enumeration)");
                }
            }
        }
    }

    Ok(())
}

async fn capture_frame_data(state: &AppState) -> Result<(Vec<u8>, v4l::buffer::Metadata), String> {
    let mut session = state.session.lock().await;
    let result = session
        .capture_frame()
        .map_err(|e| format!("Failed to capture frame: {e}"));

    // Update frame stats
    if let Ok((ref _frame_data, ref metadata)) = result {
        let mut stats = state.stats.lock().await;
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(stats.last_frame_time).as_secs_f32();

        if elapsed > 0.0 {
            let fps = 1.0 / elapsed;
            stats.fps_samples.push(fps);
            if stats.fps_samples.len() > 10 {
                stats.fps_samples.remove(0);
            }
        }

        stats.total_frames += 1;
        stats.last_frame_time = now;

        // Store timestamp from metadata
        stats.last_timestamp_sec = metadata.timestamp.sec;
        stats.last_timestamp_usec = metadata.timestamp.usec;

        // Update temperatures every second
        if now.duration_since(stats.last_temp_update).as_secs() >= 1 {
            let (fpga, pcb) = read_sensor_temperatures(&state.device_path);
            stats.fpga_temp_celsius = fpga;
            stats.pcb_temp_celsius = pcb;
            stats.last_temp_update = now;
        }
    }

    result
}

fn compute_histogram(pixels: &[u8]) -> Vec<u32> {
    let mut histogram = vec![0u32; 256];
    for &pixel in pixels {
        histogram[pixel as usize] += 1;
    }
    histogram
}

fn process_raw_to_pixels(frame_data: &[u8], width: u32, height: u32) -> Vec<u8> {
    // Only the 8096x6324 resolution has 96 pixel padding
    // All other resolutions have different stride patterns
    let stride = if width == 8096 && height == 6324 {
        // Max resolution: 96 pixel padding (192 bytes)
        (width as usize + 96) * 2
    } else {
        // For all other resolutions, calculate stride from actual data size
        frame_data.len() / height as usize
    };

    let stride_pixels = stride / 2;
    let padding_pixels = stride_pixels - width as usize;
    let mut pixels_8bit = Vec::with_capacity((width * height) as usize);

    debug!(
        "Frame {width}x{height}: stride={stride} bytes ({width}px + {padding_pixels}px padding)"
    );

    for y in 0..height {
        let row_start = (y as usize) * stride;
        for x in 0..width {
            let pixel_offset = row_start + (x as usize) * 2;
            if pixel_offset + 1 < frame_data.len() {
                let value =
                    u16::from_le_bytes([frame_data[pixel_offset], frame_data[pixel_offset + 1]]);
                pixels_8bit.push((value >> 8) as u8);
            }
        }
    }
    pixels_8bit
}

async fn raw_frame_endpoint(State(state): State<Arc<AppState>>) -> Response {
    match capture_frame_data(&state).await {
        Ok((frame_data, metadata)) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header("X-Frame-Width", state.width.to_string())
            .header("X-Frame-Height", state.height.to_string())
            .header("X-Frame-Format", "RG16")
            .header("X-Timestamp-Sec", metadata.timestamp.sec.to_string())
            .header("X-Timestamp-Usec", metadata.timestamp.usec.to_string())
            .body(Body::from(frame_data))
            .unwrap(),
        Err(e) => Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(e))
            .unwrap(),
    }
}

async fn jpeg_frame_endpoint(State(state): State<Arc<AppState>>) -> Response {
    match capture_frame_data(&state).await {
        Ok((frame_data, metadata)) => {
            let pixels_8bit = process_raw_to_pixels(&frame_data, state.width, state.height);

            // Compute and store histogram for stats
            {
                let histogram = compute_histogram(&pixels_8bit);
                let mut stats = state.stats.lock().await;
                stats.last_histogram = histogram;
            }

            match ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(state.width, state.height, pixels_8bit)
            {
                Some(img) => {
                    let mut jpeg_bytes = Vec::new();
                    let mut cursor = std::io::Cursor::new(&mut jpeg_bytes);
                    if img.write_to(&mut cursor, image::ImageFormat::Jpeg).is_ok() {
                        Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "image/jpeg")
                            .header("X-Timestamp-Sec", metadata.timestamp.sec.to_string())
                            .header("X-Timestamp-Usec", metadata.timestamp.usec.to_string())
                            .body(Body::from(jpeg_bytes))
                            .unwrap()
                    } else {
                        Response::builder()
                            .status(StatusCode::INTERNAL_SERVER_ERROR)
                            .body(Body::from("Failed to encode JPEG"))
                            .unwrap()
                    }
                }
                None => Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::from("Failed to create image buffer"))
                    .unwrap(),
            }
        }
        Err(e) => Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(e))
            .unwrap(),
    }
}

async fn stats_endpoint(State(state): State<Arc<AppState>>) -> Response {
    let stats = state.stats.lock().await;

    // Calculate average FPS from samples
    let avg_fps = if stats.fps_samples.is_empty() {
        0.0
    } else {
        stats.fps_samples.iter().sum::<f32>() / stats.fps_samples.len() as f32
    };

    // Downsample histogram to 64 bins for display
    let mut histogram_64 = vec![0u32; 64];
    for (i, chunk) in stats.last_histogram.chunks(4).enumerate() {
        if i < 64 {
            histogram_64[i] = chunk.iter().sum();
        }
    }

    let histogram_str = histogram_64
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let fpga_temp_str = match stats.fpga_temp_celsius {
        Some(temp) => format!("{temp:.3}"),
        None => "null".to_string(),
    };

    let pcb_temp_str = match stats.pcb_temp_celsius {
        Some(temp) => format!("{temp:.3}"),
        None => "null".to_string(),
    };

    let json = format!(
        r#"{{"total_frames": {}, "current_fps": {:.1}, "errors": 0, "fpga_temp_c": {}, "pcb_temp_c": {}, "timestamp_sec": {}, "timestamp_usec": {}, "histogram": [{}]}}"#,
        stats.total_frames,
        avg_fps,
        fpga_temp_str,
        pcb_temp_str,
        stats.last_timestamp_sec,
        stats.last_timestamp_usec,
        histogram_str
    );

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))
        .unwrap()
}

const CAMERA_STATUS_HTML: &str = include_str!("../../templates/camera_status.html");

async fn camera_status_page(State(state): State<Arc<AppState>>) -> Html<String> {
    // Collect camera metadata
    let metadata = collect_camera_metadata(&state.device_path).unwrap_or_default();

    // Get all available resolutions
    let resolutions = [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8096, 6324),
    ];

    // Build resolutions list HTML
    let resolutions_list = resolutions
        .iter()
        .map(|(w, h)| {
            let class = if *w == state.width && *h == state.height {
                "resolution-active"
            } else {
                "resolution-inactive"
            };
            format!(r#"<div class="resolution-item {class}">{w}x{h}</div>"#)
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Build test patterns HTML
    let test_patterns = metadata
        .test_patterns
        .iter()
        .map(|t| format!(r#"<div class="metadata-item">{t}</div>"#))
        .collect::<Vec<_>>()
        .join("\n");

    // Build HTML page with template
    let html = CAMERA_STATUS_HTML
        .replace("{device}", &state.device_path)
        .replace("{driver}", &metadata.driver)
        .replace("{card}", &metadata.card)
        .replace("{width}", &state.width.to_string())
        .replace("{height}", &state.height.to_string())
        .replace("{resolutions_list}", &resolutions_list)
        .replace("{test_patterns}", &test_patterns);

    Html(html)
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

    let app = Router::new()
        .route("/", get(camera_status_page))
        .route("/jpeg", get(jpeg_frame_endpoint))
        .route("/raw", get(raw_frame_endpoint))
        .route("/stats", get(stats_endpoint))
        .with_state(state);

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
}

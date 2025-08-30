use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{Html, Response},
    routing::get,
    Router,
};
use clap::Parser;
use flight_software::v4l2_capture::{CameraConfig, CaptureSession};
use image::{ImageBuffer, Luma};
use log::{debug, error, info, warn};
use std::ffi::CStr;
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

#[derive(Default)]
struct CameraMetadata {
    driver: String,
    card: String,
    bus: String,
    formats: Vec<String>,
    resolutions: Vec<String>,
    controls: Vec<String>,
    test_patterns: Vec<String>,
}

#[derive(Debug, Clone)]
struct Resolution {
    width: u32,
    height: u32,
    fps: f64,
}

fn get_available_resolutions(device_path: &str) -> anyhow::Result<Vec<Resolution>> {
    let device = Device::with_path(device_path)?;
    let mut resolutions = Vec::new();

    // Get first format (RG16)
    if let Some(fmt) = device.enum_formats()?.into_iter().next() {
        if let Ok(framesizes) = device.enum_framesizes(fmt.fourcc) {
            for size in framesizes {
                if let v4l::framesize::FrameSizeEnum::Discrete(discrete) = size.size {
                    // Get frame rate for this resolution
                    let fps = if let Ok(intervals) =
                        device.enum_frameintervals(fmt.fourcc, discrete.width, discrete.height)
                    {
                        if let Some(interval) = intervals.into_iter().next() {
                            match interval.interval {
                                v4l::frameinterval::FrameIntervalEnum::Discrete(disc) => {
                                    disc.denominator as f64 / disc.numerator as f64
                                }
                                _ => 30.0, // Default fps
                            }
                        } else {
                            30.0
                        }
                    } else {
                        30.0
                    };

                    resolutions.push(Resolution {
                        width: discrete.width,
                        height: discrete.height,
                        fps,
                    });
                }
            }
        }
    }

    Ok(resolutions)
}

fn query_menu_item(fd: i32, ctrl_id: u32, index: u32) -> Option<String> {
    unsafe {
        #[repr(C)]
        struct v4l2_querymenu {
            id: u32,
            index: u32,
            name: [u8; 32],
            reserved: u32,
        }

        let mut querymenu: v4l2_querymenu = std::mem::zeroed();
        querymenu.id = ctrl_id;
        querymenu.index = index;

        // VIDIOC_QUERYMENU ioctl value
        const VIDIOC_QUERYMENU: std::os::raw::c_ulong = 0xc02c5625;

        let ret = libc::ioctl(
            fd,
            VIDIOC_QUERYMENU,
            &mut querymenu as *mut _ as *mut std::os::raw::c_void,
        );

        if ret == 0 {
            // Check if it's a named menu item
            if querymenu.name[0] != 0 {
                let c_str = CStr::from_ptr(querymenu.name.as_ptr() as *const std::os::raw::c_char);
                c_str.to_str().ok().map(|s| s.to_string())
            } else {
                Some("(unnamed item)".to_string())
            }
        } else {
            None
        }
    }
}

fn collect_camera_metadata(device_path: &str) -> anyhow::Result<CameraMetadata> {
    let device = Device::with_path(device_path)?;
    let mut metadata = CameraMetadata::default();

    // Query device capabilities
    let caps = device.query_caps()?;
    metadata.driver = caps.driver.clone();
    metadata.card = caps.card.clone();
    metadata.bus = caps.bus.clone();

    // List supported formats
    let formats = device.enum_formats()?;
    for fmt in formats {
        let fourcc_bytes = fmt.fourcc.repr;
        let fourcc_str = std::str::from_utf8(&fourcc_bytes).unwrap_or("????");
        metadata
            .formats
            .push(format!("{} ({})", fourcc_str, fmt.description));

        // List frame sizes for this format
        if let Ok(framesizes) = device.enum_framesizes(fmt.fourcc) {
            for size in framesizes.into_iter().take(5) {
                // Limit to first 5
                if let v4l::framesize::FrameSizeEnum::Discrete(discrete) = size.size {
                    metadata
                        .resolutions
                        .push(format!("{}x{}", discrete.width, discrete.height));
                }
            }
        }
    }

    // List camera controls
    if let Ok(controls) = device.query_controls() {
        for ctrl in controls {
            let control_info = format!(
                "{}: {} (ID: {})",
                ctrl.name,
                match ctrl.typ {
                    v4l::control::Type::Integer => "Integer",
                    v4l::control::Type::Boolean => "Boolean",
                    v4l::control::Type::Menu => "Menu",
                    v4l::control::Type::Integer64 => "Integer64",
                    _ => "Unknown",
                },
                ctrl.id
            );
            metadata.controls.push(control_info);

            // Special handling for Test Pattern
            if ctrl.name == "Test Pattern" && ctrl.typ == v4l::control::Type::Menu {
                if let Ok(fd) = std::fs::File::open(device_path) {
                    let raw_fd = fd.as_raw_fd();
                    let mut index = ctrl.minimum as u32;
                    let max = ctrl.maximum as u32;
                    while index <= max {
                        if let Some(menu_name) = query_menu_item(raw_fd, ctrl.id, index) {
                            metadata
                                .test_patterns
                                .push(format!("[{index}] {menu_name}"));
                        }
                        index += 1;
                    }
                }
            }
        }
    }

    Ok(metadata)
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

async fn capture_frame_data(state: &AppState) -> Result<Vec<u8>, String> {
    let mut session = state.session.lock().await;
    let result = session
        .capture_frame()
        .map_err(|e| format!("Failed to capture frame: {e}"));

    // Update frame stats
    if result.is_ok() {
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

fn read_sensor_temperatures(device_path: &str) -> (Option<f32>, Option<f32>) {
    // Read both FPGA and PCB temperatures
    let mut fpga_temp = None;
    let mut pcb_temp = None;

    if let Ok(device) = Device::with_path(device_path) {
        // Try iterating through controls to find specific temperature controls
        if let Ok(controls) = device.query_controls() {
            debug!("Enumerating controls for temperature readings...");
            for ctrl in controls {
                let name_lower = ctrl.name.to_lowercase();
                debug!("Found control: '{}' (type: {:?})", ctrl.name, ctrl.typ);

                // Check for FPGA temperature
                if name_lower.contains("fpga") && name_lower.contains("temperature") {
                    debug!("Matched FPGA temperature control");
                    if let Ok(control) = device.control(ctrl.id) {
                        match control.value {
                            v4l::control::Value::Integer(val) => {
                                fpga_temp = Some(val as f32 / 1000.0);
                                debug!(
                                    "FPGA temp (Integer): {} raw, {:.3}°C",
                                    val,
                                    val as f32 / 1000.0
                                );
                            }
                            _ => warn!("FPGA temp unexpected type: {:?}", control.value),
                        }
                    }
                }

                // Check for PCB/sensor temperature
                if name_lower.contains("sensor")
                    && name_lower.contains("pcb")
                    && name_lower.contains("temperature")
                {
                    debug!("Matched PCB temperature control");
                    if let Ok(control) = device.control(ctrl.id) {
                        match control.value {
                            v4l::control::Value::Integer(val) => {
                                pcb_temp = Some(val as f32 / 1000.0);
                                debug!(
                                    "PCB temp (Integer): {} raw, {:.3}°C",
                                    val,
                                    val as f32 / 1000.0
                                );
                            }
                            _ => warn!("PCB temp unexpected type: {:?}", control.value),
                        }
                    }
                }
            }
        }
    }

    debug!("Final temperatures - FPGA: {fpga_temp:?}°C, PCB: {pcb_temp:?}°C");
    (fpga_temp, pcb_temp)
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
        Ok(frame_data) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header("X-Frame-Width", state.width.to_string())
            .header("X-Frame-Height", state.height.to_string())
            .header("X-Frame-Format", "RG16")
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
        Ok(frame_data) => {
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
        r#"{{"total_frames": {}, "current_fps": {:.1}, "errors": 0, "fpga_temp_c": {}, "pcb_temp_c": {}, "histogram": [{}]}}"#,
        stats.total_frames, avg_fps, fpga_temp_str, pcb_temp_str, histogram_str
    );

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))
        .unwrap()
}

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

    // Build HTML page with 3-column layout
    let html = format!(
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>V4L2 Camera Monitor</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Courier New', monospace; 
            background: #0a0a0a; 
            color: #00ff00;
            display: flex;
            height: 100vh;
            overflow: hidden;
            font-size: 32px;
        }}
        
        .column {{
            height: 100vh;
            padding: 20px;
            overflow-y: auto;
        }}
        
        .left-panel {{
            width: 20%;
            background: #111;
            border-right: 1px solid #00ff00;
        }}
        
        .center-panel {{
            width: 60%;
            background: #000;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding-top: 20px;
        }}
        
        .right-panel {{
            width: 20%;
            background: #111;
            border-left: 1px solid #00ff00;
        }}
        
        h2 {{
            color: #00ff00;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 1px solid #00ff00;
            padding-bottom: 5px;
        }}
        
        .metadata-item {{
            margin: 10px 0;
            font-size: 0.9em;
        }}
        
        .metadata-label {{
            color: #00aa00;
        }}
        
        .resolution-item {{
            margin: 5px 0;
            font-size: 0.85em;
        }}
        
        .resolution-active {{
            color: #00ff00;
            font-weight: bold;
        }}
        
        .resolution-inactive {{
            color: #666666;
        }}
        
        #camera-frame {{
            max-width: 100%;
            max-height: 75vh;
            border: 2px solid #00ff00;
            background: #111;
            margin-top: 10px;
        }}
        
        .frame-info {{
            margin-bottom: 20px;
            text-align: center;
            font-size: 1.1em;
            order: -1;
        }}
        
        .stats-placeholder {{
            border: 1px dashed #00ff00;
            padding: 20px;
            margin: 10px 0;
            min-height: 100px;
            color: #00aa00;
            font-size: 0.9em;
        }}
        
        .error {{
            color: #ff0000;
        }}
        
        .loading {{
            color: #ffff00;
        }}
    </style>
</head>
<body>
    <div class="column left-panel">
        <h2>Camera Info</h2>
        <div class="metadata-item">
            <span class="metadata-label">Status:</span><br>
            <span id="connection-status" class="loading">Connecting...</span>
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Device:</span><br>{device}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Driver:</span><br>{driver}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Card:</span><br>{card}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Resolution:</span><br>{width}x{height}
        </div>
        
        <h2 style="margin-top: 30px;">Resolutions</h2>
        {resolutions_list}
        
        <h2 style="margin-top: 30px;">Test Patterns</h2>
        {test_patterns}
        
        <h2 style="margin-top: 30px;">Endpoints</h2>
        <div class="metadata-item">
            <a href="/jpeg" style="color: #00ff00;">JPEG Frame</a><br>
            <a href="/raw" style="color: #00ff00;">Raw Frame</a><br>
            <a href="/stats" style="color: #00ff00;">Frame Stats (JSON)</a>
        </div>
    </div>
    
    <div class="column center-panel">
        <div class="frame-info">
            <span id="update-time"></span>
        </div>
        <img id="camera-frame" src="/jpeg" alt="Camera Frame">
    </div>
    
    <div class="column right-panel">
        <h2>Statistics</h2>
        <div class="stats-placeholder">
            <div id="stats-fps">FPS: Calculating...</div>
            <div id="stats-frames">Frames: 0</div>
            <div id="stats-fpga-temp">FPGA Temp: --°C</div>
            <div id="stats-pcb-temp">PCB Temp: --°C</div>
            <div id="stats-errors">Errors: 0</div>
        </div>
        
        <h2 style="margin-top: 30px;">Histogram</h2>
        <canvas id="histogram-canvas" width="300" height="150" style="width: 100%; border: 1px solid #00ff00; background: #111;"></canvas>
    </div>
    
    <script>
        let errorCount = 0;
        
        function updateImage() {{
            const img = document.getElementById('camera-frame');
            const newImg = new Image();
            
            newImg.onload = function() {{
                img.src = newImg.src;
                
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').className = '';
                
                // Show time with millisecond precision (before AM/PM)
                const updateTime = new Date();
                const timeOptions = {{ hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' }};
                const baseTime = updateTime.toLocaleTimeString('en-US', timeOptions);
                const ms = String(updateTime.getMilliseconds()).padStart(3, '0');
                // Insert milliseconds before AM/PM
                const timeStr = baseTime.replace(/(\d{{2}}:\d{{2}}:\d{{2}})( [AP]M)/, '$1.' + ms + '$2');
                document.getElementById('update-time').textContent = 
                    'Updated: ' + timeStr;
                
                // Schedule next update
                setTimeout(updateImage, 100); // 10 FPS target
            }};
            
            newImg.onerror = function() {{
                errorCount++;
                document.getElementById('stats-errors').textContent = 'Errors: ' + errorCount;
                document.getElementById('connection-status').textContent = 'Connection Error';
                document.getElementById('connection-status').className = 'error';
                
                // Retry after delay
                setTimeout(updateImage, 1000);
            }};
            
            // Force reload with cache buster
            newImg.src = '/jpeg?t=' + Date.now();
        }}
        
        // Stats update function
        function updateStats() {{
            fetch('/stats')
                .then(response => response.json())
                .then(data => {{
                    document.getElementById('stats-fps').textContent = 'FPS: ' + data.current_fps.toFixed(1);
                    document.getElementById('stats-frames').textContent = 'Frames: ' + data.total_frames;
                    document.getElementById('stats-errors').textContent = 'Errors: ' + data.errors;
                    
                    // Display FPGA temperature
                    if (data.fpga_temp_c !== null && data.fpga_temp_c !== undefined) {{
                        document.getElementById('stats-fpga-temp').textContent = 'FPGA Temp: ' + data.fpga_temp_c.toFixed(3) + '°C';
                    }} else {{
                        document.getElementById('stats-fpga-temp').textContent = 'FPGA Temp: --°C';
                    }}
                    
                    // Display PCB temperature
                    if (data.pcb_temp_c !== null && data.pcb_temp_c !== undefined) {{
                        document.getElementById('stats-pcb-temp').textContent = 'PCB Temp: ' + data.pcb_temp_c.toFixed(3) + '°C';
                    }} else {{
                        document.getElementById('stats-pcb-temp').textContent = 'PCB Temp: --°C';
                    }}
                    
                    // Draw histogram
                    if (data.histogram) {{
                        drawHistogram(data.histogram);
                    }}
                }})
                .catch(err => {{
                    console.error('Stats fetch error:', err);
                }});
        }}
        
        function drawHistogram(histogram) {{
            const canvas = document.getElementById('histogram-canvas');
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            const barWidth = width / histogram.length;
            
            // Clear canvas
            ctx.fillStyle = '#111';
            ctx.fillRect(0, 0, width, height);
            
            // Find max value for scaling
            const maxVal = Math.max(...histogram, 1);
            
            // Draw bars
            ctx.fillStyle = '#00ff00';
            histogram.forEach((value, i) => {{
                const barHeight = (value / maxVal) * (height - 10);
                const x = i * barWidth;
                const y = height - barHeight;
                ctx.fillRect(x, y, barWidth - 1, barHeight);
            }});
            
            // Draw grid lines
            ctx.strokeStyle = '#004400';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 4; i++) {{
                const y = (height / 4) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }}
        }}
        
        // Start updates
        updateImage();
        setInterval(updateStats, 1000); // Update stats every second
    </script>
</body>
</html>
"#,
        device = state.device_path,
        driver = metadata.driver,
        card = metadata.card,
        width = state.width,
        height = state.height,
        resolutions_list = resolutions
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
            .join("\n"),
        test_patterns = metadata
            .test_patterns
            .iter()
            .map(|t| format!(r#"<div class="metadata-item">{t}</div>"#))
            .collect::<Vec<_>>()
            .join("\n")
    );

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
            info!("[{}] {}x{} @ {:.1} fps", i, res.width, res.height, res.fps);
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
    info!(
        "  Resolution [{}]: {}x{} @ {:.1} fps",
        args.resolution, selected.width, selected.height, selected.fps
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

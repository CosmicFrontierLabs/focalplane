use axum::{
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{Html, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};
use shared::camera_interface::{CameraInterface, FrameMetadata, SensorGeometry};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify, RwLock};

use crate::calibration_overlay::{
    analyze_calibration_pattern, render_annotated_image, render_svg_overlay,
};
use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct CommonServerArgs {
    #[arg(short = 'p', long, default_value = "3000")]
    pub port: u16,

    #[arg(short = 'b', long, default_value = "0.0.0.0")]
    pub bind_address: String,

    #[arg(short = 'e', long, default_value = "10")]
    pub exposure_ms: u64,

    #[arg(short = 'g', long, default_value = "1.0")]
    pub gain: f64,
}

type TimestampedFrame = (Array2<u16>, FrameMetadata, std::time::Instant);

#[derive(Debug, Clone)]
pub struct ZoomRegion {
    pub patch: Array2<u16>,
    pub center: (usize, usize),
    pub frame_number: u64,
    pub timestamp: std::time::Instant,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ZoomCoords {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug, Deserialize)]
pub struct ZoomQueryParams {
    pub x: usize,
    pub y: usize,
    #[serde(default = "default_zoom_size")]
    pub size: usize,
}

fn default_zoom_size() -> usize {
    64
}

pub struct AppState<C: CameraInterface> {
    pub camera: Arc<Mutex<C>>,
    pub stats: Arc<Mutex<FrameStats>>,
    pub latest_frame: Arc<RwLock<Option<TimestampedFrame>>>,
    pub latest_annotated: Arc<RwLock<Option<(DynamicImage, u64)>>>,
    pub latest_overlay_svg: Arc<RwLock<Option<(String, u64)>>>,
    pub annotated_notify: Arc<Notify>,
    pub zoom_region: Arc<RwLock<Option<ZoomRegion>>>,
    pub zoom_notify: Arc<Notify>,
    pub bit_depth: u8,
    pub camera_name: String,
    pub camera_geometry: SensorGeometry,
}

#[derive(Debug, Clone)]
pub struct FrameStats {
    pub total_frames: u64,
    pub fps_samples: Vec<f32>,
    #[allow(dead_code)]
    pub last_frame_time: std::time::Instant,
    pub last_temperatures: std::collections::HashMap<String, f64>,
    pub histogram: Vec<u32>,
    pub histogram_mean: f64,
    pub histogram_max: u16,
    pub capture_timing_ms: Vec<f32>,
    pub analysis_timing_ms: Vec<f32>,
    pub render_timing_ms: Vec<f32>,
    pub total_pipeline_ms: Vec<f32>,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            total_frames: 0,
            fps_samples: Vec::new(),
            last_frame_time: std::time::Instant::now(),
            last_temperatures: std::collections::HashMap::new(),
            histogram: Vec::new(),
            histogram_mean: 0.0,
            histogram_max: 0,
            capture_timing_ms: Vec::new(),
            analysis_timing_ms: Vec::new(),
            render_timing_ms: Vec::new(),
            total_pipeline_ms: Vec::new(),
        }
    }
}

fn extract_patch(
    frame: &Array2<u16>,
    center_x: usize,
    center_y: usize,
    patch_size: usize,
) -> Array2<u16> {
    let half_size = patch_size / 2;
    let frame_height = frame.nrows();
    let frame_width = frame.ncols();

    let x_start = center_x.saturating_sub(half_size);
    let y_start = center_y.saturating_sub(half_size);
    let x_end = (x_start + patch_size).min(frame_width);
    let y_end = (y_start + patch_size).min(frame_height);

    let actual_width = x_end - x_start;
    let actual_height = y_end - y_start;

    let mut patch = Array2::zeros((patch_size, patch_size));

    if actual_width > 0 && actual_height > 0 {
        let frame_slice = frame.slice(s![y_start..y_end, x_start..x_end]);
        patch
            .slice_mut(s![0..actual_height, 0..actual_width])
            .assign(&frame_slice);
    }

    patch
}

fn compute_histogram_u16(frame: &Array2<u16>) -> Vec<u32> {
    let mut histogram = vec![0u32; 256];
    for &pixel in frame.iter() {
        let bin = ((pixel as u32 * 256) / 65536) as usize;
        let bin = bin.min(255);
        histogram[bin] += 1;
    }
    histogram
}

fn compute_mean_u16(frame: &Array2<u16>) -> f64 {
    let sum: u64 = frame.iter().map(|&v| v as u64).sum();
    let count = frame.len() as u64;
    if count > 0 {
        sum as f64 / count as f64
    } else {
        0.0
    }
}

fn process_raw_to_pixels(frame: &Array2<u16>) -> Vec<u8> {
    let max_val = *frame.iter().max().unwrap_or(&1) as f32;
    let scale = if max_val > 0.0 { 255.0 / max_val } else { 1.0 };

    frame
        .iter()
        .map(|&val| ((val as f32) * scale) as u8)
        .collect()
}

async fn jpeg_frame_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    let frame_data = {
        let latest = state.latest_frame.read().await;
        latest.clone()
    };

    let (frame, _metadata) = match frame_data {
        Some((f, m, _timestamp)) => (f, m),
        None => {
            return Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(Body::from("No frame available yet"))
                .unwrap()
        }
    };

    let pixels_8bit = process_raw_to_pixels(&frame);

    let mut stats = state.stats.lock().await;
    stats.histogram = compute_histogram_u16(&frame);
    stats.histogram_mean = compute_mean_u16(&frame);
    stats.histogram_max = *frame.iter().max().unwrap_or(&0);
    drop(stats);

    let height = frame.nrows() as u32;
    let width = frame.ncols() as u32;

    let img = match ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, pixels_8bit) {
        Some(img) => img,
        None => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("Failed to create image buffer"))
                .unwrap()
        }
    };

    let mut jpeg_bytes = Vec::new();
    if let Err(e) = img.write_to(
        &mut std::io::Cursor::new(&mut jpeg_bytes),
        image::ImageFormat::Jpeg,
    ) {
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(format!("Failed to encode JPEG: {e}")))
            .unwrap();
    }

    Response::builder()
        .header(header::CONTENT_TYPE, "image/jpeg")
        .body(Body::from(jpeg_bytes))
        .unwrap()
}

async fn raw_frame_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    let frame_data = {
        let latest = state.latest_frame.read().await;
        latest.clone()
    };

    let (frame, metadata) = match frame_data {
        Some((f, m, _timestamp)) => (f, m),
        None => {
            return Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(Body::from("No frame available yet"))
                .unwrap()
        }
    };

    let pixels_8bit = process_raw_to_pixels(&frame);

    let mut stats = state.stats.lock().await;
    stats.histogram = compute_histogram_u16(&frame);
    stats.histogram_mean = compute_mean_u16(&frame);
    stats.histogram_max = *frame.iter().max().unwrap_or(&0);
    drop(stats);

    let height = frame.nrows();
    let width = frame.ncols();

    let encoded = base64::engine::general_purpose::STANDARD.encode(&pixels_8bit);

    let response = test_bench_shared::RawFrameResponse {
        width,
        height,
        timestamp_sec: metadata.timestamp.seconds,
        timestamp_nanos: metadata.timestamp.nanos,
        temperatures: metadata.temperatures.clone(),
        exposure_us: metadata.exposure.as_micros(),
        frame_number: metadata.frame_number,
        image_base64: encoded,
    };

    let json = serde_json::to_string(&response).unwrap();

    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))
        .unwrap()
}

async fn stats_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    let stats = state.stats.lock().await;
    let avg_fps = if stats.fps_samples.is_empty() {
        0.0
    } else {
        stats.fps_samples.iter().sum::<f32>() / stats.fps_samples.len() as f32
    };

    let avg_capture_ms = if stats.capture_timing_ms.is_empty() {
        0.0
    } else {
        stats.capture_timing_ms.iter().sum::<f32>() / stats.capture_timing_ms.len() as f32
    };

    let avg_analysis_ms = if stats.analysis_timing_ms.is_empty() {
        0.0
    } else {
        stats.analysis_timing_ms.iter().sum::<f32>() / stats.analysis_timing_ms.len() as f32
    };

    let avg_render_ms = if stats.render_timing_ms.is_empty() {
        0.0
    } else {
        stats.render_timing_ms.iter().sum::<f32>() / stats.render_timing_ms.len() as f32
    };

    let avg_total_pipeline_ms = if stats.total_pipeline_ms.is_empty() {
        0.0
    } else {
        stats.total_pipeline_ms.iter().sum::<f32>() / stats.total_pipeline_ms.len() as f32
    };

    let response = test_bench_shared::CameraStats {
        total_frames: stats.total_frames,
        avg_fps,
        temperatures: stats.last_temperatures.clone(),
        histogram: stats.histogram.clone(),
        histogram_mean: stats.histogram_mean,
        histogram_max: stats.histogram_max,
        timing: Some(test_bench_shared::CameraTimingStats {
            avg_capture_ms,
            avg_analysis_ms,
            avg_render_ms,
            avg_total_pipeline_ms,
            capture_samples: stats.capture_timing_ms.len(),
            analysis_samples: stats.analysis_timing_ms.len(),
        }),
    };

    let json = serde_json::to_string(&response).unwrap();

    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))
        .unwrap()
}

async fn camera_status_page<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Html<String> {
    let width = state.camera_geometry.width();
    let height = state.camera_geometry.height();
    let camera_name = &state.camera_name;

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Camera Monitor</title>
    <link rel="stylesheet" href="/static/shared-styles.css" />
    <script type="module">
        import init from '/static/camera_wasm.js';
        init();
    </script>
</head>
<body>
    <div id="app" data-device="{camera_name}" data-width="{width}" data-height="{height}"></div>
</body>
</html>"#
    );

    Html(html)
}

async fn annotated_frame_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    use std::time::Instant;

    let start_total = Instant::now();

    state.annotated_notify.notified().await;

    let annotated_opt = {
        let latest = state.latest_annotated.read().await;
        latest.clone()
    };

    match annotated_opt {
        Some((annotated_img, frame_num)) => {
            let rgb_img = annotated_img.to_rgb8();

            let start_compress = Instant::now();
            let mut jpeg_bytes = Vec::new();
            if let Err(e) = rgb_img.write_to(
                &mut std::io::Cursor::new(&mut jpeg_bytes),
                image::ImageFormat::Jpeg,
            ) {
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::from(format!("Failed to encode JPEG: {e}")))
                    .unwrap();
            }
            let compress_time = start_compress.elapsed();
            let total_time = start_total.elapsed();

            tracing::debug!(
                "Annotated frame (cached) timing: frame={}, compress={:.1}ms, total={:.1}ms",
                frame_num,
                compress_time.as_secs_f64() * 1000.0,
                total_time.as_secs_f64() * 1000.0
            );

            Response::builder()
                .header(header::CONTENT_TYPE, "image/jpeg")
                .header("X-Frame-Number", frame_num.to_string())
                .body(Body::from(jpeg_bytes))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .body(Body::from("No annotated frame available yet"))
            .unwrap(),
    }
}

async fn annotated_raw_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    state.annotated_notify.notified().await;

    let annotated_opt = {
        let latest = state.latest_annotated.read().await;
        latest.clone()
    };

    match annotated_opt {
        Some((annotated_img, frame_num)) => {
            let gray_img = annotated_img.to_luma8();
            let (width, height) = gray_img.dimensions();
            let raw_bytes = gray_img.into_raw();

            Response::builder()
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .header("X-Image-Width", width.to_string())
                .header("X-Image-Height", height.to_string())
                .header("X-Image-Format", "GRAY8")
                .header("X-Frame-Number", frame_num.to_string())
                .body(Body::from(raw_bytes))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .body(Body::from("No annotated frame available yet"))
            .unwrap(),
    }
}

async fn overlay_svg_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    state.annotated_notify.notified().await;

    let svg_opt = {
        let latest = state.latest_overlay_svg.read().await;
        latest.clone()
    };

    match svg_opt {
        Some((svg_data, frame_num)) => Response::builder()
            .header(header::CONTENT_TYPE, "image/svg+xml")
            .header("X-Frame-Number", frame_num.to_string())
            .body(Body::from(svg_data))
            .unwrap(),
        None => Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .body(Body::from("No overlay SVG available yet"))
            .unwrap(),
    }
}

async fn fits_frame_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    let frame_data = {
        let latest = state.latest_frame.read().await;
        latest.clone()
    };

    let (frame, metadata) = match frame_data {
        Some((f, m, _timestamp)) => (f, m),
        None => {
            return Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(Body::from("No frame available yet"))
                .unwrap()
        }
    };

    // Create FITS file in memory using a temporary file
    let temp_file = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(format!("Failed to create temp file: {e}")))
                .unwrap()
        }
    };

    // Write frame to FITS file
    let mut fits_data = std::collections::HashMap::new();
    fits_data.insert(
        "FRAME".to_string(),
        simulator::io::fits::FitsDataType::UInt16(frame),
    );

    if let Err(e) = simulator::io::fits::write_typed_fits(&fits_data, temp_file.path()) {
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(format!("Failed to write FITS: {e}")))
            .unwrap();
    }

    // Read the FITS file into memory
    let fits_bytes = match std::fs::read(temp_file.path()) {
        Ok(bytes) => bytes,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(format!("Failed to read FITS file: {e}")))
                .unwrap()
        }
    };

    // Generate filename with frame number
    let filename = format!("frame_{:06}.fits", metadata.frame_number);

    Response::builder()
        .header(header::CONTENT_TYPE, "application/fits")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{filename}\""),
        )
        .body(Body::from(fits_bytes))
        .unwrap()
}

async fn set_zoom_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
    Json(coords): Json<ZoomCoords>,
) -> Response {
    let frame_data = {
        let latest = state.latest_frame.read().await;
        latest.clone()
    };

    match frame_data {
        Some((frame, _metadata, _timestamp)) => {
            let patch = extract_patch(&frame, coords.x, coords.y, 64);

            let mut zoom = state.zoom_region.write().await;
            *zoom = Some(ZoomRegion {
                patch,
                center: (coords.x, coords.y),
                frame_number: _metadata.frame_number,
                timestamp: std::time::Instant::now(),
            });
            drop(zoom);

            state.zoom_notify.notify_waiters();

            tracing::info!("Zoom region set at ({}, {})", coords.x, coords.y);

            Response::builder()
                .status(StatusCode::OK)
                .body(Body::from("Zoom region set"))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .body(Body::from("No frame available yet"))
            .unwrap(),
    }
}

async fn get_zoom_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
    axum::extract::Query(params): axum::extract::Query<ZoomQueryParams>,
) -> Response {
    let frame_data = {
        let latest = state.latest_frame.read().await;
        latest.clone()
    };

    match frame_data {
        Some((frame, metadata, _timestamp)) => {
            let patch = extract_patch(&frame, params.x, params.y, params.size);
            let scale_factor = 4;
            let scaled_size = params.size * scale_factor;

            let max_val = *patch.iter().max().unwrap_or(&1) as f32;
            let scale = if max_val > 0.0 { 255.0 / max_val } else { 1.0 };

            let mut scaled_patch = Vec::with_capacity(scaled_size * scaled_size);
            for y in 0..scaled_size {
                let src_y = y / scale_factor;
                for x in 0..scaled_size {
                    let src_x = x / scale_factor;
                    if src_y < patch.nrows() && src_x < patch.ncols() {
                        let val = patch[[src_y, src_x]];
                        let scaled_val = ((val as f32) * scale) as u8;
                        scaled_patch.push(scaled_val);
                    } else {
                        scaled_patch.push(0);
                    }
                }
            }

            let img = match ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                scaled_size as u32,
                scaled_size as u32,
                scaled_patch,
            ) {
                Some(img) => img,
                None => {
                    return Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::from("Failed to create image buffer"))
                        .unwrap()
                }
            };

            let mut jpeg_bytes = Vec::new();
            if let Err(e) = img.write_to(
                &mut std::io::Cursor::new(&mut jpeg_bytes),
                image::ImageFormat::Jpeg,
            ) {
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::from(format!("Failed to encode JPEG: {e}")))
                    .unwrap();
            }

            Response::builder()
                .header(header::CONTENT_TYPE, "image/jpeg")
                .header("X-Center-X", params.x.to_string())
                .header("X-Center-Y", params.y.to_string())
                .header("X-Frame-Number", metadata.frame_number.to_string())
                .body(Body::from(jpeg_bytes))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .body(Body::from("No frame available yet"))
            .unwrap(),
    }
}

async fn logging_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request,
    next: Next,
) -> Response {
    let start = std::time::Instant::now();
    let method = req.method().clone();
    let uri = req.uri().clone();

    let response = next.run(req).await;

    let elapsed = start.elapsed();
    tracing::info!(
        "{} {} from {} - {:.1}ms",
        method,
        uri.path(),
        addr.ip(),
        elapsed.as_secs_f64() * 1000.0
    );

    response
}

pub fn create_router<C: CameraInterface + 'static>(state: Arc<AppState<C>>) -> Router {
    use axum::routing::get_service;
    use tower_http::services::ServeDir;

    Router::new()
        .route("/", get(camera_status_page::<C>))
        .route("/jpeg", get(jpeg_frame_endpoint::<C>))
        .route("/raw", get(raw_frame_endpoint::<C>))
        .route("/fits", get(fits_frame_endpoint::<C>))
        .route("/annotated", get(annotated_frame_endpoint::<C>))
        .route("/annotated_raw", get(annotated_raw_endpoint::<C>))
        .route("/overlay-svg", get(overlay_svg_endpoint::<C>))
        .route("/stats", get(stats_endpoint::<C>))
        .route(
            "/zoom",
            post(set_zoom_endpoint::<C>).get(get_zoom_endpoint::<C>),
        )
        .nest_service(
            "/static",
            get_service(ServeDir::new("test-bench-frontend/dist/camera")),
        )
        .with_state(state)
        .layer(middleware::from_fn(logging_middleware))
}

pub async fn run_server<C: CameraInterface + Send + 'static>(
    mut camera: C,
    args: CommonServerArgs,
) -> anyhow::Result<()> {
    use tracing::info;

    let exposure = std::time::Duration::from_millis(args.exposure_ms);
    camera
        .set_exposure(exposure)
        .map_err(|e| anyhow::anyhow!("Failed to set exposure: {e}"))?;
    info!("Set camera exposure to {}ms", args.exposure_ms);

    camera
        .set_gain(args.gain)
        .map_err(|e| anyhow::anyhow!("Failed to set gain: {e}"))?;
    info!("Set camera gain to {}", args.gain);

    let bit_depth = camera.get_bit_depth();
    info!("Camera bit depth: {}", bit_depth);

    let camera_name = camera.name().to_string();
    let camera_geometry = camera.geometry();
    info!(
        "Camera: {} ({}x{})",
        camera_name,
        camera_geometry.width(),
        camera_geometry.height()
    );

    let state = Arc::new(AppState {
        camera: Arc::new(Mutex::new(camera)),
        stats: Arc::new(Mutex::new(FrameStats::default())),
        latest_frame: Arc::new(RwLock::new(None)),
        latest_annotated: Arc::new(RwLock::new(None)),
        latest_overlay_svg: Arc::new(RwLock::new(None)),
        annotated_notify: Arc::new(Notify::new()),
        zoom_region: Arc::new(RwLock::new(None)),
        zoom_notify: Arc::new(Notify::new()),
        bit_depth: bit_depth.as_u8(),
        camera_name,
        camera_geometry,
    });

    info!("Starting background capture loop...");
    let capture_state = state.clone();
    std::thread::spawn(move || {
        capture_loop_blocking(capture_state);
    });

    info!("Starting background analysis loop...");
    let analysis_state = state.clone();
    tokio::spawn(async move {
        analysis_loop(analysis_state).await;
    });

    let app = create_router(state);

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

pub fn capture_loop_blocking<C: CameraInterface + Send + 'static>(state: Arc<AppState<C>>) {
    let mut camera = state.camera.blocking_lock();

    let state_clone = state.clone();
    let result = camera.stream(&mut |frame, metadata| {
        let start_capture = std::time::Instant::now();
        let frame_num = metadata.frame_number;

        let frame_owned = frame.clone();
        let metadata_owned = metadata.clone();

        {
            let mut stats = state_clone.stats.blocking_lock();
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(stats.last_frame_time).as_secs_f32();

            if elapsed > 0.0 {
                let fps = 1.0 / elapsed;
                stats.fps_samples.push(fps);
                if stats.fps_samples.len() > 10 {
                    stats.fps_samples.remove(0);
                }
            }

            let capture_time = start_capture.elapsed();
            stats
                .capture_timing_ms
                .push(capture_time.as_secs_f32() * 1000.0);
            if stats.capture_timing_ms.len() > 100 {
                stats.capture_timing_ms.remove(0);
            }

            stats.total_frames += 1;
            stats.last_frame_time = now;
            stats.last_temperatures = metadata_owned.temperatures.clone();

            tracing::debug!(
                "Stream: {:.1}ms, frame_num={}",
                capture_time.as_secs_f64() * 1000.0,
                frame_num
            );
        }

        {
            let capture_timestamp = std::time::Instant::now();
            let mut latest = state_clone.latest_frame.blocking_write();
            *latest = Some((frame_owned, metadata_owned, capture_timestamp));
        }

        true
    });

    if let Err(e) = result {
        tracing::error!("Camera stream error: {e}");
    }
}

pub async fn analysis_loop<C: CameraInterface + Send + 'static>(state: Arc<AppState<C>>) {
    let mut last_analysis_time = std::time::Instant::now();

    loop {
        let start_total = std::time::Instant::now();
        let analysis_interval = start_total.duration_since(last_analysis_time);

        let frame_data = {
            let latest = state.latest_frame.read().await;
            latest.clone()
        };

        if let Some((frame, metadata, capture_timestamp)) = frame_data {
            let frame_age = start_total.duration_since(capture_timestamp);
            let bit_depth = state.bit_depth;
            let frame_num = metadata.frame_number;

            let width = state.camera_geometry.width() as u32;
            let height = state.camera_geometry.height() as u32;

            let frame_for_analysis = frame.clone();
            let result = tokio::task::spawn_blocking(move || {
                let start_analysis = std::time::Instant::now();
                let analysis = analyze_calibration_pattern(&frame_for_analysis, bit_depth)?;
                let analysis_time = start_analysis.elapsed();

                let start_render = std::time::Instant::now();
                let annotated_img = render_annotated_image(&frame_for_analysis, &analysis)?;
                let svg_overlay = render_svg_overlay(width, height, &analysis)?;
                let render_time = start_render.elapsed();

                Ok::<_, anyhow::Error>((annotated_img, svg_overlay, analysis_time, render_time))
            })
            .await;

            match result {
                Ok(Ok((annotated_img, svg_overlay, analysis_time, render_time))) => {
                    let total_time = start_total.elapsed();
                    last_analysis_time = start_total;

                    let mut stats = state.stats.lock().await;
                    stats
                        .analysis_timing_ms
                        .push(analysis_time.as_secs_f32() * 1000.0);
                    if stats.analysis_timing_ms.len() > 100 {
                        stats.analysis_timing_ms.remove(0);
                    }

                    stats
                        .render_timing_ms
                        .push(render_time.as_secs_f32() * 1000.0);
                    if stats.render_timing_ms.len() > 100 {
                        stats.render_timing_ms.remove(0);
                    }

                    stats
                        .total_pipeline_ms
                        .push(total_time.as_secs_f32() * 1000.0);
                    if stats.total_pipeline_ms.len() > 100 {
                        stats.total_pipeline_ms.remove(0);
                    }
                    drop(stats);

                    let mut latest = state.latest_annotated.write().await;
                    *latest = Some((annotated_img, frame_num));
                    drop(latest);

                    let mut latest_svg = state.latest_overlay_svg.write().await;
                    *latest_svg = Some((svg_overlay, frame_num));
                    drop(latest_svg);

                    state.annotated_notify.notify_waiters();

                    let zoom_center = {
                        let zoom = state.zoom_region.read().await;
                        zoom.as_ref().map(|z| z.center)
                    };

                    if let Some((cx, cy)) = zoom_center {
                        let patch = extract_patch(&frame, cx, cy, 64);
                        let mut zoom = state.zoom_region.write().await;
                        *zoom = Some(ZoomRegion {
                            patch,
                            center: (cx, cy),
                            frame_number: frame_num,
                            timestamp: std::time::Instant::now(),
                        });
                        drop(zoom);
                        state.zoom_notify.notify_waiters();
                    }

                    tracing::debug!(
                        "Pipeline: frame={}, interval={:.1}ms, age={:.1}ms, analysis={:.1}ms, render={:.1}ms, total={:.1}ms",
                        frame_num,
                        analysis_interval.as_secs_f64() * 1000.0,
                        frame_age.as_secs_f64() * 1000.0,
                        analysis_time.as_secs_f64() * 1000.0,
                        render_time.as_secs_f64() * 1000.0,
                        total_time.as_secs_f64() * 1000.0
                    );
                }
                Ok(Err(e)) => {
                    tracing::error!("Analysis/render error in analysis loop: {e}");
                }
                Err(e) => {
                    tracing::error!("Task join error in analysis loop: {e}");
                }
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}

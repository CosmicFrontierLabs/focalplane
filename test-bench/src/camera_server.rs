use axum::{
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        Response,
    },
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use hardware::pi::S330;
use image::{DynamicImage, ImageBuffer, Luma};
use monocle::config::{FgsConfig, GuideStarFilters};
use monocle::state::FgsEvent;
use monocle::{apply_camera_settings, FgsState, FineGuidanceSystem};
use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};
use shared::bad_pixel_map::BadPixelMap;
use shared::camera_interface::{CameraInterface, FrameMetadata, SensorGeometry};
use shared::frame_writer::{FrameFormat, FrameWriterHandle};
use shared::image_proc::u16_to_gray_image;
use shared::tracking_message::TrackingMessage;
use shared_wasm::{
    CameraStats, CameraTimingStats, ExportSettings, ExportStatus, FrameExportMetadata,
    FsmMoveRequest, FsmStatus, RawFrameResponse, TrackingEnableRequest, TrackingPosition,
    TrackingSettings, TrackingState, TrackingStatus,
};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, Notify, RwLock};

use crate::camera_init::ExposureArgs;
use crate::embedded_assets::{serve_fgs_frontend, serve_fgs_index_with_data};
use crate::mjpeg::encode_ndarray_jpeg;
use crate::ws_stream::{WsBroadcaster, WsFrame};
use axum::extract::ws::WebSocketUpgrade;
use clap::Args;

/// Common command-line arguments for camera server binaries.
#[derive(Args, Debug, Clone)]
pub struct CommonServerArgs {
    #[arg(
        short = 'p',
        long,
        default_value = "3000",
        help = "HTTP server port",
        long_help = "TCP port for the HTTP/REST server. The web UI will be available at \
            http://<bind_address>:<port>/. Default: 3000."
    )]
    pub port: u16,

    #[arg(
        short = 'b',
        long,
        default_value = "0.0.0.0",
        help = "HTTP server bind address",
        long_help = "IP address to bind the HTTP server to. Use '0.0.0.0' to listen on all \
            interfaces (required for remote access), or '127.0.0.1' for localhost-only access."
    )]
    pub bind_address: String,

    #[command(flatten)]
    pub exposure: ExposureArgs,

    #[arg(
        short = 'g',
        long,
        default_value = "100.0",
        help = "Initial camera gain setting",
        long_help = "Initial analog gain setting for the camera sensor. Can be adjusted at \
            runtime via the web UI. Higher gain amplifies both signal and noise. The valid \
            range depends on the camera model. Typical range: 0-500."
    )]
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

/// Minimum interval between MJPEG frame publishes (5 fps max).
const MJPEG_MIN_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);

/// Shared tracking state that can be accessed from both the capture loop and HTTP handlers
pub struct TrackingSharedState {
    /// Whether tracking mode is enabled
    pub enabled: AtomicBool,
    /// Signal to request restart of capture loop (e.g., when toggling tracking)
    pub restart_requested: AtomicBool,
    /// Current tracking state (serialized as JSON for thread-safe access)
    pub status: RwLock<TrackingStatus>,
    /// Total tracking updates
    pub total_updates: AtomicU64,
    /// Runtime-adjustable tracking settings
    pub settings: RwLock<TrackingSettings>,
    /// Export settings (CSV and frame export)
    pub export_settings: RwLock<ExportSettings>,
    /// Export statistics
    pub export_csv_records: AtomicU64,
    pub export_frames_written: AtomicU64,
    pub export_last_error: RwLock<Option<String>>,
    /// Broadcast channel for SSE subscribers
    pub sse_tx: broadcast::Sender<TrackingMessage>,
}

impl Default for TrackingSharedState {
    fn default() -> Self {
        // Buffer size 64 allows ~1 second of updates at 60Hz
        // Slow clients will receive Lagged error and can recover
        let (sse_tx, _) = broadcast::channel(64);
        Self {
            enabled: AtomicBool::new(false),
            restart_requested: AtomicBool::new(false),
            status: RwLock::new(TrackingStatus::default()),
            total_updates: AtomicU64::new(0),
            settings: RwLock::new(TrackingSettings::default()),
            export_settings: RwLock::new(ExportSettings::default()),
            export_csv_records: AtomicU64::new(0),
            export_frames_written: AtomicU64::new(0),
            export_last_error: RwLock::new(None),
            sse_tx,
        }
    }
}

/// Configuration for tracking mode (passed to the capture loop).
#[derive(Clone)]
pub struct TrackingConfig {
    pub acquisition_frames: usize,
    pub roi_size: usize,
    pub detection_threshold_sigma: f64,
    pub snr_min: f64,
    pub snr_dropout_threshold: f64,
    pub fwhm: f64,
    pub bad_pixel_map: BadPixelMap,
    pub saturation_value: f64,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            acquisition_frames: 5,
            roi_size: 128,
            detection_threshold_sigma: 5.0,
            snr_min: 10.0,
            snr_dropout_threshold: 3.0,
            fwhm: 7.0,
            bad_pixel_map: BadPixelMap::empty(),
            saturation_value: 65535.0,
        }
    }
}

/// Shared FSM (Fast Steering Mirror) state for web control.
pub struct FsmSharedState {
    /// The S-330 FSM controller (wrapped in Mutex for thread-safe access)
    pub fsm: std::sync::Mutex<S330>,
    /// Current X position in microradians
    pub x_urad: std::sync::atomic::AtomicU64,
    /// Current Y position in microradians
    pub y_urad: std::sync::atomic::AtomicU64,
    /// Travel range for X axis (min, max)
    pub x_range: (f64, f64),
    /// Travel range for Y axis (min, max)
    pub y_range: (f64, f64),
    /// Last error message (if any)
    pub last_error: RwLock<Option<String>>,
}

impl FsmSharedState {
    /// Get current X position as f64
    pub fn get_x(&self) -> f64 {
        f64::from_bits(self.x_urad.load(Ordering::SeqCst))
    }

    /// Get current Y position as f64
    pub fn get_y(&self) -> f64 {
        f64::from_bits(self.y_urad.load(Ordering::SeqCst))
    }

    /// Set current X position
    pub fn set_x(&self, x: f64) {
        self.x_urad.store(x.to_bits(), Ordering::SeqCst);
    }

    /// Set current Y position
    pub fn set_y(&self, y: f64) {
        self.y_urad.store(y.to_bits(), Ordering::SeqCst);
    }
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
    /// Tracking-related shared state (None if tracking not available)
    pub tracking: Option<Arc<TrackingSharedState>>,
    /// Tracking configuration (None if tracking not available)
    pub tracking_config: Option<TrackingConfig>,
    /// FSM control state (None if FSM not configured)
    pub fsm: Option<Arc<FsmSharedState>>,
    /// WebSocket broadcaster for streaming camera frames (with proper close events)
    pub ws_stream: Arc<WsBroadcaster>,
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

    let mut stats = state.stats.lock().await;
    stats.histogram = compute_histogram_u16(&frame);
    stats.histogram_mean = compute_mean_u16(&frame);
    stats.histogram_max = *frame.iter().max().unwrap_or(&0);
    drop(stats);

    let img = u16_to_gray_image(&frame);

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

/// WebSocket image streaming endpoint.
///
/// Provides image streaming with proper connection lifecycle management.
/// Clients receive binary frames with metadata and get clean close events
/// when the stream needs to restart (e.g., frame size change).
///
/// Protocol:
/// - First 4 bytes: width (u32 LE)
/// - Next 4 bytes: height (u32 LE)
/// - Next 8 bytes: frame_number (u64 LE)
/// - Remaining: JPEG data
async fn ws_stream_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
    ws: WebSocketUpgrade,
) -> Response {
    let broadcaster = state.ws_stream.clone();
    ws.on_upgrade(move |socket| crate::ws_stream::ws_stream_handler(socket, broadcaster))
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

    let img = u16_to_gray_image(&frame);

    let mut stats = state.stats.lock().await;
    stats.histogram = compute_histogram_u16(&frame);
    stats.histogram_mean = compute_mean_u16(&frame);
    stats.histogram_max = *frame.iter().max().unwrap_or(&0);
    drop(stats);

    let height = frame.nrows();
    let width = frame.ncols();

    let encoded = base64::engine::general_purpose::STANDARD.encode(img.as_raw());

    let response = RawFrameResponse {
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

    let response = CameraStats {
        total_frames: stats.total_frames,
        avg_fps,
        temperatures: stats.last_temperatures.clone(),
        histogram: stats.histogram.clone(),
        histogram_mean: stats.histogram_mean,
        histogram_max: stats.histogram_max,
        timing: Some(CameraTimingStats {
            avg_capture_ms,
            avg_analysis_ms,
            avg_render_ms,
            avg_total_pipeline_ms,
            capture_samples: stats.capture_timing_ms.len(),
            analysis_samples: stats.analysis_timing_ms.len(),
        }),
        device_name: state.camera_name.clone(),
        width: state.camera_geometry.width() as u32,
        height: state.camera_geometry.height() as u32,
    };

    let json = serde_json::to_string(&response).unwrap();

    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))
        .unwrap()
}

async fn camera_status_page<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    let width = state.camera_geometry.width();
    let height = state.camera_geometry.height();
    let camera_name = &state.camera_name;

    serve_fgs_index_with_data(camera_name, width, height)
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

async fn tracking_status_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    match &state.tracking {
        Some(tracking) => {
            let status = tracking.status.read().await.clone();
            let json = serde_json::to_string(&status).unwrap();
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Tracking not available"))
            .unwrap(),
    }
}

/// SSE endpoint for real-time tracking events
///
/// Streams TrackingMessage updates to multiple subscribers via Server-Sent Events.
/// Each message is sent as a JSON-encoded "tracking" event.
async fn tracking_events_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>, StatusCode> {
    let tracking = match &state.tracking {
        Some(t) => t.clone(),
        None => return Err(StatusCode::NOT_FOUND),
    };

    let mut rx = tracking.sse_tx.subscribe();

    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    if let Ok(json) = serde_json::to_string(&msg) {
                        yield Ok(Event::default().event("tracking").data(json));
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    // Timestamps in messages allow clients to detect gaps
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::default()
            .interval(std::time::Duration::from_secs(15))
            .text("keepalive"),
    ))
}

async fn tracking_enable_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
    Json(req): Json<TrackingEnableRequest>,
) -> Response {
    match &state.tracking {
        Some(tracking) => {
            let was_enabled = tracking.enabled.swap(req.enabled, Ordering::SeqCst);
            if was_enabled != req.enabled {
                tracking.restart_requested.store(true, Ordering::SeqCst);
                tracing::info!(
                    "Tracking {} (restart requested)",
                    if req.enabled { "enabled" } else { "disabled" }
                );
            }

            let mut status = tracking.status.write().await;
            status.enabled = req.enabled;
            if !req.enabled {
                status.state = TrackingState::Idle;
                status.position = None;
                status.num_guide_stars = 0;
            }
            drop(status);

            let status = tracking.status.read().await.clone();
            let json = serde_json::to_string(&status).unwrap();
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Tracking not available"))
            .unwrap(),
    }
}

async fn tracking_settings_get_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    match &state.tracking {
        Some(tracking) => {
            let settings = tracking.settings.read().await.clone();
            let json = serde_json::to_string(&settings).unwrap();
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Tracking not available"))
            .unwrap(),
    }
}

async fn tracking_settings_post_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
    Json(new_settings): Json<TrackingSettings>,
) -> Response {
    match &state.tracking {
        Some(tracking) => {
            // Update settings
            {
                let mut settings = tracking.settings.write().await;
                *settings = new_settings.clone();
            }

            // Request restart to apply new settings
            if tracking.enabled.load(Ordering::SeqCst) {
                tracking.restart_requested.store(true, Ordering::SeqCst);
                tracing::info!("Tracking settings updated, restart requested to apply");
            } else {
                tracing::info!("Tracking settings updated");
            }

            let json = serde_json::to_string(&new_settings).unwrap();
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Tracking not available"))
            .unwrap(),
    }
}

async fn export_status_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    match &state.tracking {
        Some(tracking) => {
            let settings = tracking.export_settings.read().await.clone();
            let last_error = tracking.export_last_error.read().await.clone();
            let status = ExportStatus {
                csv_records_written: tracking.export_csv_records.load(Ordering::SeqCst),
                frames_exported: tracking.export_frames_written.load(Ordering::SeqCst),
                settings,
                last_error,
            };
            let json = serde_json::to_string(&status).unwrap();
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Tracking not available"))
            .unwrap(),
    }
}

async fn export_settings_post_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
    Json(new_settings): Json<ExportSettings>,
) -> Response {
    match &state.tracking {
        Some(tracking) => {
            // Update settings
            {
                let mut settings = tracking.export_settings.write().await;
                *settings = new_settings.clone();
            }

            // If enabling export, signal restart to create new writers
            if (new_settings.csv_enabled || new_settings.frames_enabled)
                && tracking.enabled.load(Ordering::SeqCst)
            {
                tracking.restart_requested.store(true, Ordering::SeqCst);
                tracing::info!("Export settings updated, restart requested to apply");
            }

            tracing::info!(
                "Export settings updated: csv={} ({}), frames={} ({})",
                new_settings.csv_enabled,
                new_settings.csv_filename,
                new_settings.frames_enabled,
                new_settings.frames_directory
            );

            let json = serde_json::to_string(&new_settings).unwrap();
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Tracking not available"))
            .unwrap(),
    }
}

// ==================== FSM Endpoints ====================

async fn fsm_status_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Response {
    match &state.fsm {
        Some(fsm_state) => {
            let last_error = fsm_state.last_error.read().await.clone();
            let status = FsmStatus {
                connected: true,
                x_urad: fsm_state.get_x(),
                y_urad: fsm_state.get_y(),
                x_min: fsm_state.x_range.0,
                x_max: fsm_state.x_range.1,
                y_min: fsm_state.y_range.0,
                y_max: fsm_state.y_range.1,
                last_error,
            };
            let json = serde_json::to_string(&status).unwrap();
            Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("FSM not available"))
            .unwrap(),
    }
}

async fn fsm_move_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
    Json(request): Json<FsmMoveRequest>,
) -> Response {
    match &state.fsm {
        Some(fsm_state) => {
            // Clamp positions to valid range
            let x = request
                .x_urad
                .clamp(fsm_state.x_range.0, fsm_state.x_range.1);
            let y = request
                .y_urad
                .clamp(fsm_state.y_range.0, fsm_state.y_range.1);

            // Move FSM (blocking operation on std::sync::Mutex)
            let result = {
                let mut fsm = fsm_state.fsm.lock().unwrap();
                fsm.move_to(x, y)
            };

            match result {
                Ok(()) => {
                    // Update cached position
                    fsm_state.set_x(x);
                    fsm_state.set_y(y);
                    *fsm_state.last_error.write().await = None;

                    let status = FsmStatus {
                        connected: true,
                        x_urad: x,
                        y_urad: y,
                        x_min: fsm_state.x_range.0,
                        x_max: fsm_state.x_range.1,
                        y_min: fsm_state.y_range.0,
                        y_max: fsm_state.y_range.1,
                        last_error: None,
                    };
                    let json = serde_json::to_string(&status).unwrap();
                    Response::builder()
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(json))
                        .unwrap()
                }
                Err(e) => {
                    let error_msg = format!("FSM move failed: {e}");
                    tracing::error!("{}", error_msg);
                    *fsm_state.last_error.write().await = Some(error_msg.clone());

                    Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::from(error_msg))
                        .unwrap()
                }
            }
        }
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("FSM not available"))
            .unwrap(),
    }
}

pub fn create_router<C: CameraInterface + 'static>(state: Arc<AppState<C>>) -> Router {
    Router::new()
        .route("/", get(camera_status_page::<C>))
        .route("/jpeg", get(jpeg_frame_endpoint::<C>))
        .route("/ws-stream", get(ws_stream_endpoint::<C>))
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
        .route("/tracking/status", get(tracking_status_endpoint::<C>))
        .route("/tracking/events", get(tracking_events_endpoint::<C>))
        .route("/tracking/enable", post(tracking_enable_endpoint::<C>))
        .route(
            "/tracking/settings",
            get(tracking_settings_get_endpoint::<C>).post(tracking_settings_post_endpoint::<C>),
        )
        .route(
            "/tracking/export",
            get(export_status_endpoint::<C>).post(export_settings_post_endpoint::<C>),
        )
        .route("/fsm/status", get(fsm_status_endpoint::<C>))
        .route("/fsm/move", post(fsm_move_endpoint::<C>))
        .fallback(get(serve_fgs_frontend))
        .with_state(state)
        .layer(middleware::from_fn(logging_middleware))
}

pub async fn run_server<C: CameraInterface + Send + 'static>(
    mut camera: C,
    args: CommonServerArgs,
) -> anyhow::Result<()> {
    use tracing::info;

    let exposure = args.exposure.as_duration();
    camera
        .set_exposure(exposure)
        .map_err(|e| anyhow::anyhow!("Failed to set exposure: {e}"))?;
    info!("Set camera exposure to {}ms", args.exposure.exposure_ms);

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

    let ws_stream = Arc::new(WsBroadcaster::new(4));

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
        tracking: None,
        tracking_config: None,
        fsm: None,
        ws_stream,
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
    let mut last_mjpeg_publish = std::time::Instant::now() - MJPEG_MIN_INTERVAL;

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
            *latest = Some((frame_owned.clone(), metadata_owned, capture_timestamp));
        }

        // Publish to WebSocket stream (rate limited, only encode if subscribers)
        if state_clone.ws_stream.subscriber_count() > 0
            && last_mjpeg_publish.elapsed() >= MJPEG_MIN_INTERVAL
        {
            if let Some(jpeg_data) = encode_ndarray_jpeg(&frame_owned, 80) {
                state_clone.ws_stream.publish(WsFrame {
                    jpeg_data,
                    frame_number: frame_num,
                    width: frame_owned.ncols() as u32,
                    height: frame_owned.nrows() as u32,
                });
                last_mjpeg_publish = std::time::Instant::now();
            }
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
            let frame_num = metadata.frame_number;

            let frame_for_render = frame.clone();
            let result = tokio::task::spawn_blocking(move || {
                let start_render = std::time::Instant::now();
                let gray_img = u16_to_gray_image(&frame_for_render);
                let annotated_img = DynamicImage::ImageLuma8(gray_img);
                let render_time = start_render.elapsed();

                Ok::<_, anyhow::Error>((annotated_img, render_time))
            })
            .await;

            match result {
                Ok(Ok((annotated_img, render_time))) => {
                    let total_time = start_total.elapsed();
                    last_analysis_time = start_total;

                    let mut stats = state.stats.lock().await;
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
                        "Pipeline: frame={}, interval={:.1}ms, age={:.1}ms, render={:.1}ms, total={:.1}ms",
                        frame_num,
                        analysis_interval.as_secs_f64() * 1000.0,
                        frame_age.as_secs_f64() * 1000.0,
                        render_time.as_secs_f64() * 1000.0,
                        total_time.as_secs_f64() * 1000.0
                    );
                }
                Ok(Err(e)) => {
                    tracing::error!("Render error in analysis loop: {e}");
                }
                Err(e) => {
                    tracing::error!("Task join error in analysis loop: {e}");
                }
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}

/// Run the camera server with tracking support enabled
pub async fn run_server_with_tracking<C: CameraInterface + Send + 'static>(
    mut camera: C,
    args: CommonServerArgs,
    tracking_config: TrackingConfig,
    fsm: Option<Arc<FsmSharedState>>,
) -> anyhow::Result<()> {
    use tracing::info;

    let exposure = args.exposure.as_duration();
    camera
        .set_exposure(exposure)
        .map_err(|e| anyhow::anyhow!("Failed to set exposure: {e}"))?;
    info!("Set camera exposure to {}ms", args.exposure.exposure_ms);

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

    let tracking_state = Arc::new(TrackingSharedState {
        settings: RwLock::new(TrackingSettings {
            acquisition_frames: tracking_config.acquisition_frames,
            roi_size: tracking_config.roi_size,
            detection_threshold_sigma: tracking_config.detection_threshold_sigma,
            snr_min: tracking_config.snr_min,
            snr_dropout_threshold: tracking_config.snr_dropout_threshold,
            fwhm: tracking_config.fwhm,
        }),
        ..Default::default()
    });

    let ws_stream = Arc::new(WsBroadcaster::new(4));

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
        tracking: Some(tracking_state),
        tracking_config: Some(tracking_config),
        fsm,
        ws_stream,
    });

    info!("Starting background capture loop with tracking support...");
    let capture_state = state.clone();
    std::thread::spawn(move || {
        capture_loop_with_tracking(capture_state);
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
    info!("Tracking status: http://{}/tracking/status", addr);
    info!("Tracking enable: POST http://{}/tracking/enable", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .map_err(|e| anyhow::anyhow!("Server error: {e}"))?;

    Ok(())
}

fn convert_fgs_state_to_shared(state: &FgsState) -> TrackingState {
    match state {
        FgsState::Idle => TrackingState::Idle,
        FgsState::Acquiring { frames_collected } => TrackingState::Acquiring {
            frames_collected: *frames_collected,
        },
        FgsState::Calibrating => TrackingState::Calibrating,
        FgsState::Tracking { frames_processed } => TrackingState::Tracking {
            frames_processed: *frames_processed,
        },
        FgsState::Reacquiring { attempts } => TrackingState::Reacquiring {
            attempts: *attempts,
        },
    }
}

/// Capture loop with integrated tracking support
pub fn capture_loop_with_tracking<C: CameraInterface + Send + 'static>(state: Arc<AppState<C>>) {
    use monocle::CameraSettingsUpdate;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    let tracking_shared = state
        .tracking
        .as_ref()
        .expect("capture_loop_with_tracking requires tracking state");
    let tracking_config = state
        .tracking_config
        .as_ref()
        .expect("capture_loop_with_tracking requires tracking config");

    // Store fixed config values that don't change at runtime
    let bad_pixel_map = tracking_config.bad_pixel_map.clone();
    let saturation_value = tracking_config.saturation_value;

    // Get ROI alignment from camera (immutable hardware property)
    let (roi_h_alignment, roi_v_alignment) = {
        let camera = state.camera.blocking_lock();
        camera.get_roi_offset_alignment()
    };

    let fgs: Arc<StdMutex<Option<FineGuidanceSystem>>> = Arc::new(StdMutex::new(None));
    let pending_settings: Arc<StdMutex<Vec<CameraSettingsUpdate>>> =
        Arc::new(StdMutex::new(Vec::new()));
    let clear_roi_requested: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let mut last_mjpeg_publish = std::time::Instant::now() - MJPEG_MIN_INTERVAL;

    loop {
        let loop_iteration_start = std::time::Instant::now();
        let mut camera = state.camera.blocking_lock();

        let state_clone = state.clone();
        let tracking_for_loop = tracking_shared.clone();
        let fgs_clone = fgs.clone();
        let pending_settings_clone = pending_settings.clone();
        let clear_roi_clone = clear_roi_requested.clone();
        let tracking_shared_for_cb = tracking_shared.clone();
        let bad_pixel_map_clone = bad_pixel_map.clone();

        let stream_result = camera.stream(&mut |frame, metadata| {
            let start_capture = std::time::Instant::now();
            let frame_num = metadata.frame_number;
            let frame_owned = frame.clone();
            let metadata_owned = metadata.clone();

            // Check if restart requested (e.g., tracking toggled)
            if tracking_for_loop.restart_requested.swap(false, Ordering::SeqCst) {
                tracing::info!("Restart requested, stopping stream");
                // Reset FGS if it exists - this clears ROI if one was active
                let mut fgs_guard = fgs_clone.lock().unwrap();
                if let Some(fgs) = fgs_guard.as_mut() {
                    let camera_updates = fgs.reset();
                    for update in camera_updates {
                        if matches!(update, CameraSettingsUpdate::ClearROI) {
                            clear_roi_clone.store(true, Ordering::SeqCst);
                            tracing::info!("FGS reset, returning to full frame");
                        }
                    }
                }
                *fgs_guard = None;
                return false;
            }

            let tracking_enabled = tracking_for_loop.enabled.load(Ordering::SeqCst);

            // Process through FGS if tracking is enabled
            if tracking_enabled {
                let mut fgs_guard = fgs_clone.lock().unwrap();

                // Initialize FGS if needed
                if fgs_guard.is_none() {
                    // Read current settings from shared state
                    let settings = tracking_for_loop.settings.blocking_read().clone();
                    let fgs_config = FgsConfig {
                        acquisition_frames: settings.acquisition_frames,
                        filters: GuideStarFilters {
                            detection_threshold_sigma: settings.detection_threshold_sigma,
                            snr_min: settings.snr_min,
                            diameter_range: (2.0, 100000.0),
                            aspect_ratio_max: 2.5,
                            saturation_value,
                            saturation_search_radius: 3.0,
                            minimum_edge_distance: 40.0,
                            bad_pixel_map: bad_pixel_map_clone.clone(),
                            minimum_bad_pixel_distance: 5.0,
                        },
                        roi_size: settings.roi_size,
                        max_reacquisition_attempts: 5,
                        centroid_radius_multiplier: 3.0,
                        fwhm: settings.fwhm,
                        snr_dropout_threshold: settings.snr_dropout_threshold,
                        noise_estimation_downsample: 16,
                    };
                    tracing::info!(
                        "Initializing Fine Guidance System with settings: acq_frames={}, roi={}, sigma={:.1}, snr_min={:.1}, fwhm={:.1}",
                        settings.acquisition_frames, settings.roi_size, settings.detection_threshold_sigma, settings.snr_min, settings.fwhm
                    );
                    let mut new_fgs = FineGuidanceSystem::new(fgs_config, (roi_h_alignment, roi_v_alignment));

                    // Set up export (CSV and frame writer) if enabled
                    let export_settings = tracking_for_loop.export_settings.blocking_read().clone();

                    // Create CSV writer if enabled
                    let csv_writer: Option<Arc<StdMutex<csv::Writer<std::fs::File>>>> =
                        if export_settings.csv_enabled {
                            match std::fs::File::create(&export_settings.csv_filename) {
                                Ok(file) => {
                                    let mut writer = csv::WriterBuilder::new()
                                        .has_headers(true)
                                        .from_writer(file);
                                    if let Err(e) = writer.write_record(["track_id", "x", "y", "timestamp_sec", "timestamp_nanos", "flux", "diameter"]) {
                                        tracing::error!("Failed to write CSV header: {}", e);
                                        None
                                    } else {
                                        tracing::info!("CSV export enabled: {}", export_settings.csv_filename);
                                        Some(Arc::new(StdMutex::new(writer)))
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Failed to create CSV file {}: {}", export_settings.csv_filename, e);
                                    let mut err = tracking_for_loop.export_last_error.blocking_write();
                                    *err = Some(format!("Failed to create CSV: {e}"));
                                    None
                                }
                            }
                        } else {
                            None
                        };

                    // Create frame writer if enabled
                    let frame_writer: Option<(Arc<FrameWriterHandle>, PathBuf)> =
                        if export_settings.frames_enabled {
                            let export_dir = PathBuf::from(&export_settings.frames_directory);
                            match FrameWriterHandle::new(4, 100) {
                                Ok(writer) => {
                                    tracing::info!("Frame export enabled: {}", export_dir.display());
                                    Some((Arc::new(writer), export_dir))
                                }
                                Err(e) => {
                                    tracing::error!("Failed to create frame writer: {}", e);
                                    let mut err = tracking_for_loop.export_last_error.blocking_write();
                                    *err = Some(format!("Failed to create frame writer: {e}"));
                                    None
                                }
                            }
                        } else {
                            None
                        };

                    // Register callback for tracking updates
                    let tracking_cb = tracking_shared_for_cb.clone();
                    let csv_for_cb = csv_writer.clone();
                    let frame_writer_for_cb = frame_writer.clone();
                    new_fgs.register_callback(move |event| {
                        use monocle::callback::FgsCallbackEvent;
                        match event {
                            FgsCallbackEvent::TrackingStarted {
                                track_id,
                                initial_position,
                                num_guide_stars,
                            } => {
                                tracing::info!(
                                    "🎯 TRACKING LOCKED - track_id: {}, position: ({:.2}, {:.2}), guide stars: {}",
                                    track_id,
                                    initial_position.x,
                                    initial_position.y,
                                    num_guide_stars
                                );
                                let mut status = tracking_cb.status.blocking_write();
                                status.position = Some(TrackingPosition {
                                    track_id: *track_id,
                                    x: initial_position.x,
                                    y: initial_position.y,
                                    snr: initial_position.snr,
                                    timestamp_sec: initial_position.timestamp.seconds,
                                    timestamp_nanos: initial_position.timestamp.nanos,
                                });
                                status.num_guide_stars = *num_guide_stars;

                                // Publish to SSE broadcast (ignore if no subscribers)
                                let msg = TrackingMessage::new(
                                    *track_id,
                                    initial_position.x,
                                    initial_position.y,
                                    initial_position.timestamp,
                                    initial_position.shape.clone(),
                                );
                                let _ = tracking_cb.sse_tx.send(msg);
                            }
                            FgsCallbackEvent::TrackingUpdate { track_id, position } => {
                                tracing::debug!(
                                    "📍 Tracking update - track_id: {}, position: ({:.4}, {:.4}), snr: {:.2}",
                                    track_id,
                                    position.x,
                                    position.y,
                                    position.snr
                                );
                                tracking_cb.total_updates.fetch_add(1, Ordering::SeqCst);
                                let mut status = tracking_cb.status.blocking_write();
                                status.position = Some(TrackingPosition {
                                    track_id: *track_id,
                                    x: position.x,
                                    y: position.y,
                                    snr: position.snr,
                                    timestamp_sec: position.timestamp.seconds,
                                    timestamp_nanos: position.timestamp.nanos,
                                });
                                status.total_updates =
                                    tracking_cb.total_updates.load(Ordering::SeqCst);

                                // Publish to SSE broadcast (ignore if no subscribers)
                                let msg = TrackingMessage::new(
                                    *track_id,
                                    position.x,
                                    position.y,
                                    position.timestamp,
                                    position.shape.clone(),
                                );
                                let _ = tracking_cb.sse_tx.send(msg);

                                // Write to CSV if configured
                                if let Some(ref csv) = csv_for_cb {
                                    if let Ok(mut writer) = csv.lock() {
                                        if let Err(e) = writer.write_record([
                                            track_id.to_string(),
                                            position.x.to_string(),
                                            position.y.to_string(),
                                            position.timestamp.seconds.to_string(),
                                            position.timestamp.nanos.to_string(),
                                            position.shape.flux.to_string(),
                                            position.shape.diameter.to_string(),
                                        ]) {
                                            tracing::warn!("Failed to write CSV record: {}", e);
                                        } else {
                                            let _ = writer.flush();
                                            tracking_cb.export_csv_records.fetch_add(1, Ordering::SeqCst);
                                        }
                                    }
                                }
                            }
                            FgsCallbackEvent::TrackingLost { track_id, reason } => {
                                tracing::warn!(
                                    "⚠️ TRACKING LOST - track_id: {}, reason: {:?}",
                                    track_id,
                                    reason
                                );
                                let mut status = tracking_cb.status.blocking_write();
                                status.position = None;
                                status.num_guide_stars = 0;
                            }
                            FgsCallbackEvent::FrameProcessed {
                                frame_number,
                                timestamp,
                                frame_data,
                                track_id,
                                position,
                            } => {
                                // Export frame as PNG if configured
                                if let Some((ref writer, ref export_dir)) = frame_writer_for_cb {
                                    let png_path = export_dir.join(format!("frame_{frame_number:06}.png"));

                                    // Also write JSON metadata
                                    let metadata = FrameExportMetadata {
                                        frame_number: *frame_number,
                                        timestamp_sec: timestamp.seconds,
                                        timestamp_nanos: timestamp.nanos,
                                        track_id: *track_id,
                                        centroid_x: position.as_ref().map(|p| p.x),
                                        centroid_y: position.as_ref().map(|p| p.y),
                                        width: frame_data.ncols(),
                                        height: frame_data.nrows(),
                                    };

                                    let json_path = export_dir.join(format!("frame_{frame_number:06}.json"));
                                    if let Err(e) = std::fs::create_dir_all(export_dir) {
                                        tracing::warn!("Failed to create export dir: {e}");
                                    } else if let Err(e) = serde_json::to_string(&metadata)
                                        .map_err(std::io::Error::other)
                                        .and_then(|json| std::fs::write(&json_path, json))
                                    {
                                        tracing::warn!("Failed to write metadata JSON: {e}");
                                    }

                                    if writer.write_frame_nonblocking(frame_data.as_ref(), png_path, FrameFormat::Png) {
                                        tracking_cb.export_frames_written.fetch_add(1, Ordering::SeqCst);
                                    } else {
                                        tracing::debug!("Frame export queue full, dropping frame {}", frame_number);
                                    }
                                }
                            }
                            FgsCallbackEvent::FrameSizeMismatch { .. } => {}
                        }
                    });

                    // Start acquisition
                    if let Err(e) = new_fgs.process_event(FgsEvent::StartFgs) {
                        tracing::error!("Failed to start FGS: {e}");
                    }

                    *fgs_guard = Some(new_fgs);
                }

                // Process frame through FGS
                if let Some(ref mut fgs_instance) = *fgs_guard {
                    match fgs_instance.process_frame(frame.view(), metadata.timestamp) {
                        Ok((_update, settings)) => {
                            // Update tracking state
                            {
                                let mut status = tracking_for_loop.status.blocking_write();
                                status.state = convert_fgs_state_to_shared(fgs_instance.state());
                            }

                            // Handle camera settings changes (ROI)
                            if !settings.is_empty() {
                                tracing::info!(
                                    "Camera settings changed ({} updates), stopping stream to apply",
                                    settings.len()
                                );
                                *pending_settings_clone.lock().unwrap() = settings;
                                return false; // Restart stream
                            }

                            // Check if FGS went idle (reacquisition failed)
                            if matches!(fgs_instance.state(), FgsState::Idle) {
                                tracing::info!("FGS went idle, restarting acquisition");
                                if let Err(e) = fgs_instance.process_event(FgsEvent::StartFgs) {
                                    tracing::error!("Failed to restart FGS: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Frame processing error: {e}");
                        }
                    }
                }
            } else {
                // Tracking disabled - reset FGS and return to full-frame capture
                let mut fgs_guard = fgs_clone.lock().unwrap();
                if let Some(fgs) = fgs_guard.as_mut() {
                    let camera_updates = fgs.reset();
                    for update in camera_updates {
                        if matches!(update, CameraSettingsUpdate::ClearROI) {
                            clear_roi_clone.store(true, Ordering::SeqCst);
                        }
                    }
                    *fgs_guard = None;
                    tracing::info!("Tracking disabled, FGS reset, returning to full frame");
                    return false; // Restart stream to apply ROI clear
                }
            }

            // Update stats and latest frame (same as regular capture loop)
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
                *latest = Some((frame_owned.clone(), metadata_owned, capture_timestamp));
            }

            // Publish to WebSocket stream (rate limited, only encode if subscribers)
            if state_clone.ws_stream.subscriber_count() > 0
                && last_mjpeg_publish.elapsed() >= MJPEG_MIN_INTERVAL
            {
                if let Some(jpeg_data) = encode_ndarray_jpeg(&frame_owned, 80) {
                    state_clone.ws_stream.publish(WsFrame {
                        jpeg_data,
                        frame_number: frame_num,
                        width: frame_owned.ncols() as u32,
                        height: frame_owned.nrows() as u32,
                    });
                    last_mjpeg_publish = std::time::Instant::now();
                }
            }

            true
        });

        // Apply any pending camera settings after stream ends
        let settings = {
            let mut guard = pending_settings.lock().unwrap();
            std::mem::take(&mut *guard)
        };
        if !settings.is_empty() {
            let apply_start = std::time::Instant::now();
            tracing::info!("Applying {} camera settings...", settings.len());
            if let Err(errors) = apply_camera_settings(&mut *camera, settings) {
                for err in &errors {
                    tracing::error!("Camera settings error: {}", err);
                }
                // ROI failed - clear FGS to restart from scratch with full-frame
                tracing::warn!("Camera settings failed, clearing FGS to restart acquisition");
                let mut fgs_guard = fgs.lock().unwrap();
                *fgs_guard = None;
                // Request ROI clear to go back to full frame
                clear_roi_requested.store(true, Ordering::SeqCst);
            }
            tracing::info!(
                "apply_camera_settings took {:.1}ms",
                apply_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Clear ROI if requested
        if clear_roi_requested.swap(false, Ordering::SeqCst) {
            let clear_start = std::time::Instant::now();
            tracing::info!("Clearing camera ROI...");
            if let Err(e) = camera.clear_roi() {
                tracing::warn!("Failed to clear ROI: {e}");
            }
            tracing::info!(
                "clear_roi took {:.1}ms",
                clear_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        drop(camera); // Release lock before sleeping on error

        match stream_result {
            Ok(_) => {
                tracing::info!(
                    "Stream ended, restarting... (loop iteration took {:.1}ms)",
                    loop_iteration_start.elapsed().as_secs_f64() * 1000.0
                );
            }
            Err(e) => {
                tracing::error!("Camera stream error: {e}, retrying in 1s...");
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }
}

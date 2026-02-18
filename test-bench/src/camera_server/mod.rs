mod capture;
pub use capture::{analysis_loop, capture_loop_blocking, capture_loop_with_tracking, run_server};

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
use ndarray::{s, Array2};
use serde::Deserialize;
use shared::bad_pixel_map::BadPixelMap;
use shared::camera_interface::{CameraInterface, FrameMetadata, SensorGeometry};
use shared::image_proc::u16_to_gray_image;
use shared::tracking_message::TrackingMessage;
use shared_wasm::{
    CameraStats, CameraTimingStats, CommandError, FgsWsCommand, FgsWsMessage, FsmConnectRequest,
    FsmMoveRequest, FsmStatus, RawFrameResponse, StarDetectionSettings, TrackingEnableRequest,
    TrackingSettings, TrackingState, TrackingStatus,
};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, Mutex, Notify, RwLock};

use crate::camera_init::ExposureArgs;
use crate::embedded_assets::{serve_fgs_frontend, serve_fgs_index_with_data};
use crate::ws_log_stream::{ws_log_handler, LogBroadcaster};
use crate::ws_stream::{WsBroadcaster, WsFrame};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use clap::Args;
use futures::{SinkExt, StreamExt};

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

    #[arg(
        long,
        default_value = "4",
        help = "Number of JPEG encoding worker threads",
        long_help = "Number of threads dedicated to encoding camera frames as JPEG for \
            WebSocket streaming. Workers pull from a shared queue, so more workers reduce \
            encoding latency when multiple clients are connected. Typical range: 1-8."
    )]
    pub encoding_threads: usize,
}

pub(crate) type TimestampedFrame = (Array2<u16>, FrameMetadata, Instant);

/// Frame data shared between capture callback, FGS thread, and encoding workers via Arc.
pub(crate) type SharedFrame = Arc<(Array2<u16>, FrameMetadata)>;

#[derive(Debug, Clone)]
pub struct ZoomRegion {
    pub patch: Array2<u16>,
    pub center: (usize, usize),
    pub frame_number: u64,
    pub timestamp: Instant,
}

/// Query parameters for the `/ws/frames` endpoint.
///
/// When `zoom` is provided as "x,y", returns zoomed frames centered at that position.
/// When omitted, returns the main camera stream.
#[derive(Debug, Default, Deserialize)]
pub struct FramesQueryParams {
    /// Zoom center as "x,y" coordinates (e.g., "320,240")
    #[serde(default, deserialize_with = "deserialize_zoom_coords")]
    pub zoom: Option<(usize, usize)>,
}

fn deserialize_zoom_coords<'de, D>(deserializer: D) -> Result<Option<(usize, usize)>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(deserializer)?;
    match opt {
        None => Ok(None),
        Some(s) => {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() != 2 {
                return Err(serde::de::Error::custom("zoom must be 'x,y'"));
            }
            let x = parts[0]
                .parse()
                .map_err(|_| serde::de::Error::custom("invalid x coordinate"))?;
            let y = parts[1]
                .parse()
                .map_err(|_| serde::de::Error::custom("invalid y coordinate"))?;
            Ok(Some((x, y)))
        }
    }
}

/// Query parameters for the zoom HTTP endpoints (`POST /zoom`, `GET /zoom`).
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
pub(crate) const MJPEG_MIN_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);

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

/// Broadcaster for FGS status updates to WebSocket clients.
///
/// Follows the same pattern as `LogBroadcaster`. Publishes `FgsWsMessage`
/// variants when state changes, replacing HTTP polling.
pub struct FgsStatusBroadcaster {
    tx: broadcast::Sender<FgsWsMessage>,
}

impl FgsStatusBroadcaster {
    /// Create a new status broadcaster with the given buffer capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Publish a status update to all connected WebSocket clients.
    pub fn publish(&self, msg: FgsWsMessage) {
        let _ = self.tx.send(msg);
    }

    /// Subscribe to the status stream.
    pub fn subscribe(&self) -> broadcast::Receiver<FgsWsMessage> {
        self.tx.subscribe()
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
    /// FSM control state (None if not connected, swappable at runtime)
    pub fsm: Arc<RwLock<Option<Arc<FsmSharedState>>>>,
    /// FSM controller IP address for runtime connect/disconnect
    pub fsm_ip: String,
    /// WebSocket broadcaster for streaming camera frames (with proper close events)
    pub ws_stream: Arc<WsBroadcaster>,
    /// WebSocket broadcaster for streaming zoom region frames
    pub ws_zoom: Arc<WsBroadcaster>,
    /// Current zoom center coordinates, set by WebSocket clients
    pub ws_zoom_center: Arc<std::sync::RwLock<Option<(usize, usize)>>>,
    /// WebSocket broadcaster for log streaming
    pub log_broadcaster: Arc<LogBroadcaster>,
    /// WebSocket broadcaster for FGS status updates
    pub fgs_status: Arc<FgsStatusBroadcaster>,
    /// Star detection overlay settings
    pub star_detection_settings: Arc<RwLock<StarDetectionSettings>>,
    /// Number of JPEG encoding worker threads
    pub encoding_threads: usize,
}

#[derive(Debug, Clone)]
pub struct SlidingWindow {
    samples: VecDeque<f32>,
    capacity: usize,
}

impl SlidingWindow {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, value: f32) {
        if self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(value);
    }

    pub fn average(&self) -> f32 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.samples.iter().sum::<f32>() / self.samples.len() as f32
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

#[derive(Debug, Clone)]
pub struct FrameStats {
    pub total_frames: u64,
    pub fps_samples: SlidingWindow,
    pub last_frame_time: Instant,
    pub last_temperatures: std::collections::HashMap<String, f64>,
    pub histogram: Vec<u32>,
    pub histogram_mean: f64,
    pub histogram_max: u16,
    pub capture_timing_ms: SlidingWindow,
    pub analysis_timing_ms: SlidingWindow,
    pub render_timing_ms: SlidingWindow,
    pub total_pipeline_ms: SlidingWindow,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            total_frames: 0,
            fps_samples: SlidingWindow::new(10),
            last_frame_time: Instant::now(),
            last_temperatures: std::collections::HashMap::new(),
            histogram: Vec::new(),
            histogram_mean: 0.0,
            histogram_max: 0,
            capture_timing_ms: SlidingWindow::new(100),
            analysis_timing_ms: SlidingWindow::new(100),
            render_timing_ms: SlidingWindow::new(100),
            total_pipeline_ms: SlidingWindow::new(100),
        }
    }
}

pub(crate) fn extract_patch(
    frame: &Array2<u16>,
    center_x: usize,
    center_y: usize,
    patch_size: usize,
) -> Array2<u16> {
    let half_size = patch_size / 2;
    let frame_height = frame.nrows();
    let frame_width = frame.ncols();

    let x_start = center_x
        .saturating_sub(half_size)
        .min(frame_width.saturating_sub(1));
    let y_start = center_y
        .saturating_sub(half_size)
        .min(frame_height.saturating_sub(1));
    let x_end = (x_start + patch_size).min(frame_width);
    let y_end = (y_start + patch_size).min(frame_height);

    let actual_width = x_end.saturating_sub(x_start);
    let actual_height = y_end.saturating_sub(y_start);

    let mut patch = Array2::zeros((patch_size, patch_size));

    if actual_width > 0 && actual_height > 0 {
        let frame_slice = frame.slice(s![y_start..y_end, x_start..x_end]);
        patch
            .slice_mut(s![0..actual_height, 0..actual_width])
            .assign(&frame_slice);
    }

    patch
}

/// Encode a zoom patch as a JPEG WsFrame, scaled 4x with normalized intensity.
pub(crate) fn encode_zoom_frame(
    frame: &Array2<u16>,
    center_x: usize,
    center_y: usize,
    frame_number: u64,
) -> Option<WsFrame> {
    let patch_size = 128;
    let patch = extract_patch(frame, center_x, center_y, patch_size);
    let scale_factor = 4;
    let scaled_size = patch_size * scale_factor;

    let max_val = *patch.iter().max().unwrap_or(&1) as f32;
    let scale = if max_val > 0.0 { 255.0 / max_val } else { 1.0 };

    let mut scaled_patch = Vec::with_capacity(scaled_size * scaled_size);
    for y in 0..scaled_size {
        let src_y = y / scale_factor;
        for x in 0..scaled_size {
            let src_x = x / scale_factor;
            if src_y < patch.nrows() && src_x < patch.ncols() {
                scaled_patch.push(((patch[[src_y, src_x]] as f32) * scale) as u8);
            } else {
                scaled_patch.push(0);
            }
        }
    }

    let img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
        scaled_size as u32,
        scaled_size as u32,
        scaled_patch,
    )?;

    let mut jpeg_bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut jpeg_bytes),
        image::ImageFormat::Jpeg,
    )
    .ok()?;

    Some(WsFrame {
        jpeg_data: bytes::Bytes::from(jpeg_bytes),
        frame_number,
        width: scaled_size as u32,
        height: scaled_size as u32,
    })
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

/// WebSocket endpoint for streaming camera frames as binary data.
///
/// When called without query params, streams the main camera feed.
/// When called with `?zoom=320,240`, streams a zoomed crop centered at those coordinates.
///
/// Protocol:
/// - First 4 bytes: width (u32 LE)
/// - Next 4 bytes: height (u32 LE)
/// - Next 8 bytes: frame_number (u64 LE)
/// - Remaining: JPEG data
async fn ws_frames_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
    axum::extract::Query(params): axum::extract::Query<FramesQueryParams>,
    ws: WebSocketUpgrade,
) -> Response {
    match params.zoom {
        Some((x, y)) => {
            // Zoom mode: stream cropped frames
            let broadcaster = state.ws_zoom.clone();
            let zoom_center = state.ws_zoom_center.clone();

            {
                let mut center = zoom_center.write().unwrap();
                *center = Some((x, y));
            }

            ws.on_upgrade(move |socket| async move {
                crate::ws_stream::ws_stream_handler(socket, broadcaster.clone()).await;

                if broadcaster.subscriber_count() == 0 {
                    let mut center = zoom_center.write().unwrap();
                    *center = None;
                }
                tracing::debug!("Zoom WebSocket connection closed");
            })
        }
        None => {
            // Main stream mode
            let broadcaster = state.ws_stream.clone();
            ws.on_upgrade(move |socket| crate::ws_stream::ws_stream_handler(socket, broadcaster))
        }
    }
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
    Json(coords): Json<ZoomQueryParams>,
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
                timestamp: Instant::now(),
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
    let start = Instant::now();
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

// ==================== WebSocket Command Handlers ====================

/// Build a `CommandError` from a command name and a displayable error.
fn cmd_err(command: &str, err: impl std::fmt::Display) -> CommandError {
    CommandError {
        command: command.into(),
        message: err.to_string(),
    }
}

async fn handle_set_tracking_enabled<C: CameraInterface + 'static>(
    state: &AppState<C>,
    req: TrackingEnableRequest,
) -> Result<(), CommandError> {
    let tracking = state
        .tracking
        .as_ref()
        .ok_or_else(|| cmd_err("SetTrackingEnabled", "Tracking not available"))?;

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
    state
        .fgs_status
        .publish(FgsWsMessage::TrackingStatus(status));
    Ok(())
}

async fn handle_set_tracking_settings<C: CameraInterface + 'static>(
    state: &AppState<C>,
    new_settings: TrackingSettings,
) -> Result<(), CommandError> {
    let tracking = state
        .tracking
        .as_ref()
        .ok_or_else(|| cmd_err("SetTrackingSettings", "Tracking not available"))?;

    {
        let mut settings = tracking.settings.write().await;
        *settings = new_settings.clone();
    }

    if tracking.enabled.load(Ordering::SeqCst) {
        tracking.restart_requested.store(true, Ordering::SeqCst);
        tracing::info!("Tracking settings updated, restart requested to apply");
    } else {
        tracing::info!("Tracking settings updated");
    }

    state
        .fgs_status
        .publish(FgsWsMessage::TrackingSettings(new_settings));
    Ok(())
}

async fn handle_set_detection_settings<C: CameraInterface + 'static>(
    state: &AppState<C>,
    new_settings: StarDetectionSettings,
) -> Result<(), CommandError> {
    {
        let mut settings = state.star_detection_settings.write().await;
        *settings = new_settings.clone();
    }
    state
        .fgs_status
        .publish(FgsWsMessage::DetectionSettings(new_settings));
    Ok(())
}

async fn handle_move_fsm<C: CameraInterface + 'static>(
    state: &AppState<C>,
    request: FsmMoveRequest,
) -> Result<(), CommandError> {
    let fsm_guard = state.fsm.read().await;
    let fsm_state = fsm_guard
        .as_ref()
        .ok_or_else(|| cmd_err("MoveFsm", "FSM not connected"))?;

    let x = request
        .x_urad
        .clamp(fsm_state.x_range.0, fsm_state.x_range.1);
    let y = request
        .y_urad
        .clamp(fsm_state.y_range.0, fsm_state.y_range.1);

    let fsm_arc = Arc::clone(fsm_state);
    drop(fsm_guard);

    let result = tokio::task::spawn_blocking(move || {
        let mut fsm = fsm_arc.fsm.lock().unwrap();
        fsm.move_to(x, y)
    })
    .await
    .map_err(|e| cmd_err("MoveFsm", format!("FSM task failed: {e}")))?;

    let fsm_guard = state.fsm.read().await;
    let fsm_state = fsm_guard
        .as_ref()
        .ok_or_else(|| cmd_err("MoveFsm", "FSM disconnected during move"))?;

    match result {
        Ok(()) => {
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
            state.fgs_status.publish(FgsWsMessage::FsmStatus(status));
            Ok(())
        }
        Err(e) => {
            let error_msg = format!("FSM move failed: {e}");
            tracing::error!("{}", error_msg);
            *fsm_state.last_error.write().await = Some(error_msg.clone());
            Err(cmd_err("MoveFsm", &error_msg))
        }
    }
}

async fn handle_set_fsm_connected<C: CameraInterface + 'static>(
    state: &AppState<C>,
    request: FsmConnectRequest,
) -> Result<(), CommandError> {
    if request.connected {
        // Connect to FSM
        let ip = state.fsm_ip.clone();
        tracing::info!("Connecting to FSM at {ip}...");

        let result = tokio::task::spawn_blocking(move || S330::connect_ip(&ip))
            .await
            .map_err(|e| cmd_err("SetFsmConnected", format!("FSM connect task failed: {e}")))?;

        let mut fsm = result.map_err(|e| {
            tracing::error!("FSM connection failed: {e}");
            cmd_err("SetFsmConnected", format!("FSM connection failed: {e}"))
        })?;

        let (x_range, y_range) = fsm.get_travel_ranges().map_err(|e| {
            tracing::error!("Failed to get FSM travel ranges: {e}");
            cmd_err(
                "SetFsmConnected",
                format!("Failed to get travel ranges: {e}"),
            )
        })?;

        let (x, y) = fsm.get_position().map_err(|e| {
            tracing::error!("Failed to get FSM position: {e}");
            cmd_err("SetFsmConnected", format!("Failed to get position: {e}"))
        })?;

        let fsm_state = Arc::new(FsmSharedState {
            fsm: std::sync::Mutex::new(fsm),
            x_urad: AtomicU64::new(x.to_bits()),
            y_urad: AtomicU64::new(y.to_bits()),
            x_range,
            y_range,
            last_error: RwLock::new(None),
        });

        *state.fsm.write().await = Some(fsm_state);

        let status = FsmStatus {
            connected: true,
            x_urad: x,
            y_urad: y,
            x_min: x_range.0,
            x_max: x_range.1,
            y_min: y_range.0,
            y_max: y_range.1,
            last_error: None,
        };
        state.fgs_status.publish(FgsWsMessage::FsmStatus(status));
        tracing::info!("FSM connected at ({x:.1}, {y:.1}) µrad");
        Ok(())
    } else {
        // Disconnect FSM
        tracing::info!("Disconnecting FSM...");
        let old = state.fsm.write().await.take();
        drop(old);

        state
            .fgs_status
            .publish(FgsWsMessage::FsmStatus(FsmStatus::default()));
        tracing::info!("FSM disconnected");
        Ok(())
    }
}

/// Dispatch a WebSocket command to the appropriate handler.
async fn dispatch_command<C: CameraInterface + 'static>(
    state: &AppState<C>,
    cmd: FgsWsCommand,
) -> Result<(), CommandError> {
    match cmd {
        FgsWsCommand::SetTrackingEnabled(req) => handle_set_tracking_enabled(state, req).await,
        FgsWsCommand::SetTrackingSettings(s) => handle_set_tracking_settings(state, s).await,
        FgsWsCommand::SetDetectionSettings(s) => handle_set_detection_settings(state, s).await,
        FgsWsCommand::MoveFsm(req) => handle_move_fsm(state, req).await,
        FgsWsCommand::SetFsmConnected(req) => handle_set_fsm_connected(state, req).await,
    }
}

/// WebSocket endpoint for log streaming.
///
/// Streams log entries to connected clients as JSON messages.
async fn ws_log_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
    axum::extract::Query(params): axum::extract::Query<crate::ws_log_stream::LogStreamParams>,
    ws: WebSocketUpgrade,
) -> Response {
    let broadcaster = state.log_broadcaster.clone();
    ws.on_upgrade(move |socket| ws_log_handler(socket, broadcaster, params.level))
}

/// Bidirectional WebSocket handler for FGS status streaming and commands.
///
/// Sends a snapshot of current state on connect, then streams `FgsWsMessage`
/// updates while accepting `FgsWsCommand` messages from the client.
async fn ws_status_handler<C: CameraInterface + 'static>(ws: WebSocket, state: Arc<AppState<C>>) {
    let (mut sender, mut receiver) = ws.split();
    let mut rx = state.fgs_status.subscribe();

    // Send initial state snapshot so client has data immediately
    let snapshot = collect_status_snapshot(&state).await;
    for msg in snapshot {
        if let Ok(json) = serde_json::to_string(&msg) {
            if sender.send(Message::Text(json)).await.is_err() {
                return;
            }
        }
    }

    let mut last_lag_log = tokio::time::Instant::now() - tokio::time::Duration::from_secs(2);

    loop {
        tokio::select! {
            // Broadcast status updates to this client
            result = rx.recv() => {
                match result {
                    Ok(msg) => {
                        if let Ok(json) = serde_json::to_string(&msg) {
                            if sender.send(Message::Text(json)).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        // Rate-limited: sending a WS message here would make
                        // slow clients fall further behind, creating a
                        // feedback loop that floods both the socket and logs.
                        if last_lag_log.elapsed() >= tokio::time::Duration::from_secs(1) {
                            tracing::debug!("WebSocket client lagged, skipped {n} messages");
                            last_lag_log = tokio::time::Instant::now();
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            // Process incoming commands from this client
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<FgsWsCommand>(&text) {
                            Ok(cmd) => {
                                if let Err(err) = dispatch_command(&state, cmd).await {
                                    if let Ok(json) = serde_json::to_string(
                                        &FgsWsMessage::CommandError(err),
                                    ) {
                                        let _ = sender.send(Message::Text(json)).await;
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::debug!("Invalid WS command: {e}");
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    Some(Ok(_)) => {}
                }
            }
        }
    }
}

/// Collect a snapshot of all current state for new WebSocket clients.
async fn collect_status_snapshot<C: CameraInterface + 'static>(
    state: &AppState<C>,
) -> Vec<FgsWsMessage> {
    let mut msgs = Vec::new();

    // Camera stats
    let stats = state.stats.lock().await;
    msgs.push(FgsWsMessage::CameraStats(CameraStats {
        total_frames: stats.total_frames,
        avg_fps: stats.fps_samples.average(),
        temperatures: stats.last_temperatures.clone(),
        histogram: stats.histogram.clone(),
        histogram_mean: stats.histogram_mean,
        histogram_max: stats.histogram_max,
        timing: Some(CameraTimingStats {
            avg_capture_ms: stats.capture_timing_ms.average(),
            avg_analysis_ms: stats.analysis_timing_ms.average(),
            avg_render_ms: stats.render_timing_ms.average(),
            avg_total_pipeline_ms: stats.total_pipeline_ms.average(),
            capture_samples: stats.capture_timing_ms.len(),
            analysis_samples: stats.analysis_timing_ms.len(),
        }),
        device_name: state.camera_name.clone(),
        width: state.camera_geometry.width() as u32,
        height: state.camera_geometry.height() as u32,
    }));
    drop(stats);

    // Tracking status + settings
    if let Some(ref tracking) = state.tracking {
        let status = tracking.status.read().await;
        msgs.push(FgsWsMessage::TrackingStatus(status.clone()));
        drop(status);

        let settings = tracking.settings.read().await;
        msgs.push(FgsWsMessage::TrackingSettings(settings.clone()));
        drop(settings);
    }

    // FSM status (always send, even when disconnected)
    let fsm_guard = state.fsm.read().await;
    if let Some(ref fsm) = *fsm_guard {
        msgs.push(FgsWsMessage::FsmStatus(FsmStatus {
            connected: true,
            x_urad: f64::from_bits(fsm.x_urad.load(Ordering::SeqCst)),
            y_urad: f64::from_bits(fsm.y_urad.load(Ordering::SeqCst)),
            x_min: fsm.x_range.0,
            x_max: fsm.x_range.1,
            y_min: fsm.y_range.0,
            y_max: fsm.y_range.1,
            last_error: fsm.last_error.read().await.clone(),
        }));
    } else {
        msgs.push(FgsWsMessage::FsmStatus(FsmStatus::default()));
    }
    drop(fsm_guard);

    // Star detection settings
    let detection = state.star_detection_settings.read().await;
    msgs.push(FgsWsMessage::DetectionSettings(detection.clone()));

    msgs
}

async fn ws_status_endpoint<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| ws_status_handler(socket, state))
}

pub fn create_router<C: CameraInterface + 'static>(state: Arc<AppState<C>>) -> Router {
    Router::new()
        .route("/", get(camera_status_page::<C>))
        .route("/jpeg", get(jpeg_frame_endpoint::<C>))
        .route("/ws/frames", get(ws_frames_endpoint::<C>))
        .route("/raw", get(raw_frame_endpoint::<C>))
        .route("/fits", get(fits_frame_endpoint::<C>))
        .route("/annotated", get(annotated_frame_endpoint::<C>))
        .route("/annotated_raw", get(annotated_raw_endpoint::<C>))
        .route("/overlay-svg", get(overlay_svg_endpoint::<C>))
        .route(
            "/zoom",
            post(set_zoom_endpoint::<C>).get(get_zoom_endpoint::<C>),
        )
        .route("/tracking/events", get(tracking_events_endpoint::<C>))
        .route("/ws/status", get(ws_status_endpoint::<C>))
        .route("/logs", get(ws_log_endpoint::<C>))
        .fallback(get(serve_fgs_frontend))
        .with_state(state)
        .layer(middleware::from_fn(logging_middleware))
}

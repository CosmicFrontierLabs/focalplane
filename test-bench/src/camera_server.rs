use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{Html, Response},
    routing::get,
    Router,
};
use base64::Engine;
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::Array2;
use rust_embed::RustEmbed;
use shared::camera_interface::{CameraInterface, FrameMetadata};
use std::sync::Arc;
use tokio::sync::{Mutex, Notify, RwLock};

use crate::calibration_overlay::{analyze_calibration_pattern, render_annotated_image};

#[derive(RustEmbed)]
#[folder = "templates/"]
struct Templates;

type TimestampedFrame = (Array2<u16>, FrameMetadata, std::time::Instant);

pub struct AppState<C: CameraInterface> {
    pub camera: Arc<Mutex<C>>,
    pub stats: Arc<Mutex<FrameStats>>,
    pub latest_frame: Arc<RwLock<Option<TimestampedFrame>>>,
    pub latest_annotated: Arc<RwLock<Option<(DynamicImage, u64)>>>,
    pub annotated_notify: Arc<Notify>,
    pub bit_depth: u8,
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

async fn capture_frame_data<C: CameraInterface>(
    state: &AppState<C>,
) -> Result<(Array2<u16>, FrameMetadata), String> {
    let mut camera = state.camera.lock().await;
    let result = camera
        .capture_frame()
        .map_err(|e| format!("Failed to capture frame: {e}"));

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
        stats.last_temperatures = metadata.temperatures.clone();
    }

    result.map_err(|e| e.to_string())
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
    let (frame, _metadata) = match capture_frame_data(&state).await {
        Ok(data) => data,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(format!("Failed to capture frame: {e}")))
                .unwrap()
        }
    };

    let pixels_8bit = process_raw_to_pixels(&frame);

    let mut stats = state.stats.lock().await;
    stats.histogram = compute_histogram_u16(&frame);
    stats.histogram_mean = compute_mean_u16(&frame);
    stats.histogram_max = *frame.iter().max().unwrap_or(&0);
    drop(stats);

    let camera = state.camera.lock().await;
    let config = camera.get_config();
    let width = config.width as u32;
    let height = config.height as u32;
    drop(camera);

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
    let (frame, metadata) = match capture_frame_data(&state).await {
        Ok(data) => data,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(format!("Failed to capture frame: {e}")))
                .unwrap()
        }
    };

    let pixels_8bit = process_raw_to_pixels(&frame);

    let mut stats = state.stats.lock().await;
    stats.histogram = compute_histogram_u16(&frame);
    stats.histogram_mean = compute_mean_u16(&frame);
    stats.histogram_max = *frame.iter().max().unwrap_or(&0);
    drop(stats);

    let camera = state.camera.lock().await;
    let config = camera.get_config();
    let width = config.width;
    let height = config.height;
    drop(camera);

    let encoded = base64::engine::general_purpose::STANDARD.encode(&pixels_8bit);

    let json = serde_json::json!({
        "width": width,
        "height": height,
        "timestamp_sec": metadata.timestamp.seconds,
        "timestamp_nanos": metadata.timestamp.nanos,
        "temperatures": metadata.temperatures,
        "exposure_us": metadata.exposure.as_micros(),
        "frame_number": metadata.frame_number,
        "image_base64": encoded,
    });

    let json = serde_json::to_string(&json).unwrap();

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

    let json = serde_json::json!({
        "total_frames": stats.total_frames,
        "avg_fps": avg_fps,
        "temperatures": stats.last_temperatures,
        "histogram": stats.histogram,
        "histogram_mean": stats.histogram_mean,
        "histogram_max": stats.histogram_max,
        "timing": {
            "avg_capture_ms": avg_capture_ms,
            "avg_analysis_ms": avg_analysis_ms,
            "avg_render_ms": avg_render_ms,
            "avg_total_pipeline_ms": avg_total_pipeline_ms,
            "capture_samples": stats.capture_timing_ms.len(),
            "analysis_samples": stats.analysis_timing_ms.len(),
        }
    });

    let json = serde_json::to_string(&json).unwrap();

    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))
        .unwrap()
}

async fn camera_status_page<C: CameraInterface + 'static>(
    State(state): State<Arc<AppState<C>>>,
) -> Html<String> {
    let template_content = Templates::get("camera_status.html")
        .map(|file| String::from_utf8_lossy(&file.data).to_string())
        .unwrap_or_else(|| "<html><body>Template not found</body></html>".to_string());

    let camera = state.camera.lock().await;
    let config = camera.get_config();
    let width = config.width;
    let height = config.height;
    let camera_name = camera.name().to_string();
    drop(camera);

    let html = template_content
        .replace("{device}", &camera_name)
        .replace("{width}", &width.to_string())
        .replace("{height}", &height.to_string())
        .replace("{resolutions_list}", "")
        .replace("{test_patterns}", "");

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

pub fn create_router<C: CameraInterface + 'static>(state: Arc<AppState<C>>) -> Router {
    Router::new()
        .route("/", get(camera_status_page::<C>))
        .route("/jpeg", get(jpeg_frame_endpoint::<C>))
        .route("/raw", get(raw_frame_endpoint::<C>))
        .route("/annotated", get(annotated_frame_endpoint::<C>))
        .route("/annotated_raw", get(annotated_raw_endpoint::<C>))
        .route("/stats", get(stats_endpoint::<C>))
        .with_state(state)
}

pub async fn capture_loop<C: CameraInterface + Send + 'static>(state: Arc<AppState<C>>) {
    loop {
        let start_capture = std::time::Instant::now();

        let mut camera = state.camera.lock().await;
        let result = camera.capture_frame();
        drop(camera);

        let capture_time = start_capture.elapsed();

        match result {
            Ok((frame, metadata)) => {
                let frame_num = metadata.frame_number;

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

                stats
                    .capture_timing_ms
                    .push(capture_time.as_secs_f32() * 1000.0);
                if stats.capture_timing_ms.len() > 100 {
                    stats.capture_timing_ms.remove(0);
                }

                stats.total_frames += 1;
                stats.last_frame_time = now;
                stats.last_temperatures = metadata.temperatures.clone();
                drop(stats);

                let capture_timestamp = std::time::Instant::now();
                let mut latest = state.latest_frame.write().await;
                *latest = Some((frame, metadata, capture_timestamp));

                tracing::info!(
                    "Capture: {:.1}ms, frame_num={}",
                    capture_time.as_secs_f64() * 1000.0,
                    frame_num
                );
            }
            Err(e) => {
                tracing::error!("Capture loop error: {e}");
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
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
            let result = tokio::task::spawn_blocking(move || {
                let start_analysis = std::time::Instant::now();
                let analysis = analyze_calibration_pattern(&frame, bit_depth)?;
                let analysis_time = start_analysis.elapsed();

                let start_render = std::time::Instant::now();
                let annotated_img = render_annotated_image(&frame, &analysis)?;
                let render_time = start_render.elapsed();

                Ok::<_, anyhow::Error>((annotated_img, analysis_time, render_time))
            })
            .await;

            match result {
                Ok(Ok((annotated_img, analysis_time, render_time))) => {
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

                    state.annotated_notify.notify_waiters();

                    tracing::info!(
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

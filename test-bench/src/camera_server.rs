use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{Html, Response},
    routing::get,
    Router,
};
use base64::Engine;
use image::{ImageBuffer, Luma};
use ndarray::Array2;
use rust_embed::RustEmbed;
use shared::camera_interface::{CameraInterface, FrameMetadata};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(RustEmbed)]
#[folder = "templates/"]
struct Templates;

pub struct AppState<C: CameraInterface> {
    pub camera: Arc<Mutex<C>>,
    pub stats: Arc<Mutex<FrameStats>>,
}

#[derive(Debug, Clone)]
pub struct FrameStats {
    pub total_frames: u64,
    pub fps_samples: Vec<f32>,
    #[allow(dead_code)]
    pub last_frame_time: std::time::Instant,
    pub last_temperatures: std::collections::HashMap<String, f64>,
    pub histogram: Vec<u32>,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            total_frames: 0,
            fps_samples: Vec::new(),
            last_frame_time: std::time::Instant::now(),
            last_temperatures: std::collections::HashMap::new(),
            histogram: Vec::new(),
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

fn compute_histogram(pixels: &[u8]) -> Vec<u32> {
    let mut histogram = vec![0u32; 256];
    for &pixel in pixels {
        histogram[pixel as usize] += 1;
    }
    histogram
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
    stats.histogram = compute_histogram(&pixels_8bit);
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
    stats.histogram = compute_histogram(&pixels_8bit);
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

    let json = serde_json::json!({
        "total_frames": stats.total_frames,
        "avg_fps": avg_fps,
        "temperatures": stats.last_temperatures,
        "histogram": stats.histogram,
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

pub fn create_router<C: CameraInterface + 'static>(state: Arc<AppState<C>>) -> Router {
    Router::new()
        .route("/", get(camera_status_page::<C>))
        .route("/jpeg", get(jpeg_frame_endpoint::<C>))
        .route("/raw", get(raw_frame_endpoint::<C>))
        .route("/stats", get(stats_endpoint::<C>))
        .with_state(state)
}

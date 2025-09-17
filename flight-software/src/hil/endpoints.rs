use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{Html, Response},
    routing::get,
    Router,
};
use image::{ImageBuffer, Luma};
use std::sync::Arc;

use crate::camera::neutralino_imx455::read_sensor_temperatures;
use crate::camera::v4l2_utils::collect_camera_metadata;

use super::image_utils::{compute_histogram, process_raw_to_pixels};
use super::state::AppState;

pub async fn capture_frame_data(
    state: &AppState,
) -> Result<(Vec<u8>, v4l::buffer::Metadata), String> {
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

pub async fn raw_frame_endpoint(State(state): State<Arc<AppState>>) -> Response {
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

pub async fn jpeg_frame_endpoint(State(state): State<Arc<AppState>>) -> Response {
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

pub async fn stats_endpoint(State(state): State<Arc<AppState>>) -> Response {
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

pub async fn camera_status_page(State(state): State<Arc<AppState>>) -> Html<String> {
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

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(camera_status_page))
        .route("/jpeg", get(jpeg_frame_endpoint))
        .route("/raw", get(raw_frame_endpoint))
        .route("/stats", get(stats_endpoint))
        .with_state(state)
}

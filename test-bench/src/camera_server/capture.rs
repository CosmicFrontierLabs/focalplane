use image::DynamicImage;
use monocle::{apply_camera_settings, FgsState};
use shared::camera_interface::CameraInterface;
use shared::image_proc::u16_to_gray_image;
use shared::tracking_message::TrackingMessage;
use shared_wasm::{
    CameraStats, CameraTimingStats, FgsWsMessage, StarDetectionSettings, TrackingPosition,
    TrackingSettings, TrackingState,
};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, Notify, RwLock};

use crate::mjpeg::encode_ndarray_jpeg;
use crate::ws_log_stream::LogBroadcaster;
use crate::ws_stream::{WsBroadcaster, WsFrame};

use super::{
    create_router, encode_zoom_frame, extract_patch, AppState, CommonServerArgs,
    FgsStatusBroadcaster, FrameStats, FsmSharedState, SharedFrame, TrackingConfig,
    TrackingSharedState, ZoomRegion, MJPEG_MIN_INTERVAL,
};

pub fn capture_loop_blocking<C: CameraInterface + Send + 'static>(state: Arc<AppState<C>>) {
    let mut camera = state.camera.blocking_lock();
    let mut last_mjpeg_publish = Instant::now() - MJPEG_MIN_INTERVAL;

    let state_clone = state.clone();
    let result = camera.stream(&mut |frame, metadata| {
        let start_capture = Instant::now();
        let frame_num = metadata.frame_number;

        let frame_owned = frame.clone();
        let metadata_owned = metadata.clone();

        {
            let mut stats = state_clone.stats.blocking_lock();
            let now = Instant::now();
            let elapsed = now.duration_since(stats.last_frame_time).as_secs_f32();

            if elapsed > 0.0 {
                let fps = 1.0 / elapsed;
                stats.fps_samples.push(fps);
            }

            let capture_time = start_capture.elapsed();
            stats
                .capture_timing_ms
                .push(capture_time.as_secs_f32() * 1000.0);

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
            let capture_timestamp = Instant::now();
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
                last_mjpeg_publish = Instant::now();
            }
        }

        // Publish zoom frame if subscribers and coordinates are set
        if state_clone.ws_zoom.subscriber_count() > 0 {
            let center = { *state_clone.ws_zoom_center.read().unwrap() };
            if let Some((cx, cy)) = center {
                if let Some(zoom_frame) = encode_zoom_frame(&frame_owned, cx, cy, frame_num) {
                    state_clone.ws_zoom.publish(zoom_frame);
                }
            }
        }

        true
    });

    if let Err(e) = result {
        tracing::error!("Camera stream error: {e}");
    }
}

pub async fn analysis_loop<C: CameraInterface + Send + 'static>(state: Arc<AppState<C>>) {
    let mut last_analysis_time = Instant::now();

    loop {
        let start_total = Instant::now();
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
                let start_render = Instant::now();
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

                    stats
                        .total_pipeline_ms
                        .push(total_time.as_secs_f32() * 1000.0);

                    // Publish CameraStats to WebSocket clients
                    state
                        .fgs_status
                        .publish(FgsWsMessage::CameraStats(CameraStats {
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
                            timestamp: Instant::now(),
                        });
                        drop(zoom);
                        state.zoom_notify.notify_waiters();
                    }

                    tracing::trace!(
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

/// Run the camera server.
///
/// Starts an HTTP server with camera streaming, tracking, FSM control,
/// and log streaming endpoints.
pub async fn run_server<C: CameraInterface + Send + 'static>(
    mut camera: C,
    args: CommonServerArgs,
    tracking_config: TrackingConfig,
    fsm: Option<Arc<FsmSharedState>>,
    log_broadcaster: Arc<LogBroadcaster>,
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
    let ws_zoom = Arc::new(WsBroadcaster::new(4));

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
        ws_zoom,
        ws_zoom_center: Arc::new(std::sync::RwLock::new(None)),
        log_broadcaster,
        fgs_status: Arc::new(FgsStatusBroadcaster::new(64)),
        star_detection_settings: Arc::new(RwLock::new(StarDetectionSettings::default())),
        encoding_threads: args.encoding_threads,
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

/// Worker thread that runs FGS tracking on received frames.
///
/// Takes the `FineGuidanceSystem` from `shared_fgs` on entry, and puts it back when
/// the channel disconnects (stream restart). This preserves tracking state across
/// V4L2 stream stop/start cycles needed for ROI changes.
fn fgs_worker<C: CameraInterface + 'static>(
    rx: std::sync::mpsc::Receiver<SharedFrame>,
    state: Arc<AppState<C>>,
    tracking_shared: Arc<TrackingSharedState>,
    tracking_config: TrackingConfig,
    pending_settings: Arc<std::sync::Mutex<Vec<monocle::CameraSettingsUpdate>>>,
    roi_h_alignment: usize,
    roi_v_alignment: usize,
    shared_fgs: Arc<std::sync::Mutex<Option<monocle::FineGuidanceSystem>>>,
    clear_roi_requested: Arc<AtomicBool>,
) {
    use monocle::config::{FgsConfig, GuideStarFilters};
    use monocle::state::FgsEvent;
    use monocle::FineGuidanceSystem;

    let mut fgs: Option<FineGuidanceSystem> = shared_fgs.lock().unwrap().take();

    while let Ok(shared) = rx.recv() {
        let (frame, metadata) = shared.as_ref();

        let tracking_enabled = tracking_shared.enabled.load(Ordering::SeqCst);

        if tracking_enabled {
            // Initialize FGS if needed
            if fgs.is_none() {
                let settings = tracking_shared.settings.blocking_read().clone();
                let fgs_config = FgsConfig {
                    acquisition_frames: settings.acquisition_frames,
                    filters: GuideStarFilters {
                        detection_threshold_sigma: settings.detection_threshold_sigma,
                        snr_min: settings.snr_min,
                        diameter_range: (2.0, 100000.0),
                        aspect_ratio_max: 2.5,
                        saturation_value: tracking_config.saturation_value,
                        saturation_search_radius: 3.0,
                        minimum_edge_distance: 40.0,
                        bad_pixel_map: tracking_config.bad_pixel_map.clone(),
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
                let mut new_fgs =
                    FineGuidanceSystem::new(fgs_config, (roi_h_alignment, roi_v_alignment));

                // Register callback for tracking updates
                let tracking_cb = tracking_shared.clone();
                let fgs_status_cb = state.fgs_status.clone();
                new_fgs.register_callback(move |event| {
                    use monocle::callback::FgsCallbackEvent;
                    match event {
                        FgsCallbackEvent::TrackingStarted {
                            track_id,
                            initial_position,
                            num_guide_stars,
                        } => {
                            tracing::info!(
                                "TRACKING LOCKED - track_id: {}, position: ({:.2}, {:.2}), guide stars: {}",
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
                            fgs_status_cb
                                .publish(FgsWsMessage::TrackingStatus(status.clone()));

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
                                "Tracking update - track_id: {}, position: ({:.4}, {:.4}), snr: {:.2}",
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
                            fgs_status_cb
                                .publish(FgsWsMessage::TrackingStatus(status.clone()));

                            let msg = TrackingMessage::new(
                                *track_id,
                                position.x,
                                position.y,
                                position.timestamp,
                                position.shape.clone(),
                            );
                            let _ = tracking_cb.sse_tx.send(msg);
                        }
                        FgsCallbackEvent::TrackingLost { track_id, reason } => {
                            tracing::warn!(
                                "TRACKING LOST - track_id: {}, reason: {:?}",
                                track_id,
                                reason
                            );
                            let mut status = tracking_cb.status.blocking_write();
                            status.position = None;
                            status.num_guide_stars = 0;
                            fgs_status_cb
                                .publish(FgsWsMessage::TrackingStatus(status.clone()));
                        }
                        FgsCallbackEvent::FrameProcessed { .. } => {}
                        FgsCallbackEvent::FrameSizeMismatch { .. } => {}
                    }
                });

                if let Err(e) = new_fgs.process_event(FgsEvent::StartFgs) {
                    tracing::error!("Failed to start FGS: {e}");
                }

                fgs = Some(new_fgs);
            }

            // Process frame through FGS
            if let Some(ref mut fgs_instance) = fgs {
                match fgs_instance.process_frame(frame.view(), metadata.timestamp) {
                    Ok((_update, settings)) => {
                        {
                            let mut status = tracking_shared.status.blocking_write();
                            status.state = convert_fgs_state_to_shared(fgs_instance.state());
                            state
                                .fgs_status
                                .publish(FgsWsMessage::TrackingStatus(status.clone()));
                        }

                        if !settings.is_empty() {
                            tracing::info!(
                                "Camera settings changed ({} updates), signaling capture restart",
                                settings.len()
                            );
                            *pending_settings.lock().unwrap() = settings;
                            tracking_shared
                                .restart_requested
                                .store(true, Ordering::SeqCst);
                        }

                        if matches!(fgs_instance.state(), monocle::FgsState::Idle) {
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
        } else if let Some(ref mut fgs_instance) = fgs {
            // Tracking disabled — reset FGS and signal ROI clear
            use monocle::CameraSettingsUpdate;
            let camera_updates = fgs_instance.reset();
            for update in camera_updates {
                if matches!(update, CameraSettingsUpdate::ClearROI) {
                    tracing::info!("FGS reset, signaling ROI clear");
                    clear_roi_requested.store(true, Ordering::SeqCst);
                    tracking_shared
                        .restart_requested
                        .store(true, Ordering::SeqCst);
                }
            }
            fgs = None;
            tracing::info!("Tracking disabled, FGS reset");
        }
    }

    // Channel disconnected — preserve FGS state for next stream iteration
    *shared_fgs.lock().unwrap() = fgs;
    tracing::debug!("FGS worker exiting, FGS state preserved");
}

/// Worker thread that encodes frames as JPEG and publishes to WebSocket streams.
///
/// Multiple instances run concurrently, pulling from a shared crossbeam MPMC channel.
/// Each worker rate-limits its own publishes and skips encoding when no subscribers
/// are connected.
fn encoding_worker(
    rx: crossbeam_channel::Receiver<SharedFrame>,
    ws_stream: Arc<WsBroadcaster>,
    ws_zoom: Arc<WsBroadcaster>,
    ws_zoom_center: Arc<std::sync::RwLock<Option<(usize, usize)>>>,
) {
    let mut last_publish = Instant::now() - MJPEG_MIN_INTERVAL;

    while let Ok(shared) = rx.recv() {
        let (frame, metadata) = shared.as_ref();
        let frame_num = metadata.frame_number;

        if ws_stream.subscriber_count() > 0 && last_publish.elapsed() >= MJPEG_MIN_INTERVAL {
            if let Some(jpeg_data) = encode_ndarray_jpeg(frame, 80) {
                ws_stream.publish(WsFrame {
                    jpeg_data,
                    frame_number: frame_num,
                    width: frame.ncols() as u32,
                    height: frame.nrows() as u32,
                });
                last_publish = Instant::now();
            }
        }

        if ws_zoom.subscriber_count() > 0 {
            let center = { *ws_zoom_center.read().unwrap() };
            if let Some((cx, cy)) = center {
                if let Some(zoom_frame) = encode_zoom_frame(frame, cx, cy, frame_num) {
                    ws_zoom.publish(zoom_frame);
                }
            }
        }
    }
}

/// Capture loop with integrated tracking support.
///
/// Frames are captured in the callback and dispatched to worker threads:
/// - FGS thread: receives frames via bounded mpsc (try_send, drops with warning)
/// - Encoding workers: pull from crossbeam MPMC channel for JPEG encoding
pub fn capture_loop_with_tracking<C: CameraInterface + Send + 'static>(state: Arc<AppState<C>>) {
    use monocle::CameraSettingsUpdate;
    use std::time::Duration;

    let tracking_shared = state
        .tracking
        .as_ref()
        .expect("capture_loop_with_tracking requires tracking state");
    let tracking_config = state
        .tracking_config
        .as_ref()
        .expect("capture_loop_with_tracking requires tracking config")
        .clone();

    // Get ROI alignment from camera (immutable hardware property)
    let (roi_h_alignment, roi_v_alignment) = {
        let camera = state.camera.blocking_lock();
        camera.get_roi_offset_alignment()
    };

    let pending_settings: Arc<std::sync::Mutex<Vec<CameraSettingsUpdate>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let clear_roi_requested: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

    // FGS instance persists across stream restarts (ROI changes require stop/restart)
    let shared_fgs: Arc<std::sync::Mutex<Option<monocle::FineGuidanceSystem>>> =
        Arc::new(std::sync::Mutex::new(None));

    loop {
        let loop_iteration_start = Instant::now();
        let mut camera = state.camera.blocking_lock();

        // Create channels for this stream iteration
        let (fgs_tx, fgs_rx) = std::sync::mpsc::sync_channel::<SharedFrame>(2);
        let (encode_tx, encode_rx) = crossbeam_channel::bounded::<SharedFrame>(2);

        // Spawn FGS worker thread
        let fgs_state = state.clone();
        let fgs_tracking = tracking_shared.clone();
        let fgs_config = tracking_config.clone();
        let fgs_pending = pending_settings.clone();
        let fgs_shared = shared_fgs.clone();
        let fgs_clear_roi = clear_roi_requested.clone();
        let fgs_handle = std::thread::spawn(move || {
            fgs_worker(
                fgs_rx,
                fgs_state,
                fgs_tracking,
                fgs_config,
                fgs_pending,
                roi_h_alignment,
                roi_v_alignment,
                fgs_shared,
                fgs_clear_roi,
            );
        });

        // Spawn encoding worker threads
        let num_encoders = state.encoding_threads;
        let encode_handles: Vec<_> = (0..num_encoders)
            .map(|_| {
                let rx = encode_rx.clone();
                let ws_stream = state.ws_stream.clone();
                let ws_zoom = state.ws_zoom.clone();
                let ws_zoom_center = state.ws_zoom_center.clone();
                std::thread::spawn(move || {
                    encoding_worker(rx, ws_stream, ws_zoom, ws_zoom_center);
                })
            })
            .collect();
        drop(encode_rx);

        let state_clone = state.clone();
        let tracking_for_loop = tracking_shared.clone();

        let stream_result = camera.stream(&mut |frame, metadata| {
            let start_capture = Instant::now();
            let frame_num = metadata.frame_number;

            // Check if restart requested (e.g., tracking toggled, ROI change)
            if tracking_for_loop
                .restart_requested
                .swap(false, Ordering::SeqCst)
            {
                tracing::info!("Restart requested, stopping stream");
                return false;
            }

            let shared: SharedFrame = Arc::new((frame.clone(), metadata.clone()));

            // FGS: try_send, warn + drop if behind
            if tracking_for_loop.enabled.load(Ordering::SeqCst) {
                if fgs_tx.try_send(Arc::clone(&shared)).is_err() {
                    tracing::warn!("FGS behind, dropping frame {}", frame_num);
                }
            }

            // Encoding: try_send to worker pool (drop silently if both workers busy)
            let _ = encode_tx.try_send(shared.clone());

            // Stats + latest frame (keep inline for accurate timing)
            {
                let mut stats = state_clone.stats.blocking_lock();
                let now = Instant::now();
                let elapsed = now.duration_since(stats.last_frame_time).as_secs_f32();

                if elapsed > 0.0 {
                    let fps = 1.0 / elapsed;
                    stats.fps_samples.push(fps);
                }

                let capture_time = start_capture.elapsed();
                stats
                    .capture_timing_ms
                    .push(capture_time.as_secs_f32() * 1000.0);

                stats.total_frames += 1;
                stats.last_frame_time = now;
                stats.last_temperatures = metadata.temperatures.clone();

                tracing::debug!(
                    "Capture: {:.1}ms, frame_num={}",
                    capture_time.as_secs_f64() * 1000.0,
                    frame_num
                );
            }

            {
                let (ref frame_data, ref meta) = *shared;
                let capture_timestamp = Instant::now();
                let mut latest = state_clone.latest_frame.blocking_write();
                *latest = Some((frame_data.clone(), meta.clone(), capture_timestamp));
            }

            true
        });

        // Drop senders so worker threads see channel disconnect and exit
        drop(fgs_tx);
        drop(encode_tx);

        // Wait for worker threads to finish
        let _ = fgs_handle.join();
        for handle in encode_handles {
            let _ = handle.join();
        }

        // Apply any pending camera settings after stream ends
        let settings = {
            let mut guard = pending_settings.lock().unwrap();
            std::mem::take(&mut *guard)
        };
        if !settings.is_empty() {
            let apply_start = Instant::now();
            tracing::info!("Applying {} camera settings...", settings.len());
            if let Err(errors) = apply_camera_settings(&mut *camera, settings) {
                for err in &errors {
                    tracing::error!("Camera settings error: {}", err);
                }
                tracing::warn!("Camera settings failed, requesting ROI clear");
                clear_roi_requested.store(true, Ordering::SeqCst);
            }
            tracing::info!(
                "apply_camera_settings took {:.1}ms",
                apply_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Clear ROI if requested
        if clear_roi_requested.swap(false, Ordering::SeqCst) {
            let clear_start = Instant::now();
            tracing::info!("Clearing camera ROI...");
            if let Err(e) = camera.clear_roi() {
                tracing::warn!("Failed to clear ROI: {e}");
            }
            tracing::info!(
                "clear_roi took {:.1}ms",
                clear_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        drop(camera);

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

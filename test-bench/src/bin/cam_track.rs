//! Camera tracking binary using monocle FGS for all camera types.

use anyhow::{Context, Result};
use clap::Parser;
use monocle::{
    callback::FgsCallbackEvent,
    config::{FgsConfig, GuideStarFilters},
    state::FgsEvent,
    FineGuidanceSystem,
};
use serde::Serialize;
use shared::camera_interface::{CameraInterface, Timestamp};
use shared::config_storage::ConfigStorage;
use shared::frame_writer::{FrameFormat, FrameWriterHandle};
use shared::tracking_message::TrackingMessage;
use shared::zmq::TypedZmqPublisher;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use test_bench::camera_init::{initialize_camera, CameraArgs};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize)]
struct FrameMetadata {
    frame_number: usize,
    timestamp: Timestamp,
    track_id: Option<u32>,
    centroid_x: Option<f64>,
    centroid_y: Option<f64>,
    width: usize,
    height: usize,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Tracking binary for all camera types")]
struct Args {
    #[command(flatten)]
    camera: CameraArgs,

    #[arg(long, default_value = "5")]
    acquisition_frames: usize,

    #[arg(long, default_value = "64")]
    roi_size: usize,

    #[arg(long, default_value = "5.0")]
    detection_threshold_sigma: f64,

    #[arg(long, default_value = "10.0")]
    snr_min: f64,

    #[arg(
        long,
        default_value = "3.0",
        help = "SNR threshold below which tracking is lost"
    )]
    snr_dropout_threshold: f64,

    #[arg(
        short = 'e',
        long,
        default_value = "25",
        help = "Camera exposure time in milliseconds"
    )]
    exposure_ms: u64,

    #[arg(short = 'g', long, help = "Camera gain setting")]
    gain: Option<f64>,

    #[arg(
        long,
        help = "Maximum runtime in seconds (runs indefinitely if not specified)"
    )]
    max_runtime_secs: Option<u64>,

    #[arg(
        long,
        help = "Output tracking data to CSV file (track_id,x,y,timestamp)"
    )]
    csv_output: Option<String>,

    #[arg(
        long,
        help = "Export frames to directory as PNG + JSON (enables background export thread pool)"
    )]
    export_frames: Option<PathBuf>,

    #[arg(
        long,
        default_value = "4",
        help = "Number of worker threads for frame export (requires --export-frames)"
    )]
    export_workers: usize,

    #[arg(
        long,
        default_value = "100",
        help = "Frame export queue buffer size (number of frames to buffer in RAM)"
    )]
    export_buffer_size: usize,

    #[arg(
        long,
        help = "ZMQ PUB socket bind address for tracking updates (e.g., tcp://*:5555)"
    )]
    zmq_pub: Option<String>,
}

fn log_fgs_state(state: &monocle::FgsState) {
    match state {
        monocle::FgsState::Acquiring { frames_collected } => {
            if frames_collected % 5 == 0 {
                info!("Acquiring... collected {} frames", frames_collected);
            }
        }
        monocle::FgsState::Calibrating => {
            info!("Calibrating guide stars...");
        }
        monocle::FgsState::Tracking { frames_processed } => {
            if frames_processed % 100 == 0 && *frames_processed > 0 {
                info!("Tracking... processed {} frames", frames_processed);
            }
        }
        monocle::FgsState::Reacquiring { attempts } => {
            warn!("Reacquiring lock... attempt {}", attempts);
        }
        monocle::FgsState::Idle => {
            info!("System idle");
        }
    }
}

fn run_tracking_stream(
    camera: &mut Box<dyn CameraInterface>,
    fgs: &mut FineGuidanceSystem,
    max_runtime_secs: Option<u64>,
    start_time: Instant,
) -> Result<()> {
    loop {
        info!("Starting camera stream");
        let pending_settings: Arc<Mutex<Vec<monocle::CameraSettingsUpdate>>> =
            Arc::new(Mutex::new(Vec::new()));
        let pending_settings_clone = pending_settings.clone();
        let should_exit = Arc::new(Mutex::new(false));
        let should_exit_clone = should_exit.clone();
        let should_restart_fgs = Arc::new(Mutex::new(false));
        let should_restart_fgs_clone = should_restart_fgs.clone();

        let stream_result = camera.stream(&mut |frame, metadata| {
            if let Some(max_secs) = max_runtime_secs {
                if start_time.elapsed().as_secs() >= max_secs {
                    info!("Max runtime of {} seconds reached, exiting", max_secs);
                    *should_exit_clone.lock().unwrap() = true;
                    return false;
                }
            }

            match fgs.process_frame(frame.view(), metadata.timestamp) {
                Ok((_update, settings)) => {
                    if !settings.is_empty() {
                        info!(
                            "Camera settings changed ({} updates), stopping stream to apply...",
                            settings.len()
                        );
                        *pending_settings_clone.lock().unwrap() = settings;
                        return false;
                    }

                    // Check if FGS is idle (reacquisition failed) and trigger restart
                    if matches!(fgs.state(), monocle::FgsState::Idle) {
                        info!("FGS went idle, will restart acquisition");
                        *should_restart_fgs_clone.lock().unwrap() = true;
                        return false;
                    }

                    log_fgs_state(fgs.state());
                    true
                }
                Err(e) => {
                    warn!("Frame processing error: {}", e);
                    true
                }
            }
        });

        match stream_result {
            Ok(_) => {
                if *should_exit.lock().unwrap() {
                    info!("Exiting tracking stream");
                    break;
                }

                let settings = pending_settings.lock().unwrap().clone();
                if !settings.is_empty() {
                    info!("Applying {} camera settings...", settings.len());
                    monocle::apply_camera_settings(camera, settings);
                    info!("Settings applied, restarting stream");
                } else if *should_restart_fgs.lock().unwrap() {
                    info!("Restarting FGS acquisition...");
                    match fgs.process_event(FgsEvent::StartFgs) {
                        Ok((_, new_settings)) => {
                            monocle::apply_camera_settings(camera, new_settings);
                            info!("FGS restarted, continuing stream");
                        }
                        Err(e) => {
                            warn!("Failed to restart FGS: {}, retrying...", e);
                        }
                    }
                } else {
                    info!("Stream completed unexpectedly, restarting...");
                }
            }
            Err(e) => {
                warn!("Camera stream error: {}, restarting...", e);
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }

    Ok(())
}

fn save_frame_metadata(metadata: &FrameMetadata, export_dir: &PathBuf) -> Result<()> {
    let base_name = format!("frame_{:06}", metadata.frame_number);
    let json_path = export_dir.join(format!("{}.json", base_name));
    let json_data =
        serde_json::to_string_pretty(metadata).context("Failed to serialize frame metadata")?;
    std::fs::create_dir_all(export_dir)
        .with_context(|| format!("Failed to create directory {}", export_dir.display()))?;
    std::fs::write(&json_path, json_data)
        .with_context(|| format!("Failed to write JSON to {}", json_path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Initializing camera...");
    let mut camera = initialize_camera(&args.camera)?;

    let exposure = Duration::from_millis(args.exposure_ms);
    info!("Setting camera exposure to {}ms", args.exposure_ms);
    camera
        .set_exposure(exposure)
        .map_err(|e| anyhow::anyhow!("Failed to set camera exposure: {e}"))?;

    if let Some(gain) = args.gain {
        info!("Setting camera gain to {}", gain);
        camera
            .set_gain(gain)
            .map_err(|e| anyhow::anyhow!("Failed to set camera gain: {e}"))?;
    }

    let config_store = ConfigStorage::new().context("Failed to initialize config storage")?;

    info!(
        "Loading bad pixel map for {} (serial: {})",
        camera.name(),
        camera.get_serial()
    );

    let bad_pixel_map = config_store
        .get_bad_pixel_map(camera.name(), &camera.get_serial())
        .with_context(|| {
            format!(
                "No bad pixel map found for camera {} (serial: {}). Please run dark_frame_analysis first.",
                camera.name(),
                camera.get_serial()
            )
        })?
        .with_context(|| {
            format!(
                "Failed to load bad pixel map for camera {} (serial: {})",
                camera.name(),
                camera.get_serial()
            )
        })?;

    info!(
        "Loaded bad pixel map with {} bad pixels",
        bad_pixel_map.num_bad_pixels()
    );

    let config = FgsConfig {
        acquisition_frames: args.acquisition_frames,
        filters: GuideStarFilters {
            detection_threshold_sigma: args.detection_threshold_sigma,
            snr_min: args.snr_min,
            diameter_range: (2.0, 100000.0),
            aspect_ratio_max: 2.5,
            saturation_value: camera.saturation_value(),
            saturation_search_radius: 3.0,
            minimum_edge_distance: 40.0,
            bad_pixel_map,
            minimum_bad_pixel_distance: 5.0,
        },
        roi_size: args.roi_size,
        max_reacquisition_attempts: 5,
        centroid_radius_multiplier: 3.0,
        fwhm: 7.0,
        snr_dropout_threshold: args.snr_dropout_threshold,
    };

    let csv_writer = if let Some(ref csv_path) = args.csv_output {
        info!("Creating CSV output file: {}", csv_path);
        let file = std::fs::File::create(csv_path)
            .with_context(|| format!("Failed to create CSV file: {csv_path}"))?;
        let mut writer = csv::WriterBuilder::new()
            .has_headers(true)
            .from_writer(file);
        writer
            .write_record(&["track_id", "x", "y", "timestamp"])
            .context("Failed to write CSV header")?;
        Some(Arc::new(Mutex::new(writer)))
    } else {
        None
    };

    let frame_writer: Option<(Arc<FrameWriterHandle>, PathBuf)> =
        if let Some(ref export_dir) = args.export_frames {
            info!(
                "Initializing frame export to: {} with {} workers",
                export_dir.display(),
                args.export_workers
            );

            let writer = FrameWriterHandle::new(args.export_workers, args.export_buffer_size)
                .context("Failed to create frame writer")?;

            Some((Arc::new(writer), export_dir.clone()))
        } else {
            None
        };

    let current_tracking_state: Arc<Mutex<Option<(u32, f64, f64, Timestamp)>>> =
        Arc::new(Mutex::new(None));

    let zmq_publisher: Option<Arc<TypedZmqPublisher<TrackingMessage>>> =
        if let Some(ref bind_addr) = args.zmq_pub {
            info!("Creating ZMQ PUB socket binding to {}", bind_addr);
            let ctx = zmq::Context::new();
            let socket = ctx
                .socket(zmq::PUB)
                .context("Failed to create ZMQ PUB socket")?;
            socket
                .bind(bind_addr)
                .with_context(|| format!("Failed to bind ZMQ socket to {bind_addr}"))?;
            info!("ZMQ PUB socket bound to {}", bind_addr);
            Some(Arc::new(TypedZmqPublisher::new(socket)))
        } else {
            None
        };

    info!("Creating Fine Guidance System");
    let mut fgs = FineGuidanceSystem::new(config);

    let csv_writer_clone = csv_writer.clone();
    let tracking_state_clone = current_tracking_state.clone();
    let frame_writer_clone = frame_writer
        .as_ref()
        .map(|(writer, dir)| (writer.clone(), dir.clone()));
    let zmq_publisher_clone = zmq_publisher.clone();
    let _callback_id = fgs.register_callback(move |event| match event {
        FgsCallbackEvent::TrackingStarted {
            track_id,
            initial_position,
            num_guide_stars,
        } => {
            info!(
                "🎯 TRACKING LOCKED - track_id: {}, pixel location: (x={:.2}, y={:.2}), guide stars: {}",
                track_id, initial_position.x, initial_position.y, num_guide_stars
            );
            if let Ok(mut state) = tracking_state_clone.lock() {
                *state = Some((
                    *track_id,
                    initial_position.x,
                    initial_position.y,
                    initial_position.timestamp,
                ));
            }
        }
        FgsCallbackEvent::TrackingUpdate { track_id, position } => {
            info!(
                "📍 Tracking update - track_id: {}, pixel location: (x={:.4}, y={:.4}), timestamp: {}",
                track_id, position.x, position.y, position.timestamp
            );

            if let Ok(mut state) = tracking_state_clone.lock() {
                *state = Some((*track_id, position.x, position.y, position.timestamp));
            }

            if let Some(ref writer) = csv_writer_clone {
                if let Ok(mut csv) = writer.lock() {
                    if let Err(e) = csv.write_record(&[
                        track_id.to_string(),
                        position.x.to_string(),
                        position.y.to_string(),
                        position.timestamp.to_string(),
                    ]) {
                        warn!("Failed to write CSV record: {}", e);
                    }
                    if let Err(e) = csv.flush() {
                        warn!("Failed to flush CSV writer: {}", e);
                    }
                }
            }

            if let Some(ref publisher) = zmq_publisher_clone {
                let msg = TrackingMessage {
                    track_id: *track_id,
                    x: position.x,
                    y: position.y,
                    timestamp: position.timestamp,
                };
                if let Err(e) = publisher.send(&msg) {
                    warn!("Failed to send ZMQ message: {}", e);
                }
            }
        }
        FgsCallbackEvent::TrackingLost {
            track_id,
            last_position,
            reason,
        } => {
            warn!(
                "⚠️  TRACKING LOST - track_id: {}, last pixel location: (x={:.2}, y={:.2}), reason: {:?}",
                track_id, last_position.x, last_position.y, reason
            );
        }
        FgsCallbackEvent::FrameSizeMismatch {
            expected_width,
            expected_height,
            actual_width,
            actual_height,
        } => {
            warn!(
                "⚠️  Frame size mismatch - expected: {}x{}, actual: {}x{}",
                expected_width, expected_height, actual_width, actual_height
            );
        }
        FgsCallbackEvent::FrameProcessed {
            frame_number,
            timestamp,
            frame_data,
            track_id,
            position,
        } => {
            if let Some((ref writer, ref export_dir)) = frame_writer_clone {
                let (height, width) = frame_data.dim();
                let metadata = FrameMetadata {
                    frame_number: *frame_number,
                    timestamp: *timestamp,
                    track_id: *track_id,
                    centroid_x: position.as_ref().map(|p| p.x),
                    centroid_y: position.as_ref().map(|p| p.y),
                    width,
                    height,
                };

                if let Err(e) = save_frame_metadata(&metadata, export_dir) {
                    warn!("Failed to save metadata for frame {}: {}", frame_number, e);
                }

                let png_path = export_dir.join(format!("frame_{:06}.png", frame_number));
                if !writer.write_frame_nonblocking(frame_data, png_path, FrameFormat::Png) {
                    warn!("Frame export queue full, dropping frame {frame_number}");
                }
            }
        }
    });

    info!("Starting FGS acquisition");
    let (_update, settings) = fgs
        .process_event(FgsEvent::StartFgs)
        .map_err(|e| anyhow::anyhow!("Failed to start FGS: {e}"))?;
    monocle::apply_camera_settings(&mut camera, settings);

    let start_time = Instant::now();
    if let Some(max_secs) = args.max_runtime_secs {
        info!(
            "Entering tracking loop - will exit after {} seconds",
            max_secs
        );
    } else {
        info!("Entering tracking loop - press Ctrl+C to exit");
    }

    run_tracking_stream(&mut camera, &mut fgs, args.max_runtime_secs, start_time)
}

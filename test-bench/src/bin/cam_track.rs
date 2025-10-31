//! Camera tracking binary using monocle FGS for all camera types.

use anyhow::{Context, Result};
use clap::Parser;
use monocle::{
    callback::FgsCallbackEvent,
    config::{FgsConfig, GuideStarFilters},
    state::FgsEvent,
    FineGuidanceSystem,
};
use shared::{camera_interface::CameraInterface, config_storage::ConfigStorage};
use simulator::io::fits::{write_typed_fits, FitsDataType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use test_bench::camera_init::{initialize_camera, CameraArgs};
use tracing::{info, warn};

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
        short = 'e',
        long,
        default_value = "25",
        help = "Camera exposure time in milliseconds"
    )]
    exposure_ms: u64,

    #[arg(short = 'g', long, help = "Camera gain setting")]
    gain: Option<f64>,

    #[arg(
        short = 't',
        long,
        help = "Maximum runtime in seconds (runs indefinitely if not specified)"
    )]
    max_runtime_secs: Option<u64>,

    #[arg(long, help = "Save averaged acquisition frames as FITS file")]
    save_fits: bool,

    #[arg(
        long,
        help = "Output tracking data to CSV file (track_id,x,y,timestamp)"
    )]
    csv_output: Option<String>,
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

    info!("Creating Fine Guidance System");
    let mut fgs = FineGuidanceSystem::new(camera, config);

    let csv_writer_clone = csv_writer.clone();
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
        }
        FgsCallbackEvent::TrackingUpdate { track_id, position } => {
            info!(
                "📍 Tracking update - track_id: {}, pixel location: (x={:.4}, y={:.4}), timestamp: {}",
                track_id, position.x, position.y, position.timestamp
            );

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
    });

    info!("Starting FGS acquisition");
    fgs.process_event(FgsEvent::StartFgs)
        .map_err(|e| anyhow::anyhow!("Failed to start FGS: {e}"))?;

    let start_time = Instant::now();
    if let Some(max_secs) = args.max_runtime_secs {
        info!(
            "Entering tracking loop - will exit after {} seconds",
            max_secs
        );
    } else {
        info!("Entering tracking loop - press Ctrl+C to exit");
    }

    let mut prev_state = fgs.state().clone();
    let mut fits_saved = false;

    loop {
        if let Some(max_secs) = args.max_runtime_secs {
            if start_time.elapsed().as_secs() >= max_secs {
                info!("Max runtime of {} seconds reached, exiting", max_secs);
                break;
            }
        }
        match fgs.process_next_frame() {
            Ok(_update) => {
                let state = fgs.state();
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

                if args.save_fits
                    && !fits_saved
                    && matches!(prev_state, monocle::FgsState::Calibrating)
                    && !matches!(state, monocle::FgsState::Calibrating)
                {
                    if let Some(averaged_frame) = fgs.get_averaged_frame() {
                        info!("Saving averaged acquisition frames to FITS file...");
                        let mut data = HashMap::new();
                        data.insert(
                            "AVERAGED".to_string(),
                            FitsDataType::Float64(averaged_frame),
                        );

                        let fits_path = "cam_track_averaged_frame.fits";
                        match write_typed_fits(&data, fits_path) {
                            Ok(_) => {
                                info!("Averaged frame saved to: {}", fits_path);
                                fits_saved = true;
                            }
                            Err(e) => {
                                warn!("Failed to save FITS file: {}", e);
                            }
                        }
                    }
                }

                prev_state = state.clone();
            }
            Err(e) => {
                warn!("Frame processing error: {}", e);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }

    Ok(())
}

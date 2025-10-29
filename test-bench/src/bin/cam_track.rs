//! PlayerOne astronomy camera tracking binary using monocle FGS.
//!
//! IMPORTANT: This binary CANNOT be combined with cam_serve_nsv in the same executable.
//! The v4l2 libraries used by NSV455 cameras enumerate and claim USB video devices at
//! program initialization, which conflicts with the PlayerOne SDK's USB device access.

use anyhow::Result;
use clap::Parser;
use monocle::{
    callback::FgsCallbackEvent,
    config::{FgsConfig, GuideStarFilters},
    state::FgsEvent,
    FineGuidanceSystem,
};
use shared::camera_interface::CameraInterface;
use std::time::{Duration, Instant};
use test_bench::poa::camera::PlayerOneCamera;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Tracking binary for PlayerOne astronomy cameras"
)]
struct Args {
    #[arg(short = 'i', long, default_value = "0")]
    camera_id: i32,

    #[arg(long, default_value = "5")]
    acquisition_frames: usize,

    #[arg(long, default_value = "32")]
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

    #[arg(
        short = 't',
        long,
        help = "Maximum runtime in seconds (runs indefinitely if not specified)"
    )]
    max_runtime_secs: Option<u64>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Initializing PlayerOne camera with ID {}", args.camera_id);
    let mut camera = PlayerOneCamera::new(args.camera_id)
        .map_err(|e| anyhow::anyhow!("Failed to initialize POA camera: {e}"))?;

    let exposure = Duration::from_millis(args.exposure_ms);
    info!("Setting camera exposure to {}ms", args.exposure_ms);
    camera
        .set_exposure(exposure)
        .map_err(|e| anyhow::anyhow!("Failed to set camera exposure: {e}"))?;

    let config = FgsConfig {
        acquisition_frames: args.acquisition_frames,
        filters: GuideStarFilters {
            detection_threshold_sigma: args.detection_threshold_sigma,
            snr_min: args.snr_min,
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: camera.saturation_value(),
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
            bad_pixel_map: shared::bad_pixel_map::BadPixelMap::empty(),
            minimum_bad_pixel_distance: 5.0,
        },
        roi_size: args.roi_size,
        max_reacquisition_attempts: 5,
        centroid_radius_multiplier: 3.0,
        fwhm: 3.0,
    };

    info!("Creating Fine Guidance System");
    let mut fgs = FineGuidanceSystem::new(camera, config);

    let _callback_id = fgs.register_callback(|event| match event {
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
            }
            Err(e) => {
                warn!("Frame processing error: {}", e);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }

    Ok(())
}

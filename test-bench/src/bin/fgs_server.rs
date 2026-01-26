//! Unified camera server with tracking support.
//!
//! Combines the functionality of cam_serve (web UI, image streaming) and
//! cam_track (monocle FGS tracking) into a single binary. When tracking
//! is enabled via the web UI, the system detects spots, locks on, and tracks them.

use anyhow::Context;
use clap::Parser;
use hardware::pi::S330;
use shared::config_storage::ConfigStorage;
use std::sync::Arc;
use test_bench::camera_init::{initialize_camera, CameraArgs};
use test_bench::camera_server::{CommonServerArgs, FsmSharedState, TrackingConfig};
use tracing::info;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Unified camera server with FGS tracking",
    long_about = "Combined camera server and Fine Guidance System (FGS) tracking.\n\n\
        This server provides:\n  \
        - Web UI for camera control and image viewing\n  \
        - Real-time image streaming to browser\n  \
        - FGS tracking when enabled via the web interface\n  \
        - Optional ZMQ publishing of tracking updates\n  \
        - Optional FSM (Fast Steering Mirror) control via web UI\n\n\
        Prerequisites:\n  \
        - Run dark_frame_analysis to generate bad pixel map (recommended)\n  \
        - Build frontends: ./scripts/build-yew-frontends.sh\n\n\
        The web UI is available at http://localhost:<port> after startup."
)]
struct Args {
    #[command(flatten)]
    camera: CameraArgs,

    #[command(flatten)]
    server: CommonServerArgs,

    #[arg(
        long,
        default_value = "5",
        help = "Number of frames for FGS acquisition phase",
        long_help = "Number of full-frame images to collect during the FGS acquisition \
            phase before selecting guide stars. More frames improve detection reliability \
            but increase time to first lock. Typical range: 3-10 frames."
    )]
    acquisition_frames: usize,

    #[arg(
        long,
        default_value = "128",
        help = "Size of the tracking ROI in pixels (square)",
        long_help = "Size of the region-of-interest (ROI) window used for tracking each \
            guide star, in pixels. The ROI must be large enough to contain the full PSF \
            plus margin for motion, but small enough to minimize readout time. Must be \
            compatible with camera ROI alignment constraints. Typical range: 32-256 pixels."
    )]
    roi_size: usize,

    #[arg(
        long,
        default_value = "5.0",
        help = "Sigma threshold for star detection",
        long_help = "Detection threshold in standard deviations above background noise. \
            Stars with peak signal exceeding (background + threshold * noise) are detected. \
            Lower values detect fainter stars but may produce false positives from noise. \
            Typical range: 3.0-7.0 sigma."
    )]
    detection_threshold_sigma: f64,

    #[arg(
        long,
        default_value = "10.0",
        help = "Minimum signal-to-noise ratio for guide star selection",
        long_help = "Minimum signal-to-noise ratio (SNR) required for a detected star to be \
            selected as a guide star. Higher SNR stars provide more precise centroids but \
            may limit available guide stars. Typical range: 5.0-20.0."
    )]
    snr_min: f64,

    #[arg(
        long,
        default_value = "3.0",
        help = "SNR threshold below which tracking is lost",
        long_help = "If the tracked star's SNR drops below this threshold, tracking is \
            considered lost and reacquisition begins. Should be lower than snr_min to \
            provide hysteresis. Typical range: 2.0-5.0."
    )]
    snr_dropout_threshold: f64,

    #[arg(
        long,
        default_value = "7.0",
        help = "Expected FWHM of stars in pixels",
        long_help = "Expected full-width at half-maximum (FWHM) of the point spread function \
            in pixels. This affects the centroiding aperture size and SNR calculations. \
            Should match the actual PSF size from your optical system. \
            Typical range: 2.0-15.0 pixels."
    )]
    fwhm: f64,

    #[arg(
        long,
        help = "PI E-727 FSM controller IP address (enables FSM control)",
        long_help = "IP address of the PI E-727 piezo controller for the S-330 Fast Steering \
            Mirror. When specified, the web UI will include X/Y position sliders for manual \
            FSM control. The FSM is initialized with autozero and servo enable on startup.\n\n\
            Example: --fsm-ip 192.168.15.210",
        value_name = "IP"
    )]
    fsm_ip: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Initializing camera...");
    let camera = initialize_camera(&args.camera)?;

    let config_store = ConfigStorage::new().context("Failed to initialize config storage")?;

    info!(
        "Loading bad pixel map for {} (serial: {})",
        camera.name(),
        camera.get_serial()
    );

    let bad_pixel_map = match config_store.get_bad_pixel_map(camera.name(), &camera.get_serial()) {
        Some(Ok(map)) => {
            info!(
                "Loaded bad pixel map with {} bad pixels",
                map.num_bad_pixels()
            );
            map
        }
        Some(Err(e)) => {
            tracing::warn!(
                "Failed to load bad pixel map for camera {} (serial: {}): {}, using empty map",
                camera.name(),
                camera.get_serial(),
                e
            );
            shared::bad_pixel_map::BadPixelMap::empty()
        }
        None => {
            tracing::warn!(
                "No bad pixel map found for camera {} (serial: {}), using empty map",
                camera.name(),
                camera.get_serial()
            );
            shared::bad_pixel_map::BadPixelMap::empty()
        }
    };

    // Get ROI alignment constraints from camera
    let (roi_h_alignment, roi_v_alignment) = camera.get_roi_offset_alignment();
    info!(
        "Camera ROI alignment constraints: h={}, v={}",
        roi_h_alignment, roi_v_alignment
    );

    let tracking_config = TrackingConfig {
        acquisition_frames: args.acquisition_frames,
        roi_size: args.roi_size,
        detection_threshold_sigma: args.detection_threshold_sigma,
        snr_min: args.snr_min,
        snr_dropout_threshold: args.snr_dropout_threshold,
        fwhm: args.fwhm,
        bad_pixel_map,
        saturation_value: camera.saturation_value(),
        roi_h_alignment,
        roi_v_alignment,
    };

    // Initialize FSM if IP address provided
    let fsm_state = if let Some(ref fsm_ip) = args.fsm_ip {
        info!("Connecting to FSM at {}...", fsm_ip);
        match S330::connect_ip(fsm_ip) {
            Ok(mut fsm) => {
                info!("FSM connected, getting travel ranges...");
                match fsm.get_travel_ranges() {
                    Ok((x_range, y_range)) => {
                        info!(
                            "FSM travel ranges: X=[{:.1}, {:.1}], Y=[{:.1}, {:.1}] µrad",
                            x_range.0, x_range.1, y_range.0, y_range.1
                        );
                        let (x, y) = fsm.get_position().unwrap_or((0.0, 0.0));
                        Some(Arc::new(FsmSharedState {
                            fsm: std::sync::Mutex::new(fsm),
                            x_urad: std::sync::atomic::AtomicU64::new(x.to_bits()),
                            y_urad: std::sync::atomic::AtomicU64::new(y.to_bits()),
                            x_range,
                            y_range,
                            last_error: tokio::sync::RwLock::new(None),
                        }))
                    }
                    Err(e) => {
                        tracing::error!("Failed to get FSM travel ranges: {}, FSM disabled", e);
                        None
                    }
                }
            }
            Err(e) => {
                tracing::error!(
                    "Failed to connect to FSM at {}: {}, FSM disabled",
                    fsm_ip,
                    e
                );
                None
            }
        }
    } else {
        None
    };

    if fsm_state.is_some() {
        info!("FSM control enabled");
    }

    info!("Starting unified camera server with tracking support...");
    test_bench::camera_server::run_server_with_tracking(
        camera,
        args.server,
        tracking_config,
        fsm_state,
    )
    .await
}

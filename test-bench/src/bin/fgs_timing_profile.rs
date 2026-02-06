use clap::Parser;
use monocle::{
    config::FgsConfig,
    state::{FgsEvent, FgsState},
    CameraSettingsUpdate, FineGuidanceSystem,
};
use monocle_harness::{
    create_guide_star_catalog, create_jbt_hwk_camera_with_catalog_and_motion,
    motion_profiles::StaticPointing,
};
use ndarray::Array2;
use shared::bad_pixel_map::BadPixelMap;
use shared::camera_interface::{CameraInterface, FrameMetadata};
use starfield::Equatorial;
use std::sync::Arc;
use std::time::{Duration, Instant};
use test_bench::camera_init::{initialize_camera, CameraArgs, ExposureArgs};

/// Profile FGS tracking startup timing across multiple iterations
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Profile FGS tracking startup timing",
    long_about = "Runs the FGS state machine through Idle -> Acquiring -> Calibrating -> Tracking \
        multiple times and reports timing distributions for each phase.\n\n\
        Supports both simulated (default) and real hardware cameras via --camera-type."
)]
struct Args {
    #[command(flatten)]
    camera: CameraArgs,

    #[command(flatten)]
    exposure: ExposureArgs,

    #[arg(
        short = 'n',
        long,
        default_value_t = 20,
        help = "Number of test iterations"
    )]
    iterations: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Frames for acquisition phase (default 5)"
    )]
    acquisition_frames: usize,

    #[arg(long, default_value_t = 128, help = "ROI size in pixels (default 128)")]
    roi_size: usize,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "Minimum SNR for guide star selection"
    )]
    min_snr: f64,

    #[arg(
        long,
        default_value_t = 3.0,
        help = "Point spread function FWHM in pixels"
    )]
    fwhm: f64,

    #[arg(short, long, help = "Print per-iteration details")]
    verbose: bool,

    #[arg(
        long,
        help = "Use simulator camera with synthetic star field instead of CameraArgs"
    )]
    simulator: bool,
}

/// Timing measurements for a single FGS startup run
#[derive(Debug, Clone)]
struct IterationTimings {
    fgs_init_ms: f64,
    stream_start_ms: f64,
    acq_frame_times_ms: Vec<f64>,
    acq_total_ms: f64,
    calibration_ms: f64,
    stream_restart_ms: f64,
    first_track_frame_ms: f64,
    time_to_track_ms: f64,
}

/// Create a fresh SimulatorCamera pointed at Orion
fn create_simulator_camera(exposure: Duration) -> Box<dyn CameraInterface> {
    let pointing = Equatorial::from_degrees(83.0, -5.0);
    let catalog = Arc::new(create_guide_star_catalog(&pointing));
    let motion = Box::new(StaticPointing::new(83.0, -5.0));
    let mut camera = create_jbt_hwk_camera_with_catalog_and_motion(catalog, motion);
    camera
        .set_exposure(exposure)
        .expect("Failed to set exposure");
    Box::new(camera)
}

/// Apply camera settings updates to a trait object camera
fn apply_settings(
    camera: &mut dyn CameraInterface,
    updates: Vec<CameraSettingsUpdate>,
) -> Result<(), String> {
    for setting in updates {
        match setting {
            CameraSettingsUpdate::SetROI(roi) => {
                camera
                    .set_roi(roi)
                    .map_err(|e| format!("Failed to set ROI: {e}"))?;
            }
            CameraSettingsUpdate::ClearROI => {
                camera
                    .clear_roi()
                    .map_err(|e| format!("Failed to clear ROI: {e}"))?;
            }
        }
    }
    Ok(())
}

/// Run one complete FGS startup cycle using stream() and measure timings.
///
/// Uses the two-stream pattern matching real hardware:
/// 1. Full-frame stream for acquisition + calibration
/// 2. Stop stream, apply ROI settings
/// 3. ROI stream for tracking
fn run_iteration(
    camera: &mut dyn CameraInterface,
    args: &Args,
) -> Result<IterationTimings, String> {
    // Reset camera to full frame
    camera
        .clear_roi()
        .map_err(|e| format!("Failed to clear ROI: {e}"))?;

    // Measure FGS init
    let t0 = Instant::now();
    let (roi_h_alignment, roi_v_alignment) = camera.get_roi_offset_alignment();
    let config = FgsConfig {
        acquisition_frames: args.acquisition_frames,
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: args.min_snr,
            diameter_range: (2.0, 100.0),
            aspect_ratio_max: 2.5,
            saturation_value: camera.saturation_value() * 0.95,
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
            bad_pixel_map: BadPixelMap::empty(),
            minimum_bad_pixel_distance: 5.0,
        },
        roi_size: args.roi_size,
        max_reacquisition_attempts: 3,
        centroid_radius_multiplier: 3.0,
        fwhm: args.fwhm,
        snr_dropout_threshold: 3.0,
        noise_estimation_downsample: 16,
    };
    let mut fgs = FineGuidanceSystem::new(config, (roi_h_alignment, roi_v_alignment));
    let fgs_init_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Start FGS
    let time_to_track_start = Instant::now();
    let (_update, settings) = fgs
        .process_event(FgsEvent::StartFgs)
        .map_err(|e| e.to_string())?;
    apply_settings(camera, settings)?;

    // Phase 1: Full-frame stream for acquisition + calibration
    // Total frames needed: acquisition_frames + 1 calibration frame
    let total_acq_frames = args.acquisition_frames + 1;
    let mut acq_frame_times_ms = Vec::with_capacity(args.acquisition_frames);
    let mut calibration_ms = 0.0f64;
    let mut pending_settings: Vec<CameraSettingsUpdate> = Vec::new();
    let mut frame_count = 0usize;
    let mut acq_error: Option<String> = None;

    let t_stream_start = Instant::now();
    let acq_start = Instant::now();

    camera
        .stream(&mut |frame: &Array2<u16>, metadata: &FrameMetadata| {
            if frame_count == 0 {
                // First frame received - stream is now active
            }

            let t_frame = Instant::now();
            match fgs.process_frame(frame.view(), metadata.timestamp) {
                Ok((_update, settings)) => {
                    let elapsed = t_frame.elapsed().as_secs_f64() * 1000.0;

                    if frame_count < args.acquisition_frames {
                        acq_frame_times_ms.push(elapsed);
                    } else {
                        // Calibration frame
                        calibration_ms = elapsed;
                        pending_settings = settings;
                    }

                    frame_count += 1;
                    frame_count < total_acq_frames
                }
                Err(e) => {
                    acq_error = Some(e.to_string());
                    false
                }
            }
        })
        .map_err(|e| format!("Acquisition stream failed: {e}"))?;

    if let Some(e) = acq_error {
        return Err(e);
    }

    let stream_start_ms = (acq_start.elapsed().as_secs_f64() * 1000.0)
        - acq_frame_times_ms.iter().sum::<f64>()
        - calibration_ms;
    let acq_total_ms = acq_start.elapsed().as_secs_f64() * 1000.0;

    // Phase 2: Apply ROI settings (stream restart)
    let t_restart = Instant::now();
    apply_settings(camera, pending_settings)?;
    let stream_restart_ms_settings = t_restart.elapsed().as_secs_f64() * 1000.0;

    // Verify we're tracking
    if !matches!(fgs.state(), FgsState::Tracking { .. }) {
        return Err("FGS did not enter Tracking state".to_string());
    }

    // Phase 3: ROI stream for first tracking frame
    let mut first_track_frame_ms = 0.0f64;
    let mut track_error: Option<String> = None;

    camera
        .stream(&mut |frame: &Array2<u16>, metadata: &FrameMetadata| {
            let t_track = Instant::now();
            match fgs.process_frame(frame.view(), metadata.timestamp) {
                Ok((_update, _settings)) => {
                    first_track_frame_ms = t_track.elapsed().as_secs_f64() * 1000.0;
                }
                Err(e) => {
                    track_error = Some(e.to_string());
                }
            }
            false
        })
        .map_err(|e| format!("Tracking stream failed: {e}"))?;

    if let Some(e) = track_error {
        return Err(e);
    }

    // Stream restart includes settings apply + new stream startup overhead
    let stream_restart_ms = t_restart.elapsed().as_secs_f64() * 1000.0 - first_track_frame_ms;
    let _ = stream_restart_ms_settings; // included in stream_restart_ms

    let time_to_track_ms = time_to_track_start.elapsed().as_secs_f64() * 1000.0;
    let _ = t_stream_start; // used indirectly via acq_start

    Ok(IterationTimings {
        fgs_init_ms,
        stream_start_ms,
        acq_frame_times_ms,
        acq_total_ms,
        calibration_ms,
        stream_restart_ms,
        first_track_frame_ms,
        time_to_track_ms,
    })
}

struct Stats {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    p50: f64,
    p90: f64,
    p99: f64,
}

fn compute_stats(samples: &[f64]) -> Stats {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let percentile = |p: f64| -> f64 {
        let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };

    Stats {
        mean,
        std,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        p50: percentile(50.0),
        p90: percentile(90.0),
        p99: percentile(99.0),
    }
}

fn print_row(label: &str, stats: &Stats) {
    println!(
        "{:<22} {:>8.2}ms {:>8.2}ms {:>8.2}ms {:>8.2}ms {:>8.2}ms {:>8.2}ms {:>8.2}ms",
        label, stats.mean, stats.std, stats.min, stats.max, stats.p50, stats.p90, stats.p99
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    let camera_desc = if args.simulator {
        "simulator (synthetic star field)".to_string()
    } else {
        format!("{:?}", args.camera.camera_type)
    };

    println!(
        "FGS Timing Profile ({} iterations, {} acquisition frames)",
        args.iterations, args.acquisition_frames
    );
    println!("Camera: {camera_desc}");
    println!(
        "ROI size: {}px, Min SNR: {:.1}, FWHM: {:.1}",
        args.roi_size, args.min_snr, args.fwhm
    );
    println!("{}", "=".repeat(100));

    // Create camera - either simulator or from CameraArgs
    let mut camera: Box<dyn CameraInterface> = if args.simulator {
        create_simulator_camera(args.exposure.as_duration())
    } else {
        let mut cam = initialize_camera(&args.camera)?;
        cam.set_exposure(args.exposure.as_duration())
            .map_err(|e| format!("Failed to set exposure: {e}"))?;
        cam
    };

    let mut all_timings: Vec<IterationTimings> = Vec::with_capacity(args.iterations);

    for i in 0..args.iterations {
        match run_iteration(camera.as_mut(), &args) {
            Ok(timings) => {
                if args.verbose {
                    let acq_mean = timings.acq_frame_times_ms.iter().sum::<f64>()
                        / timings.acq_frame_times_ms.len() as f64;
                    println!(
                        "  [{:>3}/{}] init={:.2}ms  stream={:.1}ms  acq={:.1}ms ({:.1}ms/frame)  cal={:.1}ms  restart={:.1}ms  track={:.2}ms  total={:.1}ms",
                        i + 1, args.iterations,
                        timings.fgs_init_ms,
                        timings.stream_start_ms,
                        timings.acq_total_ms,
                        acq_mean,
                        timings.calibration_ms,
                        timings.stream_restart_ms,
                        timings.first_track_frame_ms,
                        timings.time_to_track_ms,
                    );
                } else {
                    print!(".");
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                }
                all_timings.push(timings);
            }
            Err(e) => {
                eprintln!("\n  [{}/{}] FAILED: {}", i + 1, args.iterations, e);
            }
        }
    }

    if !args.verbose {
        println!();
    }

    if all_timings.is_empty() {
        eprintln!("No successful iterations");
        return Ok(());
    }

    // Collect samples for each phase
    let fgs_init: Vec<f64> = all_timings.iter().map(|t| t.fgs_init_ms).collect();
    let stream_start: Vec<f64> = all_timings.iter().map(|t| t.stream_start_ms).collect();
    let acq_per_frame: Vec<f64> = all_timings
        .iter()
        .flat_map(|t| t.acq_frame_times_ms.iter().copied())
        .collect();
    let acq_total: Vec<f64> = all_timings.iter().map(|t| t.acq_total_ms).collect();
    let calibration: Vec<f64> = all_timings.iter().map(|t| t.calibration_ms).collect();
    let stream_restart: Vec<f64> = all_timings.iter().map(|t| t.stream_restart_ms).collect();
    let first_track: Vec<f64> = all_timings.iter().map(|t| t.first_track_frame_ms).collect();
    let time_to_track: Vec<f64> = all_timings.iter().map(|t| t.time_to_track_ms).collect();

    // Print results
    println!();
    println!(
        "{:<22} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Phase", "Mean", "Std", "Min", "Max", "P50", "P90", "P99"
    );
    println!("{}", "-".repeat(100));

    print_row("FGS init", &compute_stats(&fgs_init));
    print_row("Stream start", &compute_stats(&stream_start));
    print_row("Acq frame (each)", &compute_stats(&acq_per_frame));
    print_row("Acq total", &compute_stats(&acq_total));
    print_row("Calibration", &compute_stats(&calibration));
    print_row("Stream restart", &compute_stats(&stream_restart));
    print_row("First track frame", &compute_stats(&first_track));
    print_row("Time-to-track", &compute_stats(&time_to_track));

    println!("{}", "-".repeat(100));
    println!(
        "\n{}/{} iterations succeeded",
        all_timings.len(),
        args.iterations
    );

    Ok(())
}

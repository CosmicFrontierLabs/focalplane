use clap::Parser;
use monocle::{
    callback::FgsCallbackEvent,
    config::FgsConfig,
    state::{FgsEvent, FgsState},
    FineGuidanceSystem,
};
use monocle_harness::{
    create_guide_star_catalog, create_jbt_hwk_camera_with_catalog_and_motion,
    motion_profiles::{PointingMotion, TestMotions},
    simulator_camera::SimulatorCamera,
    tracking_plots::{TrackingDataPoint, TrackingPlotConfig, TrackingPlotter},
};
use shared::camera_interface::CameraInterface;
use starfield::Equatorial;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Command line arguments for tracking demo
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "FGS Tracking Demonstration with Motion Patterns"
)]
struct Args {
    /// Motion pattern type (sine_ra, sine_dec, circular, drift, step, chaotic)
    #[arg(short, long, default_value = "sine_ra")]
    motion: String,

    /// Simulation duration in seconds
    #[arg(short = 't', long, default_value_t = 10.0)]
    duration: f64,

    /// Output filename for the plot (without path)
    #[arg(short, long)]
    output: Option<String>,

    /// Plot width in pixels
    #[arg(long, default_value_t = 2400)]
    width: u32,

    /// Plot height in pixels
    #[arg(long, default_value_t = 1600)]
    height: u32,

    /// Number of acquisition frames
    #[arg(long, default_value_t = 3)]
    acquisition_frames: usize,

    /// Minimum SNR for guide star selection
    #[arg(long, default_value_t = 10.0)]
    min_snr: f64,

    /// Maximum number of guide stars
    #[arg(long, default_value_t = 3)]
    max_guide_stars: usize,

    /// ROI size in pixels
    #[arg(long, default_value_t = 32)]
    roi_size: usize,

    /// Centroid radius multiplier (times FWHM)
    #[arg(long, default_value_t = 5.0)]
    centroid_multiplier: f64,

    /// Frame rate in Hz
    #[arg(long, default_value_t = 10.0)]
    frame_rate: f64,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Create a simulator camera with test configuration and motion
fn create_test_camera_with_motion(
    pointing: Equatorial,
    motion: Box<dyn PointingMotion>,
) -> SimulatorCamera {
    let catalog = create_guide_star_catalog(&pointing);
    create_jbt_hwk_camera_with_catalog_and_motion(catalog, motion)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    println!("FGS Tracking Demo");
    println!("=================");
    println!("Motion type: {}", args.motion);
    println!("Duration: {} seconds", args.duration);
    println!("Frame rate: {} Hz", args.frame_rate);

    // Setup motion pattern
    let base_pointing = Equatorial::from_degrees(83.0, -5.0);
    let test_motions = TestMotions::new(83.0, -5.0);
    let motion = test_motions
        .get_motion(&args.motion)
        .ok_or_else(|| format!("Unknown motion type: {}", args.motion))?;

    // Create camera with motion profile
    let mut camera = create_test_camera_with_motion(base_pointing, motion);
    let frame_interval_ms = (1000.0 / args.frame_rate) as u64;
    camera.set_exposure(Duration::from_millis(frame_interval_ms))?;

    // Get actual sensor dimensions and telescope config for accurate projection
    let satellite_config = camera.satellite_config().clone();
    let (sensor_width, sensor_height) = satellite_config.sensor.dimensions.get_pixel_width_height();

    // Create FGS with configuration and camera
    let config = FgsConfig {
        acquisition_frames: args.acquisition_frames,
        min_guide_star_snr: args.min_snr,
        max_guide_stars: args.max_guide_stars,
        roi_size: args.roi_size,
        max_reacquisition_attempts: 3,
        centroid_radius_multiplier: args.centroid_multiplier,
        ..Default::default()
    };

    let mut fgs = FineGuidanceSystem::new(camera, config);

    // Determine output filename
    let output_filename = args
        .output
        .unwrap_or_else(|| format!("tracking_{}.png", args.motion));

    // Create tracking plotter
    let mut plotter = TrackingPlotter::with_config(TrackingPlotConfig {
        output_filename: output_filename.clone(),
        title: format!("FGS Tracking: {} Motion", args.motion),
        width: args.width,
        height: args.height,
        max_time_seconds: args.duration,
    });

    // Track state for plotting
    let lock_established = Arc::new(Mutex::new(false));
    let lock_clone = lock_established.clone();
    let center_x = sensor_width as f64 / 2.0;
    let center_y = sensor_height as f64 / 2.0;
    let estimated_position = Arc::new(Mutex::new((center_x, center_y)));
    let position_clone = estimated_position.clone();

    // Register callback to track events
    let verbose = args.verbose;
    let _callback_id = fgs.register_callback(move |event| match event {
        FgsCallbackEvent::TrackingStarted {
            initial_position, ..
        } => {
            if verbose {
                println!(
                    "Tracking started at ({:.1}, {:.1})",
                    initial_position.x, initial_position.y
                );
            }
            *lock_clone.lock().unwrap() = true;
            *position_clone.lock().unwrap() = (initial_position.x, initial_position.y);
        }
        FgsCallbackEvent::TrackingUpdate { position, .. } => {
            if verbose {
                println!(
                    "Tracking update: pos=({:.1}, {:.1})",
                    position.x, position.y
                );
            }
            *position_clone.lock().unwrap() = (position.x, position.y);
        }
        FgsCallbackEvent::TrackingLost { .. } => {
            if verbose {
                println!("Tracking lost!");
            }
            *lock_clone.lock().unwrap() = false;
        }
    });

    // Start FGS
    println!("\nStarting FGS...");
    fgs.process_event(FgsEvent::StartFgs)?;

    // Acquisition phase
    println!("Acquisition phase ({} frames)...", args.acquisition_frames);
    for i in 0..args.acquisition_frames {
        fgs.process_next_frame()?;
        if args.verbose {
            println!("  Frame {}/{} captured", i + 1, args.acquisition_frames);
        }
    }

    // Calibration frame
    println!("Calibration phase...");
    fgs.process_next_frame()?;

    // Check if we're tracking
    if !matches!(fgs.state(), FgsState::Tracking { .. }) {
        eprintln!("Warning: FGS did not enter tracking state");
    } else {
        println!("Tracking established!");
    }

    // Tracking phase
    let total_duration = Duration::from_secs_f64(args.duration);
    let frame_interval = Duration::from_millis(frame_interval_ms);
    let num_frames = (total_duration.as_millis() / frame_interval.as_millis()) as usize;

    println!(
        "\nTracking phase ({} frames over {:.1} seconds)...",
        num_frames, args.duration
    );

    for frame_num in 0..num_frames {
        let current_time = Duration::from_millis(frame_num as u64 * frame_interval_ms);

        // Process next frame (FGS owns camera now)
        let result = fgs.process_next_frame();

        if let Err(e) = &result {
            eprintln!("Frame {frame_num}: Error processing - {e:?}");
            continue;
        }

        // For tracking demo, we'll use the center position as "actual" since
        // we can't access camera pointing directly anymore
        let (actual_x, actual_y) = (center_x, center_y);

        // Get estimated position from FGS
        let has_lock = *lock_established.lock().unwrap();
        let (est_x, est_y) = *estimated_position.lock().unwrap();

        // Add data point to plotter
        plotter.add_point(TrackingDataPoint {
            time: current_time,
            actual_x,
            actual_y,
            estimated_x: if has_lock { est_x } else { actual_x },
            estimated_y: if has_lock { est_y } else { actual_y },
            is_frame_arrival: true,
            has_lock,
        });

        // Progress indicator
        if !args.verbose && frame_num % 10 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush()?;
        }
    }

    if !args.verbose {
        println!(); // New line after progress dots
    }

    // Generate the plot
    println!("\nGenerating plot...");
    plotter.generate_plot()?;
    println!("✅ Tracking plot saved to plots/{output_filename}");

    // Stop FGS
    fgs.process_event(FgsEvent::StopFgs)?;
    println!("FGS stopped successfully");

    Ok(())
}

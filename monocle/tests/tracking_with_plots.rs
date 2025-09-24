use monocle::{
    simulator_camera::SimulatorCamera,
    test_motions::TestMotions,
    tracking_plots::{TrackingDataPoint, TrackingPlotConfig, TrackingPlotter},
    FgsCallbackEvent, FgsConfig, FgsEvent, FgsState, FineGuidanceSystem,
};
use shared::camera_interface::CameraInterface;
use simulator::hardware::sensor::models as sensor_models;
use simulator::hardware::{SatelliteConfig, TelescopeConfig};
use simulator::units::{Length, LengthExt, Temperature, TemperatureExt};
use starfield::catalogs::binary_catalog::{BinaryCatalog, MinimalStar};
use starfield::Equatorial;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Create a test star catalog for tracking
fn create_test_catalog() -> BinaryCatalog {
    // Create catalog with bright stars suitable for guiding
    let stars = vec![
        MinimalStar::new(1, 83.0, -5.0, 3.0),   // Very bright star
        MinimalStar::new(2, 83.1, -5.1, 4.0),   // Bright star nearby
        MinimalStar::new(3, 83.2, -4.9, 5.0),   // Another bright star
        MinimalStar::new(4, 83.05, -5.15, 6.0), // Dimmer star
        MinimalStar::new(5, 82.95, -5.05, 7.0), // Even dimmer
    ];
    BinaryCatalog::from_stars(stars, "Test catalog for tracking")
}

/// Create a simulator camera with test configuration
fn create_test_camera(pointing: Equatorial) -> SimulatorCamera {
    let telescope = TelescopeConfig::new(
        "Test",
        Length::from_millimeters(80.0),
        Length::from_millimeters(400.0),
        0.9,
    );
    let sensor = sensor_models::IMX455.clone().with_dimensions(1024, 1024);
    let catalog = create_test_catalog();
    let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(0.0));

    SimulatorCamera::new(satellite, catalog, pointing)
}

#[test]
fn test_tracking_with_sinusoidal_motion_plot() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Check if we should generate full plots or run fast tests
    let generate_plots = std::env::var("MONOCLE_GENERATE_PLOTS").is_ok();
    let (duration_secs, plot_width, plot_height) = if generate_plots {
        println!("MONOCLE_GENERATE_PLOTS set - generating full resolution plots");
        (10, 2400, 1600)
    } else {
        println!("Running fast test mode - set MONOCLE_GENERATE_PLOTS=1 for full plots");
        (1, 800, 600) // Just 1 second for fast testing
    };

    println!("Starting tracking test with sinusoidal motion and plotting...");

    // Create FGS with test configuration
    let config = FgsConfig {
        acquisition_frames: 3,
        min_guide_star_snr: 10.0,
        max_guide_stars: 3,
        roi_size: 32,
        max_reacquisition_attempts: 3,
        ..Default::default()
    };

    let mut fgs = FineGuidanceSystem::new(config);

    // Create tracking plotter with resolution based on mode
    let mut plotter = TrackingPlotter::with_config(TrackingPlotConfig {
        output_filename: "tracking_sinusoidal.png".to_string(),
        title: "FGS Tracking: Sinusoidal Motion".to_string(),
        width: plot_width,
        height: plot_height,
        max_time_seconds: duration_secs as f64,
    });

    // Setup motion pattern (sinusoidal in RA)
    let base_pointing = Equatorial::from_degrees(83.0, -5.0);
    let test_motions = TestMotions::new(83.0, -5.0);
    let motion = test_motions
        .get_motion("sine_ra")
        .expect("Failed to get motion");

    // Create camera with initial pointing
    let mut camera = create_test_camera(base_pointing);
    camera.set_exposure(Duration::from_millis(100)).unwrap();

    // Track state for plotting
    let lock_established = Arc::new(Mutex::new(false));
    let lock_clone = lock_established.clone();
    let estimated_position = Arc::new(Mutex::new((512.0, 512.0))); // Center of frame initially
    let position_clone = estimated_position.clone();

    // Register callback to track events
    let _callback_id = fgs.register_callback(move |event| match event {
        FgsCallbackEvent::TrackingStarted {
            initial_position, ..
        } => {
            println!(
                "Tracking started at ({:.1}, {:.1})",
                initial_position.x, initial_position.y
            );
            *lock_clone.lock().unwrap() = true;
            *position_clone.lock().unwrap() = (initial_position.x, initial_position.y);
        }
        FgsCallbackEvent::TrackingUpdate {
            position,
            delta_x,
            delta_y,
            ..
        } => {
            // Use the position estimate directly from FGS
            println!(
                "Tracking update: pos=({:.1}, {:.1}), delta=({:.2}, {:.2})",
                position.x, position.y, delta_x, delta_y
            );

            // The position field contains the current estimated position of the tracked star
            *position_clone.lock().unwrap() = (position.x, position.y);
        }
        FgsCallbackEvent::TrackingLost { .. } => {
            println!("Tracking lost!");
            *lock_clone.lock().unwrap() = false;
        }
    });

    // Start FGS
    fgs.process_event(FgsEvent::StartFgs).unwrap();
    assert!(matches!(fgs.state(), FgsState::Acquiring { .. }));

    // Simulate tracking over time
    let total_duration = Duration::from_secs(duration_secs);
    let frame_interval = Duration::from_millis(100);
    let num_frames = total_duration.as_millis() / frame_interval.as_millis();

    println!(
        "Processing {} frames over {} seconds",
        num_frames,
        total_duration.as_secs()
    );

    for frame_num in 0..num_frames {
        let current_time = Duration::from_millis(frame_num as u64 * 100);

        // Update camera pointing based on motion
        let actual_pointing = motion.get_pointing(current_time);
        camera.set_pointing(actual_pointing).unwrap();

        // Capture frame and process
        let (frame, _metadata) = camera.capture_frame().unwrap();
        let result = fgs.process_frame(frame.view());

        if let Err(e) = &result {
            println!("Frame {}: Error processing - {:?}", frame_num, e);
            continue;
        }

        // Calculate actual position in pixels (center + offset from base pointing)
        let ra_offset_deg = actual_pointing.ra.to_degrees() - base_pointing.ra.to_degrees();
        let dec_offset_deg = actual_pointing.dec.to_degrees() - base_pointing.dec.to_degrees();

        // Convert to pixels (assuming ~1 arcsec/pixel scale for this test)
        let actual_x = 512.0 + ra_offset_deg * 3600.0;
        let actual_y = 512.0 + dec_offset_deg * 3600.0;

        // Get estimated position from FGS
        let has_lock = *lock_established.lock().unwrap();
        let (est_x, est_y) = *estimated_position.lock().unwrap();

        // Debug: Print motion details every 10 frames
        if frame_num % 10 == 0 && has_lock {
            println!(
                "Frame {}: Actual motion = ({:.2}, {:.2}) pixels, Estimated = ({:.2}, {:.2})",
                frame_num,
                actual_x - 512.0,
                actual_y - 512.0,
                est_x - 512.0,
                est_y - 512.0
            );
        }

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

        // Print state periodically
        if frame_num % 10 == 0 {
            println!(
                "Frame {}: State = {:?}, Lock = {}",
                frame_num,
                fgs.state(),
                has_lock
            );
        }
    }

    // Generate the plot
    plotter.generate_plot().expect("Failed to generate plot");
    println!("✅ Tracking plot saved to plots/tracking_sinusoidal.png");

    // Stop FGS
    fgs.process_event(FgsEvent::StopFgs).unwrap();
    assert_eq!(fgs.state(), &FgsState::Idle);
}

#[test]
fn test_tracking_with_circular_motion_plot() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Check if we should generate full plots or run fast tests
    let generate_plots = std::env::var("MONOCLE_GENERATE_PLOTS").is_ok();
    let (duration_secs, plot_width, plot_height) = if generate_plots {
        (15, 2400, 1600)
    } else {
        (1, 800, 600) // Fast mode
    };

    println!("Starting tracking test with circular motion and plotting...");

    // Create FGS
    let config = FgsConfig {
        acquisition_frames: 3,
        min_guide_star_snr: 10.0,
        max_guide_stars: 3,
        roi_size: 32,
        ..Default::default()
    };

    let mut fgs = FineGuidanceSystem::new(config);

    // Create tracking plotter with resolution based on mode
    let mut plotter = TrackingPlotter::with_config(TrackingPlotConfig {
        output_filename: "tracking_circular.png".to_string(),
        title: "FGS Tracking: Circular Motion".to_string(),
        width: plot_width,
        height: plot_height,
        max_time_seconds: duration_secs as f64,
    });

    // Setup circular motion
    let base_pointing = Equatorial::from_degrees(83.0, -5.0);
    let test_motions = TestMotions::new(83.0, -5.0);
    let motion = test_motions
        .get_motion("circular")
        .expect("Failed to get circular motion");

    // Create camera
    let mut camera = create_test_camera(base_pointing);
    camera.set_exposure(Duration::from_millis(100)).unwrap();

    // Tracking state
    let lock_established = Arc::new(Mutex::new(false));
    let lock_clone = lock_established.clone();
    let estimated_position = Arc::new(Mutex::new((512.0, 512.0)));
    let position_clone = estimated_position.clone();

    // Register callback
    let _callback_id = fgs.register_callback(move |event| match event {
        FgsCallbackEvent::TrackingStarted {
            initial_position, ..
        } => {
            *lock_clone.lock().unwrap() = true;
            *position_clone.lock().unwrap() = (initial_position.x, initial_position.y);
        }
        FgsCallbackEvent::TrackingUpdate { position, .. } => {
            *position_clone.lock().unwrap() = (position.x, position.y);
        }
        FgsCallbackEvent::TrackingLost { .. } => {
            *lock_clone.lock().unwrap() = false;
        }
    });

    // Start FGS
    fgs.process_event(FgsEvent::StartFgs).unwrap();

    // Process frames based on duration
    let total_frames = duration_secs * 10; // 10 Hz frame rate
    for frame_num in 0..total_frames {
        let current_time = Duration::from_millis(frame_num as u64 * 100);

        // Update pointing
        let actual_pointing = motion.get_pointing(current_time);
        camera.set_pointing(actual_pointing).unwrap();

        // Capture and process
        let (frame, _) = camera.capture_frame().unwrap();
        let _ = fgs.process_frame(frame.view());

        // Calculate positions
        let ra_offset_deg = actual_pointing.ra.to_degrees() - base_pointing.ra.to_degrees();
        let dec_offset_deg = actual_pointing.dec.to_degrees() - base_pointing.dec.to_degrees();
        let actual_x = 512.0 + ra_offset_deg * 3600.0;
        let actual_y = 512.0 + dec_offset_deg * 3600.0;

        let has_lock = *lock_established.lock().unwrap();
        let (est_x, est_y) = *estimated_position.lock().unwrap();

        // Add plot point
        plotter.add_point(TrackingDataPoint {
            time: current_time,
            actual_x,
            actual_y,
            estimated_x: if has_lock { est_x } else { actual_x },
            estimated_y: if has_lock { est_y } else { actual_y },
            is_frame_arrival: true,
            has_lock,
        });
    }

    // Generate plot
    plotter.generate_plot().expect("Failed to generate plot");
    println!("✅ Tracking plot saved to plots/tracking_circular.png");

    // Stop FGS
    fgs.process_event(FgsEvent::StopFgs).unwrap();
}

#[test]
fn test_tracking_with_chaotic_motion_plot() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Check if we should generate full plots or run fast tests
    let generate_plots = std::env::var("MONOCLE_GENERATE_PLOTS").is_ok();
    let (duration_secs, plot_width, plot_height) = if generate_plots {
        (20, 2400, 1600)
    } else {
        (2, 800, 600) // 2 seconds for chaotic - need slightly more to see pattern
    };

    println!("Starting tracking test with chaotic motion and plotting...");

    // Create FGS with more forgiving settings for chaotic motion
    let config = FgsConfig {
        acquisition_frames: 5,
        min_guide_star_snr: 8.0,
        max_guide_stars: 4,
        roi_size: 48,
        max_reacquisition_attempts: 5,
        ..Default::default()
    };

    let mut fgs = FineGuidanceSystem::new(config);

    // Create tracking plotter with resolution based on mode
    let mut plotter = TrackingPlotter::with_config(TrackingPlotConfig {
        output_filename: "tracking_chaotic.png".to_string(),
        title: "FGS Tracking: Chaotic Motion".to_string(),
        width: plot_width,
        height: plot_height,
        max_time_seconds: duration_secs as f64,
    });

    // Setup chaotic motion
    let base_pointing = Equatorial::from_degrees(83.0, -5.0);
    let test_motions = TestMotions::new(83.0, -5.0);
    let motion = test_motions
        .get_motion("chaotic")
        .expect("Failed to get chaotic motion");

    // Create camera
    let mut camera = create_test_camera(base_pointing);
    camera.set_exposure(Duration::from_millis(100)).unwrap();

    // Tracking state
    let lock_established = Arc::new(Mutex::new(false));
    let lock_clone = lock_established.clone();
    let estimated_position = Arc::new(Mutex::new((512.0, 512.0)));
    let position_clone = estimated_position.clone();

    // Register callback
    let _callback_id = fgs.register_callback(move |event| match event {
        FgsCallbackEvent::TrackingStarted {
            initial_position,
            num_guide_stars,
            ..
        } => {
            println!("Lock established with {} guide stars", num_guide_stars);
            *lock_clone.lock().unwrap() = true;
            *position_clone.lock().unwrap() = (initial_position.x, initial_position.y);
        }
        FgsCallbackEvent::TrackingUpdate {
            position,
            num_stars_used,
            ..
        } => {
            if *num_stars_used > 0 {
                *position_clone.lock().unwrap() = (position.x, position.y);
            }
        }
        FgsCallbackEvent::TrackingLost { reason, .. } => {
            println!("Lock lost: {:?}", reason);
            *lock_clone.lock().unwrap() = false;
        }
    });

    // Start FGS
    fgs.process_event(FgsEvent::StartFgs).unwrap();

    // Process frames
    let total_frames = duration_secs * 10; // 10 Hz frame rate
    for frame_num in 0..total_frames {
        let current_time = Duration::from_millis(frame_num as u64 * 100);

        // Update pointing with chaotic motion
        let actual_pointing = motion.get_pointing(current_time);
        camera.set_pointing(actual_pointing).unwrap();

        // Capture and process
        let (frame, _) = camera.capture_frame().unwrap();
        let _ = fgs.process_frame(frame.view());

        // Calculate positions
        let ra_offset_deg = actual_pointing.ra.to_degrees() - base_pointing.ra.to_degrees();
        let dec_offset_deg = actual_pointing.dec.to_degrees() - base_pointing.dec.to_degrees();
        let actual_x = 512.0 + ra_offset_deg * 3600.0;
        let actual_y = 512.0 + dec_offset_deg * 3600.0;

        let has_lock = *lock_established.lock().unwrap();
        let (est_x, est_y) = *estimated_position.lock().unwrap();

        // Add plot point
        plotter.add_point(TrackingDataPoint {
            time: current_time,
            actual_x,
            actual_y,
            estimated_x: if has_lock { est_x } else { actual_x },
            estimated_y: if has_lock { est_y } else { actual_y },
            is_frame_arrival: true,
            has_lock,
        });

        // Status updates
        if frame_num % 20 == 0 {
            println!(
                "Frame {}: State = {:?}, Lock = {}, Pos = ({:.1}, {:.1})",
                frame_num,
                fgs.state(),
                has_lock,
                actual_x,
                actual_y
            );
        }
    }

    // Generate plot
    plotter.generate_plot().expect("Failed to generate plot");
    println!("✅ Tracking plot saved to plots/tracking_chaotic.png");

    // Stop FGS
    fgs.process_event(FgsEvent::StopFgs).unwrap();
}

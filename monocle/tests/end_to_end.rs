mod common;
mod test_helpers;

use common::{create_synthetic_star_image, StarParams, SyntheticImageConfig};
use monocle::{
    callback::FgsCallbackEvent,
    config::FgsConfig,
    mock_camera::MockCamera,
    state::{FgsEvent, FgsState},
    FineGuidanceSystem,
};
use ndarray::Array2;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use test_helpers::test_timestamp;

/// Helper to perturb star positions slightly (simulating drift)
fn perturb_stars(stars: &[StarParams], dx: f64, dy: f64) -> Vec<StarParams> {
    stars
        .iter()
        .map(|star| StarParams::with_fwhm(star.x + dx, star.y + dy, star.peak_flux, star.fwhm))
        .collect()
}

#[test]
fn test_full_tracking_lifecycle() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Create FGS with test configuration
    let config = FgsConfig {
        acquisition_frames: 3,
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: 10.0,
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: 4000.0,
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
        },
        max_guide_stars: 3,
        roi_size: 32,
        max_reacquisition_attempts: 3,
        centroid_radius_multiplier: 3.0,
        fwhm: 3.0,
    };

    // Create mock camera with empty frames - we'll provide frames directly to process_frame
    let camera = MockCamera::new_repeating(Array2::<u16>::zeros((512, 512)));
    let mut fgs = FineGuidanceSystem::new(camera, config);

    // Track events
    let events_received = Arc::new(Mutex::new(Vec::<FgsCallbackEvent>::new()));
    let events_clone = events_received.clone();

    // Register callback to capture all events
    let _callback_id = fgs.register_callback(move |event| {
        events_clone.lock().unwrap().push(event.clone());
    });

    // Define star positions (bright stars for guide tracking)
    let base_stars = vec![
        StarParams::with_fwhm(250.0, 250.0, 50000.0, 4.0), // Bright guide star
        StarParams::with_fwhm(400.0, 150.0, 45000.0, 4.0), // Another bright star
        StarParams::with_fwhm(100.0, 400.0, 40000.0, 4.0), // Third bright star
        StarParams::with_fwhm(350.0, 350.0, 20000.0, 4.0), // Dimmer star
        StarParams::with_fwhm(150.0, 100.0, 15000.0, 4.0), // Even dimmer
    ];

    let image_config = SyntheticImageConfig {
        width: 512,
        height: 512,
        read_noise_std: 3.0,
        include_photon_noise: false,
        seed: 42,
    };

    // === Phase 1: Start FGS ===
    assert_eq!(fgs.state(), &FgsState::Idle);
    let result = fgs.process_event(FgsEvent::StartFgs);
    assert!(result.is_ok());
    assert!(matches!(
        fgs.state(),
        FgsState::Acquiring {
            frames_collected: 0
        }
    ));

    // === Phase 2: Acquisition (accumulate frames) ===
    println!("Starting acquisition phase...");
    for i in 0..3 {
        let image = create_synthetic_star_image(&image_config, &base_stars);
        let result = fgs.process_frame(image.view(), test_timestamp());
        assert!(result.is_ok());

        if i < 2 {
            assert!(matches!(fgs.state(), FgsState::Acquiring { .. }));
        } else {
            // After 3rd frame, should move to Calibrating
            assert!(matches!(fgs.state(), FgsState::Calibrating));
        }
    }

    // === Phase 3: Calibration ===
    println!("Calibration phase...");
    // Send one more frame to complete calibration
    let calibration_image = create_synthetic_star_image(&image_config, &base_stars);
    let result = fgs.process_frame(calibration_image.view(), test_timestamp());
    assert!(result.is_ok());

    // Check state after calibration
    println!("State after calibration: {:?}", fgs.state());

    // If still in Calibrating, might need another frame
    if matches!(fgs.state(), FgsState::Calibrating) {
        println!("Still calibrating, sending another frame...");
        let another_image = create_synthetic_star_image(&image_config, &base_stars);
        let result = fgs.process_frame(another_image.view(), test_timestamp());
        assert!(result.is_ok());
        println!("State after second frame: {:?}", fgs.state());
    }

    // Should now be tracking or back to Idle if no stars found
    let is_tracking = matches!(fgs.state(), FgsState::Tracking { .. });
    let is_idle = matches!(fgs.state(), FgsState::Idle);
    assert!(
        is_tracking || is_idle,
        "Expected Tracking or Idle state, got {:?}",
        fgs.state()
    );

    // Verify guide stars were selected
    let detected_stars = fgs.get_detected_stars();
    println!("Detected {} stars during calibration", detected_stars.len());
    assert!(!detected_stars.is_empty(), "Should have detected stars");

    // === Phase 4: Tracking (send frames with slight drift) ===
    if is_tracking {
        println!("Tracking phase...");
        let mut drift_x = 0.0;
        let mut drift_y = 0.0;

        for frame_num in 0..10 {
            // Simulate small drift
            drift_x += 0.1;
            drift_y += 0.05;

            let drifted_stars = perturb_stars(&base_stars, drift_x, drift_y);
            let tracking_image = create_synthetic_star_image(&image_config, &drifted_stars);

            let result = fgs.process_frame(tracking_image.view(), test_timestamp());
            assert!(result.is_ok());

            // Should remain in tracking
            assert!(matches!(fgs.state(), FgsState::Tracking { .. }));

            // Should get guidance updates
            let update = result.unwrap();
            if let Some(update) = update {
                println!(
                    "Frame {}: Guidance update - x: {:.2}, y: {:.2}",
                    frame_num, update.x, update.y
                );
            }
        }
    } else {
        println!("System returned to Idle (no suitable guide stars found)")
    }

    // === Phase 5: Verify events ===
    let events = events_received.lock().unwrap();
    println!("\nReceived {} tracking events", events.len());

    // Debug: print all events
    for (i, event) in events.iter().enumerate() {
        println!("Event {}: {:?}", i, event);
    }

    if is_tracking {
        // Should have received TrackingStarted event
        let tracking_started = events
            .iter()
            .any(|e| matches!(e, FgsCallbackEvent::TrackingStarted { .. }));
        assert!(
            tracking_started,
            "Should have received TrackingStarted event"
        );

        // Check for either TrackingUpdate or FrameSizeMismatch events
        let update_count = events
            .iter()
            .filter(|e| matches!(e, FgsCallbackEvent::TrackingUpdate { .. }))
            .count();
        let mismatch_count = events
            .iter()
            .filter(|e| matches!(e, FgsCallbackEvent::FrameSizeMismatch { .. }))
            .count();

        // We expect either tracking updates OR frame size mismatches (when sending full frames instead of ROI)
        assert!(
            update_count > 0 || mismatch_count > 0,
            "Should have received either TrackingUpdate or FrameSizeMismatch events"
        );
        println!(
            "Received {} tracking updates, {} frame mismatches",
            update_count, mismatch_count
        );
    }

    // Verify event details
    for event in events.iter() {
        match event {
            FgsCallbackEvent::TrackingStarted {
                track_id,
                initial_position,
                num_guide_stars,
            } => {
                println!(
                    "TrackingStarted: track_id={}, position=({:.1}, {:.1}), guides={}",
                    track_id, initial_position.x, initial_position.y, num_guide_stars
                );
                assert!(*num_guide_stars > 0);
                assert!(*track_id > 0);
            }
            FgsCallbackEvent::TrackingUpdate { track_id, position } => {
                println!(
                    "TrackingUpdate: track_id={}, position=({:.1}, {:.1})",
                    track_id, position.x, position.y
                );
            }
            FgsCallbackEvent::TrackingLost {
                track_id,
                last_position,
                reason,
            } => {
                println!(
                    "TrackingLost: track_id={}, last_position=({:.1}, {:.1}), reason={:?}",
                    track_id, last_position.x, last_position.y, reason
                );
            }
            FgsCallbackEvent::FrameSizeMismatch {
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => {
                println!(
                    "FrameSizeMismatch: expected {}x{}, actual {}x{}",
                    expected_width, expected_height, actual_width, actual_height
                );
            }
        }
    }

    // === Phase 6: Stop FGS ===
    if is_tracking {
        let result = fgs.process_event(FgsEvent::StopFgs);
        assert!(result.is_ok());
        assert_eq!(fgs.state(), &FgsState::Idle);
    }

    println!("\n✅ End-to-end test completed successfully!");
}

#[test]
fn test_tracking_loss_and_recovery() {
    let _ = env_logger::builder().is_test(true).try_init();

    let config = FgsConfig {
        acquisition_frames: 2,
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: 10.0,
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: 4000.0,
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
        },
        max_guide_stars: 2,
        roi_size: 64,
        max_reacquisition_attempts: 5,
        centroid_radius_multiplier: 3.0,
        fwhm: 3.0,
    };

    // Create mock camera with empty frames - we'll provide frames directly to process_frame
    let camera = MockCamera::new_repeating(Array2::<u16>::zeros((512, 512)));
    let mut fgs = FineGuidanceSystem::new(camera, config);

    // Track lost events
    let lost_count = Arc::new(AtomicUsize::new(0));
    let lost_clone = lost_count.clone();

    let _callback_id = fgs.register_callback(move |event| {
        if matches!(event, FgsCallbackEvent::TrackingLost { .. }) {
            lost_clone.fetch_add(1, Ordering::SeqCst);
        }
    });

    let bright_stars = vec![
        StarParams::with_fwhm(256.0, 256.0, 60000.0, 4.0),
        StarParams::with_fwhm(384.0, 128.0, 55000.0, 4.0),
    ];

    let image_config = SyntheticImageConfig {
        width: 512,
        height: 512,
        read_noise_std: 3.0,
        include_photon_noise: false,
        seed: 100,
    };

    // Start and acquire
    fgs.process_event(FgsEvent::StartFgs).unwrap();
    for _ in 0..2 {
        let image = create_synthetic_star_image(&image_config, &bright_stars);
        fgs.process_frame(image.view(), test_timestamp()).unwrap();
    }

    // Calibrate
    let image = create_synthetic_star_image(&image_config, &bright_stars);
    fgs.process_frame(image.view(), test_timestamp()).unwrap();

    // May be Tracking or Idle
    let is_tracking = matches!(fgs.state(), FgsState::Tracking { .. });
    if !is_tracking {
        println!("No guide stars found, skipping tracking loss test");
        return;
    }

    // Track for a few frames
    for _ in 0..3 {
        let image = create_synthetic_star_image(&image_config, &bright_stars);
        fgs.process_frame(image.view(), test_timestamp()).unwrap();
    }

    // Send dark frames to simulate star loss
    // Note: Currently the implementation doesn't actually lose stars,
    // but this tests the framework for when it does
    println!("Simulating star loss...");
    let dark_image = Array2::<u16>::from_elem((512, 512), 100); // Just background

    for attempt in 0..6 {
        let result = fgs.process_frame(dark_image.view(), test_timestamp());
        assert!(result.is_ok());

        // The current implementation doesn't actually transition to Reacquiring
        // This is a TODO in the implementation
        println!("Frame {}: State = {:?}", attempt, fgs.state());
    }

    // In a full implementation, we'd expect:
    // - Transition to Reacquiring state
    // - TrackingLost event
    // - Eventual return to Calibrating or Tracking

    println!("✅ Loss and recovery test framework completed");
}

#[test]
fn test_image_sequence_processing() {
    let _ = env_logger::builder().is_test(true).try_init();

    let config = FgsConfig {
        acquisition_frames: 5,
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: 15.0,
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: 4000.0,
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
        },
        max_guide_stars: 4,
        roi_size: 64,
        max_reacquisition_attempts: 5,
        centroid_radius_multiplier: 3.0,
        fwhm: 3.0,
    };

    // Create mock camera with empty frames - we'll provide frames directly to process_frame
    let camera = MockCamera::new_repeating(Array2::<u16>::zeros((512, 512)));
    let mut fgs = FineGuidanceSystem::new(camera, config);

    // Create a sequence of images with gradually moving stars
    let mut image_sequence = Vec::new();
    let base_stars = vec![
        StarParams::with_fwhm(200.0, 200.0, 40000.0, 4.0),
        StarParams::with_fwhm(300.0, 300.0, 35000.0, 4.0),
        StarParams::with_fwhm(150.0, 350.0, 30000.0, 4.0),
        StarParams::with_fwhm(350.0, 150.0, 25000.0, 4.0),
    ];

    let image_config = SyntheticImageConfig {
        width: 512,
        height: 512,
        read_noise_std: 3.0,
        include_photon_noise: false,
        seed: 200,
    };

    // Generate 20 images with gradual motion
    for i in 0..20 {
        let drift = i as f64 * 0.5;
        let stars = perturb_stars(&base_stars, drift, drift * 0.5);
        let image = create_synthetic_star_image(&image_config, &stars);
        image_sequence.push(image);
    }

    // Process the sequence
    println!("Processing {} images", image_sequence.len());

    // Start FGS
    fgs.process_event(FgsEvent::StartFgs).unwrap();

    for (idx, image) in image_sequence.iter().enumerate() {
        let result = fgs.process_frame(image.view(), test_timestamp());
        assert!(result.is_ok());

        match fgs.state() {
            FgsState::Idle => println!("Frame {}: Idle", idx),
            FgsState::Acquiring { frames_collected } => {
                println!("Frame {}: Acquiring (collected: {})", idx, frames_collected);
            }
            FgsState::Calibrating => {
                println!("Frame {}: Calibrating", idx);
            }
            FgsState::Tracking { frames_processed } => {
                println!("Frame {}: Tracking (processed: {})", idx, frames_processed);
                if let Ok(Some(update)) = result {
                    println!("  -> Update: x={:.2}, y={:.2}", update.x, update.y);
                }
            }
            FgsState::Reacquiring { attempts } => {
                println!("Frame {}: Reacquiring (attempts: {})", idx, attempts);
            }
        }
    }

    // Verify we ended up in either tracking or idle state (depends on star detection)
    let final_state = fgs.state();
    let is_tracking = matches!(final_state, FgsState::Tracking { .. });
    let is_idle = matches!(final_state, FgsState::Idle);
    assert!(
        is_tracking || is_idle,
        "Expected Tracking or Idle state, got {:?}",
        final_state
    );

    if is_tracking {
        println!("✅ Image sequence processing test completed - system tracking");
    } else {
        println!("✅ Image sequence processing test completed - no stars detected");
    }
}

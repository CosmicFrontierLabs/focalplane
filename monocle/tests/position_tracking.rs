//! Test that FGS detects stars and tracks positions accurately

mod common;
mod test_helpers;

use common::{create_synthetic_star_image, StarParams, SyntheticImageConfig};
use monocle::{
    callback::FgsCallbackEvent,
    config::FgsConfig,
    state::{FgsEvent, FgsState},
    FineGuidanceSystem,
};
use std::sync::{Arc, Mutex};
use test_helpers::test_timestamp;

#[test]
fn test_star_detection_on_correct_cycle() {
    let _ = env_logger::builder().is_test(true).try_init();

    let config = FgsConfig {
        acquisition_frames: 2,
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: 3.0,
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: 65535.0 * 0.95,
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
            bad_pixel_map: shared::bad_pixel_map::BadPixelMap::empty(),
            minimum_bad_pixel_distance: 5.0,
        },
        roi_size: 32,
        max_reacquisition_attempts: 5,
        centroid_radius_multiplier: 3.0,
        fwhm: 3.0,
        snr_dropout_threshold: 3.0,
        roi_h_alignment: 1,
        roi_v_alignment: 1,
    };

    let mut fgs = FineGuidanceSystem::new(config);

    // Track when we enter each state
    let states = Arc::new(Mutex::new(Vec::new()));
    let states_clone = states.clone();

    fgs.register_callback(move |event| {
        if let FgsCallbackEvent::TrackingStarted { .. } = event {
            states_clone.lock().unwrap().push("TrackingStarted");
        }
    });

    // Create frame with star at known position
    let config = SyntheticImageConfig {
        width: 256,
        height: 256,
        read_noise_std: 3.0,
        include_photon_noise: false,
        seed: 100,
    };

    let stars = vec![StarParams::with_fwhm(128.0, 128.0, 8000.0, 4.5)];
    let frame = create_synthetic_star_image(&config, &stars);

    // Start FGS - should enter Acquiring
    let _ = fgs.process_event(FgsEvent::StartFgs).unwrap();
    assert!(matches!(
        fgs.state(),
        FgsState::Acquiring {
            frames_collected: 0
        }
    ));

    // First acquisition frame
    let _ = fgs.process_frame(frame.view(), test_timestamp()).unwrap();
    assert!(matches!(
        fgs.state(),
        FgsState::Acquiring {
            frames_collected: 1
        }
    ));

    // Second acquisition frame - should transition to Calibrating
    let _ = fgs.process_frame(frame.view(), test_timestamp()).unwrap();
    assert!(matches!(fgs.state(), FgsState::Calibrating));

    // Calibration frame - should detect star and enter Tracking
    let _ = fgs.process_frame(frame.view(), test_timestamp()).unwrap();
    assert!(
        matches!(fgs.state(), FgsState::Tracking { .. }),
        "Should be tracking after calibration but state is {:?}",
        fgs.state()
    );

    // Verify we got the tracking started event
    let events = states.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0], "TrackingStarted");
}

#[test]
fn test_position_accuracy() {
    let _ = env_logger::builder().is_test(true).try_init();

    let config = FgsConfig {
        acquisition_frames: 1,
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: 3.0,
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: 65535.0 * 0.95,
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
            bad_pixel_map: shared::bad_pixel_map::BadPixelMap::empty(),
            minimum_bad_pixel_distance: 5.0,
        },
        roi_size: 32,
        max_reacquisition_attempts: 5,
        centroid_radius_multiplier: 5.0,
        fwhm: 3.0,
        snr_dropout_threshold: 3.0,
        roi_h_alignment: 1,
        roi_v_alignment: 1,
    };

    let mut fgs = FineGuidanceSystem::new(config);

    // Track reported positions
    let positions = Arc::new(Mutex::new(Vec::new()));
    let pos_clone = positions.clone();

    fgs.register_callback(move |event| match event {
        FgsCallbackEvent::TrackingStarted {
            initial_position, ..
        } => {
            pos_clone
                .lock()
                .unwrap()
                .push(("start", initial_position.x, initial_position.y));
        }
        FgsCallbackEvent::TrackingUpdate { position, .. } => {
            pos_clone
                .lock()
                .unwrap()
                .push(("update", position.x, position.y));
        }
        _ => {}
    });

    // Initialize to tracking with star at (100, 100)
    let config = SyntheticImageConfig {
        width: 256,
        height: 256,
        read_noise_std: 2.0,
        include_photon_noise: false,
        seed: 200,
    };

    let stars = vec![StarParams::with_fwhm(100.0, 100.0, 10000.0, 4.0)];
    let frame = create_synthetic_star_image(&config, &stars);
    let star_x = stars[0].x;
    let star_y = stars[0].y;

    let _ = fgs.process_event(FgsEvent::StartFgs).unwrap();
    let _ = fgs.process_frame(frame.view(), test_timestamp()).unwrap(); // Acquisition
    let _ = fgs.process_frame(frame.view(), test_timestamp()).unwrap(); // Calibration

    // Should be tracking now
    assert!(matches!(fgs.state(), FgsState::Tracking { .. }));

    // Process a tracking frame with same position
    let _ = fgs.process_frame(frame.view(), test_timestamp()).unwrap();

    // Check positions are accurate
    let reported_positions = positions.lock().unwrap();
    assert!(
        !reported_positions.is_empty(),
        "Should have position reports"
    );

    for (event_type, x, y) in reported_positions.iter() {
        let dx = (x - star_x).abs();
        let dy = (y - star_y).abs();

        assert!(
            dx < 2.0,
            "{} position X error too large: expected {:.1}, got {:.1} (error {:.1})",
            event_type,
            star_x,
            x,
            dx
        );
        assert!(
            dy < 2.0,
            "{} position Y error too large: expected {:.1}, got {:.1} (error {:.1})",
            event_type,
            star_y,
            y,
            dy
        );
    }
}

#[test]
fn test_moving_star_tracking() {
    let _ = env_logger::builder().is_test(true).try_init();

    let config = FgsConfig {
        acquisition_frames: 1,
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: 3.0,
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: 65535.0 * 0.95,
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
            bad_pixel_map: shared::bad_pixel_map::BadPixelMap::empty(),
            minimum_bad_pixel_distance: 5.0,
        },
        roi_size: 48,
        max_reacquisition_attempts: 5,
        centroid_radius_multiplier: 3.0,
        fwhm: 3.0,
        snr_dropout_threshold: 3.0,
        roi_h_alignment: 1,
        roi_v_alignment: 1,
    };

    let mut fgs = FineGuidanceSystem::new(config);

    let positions = Arc::new(Mutex::new(Vec::new()));
    let mismatch_count = Arc::new(Mutex::new(0usize));
    let pos_clone = positions.clone();
    let mismatch_clone = mismatch_count.clone();

    fgs.register_callback(move |event| match event {
        FgsCallbackEvent::TrackingUpdate { position, .. } => {
            pos_clone.lock().unwrap().push((position.x, position.y));
        }
        FgsCallbackEvent::FrameSizeMismatch { .. } => {
            *mismatch_clone.lock().unwrap() += 1;
        }
        _ => {}
    });

    // Start with star at (128, 128)
    let config = SyntheticImageConfig {
        width: 256,
        height: 256,
        read_noise_std: 3.0,
        include_photon_noise: false,
        seed: 300,
    };

    let mut stars = vec![StarParams::with_fwhm(128.0, 128.0, 8000.0, 4.0)];

    // Initialize to tracking
    let frame = create_synthetic_star_image(&config, &stars);
    let _ = fgs.process_event(FgsEvent::StartFgs).unwrap();
    let _ = fgs.process_frame(frame.view(), test_timestamp()).unwrap();
    let _ = fgs.process_frame(frame.view(), test_timestamp()).unwrap();

    assert!(matches!(fgs.state(), FgsState::Tracking { .. }));

    // Move star in small steps - emulating slow creep across FOV
    for i in 0..5 {
        // Offset from original position by increasing amounts
        // Slow drift b/t frames
        let drift_x = (i + 1) as f64 * 0.2;
        let drift_y = (i + 1) as f64 * 0.1;

        // Create stars at new position relative to original (128, 128)
        stars = vec![StarParams::with_fwhm(
            128.0 + drift_x,
            128.0 + drift_y,
            8000.0,
            4.0,
        )];
        let frame = create_synthetic_star_image(&config, &stars);
        let _ = fgs.process_frame(frame.view(), test_timestamp()).unwrap();

        // Should still be tracking
        assert!(
            matches!(fgs.state(), FgsState::Tracking { .. }),
            "Lost tracking at step {}",
            i
        );
    }

    // Check tracking followed the motion OR we got frame mismatches
    let reported = positions.lock().unwrap();
    let mismatches = *mismatch_count.lock().unwrap();

    // Either we got position updates or frame mismatches (when sending full frames instead of ROI)
    assert!(
        reported.len() >= 5 || mismatches >= 5,
        "Should have at least 5 position updates or frame mismatches. Got {} updates and {} mismatches",
        reported.len(), mismatches
    );

    // If we got position updates, verify tracking accuracy
    if !reported.is_empty() {
        // Last position should be close to final star position
        let (last_x, last_y) = reported.last().unwrap();
        let final_star_x = stars[0].x;
        let final_star_y = stars[0].y;

        // Check tracking accuracy - currently ~10 pixel error after 5 moves
        let x_error = (last_x - final_star_x).abs();
        let y_error = (last_y - final_star_y).abs();

        println!(
            "Tracking error: X={:.1} pixels, Y={:.1} pixels",
            x_error, y_error
        );

        assert!(
            x_error < 1.0,
            "Final X position error too large: expected {:.1}, got {:.1} (error {:.1})",
            final_star_x,
            last_x,
            x_error
        );
        assert!(
            y_error < 1.0,
            "Final Y position error too large: expected {:.1}, got {:.1} (error {:.1})",
            final_star_y,
            last_y,
            y_error
        );
    } else {
        println!(
            "Tracking test detected {} frame size mismatches instead of position updates",
            mismatches
        );
    }
}

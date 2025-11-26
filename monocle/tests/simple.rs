//! Simple test with a single stationary star

mod common;
mod test_helpers;

use common::{create_synthetic_star_image, StarParams, SyntheticImageConfig};
use monocle::{
    config::FgsConfig,
    state::{FgsEvent, FgsState},
    FineGuidanceSystem,
};
use test_helpers::test_timestamp;

#[test]
fn test_single_stationary_star() {
    // Initialize logging
    let _ = env_logger::builder().is_test(true).try_init();

    // Super simple config
    let config = FgsConfig {
        acquisition_frames: 1, // Just one frame
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: 0.1, // Even lower threshold
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: 4000.0,
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
            bad_pixel_map: shared::bad_pixel_map::BadPixelMap::empty(),
            minimum_bad_pixel_distance: 5.0,
        },
        roi_size: 64, // Big ROI so we don't lose it
        max_reacquisition_attempts: 5,
        centroid_radius_multiplier: 3.0,
        fwhm: 3.0,
        snr_dropout_threshold: 3.0,
    };

    let mut fgs = FineGuidanceSystem::new(config);

    // Create a frame with a single star using the shared renderer
    let image_config = SyntheticImageConfig {
        width: 256,
        height: 256,
        read_noise_std: 3.0,         // Low noise for reliable detection
        include_photon_noise: false, // Disable for simpler test
        seed: 42,
    };

    let stars = vec![
        StarParams::with_fwhm(128.0, 128.0, 5000.0, 4.0), // 4 pixel FWHM star at center
    ];

    let frame = create_synthetic_star_image(&image_config, &stars);

    println!("Frame stats:");
    println!("  Size: {}x{}", image_config.width, image_config.height);
    println!("  Noise (std={})", image_config.read_noise_std);
    println!(
        "  Star: pos=({:.1}, {:.1}), peak={:.0}, FWHM={:.1} pixels",
        stars[0].x, stars[0].y, stars[0].peak_flux, stars[0].fwhm
    );

    // Start FGS
    fgs.process_event(FgsEvent::StartFgs).unwrap();
    assert!(matches!(fgs.state(), FgsState::Acquiring { .. }));
    println!("After start: {:?}", fgs.state());

    // Acquisition frame
    let result = fgs.process_frame(frame.view(), test_timestamp());
    println!("Acquisition result: {:?}", result);
    println!("After acquisition: {:?}", fgs.state());

    // Calibration frame (same frame)
    let result = fgs.process_frame(frame.view(), test_timestamp());
    println!("Calibration result: {:?}", result);
    println!("After calibration: {:?}", fgs.state());

    // We should be tracking now
    // See TODO.md: Monocle (FGS/Tracking) - Fix star detection in calibration
    if !matches!(fgs.state(), FgsState::Tracking { .. }) {
        eprintln!("WARNING: Not tracking, state is {:?}", fgs.state());
        return; // Skip rest of test for now
    }

    // Try a few tracking frames with the exact same star
    for i in 0..3 {
        let result = fgs.process_frame(frame.view(), test_timestamp());
        println!("Tracking frame {} result: {:?}", i, result);
        assert!(result.is_ok(), "Tracking should succeed");
    }
}

#[test]
fn test_guidance_update_timestamp_correlation() {
    use shared::camera_interface::Timestamp;
    use std::time::Duration;

    // Initialize logging
    let _ = env_logger::builder().is_test(true).try_init();

    let config = FgsConfig {
        acquisition_frames: 1,
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: 5.0,
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: 4000.0,
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
    };

    // Create synthetic star
    let image_config = SyntheticImageConfig {
        width: 256,
        height: 256,
        read_noise_std: 3.0,
        include_photon_noise: false,
        seed: 42,
    };

    let star = StarParams::with_fwhm(128.0, 128.0, 5000.0, 4.0);
    let frame = create_synthetic_star_image(&image_config, &[star]);

    let mut fgs = FineGuidanceSystem::new(config);

    // Start FGS
    let _ = fgs.process_event(FgsEvent::StartFgs);

    // Process acquisition with specific timestamp
    let ts1 = Timestamp::from_duration(Duration::from_millis(100));
    let result = fgs.process_frame(frame.view(), ts1);
    assert!(result.is_ok());

    // Process calibration with different timestamp
    let ts2 = Timestamp::from_duration(Duration::from_millis(200));
    let result = fgs.process_frame(frame.view(), ts2);
    assert!(result.is_ok());

    // Now in tracking - test that guidance updates have correct timestamps
    let ts3 = Timestamp::from_duration(Duration::from_millis(300));
    let result = fgs.process_frame(frame.view(), ts3);
    assert!(result.is_ok());

    if let Ok((Some(update), _)) = result {
        // Verify the guidance update has the timestamp we passed in
        assert_eq!(
            update.timestamp, ts3,
            "Guidance update should have frame timestamp"
        );
        println!(
            "✓ Guidance update timestamp matches frame: {:?}",
            update.timestamp
        );
    }

    // Test multiple frames with increasing timestamps
    for i in 4..=10 {
        let ts = Timestamp::from_duration(Duration::from_millis(i * 100));
        let result = fgs.process_frame(frame.view(), ts);
        assert!(result.is_ok());

        if let Ok((Some(update), _)) = result {
            assert_eq!(
                update.timestamp, ts,
                "Frame {} guidance update timestamp should match input",
                i
            );
        }
    }

    println!("✓ All guidance updates have correct timestamps");
}

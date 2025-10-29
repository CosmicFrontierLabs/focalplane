//! Minimal integration test to verify monocle can use simulator components
//!
//! This is a simplified test that just verifies the basic integration works

use monocle::{
    config::FgsConfig,
    state::{FgsEvent, FgsState},
    FineGuidanceSystem,
};
use monocle_harness::{create_jbt_hwk_camera, create_jbt_hwk_test_satellite};
use simulator::photometry::zodiacal::SolarAngularCoordinates;

#[test]
fn test_basic_setup() {
    // Use standardized test configuration
    let satellite = create_jbt_hwk_test_satellite();

    // Verify the configuration is created correctly
    assert_eq!(satellite.telescope.name, "Cosmic Frontier JBT .5m");
    assert_eq!(satellite.sensor.name, "HWK4123");
    assert_eq!(
        satellite.sensor.dimensions.get_pixel_width_height(),
        (512, 512)
    );

    // Verify solar angles creation works
    let _solar_angles = SolarAngularCoordinates::new(90.0_f64.to_radians(), 30.0_f64.to_radians());

    // Verify FGS can be created
    let fgs_config = FgsConfig {
        acquisition_frames: 3,
        filters: monocle::config::GuideStarFilters {
            detection_threshold_sigma: 5.0,
            snr_min: 10.0,
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
    };

    let camera = create_jbt_hwk_camera();
    let mut fgs = FineGuidanceSystem::new(camera, fgs_config);

    // Verify FGS starts in idle state
    assert!(matches!(fgs.state(), FgsState::Idle));

    // Test state transitions
    assert!(fgs.process_event(FgsEvent::StartFgs).is_ok());
    assert!(matches!(fgs.state(), FgsState::Acquiring { .. }));
}

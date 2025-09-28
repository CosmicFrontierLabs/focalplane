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
        min_guide_star_snr: 10.0,
        max_guide_stars: 3,
        ..Default::default()
    };

    let camera = create_jbt_hwk_camera();
    let mut fgs = FineGuidanceSystem::new(camera, fgs_config);

    // Verify FGS starts in idle state
    assert!(matches!(fgs.state(), FgsState::Idle));

    // Test state transitions
    assert!(fgs.process_event(FgsEvent::StartFgs).is_ok());
    assert!(matches!(fgs.state(), FgsState::Acquiring { .. }));
}

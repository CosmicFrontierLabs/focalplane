//! Minimal integration test to verify monocle can use simulator components
//!
//! This is a simplified test that just verifies the basic integration works

use monocle::{FgsConfig, FgsEvent, FineGuidanceSystem};
use simulator::hardware::sensor::models as sensor_models;
use simulator::hardware::{SatelliteConfig, TelescopeConfig};
use simulator::photometry::zodiacal::SolarAngularCoordinates;
use simulator::units::{LengthExt, Temperature, TemperatureExt};
use starfield::catalogs::binary_catalog::BinaryCatalog;

#[test]
fn test_basic_setup() {
    // Verify we can create telescope config
    let aperture = simulator::units::Length::from_millimeters(80.0);
    let focal_length = simulator::units::Length::from_millimeters(400.0);
    let telescope = TelescopeConfig::new("Test Telescope", aperture, focal_length, 0.9);

    // Verify we can get sensor model
    let sensor = sensor_models::IMX455.clone();
    let temperature = Temperature::from_celsius(0.0);

    // Verify we can create satellite config
    let _satellite = SatelliteConfig::new(telescope, sensor, temperature);

    // Verify solar angles creation works
    let _solar_angles = SolarAngularCoordinates::new(90.0_f64.to_radians(), 30.0_f64.to_radians());

    // Verify FGS can be created
    let fgs_config = FgsConfig {
        acquisition_frames: 3,
        min_guide_star_snr: 10.0,
        max_guide_stars: 3,
        ..Default::default()
    };

    let mut fgs = FineGuidanceSystem::new(fgs_config);

    // Verify basic state transitions work
    assert_eq!(fgs.state(), &monocle::FgsState::Idle);
    let result = fgs.process_event(FgsEvent::StartFgs);
    assert!(result.is_ok());

    println!("✅ Basic integration test passed!");
}

#[test]
fn test_catalog_loading() {
    // Try to load catalog if available
    match BinaryCatalog::load("cats/test.bin") {
        Ok(catalog) => {
            println!("Loaded {} stars from catalog", catalog.len());

            // Verify catalog has stars
            assert!(catalog.len() > 0, "Catalog should have stars");

            println!("✅ Catalog loading test passed!");
        }
        Err(_) => {
            println!("⚠️ Catalog not available at cats/test.bin, skipping catalog test");
        }
    }
}

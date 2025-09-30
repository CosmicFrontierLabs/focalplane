//! Helper functions for monocle harness tests
//!
//! Provides common utilities for building test configurations including
//! standard satellites, cameras, and catalogs.

use crate::motion_profiles::StaticPointing;
use crate::SimulatorCamera;
use simulator::hardware::{
    sensor::models::HWK4123, telescope::models::COSMIC_FRONTIER_JBT_50CM, SatelliteConfig,
};
use simulator::units::{Temperature, TemperatureExt};
use starfield::catalogs::binary_catalog::{BinaryCatalog, MinimalStar};
use starfield::Equatorial;
use std::sync::Arc;

/// Creates a standard test satellite configuration using JBT 50cm telescope
/// and HWK4123 sensor reduced to 512x512 pixels.
///
/// This configuration is useful for fast tests that need realistic optics
/// but don't require full sensor resolution.
///
/// # Returns
/// * `SatelliteConfig` - Configured satellite with JBT 50cm and 512x512 HWK4123
pub fn create_jbt_hwk_test_satellite() -> SatelliteConfig {
    // JBT 50cm telescope (48.5cm aperture, f/12.3)
    let telescope = COSMIC_FRONTIER_JBT_50CM.clone();

    // HWK4123 sensor reduced to 512x512 for faster tests
    let sensor = HWK4123.clone().with_dimensions(512, 512);

    // Standard operating temperature
    let temperature = Temperature::from_celsius(-10.0);

    SatelliteConfig::new(telescope, sensor, temperature)
}

/// Creates a standard test SimulatorCamera with JBT 50cm telescope,
/// 512x512 HWK4123 sensor, and a simple test catalog.
///
/// # Returns
/// * `SimulatorCamera` - Configured camera ready for testing
pub fn create_jbt_hwk_camera() -> SimulatorCamera {
    let satellite = create_jbt_hwk_test_satellite();
    let catalog = Arc::new(create_simple_test_catalog());
    let motion = Box::new(StaticPointing::new(0.0, 0.0));

    SimulatorCamera::new(satellite, catalog, motion)
}

/// Creates a test SimulatorCamera with custom catalog
///
/// # Arguments
/// * `catalog` - Custom star catalog to use (wrapped in Arc)
///
/// # Returns
/// * `SimulatorCamera` - Configured camera with custom catalog
pub fn create_jbt_hwk_camera_with_catalog(catalog: Arc<BinaryCatalog>) -> SimulatorCamera {
    let satellite = create_jbt_hwk_test_satellite();
    let motion = Box::new(StaticPointing::new(0.0, 0.0));
    SimulatorCamera::new(satellite, catalog, motion)
}

/// Creates a test SimulatorCamera with custom catalog and motion
///
/// # Arguments
/// * `catalog` - Custom star catalog to use (wrapped in Arc)
/// * `motion` - Pointing motion profile
///
/// # Returns
/// * `SimulatorCamera` - Configured camera with custom catalog and motion
pub fn create_jbt_hwk_camera_with_catalog_and_motion(
    catalog: Arc<BinaryCatalog>,
    motion: Box<dyn crate::motion_profiles::PointingMotion>,
) -> SimulatorCamera {
    let satellite = create_jbt_hwk_test_satellite();
    SimulatorCamera::new(satellite, catalog, motion)
}

/// Creates a catalog with a single star at specified position
///
/// # Arguments
/// * `ra_deg` - Right ascension in degrees
/// * `dec_deg` - Declination in degrees
/// * `magnitude` - Star magnitude
///
/// # Returns
/// * `BinaryCatalog` - Catalog with single star
pub fn create_single_star_catalog(ra_deg: f64, dec_deg: f64, magnitude: f64) -> BinaryCatalog {
    let star = MinimalStar::new(1, ra_deg, dec_deg, magnitude);
    BinaryCatalog::from_stars(vec![star], "Single star catalog")
}

/// Creates a simple test catalog with a few bright stars
///
/// # Returns
/// * `BinaryCatalog` - Test catalog with 5 stars of varying brightness
pub fn create_simple_test_catalog() -> BinaryCatalog {
    let stars = vec![
        MinimalStar::new(1, 0.0, 0.0, 5.0),  // Magnitude 5 star at origin
        MinimalStar::new(2, 0.1, 0.0, 6.0),  // Magnitude 6 star
        MinimalStar::new(3, 0.0, 0.1, 7.0),  // Magnitude 7 star
        MinimalStar::new(4, -0.1, 0.0, 8.0), // Magnitude 8 star
        MinimalStar::new(5, 0.0, -0.1, 9.0), // Magnitude 9 star
    ];

    BinaryCatalog::from_stars(stars, "Simple test catalog")
}

/// Creates a test catalog suitable for FGS testing with guide stars
///
/// # Arguments
/// * `pointing` - Telescope pointing direction
///
/// # Returns
/// * `BinaryCatalog` - Catalog with bright guide stars near the pointing
pub fn create_guide_star_catalog(pointing: &Equatorial) -> BinaryCatalog {
    // Convert from radians to degrees for catalog
    let ra_deg = pointing.ra * 180.0 / std::f64::consts::PI;
    let dec_deg = pointing.dec * 180.0 / std::f64::consts::PI;

    // Create bright guide stars clustered around the pointing
    let stars = vec![
        MinimalStar::new(1, ra_deg, dec_deg, 3.0), // Very bright central star
        MinimalStar::new(2, ra_deg + 0.01, dec_deg, 4.0), // Bright nearby star
        MinimalStar::new(3, ra_deg - 0.01, dec_deg, 4.5), // Another bright star
        MinimalStar::new(4, ra_deg, dec_deg + 0.01, 5.0), // Medium bright star
        MinimalStar::new(5, ra_deg, dec_deg - 0.01, 5.5), // Medium star
        // Add some dimmer background stars
        MinimalStar::new(6, ra_deg + 0.02, dec_deg + 0.02, 8.0),
        MinimalStar::new(7, ra_deg - 0.02, dec_deg + 0.02, 9.0),
        MinimalStar::new(8, ra_deg + 0.02, dec_deg - 0.02, 10.0),
        MinimalStar::new(9, ra_deg - 0.02, dec_deg - 0.02, 11.0),
    ];

    BinaryCatalog::from_stars(stars, "Guide star test catalog")
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_create_jbt_hwk_satellite() {
        let satellite = create_jbt_hwk_test_satellite();

        // Check telescope name
        assert_eq!(satellite.telescope.name, "Cosmic Frontier JBT .5m");

        // Check sensor name
        assert_eq!(satellite.sensor.name, "HWK4123");

        // Check pixel dimensions
        assert_eq!(
            satellite.sensor.dimensions.get_pixel_width_height(),
            (512, 512)
        );
    }

    #[test]
    fn test_create_jbt_hwk_camera() {
        let camera = create_jbt_hwk_camera();

        // Just verify it creates successfully - camera is tested elsewhere
        // Check the underlying satellite config matches
        let satellite = camera.satellite_config();
        assert_eq!(satellite.telescope.name, "Cosmic Frontier JBT .5m");
        assert_eq!(satellite.sensor.name, "HWK4123");
    }

    #[test]
    fn test_create_simple_catalog() {
        let catalog = create_simple_test_catalog();

        // Check star count
        let stars: Vec<_> = catalog.stars().iter().collect();
        assert_eq!(stars.len(), 5);

        // Check first star
        assert_eq!(stars[0].id, 1);
        assert_relative_eq!(stars[0].position.ra, 0.0, epsilon = 1e-6);
        assert_relative_eq!(stars[0].position.dec, 0.0, epsilon = 1e-6);
        assert_relative_eq!(stars[0].magnitude, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_create_single_star_catalog() {
        let catalog = create_single_star_catalog(45.0, 30.0, 7.5);

        // Check star count
        let stars: Vec<_> = catalog.stars().iter().collect();
        assert_eq!(stars.len(), 1);

        // Check star properties
        // Note: MinimalStar stores position as Equatorial, which is in radians
        assert_eq!(stars[0].id, 1);
        assert_relative_eq!(stars[0].position.ra, 45.0f64.to_radians(), epsilon = 1e-6);
        assert_relative_eq!(stars[0].position.dec, 30.0f64.to_radians(), epsilon = 1e-6);
        assert_relative_eq!(stars[0].magnitude, 7.5, epsilon = 1e-6);
    }

    #[test]
    fn test_create_guide_star_catalog() {
        let pointing = Equatorial::from_degrees(83.0, -5.0);
        let catalog = create_guide_star_catalog(&pointing);

        // Check star count
        let stars: Vec<_> = catalog.stars().iter().collect();
        assert_eq!(stars.len(), 9);

        // Check brightest star is at pointing
        // Note: MinimalStar stores position as Equatorial, which is in radians
        let expected_ra = 83.0f64.to_radians();
        let expected_dec = (-5.0f64).to_radians();

        assert_relative_eq!(stars[0].position.ra, expected_ra, epsilon = 1e-6);
        assert_relative_eq!(stars[0].position.dec, expected_dec, epsilon = 1e-6);
        assert_relative_eq!(stars[0].magnitude, 3.0, epsilon = 1e-6);
    }
}

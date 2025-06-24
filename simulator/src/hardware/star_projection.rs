//! Star projection utilities for telescope simulation
//!
//! This module provides functions to calculate the field of view, projected area,
//! and other parameters needed to simulate how stars will appear through a telescope
//! on a specific sensor.

use std::time::Duration;

use crate::{photometry::BlackbodyStellarSpectrum, Spectrum};

use super::{SensorConfig, TelescopeConfig};
use starfield::catalogs::{StarData, StarPosition};

// A majority of stars have a B-V color index around 1.4
// This is used when the catalog does not provide a B-V value
// In practice this is mostly for faint stars, but this is a
// reasonable guess for stuff on the main sequence
const DEFAULT_BV: f64 = 1.4;

/// Calculate the diameter of the field of view in degrees
///
/// This function calculates the circular diameter that would fully
/// encompass a sensor's field of view when mounted on a telescope.
///
/// # Arguments
/// * `telescope` - The telescope configuration
/// * `sensor` - The sensor configuration
///
/// # Returns
/// Field diameter in degrees
pub fn field_diameter(telescope: &TelescopeConfig, sensor: &SensorConfig) -> f64 {
    // Get sensor dimensions in microns
    let (width_um, height_um) = sensor.dimensions_um();

    // Calculate the diagonal size of the sensor in microns
    let diagonal_um = (width_um.powi(2) + height_um.powi(2)).sqrt();

    // Convert to meters
    let diagonal_m = diagonal_um * 1.0e-6;

    // Use the plate scale to convert to an angle
    // angle in radians = diagonal / focal_length
    let angle_rad = diagonal_m / telescope.focal_length_m;

    // Convert to degrees
    angle_rad.to_degrees()
}

/// Calculate the projected pixel scale in arcseconds per pixel
///
/// # Arguments
/// * `telescope` - The telescope configuration
/// * `sensor` - The sensor configuration
///
/// # Returns
/// Pixel scale in arcseconds per pixel
pub fn pixel_scale(telescope: &TelescopeConfig, sensor: &SensorConfig) -> f64 {
    // Get the plate scale in arcsec/mm
    let plate_scale_arcsec_per_mm = telescope.plate_scale_arcsec_per_mm();

    // Convert to arcsec/pixel using the pixel size
    plate_scale_arcsec_per_mm * (sensor.pixel_size_um / 1000.0)
}

/// Convert star magnitude to flux in photons
///
/// # Arguments
/// * `magnitude` - The star's apparent magnitude
/// * `exposure_time` - Exposure time in seconds
/// * `telescope` - The telescope configuration
/// * `sensor` - The sensor configuration
/// * `wavelength_nm` - The wavelength in nanometers
///
/// # Returns
/// Expected photo electrons captured by the sensor for the given star
pub fn star_data_to_electrons(
    star_data: &StarData,
    exposure: &Duration,
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
) -> f64 {
    let spectrum = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(
        star_data.b_v.unwrap_or(DEFAULT_BV),
        star_data.magnitude,
    );
    let aperture_cm2 = telescope.collecting_area_m2() * 1.0e4; // Convert m^2 to cm^2
    spectrum.photo_electrons(&sensor.quantum_efficiency, aperture_cm2, exposure)
}

/// Filter stars that would be visible in the field of view
///
/// # Arguments
/// * `stars` - Vector of star data objects with ra and dec fields
/// * `center_ra` - Right ascension of field center in degrees
/// * `center_dec` - Declination of field center in degrees
/// * `field_diameter` - Diameter of field in degrees
///
/// # Returns
/// Vector of references to stars that are within the field
pub fn filter_stars_in_field<T>(
    stars: &[T],
    center_ra: f64,
    center_dec: f64,
    field_diameter: f64,
) -> Vec<&T>
where
    T: StarPosition,
{
    // Field radius in degrees
    let field_radius = field_diameter / 2.0;

    // Convert to radians for calculations
    let center_ra_rad = center_ra.to_radians();
    let center_dec_rad = center_dec.to_radians();
    let field_radius_rad = field_radius.to_radians();

    // Filter stars that fall within the field
    stars
        .iter()
        .filter(|star| {
            let star_ra_rad = star.ra().to_radians();
            let star_dec_rad = star.dec().to_radians();

            // Angular distance using haversine formula
            let d_ra = star_ra_rad - center_ra_rad;
            let d_dec = star_dec_rad - center_dec_rad;

            let a = (d_dec / 2.0).sin().powi(2)
                + center_dec_rad.cos() * star_dec_rad.cos() * (d_ra / 2.0).sin().powi(2);
            let angular_distance = 2.0 * a.sqrt().asin();

            // Include stars within field radius
            angular_distance <= field_radius_rad
        })
        .collect()
}

// We're now using the StarPosition trait from starfield::catalogs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::sensor::models as sensor_models;
    use crate::hardware::telescope::models as telescope_models;
    use float_cmp::approx_eq;
    use starfield::Equatorial;

    #[test]
    fn test_field_diameter() {
        let telescope = telescope_models::FINAL_1M.clone();
        let sensor = sensor_models::GSENSE4040BSI.clone();

        // Calculate expected field diameter
        let (width_um, height_um) = sensor.dimensions_um();
        let diagonal_um = (width_um.powi(2) + height_um.powi(2)).sqrt();
        let diagonal_m = diagonal_um * 1.0e-6;
        let expected_angle_rad = diagonal_m / telescope.focal_length_m;
        let expected_angle_deg = expected_angle_rad.to_degrees();

        let calculated = field_diameter(&telescope, &sensor);

        assert!(approx_eq!(
            f64,
            calculated,
            expected_angle_deg,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_pixel_scale() {
        let telescope = telescope_models::FINAL_1M.clone();
        let sensor = sensor_models::GSENSE4040BSI.clone();

        // Calculate expected pixel scale
        let arcsec_per_mm = telescope.plate_scale_arcsec_per_mm();
        let expected_scale = arcsec_per_mm * (sensor.pixel_size_um / 1000.0);

        let calculated = pixel_scale(&telescope, &sensor);

        assert!(approx_eq!(f64, calculated, expected_scale, epsilon = 1e-6));
    }

    // Test struct for StarPosition trait
    #[derive(Debug, Clone)]
    struct TestStar {
        ra: f64,
        dec: f64,
    }

    impl starfield::catalogs::StarPosition for TestStar {
        fn ra(&self) -> f64 {
            self.ra
        }

        fn dec(&self) -> f64 {
            self.dec
        }
    }

    #[test]
    fn test_filter_stars_in_field() {
        // Create test stars
        let stars = vec![
            TestStar {
                ra: 100.0,
                dec: 45.0,
            }, // Center
            TestStar {
                ra: 100.1,
                dec: 45.0,
            }, // Near center
            TestStar {
                ra: 101.0,
                dec: 45.0,
            }, // 1 degree away in RA
            TestStar {
                ra: 100.0,
                dec: 46.0,
            }, // 1 degree away in Dec
            TestStar {
                ra: 105.0,
                dec: 45.0,
            }, // 5 degrees away in RA
            TestStar {
                ra: 100.0,
                dec: 50.0,
            }, // 5 degrees away in Dec
        ];

        // Test with 2 degree field diameter
        let field_stars = filter_stars_in_field(&stars, 100.0, 45.0, 2.0);

        // Count how many stars are definitely in the field
        // Stars exactly 1 degree away might be at the border due to numeric precision in the calculation
        assert!(field_stars.len() >= 3);

        // Test with smaller field
        let field_stars = filter_stars_in_field(&stars, 100.0, 45.0, 0.5);

        // Should only include center and near center
        assert_eq!(field_stars.len(), 2);
    }

    #[test]
    fn test_magnitude_to_electron_different_telescopes() {
        let small_telescope = telescope_models::DEMO_50CM.clone();
        let large_telescope = telescope_models::FINAL_1M.clone();
        let sensor = sensor_models::GSENSE4040BSI.clone();

        let star_data = StarData {
            id: 0,
            position: Equatorial::from_degrees(0.0, 0.0),
            magnitude: 2.0,
            b_v: None,
        };

        let second = Duration::from_secs_f64(1.0);
        let elec_small = star_data_to_electrons(&star_data, &second, &small_telescope, &sensor);
        let elec_large = star_data_to_electrons(&star_data, &second, &large_telescope, &sensor);

        // Aperture ratio squared: 1.0^2 / 0.5^2 = 4.0
        // But also need to consider light efficiency differences
        let expected_ratio = (large_telescope.aperture_m / small_telescope.aperture_m).powi(2)
            * (large_telescope.light_efficiency / small_telescope.light_efficiency);

        assert!(approx_eq!(
            f64,
            elec_large / elec_small,
            expected_ratio,
            epsilon = 1e-6
        ));
    }
}

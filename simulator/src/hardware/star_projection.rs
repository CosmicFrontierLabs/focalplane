//! Star projection utilities for telescope simulation
//!
//! This module provides functions to calculate the field of view, projected area,
//! and other parameters needed to simulate how stars will appear through a telescope
//! on a specific sensor.

use super::{SensorConfig, TelescopeConfig};
use starfield::catalogs::StarPosition;

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
/// Expected photon flux at the sensor for the given star
pub fn magnitude_to_photon_flux(
    magnitude: f64,
    exposure_time: f64,
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    wavelength_nm: f64,
) -> f64 {
    // From wikipedia:
    //  "The zero point of the apparent bolometric magnitude scale is based on the definition
    //   that an apparent bolometric magnitude of 0 mag is equivalent to a received irradiance
    //   of 2.518×10−8 watts per square metre (W·m−2).[16]"
    let irradiance_mag0 = 2.518e-8; // [J/s/m²] = [W/m²]
    let power_at_aperture = irradiance_mag0 * telescope.effective_collecting_area_m2(); // [J/s] = [W]
                                                                                        // Convert to photons using E = h * c / λ
                                                                                        // h = Planck's constant, c = speed of light
                                                                                        // λ = wavelength in meters
                                                                                        // E = energy per photon in joules
                                                                                        // photons = energy / energy_per_photon
    let energy_per_photon = 6.626e-34 * 3.0e8 / (wavelength_nm * 1.0e-9); // [J/photon]

    let photon_rate_mag0 = power_at_aperture / energy_per_photon; // [photons/s]

    // This formula gives us photon flux at magnitude 0
    // We scale by magnitude_factor which incorporates the actual magnitude
    // Base conversion from magnitude to relative flux
    // Pogson equation: flux = 10^(-0.4 * magnitude)
    let magnitude_factor = 10.0_f64.powf(-0.4 * magnitude); // [dimensionless]
    let photon_rate_actual = photon_rate_mag0 * magnitude_factor; // [photons/s]
    let photons_collected = photon_rate_actual * exposure_time; // [photons]

    // Apply quantum efficiency of the sensor at this wavelength
    let qe = sensor.qe_at_wavelength(wavelength_nm as u32);
    let detected_photons = photons_collected * qe; // [detected photons]

    detected_photons // [detected photons]
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
    fn test_magnitude_to_photon_flux() {
        let telescope = telescope_models::FINAL_1M.clone();
        let sensor = sensor_models::GSENSE4040BSI.clone();

        // Compare relative fluxes
        let flux_mag0 = magnitude_to_photon_flux(0.0, 1.0, &telescope, &sensor, 550.0);
        let flux_mag5 = magnitude_to_photon_flux(5.0, 1.0, &telescope, &sensor, 550.0);

        // A 5 magnitude difference should be a factor of 100 in flux
        assert!(approx_eq!(
            f64,
            flux_mag0 / flux_mag5,
            100.0,
            epsilon = 1e-6
        ));

        // Test exposure time scaling
        let flux_1s = magnitude_to_photon_flux(2.0, 1.0, &telescope, &sensor, 550.0);
        let flux_2s = magnitude_to_photon_flux(2.0, 2.0, &telescope, &sensor, 550.0);

        assert!(approx_eq!(f64, flux_2s / flux_1s, 2.0, epsilon = 1e-6));
    }

    #[test]
    fn test_magnitude_to_photon_flux_zero_point() {
        let telescope = telescope_models::FINAL_1M.clone();
        let sensor = sensor_models::GSENSE4040BSI.clone();
        let wavelength_nm = 550.0;

        // Calculate expected flux for magnitude 0
        let magnitude_factor = 10.0_f64.powf(-0.4 * 0.0); // 1.0 for mag 0
        let energy_per_photon = 6.626e-34 * 3.0e8 / (wavelength_nm * 1.0e-9);
        let irradiance_mag0 = 2.518e-8; // Same as in the function
        let power_at_aperture = irradiance_mag0 * telescope.effective_collecting_area_m2();
        let photon_rate_mag0 = power_at_aperture / energy_per_photon;
        let photon_rate_actual = photon_rate_mag0 * magnitude_factor;
        let photons_collected = photon_rate_actual * 1.0; // 1 second exposure
        let quantum_efficiency = sensor.qe_at_wavelength(wavelength_nm as u32);
        let expected_flux = photons_collected * quantum_efficiency;

        let calculated_flux =
            magnitude_to_photon_flux(0.0, 1.0, &telescope, &sensor, wavelength_nm);

        assert!(approx_eq!(
            f64,
            calculated_flux,
            expected_flux,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_magnitude_to_photon_flux_with_different_wavelengths() {
        let telescope = telescope_models::FINAL_1M.clone();
        let sensor = sensor_models::IMX455.clone();
        let magnitude = 3.0;
        let exposure_time = 1.0;

        // Test two different wavelengths with known QE differences
        let flux_400nm =
            magnitude_to_photon_flux(magnitude, exposure_time, &telescope, &sensor, 400.0);
        let flux_500nm =
            magnitude_to_photon_flux(magnitude, exposure_time, &telescope, &sensor, 500.0);

        // Get actual QE values from sensor configuration
        let qe_400nm = sensor.qe_at_wavelength(400);
        let qe_500nm = sensor.qe_at_wavelength(500);
        let qe_ratio = qe_500nm / qe_400nm;

        // In our updated implementation, energy per photon is directly included in the calculation
        // The actual ratio we expect is influenced by both QE and wavelength (energy per photon)
        // E = h*c/λ, so energy ratio = λ₁/λ₂
        let energy_ratio = 500.0 / 400.0; // Longer wavelength = less energy per photon

        // Our expected ratio combines both QE and wavelength effects
        // More photons detected both from better QE and lower energy per photon (more photons for same power)
        let expected_ratio = qe_ratio * energy_ratio;

        println!(
            "QE 400nm: {}, QE 500nm: {}, QE Ratio: {}",
            qe_400nm, qe_500nm, qe_ratio
        );
        println!("Energy ratio (500nm/400nm): {}", energy_ratio);
        println!(
            "Flux 400nm: {}, Flux 500nm: {}, Ratio: {}",
            flux_400nm,
            flux_500nm,
            flux_500nm / flux_400nm
        );
        println!("Expected ratio: {}", expected_ratio);

        assert!(approx_eq!(
            f64,
            flux_500nm / flux_400nm,
            expected_ratio,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_magnitude_to_photon_flux_different_telescopes() {
        let small_telescope = telescope_models::DEMO_50CM.clone();
        let large_telescope = telescope_models::FINAL_1M.clone();
        let sensor = sensor_models::GSENSE4040BSI.clone();

        let flux_small = magnitude_to_photon_flux(2.0, 1.0, &small_telescope, &sensor, 550.0);
        let flux_large = magnitude_to_photon_flux(2.0, 1.0, &large_telescope, &sensor, 550.0);

        // Aperture ratio squared: 1.0^2 / 0.5^2 = 4.0
        // But also need to consider light efficiency differences
        let expected_ratio = (large_telescope.aperture_m / small_telescope.aperture_m).powi(2)
            * (large_telescope.light_efficiency / small_telescope.light_efficiency);

        assert!(approx_eq!(
            f64,
            flux_large / flux_small,
            expected_ratio,
            epsilon = 1e-6
        ));
    }
}

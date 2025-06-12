//! Zodiacal light brightness model
//!
//! This module provides functionality for interpolating zodiacal light data.
//! It implements a bilinear interpolation approach for estimating zodiacal light brightness
//! at arbitrary ecliptic coordinates based on angular separation from the sun.
//!
//! Data source: Leinert et al. (1998) Table 16, as implemented in NASA's Dorado sensitivity model:
//! https://github.com/nasa/dorado-sensitivity/blob/main/dorado/sensitivity/data/leinert_zodi.txt
//! Original paper: https://doi.org/10.1051/aas:1998105

use ndarray::Array2;
use std::time::Duration;
use thiserror::Error;

use crate::photometry::{spectrum::Spectrum, STISZodiacalSpectrum};
use crate::{SensorConfig, TelescopeConfig};

/// Errors that can occur when working with zodiacal light data
#[derive(Error, Debug)]
pub enum ZodicalError {
    #[error("Coordinates out of range: elongation {0}°, ecliptic latitude {1}°")]
    OutOfRange(f64, f64),

    #[error("Interpolation error: {0}")]
    InterpolationError(String),

    #[error("Invalid elongation: {0}° (must be between 0° and 180°)")]
    InvalidElongation(f64),

    #[error("Invalid latitude: {0}° (must be between -90° and 90°)")]
    InvalidLatitude(f64),
}

/// Solar angular position coordinates for zodiacal light calculations
///
/// Represents the angular position of a target relative to the Sun as seen from Earth.
/// These are the coordinates used in Leinert et al. (1998) zodiacal light measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarAngularCoordinates {
    /// Angular distance between Sun and target as seen from Earth (0 to 180 degrees)
    elongation: f64,
    /// Ecliptic latitude of the target (-90 to 90 degrees)
    latitude: f64,
}

impl SolarAngularCoordinates {
    /// Create new solar angular coordinates with validation
    ///
    /// # Arguments
    ///
    /// * `elongation` - Angular distance from sun in degrees (0 to 180)
    /// * `latitude` - Ecliptic latitude in degrees (-90 to 90)
    ///
    /// # Returns
    ///
    /// A `Result` containing the validated coordinates or an error
    pub fn new(elongation: f64, latitude: f64) -> Result<Self, ZodicalError> {
        if !(0.0..=180.0).contains(&elongation) {
            return Err(ZodicalError::InvalidElongation(elongation));
        }

        if !(-90.0..=90.0).contains(&latitude) {
            return Err(ZodicalError::InvalidLatitude(latitude));
        }

        Ok(Self {
            elongation,
            latitude,
        })
    }

    /// Get the elongation in degrees
    pub fn elongation(&self) -> f64 {
        self.elongation
    }

    /// Get the latitude in degrees
    pub fn latitude(&self) -> f64 {
        self.latitude
    }

    /// Get the absolute latitude (for internal calculations)
    pub(crate) fn abs_latitude(&self) -> f64 {
        self.latitude.abs()
    }
}

/// Represents zodiacal light brightness data as a function of angular separation from the sun
pub struct ZodicalLight {
    /// The underlying data table containing brightness values in S10 units (10th magnitude stars per square degree)
    data: Array2<f64>,
}

// Leinert et al. (1998) Table 16 coordinate grids
// Ecliptic latitudes (degrees from ecliptic plane)

// Angular distances from sun (degrees) - elongation angles
// NOTE(meawoppl) - we manually added the last row (180deg elongation) to match the augmentation
// that is performed in the original Dorado model.
const LATITUDES: [f64; 11] = [
    0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 45.0, 60.0, 75.0, 90.0,
];
const ELONGATIONS: [f64; 19] = [
    0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0,
    135.0, 150.0, 165.0, 180.0,
];

// Leinert et al. (1998) Table 16 - Zodiacal light brightness in S10 units
// (10th magnitude stars per square degree)
// Each row corresponds to an ecliptic latitude (0° to 90°)
// Each column corresponds to an elongation angle from sun (0° to 180°)
// inf values represent areas too close to sun for measurement
#[rustfmt::skip]
fn zodical_raw_data() -> [[f64; 11]; 19] {
    let inf = f64::INFINITY;

    [
        [inf,    inf,    inf,    2450.0, 1260.0, 770.0, 500.0, 215.0, 117.0, 78.0, 60.0],
        [inf,    inf,    inf,    2300.0, 1200.0, 740.0, 490.0, 212.0, 117.0, 78.0, 60.0],
        [inf,    inf,    3700.0, 1930.0, 1070.0, 675.0, 460.0, 206.0, 116.0, 78.0, 60.0],
        [9000.0, 5300.0, 2690.0, 1450.0, 870.0,  590.0, 410.0, 196.0, 114.0, 78.0, 60.0],
        [5000.0, 3500.0, 1880.0, 1100.0, 710.0,  495.0, 355.0, 185.0, 110.0, 77.0, 60.0],
        [3000.0, 2210.0, 1350.0, 860.0,  585.0,  425.0, 320.0, 174.0, 106.0, 76.0, 60.0],
        [1940.0, 1460.0, 955.0,  660.0,  480.0,  365.0, 285.0, 162.0, 102.0, 74.0, 60.0],
        [1290.0, 990.0,  710.0,  530.0,  400.0,  310.0, 250.0, 151.0,  98.0, 73.0, 60.0],
        [925.0,  735.0,  545.0,  415.0,  325.0,  264.0, 220.0, 140.0,  94.0, 72.0, 60.0],
        [710.0,  570.0,  435.0,  345.0,  278.0,  228.0, 195.0, 130.0,  91.0, 70.0, 60.0],
        [395.0,  345.0,  275.0,  228.0,  190.0,  163.0, 143.0, 105.0,  81.0, 67.0, 60.0],
        [264.0,  248.0,  210.0,  177.0,  153.0,  134.0, 118.0,  91.0,  73.0, 64.0, 60.0],
        [202.0,  196.0,  176.0,  151.0,  130.0,  115.0, 103.0,  81.0,  67.0, 62.0, 60.0],
        [166.0,  164.0,  154.0,  133.0,  117.0,  104.0,  93.0,  75.0,  64.0, 60.0, 60.0],
        [147.0,  145.0,  138.0,  120.0,  108.0,   98.0,  88.0,  70.0,  60.0, 58.0, 60.0],
        [140.0,  139.0,  130.0,  115.0,  105.0,   95.0,  86.0,  70.0,  60.0, 57.0, 60.0],
        [140.0,  139.0,  129.0,  116.0,  107.0,   99.0,  91.0,  75.0,  62.0, 56.0, 60.0],
        [153.0,  150.0,  140.0,  129.0,  118.0,  110.0, 102.0,  81.0,  64.0, 56.0, 60.0],
        [180.0,  166.0,  152.0,  139.0,  127.0,  116.0, 105.0,  82.0,  65.0, 56.0, 60.0],
    ]
}

pub const LAT_OF_MIN: f64 = 75.0;
pub const ELONG_OF_MIN: f64 = 165.0;

impl Default for ZodicalLight {
    fn default() -> Self {
        Self::new()
    }
}

impl ZodicalLight {
    /// Create a new ZodicalLight instance with the embedded data
    pub fn new() -> Self {
        // Convert the hardcoded 2D array to an ndarray::Array2
        // Raw data is organized as [elongation][latitude], but we want [latitude][elongation]
        // So we transpose during construction
        let mut data = Array2::zeros((LATITUDES.len(), ELONGATIONS.len()));
        let raw_data = zodical_raw_data();

        for (elong_idx, elong_row) in raw_data.iter().enumerate() {
            for (lat_idx, &val) in elong_row.iter().enumerate() {
                data[[lat_idx, elong_idx]] = val;
            }
        }

        Self { data }
    }

    /// Find the indices and interpolation weights for a value within an array
    ///
    /// # Arguments
    ///
    /// * `array` - Array of coordinate values (sorted)
    /// * `value` - Value to locate in the array
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * The lower index
    /// * The upper index
    /// * The weight for the lower value (interpolation factor)
    fn find_indices_and_weights(array: &[f64], value: f64) -> Option<(usize, usize, f64)> {
        if array.len() < 2 {
            return None;
        }

        // Handle out of bounds cases
        if value < array[0] || value > array[array.len() - 1] {
            return None;
        }

        // Find the lower index using binary search
        let mut lower_idx = 0;
        let mut upper_idx = array.len() - 1;

        while upper_idx - lower_idx > 1 {
            let mid_idx = (lower_idx + upper_idx) / 2;
            if array[mid_idx] <= value {
                lower_idx = mid_idx;
            } else {
                upper_idx = mid_idx;
            }
        }

        // If we hit the exact value, use the same index for both
        if array[lower_idx] == value {
            return Some((lower_idx, lower_idx, 1.0));
        }

        // Compute the weight for interpolation
        let lower_val = array[lower_idx];
        let upper_val = array[upper_idx];
        let weight = (value - lower_val) / (upper_val - lower_val);

        Some((lower_idx, upper_idx, 1.0 - weight))
    }

    /// Get the zodiacal light brightness at given ecliptic coordinates using bilinear interpolation
    ///
    /// # Arguments
    ///
    /// * `coords` - Solar angular coordinates (elongation and ecliptic latitude)
    ///
    /// # Returns
    ///
    /// A `Result` containing either the brightness in S10 units (10th magnitude stars per square degree) or an error
    pub fn get_brightness(&self, coords: &SolarAngularCoordinates) -> Result<f64, ZodicalError> {
        // Find indices and weights for elongation
        let elongation = coords.elongation();
        let latitude = coords.abs_latitude();
        let (elong_idx1, elong_idx2, elong_weight) =
            Self::find_indices_and_weights(&ELONGATIONS, elongation)
                .ok_or(ZodicalError::OutOfRange(elongation, coords.latitude()))?;

        // Find indices and weights for latitude
        let (lat_idx1, lat_idx2, lat_weight) = Self::find_indices_and_weights(&LATITUDES, latitude)
            .ok_or(ZodicalError::OutOfRange(elongation, coords.latitude()))?;

        // Get the four corner values (data is indexed as [latitude][elongation])
        let q11 = self.data[[lat_idx1, elong_idx1]];
        let q12 = self.data[[lat_idx2, elong_idx1]];
        let q21 = self.data[[lat_idx1, elong_idx2]];
        let q22 = self.data[[lat_idx2, elong_idx2]];

        // Handle infinite values by using nearest finite value
        let mut valid_points = Vec::new();
        let mut valid_weights = Vec::new();

        if q11.is_finite() {
            valid_points.push(q11);
            valid_weights.push(lat_weight * elong_weight);
        }

        if q12.is_finite() {
            valid_points.push(q12);
            valid_weights.push((1.0 - lat_weight) * elong_weight);
        }

        if q21.is_finite() {
            valid_points.push(q21);
            valid_weights.push(lat_weight * (1.0 - elong_weight));
        }

        if q22.is_finite() {
            valid_points.push(q22);
            valid_weights.push((1.0 - lat_weight) * (1.0 - elong_weight));
        }

        // If we have no valid points, return an error
        if valid_points.is_empty() {
            return Err(ZodicalError::InterpolationError(
                "No valid data points for interpolation".to_string(),
            ));
        }

        // Normalize weights
        let weight_sum: f64 = valid_weights.iter().sum();
        let normalized_weights: Vec<f64> = valid_weights.iter().map(|&w| w / weight_sum).collect();

        // Calculate weighted average
        let result = valid_points
            .iter()
            .zip(normalized_weights.iter())
            .map(|(&val, &weight)| val * weight)
            .sum();

        Ok(result)
    }

    /// Get the zodiacal light brightness in magnitudes per square arcsecond
    ///
    /// Converts the brightness from S10 units (10th magnitude stars per square degree)
    /// to magnitudes per square arcsecond using the standard photometric conversion.
    ///
    /// # Arguments
    ///
    /// * `coords` - Solar angular coordinates (elongation and ecliptic latitude)
    ///
    /// # Returns
    ///
    /// A `Result` containing the brightness in magnitudes per square arcsecond or an error
    ///
    /// # Physics
    ///
    /// The conversion from S10 to magnitudes per square arcsecond follows:
    /// mag/arcsec² = 10 - 2.5 * log₁₀(S10 / 3600²)
    /// where 3600² converts from square degrees to square arcseconds
    pub fn get_brightness_mag_per_square_arcsec(
        &self,
        coords: &SolarAngularCoordinates,
    ) -> Result<f64, ZodicalError> {
        // Get brightness in S10 units
        let s10 = self.get_brightness(coords)?;

        // Convert S10 to magnitudes per square arcsecond
        Ok(10.0 - 2.5 * (s10 / (3600.0 * 3600.0)).log10())
    }

    /// Get the zodiacal light spectrum scale factor relative to a reference position
    ///
    /// Calculates a scale factor to adjust the standard zodiacal spectrum based on
    /// the brightness difference between the given coordinates and a reference position
    /// at 180° elongation and 0° latitude (anti-solar point).
    ///
    /// # Arguments
    ///
    /// * `coords` - Solar angular coordinates (elongation and ecliptic latitude)
    ///
    /// # Returns
    ///
    /// A `Result` containing the scale factor (dimensionless) or an error
    ///
    /// # Physics
    ///
    /// The scale factor converts magnitude differences to flux ratios:
    /// scale_factor = 10^(-0.4 * Δmag)
    /// where Δmag = mag(target) - mag(reference)
    ///
    /// This allows scaling a reference zodiacal spectrum to match the brightness
    /// at any position in the sky while preserving spectral shape.
    pub fn get_spectrum_scale_factor(
        &self,
        coords: &SolarAngularCoordinates,
    ) -> Result<f64, ZodicalError> {
        // Get brightness in S10 units
        let reference_coords = SolarAngularCoordinates::new(180.0, 0.0)?;
        let reference = self.get_brightness_mag_per_square_arcsec(&reference_coords)?;
        let s10 = self.get_brightness_mag_per_square_arcsec(coords)?;
        let mag_diff = s10 - reference;
        Ok(10_f64.powf(-0.4 * mag_diff))
    }

    /// Get a scaled zodiacal spectrum for the given ecliptic coordinates
    ///
    /// Returns a zodiacal spectrum scaled to match the brightness at the specified
    /// position. Uses the STIS zodiacal spectrum as a reference template and scales
    /// it based on the brightness difference from the anti-solar point.
    ///
    /// # Arguments
    ///
    /// * `coords` - Solar angular coordinates (elongation and ecliptic latitude)
    ///
    /// # Returns
    ///
    /// A `Result` containing the scaled `STISZodiacalSpectrum` or an error
    ///
    /// # Physics
    ///
    /// The spectrum maintains the same spectral energy distribution (SED) shape
    /// as measured by STIS, but is scaled in overall brightness to match the
    /// zodiacal light intensity at the target coordinates. This assumes the
    /// zodiacal spectrum shape is constant across the sky.
    pub fn get_zodical_spectrum(
        &self,
        coords: &SolarAngularCoordinates,
    ) -> Result<STISZodiacalSpectrum, ZodicalError> {
        // Get the scale factor based on a reference point
        let scale_factor = self.get_spectrum_scale_factor(coords)?;
        Ok(STISZodiacalSpectrum::new(scale_factor))
    }

    /// Generates zodiacal light noise as photoelectrons for a sensor.
    ///
    /// This function computes the contribution of zodiacal light (scattered sunlight from interplanetary dust)
    /// to the noise in an astronomical image. The result is expressed in units of photoelectrons.
    ///
    /// # Arguments
    /// * `sensor` - Configuration of the sensor
    /// * `telescope` - Configuration of the telescope
    /// * `exposure` - Duration of the exposure
    /// * `coords` - Solar angular coordinates (elongation and ecliptic latitude)
    ///
    /// # Returns
    /// * An ndarray::Array2<f64> with dimensions matching the sensor, where each element
    ///   represents the number of photoelectrons generated by zodiacal light in that pixel
    /// # Note
    /// This assumed the light is uniformly distributed across the sensor pixels. (no vignetting)
    /// Ideally we should also interpolate the value across the sensor, as it is not uniform
    /// across the field of view.
    pub fn generate_zodical_background(
        &self,
        sensor: &SensorConfig,
        telescope: &TelescopeConfig,
        exposure: &Duration,
        coords: &SolarAngularCoordinates,
    ) -> Array2<f64> {
        // Convert telescope focal length from meters to mm
        let focal_length_mm = telescope.focal_length_m * 1000.0;

        // Calculate pixel scale in arcseconds per pixel
        // Pixel scale = (206265 * pixel_size) / focal_length
        // where 206265 is the number of arcseconds in a radian
        let pixel_scale_arcsec_per_pixel =
            206265.0 * (sensor.pixel_size_um / 1000.0) / focal_length_mm;

        let z_spect = self
            .get_zodical_spectrum(coords)
            .expect("Unable to generate zodical spectrum?");

        let pixel_solid_angle_arcsec2 = pixel_scale_arcsec_per_pixel * pixel_scale_arcsec_per_pixel;

        // Compute the photoelectrons per solid angle and multiply by the pixel solid angle
        let aperture_cmsq = telescope.aperture_m * 10000.0;
        let mean_pe = z_spect.photo_electrons(&sensor.quantum_efficiency, aperture_cmsq, exposure)
            * pixel_solid_angle_arcsec2
            * telescope.light_efficiency;

        Array2::ones((sensor.height_px as usize, sensor.width_px as usize)) * mean_pe
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_dorado_match() {
        // Running the tests on this branch, yields the following outputs
        // https://github.com/meawoppl/dorado-sensitivity/tree/mrg-augmneted

        // Input time: 2020-01-01 00:00:00.000
        // Object ecliptic coords: lon=90.29°, lat=66.56°
        // Sun ecliptic coords: lon=280.01°, lat=-0.00°
        // Angular separation from sun: lon=170.28°, lat=66.56°
        // Raw interpolated result: [23.32617604]
        // After nan handling: [23.32617604]
        // Reference value (180°, 0°): 22.143
        // Relative magnitude: 1.183
        // Final scale factor: 0.336

        // Test that the Dorado model matches the original data
        let zodical = ZodicalLight::new();

        // Check a few known points from the original data
        let coords = SolarAngularCoordinates::new(170.28, 66.56).unwrap();
        assert_relative_eq!(
            zodical.get_spectrum_scale_factor(&coords).unwrap(),
            0.336,
            epsilon = 0.01
        );
    }

    #[test]
    fn test_find_indices_and_weights() {
        let array = [0.0, 10.0, 20.0, 30.0, 40.0];

        // Test exact match
        let (idx1, idx2, weight) = ZodicalLight::find_indices_and_weights(&array, 10.0).unwrap();
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 1);
        assert_eq!(weight, 1.0);

        // Test interpolation
        let (idx1, idx2, weight) = ZodicalLight::find_indices_and_weights(&array, 15.0).unwrap();
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        assert!((weight - 0.5).abs() < 1e-10);

        // Test out of bounds (low)
        assert!(ZodicalLight::find_indices_and_weights(&array, -5.0).is_none());

        // Test out of bounds (high)
        assert!(ZodicalLight::find_indices_and_weights(&array, 45.0).is_none());
    }

    #[test]
    fn test_get_brightness() {
        let zodical = ZodicalLight::new();

        // Test a coordinate with known value (exact match to a grid point)
        // 45° elongation, 60° latitude should give 91.0 S10 (after transposing data)
        let coords = SolarAngularCoordinates::new(45.0, 60.0).unwrap();
        let brightness = zodical.get_brightness(&coords).unwrap();
        assert!((brightness - 91.0).abs() < 1e-10);

        // Test interpolation between grid points
        // Between 30° and 45° elongation, between 45° and 60° latitude
        let coords = SolarAngularCoordinates::new(37.5, 52.5).unwrap();
        let brightness = zodical.get_brightness(&coords).unwrap();
        // Expected: interpolation between surrounding values
        assert!(brightness > 80.0 && brightness < 200.0);

        // Test out of range
        assert!(SolarAngularCoordinates::new(200.0, 50.0).is_err());
        assert!(SolarAngularCoordinates::new(45.0, 150.0).is_err());
    }

    #[test]
    fn test_brightness_range() {
        let zodical = ZodicalLight::new();

        // Test a grid of points across the valid range
        // This covers the sun exclusion zone and will have a bunch
        // of points on the cusp to stress the interpolation
        // We expect the brightness to be between 40 and 10000 S10 units
        for elong in (0..=75).step_by(5) {
            for lat in (0..=105).step_by(5) {
                let elong_f64 = elong as f64;
                let lat_f64 = lat as f64;

                if let Ok(coords) = SolarAngularCoordinates::new(elong_f64, lat_f64) {
                    if let Ok(brightness) = zodical.get_brightness(&coords) {
                        if brightness.is_finite() {
                            assert!(
                                (40.0..=10000.0).contains(&brightness),
                                "Brightness at elongation={}, lat={} is {}, which is outside the expected range [40, 10000]",
                                elong_f64, lat_f64, brightness
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_data_dimensions() {
        let zodical = ZodicalLight::new();

        // Check that the data array has the correct dimensions
        assert_eq!(zodical.data.shape(), [LATITUDES.len(), ELONGATIONS.len()]);
        assert_eq!(zodical.data.shape(), [11, 19]);
    }

    #[test]
    fn test_sun_exclusion_zones() {
        let zodical = ZodicalLight::new();

        // Test that regions too close to the sun return finite values or errors
        // At 0° latitude, 0° elongation should be infinity (sun exclusion)
        let coords = SolarAngularCoordinates::new(0.0, 0.0).unwrap();
        match zodical.get_brightness(&coords) {
            Ok(brightness) => assert!(!brightness.is_finite()),
            Err(_) => (), // Also acceptable
        }

        // At higher latitudes, small elongations should have finite values
        let coords = SolarAngularCoordinates::new(15.0, 15.0).unwrap();
        let brightness = zodical.get_brightness(&coords).unwrap();
        assert!(brightness.is_finite());
        assert!(brightness > 1000.0); // Should be bright near the sun
    }

    #[test]
    fn test_zodical_light_minimum_angle() {
        let zodical = ZodicalLight::new();

        // Test the minimum angle where zodiacal light is still measurable
        // This is around 165° elongation and 75° latitude
        let coords = SolarAngularCoordinates::new(ELONG_OF_MIN, LAT_OF_MIN).unwrap();
        let min_brightness = zodical.get_brightness(&coords).unwrap();
        assert!(min_brightness.is_finite());
        assert!(min_brightness > 0.0); // Should be measurable at this point

        // Test a grid of points across the valid range
        // This covers the minimum angle and will have a bunch
        // of points on the cusp to stress the interpolation
        // We expect the brightness to be greater than 0
        for elong in (ELONG_OF_MIN as i32 - 5..=ELONG_OF_MIN as i32 + 5).step_by(1) {
            for lat in (LAT_OF_MIN as i32 - 5..=LAT_OF_MIN as i32 + 5).step_by(1) {
                let elong_f64 = elong as f64;
                let lat_f64 = lat as f64;

                if let Ok(coords) = SolarAngularCoordinates::new(elong_f64, lat_f64) {
                    if let Ok(brightness) = zodical.get_brightness(&coords) {
                        if brightness.is_finite() {
                            assert!(
                    brightness >= min_brightness,
                    "Brightness at elongation={}, lat={} is {:.3}, which is not greater than {:.3}",
                    elong_f64, lat_f64, brightness, min_brightness
                    );
                        }
                    }
                }
            }
        }
    }
}

//! Utility functions for star catalog manipulation and coordinate transformations
//!
//! This module contains utility functions for working with star catalogs and
//! transforming between coordinate systems (equatorial to pixel, etc.)

use nalgebra::Matrix3;
use starfield::framelib::inertial::InertialFrame;
use starfield::Equatorial;

use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[cfg(test)]
use crate::algo::MinMaxScan;

/// Random Equatorial coordinate generator
///
/// Generates random RA/Dec coordinates using a seeded random number generator.
/// RA is generated uniformly in [0, 360) degrees, and Dec is generated
/// uniformly in [-90, 90] degrees.
pub struct EquatorialRandomizer {
    rng: StdRng,
}

impl EquatorialRandomizer {
    /// Create a new EquatorialRandomizer with the given seed
    ///
    /// # Arguments
    /// * `seed` - Random number generator seed for reproducible results
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate a new random Equatorial coordinate
    ///
    /// # Returns
    /// * `Equatorial` - Random sky coordinates with RA in [0, 360) and Dec in [-90, 90] degrees
    pub fn generate(&mut self) -> Equatorial {
        let ra = self.rng.gen::<f64>() * 360.0; // Random RA in degrees [0, 360)
        let dec = (self.rng.gen::<f64>() - 0.5) * 180.0; // Random Dec in degrees [-90, 90]
        Equatorial::from_degrees(ra, dec)
    }
}

/// Star projector that maps celestial coordinates to image pixels
pub struct StarProjector {
    /// Center point of the projection in right ascension/declination (radians)
    pub center: Equatorial,

    /// Angular resolution in radians per pixel
    radians_per_pixel: f64,
    /// Sensor dimensions in pixels
    sensor_width: usize,
    sensor_height: usize,
    /// Rotation matrix from celestial to camera coordinates
    rotation_matrix: Matrix3<f64>,
}

impl StarProjector {
    /// Create a new StarProjector
    ///
    /// # Arguments
    /// * `center` - Equatorial coordinates of the projection center
    /// * `radians_per_pixel` - Angular resolution in radians per pixel
    /// * `sensor_width` - Width of sensor in pixels
    /// * `sensor_height` - Height of sensor in pixels
    pub fn new(
        center: &Equatorial,
        radians_per_pixel: f64,
        sensor_width: usize,
        sensor_height: usize,
    ) -> Self {
        // Calculate rotation matrix to transform from celestial to camera coordinates
        // Camera Z-axis points to center_ra/center_dec
        // Camera Y-axis points towards celestial north
        // Camera X-axis completes right-handed system

        let cos_ra = center.ra.cos();
        let sin_ra = center.ra.sin();
        let cos_dec = center.dec.cos();
        let sin_dec = center.dec.sin();

        // Z-axis (pointing to center)
        let z = Vector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec);

        // Y-axis (towards celestial north)
        let north = Vector3::new(0.0, 0.0, 1.0);
        let east = north.cross(&z).normalize();
        let y = z.cross(&east).normalize();

        // X-axis (east direction)
        let x = y.cross(&z).normalize();

        // Build rotation matrix (columns are the new basis vectors)
        let rotation_matrix = Matrix3::from_columns(&[x, y, z]);

        Self {
            center: *center,
            radians_per_pixel,
            sensor_width,
            sensor_height,
            rotation_matrix,
        }
    }

    /// Project an equatorial coordinate to pixel coordinates without bounds checking
    ///
    /// # Arguments
    /// * `equatorial` - Reference to an Equatorial instance
    ///
    /// # Returns
    /// * `Option<(f64, f64)>` - Pixel coordinates (x, y) if star is in front of camera, None if behind
    pub fn project_unbounded(&self, equatorial: &Equatorial) -> Option<(f64, f64)> {
        // Convert equatorial to cartesian unit vector
        let cartesian = equatorial.to_cartesian().to_vector3();

        // Transform to camera coordinates
        let camera_coords = self.rotation_matrix.transpose() * cartesian;

        // Check if star is in front of camera (z > 0)
        if camera_coords.z <= 0.0 {
            return None;
        }

        // Apply gnomonic (tangent plane) projection
        let x_proj = camera_coords.x / camera_coords.z;
        let y_proj = camera_coords.y / camera_coords.z;

        // Convert to pixel coordinates
        let pixel_x = (self.sensor_width as f64 / 2.0) + (x_proj / self.radians_per_pixel);
        let pixel_y = (self.sensor_height as f64 / 2.0) - (y_proj / self.radians_per_pixel);

        Some((pixel_x, pixel_y))
    }

    /// Project an equatorial coordinate to pixel coordinates
    ///
    /// # Arguments
    /// * `equatorial` - Reference to an Equatorial instance
    ///
    /// # Returns
    /// * `Option<(f64, f64)>` - Pixel coordinates (x, y) if visible, None if outside sensor
    pub fn project(&self, equatorial: &Equatorial) -> Option<(f64, f64)> {
        let (pixel_x, pixel_y) = self.project_unbounded(equatorial)?;

        // Check if within sensor bounds
        if pixel_x >= 0.0
            && pixel_x < self.sensor_width as f64
            && pixel_y >= 0.0
            && pixel_y < self.sensor_height as f64
        {
            Some((pixel_x, pixel_y))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use starfield::catalogs::StarData;
    use std::f64::consts::PI;

    /// Create a test star with the given parameters
    fn create_test_star(id: u64, ra: f64, dec: f64, magnitude: f64) -> StarData {
        // In the actual implementation, StarData contains a position field with RaDec
        // and other metadata like magnitude and b_v
        let position = Equatorial::from_degrees(ra, dec);
        StarData {
            id,
            position, // StarData stores position as RaDec
            magnitude,
            b_v: None,
        }
    }

    /// Check if two floating point values are approximately equal
    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    const ZERO_ZERO: Equatorial = Equatorial { ra: 0.0, dec: 0.0 };

    #[test]
    fn test_projector_creation() {
        StarProjector::new(
            &ZERO_ZERO, 0.001, // ~3.4 arcmin per pixel
            1920,  // Full HD width
            1080,  // Full HD height
        );
    }

    #[test]
    fn test_center_projection() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 1920, 1080);

        // Star at projection center should map to sensor center
        let center_star = Equatorial { ra: 0.0, dec: 0.0 };
        let (pixel_x, pixel_y) = projector.project(&center_star).unwrap();

        assert!((pixel_x - 960.0).abs() < 0.1);
        assert!((pixel_y - 540.0).abs() < 0.1);
    }

    #[test]
    fn test_center_field_access() {
        let test_center = Equatorial::from_degrees(120.0, -15.0);
        let projector = StarProjector::new(&test_center, 0.001, 1920, 1080);

        // Test that we can access the center field
        assert!((projector.center.ra_degrees() - 120.0).abs() < 1e-10);
        assert!((projector.center.dec_degrees() - (-15.0)).abs() < 1e-10);

        // Test that the center star maps to sensor center
        let (pixel_x, pixel_y) = projector.project(&test_center).unwrap();
        assert!((pixel_x - 960.0).abs() < 0.1);
        assert!((pixel_y - 540.0).abs() < 0.1);
    }

    #[test]
    fn test_four_corner_projection_easy() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.01, 100, 100);

        // Define stars at each corner of a 2x2 degree square centered on 0,0
        let star_top_left = Equatorial { ra: -0.1, dec: 0.1 };
        let star_top_right = Equatorial { ra: 0.1, dec: 0.1 };
        let star_bottom_left = Equatorial {
            ra: -0.1,
            dec: -0.1,
        };
        let star_bottom_right = Equatorial { ra: 0.1, dec: -0.1 };

        // Project each star
        let pixel_top_left = projector.project(&star_top_left).unwrap();
        let pixel_top_right = projector.project(&star_top_right).unwrap();
        let pixel_bottom_left = projector.project(&star_bottom_left).unwrap();
        let pixel_bottom_right = projector.project(&star_bottom_right).unwrap();

        // Assert that the projected pixels are close to the expected locations
        assert_relative_eq!(pixel_bottom_left.0, 40.0, epsilon = 0.1);
        assert_relative_eq!(pixel_bottom_left.1, 60.0, epsilon = 0.1);

        assert_relative_eq!(pixel_top_left.0, 40.0, epsilon = 0.1);
        assert_relative_eq!(pixel_top_left.1, 40.0, epsilon = 0.1);

        assert_relative_eq!(pixel_top_right.0, 60.0, epsilon = 0.1);
        assert_relative_eq!(pixel_top_right.1, 40.0, epsilon = 0.1);

        assert_relative_eq!(pixel_bottom_right.0, 60.0, epsilon = 0.1);
        assert_relative_eq!(pixel_bottom_right.1, 60.0, epsilon = 0.1);
    }

    #[test]
    fn test_rotates_unit_vectors() {
        // Generate a random unit vector using seeded RNG
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let point = Equatorial {
                ra: rng.gen_range(0.0..std::f64::consts::PI * 2.0),
                dec: rng.gen_range(-std::f64::consts::PI / 2.0..std::f64::consts::PI / 2.0),
            };

            let (width, height) = (101, 37);

            let projector = StarProjector::new(&point, 0.01, width, height);

            // Project the random unit vector
            let projected = projector.project(&point);

            assert!(
                projected.is_some(),
                "Projection failed for random unit vector"
            );

            let (px, py) = projected.unwrap();
            assert_relative_eq!(px, width as f64 / 2.0, epsilon = 0.001,);
            assert_relative_eq!(py, height as f64 / 2.0, epsilon = 0.001,);
        }
    }

    #[test]
    fn test_four_corner_projection_hard() {
        let projector = StarProjector::new(
            &Equatorial {
                ra: 0.0,
                dec: PI / 2.0,
            },
            0.01,
            100,
            100,
        );

        // Define stars at each corner of a 2x2 degree square centered on 0,0
        let eq_q1 = Equatorial {
            ra: 0.0_f64.to_radians(),
            dec: 89.9_f64.to_radians(),
        };
        let eq_q2 = Equatorial {
            ra: 90.0_f64.to_radians(),
            dec: 89.9_f64.to_radians(),
        };
        let eq_q3 = Equatorial {
            ra: 180.0_f64.to_radians(),
            dec: 89.9_f64.to_radians(),
        };
        let eq_q4 = Equatorial {
            ra: 270.0_f64.to_radians(),
            dec: 89.9_f64.to_radians(),
        };

        let eq_coords = vec![eq_q1, eq_q2, eq_q3, eq_q4];

        for coord in &eq_coords {
            let pixel = projector.project(coord);
            assert!(pixel.is_some(), "Projection failed for coord: {:?}", coord);

            let (w, h) = pixel.unwrap();

            let expected_off = 0.1_f64.to_radians() / 0.01;

            let dist = ((w - 50.0).powi(2) + (h - 50.0).powi(2)).sqrt();
            assert_relative_eq!(dist, expected_off, epsilon = 0.1);
        }
    }

    #[test]
    fn test_unbounded_vs_bounded_same_result_when_in_bounds() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 1920, 1080);

        // Star near center should be in bounds for both methods
        let star = Equatorial {
            ra: 0.001,
            dec: 0.001,
        };

        let bounded = projector.project(&star);
        let unbounded = projector.project_unbounded(&star);

        assert!(bounded.is_some());
        assert!(unbounded.is_some());

        let (bx, by) = bounded.unwrap();
        let (ux, uy) = unbounded.unwrap();

        assert_relative_eq!(bx, ux, epsilon = 1e-10);
        assert_relative_eq!(by, uy, epsilon = 1e-10);
    }

    #[test]
    fn test_unbounded_returns_coordinates_outside_sensor() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 100, 100);

        // Star far from center should be outside sensor bounds
        let far_star = Equatorial { ra: 1.0, dec: 1.0 };

        let bounded = projector.project(&far_star);
        let unbounded = projector.project_unbounded(&far_star);

        // Bounded should return None (outside sensor)
        assert!(bounded.is_none());

        // Unbounded should still return coordinates
        assert!(unbounded.is_some());
        let (x, y) = unbounded.unwrap();

        // Coordinates should be way outside sensor bounds
        assert!(!(0.0..100.0).contains(&x) || !(0.0..100.0).contains(&y));
    }

    #[test]
    fn test_unbounded_behind_camera_returns_none() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 100, 100);

        // Star behind the camera (opposite side of sky)
        let behind_star = Equatorial { ra: PI, dec: 0.0 };

        let bounded = projector.project(&behind_star);
        let unbounded = projector.project_unbounded(&behind_star);

        // Both should return None when star is behind camera
        assert!(bounded.is_none());
        assert!(unbounded.is_none());
    }

    #[test]
    fn test_unbounded_center_projection() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 1920, 1080);

        // Star at projection center
        let center_star = Equatorial { ra: 0.0, dec: 0.0 };
        let (pixel_x, pixel_y) = projector.project_unbounded(&center_star).unwrap();

        // Should map to sensor center
        assert_relative_eq!(pixel_x, 960.0, epsilon = 0.1);
        assert_relative_eq!(pixel_y, 540.0, epsilon = 0.1);
    }

    #[test]
    fn test_equatorial_randomizer_reproducible() {
        // Test that the randomizer produces the same sequence with the same seed
        let mut randomizer1 = EquatorialRandomizer::new(42);
        let mut randomizer2 = EquatorialRandomizer::new(42);

        for _ in 0..10 {
            let coord1 = randomizer1.generate();
            let coord2 = randomizer2.generate();

            assert_relative_eq!(coord1.ra_degrees(), coord2.ra_degrees(), epsilon = 1e-10);
            assert_relative_eq!(coord1.dec_degrees(), coord2.dec_degrees(), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_equatorial_randomizer_different_seeds() {
        // Test that different seeds produce different sequences
        let mut randomizer1 = EquatorialRandomizer::new(42);
        let mut randomizer2 = EquatorialRandomizer::new(43);

        let coord1 = randomizer1.generate();
        let coord2 = randomizer2.generate();

        // Should be different (extremely unlikely to be the same)
        assert!(
            coord1.ra_degrees() != coord2.ra_degrees()
                || coord1.dec_degrees() != coord2.dec_degrees()
        );
    }

    #[test]
    fn test_equatorial_randomizer_bounds() {
        // Test that generated coordinates are within expected bounds
        let mut randomizer = EquatorialRandomizer::new(123);

        for _ in 0..100 {
            let coord = randomizer.generate();
            let ra = coord.ra_degrees();
            let dec = coord.dec_degrees();

            // RA should be in [0, 360)
            assert!(ra >= 0.0 && ra < 360.0, "RA out of bounds: {}", ra);

            // Dec should be in [-90, 90]
            assert!(dec >= -90.0 && dec <= 90.0, "Dec out of bounds: {}", dec);
        }
    }

    #[test]
    fn test_equatorial_randomizer_uniform_distribution() {
        // Basic test for uniform distribution - not rigorous but catches obvious problems
        let mut randomizer = EquatorialRandomizer::new(456);
        let mut ra_values = Vec::new();
        let mut dec_values = Vec::new();

        for _ in 0..1000 {
            let coord = randomizer.generate();
            ra_values.push(coord.ra_degrees());
            dec_values.push(coord.dec_degrees());
        }

        // Check that we get values across the full range
        let ra_scan = MinMaxScan::new(&ra_values);
        let dec_scan = MinMaxScan::new(&dec_values);
        let (min_ra, max_ra) = ra_scan.min_max().unwrap();
        let (min_dec, max_dec) = dec_scan.min_max().unwrap();

        // Should span a significant portion of the range
        assert!(
            max_ra - min_ra > 200.0,
            "RA range too narrow: {} - {}",
            min_ra,
            max_ra
        );
        assert!(
            max_dec - min_dec > 100.0,
            "Dec range too narrow: {} - {}",
            min_dec,
            max_dec
        );

        // Rough check of means (should be around 180 for RA, 0 for Dec)
        let mean_ra: f64 = ra_values.iter().sum::<f64>() / ra_values.len() as f64;
        let mean_dec: f64 = dec_values.iter().sum::<f64>() / dec_values.len() as f64;

        assert!(
            mean_ra > 150.0 && mean_ra < 210.0,
            "RA mean suspicious: {}",
            mean_ra
        );
        assert!(
            mean_dec > -10.0 && mean_dec < 10.0,
            "Dec mean suspicious: {}",
            mean_dec
        );
    }
}

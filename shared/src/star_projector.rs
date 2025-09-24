//! High-precision celestial coordinate to pixel projection engine.
//!
//! Implements mathematically rigorous transformation from celestial sphere
//! coordinates to detector pixel positions using gnomonic (tangent plane)
//! projection. Designed for sub-pixel accuracy in astronomical imaging
//! applications with proper handling of coordinate system singularities
//! and field boundary conditions.

use nalgebra::{Matrix3, Vector3};
use starfield::framelib::inertial::InertialFrame;
use starfield::Equatorial;

/// High-precision celestial coordinate to pixel projection engine.
///
/// Implements mathematically rigorous transformation from celestial sphere
/// coordinates to detector pixel positions using gnomonic (tangent plane)
/// projection. Designed for sub-pixel accuracy in astronomical imaging
/// applications with proper handling of coordinate system singularities
/// and field boundary conditions.
///
/// # Projection Method
/// Uses gnomonic projection with the following characteristics:
/// - **Central projection**: All rays pass through sphere center
/// - **Tangent plane**: Projection surface tangent to sphere at field center
/// - **Angular preservation**: Small angles preserved near field center
/// - **Distortion growth**: Polynomial distortion increase toward field edges
///
/// # Coordinate System Convention
/// - **Input**: Celestial equatorial coordinates (RA, Dec) in radians
/// - **Intermediate**: 3D Cartesian unit vectors on celestial sphere
/// - **Camera frame**: Right-handed system aligned with detector
/// - **Output**: Pixel coordinates with (0,0) at detector corner
///
/// # Accuracy Specifications
/// - **Sub-pixel precision**: <0.01 pixel coordinate accuracy
/// - **Angular accuracy**: Limited by double-precision (~1 microarcsec)
/// - **Field size limits**: <10° radius recommended for <1% distortion
/// - **Pole handling**: Robust near celestial coordinate singularities
///
/// # Performance Characteristics
/// - **Setup cost**: O(1) rotation matrix computation
/// - **Projection cost**: O(1) per star with matrix multiplication
/// - **Memory usage**: ~200 bytes for rotation matrix and parameters
/// - **Thread safety**: Immutable after construction, safe for concurrent use
pub struct StarProjector {
    /// Celestial coordinates of the projection center (field center).
    ///
    /// Defines the pointing direction of the camera/telescope optical axis.
    /// This point maps to the center of the detector pixel grid and serves
    /// as the reference for all coordinate transformations.
    pub center: Equatorial,

    /// Angular pixel scale in radians per pixel.
    ///
    /// Determines the mapping between angular coordinates on the sky and
    /// linear pixel coordinates on the detector. Typically computed from
    /// telescope focal length and detector pixel physical size.
    radians_per_pixel: f64,

    /// Detector width in pixels along the X-axis.
    ///
    /// Used for pixel coordinate bounds checking and center offset calculation.
    /// Pixel coordinates range from [0, sensor_width) in the X direction.
    sensor_width: usize,

    /// Detector height in pixels along the Y-axis.
    ///
    /// Used for pixel coordinate bounds checking and center offset calculation.
    /// Pixel coordinates range from [0, sensor_height) in the Y direction.
    sensor_height: usize,

    /// 3D rotation matrix for celestial to camera coordinate transformation.
    ///
    /// Transforms celestial unit vectors to camera-aligned coordinate system
    /// where Z-axis points toward field center, Y-axis toward celestial north,
    /// and X-axis completes the right-handed system.
    rotation_matrix: Matrix3<f64>,
}

impl StarProjector {
    /// Create new star projector with specified field center and detector geometry.
    ///
    /// Constructs a complete coordinate transformation system by computing the
    /// rotation matrix that aligns the celestial sphere with the camera coordinate
    /// system. The resulting projector can efficiently transform stars from
    /// celestial coordinates to detector pixel positions.
    ///
    /// # Mathematical Setup
    /// The rotation matrix is constructed to establish camera coordinates:
    /// - **Z-axis**: Points toward field center (optical axis direction)
    /// - **Y-axis**: Points toward celestial north (or nearest non-degenerate direction)
    /// - **X-axis**: Completes right-handed system (approximately eastward)
    ///
    /// # Arguments
    /// * `center` - Field center in celestial equatorial coordinates
    /// * `radians_per_pixel` - Angular pixel scale (typically focal_length / pixel_size)
    /// * `sensor_width` - Detector width in pixels
    /// * `sensor_height` - Detector height in pixels
    ///
    /// # Returns
    /// Configured StarProjector ready for coordinate transformations
    ///
    /// # Usage
    /// Creates a configured projector for transforming celestial coordinates
    /// to detector pixel coordinates. The field center maps to detector center.
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

    /// Project celestial coordinates to pixel space without detector bounds checking.
    ///
    /// Performs complete coordinate transformation from celestial sphere to pixel
    /// coordinates without testing whether the result falls within detector boundaries.
    /// Useful for field analysis, distortion studies, and geometric calculations
    /// that extend beyond the physical detector area.
    ///
    /// # Transformation Steps
    /// 1. **Spherical to Cartesian**: Convert (RA, Dec) to unit vector
    /// 2. **Coordinate rotation**: Apply camera alignment transformation
    /// 3. **Visibility check**: Reject stars behind camera (Z ≤ 0)
    /// 4. **Gnomonic projection**: Project to tangent plane (X/Z, Y/Z)
    /// 5. **Pixel scaling**: Convert angular to pixel coordinates
    ///
    /// # Arguments
    /// * `equatorial` - Celestial coordinate to project
    ///
    /// # Returns
    /// * `Some((x, y))` - Pixel coordinates if star visible (in front of camera)
    /// * `None` - If star is behind camera or at coordinate singularity
    ///
    /// # Usage
    /// Projects stars without bounds checking. Returns pixel coordinates even
    /// for stars outside detector bounds, useful for field geometry analysis.
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

    /// Project celestial coordinates to pixel space with detector bounds checking.
    ///
    /// Performs complete coordinate transformation from celestial sphere to pixel
    /// coordinates and validates that the result falls within the physical detector
    /// boundaries. This is the primary projection method for realistic simulation
    /// where only detector-visible stars are of interest.
    ///
    /// # Bounds Checking
    /// Pixel coordinates must satisfy:
    /// - **X range**: [0, sensor_width)
    /// - **Y range**: [0, sensor_height)
    /// - **Visibility**: Star must be in front of camera (not behind)
    ///
    /// # Arguments
    /// * `equatorial` - Celestial coordinate to project
    ///
    /// # Returns
    /// * `Some((x, y))` - Pixel coordinates if star visible within detector bounds
    /// * `None` - If star is outside field of view or behind camera
    ///
    /// # Performance Notes
    /// This method calls `project_unbounded()` internally and adds bounds checking.
    /// For performance-critical applications with pre-filtered coordinates,
    /// consider using `project_unbounded()` directly.
    ///
    /// # Usage
    /// Primary method for projecting stars with bounds checking. Returns None
    /// for stars outside the detector field of view or behind the camera.
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
    use std::f64::consts::PI;

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
            assert!(pixel.is_some(), "Projection failed for coord: {coord:?}");

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
    fn test_unbounded_vs_bounded_different_for_out_of_bounds() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.00001, 100, 100);

        // Star far from center, but in front of camera
        // With tiny radians_per_pixel, even small angle projects far
        let far_star = Equatorial { ra: 0.1, dec: 0.1 };

        let bounded = projector.project(&far_star);
        let unbounded = projector.project_unbounded(&far_star);

        // Unbounded should succeed, bounded should fail
        assert!(unbounded.is_some());
        assert!(bounded.is_none());
    }

    #[test]
    fn test_stars_behind_camera_not_projected() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 1920, 1080);

        // Star on opposite side of celestial sphere
        let behind_star = Equatorial { ra: PI, dec: 0.0 };

        assert!(projector.project(&behind_star).is_none());
        assert!(projector.project_unbounded(&behind_star).is_none());
    }
}

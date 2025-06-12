//! Utility functions for star catalog manipulation and coordinate transformations
//!
//! This module contains utility functions for working with star catalogs and
//! transforming between coordinate systems (equatorial to pixel, etc.)

use nalgebra::Matrix3;
use starfield::Equatorial;
use starfield::{catalogs::StarData, framelib::inertial::InertialFrame};

use nalgebra::Vector3;

/// Star projector that maps celestial coordinates to image pixels
pub struct StarProjector {
    /// Center point of the projection in right ascension/declination (radians)
    center: Equatorial,

    /// Angular resolution in radians per pixel
    radians_per_pixel: f64,
    /// Sensor dimensions in pixels
    sensor_width: u32,
    sensor_height: u32,
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
        sensor_width: u32,
        sensor_height: u32,
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

/// Converts celestial equatorial coordinates to image pixel coordinates with sub-pixel precision.
///
/// This function transforms a sky position (RA/Dec) to pixel coordinates (x,y) in an image,
/// based on the field center, field of view, and image dimensions. The transformation uses
/// a simple tangent plane projection suitable for small fields of view.
///
/// # Arguments
/// * `position` - Sky coordinates (RA/Dec) of the object to transform
/// * `center` - Sky coordinates (Equatorial) of the field center
/// * `fov_deg` - Field of view in degrees
/// * `image_width` - Width of the image in pixels
/// * `image_height` - Height of the image in pixels
///
/// # Returns
/// * Tuple (x,y) containing pixel coordinates where:
///   - x increases toward increasing RA (typically right in the image)
///   - y increases toward increasing Dec (typically up in the image)
///   - The center of the image is at (image_width/2, image_height/2)
///   - Coordinates can be outside the image bounds for objects outside the field
pub fn equatorial_to_pixel(
    position: &Equatorial,
    center: &Equatorial,
    fov_deg: f64,
    image_width: usize,
    image_height: usize,
) -> (f64, f64) {
    // Convert to radians
    let ra_rad = position.ra_degrees().to_radians();
    let dec_rad = position.dec_degrees().to_radians();
    let center_ra_rad = center.ra_degrees().to_radians();
    let center_dec_rad = center.dec_degrees().to_radians();

    // Calculate projection factors
    // TODO(meawoppl) handle modulo of ra/dec here to make sure
    // things work properly across the 0/360 degree boundary and poles
    let x_factor = ra_rad - center_ra_rad;
    let y_factor = dec_rad - center_dec_rad;

    // Scale to pixel coordinates
    let fov_rad = fov_deg.to_radians();
    let x_pixels_per_rad = image_width as f64 / fov_rad;
    let y_pixels_per_rad = image_height as f64 / fov_rad;

    // Convert to pixel coordinates (center of image is at center_ra, center_dec)
    let x = (x_factor * x_pixels_per_rad) + (image_width as f64 / 2.0);
    let y = (y_factor * y_pixels_per_rad) + (image_height as f64 / 2.0);

    (x, y)
}

/// Saves a formatted text file containing detailed information about stars in the field.
///
/// This function creates a human-readable and machine-parsable text file containing
/// information about stars in the current field of view. The file includes:
/// - Header with field metadata (center coordinates, FOV, telescope/sensor info)
/// - One row per star with ID, RA, Dec, magnitude, B-V color index, and pixel coordinates
/// - Stars are sorted by magnitude (brightest first)
///
/// # Arguments
/// * `stars` - Slice of references to StarData objects to include in the list
/// * `center` - Sky coordinates (Equatorial) of the field center  
/// * `fov_deg` - Field of view in degrees
/// * `telescope_name` - Name of the telescope used
/// * `sensor_name` - Name of the sensor used
/// * `image_width` - Width of the image in pixels
/// * `image_height` - Height of the image in pixels
/// * `path` - Output file path
///
/// # Returns
/// * Result indicating success or an error with message
///
/// # Format
/// The output file has a CSV-like format with header lines prefixed by '#'.
/// Each star entry contains: ID, RA(°), Dec(°), Magnitude, B-V, X(px), Y(px)
pub fn save_star_list(
    stars: &[&StarData],
    center: &Equatorial,
    fov_deg: f64,
    telescope_name: &str,
    sensor_name: &str,
    image_width: usize,
    image_height: usize,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;

    // Write header
    writeln!(
        file,
        "# Star List for field centered at RA={:.4}°, Dec={:.4}°, FOV={:.4}°",
        center.ra_degrees(),
        center.dec_degrees(),
        fov_deg
    )?;
    writeln!(
        file,
        "# Telescope: {}, Sensor: {}",
        telescope_name, sensor_name
    )?;
    writeln!(file, "# Total stars in field: {}", stars.len())?;
    writeln!(file, "#")?;
    writeln!(file, "# ID, RA(°), Dec(°), Magnitude, B-V, X(px), Y(px)")?;

    // Get stars sorted by magnitude (brightest first)
    let mut sorted_stars = stars.to_vec();
    sorted_stars.sort_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap());

    // Write star data
    for star in sorted_stars {
        // Calculate pixel coordinates
        // Create RaDec for star
        let star_radec = Equatorial::from_degrees(star.ra_deg(), star.dec_deg());

        let (x, y) = equatorial_to_pixel(&star_radec, center, fov_deg, image_width, image_height);

        // Format B-V value or "N/A" if None
        let b_v_str = if let Some(b_v) = star.b_v {
            format!("{:.2}", b_v)
        } else {
            "N/A".to_string()
        };

        writeln!(
            file,
            "{}, {:.6}, {:.6}, {:.2}, {}, {:.2}, {:.2}",
            star.id,
            star.ra_deg(),
            star.dec_deg(),
            star.magnitude,
            b_v_str,
            x,
            y
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::f64::consts::PI;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use tempfile::NamedTempFile;

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

    #[test]
    fn test_equatorial_to_pixel_center() {
        // Test that the center of the field maps to the center of the image
        let center = Equatorial::from_degrees(100.0, 10.0);
        let position = center; // Same position as center
        let fov_deg = 5.0;
        let image_width = 1000;
        let image_height = 800;

        let (x, y) = equatorial_to_pixel(&position, &center, fov_deg, image_width, image_height);

        // Center of image is (width/2, height/2)
        assert!(approx_eq(x, image_width as f64 / 2.0, 1e-6));
        assert!(approx_eq(y, image_height as f64 / 2.0, 1e-6));
    }

    #[test]
    fn test_equatorial_to_pixel_offset() {
        // Test that positions offset from center map correctly
        let center = Equatorial::from_degrees(100.0, 10.0);
        let fov_deg = 5.0;
        let image_width = 1000;
        let image_height = 800;

        // Position offset by 1 degree in RA (will move in x direction)
        let position_ra_offset = Equatorial::from_degrees(101.0, 10.0);
        let (x1, _y1) = equatorial_to_pixel(
            &position_ra_offset,
            &center,
            fov_deg,
            image_width,
            image_height,
        );

        // Position offset by 1 degree in Dec (will move in y direction)
        let position_dec_offset = Equatorial::from_degrees(100.0, 11.0);
        let (_x2, y2) = equatorial_to_pixel(
            &position_dec_offset,
            &center,
            fov_deg,
            image_width,
            image_height,
        );

        // The offsets should move by a significant amount of pixels
        // Degree offset / FOV * image dimension
        let expected_x_offset = 1.0 / fov_deg * image_width as f64;
        let expected_y_offset = 1.0 / fov_deg * image_height as f64;

        assert!(x1 > image_width as f64 / 2.0); // RA increases to the right
        assert!(approx_eq(
            x1 - image_width as f64 / 2.0,
            expected_x_offset,
            1.0
        ));

        assert!(y2 > image_height as f64 / 2.0); // Dec increases upward
        assert!(approx_eq(
            y2 - image_height as f64 / 2.0,
            expected_y_offset,
            1.0
        ));
    }

    #[test]
    fn test_save_star_list() {
        // Create test stars
        let star1 = create_test_star(1, 100.0, 10.0, 5.0);
        let star2 = create_test_star(2, 101.0, 10.0, 6.0);
        let stars = [&star1, &star2];

        // Create temporary file for the star list
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();

        let center = Equatorial::from_degrees(100.0, 10.0);
        let fov_deg = 5.0;
        let telescope_name = "Test Telescope";
        let sensor_name = "Test Sensor";
        let image_width = 1000;
        let image_height = 800;

        // Save the star list
        save_star_list(
            &stars,
            &center,
            fov_deg,
            telescope_name,
            sensor_name,
            image_width,
            image_height,
            path,
        )
        .unwrap();

        // Read the file back to verify its contents
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        // Verify header
        assert!(lines[0].contains("RA=100.0000°, Dec=10.0000°, FOV=5.0000°"));
        assert!(lines[1].contains("Test Telescope"));
        assert!(lines[1].contains("Test Sensor"));
        assert!(lines[2].contains("2")); // 2 stars

        // Verify star entries (sorted by magnitude brightest first)
        assert!(lines[5].contains("1")); // ID
        assert!(lines[5].contains("5.00")); // Magnitude

        assert!(lines[6].contains("2")); // ID
        assert!(lines[6].contains("6.00")); // Magnitude
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
}

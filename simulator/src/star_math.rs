//! Utility functions for star catalog manipulation and coordinate transformations
//!
//! This module contains utility functions for working with star catalogs and
//! transforming between coordinate systems (equatorial to pixel, etc.)

use starfield::catalogs::StarData;
use starfield::RaDec;

/// Convert equatorial coordinates to pixel coordinates with sub-pixel precision
pub fn equatorial_to_pixel(
    position: &RaDec,
    center: &RaDec,
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

/// Save a text file with visible star information
pub fn save_star_list(
    stars: &[&StarData],
    center: &RaDec,
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
        let star_radec = RaDec::from_degrees(star.ra_deg(), star.dec_deg());

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
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use tempfile::NamedTempFile;

    /// Create a test star with the given parameters
    fn create_test_star(id: u64, ra: f64, dec: f64, magnitude: f64) -> StarData {
        // In the actual implementation, StarData contains a position field with RaDec
        // and other metadata like magnitude and b_v
        let position = RaDec::from_degrees(ra, dec);
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
        let center = RaDec::from_degrees(100.0, 10.0);
        let position = center.clone(); // Same position as center
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
        let center = RaDec::from_degrees(100.0, 10.0);
        let fov_deg = 5.0;
        let image_width = 1000;
        let image_height = 800;

        // Position offset by 1 degree in RA (will move in x direction)
        let position_ra_offset = RaDec::from_degrees(101.0, 10.0);
        let (x1, _y1) = equatorial_to_pixel(
            &position_ra_offset,
            &center,
            fov_deg,
            image_width,
            image_height,
        );

        // Position offset by 1 degree in Dec (will move in y direction)
        let position_dec_offset = RaDec::from_degrees(100.0, 11.0);
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

        let center = RaDec::from_degrees(100.0, 10.0);
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
}

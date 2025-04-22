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

/// Filter stars to only include those within a rectangular field of view
pub fn filter_stars_in_rectangle<'a>(
    stars: &[&'a StarData],
    center: &RaDec,
    fov_deg: f64,
    image_width: usize,
    image_height: usize,
) -> Vec<&'a StarData> {
    stars
        .iter()
        .filter(|star| {
            // Create RaDec for star
            let star_radec = RaDec::from_degrees(star.ra_deg(), star.dec_deg());

            let (x, y) =
                equatorial_to_pixel(&star_radec, center, fov_deg, image_width, image_height);
            x >= 0.0 && y >= 0.0 && x < image_width as f64 && y < image_height as f64
        })
        .copied()
        .collect()
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

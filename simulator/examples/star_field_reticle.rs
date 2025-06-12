//! Star field renderer with 3-degree square field of view and X reticle overlay
//!
//! This example creates a simple star field visualization from a catalog and applies
//! an X reticle overlay to help with star identification.

use image::{DynamicImage, Rgb, RgbImage};
use simulator::image_proc::overlay::draw_stars_with_x_markers;
use starfield::catalogs::binary_catalog::{BinaryCatalog, MinimalStar};
use starfield::catalogs::StarPosition;
use std::collections::HashMap;
use std::path::PathBuf;

/// Main function to render a star field with X reticle overlay
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Default values for star field centered on Sirius (brightest star in sky)
    let mut ra_deg = 101.3; // Sirius coordinates
    let mut dec_deg = -16.7;
    let mut fov_deg = 3.0; // 3-degree square field of view
    let mut catalog_path = "../gaia_mag16_multi.bin".to_string();
    let mut output_path = "test_output/star_field_reticle.png";
    let mut max_magnitude = 9.0; // Maximum (dimmest) star magnitude to include

    // Parse arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ra" => {
                if i + 1 < args.len() {
                    ra_deg = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --ra".into());
                }
            }
            "--dec" => {
                if i + 1 < args.len() {
                    dec_deg = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --dec".into());
                }
            }
            "--fov" => {
                if i + 1 < args.len() {
                    fov_deg = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --fov".into());
                }
            }
            "--catalog" => {
                if i + 1 < args.len() {
                    catalog_path = args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing value for --catalog".into());
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = &args[i + 1];
                    i += 2;
                } else {
                    return Err("Missing value for --output".into());
                }
            }
            "--mag-limit" => {
                if i + 1 < args.len() {
                    max_magnitude = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --mag-limit".into());
                }
            }
            _ => {
                println!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    // Show parameters
    println!("Star Field Renderer with X Reticle");
    println!("=================================");
    println!("RA: {:.4}°", ra_deg);
    println!("Dec: {:.4}°", dec_deg);
    println!("FOV: {:.4}° (square)", fov_deg);
    println!("Catalog: {}", catalog_path);
    println!("Magnitude limit: {}", max_magnitude);
    println!("Output: {}", output_path);

    // Load star catalog
    println!("Loading star catalog from {}...", catalog_path);
    let catalog_path = PathBuf::from(catalog_path);
    let catalog = BinaryCatalog::load(&catalog_path)?;

    println!("Loaded catalog with {} stars", catalog.len());

    // Filter stars by magnitude
    let stars: Vec<&MinimalStar> = catalog
        .stars()
        .iter()
        .filter(|star| star.magnitude <= max_magnitude)
        .collect();

    println!(
        "Found {} stars brighter than magnitude {}",
        stars.len(),
        max_magnitude
    );

    // Render the star field visualization
    println!("Rendering star field visualization...");

    let visualization = render_starfield_with_reticle(&stars, ra_deg, dec_deg, fov_deg)?;

    // Create output directory if it doesn't exist
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Save the image
    visualization.save(output_path)?;
    println!("Visualization saved to: {}", output_path);

    Ok(())
}

/// Convert equatorial coordinates to pixel coordinates
fn equatorial_to_pixel(
    ra: f64,
    dec: f64,
    center_ra: f64,
    center_dec: f64,
    fov_deg: f64,
    image_width: usize,
    image_height: usize,
) -> (f64, f64) {
    // Convert to radians
    let ra_rad = ra.to_radians();
    let dec_rad = dec.to_radians();
    let center_ra_rad = center_ra.to_radians();
    let center_dec_rad = center_dec.to_radians();

    // Calculate projection factors
    let x_factor = ra_rad - center_ra_rad;
    let y_factor = dec_rad - center_dec_rad;

    // Scale to pixel coordinates
    let fov_rad = fov_deg.to_radians();
    let x_pixels_per_rad = image_width as f64 / fov_rad;
    let y_pixels_per_rad = image_height as f64 / fov_rad;

    // Convert to pixel coordinates with center of image as origin
    let x = (x_factor * x_pixels_per_rad) + (image_width as f64 / 2.0);
    let y = (y_factor * y_pixels_per_rad) + (image_height as f64 / 2.0);

    (x, y)
}

/// Render a star field with X reticle overlay
fn render_starfield_with_reticle(
    stars: &[&MinimalStar],
    ra_deg: f64,
    dec_deg: f64,
    fov_deg: f64,
) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    // Set up image dimensions (square aspect ratio)
    let image_size = 1200;
    let image_width = image_size;
    let image_height = image_size;

    // Create an empty black image
    let mut rgb_image = RgbImage::new(image_width as u32, image_height as u32);
    for pixel in rgb_image.pixels_mut() {
        *pixel = Rgb([0, 0, 0]);
    }

    println!("Filtering stars in field of view...");

    // Track stars for X marker overlay
    let mut star_map = HashMap::new();
    let mut visible_star_count = 0;
    let mut bright_star_count = 0;

    // Process each star
    for star in stars {
        let magnitude = star.magnitude;

        // Check if star is in field of view (simplified calculation)
        let ra = star.ra();
        let dec = star.dec();

        // Generous boundary for stars near the field
        if (ra - ra_deg).abs() > fov_deg * 1.2 || (dec - dec_deg).abs() > fov_deg * 1.2 {
            continue;
        }

        // Convert position to pixel coordinates for precise filtering
        let (x, y) =
            equatorial_to_pixel(ra, dec, ra_deg, dec_deg, fov_deg, image_width, image_height);

        // Skip stars outside the image bounds
        if x < 0.0 || x >= image_width as f64 || y < 0.0 || y >= image_height as f64 {
            continue;
        }

        visible_star_count += 1;

        // Add all visible stars to the map, but only bright ones get X markers
        if magnitude < 8.0 {
            bright_star_count += 1;
            let label = format!("S{} (m{:.1})", bright_star_count, magnitude);

            // Calculate apparent diameter based on magnitude
            // Brighter stars appear larger
            let base_size = 5.0;
            let diameter = base_size * (1.0 / (magnitude + 2.0).powf(0.5));

            star_map.insert(label, (y, x, diameter));
        }
    }

    println!("Found {} stars in field of view", visible_star_count);
    println!(
        "Adding X reticle markers to {} bright stars",
        bright_star_count
    );

    // Convert to DynamicImage
    let base_image = DynamicImage::ImageRgb8(rgb_image);

    // Draw X markers on stars
    let marker_color = (0, 255, 0); // Green X markers
    let arm_length_factor = 2.0; // X arms are 2x the diameter of the star

    let result = draw_stars_with_x_markers(&base_image, &star_map, marker_color, arm_length_factor);

    Ok(result)
}

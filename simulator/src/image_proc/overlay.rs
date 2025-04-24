//! Overlay drawing utilities for visualizing image processing results
//!
//! This module provides functions for drawing overlays on images,
//! such as bounding boxes for star detection visualization.

use image::{DynamicImage, Rgb, RgbImage};
use std::collections::HashMap;
use tiny_skia::{Pixmap, Transform};
use usvg::{self, fontdb, Options, Tree};

/// Draw bounding boxes on an image
///
/// # Arguments
/// * `image` - Original image to draw on
/// * `bboxes` - Vector of bounding boxes as (min_row, min_col, max_row, max_col)
/// * `color` - RGB color tuple for the bounding boxes
/// * `labels` - Optional vector of labels to draw next to each box
/// * `circles` - Optional vector of (center_row, center_col, diameter) to draw circles
/// * `circle_color` - Optional color for the circles (uses color if None)
///
/// # Returns
/// * Image with bounding boxes drawn
pub fn draw_bounding_boxes(
    image: &DynamicImage,
    bboxes: &[(usize, usize, usize, usize)],
    color: (u8, u8, u8),
    labels: Option<&[String]>,
    circles: Option<&[(f64, f64, f64)]>,
    circle_color: Option<(u8, u8, u8)>,
) -> DynamicImage {
    let width = image.width();
    let height = image.height();

    // Create an SVG string with the bounding boxes
    let mut svg_data = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
        "#,
        width, height
    );

    let (r, g, b) = color;
    let color_str = format!("#{:02x}{:02x}{:02x}", r, g, b);

    // Get circle color (use box color if not specified)
    let circle_color_str = if let Some(circle_color) = circle_color {
        let (cr, cg, cb) = circle_color;
        format!("#{:02x}{:02x}{:02x}", cr, cg, cb)
    } else {
        color_str.clone()
    };

    // Draw bounding boxes first
    for (i, &(min_row, min_col, max_row, max_col)) in bboxes.iter().enumerate() {
        // Convert from array indices (row, col) to image coordinates (x, y)
        let x = min_col as f32;
        let y = min_row as f32;
        let box_width = (max_col - min_col) as f32;
        let box_height = (max_row - min_row) as f32;

        svg_data.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="{}" stroke-width="2"/>"#,
            x, y, box_width, box_height, color_str
        ));

        // Add label if provided
        if let Some(labels) = labels {
            if i < labels.len() {
                let text_x = x;
                let text_y = if y > 15.0 {
                    y - 5.0
                } else {
                    y + box_height + 15.0
                };

                svg_data.push_str(&format!(
                    r#"<text x="{}" y="{}" font-family="monospace" font-size="12" fill="{}">{}</text>"#,
                    text_x, text_y, color_str, labels[i]
                ));
            }
        }
    }

    // Draw circles if provided
    if let Some(circles) = circles {
        for &(center_row, center_col, diameter) in circles {
            // Convert from array indices (row, col) to image coordinates (x, y)
            // Use fractional part to maintain sub-pixel precision
            let cx = center_col as f32;
            let cy = center_row as f32;
            let radius = (diameter / 2.0) as f32;

            // Draw circle with solid stroke style as requested
            svg_data.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="{}" fill="none" stroke="{}" stroke-width="1.5" />"#,
                cx, cy, radius, circle_color_str
            ));
        }
    }

    svg_data.push_str("</svg>");

    overlay_to_image(image, &svg_data)
}

/// Simplified version of draw_bounding_boxes without labels or circles
///
/// This maintains backward compatibility with existing code
pub fn draw_simple_boxes(
    image: &DynamicImage,
    bboxes: &[(usize, usize, usize, usize)],
    color: (u8, u8, u8),
) -> DynamicImage {
    draw_bounding_boxes(image, bboxes, color, None, None, None)
}

/// Draw stars with both bounding boxes and size circles
///
/// This is a convenience function for drawing both bounding boxes and circles
/// showing the actual star sizes based on moment analysis.
///
/// # Arguments
/// * `image` - Original image to draw on
/// * `bboxes` - Vector of bounding boxes as (min_row, min_col, max_row, max_col)
/// * `box_color` - RGB color tuple for the bounding boxes
/// * `stars` - Vector of star detections with position and diameter information
/// * `circle_color` - RGB color tuple for the circles (separate from box color)
///
/// # Returns
/// * Image with both bounding boxes and star size circles
pub fn draw_stars_with_sizes(
    image: &DynamicImage,
    bboxes: &[(usize, usize, usize, usize)],
    box_color: (u8, u8, u8),
    stars: &[(f64, f64, f64)], // (y, x, diameter)
    circle_color: (u8, u8, u8),
) -> DynamicImage {
    // Format diameters as labels
    let labels: Vec<String> = bboxes
        .iter()
        .enumerate()
        .map(|(i, _)| format!("Box {}", i + 1))
        .collect();

    draw_bounding_boxes(
        image,
        bboxes,
        box_color,
        Some(&labels),
        Some(stars),
        Some(circle_color),
    )
}

/// Draw stars with X markers extending from the star diameter and custom labels
///
/// # Arguments
/// * `image` - Original image to draw on
/// * `stars` - Map of labels to star positions and diameters: HashMap<String, (y, x, diameter)>
/// * `marker_color` - RGB color tuple for the X markers
/// * `arm_length_factor` - Factor to multiply with star diameter to determine arm length
///                         (1.0 = same as diameter, 0.5 = half diameter)
///                         A minimum of 1 pixel is enforced to ensure visibility
///
/// # Returns
/// * Image with X markers and labeled stars
pub fn draw_stars_with_x_markers(
    image: &DynamicImage,
    stars: &HashMap<String, (f64, f64, f64)>, // Label -> (y, x, diameter)
    marker_color: (u8, u8, u8),
    arm_length_factor: f32,
) -> DynamicImage {
    let width = image.width();
    let height = image.height();

    // Create an SVG string with the X markers
    let mut svg_data = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
        "#,
        width, height
    );

    // Convert color to SVG format
    let (r, g, b) = marker_color;
    let color_str = format!("#{:02x}{:02x}{:02x}", r, g, b);

    // Draw X markers and labels for each star
    for (label, &(center_row, center_col, diameter)) in stars.iter() {
        // Convert from array indices (row, col) to image coordinates (x, y)
        let cx = center_col as f32;
        let cy = center_row as f32;
        let radius = (diameter / 2.0) as f32;

        // Calculate the end points of the X marker arms
        // Direction vectors for the four arm directions (normalized)
        let directions = [
            (1.0, 1.0),   // Down-right
            (1.0, -1.0),  // Down-left
            (-1.0, 1.0),  // Up-right
            (-1.0, -1.0), // Up-left
        ];

        for (j, &(dy, dx)) in directions.iter().enumerate() {
            // Normalize to unit vector
            let dx_f32 = dx as f32;
            let dy_f32 = dy as f32;
            let mag = (dx_f32 * dx_f32 + dy_f32 * dy_f32).sqrt();
            let nx = dx_f32 / mag;
            let ny = dy_f32 / mag;

            // Start point on circle perimeter
            let start_x = cx + nx * radius;
            let start_y = cy + ny * radius;

            // Calculate arm length based on star diameter and factor
            // Ensure minimum arm length of 1 pixel so they don't disappear for tiny stars
            let actual_arm_length = (radius * arm_length_factor).max(1.0);

            // End point after arm_length
            let end_x = start_x + nx * actual_arm_length;
            let end_y = start_y + ny * actual_arm_length;

            // Draw the arm line with increased width for better visibility
            svg_data.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2.5" />"#,
                start_x, start_y, end_x, end_y, color_str
            ));

            // Add star label at the end of top-right arm (index 2)
            if j == 2 {
                // Up-right direction
                // Position text at the end of the arm with offset (using a fixed offset of 10 pixels)
                let text_x = end_x + nx * 10.0;
                let text_y = end_y + ny * 10.0;

                // Draw the label text
                svg_data.push_str(&format!(
                    r#"<text x="{}" y="{}" font-size="12" font-weight="bold" text-anchor="start" dominant-baseline="central" fill="{}">{}</text>"#,
                    text_x, text_y, color_str, label
                ));
            }
        }

        // Draw small dot at center for better visibility
        svg_data.push_str(&format!(
            r#"<circle cx="{}" cy="{}" r="1" fill="{}" />"#,
            cx, cy, color_str
        ));
    }

    svg_data.push_str("</svg>");
    overlay_to_image(image, &svg_data)
}

/// Convert an SVG overlay to an image
///
/// # Arguments
/// * `image` - Original image
/// * `svg_data` - SVG string to render
///
/// # Returns
/// * Image with SVG overlay rendered on top
pub fn overlay_to_image(image: &DynamicImage, svg_data: &str) -> DynamicImage {
    // Parse SVG data using default options with detailed font setup

    // Create font database with system fonts
    let mut fontdb = fontdb::Database::new();
    fontdb.load_system_fonts();

    // Create options with the font database and high quality rendering settings
    let options = Options {
        fontdb: std::sync::Arc::new(fontdb),
        font_family: "DejaVu Sans".to_string(), // Widely available on Linux
        text_rendering: usvg::TextRendering::GeometricPrecision,
        shape_rendering: usvg::ShapeRendering::GeometricPrecision,
        image_rendering: usvg::ImageRendering::OptimizeQuality,
        ..Default::default()
    };

    // Parse SVG with our custom options
    let svg_tree = match Tree::from_str(svg_data, &options) {
        Ok(tree) => tree,
        Err(e) => {
            panic!("SVG parsing failed: {:?}", e);
        }
    };

    // Create a pixel buffer for the overlay
    let mut pixmap = match Pixmap::new(image.width(), image.height()) {
        Some(p) => p,
        None => {
            panic!("Pixmap creation failed");
        }
    };

    // Render SVG to the pixel buffer
    resvg::render(&svg_tree, Transform::identity(), &mut pixmap.as_mut());

    // Convert original image to RGB format
    let rgb_image: image::ImageBuffer<Rgb<u8>, Vec<u8>> = image.to_rgb8();
    let mut output_buffer = RgbImage::new(image.width(), image.height());

    // Combine original image with overlay
    for (x, y, pixel) in output_buffer.enumerate_pixels_mut() {
        let source_pixel = rgb_image.get_pixel(x, y);

        // Get SVG pixel at this position
        if let Some(overlay_pixel) = pixmap.pixel(x, y) {
            // Only apply overlay if it's not fully transparent
            if overlay_pixel.alpha() > 0 {
                *pixel = Rgb([
                    blend_channel(source_pixel[0], overlay_pixel.red(), overlay_pixel.alpha()),
                    blend_channel(
                        source_pixel[1],
                        overlay_pixel.green(),
                        overlay_pixel.alpha(),
                    ),
                    blend_channel(source_pixel[2], overlay_pixel.blue(), overlay_pixel.alpha()),
                ]);
            } else {
                *pixel = *source_pixel;
            }
        } else {
            *pixel = *source_pixel;
        }
    }

    DynamicImage::ImageRgb8(output_buffer)
}

// Helper function to blend color channels based on alpha
fn blend_channel(base: u8, overlay: u8, alpha: u8) -> u8 {
    let base_f = base as f32;
    let overlay_f = overlay as f32;
    let alpha_f = alpha as f32 / 255.0;

    // Alpha blending formula
    (base_f * (1.0 - alpha_f) + overlay_f * alpha_f).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};

    use crate::image_proc::thresholding::{
        apply_threshold, connected_components, get_bounding_boxes, otsu_threshold,
    };

    #[test]
    fn test_draw_bounding_boxes() {
        // Create a test image with some bright spots
        let width = 100;
        let height = 100;
        let mut test_image = ImageBuffer::new(width, height);

        // Create a few bright spots to simulate stars
        for x in 20..30 {
            for y in 20..25 {
                test_image.put_pixel(x, y, Luma([200]));
            }
        }

        for x in 60..65 {
            for y in 60..70 {
                test_image.put_pixel(x, y, Luma([200]));
            }
        }

        // Convert to dynamic image
        let dynamic_image = DynamicImage::ImageLuma8(test_image);

        // Run image processing pipeline
        use image::GenericImageView; // Import trait for test

        let view = ndarray::Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
            dynamic_image.get_pixel(x as u32, y as u32).0[0] as f64 / 255.0
        });

        let threshold = otsu_threshold(&view.view());
        let binary = apply_threshold(&view.view(), threshold);
        let labels = connected_components(&binary.view());
        let aabbs = get_bounding_boxes(&labels.view());
        let bboxes = crate::image_proc::aabbs_to_tuples(&aabbs);

        // Draw bounding boxes with simple interface
        let result = draw_simple_boxes(&dynamic_image, &bboxes, (255, 0, 0));

        // Verify the result is a valid image with the correct dimensions
        assert_eq!(result.width(), width);
        assert_eq!(result.height(), height);

        // Test with labels
        let labels = vec!["Star 1".to_string(), "Star 2".to_string()];
        let result_with_labels = draw_bounding_boxes(
            &dynamic_image,
            &bboxes,
            (0, 255, 0),
            Some(&labels),
            None,
            None,
        );

        assert_eq!(result_with_labels.width(), width);
        assert_eq!(result_with_labels.height(), height);
    }
}

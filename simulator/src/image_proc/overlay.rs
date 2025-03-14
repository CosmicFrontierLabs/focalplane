//! Overlay drawing utilities for visualizing image processing results
//!
//! This module provides functions for drawing overlays on images,
//! such as bounding boxes for star detection visualization.

use image::{DynamicImage, Rgb, RgbImage};
use tiny_skia::{Pixmap, Transform};
use usvg::{self, Options, Tree};

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
                    r#"<text x="{}" y="{}" font-family="sans-serif" font-size="12" fill="{}">{}</text>"#,
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

/// Convert an SVG overlay to an image
///
/// # Arguments
/// * `image` - Original image
/// * `svg_data` - SVG string to render
///
/// # Returns
/// * Image with SVG overlay rendered on top
pub fn overlay_to_image(image: &DynamicImage, svg_data: &str) -> DynamicImage {
    // Parse SVG data
    let svg_tree = Tree::from_str(svg_data, &Options::default()).expect("Failed to parse SVG");

    // Create a pixel buffer for the overlay
    let mut pixmap =
        Pixmap::new(image.width(), image.height()).expect("Failed to create pixel buffer");

    // Render SVG to the pixel buffer
    resvg::render(&svg_tree, Transform::identity(), &mut pixmap.as_mut());

    // Convert original image to RGB format
    let rgb_image = image.to_rgb8();
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
        let bboxes = get_bounding_boxes(&labels.view());

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

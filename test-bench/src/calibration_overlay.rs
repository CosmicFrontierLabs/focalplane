use anyhow::Result;
use image::DynamicImage;
use ndarray::Array2;
use shared::image_proc::image::u16_to_gray_image;
use shared::image_proc::overlay::overlay_to_image;

pub use crate::calibration::analyze::{analyze_calibration_pattern, CalibrationAnalysis};
pub use crate::calibration::apriltags::{
    build_tag_grid_map, detect_apriltags, filter_apriltags_by_size,
};
pub use crate::calibration::bars::{
    extract_bar_intensity_profile, measure_bar_sharpness, BarMeasurement, BarOrientation,
};
pub use crate::calibration::psf_pixels::{measure_psf_pixels, PixelMeasurement};

pub fn render_svg_overlay(
    width: u32,
    height: u32,
    analysis: &CalibrationAnalysis,
) -> Result<String> {
    let avg_tag_height = if !analysis.apriltag_detections.is_empty() {
        let total_height: f64 = analysis
            .apriltag_detections
            .iter()
            .map(|det| {
                let corners = det.corners();
                let top_edge = ((corners[0][0] - corners[1][0]).powi(2)
                    + (corners[0][1] - corners[1][1]).powi(2))
                .sqrt();
                let bottom_edge = ((corners[2][0] - corners[3][0]).powi(2)
                    + (corners[2][1] - corners[3][1]).powi(2))
                .sqrt();
                (top_edge + bottom_edge) / 2.0
            })
            .sum();
        total_height / analysis.apriltag_detections.len() as f64
    } else {
        100.0
    };

    let font_size = (avg_tag_height / 8.0).max(8.0) * 2.0;

    let mut svg_data = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none">"#
    );

    for det in &analysis.apriltag_detections {
        let center = det.center();
        let corners = det.corners();

        let padding = 3.0;
        for i in 0..4 {
            let next = (i + 1) % 4;
            let dx = corners[i][0] - center[0];
            let dy = corners[i][1] - center[1];
            let len = (dx * dx + dy * dy).sqrt();
            let scale = (len + padding) / len;

            let p1_x = center[0] + dx * scale;
            let p1_y = center[1] + dy * scale;

            let dx_next = corners[next][0] - center[0];
            let dy_next = corners[next][1] - center[1];
            let len_next = (dx_next * dx_next + dy_next * dy_next).sqrt();
            let scale_next = (len_next + padding) / len_next;

            let p2_x = center[0] + dx_next * scale_next;
            let p2_y = center[1] + dy_next * scale_next;

            svg_data.push_str(&format!(
                r##"<line x1="{p1_x}" y1="{p1_y}" x2="{p2_x}" y2="{p2_y}" stroke="#00ff00" stroke-width="2" />"##
            ));
        }

        let id_text = format!("ID:{}", det.id());
        svg_data.push_str(&format!(
            r##"<text x="{}" y="{}" font-size="{}" font-weight="bold" fill="#00ff00" text-anchor="middle" dominant-baseline="middle">{}</text>"##,
            center[0],
            center[1],
            font_size,
            id_text
        ));
    }

    for bar in &analysis.bar_measurements {
        let text = format!("{:.4}", bar.contrast);
        let (x, y) = bar.position;
        let x_svg = x + 1.0;
        let y_svg = y + 1.0;

        let rotation = match bar.orientation {
            BarOrientation::Horizontal => 0.0,
            BarOrientation::Vertical => -90.0,
        };

        svg_data.push_str(&format!(
            r##"<text x="{x_svg}" y="{y_svg}" font-size="{font_size}" fill="#00ff00" text-anchor="middle" dominant-baseline="middle" transform="rotate({rotation} {x_svg} {y_svg})">{text}</text>"##
        ));
    }

    for pixel in &analysis.pixel_measurements {
        let (x, y) = pixel.position;
        let x_svg = x + 1.0;
        let y_svg = y + 1.0;

        let (major_axis, minor_axis, angle_deg) = if pixel.diameter < 0.1 {
            let tiny_radius = 2.0;
            (tiny_radius, tiny_radius, 0.0)
        } else {
            let radius = pixel.diameter / 2.0;
            let major = radius * pixel.aspect_ratio.sqrt();
            let minor = radius / pixel.aspect_ratio.sqrt();
            let angle = pixel.angle.to_degrees();
            (major, minor, angle)
        };

        svg_data.push_str(&format!(
            r##"<ellipse cx="{x_svg}" cy="{y_svg}" rx="{major_axis}" ry="{minor_axis}" transform="rotate({angle_deg} {x_svg} {y_svg})" fill="none" stroke="#ff0000" stroke-width="1.5" />"##
        ));

        let diameter_text = format!("d:{:.1}", pixel.diameter);
        let diameter_offset = major_axis + font_size * 0.5;
        svg_data.push_str(&format!(
            r##"<text x="{x_svg}" y="{}" font-size="{font_size}" font-weight="bold" fill="#ff0000" text-anchor="middle" dominant-baseline="middle">{diameter_text}</text>"##,
            y_svg - diameter_offset
        ));

        let flux_text = format!("f:{:.2e}", pixel.flux);
        let flux_offset = major_axis + font_size * 0.5;
        svg_data.push_str(&format!(
            r##"<text x="{x_svg}" y="{}" font-size="{font_size}" font-weight="bold" fill="#ff0000" text-anchor="middle" dominant-baseline="middle">{flux_text}</text>"##,
            y_svg + flux_offset
        ));
    }

    svg_data.push_str("</svg>");

    Ok(svg_data)
}

pub fn render_annotated_image(
    frame: &Array2<u16>,
    analysis: &CalibrationAnalysis,
) -> Result<DynamicImage> {
    let gray_img = u16_to_gray_image(frame);
    let base_img = DynamicImage::ImageLuma8(gray_img);

    let width = base_img.width();
    let height = base_img.height();

    let svg_data = render_svg_overlay(width, height, analysis)?;

    Ok(overlay_to_image(&base_img, &svg_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display_patterns::apriltag;

    fn load_test_image(bit_depth: u8) -> Array2<u16> {
        let rgb_img =
            apriltag::generate(1024, 1024).expect("Failed to generate AprilTag test pattern");
        let (width, height) = rgb_img.dimensions();
        let mut array = Array2::zeros((height as usize, width as usize));
        let max_value = ((1u32 << bit_depth) - 1) as u32;

        for (y, mut row) in array.rows_mut().into_iter().enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                let rgb = rgb_img.get_pixel(x as u32, y as u32);
                let rgb_avg = (rgb[0] as u32 + rgb[1] as u32 + rgb[2] as u32) / 3;
                let normalized = (rgb_avg * max_value) / 255;
                let inverted = max_value - normalized;
                *pixel = inverted as u16;
            }
        }
        array
    }

    #[test]
    fn test_render_annotated_image() {
        let frame = load_test_image(8);
        let analysis = analyze_calibration_pattern(&frame, 8).expect("Analysis failed");
        let annotated = render_annotated_image(&frame, &analysis);

        assert!(
            annotated.is_ok(),
            "Should successfully render annotated image"
        );
        println!("Successfully rendered annotated image");
    }
}

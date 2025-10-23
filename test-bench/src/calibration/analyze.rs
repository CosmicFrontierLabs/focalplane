use anyhow::Result;
use apriltag::Detection;
use ndarray::Array2;
use shared::image_proc::image::{invert_monochrome, u16_to_gray_image};

use super::apriltags::{detect_apriltags, filter_apriltags_by_size};
use super::bars::{measure_bar_sharpness, BarMeasurement};
use super::psf_pixels::{measure_psf_pixels, PixelMeasurement};

#[derive(Debug)]
pub struct CalibrationAnalysis {
    pub apriltag_detections: Vec<Detection>,
    pub bar_measurements: Vec<BarMeasurement>,
    pub pixel_measurements: Vec<PixelMeasurement>,
}

pub fn analyze_calibration_pattern(
    frame: &Array2<u16>,
    bit_depth: u8,
) -> Result<CalibrationAnalysis> {
    let inverted_for_detection = invert_monochrome(frame, bit_depth);
    let gray_img = u16_to_gray_image(&inverted_for_detection);

    let all_detections = detect_apriltags(&gray_img)?;
    let apriltag_detections = filter_apriltags_by_size(all_detections, 50.0);

    let bar_measurements = if !apriltag_detections.is_empty() {
        measure_bar_sharpness(frame, &apriltag_detections, bit_depth)?
    } else {
        Vec::new()
    };

    let pixel_measurements = if !apriltag_detections.is_empty() {
        measure_psf_pixels(frame, &apriltag_detections)?
    } else {
        Vec::new()
    };

    Ok(CalibrationAnalysis {
        apriltag_detections,
        bar_measurements,
        pixel_measurements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration_overlay::render_annotated_image;
    use crate::display_patterns::apriltag;
    use image::{imageops, DynamicImage};
    use shared::test_util::get_output_dir;

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

    fn apply_gaussian_blur(frame: &Array2<u16>, sigma: f32) -> Array2<u16> {
        let gray_img = u16_to_gray_image(frame);
        let blurred = imageops::blur(&gray_img, sigma);

        let (width, height) = blurred.dimensions();
        let mut array = Array2::zeros((height as usize, width as usize));

        for (y, mut row) in array.rows_mut().into_iter().enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                *pixel = blurred.get_pixel(x as u32, y as u32)[0] as u16;
            }
        }

        array
    }

    fn rotate_image(frame: &Array2<u16>, angle_deg: f32) -> Array2<u16> {
        let gray_img = u16_to_gray_image(frame);
        let dyn_img = DynamicImage::ImageLuma8(gray_img);

        let rotated = match angle_deg as i32 {
            90 => dyn_img.rotate90(),
            180 => dyn_img.rotate180(),
            270 => dyn_img.rotate270(),
            _ => dyn_img,
        };

        let rotated_gray = rotated.to_luma8();
        let (width, height) = rotated_gray.dimensions();
        let mut array = Array2::zeros((height as usize, width as usize));

        for (y, mut row) in array.rows_mut().into_iter().enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                *pixel = rotated_gray.get_pixel(x as u32, y as u32)[0] as u16;
            }
        }

        array
    }

    #[test]
    fn test_apriltag_detection_original() {
        let frame = load_test_image(8);
        let analysis = analyze_calibration_pattern(&frame, 8).expect("Analysis failed");

        assert!(
            !analysis.apriltag_detections.is_empty(),
            "Should detect AprilTags"
        );
        println!(
            "Detected {} AprilTags in inverted image",
            analysis.apriltag_detections.len()
        );

        for det in &analysis.apriltag_detections {
            println!("  Tag ID: {}, center: {:?}", det.id(), det.center());
        }

        let annotated =
            render_annotated_image(&frame, &analysis).expect("Failed to render annotated image");
        annotated
            .save("test_apriltag_detection_original.png")
            .expect("Failed to save annotated image");
        println!("Saved annotated image to test_apriltag_detection_original.png");
    }

    #[test]
    fn test_bar_measurements_original() {
        let frame = load_test_image(8);
        let analysis = analyze_calibration_pattern(&frame, 8).expect("Analysis failed");

        assert!(
            !analysis.bar_measurements.is_empty(),
            "Should detect bar measurements"
        );
        println!(
            "Detected {} bar measurements in original image",
            analysis.bar_measurements.len()
        );

        for bar in &analysis.bar_measurements {
            println!(
                "  Bar at ({:.1}, {:.1}): contrast={:.3}, freq={:.4} cyc/px, orientation={:?}",
                bar.position.0,
                bar.position.1,
                bar.contrast,
                bar.spatial_frequency_cycles_per_pixel,
                bar.orientation
            );
            assert!(
                bar.contrast >= 0.0 && bar.contrast <= 1.0,
                "Contrast should be between 0.0 and 1.0"
            );
        }
    }

    #[test]
    fn test_pixel_measurements_original() {
        let frame = load_test_image(8);
        let analysis = analyze_calibration_pattern(&frame, 8).expect("Analysis failed");

        println!(
            "Detected {} pixel measurements in original image",
            analysis.pixel_measurements.len()
        );

        for pixel in &analysis.pixel_measurements {
            println!(
                "  Pixel at ({:.1}, {:.1}): diameter={:.2}, flux={:.0}, aspect_ratio={:.2}, angle={:.2}°",
                pixel.position.0,
                pixel.position.1,
                pixel.diameter,
                pixel.flux,
                pixel.aspect_ratio,
                pixel.angle.to_degrees()
            );
            assert!(pixel.diameter >= 0.0, "PSF diameter should be non-negative");
            assert!(pixel.flux > 0.0, "Flux should be positive");
            assert!(pixel.aspect_ratio > 0.0, "Aspect ratio should be positive");
        }
    }

    #[test]
    fn test_apriltag_detection_blurred() {
        let frame = load_test_image(8);
        let blurred = apply_gaussian_blur(&frame, 3.0);
        let analysis = analyze_calibration_pattern(&blurred, 8).expect("Analysis failed");

        println!(
            "Detected {} AprilTags in blurred image (sigma=3.0)",
            analysis.apriltag_detections.len()
        );

        if !analysis.apriltag_detections.is_empty() {
            println!("Blurred image still detects AprilTags");
        }
    }

    #[test]
    fn test_bar_sharpness_degrades_with_blur() {
        const SAVE_DEBUG_IMAGES: bool = false;

        let frame = load_test_image(8);
        let original_analysis = analyze_calibration_pattern(&frame, 8).expect("Analysis failed");

        if let Some(first_bar) = original_analysis.bar_measurements.first() {
            println!(
                "Bar spacing (approx from freq): {:.1} pixels",
                1.0 / (2.0 * first_bar.spatial_frequency_cycles_per_pixel)
            );
        }

        assert!(
            !original_analysis.bar_measurements.is_empty(),
            "Original image should have bar measurements"
        );

        let orig_avg_contrast: f64 = original_analysis
            .bar_measurements
            .iter()
            .map(|b| b.contrast)
            .sum::<f64>()
            / original_analysis.bar_measurements.len() as f64;

        let orig_avg_diameter: f64 = original_analysis
            .pixel_measurements
            .iter()
            .map(|p| p.diameter)
            .sum::<f64>()
            / original_analysis.pixel_measurements.len() as f64;

        let orig_avg_flux: f64 = original_analysis
            .pixel_measurements
            .iter()
            .map(|p| p.flux)
            .sum::<f64>()
            / original_analysis.pixel_measurements.len() as f64;

        println!("Original avg contrast: {:.6}", orig_avg_contrast);
        println!("Original avg diameter: {:.2} pixels", orig_avg_diameter);
        println!("Original avg flux: {:.0}", orig_avg_flux);

        if SAVE_DEBUG_IMAGES {
            let output_dir = get_output_dir();

            let original_annotated =
                render_annotated_image(&frame, &original_analysis).expect("Render failed");
            let original_path = output_dir.join("test_blur_original.png");
            original_annotated
                .save(&original_path)
                .expect("Failed to save original");
            println!("Saved: {}", original_path.display());
        }

        let blur_sigmas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut prev_contrast = orig_avg_contrast;

        for sigma in blur_sigmas {
            let blurred = apply_gaussian_blur(&frame, sigma);
            let blurred_analysis =
                analyze_calibration_pattern(&blurred, 8).expect("Analysis failed");

            assert!(
                !blurred_analysis.bar_measurements.is_empty(),
                "Blurred image (sigma={}) should still have bar measurements",
                sigma
            );

            let blur_avg_contrast: f64 = blurred_analysis
                .bar_measurements
                .iter()
                .map(|b| b.contrast)
                .sum::<f64>()
                / blurred_analysis.bar_measurements.len() as f64;

            let blur_avg_diameter: f64 = blurred_analysis
                .pixel_measurements
                .iter()
                .map(|p| p.diameter)
                .sum::<f64>()
                / blurred_analysis.pixel_measurements.len() as f64;

            let blur_avg_flux: f64 = blurred_analysis
                .pixel_measurements
                .iter()
                .map(|p| p.flux)
                .sum::<f64>()
                / blurred_analysis.pixel_measurements.len() as f64;

            println!(
                "σ={:.1}: contrast={:.6} ({:.2}%), diameter={:.2}px ({:.2}%), flux={:.0} ({:.2}%)",
                sigma,
                blur_avg_contrast,
                (blur_avg_contrast / orig_avg_contrast) * 100.0,
                blur_avg_diameter,
                (blur_avg_diameter / orig_avg_diameter) * 100.0,
                blur_avg_flux,
                (blur_avg_flux / orig_avg_flux) * 100.0
            );

            if SAVE_DEBUG_IMAGES {
                let output_dir = get_output_dir();
                let annotated =
                    render_annotated_image(&blurred, &blurred_analysis).expect("Render failed");
                let filename = format!("test_blur_sigma_{:.0}.png", sigma);
                let path = output_dir.join(&filename);
                annotated.save(&path).expect("Failed to save image");
                println!("Saved: {}", path.display());
            }

            assert!(
                blur_avg_contrast <= prev_contrast,
                "Contrast should not increase with more blur: σ={:.1}, prev={:.6}, current={:.6}",
                sigma,
                prev_contrast,
                blur_avg_contrast
            );

            prev_contrast = blur_avg_contrast;
        }
    }

    #[test]
    fn test_apriltag_detection_rotated_90() {
        let frame = load_test_image(8);
        let rotated = rotate_image(&frame, 90.0);
        let analysis = analyze_calibration_pattern(&rotated, 8).expect("Analysis failed");

        println!(
            "Detected {} AprilTags in 90° rotated image",
            analysis.apriltag_detections.len()
        );
        assert!(
            !analysis.apriltag_detections.is_empty(),
            "Should detect AprilTags in rotated image"
        );
    }

    #[test]
    fn test_apriltag_detection_rotated_180() {
        let frame = load_test_image(8);
        let rotated = rotate_image(&frame, 180.0);
        let analysis = analyze_calibration_pattern(&rotated, 8).expect("Analysis failed");

        println!(
            "Detected {} AprilTags in 180° rotated image",
            analysis.apriltag_detections.len()
        );
        assert!(
            !analysis.apriltag_detections.is_empty(),
            "Should detect AprilTags in 180° rotated image"
        );
    }
}

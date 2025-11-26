//! Aperture photometry utilities for astronomical sources.
//!
//! This module provides functions for collecting pixels from circular apertures
//! and background annuli, which are fundamental operations for aperture photometry.

use ndarray::ArrayView2;

/// Collect pixels from a circular aperture and background annulus.
///
/// This function collects pixel values from two regions:
/// 1. A circular aperture centered at (x_center, y_center) with radius `aperture_radius`
/// 2. A background annulus between `background_inner_radius` and `background_outer_radius`
///
/// Pixels are included if their distance from the center falls within the specified ranges.
/// This uses the "center" method where a pixel is included if its center falls within the region.
///
/// # Arguments
///
/// * `image` - The image array as f64 pixel values
/// * `x_center` - X coordinate of the aperture center (can be subpixel)
/// * `y_center` - Y coordinate of the aperture center (can be subpixel)
/// * `aperture_radius` - Radius in pixels for the measurement aperture
/// * `background_inner_radius` - Inner radius of background annulus in pixels
/// * `background_outer_radius` - Outer radius of background annulus in pixels
///
/// # Returns
///
/// A tuple of `(aperture_pixels, background_pixels)` where:
/// - `aperture_pixels` contains all pixel values within the aperture radius
/// - `background_pixels` contains all pixel values in the background annulus
pub fn collect_aperture_pixels(
    image: &ArrayView2<f64>,
    x_center: f64,
    y_center: f64,
    aperture_radius: f64,
    background_inner_radius: f64,
    background_outer_radius: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (height, width) = image.dim();

    let x_center_int = x_center.round() as isize;
    let y_center_int = y_center.round() as isize;

    let x_min = (x_center_int - background_outer_radius.ceil() as isize).max(0) as usize;
    let x_max =
        ((x_center_int + background_outer_radius.ceil() as isize + 1).min(width as isize)) as usize;
    let y_min = (y_center_int - background_outer_radius.ceil() as isize).max(0) as usize;
    let y_max = ((y_center_int + background_outer_radius.ceil() as isize + 1).min(height as isize))
        as usize;

    let mut aperture_pixels = Vec::new();
    let mut background_pixels = Vec::new();

    for y in y_min..y_max {
        for x in x_min..x_max {
            let dx = x as f64 - x_center;
            let dy = y as f64 - y_center;
            let distance = (dx * dx + dy * dy).sqrt();

            if distance <= aperture_radius {
                aperture_pixels.push(image[[y, x]]);
            } else if distance >= background_inner_radius && distance <= background_outer_radius {
                background_pixels.push(image[[y, x]]);
            }
        }
    }

    (aperture_pixels, background_pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_collect_aperture_pixels_basic() {
        let mut image = Array2::<f64>::zeros((20, 20));

        // Set aperture region to 100.0
        for i in 0..20 {
            for j in 0..20 {
                let dx = j as f64 - 10.0;
                let dy = i as f64 - 10.0;
                let distance = (dx * dx + dy * dy).sqrt();

                if distance <= 2.0 {
                    image[[i, j]] = 100.0;
                } else if distance >= 4.0 && distance <= 6.0 {
                    image[[i, j]] = 50.0;
                } else {
                    image[[i, j]] = 0.0;
                }
            }
        }

        let (aperture_pixels, background_pixels) =
            collect_aperture_pixels(&image.view(), 10.0, 10.0, 2.0, 4.0, 6.0);

        // Verify aperture pixels
        assert!(
            !aperture_pixels.is_empty(),
            "Aperture should contain pixels"
        );
        for &pixel in &aperture_pixels {
            assert!(
                (pixel - 100.0).abs() < 0.1,
                "Aperture pixels should be ~100.0"
            );
        }

        // Verify background pixels
        assert!(
            !background_pixels.is_empty(),
            "Background should contain pixels"
        );
        for &pixel in &background_pixels {
            assert!(
                (pixel - 50.0).abs() < 0.1,
                "Background pixels should be ~50.0"
            );
        }
    }

    #[test]
    fn test_collect_aperture_pixels_subpixel_center() {
        let mut image = Array2::<f64>::ones((20, 20));

        // Mark center pixel
        image[[10, 10]] = 999.0;

        let (aperture_pixels, _) =
            collect_aperture_pixels(&image.view(), 10.5, 10.3, 1.5, 3.0, 5.0);

        // Should include the center pixel
        assert!(
            aperture_pixels.contains(&999.0),
            "Should include center pixel with subpixel coordinates"
        );
    }

    #[test]
    fn test_collect_aperture_pixels_edge_case() {
        let image = Array2::<f64>::ones((10, 10));

        // Near edge
        let (aperture_pixels, background_pixels) =
            collect_aperture_pixels(&image.view(), 2.0, 2.0, 1.5, 3.0, 4.5);

        // Should handle edge clipping gracefully
        assert!(
            !aperture_pixels.is_empty(),
            "Should collect some aperture pixels"
        );
        assert!(
            !background_pixels.is_empty(),
            "Should collect some background pixels"
        );
    }

    #[test]
    fn test_collect_aperture_pixels_no_overlap() {
        let image = Array2::<f64>::ones((20, 20));

        // Aperture fully inside image
        let (aperture_pixels, background_pixels) =
            collect_aperture_pixels(&image.view(), 10.0, 10.0, 2.0, 4.0, 6.0);

        // All pixels should have value 1.0
        assert_eq!(
            aperture_pixels.iter().all(|&p| (p - 1.0).abs() < 1e-10),
            true
        );
        assert_eq!(
            background_pixels.iter().all(|&p| (p - 1.0).abs() < 1e-10),
            true
        );
    }

    #[test]
    fn test_collect_aperture_pixels_counts() {
        let image = Array2::<f64>::zeros((30, 30));

        let (aperture_pixels, background_pixels) =
            collect_aperture_pixels(&image.view(), 15.0, 15.0, 3.0, 6.0, 9.0);

        // Aperture should have roughly pi * r^2 pixels
        let expected_aperture = (std::f64::consts::PI * 3.0 * 3.0) as usize;
        assert!(
            aperture_pixels.len() >= expected_aperture - 5
                && aperture_pixels.len() <= expected_aperture + 5,
            "Aperture pixel count {} should be near {}",
            aperture_pixels.len(),
            expected_aperture
        );

        // Background annulus should have pixels
        assert!(
            background_pixels.len() > 50,
            "Background should have many pixels"
        );
    }
}

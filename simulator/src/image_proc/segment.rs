use ndarray::Array2;

use crate::image_proc::{
    convolve2d, detect_stars, gaussian_kernel, otsu_threshold, ConvolveMode, ConvolveOptions,
};

use super::StarDetection;

/// Processes an image to detect and characterize stars.
///
/// This function implements a complete star detection pipeline:
/// 1. Optionally applies Gaussian smoothing with specified sigma value
/// 2. Determines threshold value (either user-provided or automatic using Otsu's method)
/// 3. Segments the image and identifies connected regions above threshold
/// 4. Calculates precise centroid position and flux measurements for each detection
///
/// # Arguments
/// * `sensor_image` - Reference to image array (DN values from sensor)
/// * `smooth_by` - Optional sigma for Gaussian smoothing in pixels (None = no smoothing)
/// * `threshold` - Optional custom threshold value (None = use Otsu's method)
///
/// # Returns
/// * Vector of `StarDetection` objects, each containing position, flux, and shape information
pub fn do_detections(
    sensor_image: &Array2<u16>,
    smooth_by: Option<f64>,
    threshold: Option<f64>,
) -> Vec<StarDetection> {
    // Cast it into float space
    let image_array = sensor_image.mapv(|x| x as f64);

    let smoothed = match smooth_by {
        Some(smooth) => {
            // TODO(meawoppl) - make this a function of erf() + round up kernel to nearest odd multiple
            let kernel_size = 9;
            let kernel = gaussian_kernel(kernel_size, smooth);

            convolve2d(
                &image_array.view(),
                &kernel.view(),
                Some(ConvolveOptions {
                    mode: ConvolveMode::Same,
                }),
            )
        }
        None => image_array.clone(),
    };

    // Use the supplied threshold if provided, otherwise calculate Otsu's threshold
    let cutoff = match threshold {
        Some(t) => t,
        None => {
            let threshold = otsu_threshold(&smoothed.view());
            threshold
        }
    };

    // Detect stars using our new centroid-based detection
    detect_stars(&smoothed.view(), Some(cutoff))
}

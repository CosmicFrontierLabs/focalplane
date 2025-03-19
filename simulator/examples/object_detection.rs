//! Example demonstrating object detection with Otsu's thresholding
//!
//! This example shows how to detect stars in a simulated astronomical image.
//! It demonstrates a complete star detection pipeline:
//!
//! 1. Image simulation - Creates a synthetic star field with Gaussian PSFs and noise
//! 2. Image preprocessing - Applies Gaussian smoothing to reduce noise
//! 3. Thresholding - Uses Otsu's method to automatically determine threshold level
//! 4. Star detection - Identifies and characterizes stars using centroid detection
//! 5. Performance metrics - Calculates precision, recall and F1 score for the detection
//!
//! This pipeline mimics real-world astronomical image processing where stars need
//! to be accurately located against a noisy background. The centroid detection
//! provides sub-pixel accuracy for star positions, which is critical for astrometry.
//!
//! Key components demonstrated:
//! - Simulating realistic star fields with Gaussian PSFs
//! - Noise reduction with Gaussian filtering
//! - Automatic threshold selection with Otsu's method
//! - Star detection with moment-based centroid calculation
//! - Star characterization (position, flux, aspect ratio)
//! - Detection performance metrics

use ndarray::Array2;
use simulator::image_proc::{
    centroid::detect_stars,
    convolve2d::{convolve2d, gaussian_kernel, ConvolveMode, ConvolveOptions},
    thresholding::otsu_threshold,
};

fn main() {
    println!("Object Detection with Otsu's Thresholding");
    println!("=========================================");

    // Create a simulated star field
    let image = create_star_field(100, 100, 20, 5.0);

    println!("Created a 100x100 simulated star field with 20 stars");

    // Apply Gaussian blur to smooth the image
    let kernel = gaussian_kernel(5, 1.0);
    let options = ConvolveOptions {
        mode: ConvolveMode::Same,
    };
    let smoothed = convolve2d(&image.view(), &kernel.view(), Some(options));

    println!("Applied Gaussian smoothing to reduce noise");

    // Calculate Otsu's threshold
    let threshold = otsu_threshold(&smoothed.view());
    println!("Otsu's threshold: {:.6}", threshold);

    // Detect stars using our new centroid-based detection
    let stars = detect_stars(&smoothed.view(), Some(threshold));
    println!("Detected {} stars", stars.len());

    // Print star details
    println!("\nDetected stars:");
    println!(
        "{:3} | {:6} | {:6} | {:8} | {:10} | {:5}",
        "ID", "X", "Y", "Flux", "Aspect", "Valid"
    );
    println!("-----|--------|--------|----------|------------|------");

    for (i, star) in stars.iter().enumerate() {
        println!(
            "{:3} | {:6.2} | {:6.2} | {:8.2} | {:10.2} | {:5}",
            i + 1,
            star.x,
            star.y,
            star.flux,
            star.aspect_ratio,
            star.is_valid
        );
    }

    // Calculate metrics for detection (using the number of valid stars)
    let num_stars = stars.iter().filter(|s| s.is_valid).count();
    let true_positives = num_stars.min(20); // 20 is our original count of added stars
    let false_positives = if num_stars > 20 { num_stars - 20 } else { 0 };
    let false_negatives = if 20 > num_stars { 20 - num_stars } else { 0 };

    println!("\nDetection metrics:");
    println!("True positives: {}", true_positives);
    println!("False positives: {}", false_positives);
    println!("False negatives: {}", false_negatives);

    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };

    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        0.0
    };

    println!("Precision: {:.2}", precision);
    println!("Recall: {:.2}", recall);

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    println!("F1 score: {:.2}", f1);
}

/// Create a simulated star field with gaussian stars
///
/// # Arguments
/// * `width` - Width of the image in pixels
/// * `height` - Height of the image in pixels
/// * `num_stars` - Number of stars to generate
/// * `noise_level` - Noise level as a percentage (e.g., 5.0 means 5% noise)
///
/// # Returns
/// * 2D array containing the simulated star field with Gaussian PSFs and noise
///
/// This function creates a realistic astronomical image simulation by:
/// 1. Generating a uniform background noise at the specified level
/// 2. Adding a specified number of stars with Gaussian point spread functions (PSFs)
/// 3. Randomly distributing stars across the image with varying intensities
/// 4. Using varying PSF widths to simulate different seeing conditions or focus
///
/// The noise uses a uniform distribution, while the star intensities follow
/// a uniform distribution (a more realistic simulation might use a log-normal
/// distribution to better match the stellar luminosity function).
fn create_star_field(
    width: usize,
    height: usize,
    num_stars: usize,
    noise_level: f64,
) -> Array2<f64> {
    use rand::distributions::{Distribution, Uniform};
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut image = Array2::zeros((height, width));

    // Add random noise
    let noise_dist = Uniform::from(0.0..noise_level / 100.0);

    for i in 0..height {
        for j in 0..width {
            image[[i, j]] = noise_dist.sample(&mut rng);
        }
    }

    // Position distribution - keep stars away from edges to avoid truncation
    let x_dist = Uniform::from(5..width - 5);
    let y_dist = Uniform::from(5..height - 5);

    // Intensity distribution (uniform in this simple example)
    // A more realistic simulation would use a log-normal distribution
    // to better match the stellar luminosity function
    let intensity_dist = Uniform::from(0.5..1.0);

    // Add stars
    for _ in 0..num_stars {
        let x = x_dist.sample(&mut rng);
        let y = y_dist.sample(&mut rng);
        let intensity = intensity_dist.sample(&mut rng);

        // Add a Gaussian-shaped star with random sigma (PSF width)
        // This simulates different seeing conditions or focus
        add_gaussian_star(&mut image, x, y, intensity, rng.gen_range(1.0..2.5));
    }

    image
}

/// Add a gaussian-shaped star to the image
///
/// # Arguments
/// * `image` - The image array to modify
/// * `x` - X-coordinate of the star center
/// * `y` - Y-coordinate of the star center
/// * `intensity` - Peak intensity of the star
/// * `sigma` - Standard deviation of the Gaussian PSF
///
/// This function adds a star with a Gaussian point spread function (PSF)
/// to the image at the specified position. The star's brightness profile
/// follows a 2D Gaussian distribution:
///
/// I(r) = intensity * exp(-r² / (2*sigma²))
///
/// where r is the distance from the center. The function calculates
/// this distribution for each pixel within 3*sigma radius.
fn add_gaussian_star(image: &mut Array2<f64>, x: usize, y: usize, intensity: f64, sigma: f64) {
    let (height, width) = image.dim();
    // Use 3*sigma as the radius to include ~99.7% of the Gaussian's energy
    let radius = (3.0 * sigma).ceil() as i32;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = x as i32 + dx;
            let py = y as i32 + dy;

            // Skip pixels outside image bounds
            if px < 0 || px >= width as i32 || py < 0 || py >= height as i32 {
                continue;
            }

            // Calculate Gaussian intensity using the 2D Gaussian formula
            let distance_sq = (dx * dx + dy * dy) as f64;
            let value = intensity * (-(distance_sq) / (2.0 * sigma * sigma)).exp();

            // Add to image (additive because stars can overlap)
            image[[py as usize, px as usize]] += value;
        }
    }
}

// We've replaced the BoundingBox-based detection with our star detection approach
// so we no longer need the calculate_detection_metrics function

//! Example demonstrating object detection with Otsu's thresholding
//!
//! This example shows how to detect stars in a simulated astronomical image.

use ndarray::Array2;
use simulator::image_proc::{
    convolve2d::{convolve2d, gaussian_kernel, ConvolveOptions, EdgeMode},
    thresholding::{detect_objects, otsu_threshold},
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
        parallel: true,
        edge_mode: EdgeMode::Reflect,
    };
    let smoothed = convolve2d(&image, &kernel, options);

    println!("Applied Gaussian smoothing to reduce noise");

    // Calculate Otsu's threshold
    let threshold = otsu_threshold(smoothed.view(), None);
    println!("Otsu's threshold: {:.6}", threshold);

    // Detect objects
    let bboxes = detect_objects(smoothed.view(), Some(4), Some(0.3));
    println!("Detected {} objects", bboxes.len());

    // Print bounding box details
    println!("\nDetected objects:");
    println!(
        "{:3} | {:6} | {:6} | {:5} | {:5}",
        "ID", "X", "Y", "Width", "Height"
    );
    println!("-----|--------|--------|-------|-------");

    for (i, bbox) in bboxes.iter().enumerate() {
        println!(
            "{:3} | {:6} | {:6} | {:5} | {:5}",
            i + 1,
            bbox.x_min,
            bbox.y_min,
            bbox.width,
            bbox.height
        );
    }

    // Calculate metrics for detection
    let (true_positives, false_positives, false_negatives) =
        calculate_detection_metrics(&bboxes, &image, threshold);

    println!("\nDetection metrics:");
    println!("True positives: {}", true_positives);
    println!("False positives: {}", false_positives);
    println!("False negatives: {}", false_negatives);

    let precision = true_positives as f64 / (true_positives + false_positives) as f64;
    let recall = true_positives as f64 / (true_positives + false_negatives) as f64;

    println!("Precision: {:.2}", precision);
    println!("Recall: {:.2}", recall);
    println!(
        "F1 score: {:.2}",
        2.0 * precision * recall / (precision + recall)
    );
}

/// Create a simulated star field with gaussian stars
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

    // Position distribution
    let x_dist = Uniform::from(5..width - 5);
    let y_dist = Uniform::from(5..height - 5);

    // Intensity distribution (log-normal to simulate star magnitudes)
    let intensity_dist = Uniform::from(0.5..1.0);

    // Add stars
    for _ in 0..num_stars {
        let x = x_dist.sample(&mut rng);
        let y = y_dist.sample(&mut rng);
        let intensity = intensity_dist.sample(&mut rng);

        // Add a Gaussian-shaped star
        add_gaussian_star(&mut image, x, y, intensity, rng.gen_range(1.0..2.5));
    }

    image
}

/// Add a gaussian-shaped star to the image
fn add_gaussian_star(image: &mut Array2<f64>, x: usize, y: usize, intensity: f64, sigma: f64) {
    let (height, width) = image.dim();
    let radius = (3.0 * sigma).ceil() as i32;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let px = x as i32 + dx;
            let py = y as i32 + dy;

            // Skip pixels outside image bounds
            if px < 0 || px >= width as i32 || py < 0 || py >= height as i32 {
                continue;
            }

            // Calculate Gaussian intensity
            let distance_sq = (dx * dx + dy * dy) as f64;
            let value = intensity * (-(distance_sq) / (2.0 * sigma * sigma)).exp();

            // Add to image
            image[[py as usize, px as usize]] += value;
        }
    }
}

/// Calculate detection metrics
fn calculate_detection_metrics(
    detected_boxes: &[simulator::image_proc::thresholding::BoundingBox],
    ground_truth: &Array2<f64>,
    threshold: f64,
) -> (usize, usize, usize) {
    // Count bright peaks in the ground truth image
    let (height, width) = ground_truth.dim();
    let mut true_peaks = 0;

    for i in 1..height - 1 {
        for j in 1..width - 1 {
            let value = ground_truth[[i, j]];

            // Skip if below threshold
            if value < threshold {
                continue;
            }

            // Check if it's a local maximum
            let is_peak = value > ground_truth[[i - 1, j - 1]]
                && value > ground_truth[[i - 1, j]]
                && value > ground_truth[[i - 1, j + 1]]
                && value > ground_truth[[i, j - 1]]
                && value > ground_truth[[i, j + 1]]
                && value > ground_truth[[i + 1, j - 1]]
                && value > ground_truth[[i + 1, j]]
                && value > ground_truth[[i + 1, j + 1]];

            if is_peak {
                true_peaks += 1;
            }
        }
    }

    // Count true positives (correct detections)
    let mut true_positives = 0;

    // For each detection, check if it contains a peak
    for bbox in detected_boxes {
        let mut contains_peak = false;

        // Get the bbox subregion, clamping to image bounds
        let x_min = bbox.x_min.min(width - 1);
        let y_min = bbox.y_min.min(height - 1);
        let x_max = (bbox.x_min + bbox.width).min(width);
        let y_max = (bbox.y_min + bbox.height).min(height);

        // For small bboxes, check the whole bbox
        for i in y_min..y_max {
            for j in x_min..x_max {
                if i > 0 && i < height - 1 && j > 0 && j < width - 1 {
                    let value = ground_truth[[i, j]];

                    // Skip if below threshold
                    if value < threshold {
                        continue;
                    }

                    // Check if it's a local maximum
                    let is_peak = value > ground_truth[[i - 1, j - 1]]
                        && value > ground_truth[[i - 1, j]]
                        && value > ground_truth[[i - 1, j + 1]]
                        && value > ground_truth[[i, j - 1]]
                        && value > ground_truth[[i, j + 1]]
                        && value > ground_truth[[i + 1, j - 1]]
                        && value > ground_truth[[i + 1, j]]
                        && value > ground_truth[[i + 1, j + 1]];

                    if is_peak {
                        contains_peak = true;
                        break;
                    }
                }
            }
            if contains_peak {
                break;
            }
        }

        if contains_peak {
            true_positives += 1;
        }
    }

    let false_positives = detected_boxes.len() - true_positives;
    let false_negatives = true_peaks - true_positives;

    (true_positives, false_positives, false_negatives)
}

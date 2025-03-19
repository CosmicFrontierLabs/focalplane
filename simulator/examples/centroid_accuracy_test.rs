//! Precision benchmark for centroid calculation with Gaussian PSFs
//!
//! This example benchmarks the accuracy of moment-based centroid calculations
//! using a dense grid of sub-pixel positions (50x50 = 2500 test cases).
//!
//! For unit testing with smaller grids, see the test_subpixel_position_grid
//! test in the centroid.rs module.
//!
//! Key metrics calculated:
//! - Mean centroid error (across all test positions)
//! - Maximum centroid error
//! - Error distribution histograms (in X, Y, and total magnitude)
//! - Success rate for different precision thresholds
//! - Analysis of any systematic bias in X or Y directions

use ndarray::Array2;
use simulator::image_proc::centroid::calculate_star_centroid;
use simulator::image_proc::thresholding::{
    apply_threshold, connected_components, get_bounding_boxes,
};

/// Generate a 2D Gaussian PSF with specified parameters
fn create_gaussian(
    image: &mut Array2<f64>,
    center_x: f64,
    center_y: f64,
    amplitude: f64,
    sigma: f64,
) {
    let (height, width) = image.dim();

    // Compute values for a window around the center
    let x_min = (center_x - 4.0 * sigma).max(0.0) as usize;
    let x_max = (center_x + 4.0 * sigma).min(width as f64 - 1.0) as usize;
    let y_min = (center_y - 4.0 * sigma).max(0.0) as usize;
    let y_max = (center_y + 4.0 * sigma).min(height as f64 - 1.0) as usize;

    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let dx = x as f64 - center_x;
            let dy = y as f64 - center_y;
            let exponent = -(dx * dx + dy * dy) / (2.0 * sigma * sigma);
            image[[y, x]] = amplitude * exponent.exp();
        }
    }
}

fn run_single_star_test(image_size: usize, position_x: f64, position_y: f64, sigma: f64) {
    // Create an empty image
    let mut image = Array2::<f64>::zeros((image_size, image_size));

    // Generate a single star at the specified position
    create_gaussian(&mut image, position_x, position_y, 1.0, sigma);

    // Detect the star using the detection pipeline
    let threshold = 0.1;
    let binary = apply_threshold(&image.view(), threshold);
    let labeled = connected_components(&binary.view());
    let bboxes = get_bounding_boxes(&labeled.view());

    if bboxes.is_empty() {
        println!("No stars detected! Try adjusting the threshold or sigma value.");
        return;
    }

    // Calculate centroid
    let bbox = bboxes[0].to_tuple();
    let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

    // Also calculate direct centroid for comparison
    // Calculate errors
    let detected_x = star.x;
    let detected_y = star.y;

    let error_detected =
        ((position_x - detected_x).powi(2) + (position_y - detected_y).powi(2)).sqrt();

    println!(
        "Single Star Test (Image Size: {}, Sigma: {})",
        image_size, sigma
    );
    println!("True Position: ({:.3}, {:.3})", position_x, position_y);
    println!(
        "Detected Position: ({:.3}, {:.3}), Error: {:.6} pixels",
        detected_x, detected_y, error_detected
    );
    println!(
        "Star Properties: flux={:.2}, aspect_ratio={:.2}, valid={}",
        star.flux, star.aspect_ratio, star.is_valid
    );
}

/// Run a large number of sub-pixel positions to evaluate centroid accuracy
fn run_subpixel_grid_test(image_size: usize, sigma: f64) {
    // Use a large grid size for benchmarking (50x50 = 2500 positions)
    let grid_size = 50;
    let threshold = 0.1;

    println!(
        "=== Testing sub-pixel accuracy on a {}x{} grid within a single pixel ===",
        grid_size, grid_size
    );
    println!("Running {} test cases...", grid_size * grid_size);

    // Implementation of grid test
    let center_x = image_size as f64 / 2.0;
    let center_y = image_size as f64 / 2.0;

    let mut total_error = 0.0;
    let mut max_error: f64 = 0.0;
    let mut total_tests = 0;
    let mut failed_detections = 0;

    // Collect errors for histogram analysis
    let mut errors_magnitude = Vec::with_capacity(grid_size * grid_size);
    let mut errors_x = Vec::with_capacity(grid_size * grid_size);
    let mut errors_y = Vec::with_capacity(grid_size * grid_size);

    // Test a grid of sub-pixel positions
    for i in 0..grid_size {
        for j in 0..grid_size {
            let sub_x = i as f64 / grid_size as f64;
            let sub_y = j as f64 / grid_size as f64;

            let position_x = center_x + sub_x;
            let position_y = center_y + sub_y;

            // Create an empty image
            let mut image = Array2::<f64>::zeros((image_size, image_size));

            // Generate a Gaussian star at this sub-pixel position
            create_gaussian(&mut image, position_x, position_y, 1.0, sigma);

            // Process the image
            let binary = apply_threshold(&image.view(), threshold);
            let labeled = connected_components(&binary.view());
            let bboxes = get_bounding_boxes(&labeled.view());

            // Skip if no detection (shouldn't happen with these parameters)
            if bboxes.is_empty() {
                failed_detections += 1;
                continue;
            }

            // Calculate centroid
            let bbox = bboxes[0].to_tuple();
            let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

            // Calculate errors
            let error_x = position_x - star.x;
            let error_y = position_y - star.y;
            let error_detected = (error_x.powi(2) + error_y.powi(2)).sqrt();

            // Store errors for histogram
            errors_magnitude.push(error_detected);
            errors_x.push(error_x);
            errors_y.push(error_y);

            // Accumulate statistics
            total_error += error_detected;
            max_error = max_error.max(error_detected);
            total_tests += 1;
        }
    }

    // Calculate average error
    let avg_error = if total_tests > 0 {
        total_error / total_tests as f64
    } else {
        0.0
    };

    // Print summary
    if total_tests > 0 {
        println!("\nSummary:");
        println!("Average Error: {:.6} pixels", avg_error);
        println!("Maximum Error: {:.6} pixels", max_error);
        println!("Total Tests: {}", total_tests);
        if failed_detections > 0 {
            println!("Failed Detections: {}", failed_detections);
        }

        // Print detailed histograms
        print_histogram(
            "Error Magnitude Distribution (pixels)",
            &errors_magnitude,
            40,
            0.0,
            0.5,
        );
        print_histogram("Error X Distribution (pixels)", &errors_x, 100, -0.5, 0.5);
        print_histogram("Error Y Distribution (pixels)", &errors_y, 100, -0.5, 0.5);

        // Additional analysis
        println!("\nPrecision Analysis:");
        let errors_below_01 = errors_magnitude.iter().filter(|&&e| e < 0.1).count();
        let errors_below_05 = errors_magnitude.iter().filter(|&&e| e < 0.05).count();
        let errors_below_025 = errors_magnitude.iter().filter(|&&e| e < 0.025).count();
        let errors_below_01_pct = (errors_below_01 as f64 / total_tests as f64) * 100.0;
        let errors_below_05_pct = (errors_below_05 as f64 / total_tests as f64) * 100.0;
        let errors_below_025_pct = (errors_below_025 as f64 / total_tests as f64) * 100.0;

        println!(
            "Errors < 0.1 pixels: {} ({:.2}%)",
            errors_below_01, errors_below_01_pct
        );
        println!(
            "Errors < 0.05 pixels: {} ({:.2}%)",
            errors_below_05, errors_below_05_pct
        );
        println!(
            "Errors < 0.025 pixels: {} ({:.2}%)",
            errors_below_025, errors_below_025_pct
        );

        // X bias analysis
        let mean_x_error = errors_x.iter().sum::<f64>() / errors_x.len() as f64;
        println!(
            "Mean X Error: {:.6} pixels (negative = detected position is to the right)",
            mean_x_error
        );

        // Y bias analysis
        let mean_y_error = errors_y.iter().sum::<f64>() / errors_y.len() as f64;
        println!(
            "Mean Y Error: {:.6} pixels (negative = detected position is below)",
            mean_y_error
        );
    }
}

/// Print a histogram using the viz module
fn print_histogram(title: &str, data: &[f64], bins: usize, min_val: f64, max_val: f64) {
    // Format title with consistent styling
    println!("\n{}", title);
    println!("-----------------------------------------------------------");

    use viz::histogram::{Histogram, HistogramConfig};

    // Create histogram with the specified bins and range
    if let Ok(mut hist) = Histogram::new_equal_bins(min_val..max_val, bins) {
        // Configure histogram display options
        let mut config = HistogramConfig::default();
        config.max_bar_width = 40;
        config.bar_char = '#';
        config.show_empty_bins = false;
        config.title = Some(title.to_string());

        hist = hist.with_config(config);

        // Add data to histogram
        hist.add_all(data.iter().copied());

        // Print the formatted histogram
        if let Ok(formatted) = hist.format() {
            println!("{}", formatted);
        }

        // Print out of range count
        let total_in_hist = hist.total_count();
        let out_of_range = data.len() as u64 - total_in_hist;

        if out_of_range > 0 {
            println!("Out of range: {} values", out_of_range);
        }

        // Print statistics summary using the new statistical methods
        println!("\nStatistical Analysis:");
        println!("-----------------------------------------------------------");

        // Get statistical measures with proper sign formatting for mean-based values
        let mean = hist.mean().unwrap_or(0.0);
        let std_dev = hist.std_dev().unwrap_or(0.0);
        let median = hist.median().unwrap_or(0.0);
        let skewness = hist.skewness().unwrap_or(0.0);
        let kurtosis = hist.kurtosis().unwrap_or(0.0);

        // Format mean and median with consistent +/- sign
        let mean_str = format_with_sign(mean);
        let median_str = format_with_sign(median);

        println!("Mean: {}, Median: {}", mean_str, median_str);
        println!("Std. Deviation: {:.6}", std_dev);

        // Only print skewness/kurtosis if we have enough data
        if data.len() >= 4 {
            println!(
                "Skewness: {:.4} ({})",
                skewness,
                if skewness.abs() < 0.5 {
                    "approximately symmetric"
                } else if skewness > 0.0 {
                    "right-skewed"
                } else {
                    "left-skewed"
                }
            );

            println!(
                "Kurtosis: {:.4} ({})",
                kurtosis,
                if kurtosis.abs() < 0.5 {
                    "normal-like tails"
                } else if kurtosis > 0.0 {
                    "heavy tails"
                } else {
                    "light tails"
                }
            );
        }

        println!("Sample size: {}", hist.total_count());
    } else {
        println!("Error creating histogram");
    }
}

/// Format a floating-point value with a consistent +/- sign
fn format_with_sign(value: f64) -> String {
    if value >= 0.0 {
        format!("+{:.3}", value)
    } else {
        format!("{:.3}", value)
    }
}

/// Test with different sigma values to see effect on centroid accuracy
fn test_sigma_effect(image_size: usize) {
    let center_x = image_size as f64 / 2.0;
    let center_y = image_size as f64 / 2.0;
    let position_x = center_x + 0.3; // Test at a fixed 0.3 subpixel offset
    let position_y = center_y + 0.7; // Test at a fixed 0.7 subpixel offset

    println!("=== Testing effect of PSF width (sigma) on centroid accuracy ===");
    println!("Fixed position: ({:.3}, {:.3})", position_x, position_y);
    println!("Sigma\tDetected Position\t\tError\t\tAspect Ratio");
    println!("----------------------------------------------------------------------");

    // Test a range of sigma values
    for sigma in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0].iter() {
        // Create an empty image
        let mut image = Array2::<f64>::zeros((image_size, image_size));

        // Generate a single star
        create_gaussian(&mut image, position_x, position_y, 1.0, *sigma);

        // Run detection
        let threshold = 0.1;
        let binary = apply_threshold(&image.view(), threshold);
        let labeled = connected_components(&binary.view());
        let bboxes = get_bounding_boxes(&labeled.view());

        if bboxes.is_empty() {
            println!("{:.2}\tNo detection\t\t-\t\t-", sigma);
            continue;
        }

        // Calculate centroid
        let bbox = bboxes[0].to_tuple();
        let star = calculate_star_centroid(&image.view(), &labeled.view(), 1, bbox);

        // Calculate errors
        let error_detected = ((position_x - star.x).powi(2) + (position_y - star.y).powi(2)).sqrt();

        // Print results
        println!(
            "{:.2}\t({:.3}, {:.3})\t{:.6}\t{:.4}",
            sigma, star.x, star.y, error_detected, star.aspect_ratio
        );
    }
}

fn main() {
    // Test parameters
    let image_size = 256;
    let sigma = 1.0; // Standard deviation of the Gaussian PSF

    // Run basic tests at specific positions
    println!("=== Basic position tests ===\n");

    println!("Integer position:");
    run_single_star_test(image_size, 128.0, 128.0, sigma);

    println!("\nHalf-pixel position:");
    run_single_star_test(image_size, 128.5, 128.5, sigma);

    println!("\nQuarter-pixel position:");
    run_single_star_test(image_size, 128.25, 128.75, sigma);

    // Run comprehensive grid test within a single pixel
    println!("\n\n");
    run_subpixel_grid_test(image_size, sigma);

    // Test effect of PSF width on centroid accuracy
    println!("\n\n");
    test_sigma_effect(image_size);
}

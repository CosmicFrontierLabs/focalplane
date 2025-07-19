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
use simulator::algo::min_max_scan::MinMaxScan;
use simulator::image_proc::airy::PixelScaledAiryDisk;
use simulator::image_proc::detection::StarFinder;
use starfield::image::starfinders::StellarSource;

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

fn run_single_star_test(
    image_size: usize,
    position_x: f64,
    position_y: f64,
    sigma: f64,
    detector: &StarFinder,
) {
    // Create an empty image
    let mut image = Array2::<f64>::zeros((image_size, image_size));

    // Generate a single star at the specified position
    create_gaussian(&mut image, position_x, position_y, 1.0, sigma);

    // Convert f64 image to u16 for detection (scale to use full u16 range)
    let max_scan = MinMaxScan::new(image.as_slice().unwrap());
    let max_val = max_scan
        .max()
        .expect("Failed to compute max value: image contains NaN or is empty");
    let scale = if max_val > 0.0 {
        65535.0 / max_val
    } else {
        1.0
    };
    let u16_image = image.mapv(|v| (v * scale) as u16);

    // Detect the star using the unified detection interface
    // Using reasonable defaults for airy disk size and detection parameters
    let airy_disk_pixels = sigma * 2.0; // Approximate airy disk size
    let background_rms = 0.1 * scale; // Scaled background noise
    let detection_sigma = 3.0; // Detection threshold in sigmas

    let stars = match detector {
        StarFinder::Naive => {
            // Use the naive detector directly with f64 image
            use simulator::image_proc::detection::detect_stars as detect_stars_naive;
            detect_stars_naive(&image.view(), Some(0.1))
                .into_iter()
                .map(|s| Box::new(s) as Box<dyn StellarSource>)
                .collect::<Vec<_>>()
        }
        _ => {
            // Use the unified interface for DAO and IRAF
            use simulator::image_proc::detection::detect_stars_unified;
            let scaled_airy_disk = PixelScaledAiryDisk::with_fwhm(airy_disk_pixels, 550.0);
            match detect_stars_unified(
                u16_image.view(),
                *detector,
                &scaled_airy_disk,
                background_rms,
                detection_sigma,
            ) {
                Ok(stars) => stars,
                Err(e) => {
                    println!("Detection failed: {e}");
                    return;
                }
            }
        }
    };

    if stars.is_empty() {
        println!("No stars detected! Try adjusting the threshold or sigma value.");
        return;
    }

    // Use the first detected star
    let star = &stars[0];

    // Calculate errors
    let (detected_x, detected_y) = star.get_centroid();

    let error_detected =
        ((position_x - detected_x).powi(2) + (position_y - detected_y).powi(2)).sqrt();

    println!("Single Star Test [{detector:?}] (Image Size: {image_size}, Sigma: {sigma})");
    println!("True Position: ({position_x:.3}, {position_y:.3})");
    println!(
        "Detected Position: ({detected_x:.3}, {detected_y:.3}), Error: {error_detected:.6} pixels"
    );
    println!("Star Properties: flux={:.2}", star.flux());
}

/// Run a large number of sub-pixel positions to evaluate centroid accuracy
fn run_subpixel_grid_test(image_size: usize, sigma: f64, detector: &StarFinder) {
    // Use a large grid size for benchmarking (50x50 = 2500 positions)
    let grid_size = 50;

    println!(
        "=== Testing sub-pixel accuracy [{detector:?}] on a {grid_size}x{grid_size} grid within a single pixel ==="
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

            // Convert f64 image to u16 for detection
            let max_scan = MinMaxScan::new(image.as_slice().unwrap());
            let max_val = max_scan
                .max()
                .expect("Failed to compute max value: image contains NaN or is empty");
            let scale = if max_val > 0.0 {
                65535.0 / max_val
            } else {
                1.0
            };
            let u16_image = image.mapv(|v| (v * scale) as u16);

            // Process the image using appropriate detection method
            let airy_disk_pixels = sigma * 2.0;
            let background_rms = 0.1 * scale;
            let detection_sigma = 3.0;

            let stars = match detector {
                StarFinder::Naive => {
                    // Use the naive detector directly with f64 image
                    use simulator::image_proc::detection::detect_stars as detect_stars_naive;
                    detect_stars_naive(&image.view(), Some(0.1))
                        .into_iter()
                        .map(|s| Box::new(s) as Box<dyn StellarSource>)
                        .collect::<Vec<_>>()
                }
                _ => {
                    // Use the unified interface for DAO and IRAF
                    use simulator::image_proc::detection::detect_stars_unified;
                    let scaled_airy_disk = PixelScaledAiryDisk::with_fwhm(airy_disk_pixels, 550.0);
                    match detect_stars_unified(
                        u16_image.view(),
                        *detector,
                        &scaled_airy_disk,
                        background_rms,
                        detection_sigma,
                    ) {
                        Ok(stars) => stars,
                        Err(_) => {
                            failed_detections += 1;
                            continue;
                        }
                    }
                }
            };

            // Skip if no detection
            if stars.is_empty() {
                failed_detections += 1;
                continue;
            }

            // Use the first detected star
            let star = &stars[0];

            // Calculate errors
            let (star_x, star_y) = star.get_centroid();
            let error_x = position_x - star_x;
            let error_y = position_y - star_y;
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
        println!("Average Error: {avg_error:.6} pixels");
        println!("Maximum Error: {max_error:.6} pixels");
        println!("Total Tests: {total_tests}");
        if failed_detections > 0 {
            println!("Failed Detections: {failed_detections}");
        }

        // Show histogram range info
        println!("\nHistogram Range: 0.0 to {:.6} pixels", max_error * 1.1);

        // Print detailed histograms with adaptive ranges
        // For magnitude, use max error + 10% padding
        let mag_max = max_error * 1.1;
        print_histogram(
            "Error Magnitude Distribution (pixels)",
            &errors_magnitude,
            40,
            0.0,
            mag_max,
        );

        // For X/Y errors, find the actual range
        let x_scan = MinMaxScan::new(&errors_x);
        let (x_min, x_max) = x_scan
            .min_max()
            .expect("Failed to compute x error bounds: errors contain NaN");
        let y_scan = MinMaxScan::new(&errors_y);
        let (y_min, y_max) = y_scan
            .min_max()
            .expect("Failed to compute y error bounds: errors contain NaN");

        // Use symmetric range based on largest absolute value
        let xy_range = x_min
            .abs()
            .max(x_max.abs())
            .max(y_min.abs())
            .max(y_max.abs())
            * 1.1;

        print_histogram(
            "Error X Distribution (pixels)",
            &errors_x,
            40,
            -xy_range,
            xy_range,
        );
        print_histogram(
            "Error Y Distribution (pixels)",
            &errors_y,
            40,
            -xy_range,
            xy_range,
        );

        // Additional analysis with more precision levels
        println!("\nPrecision Analysis:");
        let errors_below_01 = errors_magnitude.iter().filter(|&&e| e < 0.1).count();
        let errors_below_05 = errors_magnitude.iter().filter(|&&e| e < 0.05).count();
        let errors_below_025 = errors_magnitude.iter().filter(|&&e| e < 0.025).count();
        let errors_below_01_pct = (errors_below_01 as f64 / total_tests as f64) * 100.0;
        let errors_below_05_pct = (errors_below_05 as f64 / total_tests as f64) * 100.0;
        let errors_below_025_pct = (errors_below_025 as f64 / total_tests as f64) * 100.0;

        println!("Errors < 0.1 pixels: {errors_below_01} ({errors_below_01_pct:.2}%)");
        println!("Errors < 0.05 pixels: {errors_below_05} ({errors_below_05_pct:.2}%)");
        println!("Errors < 0.025 pixels: {errors_below_025} ({errors_below_025_pct:.2}%)");

        // Add finer precision levels if average error is very small
        if avg_error < 0.01 {
            let errors_below_01_pix = errors_magnitude.iter().filter(|&&e| e < 0.01).count();
            let errors_below_005_pix = errors_magnitude.iter().filter(|&&e| e < 0.005).count();
            let errors_below_0025_pix = errors_magnitude.iter().filter(|&&e| e < 0.0025).count();
            let errors_below_01_pix_pct = (errors_below_01_pix as f64 / total_tests as f64) * 100.0;
            let errors_below_005_pix_pct =
                (errors_below_005_pix as f64 / total_tests as f64) * 100.0;
            let errors_below_0025_pix_pct =
                (errors_below_0025_pix as f64 / total_tests as f64) * 100.0;

            println!("Errors < 0.01 pixels: {errors_below_01_pix} ({errors_below_01_pix_pct:.2}%)");
            println!(
                "Errors < 0.005 pixels: {errors_below_005_pix} ({errors_below_005_pix_pct:.2}%)"
            );
            println!(
                "Errors < 0.0025 pixels: {errors_below_0025_pix} ({errors_below_0025_pix_pct:.2}%)"
            );
        }

        // X bias analysis
        let mean_x_error = errors_x.iter().sum::<f64>() / errors_x.len() as f64;
        println!(
            "Mean X Error: {mean_x_error:.6} pixels (negative = detected position is to the right)"
        );

        // Y bias analysis
        let mean_y_error = errors_y.iter().sum::<f64>() / errors_y.len() as f64;
        println!("Mean Y Error: {mean_y_error:.6} pixels (negative = detected position is below)");
    }
}

/// Print a histogram using the viz module
fn print_histogram(title: &str, data: &[f64], bins: usize, min_val: f64, max_val: f64) {
    // Format title with consistent styling
    println!("\n{title}");
    println!("-----------------------------------------------------------");

    use viz::histogram::{Histogram, HistogramConfig};

    // Create histogram with the specified bins and range
    if let Ok(mut hist) = Histogram::new_equal_bins(min_val..max_val, bins) {
        // Configure histogram display options
        let mut config = HistogramConfig {
            max_bar_width: 40,
            ..HistogramConfig::default()
        };
        config.bar_char = '#';
        config.show_empty_bins = false;
        config.title = Some(title.to_string());

        hist = hist.with_config(config);

        // Add data to histogram
        hist.add_all(data.iter().copied());

        // Print the formatted histogram
        if let Ok(formatted) = hist.format() {
            println!("{formatted}");
        }

        // Print out of range count
        let total_in_hist = hist.total_count();
        let out_of_range = data.len() as u64 - total_in_hist;

        if out_of_range > 0 {
            println!("Out of range: {out_of_range} values");
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

        println!("Mean: {mean_str}, Median: {median_str}");
        println!("Std. Deviation: {std_dev:.6}");

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
    // Use more decimal places for very small values
    if value.abs() < 0.01 {
        if value >= 0.0 {
            format!("+{value:.6}")
        } else {
            format!("{value:.6}")
        }
    } else if value >= 0.0 {
        format!("+{value:.3}")
    } else {
        format!("{value:.3}")
    }
}

/// Test with different sigma values to see effect on centroid accuracy
fn test_sigma_effect(image_size: usize, detector: &StarFinder) {
    let center_x = image_size as f64 / 2.0;
    let center_y = image_size as f64 / 2.0;
    let position_x = center_x + 0.3; // Test at a fixed 0.3 subpixel offset
    let position_y = center_y + 0.7; // Test at a fixed 0.7 subpixel offset

    println!("=== Testing effect of PSF width (sigma) on centroid accuracy [{detector:?}] ===");
    println!("Fixed position: ({position_x:.3}, {position_y:.3})");
    println!("Sigma\tDetected Position\t\tError\t\tAspect Ratio");
    println!("----------------------------------------------------------------------");

    // Test a range of sigma values
    for sigma in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0].iter() {
        // Create an empty image
        let mut image = Array2::<f64>::zeros((image_size, image_size));

        // Generate a single star
        create_gaussian(&mut image, position_x, position_y, 1.0, *sigma);

        // Convert f64 image to u16 for detection
        let max_scan = MinMaxScan::new(image.as_slice().unwrap());
        let max_val = max_scan
            .max()
            .expect("Failed to compute max value: image contains NaN or is empty");
        let scale = if max_val > 0.0 {
            65535.0 / max_val
        } else {
            1.0
        };
        let u16_image = image.mapv(|v| (v * scale) as u16);

        // Run detection using appropriate method
        let airy_disk_pixels = *sigma * 2.0;
        let background_rms = 0.1 * scale;
        let detection_sigma = 3.0;

        let stars = match detector {
            StarFinder::Naive => {
                // Use the naive detector directly with f64 image
                use simulator::image_proc::detection::detect_stars as detect_stars_naive;
                detect_stars_naive(&image.view(), Some(0.1))
                    .into_iter()
                    .map(|s| Box::new(s) as Box<dyn StellarSource>)
                    .collect::<Vec<_>>()
            }
            _ => {
                // Use the unified interface for DAO and IRAF
                use simulator::image_proc::detection::detect_stars_unified;
                let scaled_airy_disk = PixelScaledAiryDisk::with_fwhm(airy_disk_pixels, 550.0);
                match detect_stars_unified(
                    u16_image.view(),
                    *detector,
                    &scaled_airy_disk,
                    background_rms,
                    detection_sigma,
                ) {
                    Ok(stars) => stars,
                    Err(_) => {
                        println!("{sigma:.2}\tNo detection\t\t-\t\t-");
                        continue;
                    }
                }
            }
        };

        if stars.is_empty() {
            println!("{sigma:.2}\tNo detection\t\t-\t\t-");
            continue;
        }

        // Use the first detected star
        let star = &stars[0];

        // Calculate errors
        let (star_x, star_y) = star.get_centroid();
        let error_detected = ((position_x - star_x).powi(2) + (position_y - star_y).powi(2)).sqrt();

        // Print results
        println!("{sigma:.2}\t({star_x:.3}, {star_y:.3})\t{error_detected:.6}\t-");
    }
}

fn main() {
    // Initialize logging from environment variables
    env_logger::init();

    // Test parameters
    let image_size = 256;
    let sigma = 1.0; // Standard deviation of the Gaussian PSF

    // Test all three detectors
    let detectors = vec![StarFinder::Naive, StarFinder::Dao, StarFinder::Iraf];

    for detector in &detectors {
        println!("\n{}", "=".repeat(80));
        println!("Testing with {detector:?} detector");
        println!("{}\n", "=".repeat(80));

        // Run basic tests at specific positions
        println!("=== Basic position tests ===\n");

        println!("Integer position:");
        run_single_star_test(image_size, 128.0, 128.0, sigma, detector);

        println!("\nHalf-pixel position:");
        run_single_star_test(image_size, 128.5, 128.5, sigma, detector);

        println!("\nQuarter-pixel position:");
        run_single_star_test(image_size, 128.25, 128.75, sigma, detector);

        // Run comprehensive grid test within a single pixel
        println!("\n\n");
        run_subpixel_grid_test(image_size, sigma, detector);

        // Test effect of PSF width on centroid accuracy
        println!("\n\n");
        test_sigma_effect(image_size, detector);
    }
}

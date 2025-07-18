//! Performance benchmarks for star image rendering functions.
//!
//! This example provides performance benchmarks for key image processing functions,
//! particularly the `add_stars_to_image` function which is critical for simulation
//! performance. The benchmarks test various combinations of:
//!
//! - Image sizes (512×512, 1024×1024, 2048×2048)
//! - Star counts (10, 100, 1000 stars per image)
//! - PSF sizes (sigma values of 1.0, 2.0, 4.0 pixels)
//!
//! This helps identify performance bottlenecks and guide optimization efforts
//! for large-scale astronomical image simulations.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example run_benchmarks
//! ```
//!
//! For more detailed benchmarks using the `criterion` framework:
//! ```bash
//! cargo bench
//! ```

use simulator::image_proc::airy::PixelScaledAiryDisk;
use simulator::image_proc::render::{add_stars_to_image, StarInFrame};
use starfield::catalogs::StarData;
use starfield::Equatorial;
use std::time::Instant;

/// Creates a collection of test stars for benchmarking purposes.
///
/// This function generates synthetic stars with uniform flux distributed
/// in a grid pattern across a 100×100 pixel area. All stars use the same
/// placeholder astronomical data but vary in position.
///
/// # Arguments
///
/// * `count` - Number of test stars to create
/// * `flux` - Flux value to assign to all stars (in arbitrary units)
///
/// # Returns
///
/// Vector of `StarInFrame` objects ready for image rendering
///
/// # Examples
///
/// ```rust
/// let stars = create_test_stars(100, 1000.0);
/// assert_eq!(stars.len(), 100);
/// assert!(stars.iter().all(|s| s.flux == 1000.0));
/// ```
fn create_test_stars(count: usize, flux: f64) -> Vec<StarInFrame> {
    let mut stars = Vec::with_capacity(count);

    // Create test star data
    let star_data = StarData {
        id: 0,
        magnitude: 10.0,
        position: Equatorial::from_degrees(0.0, 0.0),
        b_v: None,
    };

    // Create stars with positions spread across the image
    for i in 0..count {
        let x = (i % 100) as f64;
        let y = (i / 100) as f64;

        stars.push(StarInFrame {
            x,
            y,
            flux,
            star: star_data,
        });
    }

    stars
}

/// Runs comprehensive performance benchmarks for the `add_stars_to_image` function.
///
/// This function systematically tests different combinations of image sizes,
/// star counts, and PSF parameters to measure rendering performance. Results
/// are printed in a tabular format showing configuration and execution time.
///
/// The benchmark covers:
/// - **Image sizes**: 512×512, 1024×1024, 2048×2048 pixels
/// - **Star counts**: 10, 100, 1000 stars per image
/// - **PSF sigma**: 1.0, 2.0, 4.0 pixels (controls star size)
///
/// Each combination is run once with timing measured using `std::time::Instant`.
/// For more rigorous statistical benchmarking, use `cargo bench` instead.
///
/// # Performance Notes
///
/// - Larger images scale roughly quadratically with dimension
/// - More stars scale linearly with count
/// - Larger PSF sigma increases computation per star
fn bench_add_stars_to_image() {
    println!("\n=== Benchmarking add_stars_to_image ===");
    println!("Format: image_size × image_size, star_count stars, sigma PSF");
    println!("{:<40} {:<15}", "Configuration", "Time (ms)");
    println!("{}", "-".repeat(55));

    // Test different image sizes
    let image_sizes = [512, 1024, 2048];
    // Test different star counts
    let star_counts = [10, 100, 1000];
    // Test different PSF sizes
    let sigma_values = [1.0, 2.0, 4.0];

    for &size in &image_sizes {
        for &star_count in &star_counts {
            for &sigma in &sigma_values {
                // Create test stars
                let stars = create_test_stars(star_count, 1000.0);

                // Run the benchmark
                let start = Instant::now();
                let airy_pix = PixelScaledAiryDisk::with_fwhm(sigma, 550.0);
                let _image = add_stars_to_image(size, size, &stars, airy_pix);
                let duration = start.elapsed();

                println!(
                    "{:<40} {:<15.2}",
                    format!("{}×{}, {} stars, sigma {}", size, size, star_count, sigma),
                    duration.as_secs_f64() * 1000.0
                );
            }
        }
    }
}

fn main() {
    println!("======================================");
    println!("Performance Benchmark for Simulator Functions");
    println!("======================================");

    bench_add_stars_to_image();

    println!("\nFor more detailed benchmarks, run:");
    println!("cargo bench");
}

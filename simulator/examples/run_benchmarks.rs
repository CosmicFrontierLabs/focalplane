use simulator::image_proc::airy::ScaledAiryDisk;
use simulator::image_proc::render::{add_stars_to_image, StarInFrame};
use starfield::catalogs::StarData;
use starfield::Equatorial;
use std::time::Instant;

/// Creates test stars for benchmarking
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

/// Benchmark add_stars_to_image function
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
                let airy_pix = ScaledAiryDisk::with_fwhm(sigma);
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

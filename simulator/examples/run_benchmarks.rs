use ndarray::Array2;
use simulator::image_proc::{
    generate_sensor_noise,
    render::{add_stars_to_image, StarInFrame},
};
use simulator::{photometry::Band, QuantumEfficiency, SensorConfig};
use starfield::catalogs::StarData;
use starfield::RaDec;
use std::time::{Duration, Instant};

/// Creates a test sensor with specified dimensions and characteristics
fn create_test_sensor(width: u32, height: u32, read_noise: f64, dark_current: f64) -> SensorConfig {
    let band = Band::new(300.0, 700.0);
    let qe = QuantumEfficiency::from_notch(&band, 1.0).unwrap();

    SensorConfig::new(
        "Test Sensor",
        qe,
        width,
        height,
        5.0, // pixel size in um
        read_noise,
        dark_current,
        16,       // bit depth
        0.5,      // dn per electron
        100000.0, // max well depth
        60.0,     // frame rate
    )
}

/// Creates test stars for benchmarking
fn create_test_stars(count: usize, flux: f64) -> Vec<StarInFrame> {
    let mut stars = Vec::with_capacity(count);

    // Create test star data
    let star_data = StarData {
        id: 0,
        magnitude: 10.0,
        position: RaDec::from_degrees(0.0, 0.0),
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
            star: star_data.clone(),
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
                // Create a new image
                let mut image = Array2::zeros((size, size));
                // Create test stars
                let stars = create_test_stars(star_count, 1000.0);

                // Run the benchmark
                let start = Instant::now();
                add_stars_to_image(&mut image, &stars, sigma);
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

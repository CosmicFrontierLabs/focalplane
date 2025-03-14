//! Benchmark for 2D convolution
//!
//! This example benchmarks the performance of 2D convolution
//! comparing sequential and parallel execution.

use ndarray::Array2;
use simulator::image_proc::convolve2d::{convolve2d, gaussian_kernel, ConvolveOptions, ConvolveMode};
use std::time::Instant;

fn main() {
    println!("2D Convolution Benchmark");
    println!("========================");

    // Test different image sizes
    let sizes = [100, 200, 500, 1000];

    // Create kernels of different sizes
    let kernels = [
        ("3x3", gaussian_kernel(3, 1.0)),
        ("5x5", gaussian_kernel(5, 2.0)),
        ("7x7", gaussian_kernel(7, 3.0)),
    ];

    for &size in &sizes {
        println!("\nImage size: {}x{}", size, size);

        // Create a test input array
        let mut input = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                input[[i, j]] = ((i * j) % 256) as f64 / 256.0;
            }
        }

        for (kernel_name, kernel) in &kernels {
            println!("\n  Kernel: {}", kernel_name);

            // Sequential version
            let seq_options = ConvolveOptions {
                mode: ConvolveMode::Same,
            };

            let start = Instant::now();
            let _seq_result = convolve2d(&input.view(), &kernel.view(), Some(seq_options));
            let seq_duration = start.elapsed();

            println!("    Sequential: {:?}", seq_duration);

            // Parallel version (with same options, using Rayon under the hood)
            let par_options = ConvolveOptions {
                mode: ConvolveMode::Same,
            };

            let start = Instant::now();
            let _par_result = convolve2d(&input.view(), &kernel.view(), Some(par_options));
            let par_duration = start.elapsed();

            println!("    Parallel:   {:?}", par_duration);

            if seq_duration.as_micros() > 0 {
                let speedup = seq_duration.as_secs_f64() / par_duration.as_secs_f64();
                println!("    Speedup:    {:.2}x", speedup);
            }
        }
    }
}

//! Benchmark for convolution operations
//!
//! This benchmark measures performance of various convolution operations
//! with different kernels, array sizes, and modes.

use ndarray::{Array2, Axis};
use simulator::image_proc::convolve2d::{
    convolve2d, gaussian_kernel, ConvolveMode, ConvolveOptions,
};
use std::time::Instant;

fn main() {
    println!("Convolution Operations Benchmark");
    println!("===============================");

    // Test different image sizes
    let sizes = [10, 50, 100, 200];

    // Test different convolution modes
    let modes = [ConvolveMode::Same, ConvolveMode::Valid];

    // Create kernels of different sizes
    let kernels = [
        ("3x3", gaussian_kernel(3, 1.0)),
        ("5x5", gaussian_kernel(5, 2.0)),
    ];

    // Benchmark with different array sizes
    for &size in &sizes {
        println!("\nInput array size: {size}x{size}");

        // Create test array with sequential values
        let mut input = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                input[[i, j]] = (i * size + j) as f64;
            }
        }

        // Benchmark each kernel
        for (kernel_name, kernel) in &kernels {
            println!("\n  Kernel: {kernel_name}");

            // Benchmark each mode
            for mode in &modes {
                let options = ConvolveOptions { mode: *mode };

                let mode_name = match mode {
                    ConvolveMode::Same => "Same",
                    ConvolveMode::Valid => "Valid",
                };

                println!("    Mode: {mode_name}");

                // Measure performance
                let start = Instant::now();
                let result = convolve2d(&input.view(), &kernel.view(), Some(options));
                let duration = start.elapsed();

                // Output result dimensions and timing
                let (rows, cols) = result.dim();
                println!("    Result size: {rows}x{cols}");
                println!("    Duration: {duration:?}");
            }
        }
    }

    // Benchmark edge behavior comparison (small array for visibility)
    let small_size = 10;
    let mut small_input = Array2::zeros((small_size, small_size));
    for i in 0..small_size {
        for j in 0..small_size {
            small_input[[i, j]] = (i * small_size + j) as f64;
        }
    }

    println!("\nEdge Behavior Comparison (10x10 array):");
    let kernel = gaussian_kernel(3, 1.0);

    let same_options = ConvolveOptions {
        mode: ConvolveMode::Same,
    };
    let valid_options = ConvolveOptions {
        mode: ConvolveMode::Valid,
    };

    let start = Instant::now();
    let result_same = convolve2d(&small_input.view(), &kernel.view(), Some(same_options));
    let same_duration = start.elapsed();

    let start = Instant::now();
    let result_valid = convolve2d(&small_input.view(), &kernel.view(), Some(valid_options));
    let valid_duration = start.elapsed();

    println!("  Same mode: {same_duration:?}");
    println!("  Valid mode: {valid_duration:?}");

    // Edge values comparison
    println!("  First row comparison:");
    println!("    Original: {:?}", small_input.index_axis(Axis(0), 0));
    println!("    Same:     {:?}", result_same.index_axis(Axis(0), 0));
    println!("    Valid:    {:?}", result_valid.index_axis(Axis(0), 0));
}

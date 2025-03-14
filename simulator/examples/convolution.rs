//! Example demonstrating 2D convolution
//!
//! This example shows how to use the convolve2d functionality
//! with different kernels and options.

use ndarray::{Array2, Axis};
use simulator::image_proc::convolve2d::{
    convolve2d, gaussian_kernel, ConvolveMode, ConvolveOptions,
};

fn main() {
    println!("2D Convolution Example");
    println!("======================");

    // Create a test input array with a simple pattern
    let mut input = Array2::zeros((10, 10));
    for i in 0..10 {
        for j in 0..10 {
            input[[i, j]] = (i * 10 + j) as f64;
        }
    }

    println!("Input array (10x10):");
    print_array(&input);

    // Create a Gaussian kernel
    let kernel_size = 3;
    let sigma = 1.0;
    let kernel = gaussian_kernel(kernel_size, sigma);

    println!(
        "\nGaussian kernel ({}x{}, sigma={}):",
        kernel_size, kernel_size, sigma
    );
    print_array(&kernel);

    // Perform convolution with different options
    let options = ConvolveOptions {
        mode: ConvolveMode::Same,
    };

    println!("\nConvolution with Same mode:");
    let result1 = convolve2d(&input.view(), &kernel.view(), Some(options));
    print_array(&result1);

    let options = ConvolveOptions {
        mode: ConvolveMode::Valid,
    };

    println!("\nConvolution with Valid mode:");
    let result2 = convolve2d(&input.view(), &kernel.view(), Some(options));
    print_array(&result2);

    // Compare edge behaviors
    println!("\nMode comparison (first row):");
    println!("Original: {:?}", input.index_axis(Axis(0), 0));
    println!("Same:     {:?}", result1.index_axis(Axis(0), 0));
    println!("Valid:    {:?}", result2.index_axis(Axis(0), 0));

    // Example of image smoothing with larger kernel
    println!("\nImage smoothing with larger kernel (5x5):");
    let large_kernel = gaussian_kernel(5, 2.0);
    let options = ConvolveOptions {
        mode: ConvolveMode::Same,
    };
    let smoothed = convolve2d(&input.view(), &large_kernel.view(), Some(options));
    print_array(&smoothed);
}

// Helper function to print a 2D array
fn print_array(arr: &Array2<f64>) {
    let (rows, cols) = arr.dim();

    // Limit display size for large arrays
    let max_rows = 10.min(rows);
    let max_cols = 10.min(cols);

    for i in 0..max_rows {
        for j in 0..max_cols {
            print!("{:7.2} ", arr[[i, j]]);
        }
        println!();
    }

    if rows > max_rows || cols > max_cols {
        println!("... (truncated)");
    }
}

//! 2D convolution operations for ndarray
//!
//! This module provides efficient 2D convolution operations for ndarray,
//! with optional parallel processing using rayon.

use ndarray::{Array2, Zip};
use std::ops::{Add, Mul};

/// Options for controlling the convolution operation
#[derive(Debug, Clone, Copy)]
pub struct ConvolveOptions {
    /// Whether to use parallel processing with rayon
    pub parallel: bool,
    
    /// Controls how edges are handled
    pub edge_mode: EdgeMode,
}

impl Default for ConvolveOptions {
    fn default() -> Self {
        Self {
            parallel: true,
            edge_mode: EdgeMode::Constant(0.0),
        }
    }
}

/// Edge handling modes for convolution
#[derive(Debug, Clone, Copy)]
pub enum EdgeMode {
    /// Uses a constant value for pixels outside image bounds
    Constant(f64),
    
    /// Reflects the image at the edges
    Reflect,
    
    /// Wraps around to the other side of the image
    Wrap,
    
    /// Extends the edge pixels outward
    Extend,
}

/// Convolve a 2D array with a kernel
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `kernel` - Convolution kernel
/// * `options` - Convolution options
///
/// # Returns
///
/// A new Array2 containing the convolution result
pub fn convolve2d<T>(
    input: &Array2<T>,
    kernel: &Array2<T>,
    options: ConvolveOptions,
) -> Array2<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Send + Sync + 'static,
    T: std::iter::Sum,
    T: From<f64>,
    T: num_traits::Zero,
{
    let (input_rows, input_cols) = input.dim();
    let (kernel_rows, kernel_cols) = kernel.dim();
    
    // Kernel center
    let kr = kernel_rows / 2;
    let kc = kernel_cols / 2;
    
    // Create output array with same dimensions as input
    let mut output = Array2::zeros((input_rows, input_cols));
    
    // Process the convolution
    if options.parallel {
        process_parallel(input, kernel, &mut output, kr, kc, options.edge_mode);
    } else {
        process_sequential(input, kernel, &mut output, kr, kc, options.edge_mode);
    }
    
    output
}

// Parallel processing using rayon
fn process_parallel<T>(
    input: &Array2<T>,
    kernel: &Array2<T>,
    output: &mut Array2<T>,
    kr: usize,
    kc: usize,
    edge_mode: EdgeMode,
) where
    T: Copy + Add<Output = T> + Mul<Output = T> + Send + Sync + 'static,
    T: std::iter::Sum,
    T: From<f64>,
    T: num_traits::Zero,
{
    let (input_rows, input_cols) = input.dim();
    let (kernel_rows, kernel_cols) = kernel.dim();
    
    Zip::indexed(output).par_for_each(|(i, j), out| {
        let mut sum = T::zero();
        
        for ki in 0..kernel_rows {
            for kj in 0..kernel_cols {
                // Calculate input array coordinates
                let ii = i as isize + ki as isize - kr as isize;
                let jj = j as isize + kj as isize - kc as isize;
                
                // Get input value based on edge mode
                let input_val = get_pixel(input, ii, jj, input_rows, input_cols, edge_mode);
                
                // Accumulate sum - kernel is not flipped for correlation
                sum = sum + input_val * kernel[[ki, kj]];
            }
        }
        
        *out = sum;
    });
}

// Sequential processing
fn process_sequential<T>(
    input: &Array2<T>,
    kernel: &Array2<T>,
    output: &mut Array2<T>,
    kr: usize,
    kc: usize,
    edge_mode: EdgeMode,
) where
    T: Copy + Add<Output = T> + Mul<Output = T>,
    T: std::iter::Sum,
    T: From<f64>,
    T: num_traits::Zero,
{
    let (input_rows, input_cols) = input.dim();
    let (kernel_rows, kernel_cols) = kernel.dim();
    
    for i in 0..input_rows {
        for j in 0..input_cols {
            let mut sum = T::zero();
            
            for ki in 0..kernel_rows {
                for kj in 0..kernel_cols {
                    // Calculate input array coordinates
                    let ii = i as isize + ki as isize - kr as isize;
                    let jj = j as isize + kj as isize - kc as isize;
                    
                    // Get input value based on edge mode
                    let input_val = get_pixel(input, ii, jj, input_rows, input_cols, edge_mode);
                    
                    // Accumulate sum - kernel is not flipped for correlation
                    sum = sum + input_val * kernel[[ki, kj]];
                }
            }
            
            output[[i, j]] = sum;
        }
    }
}

// Helper function to get pixel values with edge handling
fn get_pixel<T>(
    input: &Array2<T>,
    i: isize,
    j: isize,
    rows: usize,
    cols: usize,
    edge_mode: EdgeMode,
) -> T
where
    T: Copy + From<f64>,
{
    if i >= 0 && i < rows as isize && j >= 0 && j < cols as isize {
        // Within bounds
        return input[[i as usize, j as usize]];
    }
    
    match edge_mode {
        EdgeMode::Constant(value) => T::from(value),
        
        EdgeMode::Reflect => {
            let ri = reflect_index(i, rows as isize);
            let rj = reflect_index(j, cols as isize);
            input[[ri as usize, rj as usize]]
        },
        
        EdgeMode::Wrap => {
            let wi = wrap_index(i, rows as isize);
            let wj = wrap_index(j, cols as isize);
            input[[wi as usize, wj as usize]]
        },
        
        EdgeMode::Extend => {
            let ei = clamp(i, 0, rows as isize - 1);
            let ej = clamp(j, 0, cols as isize - 1);
            input[[ei as usize, ej as usize]]
        },
    }
}

// Reflect indices for edge handling
fn reflect_index(idx: isize, size: isize) -> isize {
    if idx < 0 {
        -idx - 1
    } else if idx >= size {
        2 * size - idx - 1
    } else {
        idx
    }
}

// Wrap indices for edge handling
fn wrap_index(idx: isize, size: isize) -> isize {
    ((idx % size) + size) % size
}

// Clamp indices within bounds
fn clamp(idx: isize, min: isize, max: isize) -> isize {
    if idx < min {
        min
    } else if idx > max {
        max
    } else {
        idx
    }
}

/// Create a Gaussian kernel for convolution
///
/// # Arguments
///
/// * `size` - Kernel size (must be odd)
/// * `sigma` - Standard deviation of the Gaussian
///
/// # Returns
///
/// A 2D array containing the Gaussian kernel
pub fn gaussian_kernel(size: usize, sigma: f64) -> Array2<f64> {
    assert!(size % 2 == 1, "Kernel size must be odd");
    
    let center = size / 2;
    let mut kernel = Array2::zeros((size, size));
    let mut sum = 0.0;
    
    for i in 0..size {
        for j in 0..size {
            let x = i as isize - center as isize;
            let y = j as isize - center as isize;
            let value = (-(x * x + y * y) as f64 / (2.0 * sigma * sigma)).exp();
            kernel[[i, j]] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    kernel.mapv_inplace(|x| x / sum);
    
    kernel
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_convolution_identity() {
        // Identity kernel should leave input unchanged
        let input = arr2(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        
        let kernel = arr2(&[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]);
        
        let output = convolve2d(&input, &kernel, ConvolveOptions::default());
        
        for i in 0..3 {
            for j in 0..3 {
                assert!(f64::abs(output[[i, j]] - input[[i, j]]) < 1e-10f64);
            }
        }
    }
    
    #[test]
    fn test_gaussian_kernel() {
        // Test that gaussian kernel sums to 1.0
        let kernel = gaussian_kernel(5, 1.0);
        let sum: f64 = kernel.iter().sum();
        
        assert!((sum - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_edge_modes() {
        let input = arr2(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        
        // Print input for debugging
        println!("Input matrix:");
        for i in 0..3 {
            for j in 0..3 {
                print!("{} ", input[[i, j]]);
            }
            println!();
        }
        
        let kernel = arr2(&[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]);
        
        // Print kernel for debugging
        println!("Kernel matrix:");
        for i in 0..3 {
            for j in 0..3 {
                print!("{} ", kernel[[i, j]]);
            }
            println!();
        }
        
        // Test constant edge mode
        let options = ConvolveOptions {
            parallel: false,
            edge_mode: EdgeMode::Constant(0.0),
        };
        
        let output = convolve2d(&input, &kernel, options);
        
        // Print the entire output for debugging
        println!("Output matrix:");
        for i in 0..3 {
            for j in 0..3 {
                print!("{} ", output[[i, j]]);
            }
            println!();
        }
        
        // With kernel [0,0,0; 0,0,1; 0,0,0] we expect
        // input elements shifted to the right
        assert!(f64::abs(output[[0, 0]] - 2.0) < 1e-6f64);
        assert!(f64::abs(output[[0, 1]] - 3.0) < 1e-6f64); 
        assert!(f64::abs(output[[0, 2]] - 0.0) < 1e-6f64);
        assert!(f64::abs(output[[1, 1]] - 6.0) < 1e-6f64);
        assert!(f64::abs(output[[2, 2]] - 0.0) < 1e-6f64);
    }
    
    #[test]
    fn test_parallel() {
        let input = arr2(&[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ]);
        
        let kernel = gaussian_kernel(3, 1.0);
        
        // Run with and without parallelism
        let seq_options = ConvolveOptions {
            parallel: false,
            edge_mode: EdgeMode::Constant(0.0),
        };
        
        let par_options = ConvolveOptions {
            parallel: true,
            edge_mode: EdgeMode::Constant(0.0),
        };
        
        let seq_output = convolve2d(&input, &kernel, seq_options);
        let par_output = convolve2d(&input, &kernel, par_options);
        
        // Both results should be the same
        for i in 0..5 {
            for j in 0..5 {
                assert!(f64::abs(seq_output[[i, j]] - par_output[[i, j]]) < 1e-10f64);
            }
        }
    }
}
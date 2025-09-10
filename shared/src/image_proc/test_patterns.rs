//! Test pattern generation utilities for image processing validation
//!
//! Provides functions to generate various test patterns useful for
//! validating image I/O, coordinate systems, and processing algorithms.

use ndarray::Array2;
use num_traits::Zero;

/// Generate a checkerboard pattern with specified block size
///
/// # Arguments
/// * `blocks_x` - Number of blocks in X direction
/// * `blocks_y` - Number of blocks in Y direction  
/// * `block_size` - Size of each square block in pixels
/// * `black_value` - Value for black squares
/// * `white_value` - Value for white squares
/// * `top_left_black` - If true, top-left corner starts with black
///
/// # Returns
/// Array2 containing the checkerboard pattern
pub fn generate_checkerboard<T>(
    blocks_x: usize,
    blocks_y: usize,
    block_size: usize,
    black_value: T,
    white_value: T,
    top_left_black: bool,
) -> Array2<T>
where
    T: Clone + Zero,
{
    let width = blocks_x * block_size;
    let height = blocks_y * block_size;

    let mut pattern = Array2::zeros((height, width));

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            // Determine if this block should be white
            let is_white = if top_left_black {
                (block_x + block_y) % 2 == 1
            } else {
                (block_x + block_y) % 2 == 0
            };

            let value = if is_white {
                white_value.clone()
            } else {
                black_value.clone()
            };

            // Fill the block
            for dy in 0..block_size {
                for dx in 0..block_size {
                    let y = block_y * block_size + dy;
                    let x = block_x * block_size + dx;
                    pattern[[y, x]] = value.clone();
                }
            }
        }
    }

    pattern
}

/// Generate a horizontal gradient pattern
///
/// # Arguments
/// * `width` - Width of the pattern
/// * `height` - Height of the pattern
/// * `min_value` - Minimum value (left edge)
/// * `max_value` - Maximum value (right edge)
///
/// # Returns
/// Array2 containing horizontal gradient
pub fn generate_horizontal_gradient<T>(
    width: usize,
    height: usize,
    min_value: T,
    max_value: T,
) -> Array2<T>
where
    T: Clone + num_traits::NumCast + Zero,
{
    let mut pattern = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let t = x as f64 / (width - 1) as f64;
            let min_f: f64 = num_traits::cast(min_value.clone()).unwrap();
            let max_f: f64 = num_traits::cast(max_value.clone()).unwrap();
            let value_f = min_f + t * (max_f - min_f);
            pattern[[y, x]] = num_traits::cast(value_f).unwrap();
        }
    }

    pattern
}

/// Generate a vertical gradient pattern
///
/// # Arguments
/// * `width` - Width of the pattern
/// * `height` - Height of the pattern
/// * `min_value` - Minimum value (top edge)
/// * `max_value` - Maximum value (bottom edge)
///
/// # Returns
/// Array2 containing vertical gradient
pub fn generate_vertical_gradient<T>(
    width: usize,
    height: usize,
    min_value: T,
    max_value: T,
) -> Array2<T>
where
    T: Clone + num_traits::NumCast + Zero,
{
    let mut pattern = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let t = y as f64 / (height - 1) as f64;
            let min_f: f64 = num_traits::cast(min_value.clone()).unwrap();
            let max_f: f64 = num_traits::cast(max_value.clone()).unwrap();
            let value_f = min_f + t * (max_f - min_f);
            pattern[[y, x]] = num_traits::cast(value_f).unwrap();
        }
    }

    pattern
}

/// Generate a Gaussian blob pattern
///
/// # Arguments
/// * `size` - Width and height of the square pattern
/// * `sigma` - Standard deviation of the Gaussian
/// * `amplitude` - Peak amplitude at center
///
/// # Returns
/// Array2<f32> containing Gaussian blob
pub fn generate_gaussian_blob(size: usize, sigma: f32, amplitude: f32) -> Array2<f32> {
    let mut pattern = Array2::zeros((size, size));
    let center = size as f32 / 2.0;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist_sq = dx * dx + dy * dy;
            pattern[[y, x]] = amplitude * (-dist_sq / (2.0 * sigma * sigma)).exp();
        }
    }

    pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkerboard_generation() {
        // Test 3x3 blocks, 2x2 pixels each
        let pattern = generate_checkerboard::<u8>(3, 3, 2, 0, 255, true);

        // Check dimensions
        assert_eq!(pattern.dim(), (6, 6));

        // Check top-left is black
        assert_eq!(pattern[[0, 0]], 0);
        assert_eq!(pattern[[1, 1]], 0);

        // Check adjacent block is white
        assert_eq!(pattern[[0, 2]], 255);
        assert_eq!(pattern[[1, 3]], 255);
    }

    #[test]
    fn test_gradient_generation() {
        let h_grad = generate_horizontal_gradient::<u16>(100, 50, 0, 1000);
        assert_eq!(h_grad.dim(), (50, 100));
        assert_eq!(h_grad[[0, 0]], 0);
        assert_eq!(h_grad[[0, 99]], 1000);

        let v_grad = generate_vertical_gradient::<u16>(50, 100, 0, 1000);
        assert_eq!(v_grad.dim(), (100, 50));
        assert_eq!(v_grad[[0, 0]], 0);
        assert_eq!(v_grad[[99, 0]], 1000);
    }
}

//! Bilinear interpolation for 2D data grids.
//!
//! This module provides efficient bilinear interpolation for both regular and irregular
//! 2D grids, with support for invalid data handling and configurable boundary behavior.

use ndarray::Array2;
use std::fmt;

/// Error types for bilinear interpolation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationError {
    /// Coordinate is outside the valid interpolation domain
    OutOfBounds {
        axis: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    /// No valid data points available for interpolation
    NoValidData(String),
    /// Inconsistent data dimensions
    DimensionMismatch {
        x_len: usize,
        y_len: usize,
        data_shape: (usize, usize),
    },
}

impl fmt::Display for InterpolationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpolationError::OutOfBounds {
                axis,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "{axis} coordinate {value} is outside valid range [{min}, {max}]"
                )
            }
            InterpolationError::NoValidData(msg) => {
                write!(f, "No valid data for interpolation: {msg}")
            }
            InterpolationError::DimensionMismatch {
                x_len,
                y_len,
                data_shape,
            } => {
                write!(
                    f,
                    "Data dimensions ({data_shape:?}) don't match coordinate lengths (x: {x_len}, y: {y_len})"
                )
            }
        }
    }
}

impl std::error::Error for InterpolationError {}

/// Bilinear interpolator for 2D data on regular or irregular grids.
///
/// Supports efficient interpolation with:
/// - Irregular grid spacing (binary search for coordinates)
/// - Invalid data handling (e.g., infinite values)
/// - Configurable boundary behavior
#[derive(Debug, Clone)]
pub struct BilinearInterpolator {
    /// X-axis coordinates (must be sorted in ascending order)
    x_coords: Vec<f64>,
    /// Y-axis coordinates (must be sorted in ascending order)
    y_coords: Vec<f64>,
    /// 2D data array indexed as [y_index, x_index]
    data: Array2<f64>,
    /// Whether to allow extrapolation beyond grid bounds
    allow_extrapolation: bool,
}

impl BilinearInterpolator {
    /// Create a new bilinear interpolator.
    ///
    /// # Arguments
    /// * `x_coords` - X-axis coordinates (must be sorted ascending)
    /// * `y_coords` - Y-axis coordinates (must be sorted ascending)
    /// * `data` - 2D data array with shape (y_coords.len(), x_coords.len())
    ///
    /// # Panics
    /// Panics if coordinates are not sorted or dimensions don't match
    pub fn new(
        x_coords: Vec<f64>,
        y_coords: Vec<f64>,
        data: Array2<f64>,
    ) -> Result<Self, InterpolationError> {
        // Validate dimensions
        let (ny, nx) = data.dim();
        if nx != x_coords.len() || ny != y_coords.len() {
            return Err(InterpolationError::DimensionMismatch {
                x_len: x_coords.len(),
                y_len: y_coords.len(),
                data_shape: (ny, nx),
            });
        }

        // Validate coordinates are sorted
        for i in 1..x_coords.len() {
            if x_coords[i] <= x_coords[i - 1] {
                panic!("X coordinates must be sorted in ascending order");
            }
        }
        for i in 1..y_coords.len() {
            if y_coords[i] <= y_coords[i - 1] {
                panic!("Y coordinates must be sorted in ascending order");
            }
        }

        Ok(Self {
            x_coords,
            y_coords,
            data,
            allow_extrapolation: false,
        })
    }

    /// Enable or disable extrapolation beyond grid bounds.
    pub fn with_extrapolation(mut self, allow: bool) -> Self {
        self.allow_extrapolation = allow;
        self
    }

    /// Find indices and interpolation weight for a coordinate value.
    ///
    /// Returns (lower_index, upper_index, weight) where weight is the
    /// fraction of the way from lower to upper (0.0 at lower, 1.0 at upper).
    fn find_indices_and_weight(&self, coords: &[f64], value: f64) -> Option<(usize, usize, f64)> {
        let n = coords.len();

        // Handle edge cases
        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some((0, 0, 0.0));
        }

        // Check bounds
        if value < coords[0] {
            if self.allow_extrapolation {
                return Some((0, 1, (value - coords[0]) / (coords[1] - coords[0])));
            }
            return None;
        }
        if value > coords[n - 1] {
            if self.allow_extrapolation {
                return Some((
                    n - 2,
                    n - 1,
                    (value - coords[n - 2]) / (coords[n - 1] - coords[n - 2]),
                ));
            }
            return None;
        }

        // Binary search for the interval
        let mut left = 0;
        let mut right = n - 1;

        while left < right - 1 {
            let mid = (left + right) / 2;
            if value < coords[mid] {
                right = mid;
            } else {
                left = mid;
            }
        }

        // Calculate interpolation weight
        let weight = (value - coords[left]) / (coords[right] - coords[left]);
        Some((left, right, weight))
    }

    /// Perform bilinear interpolation at the given coordinates.
    pub fn interpolate(&self, x: f64, y: f64) -> Result<f64, InterpolationError> {
        // Find x indices and weight
        let (x_low, x_high, x_weight) = self
            .find_indices_and_weight(&self.x_coords, x)
            .ok_or_else(|| InterpolationError::OutOfBounds {
                axis: "X",
                value: x,
                min: self.x_coords[0],
                max: self.x_coords[self.x_coords.len() - 1],
            })?;

        // Find y indices and weight
        let (y_low, y_high, y_weight) = self
            .find_indices_and_weight(&self.y_coords, y)
            .ok_or_else(|| InterpolationError::OutOfBounds {
                axis: "Y",
                value: y,
                min: self.y_coords[0],
                max: self.y_coords[self.y_coords.len() - 1],
            })?;

        // Get the four corner values
        let q11 = self.data[[y_low, x_low]];
        let q12 = self.data[[y_high, x_low]];
        let q21 = self.data[[y_low, x_high]];
        let q22 = self.data[[y_high, x_high]];

        // Standard bilinear interpolation
        let value = q11 * (1.0 - x_weight) * (1.0 - y_weight)
            + q21 * x_weight * (1.0 - y_weight)
            + q12 * (1.0 - x_weight) * y_weight
            + q22 * x_weight * y_weight;

        Ok(value)
    }

    /// Perform bilinear interpolation with invalid data handling.
    ///
    /// This method handles infinite or NaN values by using only valid
    /// neighboring points and renormalizing weights accordingly.
    pub fn interpolate_with_invalid_handling(
        &self,
        x: f64,
        y: f64,
    ) -> Result<f64, InterpolationError> {
        // Find x indices and weight
        let (x_low, x_high, x_weight) = self
            .find_indices_and_weight(&self.x_coords, x)
            .ok_or_else(|| InterpolationError::OutOfBounds {
                axis: "X",
                value: x,
                min: self.x_coords[0],
                max: self.x_coords[self.x_coords.len() - 1],
            })?;

        // Find y indices and weight
        let (y_low, y_high, y_weight) = self
            .find_indices_and_weight(&self.y_coords, y)
            .ok_or_else(|| InterpolationError::OutOfBounds {
                axis: "Y",
                value: y,
                min: self.y_coords[0],
                max: self.y_coords[self.y_coords.len() - 1],
            })?;

        // Get the four corner values
        let q11 = self.data[[y_low, x_low]];
        let q12 = self.data[[y_high, x_low]];
        let q21 = self.data[[y_low, x_high]];
        let q22 = self.data[[y_high, x_high]];

        // Collect valid points and their weights
        let mut valid_sum = 0.0;
        let mut weight_sum = 0.0;

        let corners = [
            (q11, (1.0 - x_weight) * (1.0 - y_weight)),
            (q21, x_weight * (1.0 - y_weight)),
            (q12, (1.0 - x_weight) * y_weight),
            (q22, x_weight * y_weight),
        ];

        for (value, weight) in &corners {
            if value.is_finite() {
                valid_sum += value * weight;
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            Ok(valid_sum / weight_sum)
        } else {
            Err(InterpolationError::NoValidData(format!(
                "All corner points are invalid at ({x}, {y})"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_simple_grid() -> BilinearInterpolator {
        let x_coords = vec![0.0, 1.0, 2.0];
        let y_coords = vec![0.0, 1.0, 2.0];
        let mut data = Array2::zeros((3, 3));

        // Fill with simple values: data[y,x] = x + y
        for y in 0..3 {
            for x in 0..3 {
                data[[y, x]] = x as f64 + y as f64;
            }
        }

        BilinearInterpolator::new(x_coords, y_coords, data).unwrap()
    }

    #[test]
    fn test_exact_grid_points() {
        let interp = create_simple_grid();

        // Test exact grid points
        assert_eq!(interp.interpolate(0.0, 0.0).unwrap(), 0.0);
        assert_eq!(interp.interpolate(1.0, 0.0).unwrap(), 1.0);
        assert_eq!(interp.interpolate(0.0, 1.0).unwrap(), 1.0);
        assert_eq!(interp.interpolate(2.0, 2.0).unwrap(), 4.0);
    }

    #[test]
    fn test_interpolation_midpoints() {
        let interp = create_simple_grid();

        // Test midpoint interpolation
        assert_relative_eq!(interp.interpolate(0.5, 0.0).unwrap(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(interp.interpolate(0.0, 0.5).unwrap(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(interp.interpolate(0.5, 0.5).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(interp.interpolate(1.5, 1.5).unwrap(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_out_of_bounds() {
        let interp = create_simple_grid();

        // Test out of bounds
        assert!(interp.interpolate(-0.1, 0.0).is_err());
        assert!(interp.interpolate(0.0, -0.1).is_err());
        assert!(interp.interpolate(2.1, 0.0).is_err());
        assert!(interp.interpolate(0.0, 2.1).is_err());
    }

    #[test]
    fn test_extrapolation() {
        let interp = create_simple_grid().with_extrapolation(true);

        // Test extrapolation
        assert_relative_eq!(
            interp.interpolate(-1.0, 0.0).unwrap(),
            -1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(interp.interpolate(3.0, 0.0).unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(
            interp.interpolate(0.0, -1.0).unwrap(),
            -1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(interp.interpolate(0.0, 3.0).unwrap(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_irregular_grid() {
        let x_coords = vec![0.0, 0.3, 1.0, 2.5];
        let y_coords = vec![0.0, 0.7, 1.5, 3.0];
        let mut data = Array2::zeros((4, 4));

        // Fill with values
        for (j, &y) in y_coords.iter().enumerate() {
            for (i, &x) in x_coords.iter().enumerate() {
                data[[j, i]] = x * y;
            }
        }

        let interp = BilinearInterpolator::new(x_coords.clone(), y_coords.clone(), data).unwrap();

        // Test exact points
        assert_eq!(interp.interpolate(0.3, 0.7).unwrap(), 0.3 * 0.7);
        assert_eq!(interp.interpolate(1.0, 1.5).unwrap(), 1.0 * 1.5);

        // Test interpolation
        let x_test = 0.5;
        let y_test = 1.0;
        let result = interp.interpolate(x_test, y_test).unwrap();

        // For f(x,y) = x*y, bilinear interpolation should give exact result
        assert_relative_eq!(result, x_test * y_test, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_data_handling() {
        let x_coords = vec![0.0, 1.0, 2.0];
        let y_coords = vec![0.0, 1.0, 2.0];
        let mut data = Array2::zeros((3, 3));

        // Fill with some invalid values
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 2.0;
        data[[1, 0]] = 3.0;
        data[[1, 1]] = f64::INFINITY; // Invalid!

        let interp = BilinearInterpolator::new(x_coords, y_coords, data).unwrap();

        // Regular interpolation should return infinity
        assert!(interp.interpolate(0.5, 0.5).unwrap().is_infinite());

        // Invalid handling should skip the infinite value
        let result = interp.interpolate_with_invalid_handling(0.5, 0.5).unwrap();
        assert!(result.is_finite());

        // Should be weighted average of the three valid corners
        let expected = (1.0 * 0.25 + 2.0 * 0.25 + 3.0 * 0.25) / 0.75;
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let x_coords = vec![0.0, 1.0];
        let y_coords = vec![0.0, 1.0, 2.0];
        let data = Array2::zeros((2, 2)); // Wrong size!

        let result = BilinearInterpolator::new(x_coords, y_coords, data);
        assert!(matches!(
            result,
            Err(InterpolationError::DimensionMismatch { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "X coordinates must be sorted")]
    fn test_unsorted_x_coords() {
        let x_coords = vec![1.0, 0.0, 2.0]; // Not sorted!
        let y_coords = vec![0.0, 1.0];
        let data = Array2::zeros((2, 3));

        BilinearInterpolator::new(x_coords, y_coords, data).unwrap();
    }

    #[test]
    #[should_panic(expected = "Y coordinates must be sorted")]
    fn test_unsorted_y_coords() {
        let x_coords = vec![0.0, 1.0];
        let y_coords = vec![1.0, 0.0]; // Not sorted!
        let data = Array2::zeros((2, 2));

        BilinearInterpolator::new(x_coords, y_coords, data).unwrap();
    }
}

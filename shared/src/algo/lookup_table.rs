//! Lookup table with quadratic interpolation for fast function evaluation.
//!
//! This module provides a generic lookup table that precomputes function values
//! at regular intervals and uses quadratic (second-order) interpolation for
//! intermediate values. This is particularly useful for expensive transcendental
//! functions like exp(), sin(), cos(), etc.
//!
//! # Features
//!
//! - Precomputed function values at regular intervals
//! - Quadratic interpolation for smooth approximation
//! - Configurable table size and domain
//! - Generic over floating point types
//!
//! # Example
//!
//! ```
//! use shared::algo::lookup_table::LookupTable;
//!
//! // Create a lookup table for exp(x) from -5 to 5 with 1000 points
//! let exp_table = LookupTable::new(-5.0, 5.0, 1000, |x| x.exp());
//!
//! // Fast evaluation using interpolation
//! let approx_value = exp_table.eval(2.3).unwrap();
//! let exact_value = 2.3_f64.exp();
//! assert!((approx_value - exact_value).abs() < 1e-6);
//! ```

use std::fmt::Debug;
use thiserror::Error;

/// Error type for lookup table operations
#[derive(Debug, Error)]
pub enum LookupError {
    /// Value is outside the domain bounds
    #[error("Value {value} is outside domain bounds ({min}, {max})")]
    OutOfBounds { value: f64, min: f64, max: f64 },
}

/// A lookup table with quadratic interpolation for fast function evaluation.
///
/// The table stores precomputed function values at regular intervals and uses
/// quadratic interpolation to approximate values between grid points.
#[derive(Debug, Clone)]
pub struct LookupTable {
    /// Domain bounds (min, max)
    domain: (f64, f64),
    /// Step size between grid points
    dx: f64,
    /// Precomputed function values
    values: Vec<f64>,
    /// Number of points in the table
    n_points: usize,
}

impl LookupTable {
    /// Create a new lookup table by precomputing function values.
    ///
    /// # Arguments
    ///
    /// * `x_min` - Lower bound of the domain
    /// * `x_max` - Upper bound of the domain
    /// * `n_points` - Number of points to precompute (minimum 3 for quadratic interpolation)
    /// * `f` - Function to evaluate at each grid point
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `x_min >= x_max`
    /// - `n_points < 3` (need at least 3 points for quadratic interpolation)
    pub fn new<F>(x_min: f64, x_max: f64, n_points: usize, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        assert!(x_min < x_max, "x_min must be less than x_max");
        assert!(
            n_points >= 3,
            "Need at least 3 points for quadratic interpolation"
        );

        let dx = (x_max - x_min) / (n_points - 1) as f64;
        let mut values = Vec::with_capacity(n_points);

        // Precompute all function values
        for i in 0..n_points {
            let x = x_min + i as f64 * dx;
            values.push(f(x));
        }

        Self {
            domain: (x_min, x_max),
            dx,
            values,
            n_points,
        }
    }

    /// Evaluate the function at a given point using quadratic interpolation.
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the function
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Interpolated function value at x
    /// * `Err(LookupError::OutOfBounds)` - If x is outside the domain
    pub fn eval(&self, x: f64) -> Result<f64, LookupError> {
        // Check if x is within bounds
        if x < self.domain.0 || x > self.domain.1 {
            return Err(LookupError::OutOfBounds {
                value: x,
                min: self.domain.0,
                max: self.domain.1,
            });
        }
        // Find the index of the nearest grid point
        let t = (x - self.domain.0) / self.dx;
        let i = t.floor() as isize;

        // Handle boundary cases
        let i = if i < 0 {
            0
        } else if i >= (self.n_points - 2) as isize {
            self.n_points - 3
        } else {
            i as usize
        };

        // Get three points for quadratic interpolation
        let x0 = self.domain.0 + i as f64 * self.dx;
        let x1 = x0 + self.dx;
        let x2 = x1 + self.dx;

        let y0 = self.values[i];
        let y1 = self.values[i + 1];
        let y2 = self.values[i + 2];

        // Quadratic interpolation using Lagrange formula
        let l0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2));
        let l1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2));
        let l2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1));

        Ok(y0 * l0 + y1 * l1 + y2 * l2)
    }

    /// Get the domain bounds of the lookup table.
    pub fn domain(&self) -> (f64, f64) {
        self.domain
    }

    /// Get the number of points in the lookup table.
    pub fn size(&self) -> usize {
        self.n_points
    }

    /// Get the step size between grid points.
    pub fn step_size(&self) -> f64 {
        self.dx
    }

    /// Check if a value is within the table's domain.
    pub fn contains(&self, x: f64) -> bool {
        x >= self.domain.0 && x <= self.domain.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_function() {
        // Linear function should be exact with quadratic interpolation
        let table = LookupTable::new(0.0, 10.0, 11, |x| 2.0 * x + 3.0);

        for x in [0.0, 2.5, 5.0, 7.3, 9.9] {
            let expected = 2.0 * x + 3.0;
            let actual = table.eval(x).unwrap();
            assert_relative_eq!(actual, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_quadratic_function() {
        // Quadratic function should be exact with quadratic interpolation
        let table = LookupTable::new(-5.0, 5.0, 21, |x| x * x - 2.0 * x + 1.0);

        for x in [-4.5, -2.0, 0.0, 1.5, 3.7, 4.9] {
            let expected = x * x - 2.0 * x + 1.0;
            let actual = table.eval(x).unwrap();
            assert_relative_eq!(actual, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_exponential_approximation() {
        let table = LookupTable::new(-2.0, 2.0, 1000, |x| x.exp());

        // Test at various points
        for x in [-1.5_f64, -0.5, 0.0, 0.5, 1.0, 1.5] {
            let expected = x.exp();
            let actual = table.eval(x).unwrap();
            // With 1000 points, should be very accurate
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sine_approximation() {
        use std::f64::consts::PI;
        let table = LookupTable::new(0.0, 2.0 * PI, 500, |x| x.sin());

        for x in [0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0] {
            let expected = x.sin();
            let actual = table.eval(x).unwrap();
            assert_relative_eq!(actual, expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_boundary_handling() {
        let table = LookupTable::new(0.0, 1.0, 11, |x| x * x);

        // Test at boundaries
        assert_relative_eq!(table.eval(0.0).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(table.eval(1.0).unwrap(), 1.0, epsilon = 1e-10);

        // Test out of bounds errors
        assert!(matches!(
            table.eval(-0.1),
            Err(LookupError::OutOfBounds { .. })
        ));
        assert!(matches!(
            table.eval(1.1),
            Err(LookupError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn test_domain_methods() {
        let table = LookupTable::new(-10.0, 10.0, 101, |x| x);

        assert_eq!(table.domain(), (-10.0, 10.0));
        assert_eq!(table.size(), 101);
        assert_relative_eq!(table.step_size(), 0.2, epsilon = 1e-10);

        assert!(table.contains(0.0));
        assert!(table.contains(-10.0));
        assert!(table.contains(10.0));
        assert!(!table.contains(-10.1));
        assert!(!table.contains(10.1));
    }

    #[test]
    #[should_panic(expected = "x_min must be less than x_max")]
    fn test_invalid_domain() {
        LookupTable::new(5.0, 3.0, 10, |x| x);
    }

    #[test]
    #[should_panic(expected = "Need at least 3 points")]
    fn test_insufficient_points() {
        LookupTable::new(0.0, 1.0, 2, |x| x);
    }

    #[test]
    fn test_high_accuracy_approximation() {
        // Test that increasing points improves accuracy
        let f = |x: f64| (x * x).exp() * x.cos();

        let coarse = LookupTable::new(-1.0, 1.0, 10, f);
        let fine = LookupTable::new(-1.0, 1.0, 100, f);
        let very_fine = LookupTable::new(-1.0, 1.0, 1000, f);

        let test_x = 0.567;
        let exact = f(test_x);

        let error_coarse = (coarse.eval(test_x).unwrap() - exact).abs();
        let error_fine = (fine.eval(test_x).unwrap() - exact).abs();
        let error_very_fine = (very_fine.eval(test_x).unwrap() - exact).abs();

        // Errors should decrease with more points
        assert!(error_fine < error_coarse);
        assert!(error_very_fine < error_fine);
        assert!(error_very_fine < 1e-8);
    }

    #[test]
    fn test_out_of_bounds_error() {
        let table = LookupTable::new(0.0, 10.0, 11, |x| x * x);

        // Values within bounds should work
        assert!(table.eval(0.0).is_ok());
        assert!(table.eval(5.0).is_ok());
        assert!(table.eval(10.0).is_ok());

        // Values outside bounds should error
        match table.eval(-1.0) {
            Err(LookupError::OutOfBounds { value, min, max }) => {
                assert_eq!(value, -1.0);
                assert_eq!(min, 0.0);
                assert_eq!(max, 10.0);
            }
            _ => panic!("Expected OutOfBounds error"),
        }

        match table.eval(10.1) {
            Err(LookupError::OutOfBounds { value, min, max }) => {
                assert_eq!(value, 10.1);
                assert_eq!(min, 0.0);
                assert_eq!(max, 10.0);
            }
            _ => panic!("Expected OutOfBounds error"),
        }
    }

    #[test]
    fn test_clone() {
        let original = LookupTable::new(0.0, 1.0, 11, |x| x * x);
        let cloned = original.clone();

        assert_eq!(cloned.domain(), original.domain());
        assert_eq!(cloned.size(), original.size());

        // Test that they produce same results
        for x in [0.1, 0.5, 0.9] {
            assert_eq!(cloned.eval(x).unwrap(), original.eval(x).unwrap());
        }
    }
}

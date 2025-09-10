//! Miscellaneous mathematical and utility algorithms.
//!
//! This module provides general-purpose mathematical functions and utilities
//! that don't fit into more specific algorithm categories. Currently includes:
//!
//! - **Linear interpolation**: Fast 1D interpolation with error handling
//! - **Numerical utilities**: Common mathematical operations for scientific computing
//! - **Coordinate conversions**: RA/Dec conversions between formats
//!
//! These functions are designed for performance and robustness in scientific
//! applications, with comprehensive error handling and input validation.

use thiserror::Error;

/// Errors that can occur during interpolation operations.
///
/// This enum provides detailed error information for interpolation failures,
/// allowing callers to handle different error conditions appropriately.
#[derive(Error, Debug)]
pub enum InterpError {
    #[error("Value {0} is out of bounds for interpolation range [{1}, {2}]")]
    OutOfBounds(f64, f64, f64),
    #[error("Input vectors must have at least 2 points")]
    InsufficientData,
    #[error("Input vectors must have the same length")]
    MismatchedLengths,
    #[error("X values must be sorted in ascending order")]
    UnsortedData,
}

/// Performs linear interpolation on 1D data using binary search for efficiency.
///
/// This function implements fast linear interpolation by:
/// 1. Validating input data (lengths, sorting, sufficient points)
/// 2. Using binary search to find the correct interval (O(log n))
/// 3. Applying linear interpolation formula: y = y₁ + t(y₂ - y₁)
///
/// where t = (x - x₁)/(x₂ - x₁) is the interpolation parameter.
///
/// # Arguments
///
/// * `x` - The x-coordinate at which to interpolate
/// * `xs` - Array of x-coordinates (must be sorted in ascending order)
/// * `ys` - Array of corresponding y-values (must match length of xs)
///
/// # Returns
///
/// * `Ok(f64)` - The interpolated y-value at position x
/// * `Err(InterpError)` - Various error conditions:
///   - `OutOfBounds`: x is outside the range [xs[0], xs[n-1]]
///   - `InsufficientData`: xs and ys have fewer than 2 points
///   - `MismatchedLengths`: xs and ys have different lengths
///   - `UnsortedData`: xs values are not in ascending order
///
/// # Examples
///
/// ```rust
/// use shared::algo::misc::interp;
///
/// let xs = vec![0.0, 1.0, 2.0, 3.0];
/// let ys = vec![0.0, 2.0, 4.0, 6.0];
///
/// // Interpolate at x=1.5
/// let result = interp(1.5, &xs, &ys).unwrap();
/// assert_eq!(result, 3.0);
///
/// // Exact match returns exact value
/// let result = interp(2.0, &xs, &ys).unwrap();
/// assert_eq!(result, 4.0);
/// ```
///
/// # Performance
///
/// O(log n) time complexity due to binary search, where n is the number of data points.
/// Memory usage is O(1) as no additional allocations are made.
pub fn interp(x: f64, xs: &[f64], ys: &[f64]) -> Result<f64, InterpError> {
    // Validate input vectors have matching lengths
    if xs.len() != ys.len() {
        return Err(InterpError::MismatchedLengths);
    }

    // Ensure we have at least 2 points for interpolation
    if xs.len() < 2 {
        return Err(InterpError::InsufficientData);
    }

    // Verify xs is sorted in ascending order
    for i in 1..xs.len() {
        if xs[i] < xs[i - 1] {
            return Err(InterpError::UnsortedData);
        }
    }

    // Check if x is within bounds
    if x < xs[0] || x > xs[xs.len() - 1] {
        return Err(InterpError::OutOfBounds(x, xs[0], xs[xs.len() - 1]));
    }

    // Use binary search to find the right interval
    // partition_point returns the index of the first element > x
    let idx = xs.partition_point(|&val| val <= x);

    // Handle edge cases
    if idx == 0 {
        // x equals xs[0]
        return Ok(ys[0]);
    }
    if idx == xs.len() {
        // x equals xs[n-1]
        return Ok(ys[xs.len() - 1]);
    }

    // Perform linear interpolation between points [idx-1] and [idx]
    let x1 = xs[idx - 1];
    let x2 = xs[idx];
    let y1 = ys[idx - 1];
    let y2 = ys[idx];

    // Calculate interpolation parameter t ∈ [0, 1]
    let t = (x - x1) / (x2 - x1);

    // Linear interpolation formula: y = y1 + t*(y2 - y1)
    Ok(y1 + t * (y2 - y1))
}

/// Normalize a vector of values to the range [0, 1] based on the maximum value.
///
/// This function scales all values in the input vector by dividing by the maximum value,
/// resulting in a normalized vector where the largest value becomes 1.0 and all other
/// values are proportionally scaled.
///
/// # Special Cases
/// - If all values are zero or the vector is empty, returns the input unchanged
/// - If all values are negative, returns the input unchanged
/// - Single element vectors are normalized to [1.0]
///
/// # Arguments
///
/// * `pts` - Vector of f64 values to normalize
///
/// # Returns
///
/// A new vector with normalized values in the range [0, 1]
///
/// # Example
///
/// ```
/// use shared::algo::misc::normalize;
///
/// let data = vec![0.5, 2.0, 1.0, 3.0];
/// let normalized = normalize(data);
/// assert_eq!(normalized, vec![0.5/3.0, 2.0/3.0, 1.0/3.0, 1.0]);
/// ```
pub fn normalize(pts: Vec<f64>) -> Vec<f64> {
    // Find maximum value using iterator max_by
    let max_val = pts
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(0.0);

    // If max is zero or negative, return original values
    if max_val <= 0.0 {
        return pts;
    }

    // Normalize by dividing all values by the maximum
    pts.iter().map(|&val| val / max_val).collect()
}

/// Converts Right Ascension from Hours, Minutes, Seconds to Decimal Degrees.
/// RA (Hours) * 15 = RA (Degrees)
pub fn ra_hms_to_deg(hours: f64, minutes: f64, seconds: f64) -> f64 {
    (hours + minutes / 60.0 + seconds / 3600.0) * 15.0
}

/// Converts Declination from Degrees, Minutes, Seconds to Decimal Degrees.
/// Handles negative declination correctly.
pub fn dec_dms_to_deg(degrees: f64, minutes: f64, seconds: f64) -> f64 {
    let sign = if degrees < 0.0 { -1.0 } else { 1.0 };
    (degrees.abs() + minutes / 60.0 + seconds / 3600.0) * sign
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0];
        assert_eq!(interp(2.0, &xs, &ys).unwrap(), 20.0);
    }

    #[test]
    fn test_linear_interpolation() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        assert_eq!(interp(1.5, &xs, &ys).unwrap(), 15.0);
        assert_eq!(interp(2.5, &xs, &ys).unwrap(), 25.0);
    }

    #[test]
    fn test_out_of_bounds() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        assert!(matches!(
            interp(0.5, &xs, &ys),
            Err(InterpError::OutOfBounds(_, _, _))
        ));
        assert!(matches!(
            interp(3.5, &xs, &ys),
            Err(InterpError::OutOfBounds(_, _, _))
        ));
    }

    #[test]
    fn test_mismatched_lengths() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0];
        assert!(matches!(
            interp(1.5, &xs, &ys),
            Err(InterpError::MismatchedLengths)
        ));
    }

    #[test]
    fn test_insufficient_data() {
        let xs = vec![1.0];
        let ys = vec![10.0];
        assert!(matches!(
            interp(1.0, &xs, &ys),
            Err(InterpError::InsufficientData)
        ));
    }

    #[test]
    fn test_unsorted_data() {
        let xs = vec![2.0, 1.0, 3.0];
        let ys = vec![20.0, 10.0, 30.0];
        assert!(matches!(
            interp(1.5, &xs, &ys),
            Err(InterpError::UnsortedData)
        ));
    }

    #[test]
    fn test_normalize() {
        // Basic normalization
        let data = vec![0.5, 2.0, 1.0, 3.0];
        let normalized = normalize(data);
        assert_eq!(normalized, vec![0.5 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0]);

        // All zeros
        let zeros = vec![0.0, 0.0, 0.0];
        let result = normalize(zeros.clone());
        assert_eq!(result, zeros);

        // Negative values
        let negative = vec![-1.0, -2.0, -3.0];
        let result = normalize(negative.clone());
        assert_eq!(result, negative);

        // Single value
        let single = vec![5.0];
        let result = normalize(single);
        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn test_ra_hms_to_deg() {
        // Test Andromeda Galaxy coordinates: 0h 42m 44.3s
        let ra_deg = ra_hms_to_deg(0.0, 42.0, 44.3);
        assert!((ra_deg - 10.6845833).abs() < 0.0001);
    }

    #[test]
    fn test_dec_dms_to_deg() {
        // Test Andromeda Galaxy coordinates: +41° 16' 9"
        let dec_deg = dec_dms_to_deg(41.0, 16.0, 9.0);
        assert!((dec_deg - 41.2691667).abs() < 0.0001);

        // Test negative declination
        let neg_dec_deg = dec_dms_to_deg(-30.0, 15.0, 30.0);
        assert!((neg_dec_deg - (-30.2583333)).abs() < 0.0001);
    }
}

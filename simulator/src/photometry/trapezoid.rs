//! Trapezoidal integration utility

use thiserror::Error;

/// Errors that can occur during trapezoidal integration
#[derive(Debug, Error)]
pub enum TrapezoidError {
    #[error("Insufficient points for integration, need at least 2 points")]
    InsufficientPoints,

    #[error("Points must be in ascending order")]
    NotAscending,
}

/// Performs trapezoidal integration of a function over a set of points.
///
/// # Arguments
///
/// * `corners` - The x coordinates of the trapezoid corners in ascending order
/// * `to_integrate` - The function to integrate
///
/// # Returns
///
/// The result of the trapezoidal integration or an error if the input is invalid.
pub fn trap_integrate<F>(corners: Vec<f32>, to_integrate: F) -> Result<f32, TrapezoidError>
where
    F: Fn(f32) -> f32,
{
    // Validate minimum point requirement for meaningful integration
    if corners.len() < 2 {
        return Err(TrapezoidError::InsufficientPoints);
    }

    // Ensure strictly ascending order for proper integration bounds
    for i in 1..corners.len() {
        if corners[i] <= corners[i - 1] {
            return Err(TrapezoidError::NotAscending);
        }
    }

    let mut integral_sum = 0.0;

    // Compute composite trapezoidal rule over all intervals
    for i in 0..corners.len() - 1 {
        let x_left = corners[i];
        let x_right = corners[i + 1];
        let y_left = to_integrate(x_left);
        let y_right = to_integrate(x_right);

        // Trapezoidal area: (base width) × (average height)
        // Mathematically: ∫[x₁,x₂] f(x)dx ≈ (x₂-x₁) × (f(x₁)+f(x₂))/2
        let interval_width = x_right - x_left;
        let average_height = (y_left + y_right) / 2.0;
        let trapezoid_area = interval_width * average_height;

        integral_sum += trapezoid_area;
    }

    Ok(integral_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_trap_integrate() {
        // Integrate f(x) = x^2 from 0 to 3 using 4 points
        // The trapezoidal approximation gives us:
        // (1-0)(0^2+1^2)/2 + (2-1)(1^2+2^2)/2 + (3-2)(2^2+3^2)/2
        // = 0.5 + 2.5 + 6.5 = 9.5
        let corners = vec![0.0, 1.0, 2.0, 3.0];
        let result = trap_integrate(corners, |x| x * x).unwrap();

        assert_relative_eq!(result, 9.5, epsilon = 1e-5);
    }

    #[test]
    fn test_insufficient_points() {
        let corners = vec![1.0];
        let result = trap_integrate(corners, |x| x);

        assert!(matches!(result, Err(TrapezoidError::InsufficientPoints)));
    }

    #[test]
    fn test_not_ascending() {
        let corners = vec![0.0, 2.0, 1.0, 3.0];
        let result = trap_integrate(corners, |x| x);

        assert!(matches!(result, Err(TrapezoidError::NotAscending)));
    }
}

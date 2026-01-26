//! 2x2 matrix utilities using nalgebra
//!
//! Provides common 2x2 matrix operations with error handling for
//! singular matrices and degenerate configurations.

use nalgebra::{Matrix2, Vector2};
use thiserror::Error;

/// Error when matrix inversion fails due to singular matrix
#[derive(Error, Debug, Clone, PartialEq)]
#[error("singular matrix: determinant={determinant:.6e}")]
pub struct SingularMatrixError {
    /// The determinant value (zero or near-zero)
    pub determinant: f64,
}

/// Error when vectors are nearly parallel (degenerate configuration)
#[derive(Error, Debug, Clone, PartialEq)]
#[error("degenerate vectors: angle between them is {angle_degrees:.2}°")]
pub struct DegenerateVectorsError {
    /// The angle between vectors in degrees
    pub angle_degrees: f64,
}

/// Threshold for considering a determinant as zero
const DETERMINANT_EPSILON: f64 = 1e-10;

/// Invert a 2x2 matrix with error handling for singular matrices
///
/// # Arguments
/// * `matrix` - The 2x2 matrix to invert
///
/// # Returns
/// * `Ok(Matrix2<f64>)` - The inverse matrix
/// * `Err(SingularMatrixError)` - If matrix is singular
pub fn invert_matrix(matrix: &Matrix2<f64>) -> Result<Matrix2<f64>, SingularMatrixError> {
    let det = matrix.determinant();

    if det.abs() < DETERMINANT_EPSILON {
        return Err(SingularMatrixError { determinant: det });
    }

    matrix
        .try_inverse()
        .ok_or(SingularMatrixError { determinant: det })
}

/// Compute the angle between two 2D vectors in degrees
///
/// # Arguments
/// * `v1` - First vector
/// * `v2` - Second vector
///
/// # Returns
/// Angle between vectors in degrees (0 to 180)
pub fn angle_between_vectors(v1: &Vector2<f64>, v2: &Vector2<f64>) -> f64 {
    let mag1 = v1.norm();
    let mag2 = v2.norm();

    if mag1 < f64::EPSILON || mag2 < f64::EPSILON {
        return 0.0;
    }

    let cos_angle = (v1.dot(v2) / (mag1 * mag2)).clamp(-1.0, 1.0);
    cos_angle.acos().to_degrees()
}

/// Build a 2x2 matrix from two column vectors with degeneracy check
///
/// # Arguments
/// * `col1` - First column vector
/// * `col2` - Second column vector
/// * `min_angle_degrees` - Minimum angle between vectors to accept (typically 5-10°)
///
/// # Returns
/// * `Ok(Matrix2<f64>)` - Matrix with col1 and col2 as columns
/// * `Err(DegenerateVectorsError)` - If vectors are nearly parallel
pub fn matrix_from_columns_checked(
    col1: Vector2<f64>,
    col2: Vector2<f64>,
    min_angle_degrees: f64,
) -> Result<Matrix2<f64>, DegenerateVectorsError> {
    let angle = angle_between_vectors(&col1, &col2);
    let valid_angle_range = min_angle_degrees..=(180.0 - min_angle_degrees);

    if !valid_angle_range.contains(&angle) {
        return Err(DegenerateVectorsError {
            angle_degrees: angle,
        });
    }

    Ok(Matrix2::from_columns(&[col1, col2]))
}

/// Create a 2x2 rotation matrix
///
/// # Arguments
/// * `angle_rad` - Rotation angle in radians (counter-clockwise)
///
/// # Returns
/// Rotation matrix
pub fn rotation_matrix(angle_rad: f64) -> Matrix2<f64> {
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    Matrix2::new(c, -s, s, c)
}

/// Create a 2x2 scaling matrix
///
/// # Arguments
/// * `sx` - Scale factor for x
/// * `sy` - Scale factor for y
///
/// # Returns
/// Scaling matrix
pub fn scale_matrix(sx: f64, sy: f64) -> Matrix2<f64> {
    Matrix2::new(sx, 0.0, 0.0, sy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_identity_transform() {
        let matrix = Matrix2::identity();
        let input = Vector2::new(3.0, 4.0);

        let output = matrix * input;

        assert_relative_eq!(output.x, input.x, epsilon = 1e-10);
        assert_relative_eq!(output.y, input.y, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_90_degrees() {
        let matrix = rotation_matrix(PI / 2.0);
        let input = Vector2::new(1.0, 0.0);

        let output = matrix * input;

        assert_relative_eq!(output.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(output.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inversion_identity() {
        let identity = Matrix2::identity();
        let inverse = invert_matrix(&identity).unwrap();

        assert_relative_eq!(inverse[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(inverse[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(inverse[(1, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(inverse[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inversion_roundtrip() {
        let matrix = Matrix2::new(2.0, 1.0, 1.0, 3.0);
        let inverse = invert_matrix(&matrix).unwrap();

        let product = matrix * inverse;

        assert_relative_eq!(product[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(product[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(product[(1, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(product[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_singular_matrix_error() {
        let matrix = Matrix2::new(1.0, 2.0, 2.0, 4.0);

        let result = invert_matrix(&matrix);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.determinant.abs() < 1e-9);
    }

    #[test]
    fn test_matrix_from_columns_orthogonal() {
        let col1 = Vector2::new(1.0, 0.0);
        let col2 = Vector2::new(0.0, 1.0);

        let matrix = matrix_from_columns_checked(col1, col2, 5.0).unwrap();

        let output = matrix * Vector2::new(5.0, 3.0);
        assert_relative_eq!(output.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(output.y, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_from_columns_degenerate() {
        let col1 = Vector2::new(1.0, 0.0);
        let col2 = Vector2::new(1.0, 0.01); // Almost parallel

        let result = matrix_from_columns_checked(col1, col2, 5.0);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.angle_degrees < 5.0);
    }

    #[test]
    fn test_angle_between_vectors() {
        // Orthogonal vectors
        let angle = angle_between_vectors(&Vector2::new(1.0, 0.0), &Vector2::new(0.0, 1.0));
        assert_relative_eq!(angle, 90.0, epsilon = 0.01);

        // Parallel vectors
        let angle = angle_between_vectors(&Vector2::new(1.0, 0.0), &Vector2::new(2.0, 0.0));
        assert_relative_eq!(angle, 0.0, epsilon = 0.01);

        // 45 degrees
        let angle = angle_between_vectors(&Vector2::new(1.0, 0.0), &Vector2::new(1.0, 1.0));
        assert_relative_eq!(angle, 45.0, epsilon = 0.01);
    }

    #[test]
    fn test_scale_matrix() {
        let scale = scale_matrix(2.0, 3.0);
        let input = Vector2::new(4.0, 5.0);

        let output = scale * input;

        assert_relative_eq!(output.x, 8.0, epsilon = 1e-10);
        assert_relative_eq!(output.y, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant() {
        let matrix = Matrix2::new(3.0, 1.0, 2.0, 4.0);
        let det = matrix.determinant();

        // det = 3*4 - 1*2 = 10
        assert_relative_eq!(det, 10.0, epsilon = 1e-10);
    }
}

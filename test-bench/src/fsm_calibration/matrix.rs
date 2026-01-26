//! FSM-specific matrix operations for axis calibration
//!
//! Wraps the shared matrix utilities with FSM-calibration-specific
//! error types and validation.

use meter_math::matrix2::{self, DegenerateVectorsError, SingularMatrixError};
use nalgebra::{Matrix2, Vector2};
use thiserror::Error;

/// Minimum angle (degrees) between FSM axes to consider calibration valid
const MIN_AXIS_ANGLE_DEGREES: f64 = 5.0;

/// Error when FSM calibration matrix is singular
#[derive(Error, Debug, Clone, PartialEq)]
#[error("singular FSM calibration matrix: determinant={determinant:.6e}")]
pub struct FsmSingularMatrixError {
    /// The determinant value (zero or near-zero)
    pub determinant: f64,
}

impl From<SingularMatrixError> for FsmSingularMatrixError {
    fn from(err: SingularMatrixError) -> Self {
        FsmSingularMatrixError {
            determinant: err.determinant,
        }
    }
}

/// Error when FSM axes are nearly parallel (degenerate calibration)
#[derive(Error, Debug, Clone, PartialEq)]
#[error("degenerate FSM axes: angle between axes is {angle_degrees:.2}°")]
pub struct FsmDegenerateAxesError {
    /// The angle between FSM axes in degrees
    pub angle_degrees: f64,
}

impl From<DegenerateVectorsError> for FsmDegenerateAxesError {
    fn from(err: DegenerateVectorsError) -> Self {
        FsmDegenerateAxesError {
            angle_degrees: err.angle_degrees,
        }
    }
}

/// Build a 2x2 transform matrix from FSM axis response vectors
///
/// The response vectors describe how sensor position changes per unit FSM command.
///
/// # Arguments
/// * `axis1_response` - (dx, dy) sensor motion per µrad for FSM axis 1
/// * `axis2_response` - (dx, dy) sensor motion per µrad for FSM axis 2
///
/// # Returns
/// * `Ok(Matrix2<f64>)` - Transform matrix where:
///   - `sensor_delta = matrix * fsm_command`
/// * `Err(FsmDegenerateAxesError)` - If axes are degenerate (parallel)
///
/// # Matrix Layout
/// ```text
/// [ axis1_dx  axis2_dx ]   [ axis1_cmd ]   [ sensor_dx ]
/// [ axis1_dy  axis2_dy ] × [ axis2_cmd ] = [ sensor_dy ]
/// ```
pub fn build_transform_matrix(
    axis1_response: Vector2<f64>,
    axis2_response: Vector2<f64>,
) -> Result<Matrix2<f64>, FsmDegenerateAxesError> {
    matrix2::matrix_from_columns_checked(axis1_response, axis2_response, MIN_AXIS_ANGLE_DEGREES)
        .map_err(Into::into)
}

/// Invert an FSM calibration matrix
///
/// # Arguments
/// * `matrix` - The 2x2 matrix to invert
///
/// # Returns
/// * `Ok(Matrix2<f64>)` - The inverse matrix
/// * `Err(FsmSingularMatrixError)` - If matrix is singular
pub fn invert_matrix(matrix: &Matrix2<f64>) -> Result<Matrix2<f64>, FsmSingularMatrixError> {
    matrix2::invert_matrix(matrix).map_err(Into::into)
}

// Re-export utility functions from shared
pub use matrix2::{rotation_matrix, scale_matrix};

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
    fn test_build_transform_orthogonal_axes() {
        let axis1_response = Vector2::new(1.0, 0.0);
        let axis2_response = Vector2::new(0.0, 1.0);

        let matrix = build_transform_matrix(axis1_response, axis2_response).unwrap();

        let output = matrix * Vector2::new(5.0, 3.0);
        assert_relative_eq!(output.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(output.y, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_build_transform_degenerate_axes() {
        let axis1_response = Vector2::new(1.0, 0.0);
        let axis2_response = Vector2::new(1.0, 0.01); // Almost parallel

        let result = build_transform_matrix(axis1_response, axis2_response);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.angle_degrees < 5.0);
    }

    #[test]
    fn test_full_calibration_workflow() {
        let axis1_response = Vector2::new(0.7, 0.3);
        let axis2_response = Vector2::new(-0.3, 0.7);

        let fsm_to_sensor = build_transform_matrix(axis1_response, axis2_response).unwrap();
        let sensor_to_fsm = invert_matrix(&fsm_to_sensor).unwrap();

        let desired_motion = Vector2::new(10.0, 5.0);
        let fsm_commands = sensor_to_fsm * desired_motion;
        let actual_motion = fsm_to_sensor * fsm_commands;

        assert_relative_eq!(actual_motion.x, desired_motion.x, epsilon = 1e-10);
        assert_relative_eq!(actual_motion.y, desired_motion.y, epsilon = 1e-10);
    }
}

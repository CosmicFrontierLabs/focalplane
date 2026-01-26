//! meter-math - Mathematical algorithms for astronomical simulations
//!
//! This crate provides core mathematical algorithms used in astronomical
//! simulations and star tracking systems, including:
//!
//! - **Quaternion** - 3D rotation representation and operations
//! - **ICP** - Iterative Closest Point algorithm for point cloud alignment
//! - **Interpolation** - Cubic spline and bilinear interpolation
//! - **Matrix** - 2D transformation matrices
//! - **Statistics** - Statistical functions (median, correlation, etc.)
//!
//! # Example
//!
//! ```text
//! use meter_math::{Quaternion, iterative_closest_point};
//! use nalgebra::Vector3;
//!
//! // Create a rotation quaternion
//! let axis = Vector3::new(0.0, 0.0, 1.0);
//! let angle = std::f64::consts::FRAC_PI_4; // 45 degrees
//! let q = Quaternion::from_axis_angle(&axis, angle);
//!
//! // Rotate a vector
//! let v = Vector3::new(1.0, 0.0, 0.0);
//! let rotated = q.rotate_vector(&v);
//! ```

pub mod bilinear;
pub mod icp;
pub mod matrix2;
pub mod quaternion;
pub mod spline;
pub mod stats;

// Re-export commonly used types
pub use bilinear::{BilinearInterpolator, InterpolationError};
pub use icp::{iterative_closest_point, ICPError, ICPResult, Locatable2d};
pub use matrix2::{
    angle_between_vectors, invert_matrix, matrix_from_columns_checked, rotation_matrix,
    scale_matrix, DegenerateVectorsError, SingularMatrixError,
};
pub use quaternion::Quaternion;
pub use spline::CubicSpline;
pub use stats::median;

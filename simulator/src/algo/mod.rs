//! Algorithms for various tasks in astronomical simulations
//!
//! This module provides algorithms for point cloud alignment, feature
//! extraction, quaternion mathematics, motion models, and other computational tasks.

pub mod icp;
pub mod motion;
pub mod quaternion;

pub use icp::{iterative_closest_point, ICPResult};
pub use motion::{MotionModel, XAxisSpinner, XYWobble};
pub use quaternion::Quaternion;

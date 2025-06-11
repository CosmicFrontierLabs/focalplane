//! Algorithms for various tasks in astronomical simulations
//!
//! This module provides algorithms for point cloud alignment, feature
//! extraction, quaternion mathematics, motion models, and other computational tasks.

pub mod icp;
pub mod misc;
pub mod motion;
pub mod parallel;
pub mod psd;
pub mod quaternion;
pub mod spline;

pub use icp::{iterative_closest_point, ICPResult};
pub use motion::{MotionModel, XAxisSpinner, XYWobble};
pub use parallel::process_array_in_parallel_chunks;
pub use quaternion::Quaternion;

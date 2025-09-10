//! Algorithms for various tasks in astronomical simulations
//!
//! This module provides algorithms for point cloud alignment, feature
//! extraction, quaternion mathematics, motion models, and other computational tasks.

pub mod bilinear;
pub mod icp;
pub mod lookup_table;
pub mod min_max_scan;
pub mod misc;
pub mod motion;
pub mod parallel;
pub mod psd;
pub mod quaternion;
pub mod spline;
pub mod stats;

pub use bilinear::{BilinearInterpolator, InterpolationError};
pub use icp::{iterative_closest_point, ICPResult};
pub use lookup_table::{LookupError, LookupTable};
pub use min_max_scan::{MinMaxError, MinMaxScan};
pub use misc::{dec_dms_to_deg, interp, normalize, ra_hms_to_deg, InterpError};
pub use motion::{MotionModel, XAxisSpinner, XYWobble};
pub use parallel::process_array_in_parallel_chunks;
pub use quaternion::Quaternion;

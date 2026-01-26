//! Algorithms for various tasks in astronomical simulations
//!
//! This module provides algorithms for motion models, lookup tables,
//! scanning, and other utilities.
//!
//! Core math algorithms (quaternion, ICP, interpolation, matrix, stats)
//! have been extracted to the `meter-math` crate.

pub mod lookup_table;
pub mod min_max_scan;
pub mod misc;
pub mod motion;
pub mod parallel;
pub mod psd;

pub use lookup_table::{LookupError, LookupTable};
pub use min_max_scan::{MinMaxError, MinMaxScan};
pub use misc::{dec_dms_to_deg, interp, normalize, ra_hms_to_deg, InterpError};
pub use motion::{MotionModel, XAxisSpinner, XYWobble};
pub use parallel::process_array_in_parallel_chunks;

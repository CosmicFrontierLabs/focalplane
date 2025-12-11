mod e727;
mod gcs;
mod s330;

pub use e727::{Axis, PiErrorCode, SpaParam, E727, RECORDER_SAMPLE_RATE_HZ};
pub use gcs::{GcsDevice, GcsError, GcsResult, DEFAULT_PORT};
pub use s330::S330;

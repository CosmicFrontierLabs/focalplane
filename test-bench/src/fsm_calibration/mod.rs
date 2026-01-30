//! FSM Axis Calibration
//!
//! This module provides calibration routines for mapping PI S-330 Fast Steering Mirror
//! (FSM) axes to FGS sensor coordinates. The calibration determines how FSM commands
//! translate to centroid motion on the detector.
//!
//! # Overview
//!
//! The calibration process:
//! 1. Move FSM to discrete step positions (negative, zero, positive)
//! 2. Wait for settle time after each move
//! 3. Collect tracking messages at each position
//! 4. Compute response vectors from centroid displacement
//! 5. Build 2x2 transform matrix mapping FSM commands to sensor motion
//!
//! # Modules
//!
//! - [`config`] - Calibration configuration parameters
//! - [`matrix`] - 2x2 transform matrix operations using nalgebra
//! - [`executor`] - Static step calibration workflow orchestration

pub mod config;
pub mod executor;
pub mod matrix;
pub mod transform;

pub use config::{FsmAxisCalibration, FsmCalibrationConfig};
pub use executor::{
    compute_response_vector, extract_intercept, CalibrationRawData, RawSample,
    StaticCalibrationError, StaticStepExecutor, StepMeasurement,
};
pub use hardware::FsmInterface;
pub use matrix::{
    build_transform_matrix, invert_matrix, FsmDegenerateAxesError, FsmSingularMatrixError,
};
pub use transform::{FsmTransform, FsmTransformError};

//! FSM Axis Calibration
//!
//! This module provides calibration routines for mapping PI S-330 Fast Steering Mirror
//! (FSM) axes to FGS sensor coordinates. The calibration determines how FSM commands
//! translate to centroid motion on the detector.
//!
//! # Overview
//!
//! The calibration process:
//! 1. Apply sinusoidal wiggle to each FSM axis independently
//! 2. Record centroid motion on the detector
//! 3. Fit sinusoids to extract response vectors
//! 4. Build 2x2 transform matrix mapping FSM commands to sensor motion
//!
//! # Modules
//!
//! - [`config`] - Calibration configuration parameters
//! - [`generator`] - Sinusoidal command generation for FSM wiggle
//! - [`fitter`] - Sinusoid fitting to extract amplitude and phase from centroid data
//! - [`matrix`] - 2x2 transform matrix operations using nalgebra
//! - [`executor`] - Calibration workflow orchestration

pub mod config;
pub mod executor;
pub mod fitter;
pub mod generator;
pub mod matrix;

pub use config::{FsmAxisCalibration, FsmCalibrationConfig};
pub use executor::{
    CalibrationError, CalibrationExecutor, CalibrationProgress, CentroidMeasurement,
    CentroidSource, FsmInterface, NoProgress, ProgressCallback,
};
pub use fitter::{FitError, SinusoidFit};
pub use generator::SinusoidGenerator;
pub use matrix::{
    build_transform_matrix, invert_matrix, FsmDegenerateAxesError, FsmSingularMatrixError,
};

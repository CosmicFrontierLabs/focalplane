//! Monocle harness for testing and simulation
//!
//! This module provides test harnesses and simulation infrastructure
//! for the monocle fine guidance system. It bridges the simulator
//! and monocle modules for testing and demonstration purposes.

pub mod motion_profiles;
pub mod runner;
pub mod simulator_camera;
pub mod tracking_plots;

pub use motion_profiles::TestMotions;
pub use runner::{run_fgs_with_callback, run_fgs_with_motion, RunnerResults};
pub use simulator_camera::SimulatorCamera;
pub use tracking_plots::{TrackingDataPoint, TrackingPlotConfig, TrackingPlotter};

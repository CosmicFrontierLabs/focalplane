//! Monocle harness for testing and simulation
//!
//! This module provides test harnesses and simulation infrastructure
//! for the monocle fine guidance system. It bridges the simulator
//! and monocle modules for testing and demonstration purposes.

pub mod helpers;
pub mod motion_profiles;
pub mod runner;
pub mod simulator_camera;
pub mod tracking_plots;

pub use helpers::{
    create_guide_star_catalog, create_jbt_hwk_camera, create_jbt_hwk_camera_with_catalog,
    create_jbt_hwk_test_satellite, create_simple_test_catalog, create_single_star_catalog,
};
pub use motion_profiles::TestMotions;
pub use runner::{run_fgs_with_callback, run_fgs_with_motion, RunnerResults};
pub use simulator_camera::SimulatorCamera;
pub use tracking_plots::{TrackingDataPoint, TrackingPlotConfig, TrackingPlotter};

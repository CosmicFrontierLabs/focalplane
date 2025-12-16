//! Estimation and control algorithms for the Fine Guidance System
//!
//! This module contains feedback controllers and state estimators used
//! for line of sight stabilization and tracking.

mod los_controller;

pub use los_controller::{LosControlOutput, LosController};

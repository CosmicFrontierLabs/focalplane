//! Noise processing module for astronomical image processing
//!
//! This module provides comprehensive noise handling capabilities:
//! - **generate**: Noise generation utilities for sensor simulation
//! - **quantify**: Noise estimation and quantification methods

pub mod generate;
pub mod quantify;

// Re-export commonly used functions for backward compatibility
pub use generate::{
    apply_poisson_photon_noise, generate_noise_with_precomputed_params, simple_normal_array,
};
pub use quantify::estimate_noise_level;

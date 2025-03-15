//! Visualization tools for meter-sim
//!
//! This crate provides visualization tools for the meter-sim project,
//! including text-based histograms, data plotting, and other visualization utilities.

use std::fmt;
use thiserror::Error;

/// Error types for visualization operations
#[derive(Debug, Error)]
pub enum VizError {
    #[error("Histogram error: {0}")]
    HistogramError(String),

    #[error("Formatting error: {0}")]
    FmtError(#[from] fmt::Error),
}

/// Result type for visualization operations
pub type Result<T> = std::result::Result<T, VizError>;

pub mod histogram;

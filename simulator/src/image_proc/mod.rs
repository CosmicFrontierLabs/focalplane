//! Image processing functionality specific to simulator.
//!
//! This module contains image processing functions that depend on
//! simulator-specific types and are not suitable for the shared module.

pub mod deposit;
pub mod render;

pub use deposit::{render_sources, splat_deposit, FrameSource, MeanFluxDeposit};

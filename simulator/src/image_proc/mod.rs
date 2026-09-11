//! Image processing functionality specific to simulator.
//!
//! This module contains image processing functions that depend on
//! simulator-specific types and are not suitable for the shared module.

#[cfg(feature = "solar-system")]
pub mod body_stamp;
pub mod compose;
pub mod deposit;
pub mod render;
pub mod sersic_splat;

pub use compose::{BodyComposite, Composite, PassContext, SecondPass};
pub use deposit::{render_sources, splat_deposit, FrameSource, MeanFluxDeposit};
pub use sersic_splat::SersicSplat;

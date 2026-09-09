//! Appearance models for resolved solar-system bodies.
//!
//! Everything under this module describes *how a surface element
//! reflects sunlight*; geometry (where the body is, how it is oriented)
//! comes from starfield, and detector radiometry lives in
//! [`crate::photometry`]. The split keeps the physics testable in
//! isolation: a [`brdf::Brdf`] can be integrated over a sphere and
//! checked against published geometric albedos without a telescope in
//! the loop.

pub mod brdf;
pub mod surface;

//! Ephemeris calculation library for astronomical bodies
//!
//! This crate provides functionality for calculating positions of celestial bodies
//! including stars, planets, and other astronomical objects.

use thiserror::Error;
use time::OffsetDateTime;

pub mod celestial;
pub mod coordinates;
pub mod time_utils;

/// Represents a point in the celestial sphere
#[derive(Debug, Clone, Copy)]
pub struct CelestialCoordinate {
    /// Right ascension in radians
    pub ra: f64,
    /// Declination in radians
    pub dec: f64,
}

/// Error types for ephemeris calculations
#[derive(Debug, Error)]
pub enum EphemerisError {
    #[error("Invalid time: {0}")]
    InvalidTime(String),

    #[error("Object not found: {0}")]
    ObjectNotFound(String),

    #[error("Calculation error: {0}")]
    CalculationError(String),
}

pub type Result<T> = std::result::Result<T, EphemerisError>;

/// Trait for objects that have a position in the sky
pub trait CelestialObject {
    /// Get the position of the object at a specific time
    fn position_at(&self, time: OffsetDateTime) -> Result<CelestialCoordinate>;
}

/// Create empty module files that will be filled later
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

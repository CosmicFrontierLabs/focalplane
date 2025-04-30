//! Photometry models and utilities

pub mod quantum_efficiency;
pub mod spectrum;
pub mod trapezoid;

pub use quantum_efficiency::QuantumEfficiency;
pub use spectrum::{Band, FlatStellarSpectrum, Spectrum};
pub use trapezoid::trap_integrate;

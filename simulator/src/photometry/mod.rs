//! Photometry models and utilities

pub mod color;
pub mod gaia;
pub mod human;
pub mod quantum_efficiency;
pub mod spectrum;
pub mod stellar;
pub mod trapezoid;

pub use color::{
    color_temperature_index, generate_temperature_sequence, spectrum_to_rgb_values,
    temperature_to_spectral_class,
};
pub use human::{HumanPhotoreceptor, HumanVision};
pub use quantum_efficiency::QuantumEfficiency;
pub use spectrum::{Band, Spectrum};
pub use stellar::{BlackbodyStellarSpectrum, FlatStellarSpectrum};
pub use trapezoid::trap_integrate;

//! Comprehensive stellar photometry and spectral analysis for astronomical simulations.
//!
//! This module provides a complete photometric toolkit for astronomical applications,
//! covering stellar spectra, filter systems, quantum efficiency models, and color
//! analysis. Essential for synthetic photometry, magnitude calculations, stellar
//! classification, and realistic color rendering of astronomical objects.
//!
//! # Core Components
//!
//! ## Spectral Models (`spectrum`, `stellar`)
//! - **BlackbodyStellarSpectrum**: Planck function stellar spectra with temperature scaling
//! - **FlatStellarSpectrum**: Uniform flux density spectra for calibration
//! - **Spectrum trait**: Common interface for all spectral energy distributions
//! - **Band**: Wavelength range definitions for filters and instruments
//!
//! ## Filter Systems (`filters`, `gaia`, `human`)
//! - **Johnson-Cousins UBV**: Standard astronomical photometric system
//! - **Gaia G-band**: ESA Gaia mission broadband passband
//! - **Human vision**: Cone cell quantum efficiency curves for color perception
//! - **Quantum efficiency**: Wavelength-dependent instrument response modeling
//!
//! ## Stellar Analysis (`color`, `stellar`)
//! - **Spectral classification**: OBAFGKM system temperature classification
//! - **Color indices**: B-V, U-B photometric color measurements
//! - **RGB conversion**: Human-perceptible stellar color calculation
//! - **Temperature sequences**: Logarithmic stellar population sampling
//!
//! ## Background Models (`zodiacal`, `stis_zodiacal`)
//! - **Zodiacal light**: Solar system dust scattering models
//! - **Scattered light**: Telescope background contributions
//! - **Angular dependencies**: Solar elongation and ecliptic latitude effects
//!
//! # Module Organization
//!
//! ## Core Spectral Framework
//! - **`spectrum`**: Fundamental spectrum trait and wavelength band definitions
//! - **`quantum_efficiency`**: Instrument response function modeling
//! - **`trapezoid`**: Numerical integration utilities for spectral calculations
//!
//! ## Stellar Models
//! - **`stellar`**: Blackbody and flat-spectrum stellar energy distributions
//! - **`color`**: Stellar color analysis and spectral classification systems
//!
//! ## Photometric Systems
//! - **`filters`**: Johnson-Cousins UBV standard photometric filters
//! - **`gaia`**: ESA Gaia mission G-band passband for space astrometry
//! - **`human`**: Human cone cell vision models for perceptual color accuracy
//!
//! ## Background Modeling
//! - **`zodiacal`**: Zodiacal light and scattered light background models
//! - **`stis_zodiacal`**: HST/STIS zodiacal light spectral measurements
//!
//! # Physical Accuracy
//!
//! All photometric models are based on:
//! - **Laboratory measurements**: Validated against instrument characterization
//! - **On-orbit calibration**: Space telescope cross-calibration studies
//! - **Standard references**: IAU, CIE, and observational astronomy standards
//! - **Physiological data**: Human vision based on cone cell measurements
//!
//! This ensures simulation results match real astronomical observations within
//! measurement uncertainties, making the toolkit suitable for mission planning,
//! algorithm validation, and scientific analysis.

pub mod color;
pub mod filters;
pub mod gaia;
pub mod human;
pub mod photoconversion;
pub mod quantum_efficiency;
pub mod spectrum;
pub mod stellar;
pub mod stis_zodiacal;
pub mod trapezoid;
pub mod zodiacal;

pub use color::{
    color_temperature_index, generate_temperature_sequence, rgb_values_to_color,
    spectrum_to_rgb_values, temperature_to_spectral_class,
};
pub use filters::{b_filter, u_filter, ubv_filters, v_filter};
pub use human::{HumanPhotoreceptor, HumanVision};
pub use photoconversion::photon_electron_fluxes;
pub use quantum_efficiency::QuantumEfficiency;
pub use spectrum::{Band, Spectrum};
pub use stellar::{BlackbodyStellarSpectrum, FlatStellarSpectrum};
pub use stis_zodiacal::STISZodiacalSpectrum;
pub use trapezoid::trap_integrate;
pub use zodiacal::ZodiacalLight;

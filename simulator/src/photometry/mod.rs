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
//! # Photometric Workflow Example
//!
//! ```rust
//! use simulator::photometry::{
//!     BlackbodyStellarSpectrum, b_filter, v_filter,
//!     temperature_to_spectral_class, spectrum_to_rgb_values, Spectrum
//! };
//! use simulator::units::{Area, AreaExt};
//! use std::time::Duration;
//!
//! // 1. Create stellar spectrum
//! let star_temp = 5778.0; // Solar temperature
//! let star = BlackbodyStellarSpectrum::new(star_temp, 1.0);
//!
//! // 2. Get photometric filters
//! let b_band = b_filter();
//! let v_band = v_filter();
//!
//! // 3. Calculate synthetic photometry
//! let exposure = Duration::from_secs(30);
//! let aperture = Area::from_square_centimeters(100.0); // 100 cm²
//! let b_flux = star.photo_electrons(&b_band, aperture, &exposure);
//! let v_flux = star.photo_electrons(&v_band, aperture, &exposure);
//!
//! // 4. Derive stellar properties
//! let bv_color = -2.5 * (b_flux / v_flux).log10();
//! let spectral_class = temperature_to_spectral_class(star_temp);
//! let (r, g, b) = spectrum_to_rgb_values(&star);
//!
//! println!("Star analysis:");
//! println!("  Temperature: {:.0} K", star_temp);
//! println!("  Spectral class: {}", spectral_class);
//! println!("  B-V color: {:.3} mag", bv_color);
//! println!("  RGB color: ({:.2}, {:.2}, {:.2})", r, g, b);
//! ```
//!
//! # Color Calibration Example
//!
//! ```rust
//! use simulator::photometry::{
//!     HumanVision, BlackbodyStellarSpectrum, gaia::GAIA_PASSBAND, Spectrum
//! };
//! use simulator::units::{Area, AreaExt};
//! use std::time::Duration;
//!
//! // Create stellar spectra for different types
//! let stars = [
//!     ("Vega", 9602.0),     // A0V standard
//!     ("Sun", 5778.0),      // G2V solar
//!     ("Arcturus", 4286.0), // K1.5III giant
//! ];
//!
//! // Get human vision responses
//! let red_qe = HumanVision::red();
//! let green_qe = HumanVision::green_blue();
//! let blue_qe = HumanVision::blue();
//!
//! // Compare human vision vs Gaia photometry
//! for (name, temp) in stars.iter() {
//!     let spectrum = BlackbodyStellarSpectrum::new(*temp, 1.0);
//!     
//!     // Human RGB response
//!     let exposure = Duration::from_secs(1);
//!     let aperture = Area::from_square_centimeters(1.0);
//!     let r = spectrum.photo_electrons(&red_qe, aperture, &exposure);
//!     let g = spectrum.photo_electrons(&green_qe, aperture, &exposure);
//!     let b = spectrum.photo_electrons(&blue_qe, aperture, &exposure);
//!     
//!     // Gaia G-band response
//!     let gaia_g = spectrum.photo_electrons(&*GAIA_PASSBAND, aperture, &exposure);
//!     
//!     println!("{}: RGB({:.2}, {:.2}, {:.2}), Gaia G: {:.2e}",
//!              name, r, g, b, gaia_g);
//! }
//! ```
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

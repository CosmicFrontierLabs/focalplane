//! Fundamental spectral energy distribution modeling for astronomical photometry.
//!
//! This module provides the core infrastructure for representing and manipulating
//! stellar and background spectra in astronomical simulations. Includes wavelength
//! band definitions, physical constants, and the foundational Spectrum trait that
//! enables accurate synthetic photometry and detector response calculations.
//!
//! # Physical Framework
//!
//! All spectral calculations use rigorous physics in CGS units:
//! - **Spectral irradiance**: erg s⁻¹ cm⁻² Hz⁻¹ (power per unit area per unit frequency)
//! - **Irradiance**: erg s⁻¹ cm⁻² (total power per unit area)
//! - **Photon rates**: photons s⁻¹ cm⁻² (photon flux density)
//! - **Wavelengths**: nanometers (nm) for astronomical convenience
//!
//! # Core Components
//!
//! ## Spectrum Trait
//! Universal interface for stellar spectra, blackbody radiation,
//! and other spectral energy distributions.
//!
//! ## Wavelength Bands
//! Precise wavelength range definitions for photometric filters
//! and detector response calculations.
//!
//! ## Physical Constants
//! Accurate CGS physical constants for astronomical magnitude
//! calculations and photon energy conversions.
//!
//! # Synthetic Photometry Workflow
//! Complete pipeline for calculating detector responses from stellar
//! spectra through realistic telescope and sensor models.
//!
//! # Wavelength-Frequency Conversions
//!
//! The module handles conversions between wavelength and frequency domains:
//! - **λ = c/ν**: Wavelength from frequency
//! - **ν = c/λ**: Frequency from wavelength  
//! - **E = hν = hc/λ**: Photon energy calculations
//!
//! # Numerical Integration
//!
//! Spectral integration uses adaptive methods:
//! - **Band decomposition**: 1nm sub-bands for accurate integration
//! - **Trapezoidal rule**: Reliable numerical integration
//! - **Edge handling**: Proper treatment of band boundaries
//! - **Photon counting**: Accurate conversion from power to photon rates

use std::time::Duration;

use thiserror::Error;

use super::QuantumEfficiency;

/// Physical constants in CGS units for astronomical calculations.
///
/// Provides fundamental constants required for accurate spectral photometry,
/// magnitude systems, and photon energy calculations. All values use CGS
/// (centimeter-gram-second) units for consistency with astronomical literature.
///
/// # Usage
/// Physical constants for astronomical photon energy calculations
/// and AB magnitude zero-point reference values.
pub struct CGS {}

impl CGS {
    /// AB magnitude system zero-point flux density
    /// Units: 3631e-23 erg s⁻¹ cm⁻² Hz⁻¹
    pub const AB_ZERO_POINT_FLUX_DENSITY: f64 = 3631e-23;

    /// 1 Jansky in CGS units
    /// Units: 1e-23 erg s⁻¹ cm⁻² Hz⁻¹
    pub const JANSKY_IN_CGS: f64 = 1e-23;

    /// Planck's constant
    /// Units: 6.62607015e-27 erg⋅s (erg-seconds in CGS)
    pub const PLANCK_CONSTANT: f64 = 6.62607015e-27;

    /// Speed of light in vacuum
    /// Units: 2.99792458e10 cm/s (centimeters per second in CGS)
    pub const SPEED_OF_LIGHT: f64 = 2.99792458e10;
}

/// Errors that can occur with spectrum operations
#[derive(Debug, Error)]
pub enum SpectrumError {
    #[error("Invalid frequency: {0}")]
    InvalidFrequency(String),
}

/// Wavelength range specification for astronomical filters and detectors.
///
/// Represents a contiguous wavelength interval with well-defined lower and upper
/// bounds. Used throughout the photometry system to define detector sensitivity
/// ranges, filter passbands, and integration limits for synthetic photometry.
///
/// # Physical Constraints
/// - Wavelengths must be positive and finite
/// - Lower bound must be less than upper bound
/// - Wavelengths specified in nanometers for astronomical convenience
///
/// # Applications
/// - **Filter definitions**: Johnson UBV, Gaia G-band, custom narrow-band
/// - **Detector ranges**: CCD/CMOS sensitivity limits
/// - **Integration bounds**: Spectral photometry calculations
/// - **Bandwidth analysis**: Effective wavelength and filter width
///
/// # Usage
/// Create wavelength ranges for astronomical filters, detector sensitivity
/// ranges, and spectral integration bounds with precise nanometer definitions.
pub struct Band {
    /// Lower wavelength bound in nanometers
    pub lower_nm: f64,

    /// Upper wavelength bound in nanometers
    pub upper_nm: f64,
}

impl Band {
    /// Create a new Band directly from lower and upper bounds
    ///
    /// # Arguments
    ///
    /// * `lower_nm` - Lower wavelength bound in nanometers
    /// * `upper_nm` - Upper wavelength bound in nanometers
    ///
    /// # Returns
    ///
    /// A new Band with the specified wavelength bounds
    pub fn from_nm_bounds(lower_nm: f64, upper_nm: f64) -> Self {
        // These are programming errors, so we don't return Result
        // but panic if the range is invalid
        if !lower_nm.is_finite() || !upper_nm.is_finite() {
            panic!("Wavelength range cannot contain non-finite values");
        }

        if lower_nm > upper_nm {
            panic!(
                "Invalid wavelength range: start must be less than end, got {lower_nm}..{upper_nm}",
            );
        }
        if lower_nm < 0.0 || upper_nm < 0.0 {
            panic!("Wavelengths must be non-negative");
        }

        Self { lower_nm, upper_nm }
    }

    /// Create a new Band from frequency bounds in Hz
    ///
    /// # Arguments
    ///
    /// * `lower_freq_hz` - Lower frequency bound in Hz
    /// * `upper_freq_hz` - Upper frequency bound in Hz
    ///
    /// # Returns
    ///
    /// A Result containing the new Band or an error if the frequencies are invalid
    pub fn from_freq_bounds(lower_freq_hz: f64, upper_freq_hz: f64) -> Self {
        // Convert frequency bounds to wavelength bounds
        if lower_freq_hz <= 0.0 || upper_freq_hz <= 0.0 {
            panic!("Frequencies must be positive, got {lower_freq_hz}..{upper_freq_hz}");
        }

        // Wavelength = speed of light / frequency
        let lower_nm = CGS::SPEED_OF_LIGHT / (upper_freq_hz * 1e-7); // Convert Hz to nm
        let upper_nm = CGS::SPEED_OF_LIGHT / (lower_freq_hz * 1e-7); // Convert Hz to nm

        Self::from_nm_bounds(lower_nm, upper_nm)
    }

    /// Create a new Band centered on a given wavelength with a specified frequency spread
    ///
    /// # Arguments
    ///
    /// * `wavelength_nm` - Wavelength in nanometers to center the band on
    /// * `frequency_spread` - Frequency spread in Hz to define the width of the band
    ///
    /// # Returns
    ///
    /// A new Band centered on the specified wavelength with the given frequency spread
    pub fn centered_on(wavelength_nm: f64, frequency_spread: f64) -> Self {
        // Create a band centered on a wavelength with a given width
        if wavelength_nm <= 0.0 {
            panic!("Wavelength must be positive, got: {wavelength_nm}");
        }
        if frequency_spread <= 0.0 {
            panic!("Frequency spread must be positive, got: {frequency_spread}");
        }
        // Convert wavelength to frequency
        let center_freq = CGS::SPEED_OF_LIGHT / (wavelength_nm * 1e-7); // Convert nm to Hz
                                                                        // Calculate lower and upper frequency bounds
        let lower_freq = center_freq - frequency_spread / 2.0;
        let upper_freq = center_freq + frequency_spread / 2.0;
        // Convert back to wavelength bounds
        let lower_nm = CGS::SPEED_OF_LIGHT / (upper_freq * 1e-7); // Convert Hz to nm
        let upper_nm = CGS::SPEED_OF_LIGHT / (lower_freq * 1e-7); // Convert Hz to nm
        Self::from_nm_bounds(lower_nm, upper_nm)
    }

    /// Get the width of the band in nanometers
    ///
    /// # Returns
    ///
    /// The width of the band (upper_nm - lower_nm)
    pub fn width(&self) -> f64 {
        self.upper_nm - self.lower_nm
    }

    /// Return the center of a band in nanometers
    ///
    /// # Returns
    ///
    /// The center wavelength of the band in nanometers
    pub fn center(&self) -> f64 {
        (self.lower_nm + self.upper_nm) / 2.0
    }

    /// Get the frequency bounds of the band in Hz
    ///
    /// # Returns
    ///
    /// A tuple containing the lower and upper frequency bounds of the band in Hz
    pub fn frequency_bounds(&self) -> (f64, f64) {
        // Frequency = speed of light / wavelength
        // Wavelength in nanometers, speed of light in cm/s
        let lower_freq = CGS::SPEED_OF_LIGHT / (self.upper_nm * 1e-7);
        let upper_freq = CGS::SPEED_OF_LIGHT / (self.lower_nm * 1e-7);
        (lower_freq, upper_freq)
    }

    /// Divide this band into n equally-sized sub-bands
    ///
    /// Creates n sub-bands that completely cover the wavelength range
    /// of this band, with each sub-band having equal width in nanometers.
    ///
    /// # Arguments
    /// * `n` - Number of sub-bands to create (must be >= 1)
    ///
    /// # Returns
    /// Vec of n Band objects covering the original wavelength range
    ///
    /// # Panics
    /// Panics if n is 0
    pub fn as_n_subbands(&self, n: usize) -> Vec<Band> {
        if n == 0 {
            panic!("Number of sub-bands must be at least 1");
        }

        let mut subbands = Vec::with_capacity(n);
        let width = self.width() / n as f64;

        for i in 0..n {
            let lower = self.lower_nm + i as f64 * width;
            let upper = if i == n - 1 {
                // For the last sub-band, use exact upper bound to avoid floating point errors
                self.upper_nm
            } else {
                self.lower_nm + (i + 1) as f64 * width
            };
            subbands.push(Band::from_nm_bounds(lower, upper));
        }

        subbands
    }

    /// Create sub-bands with 1nm width covering the full band
    ///
    /// This is a convenience method that creates sub-bands of 1nm width,
    /// with the number of sub-bands equal to the ceiling of the band width.
    /// Common pattern for spectral integration.
    ///
    /// # Returns
    /// Vec of Band objects with ~1nm width each
    pub fn sub_nm_bands(&self) -> Vec<Band> {
        self.as_n_subbands(self.width().ceil() as usize)
    }
}
pub fn wavelength_to_ergs(wavelength_nm: f64) -> f64 {
    // Convert wavelength in nanometers to energy in erg
    // E = h * c / λ, where λ is in cm
    if wavelength_nm <= 0.0 {
        panic!("WARNING!!! Wavelength must be positive, got: {wavelength_nm}");
    }
    let wavelength_cm = wavelength_nm * 1e-7; // Convert nm to cm
    CGS::PLANCK_CONSTANT * CGS::SPEED_OF_LIGHT / wavelength_cm
}

/// Universal interface for astronomical spectral energy distributions.
///
/// Provides the fundamental methods required for synthetic photometry, detector
/// response calculations, and spectral analysis. All implementations must support
/// wavelength-dependent spectral irradiance evaluation and band integration.
///
/// # Physical Units (CGS)
/// - **Wavelengths**: nanometers (nm)
/// - **Spectral irradiance**: erg s⁻¹ cm⁻² Hz⁻¹ (power per area per frequency)
/// - **Irradiance**: erg s⁻¹ cm⁻² (integrated power per area)
/// - **Photon rates**: photons s⁻¹ cm⁻² (number flux density)
///
/// # Implementation Requirements
/// - **Thread safety**: Must be Send + Sync for parallel processing
/// - **Wavelength domain**: Support 300-1100 nm range minimum
/// - **Physical accuracy**: Realistic spectral energy distributions
/// - **Numerical stability**: Robust handling of integration and edge cases
///
/// # Standard Implementations
/// - **BlackbodyStellarSpectrum**: Planck function stellar spectra
/// - **FlatStellarSpectrum**: Uniform flux density (calibration)
/// - **STISZodiacalSpectrum**: Zodiacal light background model
///
/// # Usage
/// Universal interface for astronomical spectral energy distributions with
/// wavelength-dependent irradiance evaluation and photometric integration.
pub trait Spectrum: Send + Sync {
    /// Evaluate spectral irradiance at specific wavelength.
    ///
    /// Returns the power per unit area per unit frequency at the specified
    /// wavelength. This is the fundamental spectral quantity from which all
    /// other photometric measurements are derived.
    ///
    /// # Physical Interpretation
    /// The spectral irradiance F_ν represents the energy flux density per unit
    /// frequency interval, related to the more common F_λ by:
    /// F_ν = F_λ × λ²/c
    ///
    /// # Arguments
    /// * `wavelength_nm` - Wavelength in nanometers [300, 1100] typical range
    ///
    /// # Returns
    /// Spectral irradiance in erg s⁻¹ cm⁻² Hz⁻¹, or 0.0 outside spectrum range
    ///
    /// # Usage
    /// Evaluate power per unit area per unit frequency at specific wavelengths
    /// for fundamental spectral quantity calculations and photometric analysis.
    fn spectral_irradiance(&self, wavelength_nm: f64) -> f64;

    /// Calculate the integrated power within a wavelength range and aperture
    ///
    /// # Arguments
    ///
    /// * `band` - The wavelength band to integrate over
    ///
    /// # Returns
    ///
    /// The the irradiance in erg s⁻¹ cm⁻²
    fn irradiance(&self, band: &Band) -> f64;

    /// Calculate the number of photons within a wavelength range
    ///
    /// # Arguments
    ///
    /// * `band` - The wavelength band to integrate over
    /// * `aperture_cm2` - Collection aperture area in square centimeters
    /// * `duration` - Duration of the observation
    ///
    /// # Returns
    ///
    /// The number of photons detected in the specified band
    fn photons(&self, band: &Band, aperture_cm2: f64, duration: Duration) -> f64 {
        crate::photometry::photoconversion::photons(self, band, aperture_cm2, duration)
    }

    /// Calculate the photo-electrons obtained from this spectrum when using a sensor with a given quantum efficiency
    ///
    /// # Arguments
    /// * `qe` - The quantum efficiency of the sensor as a function of wavelength
    /// * `aperture_cm2` - Collection aperture area in square centimeters
    /// * `duration` - Duration of the observation
    ///
    /// # Returns
    ///
    /// The number of electrons detected in the specified band
    fn photo_electrons(
        &self,
        qe: &QuantumEfficiency,
        aperture_cm2: f64,
        duration: &Duration,
    ) -> f64 {
        crate::photometry::photoconversion::photo_electrons(self, qe, aperture_cm2, duration)
    }
}

/// Simple flat spectrum for testing purposes.
///
/// Provides constant spectral irradiance across all wavelengths.
/// Useful for unit tests and debugging spectrum-related functionality.
pub struct FlatSpectrum {
    /// Spectral irradiance value in erg s⁻¹ cm⁻² Hz⁻¹
    pub irradiance_value: f64,
}

impl FlatSpectrum {
    /// Create a new flat spectrum with specified irradiance
    pub fn new(irradiance_value: f64) -> Self {
        Self { irradiance_value }
    }

    /// Create a unit flat spectrum with irradiance = 1.0
    pub fn unit() -> Self {
        Self::new(CGS::JANSKY_IN_CGS) // 1 Jansky in CGS units
    }
}

impl Spectrum for FlatSpectrum {
    fn spectral_irradiance(&self, _wavelength: f64) -> f64 {
        self.irradiance_value
    }

    fn irradiance(&self, band: &Band) -> f64 {
        // For a flat spectrum in frequency space, integrate over the band
        let (lower_freq, upper_freq) = band.frequency_bounds();
        self.irradiance_value * (upper_freq - lower_freq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_band_frequency_bounds() {
        // Test basic conversion
        let band = Band::from_nm_bounds(400.0, 700.0);
        let (lower_freq, upper_freq) = band.frequency_bounds();

        // Manual calculation of expected values
        let expected_lower = CGS::SPEED_OF_LIGHT / (700.0 * 1e-7);
        let expected_upper = CGS::SPEED_OF_LIGHT / (400.0 * 1e-7);

        assert_relative_eq!(lower_freq, expected_lower, epsilon = 1e-10);
        assert_relative_eq!(upper_freq, expected_upper, epsilon = 1e-10);

        // Verify that lower_freq < upper_freq (because of wavelength inversion)
        assert!(lower_freq < upper_freq);

        // Test at different wavelength range
        let band2 = Band::from_nm_bounds(100.0, 200.0);
        let (lower_freq2, upper_freq2) = band2.frequency_bounds();

        // Expected values for band2
        let expected_lower2 = CGS::SPEED_OF_LIGHT / (200.0 * 1e-7);
        let expected_upper2 = CGS::SPEED_OF_LIGHT / (100.0 * 1e-7);

        assert_relative_eq!(lower_freq2, expected_lower2, epsilon = 1e-10);
        assert_relative_eq!(upper_freq2, expected_upper2, epsilon = 1e-10);
    }

    #[test]
    fn test_band_as_n_subbands() {
        // Test basic subdivision
        let band = Band::from_nm_bounds(400.0, 700.0);
        let subbands = band.as_n_subbands(3);

        assert_eq!(subbands.len(), 3);

        // Check that sub-bands cover the full range
        assert_eq!(subbands[0].lower_nm, 400.0);
        assert_eq!(subbands[2].upper_nm, 700.0);

        // Check sub-band widths (should be 100nm each)
        assert_relative_eq!(subbands[0].width(), 100.0, epsilon = 1e-10);
        assert_relative_eq!(subbands[1].width(), 100.0, epsilon = 1e-10);
        assert_relative_eq!(subbands[2].width(), 100.0, epsilon = 1e-10);

        // Check boundaries connect
        assert_eq!(subbands[0].upper_nm, subbands[1].lower_nm);
        assert_eq!(subbands[1].upper_nm, subbands[2].lower_nm);
    }

    #[test]
    fn test_band_as_n_subbands_single() {
        // Test n=1 case
        let band = Band::from_nm_bounds(500.0, 600.0);
        let subbands = band.as_n_subbands(1);

        assert_eq!(subbands.len(), 1);
        assert_eq!(subbands[0].lower_nm, 500.0);
        assert_eq!(subbands[0].upper_nm, 600.0);
        assert_eq!(subbands[0].width(), band.width());
    }

    #[test]
    fn test_band_as_n_subbands_many() {
        // Test with many sub-bands
        let band = Band::from_nm_bounds(400.0, 500.0);
        let subbands = band.as_n_subbands(10);

        assert_eq!(subbands.len(), 10);

        // Each should be 10nm wide
        for subband in &subbands {
            assert_relative_eq!(subband.width(), 10.0, epsilon = 1e-10);
        }

        // Check coverage
        assert_eq!(subbands.first().unwrap().lower_nm, 400.0);
        assert_eq!(subbands.last().unwrap().upper_nm, 500.0);

        // Check no gaps
        for i in 0..9 {
            assert_eq!(subbands[i].upper_nm, subbands[i + 1].lower_nm);
        }
    }

    #[test]
    fn test_band_as_n_subbands_odd_division() {
        // Test when width doesn't divide evenly
        let band = Band::from_nm_bounds(400.0, 403.0);
        let subbands = band.as_n_subbands(2);

        assert_eq!(subbands.len(), 2);
        assert_eq!(subbands[0].lower_nm, 400.0);
        assert_eq!(subbands[1].upper_nm, 403.0);

        // First should be 1.5nm, second should cover the rest
        assert_relative_eq!(subbands[0].width(), 1.5, epsilon = 1e-10);
        assert_relative_eq!(subbands[1].width(), 1.5, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Number of sub-bands must be at least 1")]
    fn test_band_as_n_subbands_zero() {
        let band = Band::from_nm_bounds(400.0, 700.0);
        let _ = band.as_n_subbands(0);
    }

    #[test]
    fn test_flat_spectrum_creation() {
        let spectrum = FlatSpectrum::new(1e-20);
        assert_eq!(spectrum.irradiance_value, 1e-20);

        let unit_spectrum = FlatSpectrum::unit();
        assert_eq!(unit_spectrum.irradiance_value, CGS::JANSKY_IN_CGS);
    }

    #[test]
    fn test_flat_spectrum_spectral_irradiance() {
        let spectrum = FlatSpectrum::new(5e-23);
        assert_eq!(spectrum.spectral_irradiance(500.0), 5e-23);
        assert_eq!(spectrum.spectral_irradiance(1000.0), 5e-23);
    }
}

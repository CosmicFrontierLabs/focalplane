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
//! Universal interface for all spectral energy distributions:
//! ```rust
//! use simulator::photometry::{Spectrum, Band};
//! use std::time::Duration;
//!
//! # struct ExampleSpectrum;
//! # impl Spectrum for ExampleSpectrum {
//! #     fn spectral_irradiance(&self, _: f64) -> f64 { 1e-10 }
//! #     fn irradiance(&self, _: &Band) -> f64 { 1e-8 }
//! # }
//! let spectrum = ExampleSpectrum;
//! let band = Band::from_nm_bounds(400.0, 700.0);
//!
//! // Get spectral irradiance at specific wavelength
//! let flux_550nm = spectrum.spectral_irradiance(550.0);
//!
//! // Integrate total power in band
//! let total_power = spectrum.irradiance(&band);
//!
//! // Count photons for realistic detector
//! let aperture = 100.0; // cm²
//! let exposure = Duration::from_secs(30);
//! let photon_count = spectrum.photons(&band, aperture, exposure);
//! ```
//!
//! ## Wavelength Bands
//! Precise wavelength range definitions for filters and detectors:
//! ```rust
//! use simulator::photometry::Band;
//!
//! // Johnson V-band approximation
//! let v_band = Band::from_nm_bounds(500.0, 600.0);
//!
//! // Narrow-band H-alpha filter
//! let ha_band = Band::centered_on(656.3, 2e12); // 656.3nm ± 1GHz
//!
//! // Check band properties
//! println!("V-band: {:.0}-{:.0} nm (width: {:.0} nm)",
//!          v_band.lower_nm, v_band.upper_nm, v_band.width());
//! println!("H-alpha center: {:.1} nm", ha_band.center());
//! ```
//!
//! ## Physical Constants
//! Accurate CGS physical constants for astronomical calculations:
//! ```rust
//! use simulator::photometry::spectrum::CGS;
//!
//! // AB magnitude zero point
//! println!("AB zero-point: {:.2e} erg/s/cm²/Hz", CGS::AB_ZERO_POINT_FLUX_DENSITY);
//!
//! // Fundamental constants
//! println!("Planck constant: {:.3e} erg⋅s", CGS::PLANCK_CONSTANT);
//! println!("Speed of light: {:.5e} cm/s", CGS::SPEED_OF_LIGHT);
//! ```
//!
//! # Synthetic Photometry Workflow
//!
//! ```rust
//! use simulator::photometry::{Spectrum, Band, QuantumEfficiency};
//! use std::time::Duration;
//!
//! # struct SolarSpectrum;
//! # impl Spectrum for SolarSpectrum {
//! #     fn spectral_irradiance(&self, wavelength: f64) -> f64 {
//! #         // Simplified solar spectrum
//! #         if (400.0..=700.0).contains(&wavelength) { 1e-10 } else { 0.0 }
//! #     }
//! #     fn irradiance(&self, band: &Band) -> f64 { 1e-8 * band.width() }
//! # }
//! // 1. Define stellar spectrum
//! let sun = SolarSpectrum;
//!
//! // 2. Define detector response
//! let wavelengths = vec![400.0, 500.0, 600.0, 700.0];
//! let efficiencies = vec![0.0, 0.8, 0.9, 0.0];
//! let detector = QuantumEfficiency::from_table(wavelengths, efficiencies).unwrap();
//!
//! // 3. Calculate synthetic photometry
//! let aperture = 50.0; // cm² telescope
//! let exposure = Duration::from_secs(60);
//! let photoelectrons = sun.photo_electrons(&detector, aperture, &exposure);
//!
//! println!("Detected {:.0} photoelectrons in 60s", photoelectrons);
//! ```
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
/// ```rust
/// use simulator::photometry::spectrum::CGS;
///
/// // Calculate photon energy at 550nm
/// let wavelength_cm = 550e-7; // Convert nm to cm
/// let photon_energy = CGS::PLANCK_CONSTANT * CGS::SPEED_OF_LIGHT / wavelength_cm;
/// println!("550nm photon energy: {:.2e} erg", photon_energy);
///
/// // AB magnitude zero-point reference
/// let ab_zp = CGS::AB_ZERO_POINT_FLUX_DENSITY;
/// println!("AB zero-point: {:.0e} erg/s/cm²/Hz", ab_zp);
/// ```
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
/// # Examples
/// ```rust
/// use simulator::photometry::Band;
///
/// // Standard Johnson V-band (approximate)
/// let v_band = Band::from_nm_bounds(500.0, 600.0);
/// assert_eq!(v_band.width(), 100.0);
/// assert_eq!(v_band.center(), 550.0);
///
/// // Narrow H-alpha filter
/// let ha_filter = Band::from_nm_bounds(655.0, 657.0);
/// assert_eq!(ha_filter.width(), 2.0);
///
/// // Wide visible spectrum
/// let visible = Band::from_nm_bounds(380.0, 750.0);
/// assert_eq!(visible.width(), 370.0);
/// ```
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
                "Invalid wavelength range: start must be less than end, got {}..{}",
                lower_nm, upper_nm,
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
            panic!(
                "Frequencies must be positive, got {}..{}",
                lower_freq_hz, upper_freq_hz
            );
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
            panic!("Wavelength must be positive, got: {}", wavelength_nm);
        }
        if frequency_spread <= 0.0 {
            panic!(
                "Frequency spread must be positive, got: {}",
                frequency_spread
            );
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
}

pub fn nm_sub_bands(band: &Band) -> Vec<Band> {
    // Decompose the band into integer nanometer bands
    // Special case the first and last bands
    let mut bands: Vec<Band> = Vec::new();

    let first_int_nm = band.lower_nm.ceil() as u32;
    let last_int_nm = band.upper_nm.floor() as u32;

    if band.lower_nm != first_int_nm as f64 {
        bands.push(Band::from_nm_bounds(band.lower_nm, first_int_nm as f64));
    }
    bands.extend(
        (first_int_nm..last_int_nm).map(|nm| Band::from_nm_bounds(nm as f64, nm as f64 + 1.0)),
    );

    if last_int_nm as f64 != band.upper_nm {
        bands.push(Band::from_nm_bounds(last_int_nm as f64, band.upper_nm));
    }

    bands
}

pub fn wavelength_to_ergs(wavelength_nm: f64) -> f64 {
    // Convert wavelength in nanometers to energy in erg
    // E = h * c / λ, where λ is in cm
    if wavelength_nm <= 0.0 {
        panic!(
            "WARNING!!! Wavelength must be positive, got: {}",
            wavelength_nm
        );
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
/// # Typical Usage Pattern
/// ```rust
/// use simulator::photometry::{Spectrum, Band, QuantumEfficiency};
/// use std::time::Duration;
///
/// # struct ExampleSpectrum;
/// # impl Spectrum for ExampleSpectrum {
/// #     fn spectral_irradiance(&self, wavelength: f64) -> f64 {
/// #         if (400.0..=700.0).contains(&wavelength) { 1e-10 } else { 0.0 }
/// #     }
/// #     fn irradiance(&self, band: &Band) -> f64 { 1e-8 * band.width() }
/// # }
/// fn analyze_spectrum<S: Spectrum>(spectrum: &S) {
///     // 1. Check spectral irradiance at key wavelengths
///     let blue_flux = spectrum.spectral_irradiance(450.0);  // B-band
///     let visual_flux = spectrum.spectral_irradiance(550.0); // V-band
///     let red_flux = spectrum.spectral_irradiance(650.0);   // R-band
///     
///     // 2. Calculate color indices
///     let bv_ratio = blue_flux / visual_flux;
///     
///     // 3. Integrate total power in visible spectrum
///     let visible_band = Band::from_nm_bounds(400.0, 700.0);
///     let total_power = spectrum.irradiance(&visible_band);
///     
///     // 4. Count photons for 1-meter telescope, 1-second exposure
///     let aperture = 10000.0; // cm² (1m diameter)
///     let exposure = Duration::from_secs(1);
///     let photon_count = spectrum.photons(&visible_band, aperture, exposure);
/// }
/// ```
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
    /// # Examples
    /// ```rust
    /// use simulator::photometry::Spectrum;
    ///
    /// # struct SolarSpectrum;
    /// # impl Spectrum for SolarSpectrum {
    /// #     fn spectral_irradiance(&self, wavelength: f64) -> f64 {
    /// #         // Solar-like spectrum with peak in green
    /// #         if (400.0..=700.0).contains(&wavelength) {
    /// #             match wavelength as i32 {
    /// #                 400..=500 => 0.8e-10,  // Blue
    /// #                 500..=600 => 1.2e-10,  // Green (peak)
    /// #                 600..=700 => 0.9e-10,  // Red
    /// #                 _ => 1e-10
    /// #             }
    /// #         } else { 0.0 }
    /// #     }
    /// #     fn irradiance(&self, band: &simulator::photometry::Band) -> f64 { 1e-8 }
    /// # }
    /// let sun = SolarSpectrum;
    ///
    /// // Sample key wavelengths
    /// let uv_flux = sun.spectral_irradiance(350.0);   // Near-UV
    /// let blue_flux = sun.spectral_irradiance(450.0);  // Blue
    /// let green_flux = sun.spectral_irradiance(550.0); // Green (peak)
    /// let red_flux = sun.spectral_irradiance(650.0);   // Red
    /// let ir_flux = sun.spectral_irradiance(850.0);    // Near-IR
    ///
    /// // Verify spectrum ranges
    /// assert!(green_flux > 0.0);       // In range
    /// assert!(blue_flux > 0.0);        // In range
    /// assert_eq!(uv_flux, 0.0);        // Outside spectrum range
    /// ```
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
        // Convert power to photons per second
        // E = h * c / λ, so N = P / (h * c / λ)
        // where P is power in erg/s, h is Planck's constant, c is speed of light, and λ is wavelength in cm
        let mut total_photons = 0.0;

        // Decompose the band into integer nanometer bands
        // Special case the first and last bands
        let bands = nm_sub_bands(band);

        // Integrate over each wavelength in the band
        for band in bands {
            let energy_per_photon = wavelength_to_ergs(band.center());
            let irradiance = self.irradiance(&band);
            total_photons += irradiance / energy_per_photon;
        }

        // Multiply by duration to get total photons detected
        total_photons * duration.as_secs_f64() * aperture_cm2
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
        // Convert power to photons per second
        // E = h * c / λ, so N = P / (h * c / λ)
        // where P is power in erg/s, h is Planck's constant, c is speed of light, and λ is wavelength in cm
        let mut total_electrons = 0.0;

        // Decompose the band into integer nanometer bands
        // Special case the first and last bands
        let bands = nm_sub_bands(&qe.band());

        // Integrate over each wavelength in the band
        for band in bands {
            let energy_per_photon = wavelength_to_ergs(band.center());
            let photons_in_band = self.irradiance(&band) / energy_per_photon;
            total_electrons += qe.at(band.center()) * photons_in_band;
        }

        // Multiply by duration to get total photons detected
        total_electrons * duration.as_secs_f64() * aperture_cm2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nm_sub_bands() {
        let lowest = 400.0;
        let highest = 700.0;

        let band = Band::from_nm_bounds(lowest, highest);
        let sub_bands = nm_sub_bands(&band);

        // Should create bands for each integer nm in the range
        assert_eq!(sub_bands.len(), 300); // 700 - 400

        // Check first and last bands
        assert_eq!(sub_bands[0].lower_nm, lowest);
        assert_eq!(sub_bands[0].upper_nm, lowest + 1.0);
        assert_eq!(sub_bands.last().unwrap().lower_nm, highest - 1.0);
        assert_eq!(sub_bands.last().unwrap().upper_nm, highest);

        // Check middle band
        assert_eq!(sub_bands[150].lower_nm, 550.0);
        assert_eq!(sub_bands[150].upper_nm, 551.0);

        for b in &sub_bands {
            assert!(
                b.lower_nm < b.upper_nm,
                "Band {}..{} is invalid",
                b.lower_nm,
                b.upper_nm
            );
            assert!(b.lower_nm >= lowest, "OOB {}", b.lower_nm);
            assert!(b.upper_nm <= highest, "OOB {}", b.upper_nm);
        }
    }

    #[test]
    fn test_nm_sub_bands_tiny() {
        let lowest = 10.0;
        let highest = 11.0;

        let band = Band::from_nm_bounds(lowest, highest);
        let sub_bands = nm_sub_bands(&band);

        assert_eq!(sub_bands.len(), 1); // 11 - 10

        // Check the single band created
        assert_eq!(sub_bands[0].lower_nm, lowest);
        assert_eq!(sub_bands[0].upper_nm, highest);
    }

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
}

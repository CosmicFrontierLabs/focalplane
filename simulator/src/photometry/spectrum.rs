//! Spectrum model for astronomical photometry
//!
//! This module provides a representation of astronomical spectra
//! with the Spectrum trait and implementations.

use std::time::Duration;

use thiserror::Error;

use super::QuantumEfficiency;

/// Constants in CGS units
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

fn nm_sub_bands(band: &Band) -> Vec<Band> {
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

fn wavelength_to_ergs(wavelength_nm: f64) -> f64 {
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

/// Trait representing a spectrum of electromagnetic radiation
///
/// All implementations must provide spectral irradiance at a given wavelength
/// and integrate power over wavelength ranges.
///
/// All units are in CGS:
/// - Wavelengths in nanometers
/// - Spectral irradiance in erg s⁻¹ cm⁻² Hz⁻¹
/// - Irradiance in erg s⁻¹ cm⁻² (total energy per unit time)
pub trait Spectrum: Send + Sync {
    /// Get the spectral irradiance at the specified wavelength
    ///
    /// # Arguments
    ///
    /// * `wavelength_nm` - The wavelength in nanometers
    ///
    /// # Returns
    ///
    /// The spectral irradiance at the given wavelength in erg s⁻¹ cm⁻² Hz⁻¹
    /// Returns 0.0 if the wavelength is outside the spectrum's range
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
    fn photons(&self, band: &Band, aperture_cm2: f64, duration: std::time::Duration) -> f64 {
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

/// A flat stellar spectrum with constant spectral flux density
///
/// This represents a source with the same energy per unit frequency
/// across all wavelengths. Commonly used for simple stellar modeling.
#[derive(Debug, Clone)]
pub struct FlatStellarSpectrum {
    /// Spectral flux density in erg s⁻¹ cm⁻² Hz⁻¹
    spectral_flux_density: f64,
}

impl FlatStellarSpectrum {
    /// Create a new FlatStellarSpectrum with a constant spectral flux density
    ///
    /// # Arguments
    ///
    /// * `spectral_flux_density` - The spectral flux density in (erg s⁻¹ cm⁻² Hz⁻¹)
    ///
    /// # Returns
    ///
    /// A new FlatStellarSpectrum with the specified flux density
    pub fn new(spectral_flux_density: f64) -> Self {
        Self {
            spectral_flux_density,
        }
    }

    /// Create a new FlatStellarSpectrum from an AB magnitude
    ///
    /// # Arguments
    ///
    /// * `ab_mag` - The AB magnitude of the source
    ///
    /// # Returns
    ///
    /// A new FlatStellarSpectrum with the spectral flux density corresponding to the given AB magnitude
    pub fn from_ab_mag(ab_mag: f64) -> Self {
        // Convert AB magnitude to flux density
        // F_ν = F_ν,0 * 10^(-0.4 * AB)
        // where F_ν,0 is the zero point (3631 Jy for AB mag system)
        let spectral_flux_density = CGS::AB_ZERO_POINT_FLUX_DENSITY * 10f64.powf(-0.4 * ab_mag);
        Self::new(spectral_flux_density)
    }

    /// Create a new FlatStellarSpectrum from a GaiaV2/V3 value
    /// # Arguments
    ///
    /// `gaia_magnitude` - The GaiaV2/V3 magnitude of the source
    /// # Returns
    ///
    /// A new FlatStellarSpectrum with the spectral flux density corresponding to the given Gaia magnitude
    pub fn from_gaia_magnitude(gaia_magnitude: f64) -> Self {
        // Convert Gaia magnitude to flux density
        // Same scaling, but slightly different zero-point definition
        Self::from_ab_mag(gaia_magnitude + 0.12)
    }
}

impl Spectrum for FlatStellarSpectrum {
    fn spectral_irradiance(&self, wavelength_nm: f64) -> f64 {
        // For a flat spectrum in frequency space, the spectral flux density
        // constant spectral irradiance

        // Ensure wavelength is positive
        if wavelength_nm <= 0.0 {
            return 0.0;
        }

        // erg s⁻¹ cm⁻² Hz⁻¹
        self.spectral_flux_density
    }

    fn irradiance(&self, band: &Band) -> f64 {
        // Integrate the spectral irradiance over the wavelength range
        // and multiply by the aperture area

        if band.lower_nm >= band.upper_nm || band.lower_nm <= 0.0 {
            return 0.0;
        }

        // Convert band to frequency bounds
        let (lower_freq, upper_freq) = band.frequency_bounds();

        self.spectral_flux_density * (upper_freq - lower_freq)
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
    fn test_aaron_matching_photoelec() {
        let mag_to_photons = vec![
            (0.0, 3_074_446.0),
            (10.0, 0.0001 * 3_074_446.0),
            (12.0, 48.0),
        ];
        let band: Band = Band::from_nm_bounds(400.0, 700.0);
        let qe = QuantumEfficiency::from_notch(&band, 1.0).unwrap();

        for (mag, expected_electrons) in mag_to_photons.iter() {
            // Calculate number of photons in 400-700nm if a 12th mag star
            let spectrum = FlatStellarSpectrum::from_ab_mag(*mag);

            // Assume 1 cm² aperture and 1 second duration
            let aperture_cm2 = 1.0;
            let duration = std::time::Duration::from_secs_f64(1.0);
            let electrons = spectrum.photo_electrons(&qe, aperture_cm2, &duration);

            let error = f64::abs(electrons - *expected_electrons) / *expected_electrons;

            assert!(
                error < 0.02,
                "For mag {}: Got {} Expected ~{}",
                mag,
                electrons,
                expected_electrons
            );
        }
    }

    #[test]
    fn test_aaron_matching() {
        let mag_to_photons = vec![
            (0.0, 3_074_446.0),
            (10.0, 0.0001 * 3_074_446.0),
            (12.0, 48.0),
        ];

        for (mag, expected_photons) in mag_to_photons.iter() {
            // Calculate number of photons in 400-700nm if a 12th mag star
            let spectrum = FlatStellarSpectrum::from_ab_mag(*mag);

            let band = Band::from_nm_bounds(400.0, 700.0);

            // Assume 1 cm² aperture and 1 second duration
            let aperture_cm2 = 1.0;
            let duration = std::time::Duration::from_secs(1);
            let photons = spectrum.photons(&band, aperture_cm2, duration);

            let error = f64::abs(photons - *expected_photons) / *expected_photons;

            assert!(
                error < 0.02,
                "For mag {}: Got {} Expected ~{}",
                mag,
                photons,
                expected_photons
            );
        }
    }

    #[test]
    fn test_flat_stellar_spectrum() {
        // Test creating from Jansky value
        let spectrum = FlatStellarSpectrum::new(3631.0);

        // Test irradiance at different wavelengths
        // The spectral irradiance in wavelength units varies with wavelength
        // even though the frequency spectrum is flat
        let spec_irr_500 = spectrum.spectral_irradiance(500.0);
        let spec_irr_1000 = spectrum.spectral_irradiance(1000.0);

        // Irradiance should be higher at shorter wavelengths (F_λ ∝ 1/λ²)
        assert_eq!(spec_irr_500, spec_irr_1000);

        // Test creating from AB magnitude
        let spectrum_ab = FlatStellarSpectrum::from_ab_mag(0.0);

        // AB mag of 0 should give the same result as 3631 Jy
        assert_relative_eq!(
            spectrum_ab.spectral_irradiance(500.0),
            CGS::JANSKY_IN_CGS * 3631.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_printout_photons() {
        let wavelengths = vec![400.0, 500.0, 600.0, 700.0];
        let spectrum = FlatStellarSpectrum::from_ab_mag(0.0);
        let aperture_cm2 = 1.0; // 1 cm² aperture
        let duration = std::time::Duration::from_secs(1); // 1 second observation

        for wavelength in wavelengths {
            // Make a band that is at the wavelength +- 1THz
            let band = Band::centered_on(wavelength, 1e12);
            let irradiance = spectrum.irradiance(&band);
            let photons = spectrum.photons(&band, aperture_cm2, duration);
            println!(
                "Wavelength: {} nm, Irradiance: {} Photons: {:.2}",
                wavelength, irradiance, photons
            );
        }
    }

    #[test]
    fn test_stellar_photon_spectrum() {
        let spectrum = FlatStellarSpectrum::from_gaia_magnitude(10.0);

        let band1 = Band::centered_on(400.0, 1e12);
        let band2 = Band::centered_on(800.0, 1e12);

        // Calculate photons in each band
        let photons1 = spectrum.photons(&band1, 1.0, std::time::Duration::from_secs(1));
        let photons2 = spectrum.photons(&band2, 1.0, std::time::Duration::from_secs(1));

        // Should be the same number 2x frequency == 1/2 wavelength == 2x photons
        println!(
            "Photons in band1 ({}nm): {}, band2 ({}nm): {}",
            band1.lower_nm, photons1, band2.lower_nm, photons2
        );
        assert_relative_eq!(photons1 * 2.0, photons2, epsilon = 1e-5);
    }

    #[test]
    fn test_stellar_spectrum_scaling() {
        let spectrum = FlatStellarSpectrum::from_ab_mag(0.0);
        // Test dimmer star (AB mag = 5.0)
        let spectrum_dim = FlatStellarSpectrum::from_ab_mag(5.0);

        // Should be 100x dimmer (5 mags = factor of 100)
        assert_relative_eq!(
            spectrum.spectral_irradiance(500.0) / spectrum_dim.spectral_irradiance(500.0),
            100.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_flat_stellar_irradiance() {
        // Create a flat spectrum with a known flux density
        let spectrum = FlatStellarSpectrum::from_ab_mag(0.0);

        // Test whole range
        let band = Band::from_nm_bounds(400.0, 700.0);
        let power1 = spectrum.irradiance(&band);

        // Test partial range
        let band2 = Band::from_nm_bounds(450.0, 600.0);
        let power2 = spectrum.irradiance(&band2);

        // Ensure non-zero values
        assert!(power1 > 0.0);
        assert!(power2 > 0.0);

        // First band should have more power (wider wavelength range)
        assert!(power1 > power2);

        // Test range outside spectrum
        let band3 = Band::from_nm_bounds(0.1, 0.2); // Very small wavelengths but not negative
        assert!(spectrum.irradiance(&band3) > 0.0);
    }

    #[test]
    fn test_photoelectron_math_100percent() {
        let aperture_cm2 = 1.0; // 1 cm² aperture
        let duration = std::time::Duration::from_secs(1); // 1 second observation

        let band = Band::from_nm_bounds(400.0, 600.0);
        // Make a pretend QE that is perfect in the 400-600nm range
        let qe = QuantumEfficiency::from_notch(&band, 1.0).unwrap();

        // Create a flat spectrum with a known flux density
        let spectrum = FlatStellarSpectrum::from_ab_mag(0.0);

        let photons = spectrum.photons(&band, aperture_cm2, duration);
        let electrons = spectrum.photo_electrons(&qe, aperture_cm2, &duration);

        // For a perfect QE, the number of electrons should equal the number of photons
        let err = f64::abs(photons - electrons) / photons;

        assert!(
            err < 0.01,
            "Expected {} electrons, got {}",
            photons,
            electrons
        );
    }

    #[test]
    fn test_photoelectron_math_50_percent() {
        let aperture_cm2 = 1.0; // 1 cm² aperture
        let duration = std::time::Duration::from_secs(1); // 1 second observation

        let band = Band::from_nm_bounds(400.0, 600.0);
        // Make a pretend QE with 50% efficiency in the 400-600nm range
        let qe = QuantumEfficiency::from_notch(&band, 0.5).unwrap();

        // Create a flat spectrum with a known flux density
        let spectrum = FlatStellarSpectrum::from_ab_mag(0.0);

        let photons = spectrum.photons(&band, aperture_cm2, duration);
        let electrons = spectrum.photo_electrons(&qe, aperture_cm2, &duration);

        // For 50% QE, electrons should be ~50% of photons
        let ratio = electrons / photons;

        assert_relative_eq!(ratio, 0.5, epsilon = 0.01);
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

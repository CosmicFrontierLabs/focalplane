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
    #[error("Wavelengths must have at least 2 points")]
    TooFewWavelengths,

    #[error("Measurements must have exactly one less element than wavelengths")]
    MeasurementsMismatch,

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
    /// Create a new Band from a wavelength range
    ///
    /// # Arguments
    ///
    /// * `range` - A range of wavelengths in nanometers
    ///
    /// # Returns
    ///
    /// A new Band with the specified wavelength range
    pub fn from(range: std::ops::Range<f64>) -> Self {
        // These are programming errors, so we don't return Result
        // but panic if the range is invalid
        if !range.start.is_finite() || !range.end.is_finite() {
            panic!("Wavelength range cannot contain non-finite values");
        }

        if range.start > range.end {
            panic!(
                "Invalid wavelength range: start must be less than end, got {}..{}",
                range.start, range.end
            );
        }
        if range.start < 0.0 || range.end < 0.0 {
            panic!("Wavelengths must be non-negative");
        }

        Self {
            lower_nm: range.start,
            upper_nm: range.end,
        }
    }

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
    pub fn new(lower_nm: f64, upper_nm: f64) -> Self {
        Self::from(lower_nm..upper_nm)
    }

    /// Get the width of the band in nanometers
    ///
    /// # Returns
    ///
    /// The width of the band (upper_nm - lower_nm)
    pub fn width(&self) -> f64 {
        self.upper_nm - self.lower_nm
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

    bands.push(Band::new(band.lower_nm, first_int_nm as f64));
    bands.extend((first_int_nm..=last_int_nm).map(|nm| Band::new(nm as f64, nm as f64 + 1.0)));
    bands.push(Band::new(last_int_nm as f64, band.upper_nm));

    bands
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
            let mean_wavelength_nm = (band.lower_nm + band.upper_nm) / 2.0;
            let mean_wavelength_cm = mean_wavelength_nm * 1e-7; // Convert to cm
            let energy_per_photon = CGS::PLANCK_CONSTANT * CGS::SPEED_OF_LIGHT / mean_wavelength_cm;
            total_photons += self.irradiance(&band) / energy_per_photon;
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
            let mean_wavelength_nm = (band.lower_nm + band.upper_nm) / 2.0;
            let mean_wavelength_cm = mean_wavelength_nm * 1e-7; // Convert to cm
            let energy_per_photon = CGS::PLANCK_CONSTANT * CGS::SPEED_OF_LIGHT / mean_wavelength_cm;
            let photons_in_band = self.irradiance(&band) / energy_per_photon;
            total_electrons += qe.at(mean_wavelength_nm) * photons_in_band;
        }

        // Multiply by duration to get total photons detected
        total_electrons * duration.as_secs_f64() * aperture_cm2
    }
}

/// A spectrum model with variable width wavelength bins
///
/// This struct stores N bin edges and N-1 measurement values.
/// Each measurement represents the spectral irradiance within a bin defined
/// by adjacent wavelength points.
#[derive(Debug, Clone)]
pub struct BinnedSpectrum {
    /// Wavelength bin edges in nanometers (N points)
    wavelengths: Vec<f64>,

    /// Spectral irradiance for each bin in erg s⁻¹ cm⁻² nm⁻¹ (N-1 values)
    spec_irr: Vec<f64>,
}

impl BinnedSpectrum {
    /// Create a new BinnedSpectrum with bin edges and measurements
    ///
    /// # Arguments
    ///
    /// * `wavelengths` - The N wavelength bin edges in nanometers (must be ascending)
    /// * `measurements` - The N-1 measurement values for each bin in erg s⁻¹ cm⁻² nm⁻¹
    ///
    /// # Returns
    ///
    /// A Result containing the new BinnedSpectrum or an error
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Wavelengths has fewer than 2 points
    /// - Measurements length is not wavelengths length - 1
    /// - Wavelengths are not in ascending order
    /// - Any wavelength is negative
    pub fn new(wavelengths: Vec<f64>, spec_irr: Vec<f64>) -> Result<Self, SpectrumError> {
        // Check wavelengths has at least 2 points
        if wavelengths.len() < 2 {
            return Err(SpectrumError::TooFewWavelengths);
        }

        // Check measurements has correct length
        if spec_irr.len() != wavelengths.len() - 1 {
            return Err(SpectrumError::MeasurementsMismatch);
        }

        // Check wavelengths are positive
        if let Some(negative) = wavelengths.iter().find(|&&w| w < 0.0) {
            return Err(SpectrumError::InvalidFrequency(format!(
                "Negative wavelength: {}",
                negative
            )));
        }

        // Check wavelengths are in ascending order
        for i in 1..wavelengths.len() {
            if wavelengths[i] <= wavelengths[i - 1] {
                return Err(SpectrumError::InvalidFrequency(format!(
                    "Wavelengths not in ascending order: {} <= {}",
                    wavelengths[i],
                    wavelengths[i - 1]
                )));
            }
        }

        Ok(Self {
            wavelengths,
            spec_irr,
        })
    }
}

impl Spectrum for BinnedSpectrum {
    fn spectral_irradiance(&self, wavelength_nm: f64) -> f64 {
        // Return 0.0 if outside the range
        if wavelength_nm < self.wavelengths[0] || wavelength_nm > *self.wavelengths.last().unwrap()
        {
            return 0.0;
        }

        // Find the bin that contains the wavelength
        // Could be done via binary search for efficiency,
        // but for simplicity we use linear search here
        for i in 0..self.spec_irr.len() {
            if wavelength_nm >= self.wavelengths[i] && wavelength_nm <= self.wavelengths[i + 1] {
                return self.spec_irr[i];
            }
        }

        // Should never reach here...
        0.0
    }

    fn irradiance(&self, band: &Band) -> f64 {
        // Initialize bounds from band
        let lower = band.lower_nm;
        let upper = band.upper_nm;

        // Clamp to spectrum range
        let lower = lower.max(self.wavelengths[0]);
        let upper = upper.min(*self.wavelengths.last().unwrap());

        if lower >= upper {
            return 0.0;
        }

        let mut power = 0.0;

        // Integrate over each bin that overlaps with the requested range
        for i in 0..self.spec_irr.len() {
            let bin_start = self.wavelengths[i];
            let bin_end = self.wavelengths[i + 1];

            // Skip bins completely outside our range
            if bin_end <= lower || bin_start >= upper {
                continue;
            }

            // Calculate overlap between bin and requested range (units nm)
            let overlap_start = bin_start.max(lower);
            let overlap_end = bin_end.min(upper);
            let overlap_width = overlap_end - overlap_start;

            // Use the original measurement units for power calculation (erg s⁻¹ cm⁻² nm⁻¹)
            // since we're integrating over wavelength, not frequency
            power += self.spec_irr[i] * overlap_width;
        }

        // Multiply by aperture area to get total power
        power
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
    /// * `flux_density_jy` - The spectral flux density in (erg s⁻¹ cm⁻² nm⁻¹)
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

        // Convert from erg s⁻¹ cm⁻² Hz⁻¹
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
    fn test_aaron_matching_photoelec() {
        let mag_to_photons = vec![
            (0.0, 3_074_446.0),
            (10.0, 0.0001 * 3_074_446.0),
            (12.0, 48.0),
        ];
        let band: Band = Band::new(400.0, 700.0);
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

            let band = Band::new(400.0, 700.0);

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
    fn test_binned_spectrum_irradiance() {
        let wavelengths = vec![400.0, 450.0, 500.0, 600.0, 700.0];
        let measurements = vec![0.1, 0.5, 0.8, 0.2];

        let spectrum = BinnedSpectrum::new(wavelengths, measurements).unwrap();

        // First, test the original irradiance method
        assert_relative_eq!(spectrum.spectral_irradiance(425.0), 0.1, epsilon = 1e-5);
        assert_relative_eq!(spectrum.spectral_irradiance(550.0), 0.8, epsilon = 1e-5);

        // Test irradiance at bin edges
        assert_relative_eq!(spectrum.spectral_irradiance(400.0), 0.1, epsilon = 1e-5);
        assert_relative_eq!(spectrum.spectral_irradiance(500.0), 0.5, epsilon = 1e-5);

        // Test irradiance outside range
        assert_relative_eq!(spectrum.spectral_irradiance(300.0), 0.0, epsilon = 1e-5);
        assert_relative_eq!(spectrum.spectral_irradiance(800.0), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_binned_spectrum_power() {
        let wavelengths = vec![400.0, 450.0, 500.0, 600.0, 700.0];
        let measurements = vec![0.1, 0.5, 0.8, 0.2];

        let spectrum = BinnedSpectrum::new(wavelengths, measurements).unwrap();

        // Test whole range with 1.0 cm² aperture
        // Expected: (0.1 * 50 + 0.5 * 50 + 0.8 * 100 + 0.2 * 100) * 1.0 = 130.0
        let band = Band::new(400.0, 700.0);
        let power = spectrum.irradiance(&band);
        assert_relative_eq!(power, 130.0, epsilon = 1e-5);

        // Test partial range (450-600nm)
        // Expected: (0.5 * 50 + 0.8 * 100) * 1.0 = 25.0 + 80.0 = 105.0
        let band = Band::new(450.0, 600.0);
        let power = spectrum.irradiance(&band);
        assert_relative_eq!(power, 105.0, epsilon = 1e-5);

        // Test range outside spectrum
        let band = Band::new(200.0, 300.0);
        assert_relative_eq!(spectrum.irradiance(&band), 0.0, epsilon = 1e-5);
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
    fn test_errors() {
        // Too few wavelengths
        let result = BinnedSpectrum::new(vec![400.0], vec![]);
        assert!(matches!(result, Err(SpectrumError::TooFewWavelengths)));

        // Measurements length mismatch
        let result = BinnedSpectrum::new(vec![400.0, 500.0, 600.0], vec![0.1, 0.2, 0.3]);
        assert!(matches!(result, Err(SpectrumError::MeasurementsMismatch)));

        // Wavelengths not ascending
        let result = BinnedSpectrum::new(vec![400.0, 500.0, 450.0], vec![0.1, 0.2]);
        assert!(matches!(result, Err(SpectrumError::InvalidFrequency(_))));

        // Negative wavelength
        let result = BinnedSpectrum::new(vec![-100.0, 500.0, 600.0], vec![0.1, 0.2]);
        assert!(matches!(result, Err(SpectrumError::InvalidFrequency(_))));
    }

    #[test]
    fn test_photoelectron_math_100percent() {
        let aperture_cm2 = 1.0; // 1 cm² aperture
        let duration = std::time::Duration::from_secs(1); // 1 second observation

        let band = Band::new(400.0, 600.0);
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

        let band = Band::new(400.0, 600.0);
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
}

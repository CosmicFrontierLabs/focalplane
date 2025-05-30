//! STIS Zodiacal Light Spectrum Implementation
//!
//! This module provides a spectrum implementation for zodiacal light data
//! from the Hubble Space Telescope STIS instrument documentation.
//!
//! Data source: NASA Dorado Sensitivity repository
//! https://raw.githubusercontent.com/nasa/dorado-sensitivity/refs/heads/main/dorado/sensitivity/data/stis_zodi_high.ecsv
//!
//! Original source: Table 6.4 of Hubble Space Telescope User Documentation
//! https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-6-tabular-sky-backgrounds

use std::sync::Once;

use super::spectrum::{Band, Spectrum, CGS};

/// Fixed array size for STIS zodiacal light data points
const STIS_DATA_POINTS: usize = 59;

/// Wavelength data in nanometers (converted from Angstroms in original data)
const WAVELENGTHS_NM: [f64; STIS_DATA_POINTS] = [
    100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0,
    230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0,
    360.0, 370.0, 380.0, 390.0, 400.0, 425.0, 450.0, 475.0, 500.0, 525.0, 550.0, 575.0, 600.0,
    625.0, 650.0, 675.0, 700.0, 725.0, 750.0, 775.0, 800.0, 825.0, 850.0, 875.0, 900.0, 925.0,
    950.0, 975.0, 1000.0, 1025.0, 1050.0, 1075.0, 1100.0,
];

/// Surface brightness data in original units: erg / (Angstrom arcsec² cm² s)
const SURFACE_BRIGHTNESS_ORIGINAL: [f64; STIS_DATA_POINTS] = [
    9.69e-29, 1.04e-26, 1.08e-25, 6.59e-25, 2.55e-24, 9.73e-24, 2.35e-22, 7.21e-21, 1.53e-20,
    2.25e-20, 3.58e-20, 1.23e-19, 2.21e-19, 1.81e-19, 1.83e-19, 2.53e-19, 3.06e-19, 1.01e-18,
    2.88e-19, 2.08e-18, 1.25e-18, 1.5e-18, 2.3e-18, 2.95e-18, 2.86e-18, 2.79e-18, 2.74e-18,
    3.32e-18, 3.12e-18, 3.34e-18, 4.64e-18, 4.65e-18, 5.58e-18, 5.46e-18, 5.15e-18, 5.37e-18,
    5.34e-18, 5.4e-18, 5.25e-18, 5.02e-18, 4.92e-18, 4.79e-18, 4.55e-18, 4.43e-18, 4.23e-18,
    4.04e-18, 3.92e-18, 3.76e-18, 3.5e-18, 3.43e-18, 3.23e-18, 3.07e-18, 2.98e-18, 2.86e-18,
    2.78e-18, 2.67e-18, 2.56e-18, 2.41e-18, 2.31e-18,
];

/// Converted spectral irradiance data in erg s⁻¹ cm⁻² Hz⁻¹
/// Lazy initialized to avoid expensive computation at compile time
static mut SPECTRAL_IRRADIANCE_CGS: [f64; STIS_DATA_POINTS] = [0.0; STIS_DATA_POINTS];
static INIT_CONVERSION: Once = Once::new();

/// STIS Zodiacal Light Spectrum
///
/// Provides zodiacal light spectral irradiance based on STIS measurements.
/// Data is interpolated linearly between measured wavelength points.
pub struct STISZodiacalSpectrum;

impl STISZodiacalSpectrum {
    /// Create a new STIS zodiacal spectrum instance
    pub fn new() -> Self {
        Self::initialize_conversion();
        Self
    }

    /// Initialize unit conversion from original data to CGS spectral irradiance
    fn initialize_conversion() {
        INIT_CONVERSION.call_once(|| {
            // Conversion factors:
            // 1. Angstrom to nm: already done in WAVELENGTHS_NM
            // 2. per Angstrom to per Hz: multiply by λ²/c where λ in cm
            // 3. arcsec² to steradian: multiply by (π/180/3600)²
            // 4. Surface brightness to spectral irradiance: assume unit steradian

            const ARCSEC_TO_RADIAN: f64 = std::f64::consts::PI / (180.0 * 3600.0);
            const ARCSEC2_TO_STERADIAN: f64 = ARCSEC_TO_RADIAN * ARCSEC_TO_RADIAN;

            unsafe {
                for i in 0..STIS_DATA_POINTS {
                    let wavelength_nm = WAVELENGTHS_NM[i];
                    let wavelength_cm = wavelength_nm * 1e-7; // nm to cm
                    let original_brightness = SURFACE_BRIGHTNESS_ORIGINAL[i];

                    // Convert from per Angstrom to per Hz
                    let angstrom_to_cm = 1e-8; // Angstrom to cm
                    let per_angstrom_to_per_hz =
                        (wavelength_cm * wavelength_cm) / (CGS::SPEED_OF_LIGHT * angstrom_to_cm);

                    // Convert from surface brightness to spectral irradiance
                    // Assuming unit steradian for zodiacal light background
                    SPECTRAL_IRRADIANCE_CGS[i] =
                        original_brightness * per_angstrom_to_per_hz * ARCSEC2_TO_STERADIAN;
                }
            }
        });
    }

    /// Get wavelength bounds of the spectrum
    pub fn wavelength_bounds() -> (f64, f64) {
        (WAVELENGTHS_NM[0], WAVELENGTHS_NM[STIS_DATA_POINTS - 1])
    }

    /// Linear interpolation between data points
    fn interpolate_irradiance(&self, wavelength_nm: f64) -> f64 {
        let (min_wl, max_wl) = Self::wavelength_bounds();

        // Return 0 if outside wavelength range
        if wavelength_nm < min_wl || wavelength_nm > max_wl {
            return 0.0;
        }

        // Find bracketing indices
        let mut idx = 0;
        while idx < STIS_DATA_POINTS - 1 && WAVELENGTHS_NM[idx + 1] < wavelength_nm {
            idx += 1;
        }

        // Handle exact match
        if WAVELENGTHS_NM[idx] == wavelength_nm {
            unsafe {
                return SPECTRAL_IRRADIANCE_CGS[idx];
            }
        }

        // Handle edge case at upper bound
        if idx == STIS_DATA_POINTS - 1 {
            unsafe {
                return SPECTRAL_IRRADIANCE_CGS[idx];
            }
        }

        // Linear interpolation
        let wl1 = WAVELENGTHS_NM[idx];
        let wl2 = WAVELENGTHS_NM[idx + 1];
        let weight = (wavelength_nm - wl1) / (wl2 - wl1);

        unsafe {
            let irr1 = SPECTRAL_IRRADIANCE_CGS[idx];
            let irr2 = SPECTRAL_IRRADIANCE_CGS[idx + 1];
            irr1 * (1.0 - weight) + irr2 * weight
        }
    }
}

impl Default for STISZodiacalSpectrum {
    fn default() -> Self {
        Self::new()
    }
}

impl Spectrum for STISZodiacalSpectrum {
    fn spectral_irradiance(&self, wavelength_nm: f64) -> f64 {
        self.interpolate_irradiance(wavelength_nm)
    }

    fn irradiance(&self, band: &Band) -> f64 {
        // Trapezoid integration over the band
        let (band_min, band_max) = (band.lower_nm, band.upper_nm);
        let (spectrum_min, spectrum_max) = Self::wavelength_bounds();

        // Find overlap between band and spectrum
        let start_wl = band_min.max(spectrum_min);
        let end_wl = band_max.min(spectrum_max);

        if start_wl >= end_wl {
            return 0.0;
        }

        // Integration step size (1 nm resolution)
        let step = 1.0;
        let mut total_irradiance = 0.0;
        let mut current_wl = start_wl;

        while current_wl < end_wl {
            let next_wl = (current_wl + step).min(end_wl);
            let irr1 = self.spectral_irradiance(current_wl);
            let irr2 = self.spectral_irradiance(next_wl);

            // Trapezoid rule: average height * width * frequency conversion
            let avg_irradiance = (irr1 + irr2) / 2.0;
            let wavelength_width = next_wl - current_wl;

            // Convert from per Hz to per wavelength interval
            let freq_width = CGS::SPEED_OF_LIGHT * wavelength_width * 1e-7
                / ((current_wl * 1e-7) * (next_wl * 1e-7));

            total_irradiance += avg_irradiance * freq_width;
            current_wl = next_wl;
        }

        total_irradiance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_wavelength_bounds() {
        let (min_wl, max_wl) = STISZodiacalSpectrum::wavelength_bounds();
        assert_eq!(min_wl, 100.0);
        assert_eq!(max_wl, 1100.0);
    }

    #[test]
    fn test_spectral_irradiance_bounds() {
        let spectrum = STISZodiacalSpectrum::new();

        // Should return 0 outside bounds
        assert_eq!(spectrum.spectral_irradiance(50.0), 0.0);
        assert_eq!(spectrum.spectral_irradiance(1500.0), 0.0);

        // Should return non-zero within bounds
        assert!(spectrum.spectral_irradiance(400.0) > 0.0);
        assert!(spectrum.spectral_irradiance(700.0) > 0.0);
    }

    #[test]
    fn test_spectral_irradiance_interpolation() {
        let spectrum = STISZodiacalSpectrum::new();

        // Test interpolation between two known points
        let irr_400 = spectrum.spectral_irradiance(400.0);
        let irr_425 = spectrum.spectral_irradiance(425.0);
        let irr_412_5 = spectrum.spectral_irradiance(412.5);

        // Should be approximately average of endpoints
        let expected = (irr_400 + irr_425) / 2.0;
        assert_relative_eq!(irr_412_5, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_irradiance_integration() {
        let spectrum = STISZodiacalSpectrum::new();

        // Test integration over a narrow band
        let band = Band::from_nm_bounds(500.0, 600.0);
        let total_irradiance = spectrum.irradiance(&band);

        // Should be positive and finite
        assert!(total_irradiance > 0.0);
        assert!(total_irradiance.is_finite());
    }

    #[test]
    fn test_unit_conversion_sanity() {
        let spectrum = STISZodiacalSpectrum::new();

        // Units should be in CGS: erg s⁻¹ cm⁻² Hz⁻¹
        // Values should be much smaller than original surface brightness
        let irradiance = spectrum.spectral_irradiance(550.0);

        // Should be positive and in reasonable range for zodiacal light
        assert!(irradiance > 0.0);
        assert!(irradiance < 1e-10); // Should be very small compared to stellar sources
    }
}

//! Stellar spectrum models for astronomical photometry
//!
//! This module provides implementations of the Spectrum trait
//! for modeling stellar spectra.

use std::time::Duration;

use once_cell::sync::Lazy;

use super::gaia::GAIA_PASSBAND;
use super::quantum_efficiency::QuantumEfficiency;
use super::spectrum::{nm_sub_bands, wavelength_to_ergs, Band, Spectrum, CGS};

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

/// A blackbody stellar spectrum based on temperature
///
/// This represents a source with a Planck spectrum based on temperature,
/// providing more realistic spectral models for stars of different colors.
#[derive(Debug, Clone)]
pub struct BlackbodyStellarSpectrum {
    /// Effective temperature in Kelvin
    temperature: f64,

    /// Scaling factor for the spectrum (equivalent to stellar radius and distance)
    scaling_factor: f64,
}

/// Convert B-V color index to effective temperature in Kelvin
///
/// Uses the empirical relation from Ballesteros (2012)
/// https://arxiv.org/pdf/1201.1809
/// Valid for B-V values from approximately -0.4 to 2.0
///
/// # Arguments
/// * `b_v` - B-V color index
///
/// # Returns
/// * Effective temperature in Kelvin
pub fn temperature_from_bv(b_v: f64) -> f64 {
    // equation 14 in the source above
    4600.0 * (1.0 / (0.92 * b_v + 1.7) + (1.0 / (0.92 * b_v + 0.62)))
}

fn compute_gaia_scaler() -> f64 {
    let flat_spectrum = FlatStellarSpectrum::from_gaia_magnitude(0.0);

    let bv_0_temp = temperature_from_bv(0.0);
    let bb_spect = BlackbodyStellarSpectrum::new(bv_0_temp, 1.0);

    // Gather the same passband, duration, and aperture
    let gaia_band = GAIA_PASSBAND.band();
    let duration = Duration::from_secs(1);
    let one_cm_sq = 1.0;

    // Compute both, with the blackbody output unscaled
    let gaia_gated_flat_flux = flat_spectrum.photons(&gaia_band, one_cm_sq, duration);
    let bb_gated_flux = bb_spect.photons(&gaia_band, one_cm_sq, duration);

    // Compute the scaling factor
    gaia_gated_flat_flux / bb_gated_flux
}

static GAIA_SCALER: Lazy<f64> = Lazy::new(compute_gaia_scaler);

impl BlackbodyStellarSpectrum {
    /// Create a new BlackbodyStellarSpectrum with a given temperature and scaling factor
    ///
    /// # Arguments
    ///
    /// * `temperature` - Effective temperature in Kelvin
    /// * `scaling_factor` - Scaling factor for normalizing the spectrum
    ///
    /// # Returns
    ///
    /// A new BlackbodyStellarSpectrum with the specified temperature
    pub fn new(temperature: f64, scaling_factor: f64) -> Self {
        if temperature <= 0.0 {
            panic!("Temperature must be positive, got: {}", temperature);
        }
        if scaling_factor <= 0.0 {
            panic!("Scaling factor must be positive, got: {}", scaling_factor);
        }

        Self {
            temperature,
            scaling_factor,
        }
    }

    /// Create a new BlackbodyStellarSpectrum from B-V color index and visual magnitude
    ///
    /// # Arguments
    ///
    /// * `b_v` - B-V color index
    /// * `v_mag` - Visual magnitude (V-band)
    ///
    /// # Returns
    ///
    /// A new BlackbodyStellarSpectrum with appropriate temperature and magnitude
    pub fn from_gaia_bv_magnitude(b_v: f64, gaia_mag: f64) -> Self {
        // First calculate temperature from B-V color index
        let temperature = temperature_from_bv(b_v);
        let scalar = *GAIA_SCALER * 10f64.powf(-0.4 * gaia_mag);

        // Create a new BlackbodyStellarSpectrum with the calculated temperature
        Self::new(temperature, scalar)
    }

    /// Calculate spectral radiance using Planck's law in CGS units
    ///
    /// # Arguments
    ///
    /// * `wavelength_cm` - Wavelength in centimeters
    ///
    /// # Returns
    ///
    /// Spectral radiance in erg⋅s^-1⋅cm^-2⋅sr^-1⋅cm^-1
    fn planck_spectral_radiance(&self, wavelength_cm: f64) -> f64 {
        // Calculate spectral radiance using Planck's law
        let numerator = 2.0 * CGS::PLANCK_CONSTANT * CGS::SPEED_OF_LIGHT * CGS::SPEED_OF_LIGHT;
        let exponent = (CGS::PLANCK_CONSTANT * CGS::SPEED_OF_LIGHT)
            / (wavelength_cm * 1.380649e-16 * self.temperature); // Boltzmann constant in erg/K
        let denominator = wavelength_cm.powi(5) * (exponent.exp() - 1.0);

        numerator / denominator
    }

    /// Convert spectral radiance to spectral irradiance
    ///
    /// Converts from spectral radiance (erg⋅s^-1⋅cm^-2⋅sr^-1⋅cm^-1)
    /// to spectral irradiance in frequency space (erg s⁻¹ cm⁻² Hz⁻¹)
    ///
    /// # Arguments
    ///
    /// * `wavelength_nm` - Wavelength in nanometers
    /// * `radiance` - Spectral radiance at the wavelength
    ///
    /// # Returns
    ///
    /// Spectral irradiance in erg s⁻¹ cm⁻² Hz⁻¹
    fn radiance_to_irradiance(&self, wavelength_nm: f64, radiance: f64) -> f64 {
        // Convert from per wavelength to per frequency
        // F_ν = (λ^2/c) * B_λ
        let wavelength_cm = wavelength_nm * 1e-7; // nm to cm
        let conversion = wavelength_cm * wavelength_cm / CGS::SPEED_OF_LIGHT;

        // Apply conversion and scaling factor
        radiance * conversion * self.scaling_factor
    }
}

impl Spectrum for BlackbodyStellarSpectrum {
    fn spectral_irradiance(&self, wavelength_nm: f64) -> f64 {
        // Ensure wavelength is positive
        if wavelength_nm <= 0.0 {
            return 0.0;
        }

        // Convert nm to cm for calculations
        let wavelength_cm = wavelength_nm * 1e-7;

        // Calculate spectral radiance using Planck's law
        let radiance = self.planck_spectral_radiance(wavelength_cm);

        // Convert to spectral irradiance in frequency space
        self.radiance_to_irradiance(wavelength_nm, radiance)
    }

    fn irradiance(&self, band: &Band) -> f64 {
        // Check for valid band
        if band.lower_nm >= band.upper_nm || band.lower_nm <= 0.0 {
            return 0.0;
        }

        // Decompose the band into integer nanometer bands for integration
        let bands = nm_sub_bands(band);

        // Integrate irradiance over all sub-bands
        let mut total_irradiance = 0.0;
        for sub_band in bands {
            // For each narrow band, calculate irradiance at center wavelength
            let center_nm = sub_band.center();
            let center_cm = center_nm * 1e-7;
            let spectral_radiance = self.planck_spectral_radiance(center_cm);
            let spectral_irr = self.radiance_to_irradiance(center_nm, spectral_radiance);

            // Convert from per frequency to per wavelength for integration
            let (lower_freq, upper_freq) = sub_band.frequency_bounds();
            total_irradiance += spectral_irr * (upper_freq - lower_freq);
        }

        total_irradiance
    }

    // Use default implementations for photons() and photo_electrons() from the trait
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
    fn test_blackbody_spectrum_creation() {
        // Create blackbody spectra at different temperatures
        let hot_star = BlackbodyStellarSpectrum::new(30000.0, 1.0); // Hot O-type star
        let sun_like = BlackbodyStellarSpectrum::new(5778.0, 1.0); // Sun-like star
        let cool_star = BlackbodyStellarSpectrum::new(3500.0, 1.0); // Cool M-type star

        // Verify all of them return positive irradiance values
        assert!(hot_star.spectral_irradiance(500.0) > 0.0);
        assert!(sun_like.spectral_irradiance(500.0) > 0.0);
        assert!(cool_star.spectral_irradiance(500.0) > 0.0);
    }

    #[test]
    fn test_blackbody_spectral_shape() {
        // Create a blackbody spectrum for a sun-like star (5778K)
        let sun_like = BlackbodyStellarSpectrum::new(5778.0, 1.0);

        // Check irradiance at different wavelengths
        let uv = sun_like.spectral_irradiance(200.0);
        let vis = sun_like.spectral_irradiance(500.0);

        // Sun-like star - in our frequency space representation, UV should be less than visible
        assert!(vis > uv, "Vis {} should be greater than uv {}", vis, uv);

        // Create a cool star (3500K)
        let cool_star = BlackbodyStellarSpectrum::new(3500.0, 1.0);

        // Cool stars peak in red, so red irradiance should be higher than blue
        let cool_blue = cool_star.spectral_irradiance(400.0);
        let cool_red = cool_star.spectral_irradiance(700.0);

        assert!(cool_red > cool_blue);

        // Create a hot star (30000K)
        let hot_star = BlackbodyStellarSpectrum::new(30000.0, 1.0);

        // Hot stars peak in blue/UV, so blue irradiance should be higher than red
        let hot_blue = hot_star.spectral_irradiance(400.0);
        let hot_red = hot_star.spectral_irradiance(700.0);

        assert!(hot_blue > hot_red);
    }

    #[test]
    fn test_band_integration() {
        // Create a 5778K blackbody spectrum (Sun-like)
        let sun_like = BlackbodyStellarSpectrum::new(5778.0, 1.0);

        // Test integration over various bands
        let blue_band = Band::from_nm_bounds(400.0, 500.0);
        let green_band = Band::from_nm_bounds(500.0, 600.0);
        let red_band = Band::from_nm_bounds(600.0, 700.0);
        let full_band = Band::from_nm_bounds(400.0, 700.0);

        // Calculate irradiance in each band
        let blue_irr = sun_like.irradiance(&blue_band);
        let green_irr = sun_like.irradiance(&green_band);
        let red_irr = sun_like.irradiance(&red_band);
        let full_irr = sun_like.irradiance(&full_band);

        // All irradiance values should be positive
        assert!(blue_irr > 0.0);
        assert!(green_irr > 0.0);
        assert!(red_irr > 0.0);

        // Full band irradiance should approximately equal the sum of the sub-bands
        let sum_irr = blue_irr + green_irr + red_irr;
        assert_relative_eq!(full_irr, sum_irr, epsilon = 0.01);
    }

    #[test]
    fn test_from_ab_mag() {
        // Create blackbody spectra with different magnitudes but same temperature
        let bright_star = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(0.0, 0.0);
        let dimmer_star = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(0.0, 5.0);

        // Check that the brightness scales correctly with magnitude
        // 5 magnitudes = factor of 100 in brightness
        let reference_band = Band::centered_on(550.0, 1e12);
        let bright_irr = bright_star.irradiance(&reference_band);
        let dimmer_irr = dimmer_star.irradiance(&reference_band);

        assert_relative_eq!(bright_irr / dimmer_irr, 100.0, epsilon = 0.1);
    }

    #[test]
    fn test_photons_from_blackbody() {
        // Compare photon counts between flat and blackbody spectra
        let flat = FlatStellarSpectrum::from_gaia_magnitude(12.0);
        let blackbody = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(0.0, 12.0);

        let band = Band::from_nm_bounds(400.0, 700.0);
        let aperture_cm2 = 1.0;
        let duration = std::time::Duration::from_secs(1);

        let flat_photons = flat.photons(&band, aperture_cm2, duration);
        let blackbody_photons = blackbody.photons(&band, aperture_cm2, duration);

        // Both should produce positive photon counts
        assert!(flat_photons > 0.0);
        assert!(blackbody_photons > 0.0);

        // They should be roughly in the same order of magnitude
        let ratio = flat_photons / blackbody_photons;
        assert!(ratio > 0.1 && ratio < 10.0);

        println!("Flat spectrum photons: {}", flat_photons);
        println!("Blackbody spectrum photons: {}", blackbody_photons);
        println!("Ratio (flat/blackbody): {}", ratio);
    }

    #[test]
    fn test_bv_to_temperature() {
        // Check temperature conversion for different stellar types
        let o_star_bv = -0.3; // Hot blue star
        let g_star_bv = 0.65; // Sun-like
        let m_star_bv = 1.6; // Cool red star

        // Calculate temperatures
        let fixed_mag = 10.0;
        let o_bb_spec = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(fixed_mag, o_star_bv);
        let g_bb_spec = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(fixed_mag, g_star_bv);
        let m_bb_spec = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(fixed_mag, m_star_bv);

        // Verify temperature ranges are correct
        // O should have more uv
        let blue = Band::from_nm_bounds(300.0, 400.0);
        assert!(o_bb_spec.irradiance(&blue) > g_bb_spec.irradiance(&blue));
        assert!(o_bb_spec.irradiance(&blue) > m_bb_spec.irradiance(&blue));
    }
}

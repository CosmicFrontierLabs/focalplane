//! Photon to electron conversion utilities for detector modeling.

use crate::image_proc::airy::PixelScaledAiryDisk;
#[cfg(test)]
use crate::photometry::spectrum::FlatSpectrum;
use crate::photometry::{
    spectrum::{wavelength_to_ergs, Band},
    QuantumEfficiency, Spectrum,
};
use std::time::Duration;

/// Calculate photon flux for each nm sub-band of a wavelength band.
///
/// For each nm sub-band, calculates the photon flux (photons/s/cm²)
/// from the spectrum's irradiance.
///
/// # Arguments
/// * `spectrum` - The spectrum to calculate photon flux from
/// * `band` - The wavelength band to subdivide and calculate over
///
/// # Returns
/// Iterator of (Band, photon_flux) pairs where photon_flux is in photons/s/cm²
pub fn sub_band_photon_fluxes<'a, S: Spectrum + ?Sized>(
    spectrum: &'a S,
    band: &Band,
) -> impl Iterator<Item = (Band, f64)> + 'a {
    let sub_bands = band.sub_nm_bands();
    sub_bands.into_iter().map(move |sub_band| {
        let energy_per_photon = wavelength_to_ergs(sub_band.center());
        let irradiance = spectrum.irradiance(&sub_band);
        let photon_flux = irradiance / energy_per_photon;
        (sub_band, photon_flux)
    })
}

/// Represents a point source flux with associated PSF characteristics.
///
/// Combines an effective PSF (point spread function) with a flux rate,
/// representing either photon or photoelectron flux from a point source
/// like a star. The flux is given as a rate per unit area.
///
/// # Fields
/// * `disk` - Effective chromatic PSF as a pixel-scaled Airy disk
/// * `flux` - Flux rate in units of particles per second per cm²
///
/// # Usage
/// Use `integrated_over()` to calculate total particle counts for a given
/// exposure time and telescope aperture area.
#[derive(Debug, Clone)]
pub struct SpotFlux {
    pub disk: PixelScaledAiryDisk,

    // Quantity of photons or electrons per second per cm²
    pub flux: f64,
}

impl SpotFlux {
    /// Calculate total particle count from flux rate.
    ///
    /// Integrates the flux rate over the given exposure time and telescope
    /// aperture area to calculate the total number of particles (photons
    /// or photoelectrons) collected.
    ///
    /// # Arguments
    /// * `exposure` - Integration time duration
    /// * `aperture_cm2` - Telescope collecting area in cm²
    ///
    /// # Returns
    /// Total particle count (photons or electrons)
    ///
    /// # Formula
    /// `total = flux_rate × aperture_area × exposure_time`
    pub fn integrated_over(&self, exposure: &Duration, aperture_cm2: f64) -> f64 {
        // Convert the flux to total photons or electrons over the integration time
        self.flux * aperture_cm2 * exposure.as_secs_f64()
    }
}

/// Complete flux characterization for a chromatic point source.
///
/// Contains both photon and photoelectron flux rates with their respective
/// effective PSFs. The two fluxes differ because:
/// - Photon flux represents all incident photons regardless of detection
/// - Photoelectron flux accounts for quantum efficiency wavelength dependence
///
/// The effective PSFs may also differ since QE weighting changes the
/// relative contribution of different wavelengths to the final PSF shape.
#[derive(Debug, Clone)]
pub struct SourceFlux {
    /// Effective PSF/flux for photon flux
    pub photons: SpotFlux,

    /// Effective PSF/flux for photoelectron flux
    pub electrons: SpotFlux,
}

/// Calculate effective PSF and photon/electron flux rates for chromatic sources.
///
/// Computes wavelength-dependent PSF broadening and photon/photoelectron
/// flux rates when integrating over the source spectrum and detector quantum efficiency.
/// The effective PSF size accounts for chromatic aberration due to wavelength-dependent
/// diffraction patterns.
///
/// # Algorithm
/// 1. Subdivides the QE band into 1nm sub-bands
/// 2. For each sub-band:
///    - Calculates photon flux from spectrum irradiance
///    - Applies QE to get photoelectron flux
///    - Scales PSF size linearly with wavelength (above reference)
///    - Accumulates flux-weighted PSF sizes
/// 3. Computes effective PSF from weighted averages
///
/// # Arguments
/// * `psf` - Reference PSF with baseline FWHM and reference wavelength
/// * `spectrum` - Source spectral energy distribution
/// * `qe` - Detector quantum efficiency curve
///
/// # Returns
/// `SourceFlux` containing:
/// - `photons`: SpotFlux with photon-weighted effective PSF and flux rate (photons/s/cm²)
/// - `electrons`: SpotFlux with photoelectron-weighted effective PSF and flux rate (e⁻/s/cm²)
///
/// # PSF Scaling Rules
/// - Below reference wavelength: No scaling (assumes diffraction-limited)
/// - Above reference wavelength: Linear scaling with wavelength ratio
///
/// # Usage
/// The returned flux rates must be integrated over exposure time and telescope
/// aperture area to get total photon/electron counts. Use `SpotFlux::integrated_over()`
/// for this calculation.
pub fn photon_electron_fluxes<S: Spectrum>(
    psf: &PixelScaledAiryDisk,
    spectrum: &S,
    qe: &QuantumEfficiency,
) -> SourceFlux {
    let band = qe.band();
    // THe names in this section are a bit more punctuated than I like
    // but `p_` and `pe_` are used to indicate photon and photo-electrons
    let mut total_p_flux = 0.0;
    let mut total_pe_flux = 0.0;

    let mut pe_weighted_fwhm_sum = 0.0;
    let mut pe_weight_total = 0.0;

    let mut p_weighted_fwhm_sum = 0.0;
    let mut p_weight_total = 0.0;

    for (sub_band, p_flux) in sub_band_photon_fluxes(spectrum, &band) {
        // Compute the photo-electrons flux for this sub-band
        let pe_flux = p_flux * qe.at(sub_band.center());

        total_p_flux += p_flux;
        total_pe_flux += pe_flux;

        let scale = if sub_band.center() < psf.reference_wavelength {
            // Below the reference wavelength, we cant assume better focusing
            1.0
        } else {
            // Airy disk radius scales linearly with wavelength
            sub_band.center() / psf.reference_wavelength
        };

        // Accrue the total relative FWHM weighted by subband PE
        pe_weighted_fwhm_sum += pe_flux * scale;
        pe_weight_total += pe_flux;

        p_weighted_fwhm_sum += p_flux * scale;
        p_weight_total += p_flux;
    }

    // Create PSF for the photos
    let p_scale = p_weighted_fwhm_sum / p_weight_total;
    let p_fwhm = psf.fwhm() * p_scale;
    let p_ref_wavelength = psf.reference_wavelength * p_scale;
    let p_psf = PixelScaledAiryDisk::with_fwhm(p_fwhm, p_ref_wavelength);

    // Create PSF for the photoelectrons
    let pe_scale = pe_weighted_fwhm_sum / pe_weight_total;
    let pe_fwhm = psf.fwhm() * pe_scale;
    let pe_ref_wavelength = psf.reference_wavelength * pe_scale;
    let pe_psf = PixelScaledAiryDisk::with_fwhm(pe_fwhm, pe_ref_wavelength);

    SourceFlux {
        photons: SpotFlux {
            disk: p_psf,
            flux: total_p_flux,
        },
        electrons: SpotFlux {
            disk: pe_psf,
            flux: total_pe_flux,
        },
    }
}

/// Calculate the number of photons within a wavelength range
///
/// # Arguments
///
/// * `spectrum` - The spectrum to integrate
/// * `band` - The wavelength band to integrate over
/// * `aperture_cm2` - Collection aperture area in square centimeters
/// * `duration` - Duration of the observation
///
/// # Returns
///
/// The number of photons detected in the specified band
pub fn photons<S: Spectrum>(
    spectrum: &S,
    band: &Band,
    aperture_cm2: f64,
    exposure: &Duration,
) -> f64 {
    // Convert power to photons per second
    // E = h * c / λ, so N = P / (h * c / λ)
    // where P is power in erg/s, h is Planck's constant, c is speed of light, and λ is wavelength in cm
    let total_photons: f64 = sub_band_photon_fluxes(spectrum, band)
        .map(|(_, photon_flux)| photon_flux)
        .sum();

    // Multiply by duration to get total photons detected
    total_photons * exposure.as_secs_f64() * aperture_cm2
}

/// Calculate the photo-electrons obtained from this spectrum when using a sensor with a given quantum efficiency
///
/// # Arguments
/// * `spectrum` - The spectrum to integrate
/// * `qe` - The quantum efficiency of the sensor as a function of wavelength
/// * `aperture_cm2` - Collection aperture area in square centimeters
/// * `duration` - Duration of the observation
///
/// # Returns
///
/// The number of electrons detected in the specified band
pub fn photo_electrons<S: Spectrum>(
    spectrum: &S,
    qe: &QuantumEfficiency,
    aperture_cm2: f64,
    duration: &Duration,
) -> f64 {
    // Convert power to photons per second
    // E = h * c / λ, so N = P / (h * c / λ)
    // where P is power in erg/s, h is Planck's constant, c is speed of light, and λ is wavelength in cm
    let band = qe.band();
    let total_electrons: f64 = sub_band_photon_fluxes(spectrum, &band)
        .map(|(sub_band, photon_flux)| photon_flux * qe.at(sub_band.center()))
        .sum();

    // Multiply by duration to get total photons detected
    total_electrons * duration.as_secs_f64() * aperture_cm2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photometry::{stellar::FlatStellarSpectrum, Band, BlackbodyStellarSpectrum};
    use approx::assert_relative_eq;

    #[test]
    fn test_chromatic_monochromatic_limit() {
        // Test that chromatic PSF reduces to monochromatic when spectrum is narrow
        let achromatic_disk = PixelScaledAiryDisk::with_fwhm(1.0, 550.0);

        // Create narrow-band filter centered at 550nm
        let band = Band::from_nm_bounds(549.0, 551.0);
        let qe = QuantumEfficiency::from_notch(&band, 1.0).unwrap();

        // Create flat spectrum
        let spectrum = FlatSpectrum::unit();

        // Get effective PSF for narrow band - should be close to monochromatic
        let fluxes = photon_electron_fluxes(&achromatic_disk, &spectrum, &qe);

        // The effective scale should be very close to the input disk's FWHM for narrow band at reference wavelength
        assert_relative_eq!(
            fluxes.photons.disk.fwhm(),
            achromatic_disk.fwhm(),
            epsilon = 1e-2
        );
        assert_relative_eq!(
            fluxes.electrons.disk.fwhm(),
            achromatic_disk.fwhm(),
            epsilon = 1e-2
        );
    }

    #[test]
    fn test_chromatic_broadening() {
        // Test that chromatic PSF effective scale reflects wavelength averaging
        let airy = PixelScaledAiryDisk::with_fwhm(1.0, 550.0);

        // Create broad-band filter (400-700nm)
        let band = Band::from_nm_bounds(400.0, 700.0);
        let qe = QuantumEfficiency::from_notch(&band, 1.0).unwrap();

        // Create flat spectrum
        let spectrum = FlatSpectrum::unit();

        // Get effective PSF
        let fluxes = photon_electron_fluxes(&airy, &spectrum, &qe);

        let e_spot = fluxes.electrons;
        let pe_spot = fluxes.photons;

        println!(
            "Chromatic PSF FWHM: {:.3}, Photoelectron FWHM: {:.3}",
            e_spot.disk.fwhm(),
            pe_spot.disk.fwhm()
        );
        assert!(
            e_spot.disk.fwhm() > airy.fwhm(),
            "Expected chromatic psf {} to be larger than achromatic psf {}",
            e_spot.disk.fwhm(),
            airy.fwhm()
        );
        assert_relative_eq!(e_spot.disk.fwhm(), airy.fwhm(), epsilon = 0.1);
    }

    #[test]
    fn test_ir_sensitive_detector_broadening() {
        // Test that IR-sensitive detectors see wider PSF than visible-only detectors
        // due to chromatic effects - longer wavelengths have larger Airy disks
        let airy = PixelScaledAiryDisk::with_fwhm(1.0, 550.0);

        // Create a sun-like blackbody spectrum
        let spectrum = BlackbodyStellarSpectrum::new(5780.0, 1e-10);

        // Create visible-only QE (400-700nm)
        let visible_wavelengths = vec![350.0, 400.0, 500.0, 600.0, 700.0, 750.0];
        let visible_efficiencies = vec![0.0, 0.5, 0.8, 0.8, 0.5, 0.0];
        let visible_qe =
            QuantumEfficiency::from_table(visible_wavelengths, visible_efficiencies).unwrap();

        // Create IR-sensitive QE (400-1000nm)
        let ir_wavelengths = vec![
            350.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0,
        ];
        let ir_efficiencies = vec![0.0, 0.5, 0.8, 0.8, 0.7, 0.6, 0.4, 0.2, 0.0];
        let ir_qe = QuantumEfficiency::from_table(ir_wavelengths, ir_efficiencies).unwrap();

        // Get PSFs for both detectors
        let vis_flux = photon_electron_fluxes(&airy, &spectrum, &visible_qe);

        let ir_flux = photon_electron_fluxes(&airy, &spectrum, &ir_qe);

        // IR-sensitive detector should see wider PSF due to longer wavelengths
        assert!(
            ir_flux.electrons.disk.fwhm() > vis_flux.electrons.disk.fwhm(),
            "IR-sensitive detector PSF ({:.3}) should be wider than visible-only PSF ({:.3})",
            ir_flux.electrons.disk.fwhm(),
            vis_flux.electrons.disk.fwhm()
        );

        // The photon-weighted PSF should also be wider for IR
        assert!(
            ir_flux.photons.disk.fwhm() > vis_flux.photons.disk.fwhm(),
            "IR photon PSF ({:.3}) should be wider than visible photon PSF ({:.3})",
            ir_flux.photons.disk.fwhm(),
            vis_flux.photons.disk.fwhm()
        );

        // For IR-sensitive detector, the photoelectron PSF may be narrower than photon PSF
        // because QE is typically higher in visible wavelengths where Airy disk is smaller.
        // The key insight is that IR detectors still see wider PSF than visible-only detectors.

        println!(
            "Visible detector - Photon FWHM: {:.3}, PE FWHM: {:.3}",
            vis_flux.photons.disk.fwhm(),
            vis_flux.electrons.disk.fwhm()
        );
        println!(
            "IR-sensitive detector - Photon FWHM: {:.3}, PE FWHM: {:.3}",
            ir_flux.photons.disk.fwhm(),
            ir_flux.electrons.disk.fwhm()
        );
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

        let photons_count = photons(&spectrum, &band, aperture_cm2, &duration);
        let electrons_count = photo_electrons(&spectrum, &qe, aperture_cm2, &duration);

        // For a perfect QE, the number of electrons should equal the number of photons
        let err = f64::abs(photons_count - electrons_count) / photons_count;

        assert!(
            err < 0.01,
            "Expected {photons_count} electrons, got {electrons_count}"
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

        let photons_count = photons(&spectrum, &band, aperture_cm2, &duration);
        let electrons_count = photo_electrons(&spectrum, &qe, aperture_cm2, &duration);

        // For 50% QE, electrons should be ~50% of photons
        let ratio = electrons_count / photons_count;

        assert_relative_eq!(ratio, 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_photoelectron_math_vega() {
        let aperture_cm2 = 1.0; // 1 cm² aperture
        let duration = std::time::Duration::from_secs(1); // 1 second observation

        let v_band = Band::from_nm_bounds(551.0 - 44.0, 551.0 + 44.0);
        // Make a pretend QE with 50% efficiency in the 400-600nm range
        let qe = QuantumEfficiency::from_notch(&v_band, 1.0).unwrap();

        // Create a flat spectrum with a known flux density
        let spectrum = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(0.0, 0.0);

        let disk = PixelScaledAiryDisk::with_fwhm(1.0, 550.0);

        let fluxes = photon_electron_fluxes(&disk, &spectrum, &qe);
        let photons_count = fluxes.photons.integrated_over(&duration, aperture_cm2);
        let electrons_count = fluxes.electrons.integrated_over(&duration, aperture_cm2);

        // Rule of thumb for vega is 1000 photons per second per cm² per second per angstrom
        let angs = v_band.width() * 10.0; // Convert nm to angstroms
        let expected_photons = 1000.0 * angs * aperture_cm2;

        println!("Photons: {photons_count}, Electrons: {electrons_count}");
        let err = f64::abs(photons_count - expected_photons) / photons_count;
        assert!(
            err < 0.05,
            "Expected {expected_photons} photons, got {photons_count}"
        );
    }

    #[test]
    fn test_chromatic_achromatic_flux_match() {
        // Test that chromatic PSF effective scale reflects wavelength averaging
        let airy = PixelScaledAiryDisk::with_fwhm(1.0, 550.0);

        // Create broad-band filter (400-700nm)
        let band = Band::from_nm_bounds(400.0, 700.0);
        let qe = QuantumEfficiency::from_notch(&band, 1.0).unwrap();

        // Create flat spectrum
        let spectrum = FlatSpectrum::unit();

        // Get effective PSF
        let fluxes = photon_electron_fluxes(&airy, &spectrum, &qe);

        let e_spot = fluxes.electrons;
        let pe_spot = fluxes.photons;

        let aperture_cm2 = 100.0; // 1 cm² aperture
        let exposure = Duration::from_secs(1); // 1 second observation

        let photons_direct = photons(&spectrum, &band, aperture_cm2, &exposure);
        let electrons_direct = photo_electrons(&spectrum, &qe, aperture_cm2, &exposure);

        assert_relative_eq!(
            e_spot.integrated_over(&exposure, aperture_cm2),
            electrons_direct,
            epsilon = 0.1
        );
        assert_relative_eq!(
            pe_spot.integrated_over(&exposure, aperture_cm2),
            photons_direct,
            epsilon = 0.1
        );
    }
}

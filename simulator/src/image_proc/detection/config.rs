//! Configuration utilities for star detection algorithms
//!
//! This module provides optimized configurations for different star detection
//! algorithms based on space telescope characteristics.

use starfield::image::starfinders::{DAOStarFinderConfig, IRAFStarFinderConfig};

/// Create DAOStarFinder configuration optimized for space telescope
///
/// # Arguments
/// * `airy_disk_pixels` - Airy disk diameter in pixels
/// * `background_rms` - RMS noise level of the background
/// * `detection_sigma` - Detection threshold in units of sigma (typically 5.0)
///
/// # Returns
/// DAOStarFinderConfig optimized for space telescope observations
pub fn dao_autoconfig(
    airy_disk_pixels: f64,
    background_rms: f64,
    detection_sigma: f64,
) -> DAOStarFinderConfig {
    DAOStarFinderConfig {
        threshold: detection_sigma * background_rms * 1.2,
        fwhm: 0.5 * airy_disk_pixels, // Airy disk FWHM
        ratio: 1.0,
        theta: 0.0,
        sigma_radius: 1.5,
        sharpness: 0.2..=5.0,
        roundness: -0.5..=0.5,
        exclude_border: false,
        brightest: None,
        peakmax: None,
        min_separation: 0.8 * airy_disk_pixels,
    }
}

/// Create IRAFStarFinder configuration optimized for space telescope
///
/// # Arguments
/// * `airy_disk_pixels` - Airy disk diameter in pixels
/// * `background_rms` - RMS noise level of the background
/// * `detection_sigma` - Detection threshold in units of sigma (typically 5.0)
///
/// # Returns
/// IRAFStarFinderConfig optimized for space telescope observations
pub fn iraf_autoconfig(
    airy_disk_pixels: f64,
    background_rms: f64,
    detection_sigma: f64,
) -> IRAFStarFinderConfig {
    IRAFStarFinderConfig {
        threshold: detection_sigma * background_rms,
        fwhm: 0.55 * airy_disk_pixels, // Slightly larger for IRAF
        sigma_radius: 1.5,
        minsep_fwhm: 1.5,     // 1.5 × FWHM separation
        sharpness: 0.5..=5.0, // IRAF scale is different
        roundness: 0.5..=0.5, // Tight for space telescope
        exclude_border: false,
        brightest: None,
        peakmax: None,
        min_separation: None, // Let IRAF calculate from minsep_fwhm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_telescope_configs() {
        let airy_disk = 2.5; // pixels
        let background_rms = 1.2;
        let detection_sigma = 5.0;

        let dao = dao_autoconfig(airy_disk, background_rms, detection_sigma);
        assert_eq!(dao.threshold, 6.0);
        assert_eq!(dao.fwhm, 1.25);
        assert_eq!(dao.min_separation, 2.0);

        let iraf = iraf_autoconfig(airy_disk, background_rms, detection_sigma);
        assert_eq!(iraf.threshold, 6.0);
        assert_eq!(iraf.fwhm, 1.375);
        assert_eq!(iraf.minsep_fwhm, 1.5);
    }
}

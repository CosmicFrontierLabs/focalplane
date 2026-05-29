//! Per-tile mean-electron accumulation for the motion-blur renderer.

use ndarray::Array2;
use rayon::prelude::*;

/// Per-tile mean-electron accumulator (pre-Poisson).
///
/// Stars are splatted in as mean-electron contributions (not draws) so
/// that all sub-samples, the zodiacal uniform, and the dark-current uniform
/// can be combined into a single Poisson lambda.
#[derive(Debug, Clone)]
pub struct SensorAccumulator {
    /// Accumulated mean electrons from star subsample splats.
    pub star_mean_electrons: Array2<f64>,
}

impl SensorAccumulator {
    /// Allocate a zeroed accumulator shaped `(height, width)`.
    pub fn zero(width: usize, height: usize) -> Self {
        Self {
            star_mean_electrons: Array2::zeros((height, width)),
        }
    }

    /// Splat one star's mean electrons (already integrated over `dt`, not
    /// Poisson-sampled) into the accumulator via Simpson's rule over the
    /// Airy disk.
    ///
    /// `total_electrons` is the expected number of electrons this subsample
    /// contributes across the disk — i.e. the star's mean-electron rate at
    /// the star's chromatic effective PSF, multiplied by aperture area and
    /// the subsample duration.
    pub fn splat_psf(
        &mut self,
        px: f64,
        py: f64,
        total_electrons: f64,
        psf: &shared::image_proc::airy::PixelScaledAiryDisk,
    ) {
        crate::image_proc::deposit::splat_deposit(
            &mut self.star_mean_electrons,
            px,
            py,
            total_electrons,
            psf,
        );
    }

    /// Splat one galaxy's mean electrons (already integrated over `dt`,
    /// not Poisson-sampled) into the same accumulator stars use, via
    /// the galaxy's `SersicSplat` deposit. The single-Poisson invariant
    /// (INVARIANTS §1) is preserved because galaxies land on the same
    /// `star_mean_electrons` buffer that the unified Poisson eventually
    /// samples — galaxy and star shot noise are co-sampled from the
    /// per-pixel `Poisson(λ_total)` of the combined mean.
    pub fn splat_galaxy(
        &mut self,
        px: f64,
        py: f64,
        total_electrons: f64,
        sersic: &crate::image_proc::sersic_splat::SersicSplat,
    ) {
        crate::image_proc::deposit::splat_deposit(
            &mut self.star_mean_electrons,
            px,
            py,
            total_electrons,
            sersic,
        );
    }

    /// Consume the accumulator and return the combined mean-electron image
    /// = star mean + zodiacal uniform + dark-current uniform (pre-Poisson).
    ///
    /// The accumulator's existing `star_mean_electrons` buffer is reused —
    /// no second 488 MB allocation, no read-then-write of a fresh array.
    /// The scalar background is added in place via rayon-parallel
    /// element-wise iteration (ndarray's `rayon` feature is enabled).
    pub fn into_combined_mean(mut self, zodiacal_per_px: f64, dark_per_px: f64) -> Array2<f64> {
        let bg = (zodiacal_per_px + dark_per_px).max(0.0);
        if bg > 0.0 {
            self.star_mean_electrons
                .par_iter_mut()
                .for_each(|pixel| *pixel += bg);
        }
        self.star_mean_electrons
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_accumulator_combined_mean() {
        let mut acc = SensorAccumulator::zero(4, 4);
        acc.star_mean_electrons[[1, 1]] = 5.0;
        let combined = acc.into_combined_mean(2.0, 1.0);
        assert_eq!(combined[[0, 0]], 3.0); // 0 + 2 + 1
        assert_eq!(combined[[1, 1]], 8.0); // 5 + 2 + 1
    }
}

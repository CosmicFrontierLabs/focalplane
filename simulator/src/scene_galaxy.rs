//! `GalaxyInFrame` — the per-sensor representation of a galaxy ready
//! for splatting onto a pixel buffer. Mirrors `StarInFrame` (in
//! `image_proc::render`) shape-for-shape so both source kinds flow
//! through the same `FrameSource`/`MeanFluxDeposit` pipeline.

use std::time::Duration;

use shared::units::Area;

use crate::image_proc::deposit::FrameSource;
use crate::image_proc::sersic_splat::SersicSplat;
use crate::photometry::photoconversion::SourceFlux;

/// One galaxy projected onto one sensor, ready to be splatted into a
/// mean-electron buffer alongside the per-sensor star list.
///
/// Fields parallel `StarInFrame`:
/// - `(x, y)`: sub-pixel position on the sensor
/// - `id`: catalog ID (NSAID for NSA, etc.) for caching across frames /
///   subsamples
/// - `flux`: same `SourceFlux` shape stars use, carrying the chromatic
///   electron rate from spectrum × QE; `flux.electrons.disk` is the
///   per-galaxy effective Airy disk (used by future PSF-convolved
///   deposit modes) — for v1's bare-Sérsic deposit only `flux.electrons.flux`
///   is consumed
/// - `deposit`: `SersicSplat` built once per (galaxy, sensor) at the
///   sensor's plate scale
#[derive(Clone, Debug)]
pub struct GalaxyInFrame {
    pub x: f64,
    pub y: f64,
    pub id: u64,
    pub flux: SourceFlux,
    pub deposit: SersicSplat,
}

impl FrameSource for GalaxyInFrame {
    type Deposit = SersicSplat;

    fn position_pixels(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    fn total_electrons(&self, dt: Duration, aperture: Area) -> f64 {
        self.flux.electrons.integrated_over(&dt, aperture)
    }

    fn deposit(&self) -> &Self::Deposit {
        &self.deposit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_proc::deposit::render_sources;
    use crate::photometry::photoconversion::SpotFlux;
    use crate::photometry::sersic::SersicProfile;
    use crate::sims::motion_blur::SensorAccumulator;
    use ndarray::Array2;
    use shared::image_proc::airy::PixelScaledAiryDisk;
    use shared::units::{AreaExt, LengthExt, Wavelength};
    use std::time::Duration;

    /// Build a synthetic GalaxyInFrame for tests. Hand-rolls a
    /// `SourceFlux` instead of running through the spectrum pipeline
    /// since the integration coupling lives outside this module.
    fn fixture_galaxy(x: f64, y: f64, electron_rate: f64) -> GalaxyInFrame {
        let psf = PixelScaledAiryDisk::with_fwhm(2.0, Wavelength::from_nanometers(550.0));
        let spot = SpotFlux {
            disk: psf,
            flux: electron_rate, // electrons / s / cm²
        };
        let flux = SourceFlux {
            photons: spot.clone(),
            electrons: spot,
        };
        let profile = SersicProfile {
            theta_half_arcsec: 5.0,
            n: 4.0,
            axis_ratio: 0.6,
            position_angle_deg: 30.0,
        };
        let deposit = SersicSplat::new(profile, 0.5);
        GalaxyInFrame {
            x,
            y,
            id: 42,
            flux,
            deposit,
        }
    }

    /// **INVARIANTS §2 lock for galaxies**: depositing the same
    /// galaxy via the static path (`render_sources` → into a buffer)
    /// and via the motion-blur path (`SensorAccumulator::splat_galaxy`)
    /// must produce byte-identical buffers. After PR 3 both routes
    /// hit `splat_deposit`; this test fires only if a future PR forks
    /// them.
    #[test]
    fn galaxy_static_and_motion_paths_are_byte_equal() {
        let g = fixture_galaxy(40.0, 35.5, 1234.5);
        let aperture = Area::from_square_centimeters(100.0);
        let exposure = Duration::from_secs(5);

        // Path A: render_sources (static).
        let mut buf_a = Array2::<f64>::zeros((80, 80));
        render_sources(&mut buf_a, std::slice::from_ref(&g), exposure, aperture);

        // Path B: SensorAccumulator::splat_galaxy (motion blur).
        let mut acc = SensorAccumulator::zero(80, 80);
        let total = g.flux.electrons.integrated_over(&exposure, aperture);
        acc.splat_galaxy(g.x, g.y, total, &g.deposit);

        assert_eq!(
            buf_a.as_slice().unwrap(),
            acc.star_mean_electrons.as_slice().unwrap(),
            "static and splat_galaxy paths must be byte-identical"
        );
    }

    /// Co-deposit invariant: stars + galaxies splatted onto the same
    /// buffer in either order produce the same result, and the result
    /// equals the analytic sum of the two independent renders. Locks
    /// the additivity of the deposit pipeline — important for the
    /// single-Poisson invariant since the Poisson stage applies to
    /// the *sum* of the two source kinds.
    #[test]
    fn galaxy_and_star_deposits_are_additive() {
        use crate::image_proc::render::StarInFrame;

        let g = fixture_galaxy(40.0, 35.5, 1234.5);
        let psf = PixelScaledAiryDisk::with_fwhm(2.0, Wavelength::from_nanometers(550.0));
        let spot = SpotFlux {
            disk: psf,
            flux: 567.8,
        };
        let flux = SourceFlux {
            photons: spot.clone(),
            electrons: spot,
        };
        let star = StarInFrame {
            x: 42.0,
            y: 38.0,
            spot: flux,
            star: starfield::catalogs::StarData::new(1, 100.0, 0.0, 5.0, None),
        };
        let aperture = Area::from_square_centimeters(100.0);
        let exposure = Duration::from_secs(5);

        // Independent renders.
        let mut buf_g = Array2::<f64>::zeros((80, 80));
        render_sources(&mut buf_g, std::slice::from_ref(&g), exposure, aperture);
        let mut buf_s = Array2::<f64>::zeros((80, 80));
        render_sources(&mut buf_s, std::slice::from_ref(&star), exposure, aperture);
        let analytic_sum = &buf_g + &buf_s;

        // Combined render in one buffer (galaxy first).
        let mut buf_gs = Array2::<f64>::zeros((80, 80));
        render_sources(&mut buf_gs, std::slice::from_ref(&g), exposure, aperture);
        render_sources(&mut buf_gs, std::slice::from_ref(&star), exposure, aperture);
        // And in the other order.
        let mut buf_sg = Array2::<f64>::zeros((80, 80));
        render_sources(&mut buf_sg, std::slice::from_ref(&star), exposure, aperture);
        render_sources(&mut buf_sg, std::slice::from_ref(&g), exposure, aperture);

        // Both ordered combinations must equal the analytic sum.
        assert_eq!(
            buf_gs.as_slice().unwrap(),
            analytic_sum.as_slice().unwrap(),
            "galaxy-then-star must equal analytic sum"
        );
        assert_eq!(
            buf_sg.as_slice().unwrap(),
            analytic_sum.as_slice().unwrap(),
            "star-then-galaxy must equal analytic sum"
        );
    }
}

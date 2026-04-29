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
        fixture_galaxy_at_scale(x, y, electron_rate, 0.5)
    }

    fn fixture_galaxy_at_scale(
        x: f64,
        y: f64,
        electron_rate: f64,
        arcsec_per_pixel: f64,
    ) -> GalaxyInFrame {
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
        let deposit = SersicSplat::new(profile, arcsec_per_pixel);
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

    /// **End-to-end flux roundtrip**: build a `Renderer` containing
    /// only a single galaxy (no stars), render with Poisson disabled,
    /// and verify the sum of all electron pixels equals the galaxy's
    /// expected `flux_rate × aperture × exposure` within the
    /// truncation budget.
    ///
    /// This is the most direct test that the SersicSplat normalisation,
    /// the Renderer's `from_stars_and_galaxies` plumbing, the
    /// `apply_poisson=false` short-circuit, and the per-exposure linear
    /// scaling all compose correctly. Any unit error or missing factor
    /// along the chain shows up here.
    #[test]
    fn end_to_end_flux_roundtrip_through_renderer() {
        use crate::hardware::sensor::models::GSENSE4040BSI;
        use crate::hardware::SatelliteConfig;
        use crate::hardware::TelescopeConfig;
        use crate::image_proc::render::Renderer;
        use crate::photometry::zodiacal::SolarAngularCoordinates;
        use shared::units::{Length, LengthExt as _, Temperature, TemperatureExt as _};

        // Build a satellite config with the test sensor + a fixed-FWHM
        // telescope. The exact aperture / focal length don't matter
        // for the roundtrip — they cancel out via integrated_over.
        let telescope = TelescopeConfig::new(
            "Roundtrip test",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = GSENSE4040BSI.clone();
        let temp = Temperature::from_celsius(-10.0);
        let satellite = SatelliteConfig::new(telescope, sensor, temp);
        let aperture = satellite.telescope.clear_aperture_area();

        // Place a galaxy at a centred sub-pixel position on the sensor.
        let (w, h) = satellite.sensor.dimensions.get_pixel_width_height();
        let cx = (w as f64) * 0.5;
        let cy = (h as f64) * 0.5;
        let electron_rate = 5_000.0_f64; // electrons / s / cm²
                                         // Use a well-resolved deposit (50 pixels per θ_eff) so the
                                         // adaptive Simpson sub-pixel integrator converges within the
                                         // documented 2% truncation budget.
        let g = fixture_galaxy_at_scale(cx, cy, electron_rate, 0.1);
        let exposure = Duration::from_secs(2);
        let total_in = g.flux.electrons.integrated_over(&exposure, aperture);

        // Render via the static path with Poisson disabled — the
        // resulting `star_image` is the *mean* electron buffer with
        // both stars (empty) and galaxies splatted in, scaled by
        // exposure.
        let renderer = Renderer::from_stars_and_galaxies(&[], &[g], satellite);
        let zodi = SolarAngularCoordinates::new(180.0, 60.0).unwrap();
        let result = renderer.render_with_options(&exposure, &zodi, false, None);
        let total_out: f64 = result.star_image.iter().sum();

        // Expect within the SersicSplat truncation budget: > 95% of
        // input flux is captured (loose bound covers high-n wings),
        // and never exceeds input (Simpson sub-sample is a bounded
        // integrator).
        let captured = total_out / total_in;
        assert!(
            captured > 0.95 && captured < 1.02,
            "renderer flux roundtrip: input {total_in:.4} electrons, output {total_out:.4} electrons (captured {captured:.4})"
        );
    }

    /// **INVARIANTS §1 lock — variance equals mean (Poisson identity)**.
    /// Render a galaxy-only scene N times with different RNG seeds and
    /// no read noise; per-pixel variance over the trials must equal
    /// per-pixel mean within Monte-Carlo error. Anchors the property
    /// `Var[Poisson(λ)] = E[Poisson(λ)] = λ` against any future bug
    /// that introduces a per-source or per-stamp Poisson, which would
    /// shift the variance/mean ratio away from 1.
    ///
    /// `#[ignore]` because it's slow (200 trials × full render). Run
    /// manually:
    ///   cargo test -p simulator --lib scene_galaxy --
    ///       --ignored variance_equals_mean_poisson_identity
    #[test]
    #[ignore]
    fn variance_equals_mean_poisson_identity() {
        use crate::hardware::sensor::models::GSENSE4040BSI;
        use crate::hardware::SatelliteConfig;
        use crate::hardware::TelescopeConfig;
        use crate::image_proc::render::Renderer;
        use crate::photometry::zodiacal::SolarAngularCoordinates;
        use ndarray::Array2;
        use shared::units::{Length, LengthExt as _, Temperature, TemperatureExt as _};

        let telescope = TelescopeConfig::new(
            "Poisson test",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = GSENSE4040BSI.clone();
        let temp = Temperature::from_celsius(-10.0);
        let satellite = SatelliteConfig::new(telescope, sensor, temp);

        let (w, h) = satellite.sensor.dimensions.get_pixel_width_height();
        let cx = (w as f64) * 0.5;
        let cy = (h as f64) * 0.5;
        // Lower electron rate keeps per-pixel mean small enough that
        // the Poisson identity is detectable; high rate gets dominated
        // by the variance of the variance estimator.
        let g = fixture_galaxy_at_scale(cx, cy, 50.0, 0.1);
        let renderer = Renderer::from_stars_and_galaxies(&[], &[g], satellite);
        let zodi = SolarAngularCoordinates::new(180.0, 60.0).unwrap();
        let exposure = Duration::from_secs(2);

        let n_trials = 200_u32;
        let mut sum: Option<Array2<f64>> = None;
        let mut sum_sq: Option<Array2<f64>> = None;
        for trial in 0..n_trials {
            // Render with Poisson on but force read noise off by
            // computing only the star_image (which the Renderer
            // already separates from sensor noise).
            let result = renderer.render_with_options(
                &exposure,
                &zodi,
                true, // apply Poisson
                Some(0xCAFE_F00D ^ trial as u64),
            );
            let img = result.star_image;
            sum = Some(match sum {
                Some(s) => &s + &img,
                None => img.clone(),
            });
            sum_sq = Some(match sum_sq {
                Some(s) => &s + &(&img * &img),
                None => &img * &img,
            });
        }
        let n = n_trials as f64;
        let mean = sum.unwrap() / n;
        let var = sum_sq.unwrap() / n - &mean * &mean;

        // Restrict to pixels with non-trivial flux so the variance
        // estimator isn't dominated by the zero-mean tail.
        let mut ratios = Vec::new();
        for (m, v) in mean.iter().zip(var.iter()) {
            if *m > 1.0 {
                ratios.push(v / m);
            }
        }
        assert!(!ratios.is_empty(), "no high-mean pixels to evaluate");
        let mean_ratio: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
        // Pure-Poisson identity: ratio should be 1.0 within MC bound
        // ~ √(2/(N-1)) ≈ 0.10 for N=200. Allow ±0.10 — tighter would
        // be flaky on modest sample sizes; looser would let regressions
        // through.
        assert!(
            (mean_ratio - 1.0).abs() < 0.10,
            "Poisson variance/mean ratio averaged over {} pixels = {:.4}, expected ≈ 1.0",
            ratios.len(),
            mean_ratio
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

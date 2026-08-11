//! `Galaxy` (flat, sky-truth catalog entity) and `GalaxyInFrame` (per-
//! sensor projected wrapper).
//!
//! The renderer-facing [`crate::sims::motion_blur::LightSources`] now
//! carries a flat `&[Galaxy]` slice: one entry per catalog galaxy,
//! independent of which sensor (or how many) the galaxy projects
//! onto. The motion-blur renderer expands this into per-sensor
//! `Vec<GalaxyInFrame>` lists at render start via
//! [`project_galaxies_to_sensors`], which projects each galaxy onto
//! every sensor whose extent + halo-aware padding contains the
//! galaxy's centre. This is how big galaxies that subtend multiple
//! sensors get rendered on each of them — the renderer sorts out the
//! per-sensor incidence, callers don't need to.
//!
//! `GalaxyInFrame` remains the per-sensor projected representation,
//! used by the static [`crate::image_proc::render::Renderer`] path
//! (single-sensor) and as the motion-blur renderer's internal type.

use std::time::Duration;

use nalgebra::UnitQuaternion;
use serde::{Deserialize, Serialize};
use shared::units::{Area, LengthExt};
use starfield::catalogs::{SersicProfile, StarData};
use starfield::Equatorial;

use crate::hardware::satellite::{FocalPlaneConfig, FocalPlaneProjector};
use crate::image_proc::deposit::{FrameSource, MeanFluxDeposit};
use crate::image_proc::sersic_splat::SersicSplat;
use crate::photometry::photoconversion::SourceFlux;

/// One galaxy projected onto one sensor, ready to be splatted into a
/// mean-electron buffer alongside the per-sensor star list.
///
/// Fields parallel `StarInFrame`:
/// - `(x, y)`: sub-pixel position on the sensor (derived from
///   `position` via the trajectory's mid-frame projection)
/// - `position`: sky coordinates of the galaxy centre — the
///   catalog-truth location, preserved so the renderer can emit it
///   into scene metadata without re-querying the catalog
/// - `id`: catalog ID (NSAID for NSA, etc.) for caching across frames /
///   subsamples
/// - `name`: optional human-readable catalog name (e.g. "M87",
///   "NGC 4486" for the bright-galaxy catalog); `None` for catalogs
///   that only carry numeric IDs (NSA)
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
    pub position: Equatorial,
    pub id: u64,
    pub name: Option<String>,
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

/// Flat sky-truth representation of one catalog galaxy.
///
/// Carries everything needed to deposit the galaxy onto any sensor of
/// any focal plane, without committing to a specific projection. The
/// motion-blur renderer projects each `Galaxy` onto every sensor it
/// subtends (with halo-aware padding) via
/// [`project_galaxies_to_sensors`] at render start.
///
/// Fields:
/// - `id` / `name`: catalog identifier (numeric + optional human-readable).
/// - `position`: galaxy centre in equatorial J2000 coordinates.
/// - `profile`: Sérsic shape (theta_half, n, axis_ratio, position_angle).
/// - `flux`: integrated photon / photoelectron rate at the entrance
///   aperture. Computed by the catalog builder using the focal plane's
///   reference sensor (sensor 0) QE — accurate for the homogeneous
///   arrays currently in production. Heterogeneous-array per-sensor
///   flux would belong in a render-time cache analogous to stars'
///   `FluxCache`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Galaxy {
    pub id: u64,
    pub name: Option<String>,
    pub position: Equatorial,
    pub profile: SersicProfile,
    pub flux: SourceFlux,
}

/// Project each [`Galaxy`] onto every sensor whose extent (plus a
/// per-sensor halo-aware padding equal to the galaxy's Sérsic
/// footprint) contains the projected centre.
///
/// Returns a per-sensor `Vec<GalaxyInFrame>` list ready to be fed to
/// the motion-blur renderer's per-tile splat loop. Galaxies whose
/// halo subtends multiple sensors appear in the lists of every
/// sensor they touch; galaxies whose entire footprint falls outside
/// every sensor are dropped.
///
/// The `orientation` argument is the spacecraft attitude under which
/// to project — typically the trajectory's mid-frame pose. Per-frame
/// re-projection (so galaxies follow the camera through a moving
/// trajectory) is a future improvement; for now galaxies are
/// projected once per render call.
pub fn project_galaxies_to_sensors(
    galaxies: &[Galaxy],
    fp: &FocalPlaneConfig,
    orientation: &UnitQuaternion<f64>,
) -> Vec<Vec<GalaxyInFrame>> {
    let n_sensors = fp.array.sensor_count();
    let mut per_sensor: Vec<Vec<GalaxyInFrame>> = vec![Vec::new(); n_sensors];

    for galaxy in galaxies {
        for (sensor_idx, sensor_list) in per_sensor.iter_mut().enumerate() {
            let sat = match fp.satellite_for_sensor(sensor_idx) {
                Some(s) => s,
                None => continue,
            };
            let plate_scale_arcsec_per_px = sat.plate_scale_arcsec_per_pixel();
            let deposit = SersicSplat::new(galaxy.profile, plate_scale_arcsec_per_px);
            let footprint_px = deposit.footprint_pixels() as f64;
            let pixel_size_mm = sat.sensor.pixel_size().as_millimeters();
            let padding_mm = footprint_px * pixel_size_mm;

            let probe = StarData::with_position(galaxy.id, galaxy.position, 0.0, None);
            let (px, py) = match fp.project_to_sensor(&probe, orientation, sensor_idx, padding_mm) {
                Some(p) => p,
                None => continue,
            };

            // Final AABB test against the (un-padded) sensor extent:
            // a galaxy whose centre lies within the padded region but
            // whose footprint AABB does not actually intersect any
            // pixel contributes nothing and should be dropped.
            let (w, h) = sat.sensor.dimensions.get_pixel_width_height();
            if px + footprint_px < 0.0
                || px - footprint_px >= w as f64
                || py + footprint_px < 0.0
                || py - footprint_px >= h as f64
            {
                continue;
            }

            sensor_list.push(GalaxyInFrame {
                x: px,
                y: py,
                position: galaxy.position,
                id: galaxy.id,
                name: galaxy.name.clone(),
                flux: galaxy.flux.clone(),
                deposit,
            });
        }
    }
    per_sensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_proc::deposit::render_sources;
    use crate::photometry::photoconversion::SpotFlux;
    use crate::sims::motion_blur::SensorAccumulator;
    use ndarray::Array2;
    use shared::image_proc::airy::PixelScaledAiryDisk;
    use shared::units::{AreaExt, LengthExt, Wavelength};
    use starfield::catalogs::SersicProfile;
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
        // De Vaucouleurs-like (n=4) — used by the byte-equality and
        // additivity tests where the deposit just needs to be a
        // realistic galaxy shape.
        fixture_galaxy_full(x, y, electron_rate, arcsec_per_pixel, 4.0, 5.0)
    }

    fn fixture_galaxy_full(
        x: f64,
        y: f64,
        electron_rate: f64,
        arcsec_per_pixel: f64,
        n: f64,
        theta_half_arcsec: f64,
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
            theta_half_arcsec,
            n,
            axis_ratio: 0.6,
            position_angle_deg: 30.0,
        };
        let deposit = SersicSplat::new(profile, arcsec_per_pixel);
        GalaxyInFrame {
            x,
            y,
            position: Equatorial::from_degrees(0.0, 0.0),
            id: 42,
            name: None,
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
    /// Build a small (256×256 px) test sensor + satellite config.
    /// Tiny on purpose so the `Renderer::from_stars_and_galaxies` base
    /// image fits in ~64k pixels rather than 16M for a real CMOS — the
    /// flux-conservation physics is sensor-size-invariant, and the
    /// reduced size keeps the e2e test fast under coverage
    /// instrumentation.
    fn small_test_satellite() -> crate::hardware::SatelliteConfig {
        use crate::hardware::dark_current::DarkCurrentEstimator;
        use crate::hardware::read_noise::ReadNoiseEstimator;
        use crate::hardware::sensor::{SensorConfig, SensorGeometry};
        use crate::hardware::SatelliteConfig;
        use crate::hardware::TelescopeConfig;
        use crate::photometry::QuantumEfficiency;
        use shared::units::{Length, LengthExt as _, Temperature, TemperatureExt as _};
        let telescope = TelescopeConfig::new(
            "Tiny e2e telescope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let qe = QuantumEfficiency::from_table(
            vec![400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
            vec![0.0, 0.7, 0.9, 0.85, 0.7, 0.5, 0.0],
        )
        .unwrap();
        let sensor = SensorConfig {
            name: "TinyE2E".into(),
            quantum_efficiency: qe,
            dimensions: SensorGeometry::of_width_height(256, 256, Length::from_micrometers(5.5)),
            read_noise_estimator: ReadNoiseEstimator::constant(2.0),
            dark_current_estimator: DarkCurrentEstimator::from_reference_point(
                0.01,
                Temperature::from_celsius(20.0),
            ),
            bit_depth: 16,
            dn_per_electron: 1.0,
            max_well_depth_e: 1e20,
            black_level_dn: 0,
            max_frame_rate_fps: 30.0,
        };
        SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(-10.0))
    }

    #[test]
    fn end_to_end_flux_roundtrip_through_renderer() {
        use crate::image_proc::render::Renderer;
        use crate::photometry::zodiacal::SolarAngularCoordinates;

        let satellite = small_test_satellite();
        let aperture = satellite.telescope.clear_aperture_area();

        // Place a galaxy at a centred sub-pixel position. Use n=1
        // (exponential disk, not de Vaucouleurs) so the 1e-4 SB
        // footprint is ~5 θ_eff rather than the ~30 θ_eff of n=4 —
        // fits comfortably inside a 256×256 sensor at 0.1″/pix even
        // with θ_eff = 2″ (20 px per θ_eff, well-resolved).
        let (w, h) = satellite.sensor.dimensions.get_pixel_width_height();
        let cx = (w as f64) * 0.5;
        let cy = (h as f64) * 0.5;
        let electron_rate = 5_000.0_f64; // electrons / s / cm²
        let g = fixture_galaxy_full(cx, cy, electron_rate, 0.1, 1.0, 2.0);
        let exposure = Duration::from_secs(2);
        let total_in = g.flux.electrons.integrated_over(&exposure, aperture);

        // Render via the static path with Poisson disabled — the
        // `star_image` is the *mean* electron buffer with both stars
        // (empty) and galaxies splatted in, scaled by exposure.
        let renderer = Renderer::from_stars_and_galaxies(&[], &[g], satellite);
        let zodi = SolarAngularCoordinates::new(180.0, 60.0).unwrap();
        let result = renderer.render_with_options(&exposure, &zodi, false, None);
        let total_out: f64 = result.star_image.iter().sum();

        // Expect within the SersicSplat truncation budget: > 95% of
        // input flux captured, never exceeding input (Simpson sub-
        // sample is a bounded integrator).
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
        use crate::image_proc::render::Renderer;
        use crate::photometry::zodiacal::SolarAngularCoordinates;
        use ndarray::Array2;

        let satellite = small_test_satellite();
        let (w, h) = satellite.sensor.dimensions.get_pixel_width_height();
        let cx = (w as f64) * 0.5;
        let cy = (h as f64) * 0.5;
        // Lower electron rate keeps per-pixel mean small enough that
        // the Poisson identity is detectable; high rate gets dominated
        // by the variance of the variance estimator.
        // Same n=1 / θ_eff = 2″ shape as the e2e roundtrip — see
        // there for why n=1 specifically (footprint sized to 256 sensor).
        let g = fixture_galaxy_full(cx, cy, 50.0, 0.1, 1.0, 2.0);
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

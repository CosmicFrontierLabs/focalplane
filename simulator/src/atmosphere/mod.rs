//! A molecular (Rayleigh) atmosphere around a body, single scattering.
//!
//! The visible limb of Earth is not its surface: along a tangent ray the
//! Rayleigh optical depth is about 71 × the vertical value, so the ray
//! reaches unit optical depth some 15 km above the ground at 550 nm (22 km
//! at 450 nm) and a blue-weighted glow extends past the geometric disk
//! and past the terminator. Broadband cameras detect the limb higher
//! still, ~25 km for Artemis I, once aerosols, multiple scattering and
//! the detection threshold are included. A guider that fits the limb
//! fits *this*, so the renderer has to produce it.
//!
//! # Model
//!
//! - Air density falls exponentially with height: `n(h) = n₀ e^{−h/H}`
//!   with `H` = [`RayleighAtmosphere::scale_height_km`]; nothing above
//!   [`RayleighAtmosphere::top_km`].
//! - Scattering coefficient `β(λ, h) = (τ_R(λ)/H) e^{−h/H}` where
//!   `τ_R(λ)` is the sea-level vertical Rayleigh optical depth from the
//!   Hansen & Travis (1974) fit, `τ_R = 0.008569 λ⁻⁴ (1 + 0.0113 λ⁻² +
//!   0.00013 λ⁻⁴)` for `λ` in µm: 0.0973 at 550 nm, within 1 % of
//!   Bodhaine et al. (1999) across the visible.
//! - Rayleigh phase function `P(Θ) = 3/(16π) (1 + cos²Θ)`, normalised to
//!   unit integral over the sphere.
//! - Single scattering only. For one ray toward the observer,
//!
//!   ```text
//!   T_view      = exp(−∫ β ds)                          along the ray
//!   S           = ∫ β(s) P(Θ) T_sun(s) T_view(s→obs) ds  path radiance / E☉  [sr⁻¹]
//!   T_sun(s)    = 0 if the ground shadows the point, else exp(−∫ β ds′) toward the Sun
//!   ```
//!
//!   so a surface element of reflectance `r` seen through the atmosphere
//!   contributes `r · T_sun(ground) · T_view(ground) + S`, in the same
//!   "radiance per unit solar irradiance" units the stamp uses.
//! - Spectral integration: the sensor band is split into bins weighted
//!   by `F☉(λ) Q(λ) λ`; each bin has its own `β`, and the band values of
//!   `T` and `S` are the weighted means. Multiplying a band-mean
//!   transmittance by a band-mean surface reflectance is exact for a grey
//!   surface and a small approximation otherwise.
//!
//! Not modelled here: multiple scattering (adds tens of percent to the
//! limb glow and softens the terminator), aerosols (Mie forward
//! scattering, the bright cusp extension of thin crescents), ozone
//! (Chappuis absorption, darkens the limb in the yellow-red), refraction.
//! Each is a documented extension point in the plan.

use std::f64::consts::PI;

use nalgebra::Vector3;
use shared::units::{LengthExt, Wavelength};
use thiserror::Error;

use crate::photometry::quantum_efficiency::QuantumEfficiency;
use crate::photometry::solar::TsisSolarSpectrum;

/// Sea-level vertical Rayleigh optical depth, Hansen & Travis (1974),
/// `lambda_um` in micrometres.
pub fn rayleigh_optical_depth(lambda_um: f64) -> f64 {
    let inv2 = 1.0 / (lambda_um * lambda_um);
    let inv4 = inv2 * inv2;
    0.008_569 * inv4 * (1.0 + 0.0113 * inv2 + 0.000_13 * inv4)
}

/// Rayleigh phase function for scattering angle cosine `cos_theta`,
/// sr⁻¹, unit integral over the sphere.
pub fn rayleigh_phase(cos_theta: f64) -> f64 {
    3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta)
}

/// Exponential Rayleigh atmosphere parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct RayleighAtmosphere {
    /// Body radius the heights are measured from, km.
    pub radius_km: f64,
    /// Density scale height, km.
    pub scale_height_km: f64,
    /// Height above which the atmosphere is ignored, km.
    pub top_km: f64,
    /// Multiplier on the sea-level optical depth (1.0 = Earth at
    /// 1013.25 hPa; use the surface-pressure ratio for other bodies).
    pub column_scale: f64,
    /// Steps along the view ray.
    pub view_steps: usize,
    /// Steps along each sunward ray.
    pub sun_steps: usize,
}

impl RayleighAtmosphere {
    /// Earth: 8 km scale height, 100 km top, unit column.
    pub fn earth(radius_km: f64) -> Self {
        Self {
            radius_km,
            scale_height_km: 8.0,
            top_km: 100.0,
            column_scale: 1.0,
            view_steps: 32,
            sun_steps: 12,
        }
    }

    /// Outer radius of the modelled shell, km.
    pub fn top_radius_km(&self) -> f64 {
        self.radius_km + self.top_km
    }

    /// Scattering coefficient at height `h_km` and wavelength `lambda_nm`,
    /// km⁻¹.
    pub fn beta(&self, lambda_nm: f64, h_km: f64) -> f64 {
        if h_km < 0.0 || h_km > self.top_km {
            return 0.0;
        }
        self.column_scale * rayleigh_optical_depth(lambda_nm / 1000.0) / self.scale_height_km
            * (-h_km / self.scale_height_km).exp()
    }

    /// Vertical optical depth above height `h_km` at `lambda_nm`.
    pub fn vertical_optical_depth(&self, lambda_nm: f64, h_km: f64) -> f64 {
        self.column_scale
            * rayleigh_optical_depth(lambda_nm / 1000.0)
            * ((-h_km / self.scale_height_km).exp() - (-self.top_km / self.scale_height_km).exp())
    }

    /// Optical depth along the straight segment from `from` to `to`
    /// (body-centred coordinates, km) at `lambda_nm`, by midpoint
    /// quadrature with `steps` steps.
    pub fn segment_optical_depth(
        &self,
        lambda_nm: f64,
        from: Vector3<f64>,
        to: Vector3<f64>,
        steps: usize,
    ) -> f64 {
        let d = to - from;
        let length = d.norm();
        if length == 0.0 || steps == 0 {
            return 0.0;
        }
        let ds = length / steps as f64;
        let mut tau = 0.0;
        for i in 0..steps {
            let p = from + d * ((i as f64 + 0.5) / steps as f64);
            tau += self.beta(lambda_nm, p.norm() - self.radius_km) * ds;
        }
        tau
    }

    /// Where a ray `origin + t·dir` (unit `dir`) leaves the shell, i.e.
    /// the larger root of `|origin + t dir| = top_radius`, or `None` if
    /// the ray never touches the shell.
    fn exit_parameter(&self, origin: Vector3<f64>, dir: Vector3<f64>) -> Option<f64> {
        let r_top = self.top_radius_km();
        let b = origin.dot(&dir);
        let c = origin.dot(&origin) - r_top * r_top;
        let disc = b * b - c;
        if disc < 0.0 {
            return None;
        }
        Some(-b + disc.sqrt())
    }

    /// Transmittance from `point` toward the Sun (unit `sun_dir`), zero if
    /// the ground blocks the sunward ray.
    pub fn sun_transmittance(
        &self,
        lambda_nm: f64,
        point: Vector3<f64>,
        sun_dir: Vector3<f64>,
    ) -> f64 {
        // Ground shadow: closest approach of the sunward ray to the centre
        // lies ahead of the point and inside the body.
        let t_closest = -point.dot(&sun_dir);
        if t_closest > 0.0 {
            let closest = point + sun_dir * t_closest;
            if closest.norm() < self.radius_km {
                return 0.0;
            }
        }
        let Some(t_exit) = self.exit_parameter(point, sun_dir) else {
            return 1.0;
        };
        if t_exit <= 0.0 {
            return 1.0;
        }
        let tau =
            self.segment_optical_depth(lambda_nm, point, point + sun_dir * t_exit, self.sun_steps);
        (-tau).exp()
    }

    /// Single-scattering result for one ray at one wavelength.
    ///
    /// The ray runs from `entry` toward the observer along unit `view_dir`
    /// and ends at `exit` (the shell top on the observer side). `entry` is
    /// either the ground point or the far shell boundary.
    pub fn evaluate_ray(
        &self,
        lambda_nm: f64,
        entry: Vector3<f64>,
        exit: Vector3<f64>,
        view_dir: Vector3<f64>,
        sun_dir: Vector3<f64>,
    ) -> RaySample {
        let d = exit - entry;
        let length = d.norm();
        if length == 0.0 {
            return RaySample {
                transmittance: 1.0,
                path_radiance: 0.0,
            };
        }
        let ds = length / self.view_steps as f64;
        // Scattering angle between the incident sunlight (−sun_dir) and
        // the scattered direction (view_dir).
        let cos_theta = -sun_dir.dot(&view_dir);
        let phase = rayleigh_phase(cos_theta);

        // March from the observer side inward so the transmittance to the
        // observer accumulates as we go.
        let mut tau_to_observer = 0.0;
        let mut path_radiance = 0.0;
        for i in 0..self.view_steps {
            let frac = 1.0 - (i as f64 + 0.5) / self.view_steps as f64;
            let p = entry + d * frac;
            let beta = self.beta(lambda_nm, p.norm() - self.radius_km);
            tau_to_observer += 0.5 * beta * ds;
            let t_view = (-tau_to_observer).exp();
            let t_sun = self.sun_transmittance(lambda_nm, p, sun_dir);
            path_radiance += beta * phase * t_sun * t_view * ds;
            tau_to_observer += 0.5 * beta * ds;
        }
        RaySample {
            transmittance: (-tau_to_observer).exp(),
            path_radiance,
        }
    }
}

/// Result of a single-scattering ray march at one wavelength.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaySample {
    /// Transmittance from the ray's entry point to the observer.
    pub transmittance: f64,
    /// Path radiance per unit solar irradiance, sr⁻¹.
    pub path_radiance: f64,
}

/// A ray through the atmosphere in the stamp's sky frame: `x` east,
/// `y` north, `z` toward the observer, lengths in km, origin at the
/// body centre.
#[derive(Clone, Debug, PartialEq)]
pub struct SkyRay {
    /// Impact-parameter offset of the ray from the body centre, km,
    /// `(east, north)`.
    pub offset_km: (f64, f64),
    /// Unit vector toward the Sun in the sky frame.
    pub sun_dir: Vector3<f64>,
}

/// Band-integrated atmospheric radiance and transmittance along sky
/// rays. Bound to one sensor band.
pub trait AtmosphereRadiance: Send + Sync {
    /// Outer radius of the shell, km.
    fn top_radius_km(&self) -> f64;

    /// Band-mean transmittances and path radiance for a ray at
    /// `offset_km` that **hits the ground** at radius `radius_km`
    /// (impact parameter below the surface).
    fn ground_ray(&self, ray: &SkyRay) -> GroundSample;

    /// Band-mean transmittance and path radiance for a ray that misses
    /// the ground and crosses the shell.
    fn limb_ray(&self, ray: &SkyRay) -> RaySample;
}

/// Atmospheric terms for a ray that reaches the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundSample {
    /// Transmittance from the ground point to the observer.
    pub view_transmittance: f64,
    /// Transmittance from the Sun to the ground point (zero at night).
    pub sun_transmittance: f64,
    /// Path radiance per unit solar irradiance between ground and
    /// observer, sr⁻¹.
    pub path_radiance: f64,
}

/// Errors binding an atmosphere to a sensor band.
#[derive(Debug, Error)]
pub enum AtmosphereError {
    /// The sensor band leaves the solar table's coverage.
    #[error("sensor band {lo_nm:.0}–{hi_nm:.0} nm is not covered by the solar spectrum")]
    UncoveredBand { lo_nm: f64, hi_nm: f64 },
}

/// A [`RayleighAtmosphere`] bound to a sensor band: a handful of
/// spectral bins with response weights.
#[derive(Clone, Debug)]
pub struct BoundRayleigh {
    atmosphere: RayleighAtmosphere,
    /// `(bin centre nm, weight)`, weights summing to 1.
    bins: Vec<(f64, f64)>,
}

impl BoundRayleigh {
    /// Split the QE support into `n_bins` equal-width bins weighted by
    /// `∫ F☉ Q λ dλ`.
    pub fn bind(
        atmosphere: RayleighAtmosphere,
        qe: &QuantumEfficiency,
        solar: &TsisSolarSpectrum,
        n_bins: usize,
    ) -> Result<Self, AtmosphereError> {
        let band = qe.band();
        let (lo, hi) = (band.lower_nm, band.upper_nm);
        solar
            .check_covers(&band)
            .map_err(|_| AtmosphereError::UncoveredBand {
                lo_nm: lo,
                hi_nm: hi,
            })?;
        let n = n_bins.max(1);
        let width = (hi - lo) / n as f64;
        let mut bins = Vec::with_capacity(n);
        let mut total = 0.0;
        for i in 0..n {
            let b_lo = lo + width * i as f64;
            let b_hi = b_lo + width;
            let w = solar
                .table()
                .weighted_irradiance(b_lo, b_hi, |nm| qe.at(Wavelength::from_nanometers(nm)) * nm)
                .unwrap_or(0.0);
            total += w;
            bins.push((0.5 * (b_lo + b_hi), w));
        }
        if total <= 0.0 {
            return Err(AtmosphereError::UncoveredBand {
                lo_nm: lo,
                hi_nm: hi,
            });
        }
        for bin in &mut bins {
            bin.1 /= total;
        }
        Ok(Self { atmosphere, bins })
    }

    /// The underlying model.
    pub fn atmosphere(&self) -> &RayleighAtmosphere {
        &self.atmosphere
    }

    /// Spectral bins `(centre nm, weight)`.
    pub fn bins(&self) -> &[(f64, f64)] {
        &self.bins
    }

    /// Points where the view ray enters and exits the shell (or ground),
    /// in the sky frame.
    fn ray_segment(&self, ray: &SkyRay) -> Option<(Vector3<f64>, Vector3<f64>, bool)> {
        let a = &self.atmosphere;
        let (x, y) = ray.offset_km;
        let rho2 = x * x + y * y;
        let r_top = a.top_radius_km();
        if rho2 >= r_top * r_top {
            return None;
        }
        let z_top = (r_top * r_top - rho2).sqrt();
        let exit = Vector3::new(x, y, z_top);
        let r = a.radius_km;
        if rho2 < r * r {
            let z_ground = (r * r - rho2).sqrt();
            Some((Vector3::new(x, y, z_ground), exit, true))
        } else {
            Some((Vector3::new(x, y, -z_top), exit, false))
        }
    }
}

impl AtmosphereRadiance for BoundRayleigh {
    fn top_radius_km(&self) -> f64 {
        self.atmosphere.top_radius_km()
    }

    fn ground_ray(&self, ray: &SkyRay) -> GroundSample {
        let Some((ground, exit, hits)) = self.ray_segment(ray) else {
            return GroundSample {
                view_transmittance: 1.0,
                sun_transmittance: 1.0,
                path_radiance: 0.0,
            };
        };
        debug_assert!(hits, "ground_ray called for a ray that misses the ground");
        let view_dir = Vector3::z();
        let mut out = GroundSample {
            view_transmittance: 0.0,
            sun_transmittance: 0.0,
            path_radiance: 0.0,
        };
        for &(nm, w) in &self.bins {
            let sample = self
                .atmosphere
                .evaluate_ray(nm, ground, exit, view_dir, ray.sun_dir);
            out.view_transmittance += w * sample.transmittance;
            out.sun_transmittance += w * self.atmosphere.sun_transmittance(nm, ground, ray.sun_dir);
            out.path_radiance += w * sample.path_radiance;
        }
        out
    }

    fn limb_ray(&self, ray: &SkyRay) -> RaySample {
        let Some((entry, exit, hits)) = self.ray_segment(ray) else {
            return RaySample {
                transmittance: 1.0,
                path_radiance: 0.0,
            };
        };
        debug_assert!(!hits, "limb_ray called for a ray that hits the ground");
        let view_dir = Vector3::z();
        let mut out = RaySample {
            transmittance: 0.0,
            path_radiance: 0.0,
        };
        for &(nm, w) in &self.bins {
            let sample = self
                .atmosphere
                .evaluate_ray(nm, entry, exit, view_dir, ray.sun_dir);
            out.transmittance += w * sample.transmittance;
            out.path_radiance += w * sample.path_radiance;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::sensor::create_flat_qe;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    const R_EARTH: f64 = 6378.1366;

    #[test]
    fn rayleigh_optical_depth_matches_reference_values() {
        // Bodhaine et al. (1999) sea-level values: 0.097 at 550 nm,
        // ~0.36 at 400 nm, ~0.0135 at 900 nm (λ⁻⁴ scaling of the 550 nm
        // value gives 0.0136).
        assert_relative_eq!(rayleigh_optical_depth(0.55), 0.0973, max_relative = 0.01);
        assert_relative_eq!(rayleigh_optical_depth(0.40), 0.36, max_relative = 0.03);
        assert_relative_eq!(rayleigh_optical_depth(0.90), 0.0135, max_relative = 0.03);
    }

    #[test]
    fn rayleigh_phase_integrates_to_one_over_the_sphere() {
        let n = 20_000;
        let d = PI / n as f64;
        let integral: f64 = (0..n)
            .map(|k| {
                let theta = (k as f64 + 0.5) * d;
                2.0 * PI * rayleigh_phase(theta.cos()) * theta.sin() * d
            })
            .sum();
        assert_abs_diff_eq!(integral, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn vertical_optical_depth_from_the_ground_is_the_sea_level_value() {
        let atm = RayleighAtmosphere::earth(R_EARTH);
        assert_relative_eq!(
            atm.vertical_optical_depth(550.0, 0.0),
            rayleigh_optical_depth(0.55),
            max_relative = 1e-4
        );
        // Numerical vertical integral agrees with the closed form.
        let numeric = atm.segment_optical_depth(
            550.0,
            Vector3::new(0.0, 0.0, R_EARTH),
            Vector3::new(0.0, 0.0, R_EARTH + 100.0),
            4000,
        );
        assert_relative_eq!(
            numeric,
            atm.vertical_optical_depth(550.0, 0.0),
            max_relative = 1e-3
        );
    }

    #[test]
    fn tangent_ray_optical_depth_follows_the_chapman_factor() {
        // A ray grazing at tangent height h has optical depth
        // τ_vert(h) · sqrt(2π (R+h) / H) for an exponential atmosphere.
        let atm = RayleighAtmosphere::earth(R_EARTH);
        for &h in &[0.0_f64, 20.0, 40.0] {
            let r = R_EARTH + h;
            let half_chord = ((R_EARTH + atm.top_km).powi(2) - r * r).sqrt();
            let tau = atm.segment_optical_depth(
                550.0,
                Vector3::new(r, 0.0, -half_chord),
                Vector3::new(r, 0.0, half_chord),
                8000,
            );
            let chapman =
                atm.vertical_optical_depth(550.0, h) * (2.0 * PI * r / atm.scale_height_km).sqrt();
            assert_relative_eq!(tau, chapman, max_relative = 0.03);
        }
    }

    #[test]
    fn the_visible_limb_sits_tens_of_kilometres_up() {
        // Tangent height where the grazing-ray optical depth is 1. The
        // grazing depth at the ground is τ_vert · √(2πR/H) ≈ 0.097 × 71 ≈
        // 6.9 at 550 nm, so τ = 1 falls at H ln 6.9 ≈ 15 km; ~22 km at
        // 450 nm and ~6 km at 750 nm. (Detected limbs in broadband
        // cameras sit higher, ~25 km for Artemis I, because aerosols,
        // multiple scattering and the detection threshold add to this.)
        let atm = RayleighAtmosphere::earth(R_EARTH);
        let limb_height = |nm: f64| {
            let mut lo = 0.0_f64;
            let mut hi = 100.0_f64;
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                let r = R_EARTH + mid;
                let half = ((R_EARTH + atm.top_km).powi(2) - r * r).sqrt();
                let tau = atm.segment_optical_depth(
                    nm,
                    Vector3::new(r, 0.0, -half),
                    Vector3::new(r, 0.0, half),
                    2000,
                );
                if tau > 1.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };
        let h550 = limb_height(550.0);
        let h450 = limb_height(450.0);
        let h750 = limb_height(750.0);
        assert!(
            (13.0..18.0).contains(&h550),
            "τ=1 tangent height {h550:.1} km at 550 nm"
        );
        assert!(
            h450 > h550 + 4.0,
            "blue limb {h450:.1} km should be higher than {h550:.1} km"
        );
        assert!(
            h750 < h550 - 5.0,
            "red limb {h750:.1} km should be lower than {h550:.1} km"
        );
    }

    fn bound() -> BoundRayleigh {
        let solar = TsisSolarSpectrum::load().unwrap();
        let qe = create_flat_qe(0.5);
        BoundRayleigh::bind(RayleighAtmosphere::earth(R_EARTH), &qe, &solar, 8).unwrap()
    }

    #[test]
    fn bins_are_normalised_and_blue_bins_scatter_more() {
        let b = bound();
        let total: f64 = b.bins().iter().map(|(_, w)| w).sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-12);
        let atm = b.atmosphere();
        let first = b.bins()[0].0;
        let last = b.bins()[b.bins().len() - 1].0;
        assert!(atm.beta(first, 0.0) > 4.0 * atm.beta(last, 0.0));
    }

    #[test]
    fn sub_solar_ground_ray_is_attenuated_and_brightened_by_path_radiance() {
        // Observer and Sun both along +z: nadir view of the sub-solar
        // point at zero phase.
        let b = bound();
        let ray = SkyRay {
            offset_km: (0.0, 0.0),
            sun_dir: Vector3::z(),
        };
        let g = b.ground_ray(&ray);
        // Two-way transmittance well below 1 but above exp(−2·0.4).
        assert!(
            g.view_transmittance > 0.6 && g.view_transmittance < 0.97,
            "{g:?}"
        );
        assert!(g.sun_transmittance > 0.6 && g.sun_transmittance < 0.97);
        // Backscattered path radiance of a Rayleigh column: ~τ·P(180°)
        // for small τ, i.e. a few times 1e-2 sr⁻¹ blue-weighted.
        assert!(g.path_radiance > 0.005 && g.path_radiance < 0.05, "{g:?}");
    }

    #[test]
    fn night_side_ground_ray_has_no_sunlight_but_the_limb_glows_past_the_terminator() {
        let b = bound();
        // Sun along −z (behind the body): the whole facing hemisphere is
        // night, so the ground gets no sunlight.
        let night = SkyRay {
            offset_km: (0.0, 0.0),
            sun_dir: -Vector3::z(),
        };
        let g = b.ground_ray(&night);
        assert_eq!(g.sun_transmittance, 0.0);
        assert!(
            g.path_radiance < 1e-6,
            "no single scattering in full shadow: {g:?}"
        );
        // A grazing ray 20 km above the limb still sees sunlit air on the
        // far side of the terminator: forward-scattered twilight.
        let limb = SkyRay {
            offset_km: (R_EARTH + 20.0, 0.0),
            sun_dir: -Vector3::z(),
        };
        let l = b.limb_ray(&limb);
        assert!(l.path_radiance > 1e-4, "twilight limb should glow: {l:?}");
        assert!(l.transmittance > 0.0 && l.transmittance < 1.0);
    }

    #[test]
    fn limb_glow_fades_with_tangent_height_and_vanishes_above_the_shell() {
        let b = bound();
        let sun = Vector3::new(1.0, 0.0, 0.0);
        let glow = |h: f64| {
            b.limb_ray(&SkyRay {
                offset_km: (0.0, R_EARTH + h),
                sun_dir: sun,
            })
        };
        let low = glow(10.0);
        let mid = glow(40.0);
        let high = glow(90.0);
        assert!(low.path_radiance > mid.path_radiance && mid.path_radiance > high.path_radiance);
        assert!(low.transmittance < mid.transmittance && mid.transmittance < high.transmittance);
        let above = b.limb_ray(&SkyRay {
            offset_km: (0.0, R_EARTH + 200.0),
            sun_dir: sun,
        });
        assert_eq!(above.transmittance, 1.0);
        assert_eq!(above.path_radiance, 0.0);
    }

    #[test]
    fn limb_is_opaque_at_the_ground_and_clear_high_up() {
        let b = bound();
        let sun = Vector3::x();
        // Band-mean transmittance: the red bins leak (τ ≈ 1.5 at 800 nm)
        // while the blue ones are black, so the mean sits a few percent.
        let grazing = b.limb_ray(&SkyRay {
            offset_km: (R_EARTH + 0.5, 0.0),
            sun_dir: sun,
        });
        assert!(grazing.transmittance < 0.1, "{grazing:?}");
        let clear = b.limb_ray(&SkyRay {
            offset_km: (R_EARTH + 80.0, 0.0),
            sun_dir: sun,
        });
        assert!(clear.transmittance > 0.99, "{clear:?}");
    }
}

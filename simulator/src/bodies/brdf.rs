//! Bidirectional reflectance models for planetary surfaces.
//!
//! # Convention
//!
//! Every model returns the **bidirectional reflectance**
//! `r(μ₀, μ, α) = L / E` in sr⁻¹, where `L` is the radiance leaving the
//! surface toward the observer and `E` is the solar irradiance on a
//! plane normal to the Sun at the body. Radiance in any band is then
//! `L(λ) = r · F☉(λ) / d☉²` with `F☉` the irradiance at 1 AU and `d☉`
//! the heliocentric distance in AU.
//!
//! Arguments:
//!
//! - `μ₀ = cos i`, cosine of the incidence angle (surface normal to Sun)
//! - `μ  = cos e`, cosine of the emission angle (surface normal to observer)
//! - `α`, the phase angle in radians (Sun–surface–observer)
//!
//! Models return `0` when either cosine is non-positive (unlit or
//! hidden). A Lambert surface of albedo `A` has `r = A μ₀ / π`, so a
//! Lambert sphere's geometric albedo is `2A/3`; a Lommel–Seeliger
//! sphere's is `w/8`. Both are locked by tests via
//! [`disk_integrated_reflectance`].
//!
//! # Models
//!
//! | Model | Physics | Typical use |
//! |---|---|---|
//! | [`Lambert`] | isotropic diffuse | clouds, ice, quick looks |
//! | [`LommelSeeliger`] | single scattering in a dark particulate | Moon, Mercury, Mars quick-look |
//! | [`Minnaert`] | empirical limb-darkening exponent | Venus, gas giants |
//! | [`Hapke`] | radiative-transfer model with opposition surge, particle phase function and macroscopic roughness (Hapke 1984, 2002, 2012) | Moon, Mercury, Mars with published parameter sets |
//!
//! References: Hapke, B. (2012) *Theory of Reflectance and Emittance
//! Spectroscopy*, 2nd ed., chapters 8–12; Hapke (1984) Icarus 59, 41
//! (roughness); Hapke (2002) Icarus 157, 523 (H-function approximation);
//! Sato et al. (2014) JGR Planets 119, 1775 (lunar parameter maps).

use std::f64::consts::PI;

/// A surface reflectance law. See the module docs for the convention.
pub trait Brdf: Send + Sync {
    /// Bidirectional reflectance in sr⁻¹ for incidence cosine `mu0`,
    /// emission cosine `mu` and phase angle `alpha` (radians).
    fn reflectance(&self, mu0: f64, mu: f64, alpha: f64) -> f64;
}

/// Ideal diffuse reflector: `r = A μ₀ / π`.
///
/// `albedo` is the Lambert (normal) albedo, equal to the Bond albedo
/// of a Lambert sphere. Geometric albedo is `2A/3`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lambert {
    /// Lambert albedo `A` in `[0, 1]`.
    pub albedo: f64,
}

impl Brdf for Lambert {
    fn reflectance(&self, mu0: f64, mu: f64, _alpha: f64) -> f64 {
        if mu0 <= 0.0 || mu <= 0.0 {
            return 0.0;
        }
        self.albedo * mu0 / PI
    }
}

/// Lommel–Seeliger law: `r = (w / 4π) · μ₀ / (μ₀ + μ) · P(α)`.
///
/// Single scattering from a semi-infinite particulate medium with
/// single-scattering albedo `w` and particle phase function `P(α)`
/// (normalised so `P = 1` is isotropic). It has no limb darkening at
/// zero phase, which matches the flat appearance of the full Moon.
/// Geometric albedo of a sphere with `P = 1` is `w/8`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LommelSeeliger {
    /// Single-scattering albedo `w` in `[0, 1]`.
    pub single_scattering_albedo: f64,
    /// Particle phase function; `None` means isotropic.
    pub phase_function: Option<DoubleHenyeyGreenstein>,
}

impl LommelSeeliger {
    /// Isotropic Lommel–Seeliger with single-scattering albedo `w`.
    pub fn isotropic(single_scattering_albedo: f64) -> Self {
        Self {
            single_scattering_albedo,
            phase_function: None,
        }
    }
}

impl Brdf for LommelSeeliger {
    fn reflectance(&self, mu0: f64, mu: f64, alpha: f64) -> f64 {
        if mu0 <= 0.0 || mu <= 0.0 {
            return 0.0;
        }
        let p = self.phase_function.map_or(1.0, |pf| pf.evaluate(alpha));
        self.single_scattering_albedo / (4.0 * PI) * mu0 / (mu0 + mu) * p
    }
}

/// Minnaert law: `r = (B₀ / π) · μ₀ᵏ · μᵏ⁻¹`.
///
/// Empirical limb-darkening law. `k = 1` is Lambert with albedo `B₀`;
/// `k ≈ 0.5` gives the near-uniform disk of a Lommel–Seeliger-like
/// surface; `k > 1` darkens the limb more strongly than Lambert, as the
/// cloud decks of Venus and the giant planets do in the visible.
/// Reciprocity (`r(μ₀, μ) = r(μ, μ₀)`) does not hold unless `k = 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Minnaert {
    /// Minnaert albedo `B₀`: the value of `π r` at normal incidence and
    /// emission.
    pub albedo: f64,
    /// Limb-darkening exponent `k`.
    pub k: f64,
}

impl Brdf for Minnaert {
    fn reflectance(&self, mu0: f64, mu: f64, _alpha: f64) -> f64 {
        if mu0 <= 0.0 || mu <= 0.0 {
            return 0.0;
        }
        self.albedo / PI * mu0.powf(self.k) * mu.powf(self.k - 1.0)
    }
}

/// Two-lobe Henyey–Greenstein particle phase function in the
/// phase-angle convention (Hapke 2012, eq. 6.7a):
///
/// `P(α) = (1+c)/2 · (1−b²)/(1 − 2b cos α + b²)^{3/2}
///        + (1−c)/2 · (1−b²)/(1 + 2b cos α + b²)^{3/2}`
///
/// `b ∈ [0, 1)` is the lobe width, `c` the backscatter fraction
/// parameter (`c = 1` purely backscattering, `c = −1` purely forward).
/// Normalised so that the average over the sphere is 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleHenyeyGreenstein {
    /// Asymmetry (lobe width) `b`, `0 ≤ b < 1`.
    pub b: f64,
    /// Backscatter fraction parameter `c`, `−1 ≤ c ≤ 1`.
    pub c: f64,
}

impl DoubleHenyeyGreenstein {
    /// Evaluate `P(α)` for phase angle `alpha` in radians.
    pub fn evaluate(&self, alpha: f64) -> f64 {
        let b = self.b;
        let cos_a = alpha.cos();
        let one_minus_b2 = 1.0 - b * b;
        let back = one_minus_b2 / (1.0 - 2.0 * b * cos_a + b * b).powf(1.5);
        let fwd = one_minus_b2 / (1.0 + 2.0 * b * cos_a + b * b).powf(1.5);
        0.5 * (1.0 + self.c) * back + 0.5 * (1.0 - self.c) * fwd
    }
}

/// Hapke photometric model (isotropic multiple scattering form).
///
/// `r = (w / 4π) · μ₀ₑ / (μ₀ₑ + μₑ) · [ (1 + B(α)) P(α) + H(μ₀ₑ) H(μₑ) − 1 ] · S(i, e, ψ)`
///
/// - `B(α) = B₀ / (1 + tan(α/2) / h)` is the shadow-hiding opposition
///   surge with amplitude `B₀` and angular width `h`.
/// - `P(α)` is the [`DoubleHenyeyGreenstein`] particle phase function.
/// - `H(x)` is the Chandrasekhar H-function in Hapke's (2002)
///   approximation, which carries the multiple-scattering term.
/// - `S` and the effective cosines `μ₀ₑ, μₑ` are the macroscopic
///   roughness correction of Hapke (1984) for mean slope angle `θ̄`;
///   they depend on the azimuth `ψ` between the incidence and emission
///   planes, recovered from `(μ₀, μ, α)`.
///
/// The coherent-backscatter opposition term is not included; published
/// parameter sets that fold it into `B₀`/`h` still reproduce
/// disk-integrated phase curves to a few percent at α > 2°.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hapke {
    /// Single-scattering albedo `w`.
    pub single_scattering_albedo: f64,
    /// Particle phase function.
    pub phase_function: DoubleHenyeyGreenstein,
    /// Opposition-surge amplitude `B₀`.
    pub opposition_amplitude: f64,
    /// Opposition-surge angular width `h` (radians-equivalent, as
    /// `tan(α/2)/h`).
    pub opposition_width: f64,
    /// Mean macroscopic slope angle `θ̄` in radians. `0` disables the
    /// roughness correction.
    pub roughness: f64,
}

impl Hapke {
    /// Lunar global-average parameters in the spirit of Sato et al.
    /// (2014) at 643 nm: `w = 0.33`, `b = 0.23`, `c = 0.30`,
    /// `B₀ = 1.5`, `h = 0.06`, `θ̄ = 23.4°`. Suitable for a disk-resolved
    /// Moon before per-pixel parameter maps are loaded.
    pub fn lunar_average() -> Self {
        Self {
            single_scattering_albedo: 0.33,
            phase_function: DoubleHenyeyGreenstein { b: 0.23, c: 0.30 },
            opposition_amplitude: 1.5,
            opposition_width: 0.06,
            roughness: 23.4_f64.to_radians(),
        }
    }

    /// Chandrasekhar H-function, Hapke (2002) approximation:
    /// `H(x) = 1 / (1 − w x [r₀ + (1 − 2 r₀ x)/2 · ln((1 + x)/x)])`
    /// with `γ = √(1 − w)`, `r₀ = (1 − γ)/(1 + γ)`.
    pub fn h_function(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let w = self.single_scattering_albedo;
        let gamma = (1.0 - w).max(0.0).sqrt();
        let r0 = (1.0 - gamma) / (1.0 + gamma);
        let ln_term = ((1.0 + x) / x).ln();
        1.0 / (1.0 - w * x * (r0 + 0.5 * (1.0 - 2.0 * r0 * x) * ln_term))
    }

    /// Shadow-hiding opposition surge `B(α)`.
    pub fn opposition_surge(&self, alpha: f64) -> f64 {
        if self.opposition_width <= 0.0 {
            return 0.0;
        }
        self.opposition_amplitude / (1.0 + (0.5 * alpha).tan() / self.opposition_width)
    }

    /// Roughness correction of Hapke (1984): returns
    /// `(μ₀ₑ, μₑ, S)` for incidence cosine `mu0`, emission cosine `mu`
    /// and phase angle `alpha`.
    pub fn roughness_correction(&self, mu0: f64, mu: f64, alpha: f64) -> (f64, f64, f64) {
        let theta_bar = self.roughness;
        if theta_bar <= 0.0 {
            return (mu0, mu, 1.0);
        }
        let mu0 = mu0.clamp(-1.0, 1.0);
        let mu = mu.clamp(-1.0, 1.0);
        let i = mu0.acos();
        let e = mu.acos();
        let sin_i = i.sin();
        let sin_e = e.sin();

        // Azimuth between the incidence and emission planes.
        let psi = if sin_i < 1e-12 || sin_e < 1e-12 {
            0.0
        } else {
            ((alpha.cos() - mu0 * mu) / (sin_i * sin_e))
                .clamp(-1.0, 1.0)
                .acos()
        };

        let tan_tb = theta_bar.tan();
        let cot_tb = 1.0 / tan_tb;
        let chi = 1.0 / (1.0 + PI * tan_tb * tan_tb).sqrt();
        let e1 = |x: f64| {
            if x.abs() < 1e-12 {
                0.0
            } else {
                (-(2.0 / PI) * cot_tb / x.tan()).exp()
            }
        };
        let e2 = |x: f64| {
            if x.abs() < 1e-12 {
                0.0
            } else {
                let cot_x = 1.0 / x.tan();
                (-(1.0 / PI) * cot_tb * cot_tb * cot_x * cot_x).exp()
            }
        };
        let eta = |x: f64| chi * (x.cos() + x.sin() * tan_tb * e2(x) / (2.0 - e1(x)));
        let f_psi = (-2.0 * (0.5 * psi).tan()).exp();
        let sin2_half_psi = (0.5 * psi).sin().powi(2);

        let (mu0e, mue, s) = if e >= i {
            let denom = 2.0 - e1(e) - (psi / PI) * e1(i);
            let mu0e =
                chi * (mu0 + sin_i * tan_tb * (psi.cos() * e2(e) + sin2_half_psi * e2(i)) / denom);
            let mue = chi * (mu + sin_e * tan_tb * (e2(e) - sin2_half_psi * e2(i)) / denom);
            let s = (mue / eta(e)) * (mu0 / eta(i)) * chi
                / (1.0 - f_psi + f_psi * chi * (mu0 / eta(i)));
            (mu0e, mue, s)
        } else {
            let denom = 2.0 - e1(i) - (psi / PI) * e1(e);
            let mu0e = chi * (mu0 + sin_i * tan_tb * (e2(i) - sin2_half_psi * e2(e)) / denom);
            let mue =
                chi * (mu + sin_e * tan_tb * (psi.cos() * e2(i) + sin2_half_psi * e2(e)) / denom);
            let s =
                (mue / eta(e)) * (mu0 / eta(i)) * chi / (1.0 - f_psi + f_psi * chi * (mu / eta(e)));
            (mu0e, mue, s)
        };
        (mu0e.max(0.0), mue.max(0.0), s.max(0.0))
    }
}

impl Brdf for Hapke {
    fn reflectance(&self, mu0: f64, mu: f64, alpha: f64) -> f64 {
        if mu0 <= 0.0 || mu <= 0.0 {
            return 0.0;
        }
        let (mu0e, mue, s) = self.roughness_correction(mu0, mu, alpha);
        if mu0e <= 0.0 || mue <= 0.0 {
            return 0.0;
        }
        let w = self.single_scattering_albedo;
        let p = self.phase_function.evaluate(alpha);
        let b = self.opposition_surge(alpha);
        let multiple = self.h_function(mu0e) * self.h_function(mue) - 1.0;
        let bracket = (1.0 + b) * p + multiple;
        (w / (4.0 * PI) * mu0e / (mu0e + mue) * bracket * s).max(0.0)
    }
}

/// Disk-integrated reflectance of a sphere at phase angle `alpha`:
///
/// `∫ r(μ₀, μ, α) μ dΩ`
///
/// over the visible, illuminated hemisphere, with `dΩ` the solid angle
/// of surface normals about the observer direction. This is the ratio
/// of the sphere's flux to that of a perfectly diffusing Lambert disk of
/// the same cross-section, so at `α = 0` it **is** the geometric albedo
/// and `Φ(α) = disk(α) / disk(0)` is the phase function.
///
/// Midpoint rule on a `(n_polar × 2 n_polar)` grid in emission angle
/// and azimuth. `n_polar = 400` gives four significant figures for
/// smooth laws; the terminator is resolved to `O(1/n)`.
pub fn disk_integrated_reflectance(brdf: &dyn Brdf, alpha: f64, n_polar: usize) -> f64 {
    let n_polar = n_polar.max(4);
    let n_az = 2 * n_polar;
    let d_theta = 0.5 * PI / n_polar as f64;
    let d_phi = 2.0 * PI / n_az as f64;
    let (sin_a, cos_a) = alpha.sin_cos();
    let mut total = 0.0;
    for it in 0..n_polar {
        let theta = (it as f64 + 0.5) * d_theta;
        let (sin_t, cos_t) = theta.sin_cos();
        let mu = cos_t;
        let d_omega = sin_t * d_theta * d_phi;
        for ip in 0..n_az {
            let phi = (ip as f64 + 0.5) * d_phi;
            // Normal in observer-centred frame; Sun in the x–z plane.
            let mu0 = sin_t * phi.cos() * sin_a + cos_t * cos_a;
            if mu0 <= 0.0 {
                continue;
            }
            total += brdf.reflectance(mu0, mu, alpha) * mu * d_omega;
        }
    }
    total
}

/// Geometric albedo `p = disk(0)`.
pub fn geometric_albedo(brdf: &dyn Brdf, n_polar: usize) -> f64 {
    disk_integrated_reflectance(brdf, 0.0, n_polar)
}

/// Phase integral `q = 2 ∫₀^π Φ(α) sin α dα`, so the Bond albedo is
/// `p · q`. Midpoint rule with `n_alpha` phase-angle samples.
pub fn phase_integral(brdf: &dyn Brdf, n_alpha: usize, n_polar: usize) -> f64 {
    let p = geometric_albedo(brdf, n_polar);
    if p <= 0.0 {
        return 0.0;
    }
    let n_alpha = n_alpha.max(4);
    let d_alpha = PI / n_alpha as f64;
    let mut q = 0.0;
    for ia in 0..n_alpha {
        let alpha = (ia as f64 + 0.5) * d_alpha;
        let phi = disk_integrated_reflectance(brdf, alpha, n_polar) / p;
        q += 2.0 * phi * alpha.sin() * d_alpha;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    const N: usize = 300;

    #[test]
    fn lambert_sphere_geometric_albedo_is_two_thirds() {
        let lambert = Lambert { albedo: 0.9 };
        assert_relative_eq!(geometric_albedo(&lambert, N), 0.6, max_relative = 2e-3);
    }

    #[test]
    fn lambert_sphere_bond_albedo_equals_lambert_albedo() {
        // q = 1.5 for a Lambert sphere, so p·q = A.
        let lambert = Lambert { albedo: 0.4 };
        let q = phase_integral(&lambert, 90, 120);
        assert_relative_eq!(q, 1.5, max_relative = 5e-3);
        assert_relative_eq!(
            q * geometric_albedo(&lambert, 120),
            0.4,
            max_relative = 5e-3
        );
    }

    #[test]
    fn lambert_phase_function_matches_closed_form() {
        // Φ(α) = [sin α + (π − α) cos α] / π for a Lambert sphere.
        let lambert = Lambert { albedo: 1.0 };
        let p = geometric_albedo(&lambert, N);
        for &alpha_deg in &[30.0_f64, 60.0, 90.0, 120.0] {
            let alpha = alpha_deg.to_radians();
            let phi = disk_integrated_reflectance(&lambert, alpha, N) / p;
            let expected = (alpha.sin() + (PI - alpha) * alpha.cos()) / PI;
            assert_abs_diff_eq!(phi, expected, epsilon = 3e-3);
        }
    }

    #[test]
    fn lommel_seeliger_sphere_geometric_albedo_is_w_over_eight() {
        let ls = LommelSeeliger::isotropic(0.4);
        assert_relative_eq!(geometric_albedo(&ls, N), 0.05, max_relative = 2e-3);
    }

    #[test]
    fn lommel_seeliger_full_disk_is_flat() {
        // At zero phase μ₀ = μ so r is constant across the disk.
        let ls = LommelSeeliger::isotropic(0.5);
        let centre = ls.reflectance(1.0, 1.0, 0.0);
        let limb = ls.reflectance(0.05, 0.05, 0.0);
        assert_relative_eq!(centre, limb, max_relative = 1e-12);
    }

    #[test]
    fn minnaert_k_one_is_lambert() {
        let m = Minnaert {
            albedo: 0.7,
            k: 1.0,
        };
        let l = Lambert { albedo: 0.7 };
        for &(mu0, mu) in &[(1.0, 1.0), (0.3, 0.8), (0.9, 0.1)] {
            assert_relative_eq!(
                m.reflectance(mu0, mu, 0.5),
                l.reflectance(mu0, mu, 0.5),
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn minnaert_large_k_darkens_the_limb() {
        let m = Minnaert {
            albedo: 0.7,
            k: 1.4,
        };
        let centre = m.reflectance(1.0, 1.0, 0.0);
        let limb = m.reflectance(0.2, 0.2, 0.0);
        let lambert_ratio = 0.2;
        assert!(limb / centre < lambert_ratio);
    }

    #[test]
    fn double_hg_is_normalised_over_the_sphere() {
        // (1/4π) ∫ P dΩ = (1/2) ∫₀^π P(α) sin α dα = 1.
        let pf = DoubleHenyeyGreenstein { b: 0.4, c: 0.6 };
        let n = 20_000;
        let d = PI / n as f64;
        let integral: f64 = (0..n)
            .map(|k| {
                let a = (k as f64 + 0.5) * d;
                0.5 * pf.evaluate(a) * a.sin() * d
            })
            .sum();
        assert_abs_diff_eq!(integral, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn double_hg_backscatter_lobe_peaks_at_zero_phase() {
        let pf = DoubleHenyeyGreenstein { b: 0.3, c: 1.0 };
        assert!(pf.evaluate(0.0) > pf.evaluate(PI / 2.0));
        assert!(pf.evaluate(PI / 2.0) > pf.evaluate(PI));
    }

    #[test]
    fn hapke_h_function_limits() {
        let h = Hapke {
            single_scattering_albedo: 0.5,
            ..Hapke::lunar_average()
        };
        assert_abs_diff_eq!(h.h_function(0.0), 1.0, epsilon = 1e-12);
        assert!(h.h_function(1.0) > 1.0);
        // Exact H(1) for w = 1 is 2.9078; the approximation is within ~1%.
        let conservative = Hapke {
            single_scattering_albedo: 1.0,
            ..Hapke::lunar_average()
        };
        assert_relative_eq!(conservative.h_function(1.0), 2.9078, max_relative = 0.02);
    }

    #[test]
    fn hapke_without_roughness_or_surge_reduces_to_lommel_seeliger_for_dark_surfaces() {
        // Small w: multiple scattering negligible, so Hapke → L-S with
        // the same phase function.
        let pf = DoubleHenyeyGreenstein { b: 0.2, c: 0.5 };
        let hapke = Hapke {
            single_scattering_albedo: 0.02,
            phase_function: pf,
            opposition_amplitude: 0.0,
            opposition_width: 0.0,
            roughness: 0.0,
        };
        let ls = LommelSeeliger {
            single_scattering_albedo: 0.02,
            phase_function: Some(pf),
        };
        for &(mu0, mu, a) in &[(0.9, 0.7, 0.3), (0.4, 0.9, 1.2), (0.2, 0.3, 2.0)] {
            assert_relative_eq!(
                hapke.reflectance(mu0, mu, a),
                ls.reflectance(mu0, mu, a),
                max_relative = 0.02
            );
        }
    }

    #[test]
    fn hapke_smooth_surface_obeys_helmholtz_reciprocity() {
        // The BRDF proper is r/μ₀; swapping incidence and emission must
        // leave it unchanged when there is no roughness term.
        let hapke = Hapke {
            roughness: 0.0,
            ..Hapke::lunar_average()
        };
        let a = hapke.reflectance(0.8, 0.3, 0.7) / 0.8;
        let b = hapke.reflectance(0.3, 0.8, 0.7) / 0.3;
        assert_relative_eq!(a, b, max_relative = 1e-12);
    }

    #[test]
    fn hapke_roughness_correction_is_identity_at_zero_slope() {
        let hapke = Hapke {
            roughness: 0.0,
            ..Hapke::lunar_average()
        };
        let (mu0e, mue, s) = hapke.roughness_correction(0.6, 0.4, 0.9);
        assert_eq!((mu0e, mue, s), (0.6, 0.4, 1.0));
    }

    #[test]
    fn hapke_roughness_correction_is_finite_and_darkens_large_angles() {
        let hapke = Hapke::lunar_average();
        for &(mu0, mu, a) in &[
            (1.0, 1.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.1, 0.9, 1.4),
            (0.9, 0.1, 1.4),
        ] {
            let (mu0e, mue, s) = hapke.roughness_correction(mu0, mu, a);
            assert!(mu0e.is_finite() && mue.is_finite() && s.is_finite());
            assert!(s > 0.0 && s <= 1.5, "s = {s}");
        }
        // Roughness lowers reflectance at large phase angles.
        let smooth = Hapke {
            roughness: 0.0,
            ..Hapke::lunar_average()
        };
        let alpha = 100.0_f64.to_radians();
        assert!(
            disk_integrated_reflectance(&hapke, alpha, 120)
                < disk_integrated_reflectance(&smooth, alpha, 120)
        );
    }

    #[test]
    fn hapke_opposition_surge_has_expected_shape() {
        let hapke = Hapke::lunar_average();
        assert_abs_diff_eq!(hapke.opposition_surge(0.0), 1.5, epsilon = 1e-12);
        let half = hapke.opposition_surge(2.0 * (0.06_f64).atan());
        assert_abs_diff_eq!(half, 0.75, epsilon = 1e-9);
    }

    #[test]
    fn lunar_average_geometric_albedo_is_moonlike() {
        // Visual geometric albedo of the Moon is 0.12–0.14; a global
        // parameter set with a strong opposition surge lands near it.
        let p = geometric_albedo(&Hapke::lunar_average(), 200);
        assert!((0.08..0.25).contains(&p), "p = {p}");
    }

    #[test]
    fn lunar_average_phase_curve_is_steep() {
        // The Moon drops by ~2.5 mag (factor ~10) from full to quarter.
        let hapke = Hapke::lunar_average();
        let p = geometric_albedo(&hapke, 150);
        let quarter = disk_integrated_reflectance(&hapke, PI / 2.0, 150) / p;
        assert!(quarter > 0.05 && quarter < 0.2, "Φ(90°) = {quarter}");
    }

    #[test]
    fn unlit_or_hidden_facets_return_zero() {
        let models: Vec<Box<dyn Brdf>> = vec![
            Box::new(Lambert { albedo: 1.0 }),
            Box::new(LommelSeeliger::isotropic(1.0)),
            Box::new(Minnaert {
                albedo: 1.0,
                k: 0.8,
            }),
            Box::new(Hapke::lunar_average()),
        ];
        for m in &models {
            assert_eq!(m.reflectance(-0.1, 0.5, 0.3), 0.0);
            assert_eq!(m.reflectance(0.5, 0.0, 0.3), 0.0);
        }
    }
}

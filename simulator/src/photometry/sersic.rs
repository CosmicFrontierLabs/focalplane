//! Sérsic surface-brightness profile primitives.
//!
//! **Vendored from `starfield::catalogs::SersicProfile`** (added in starfield
//! PR #122, version 0.12.0). Once a 0.12.x version of starfield is published
//! to crates.io and focalplane's `starfield` dep can be bumped, this module
//! should be replaced with a re-export of `starfield::catalogs::SersicProfile`.
//!
//! Everything below is bit-for-bit identical to the upstream evaluator, with
//! the test set extended to cross-validate against scipy reference values.
//!
//! # Physics
//!
//! The Sérsic profile is the de facto parametric model for galaxy
//! surface brightness in optical imaging:
//!
//! ```text
//! I(r) = I_e · exp[-b_n · ((r / θ_half)^(1/n) - 1)]
//! ```
//!
//! where `θ_half` is the half-light radius along the major axis, `n` is
//! the Sérsic index (`n = 0.5` Gaussian, `n = 1` exponential disk,
//! `n = 4` de Vaucouleurs bulge), and `b_n` is the constant that makes
//! `θ_half` enclose exactly half the total light: `γ(2n, b_n) = Γ(2n) / 2`.

/// Sérsic surface-brightness profile parameters for an extended source.
///
/// `axis_ratio` flattens the profile along the minor axis (1 = circular,
/// 0 = degenerate edge-on). `position_angle_deg` rotates the major axis
/// east of north (J2000) — the standard astronomical convention used
/// by the NASA-Sloan Atlas and by `astropy.modeling.functional_models.Sersic2D`
/// after the documented `theta_AstroPy = position_angle_deg + 90°`
/// translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SersicProfile {
    /// Half-light radius along the major axis, in arcseconds.
    pub theta_half_arcsec: f64,
    /// Sérsic index *n* (dimensionless, typically ~0.5 to ~6 for galaxies).
    pub n: f64,
    /// Axis ratio b/a, where b ≤ a. `1.0` is circular, `0.0` is degenerate edge-on.
    pub axis_ratio: f64,
    /// Position angle of the major axis, degrees east of north (J2000).
    pub position_angle_deg: f64,
}

impl SersicProfile {
    /// The Sérsic constant `b_n` defined by `γ(2n, b_n) = Γ(2n) / 2`.
    ///
    /// Uses the Ciotti & Bertin (1999, A&A, 352, 447) Eq. 18 asymptotic
    /// series. Empirically measured residuals against
    /// `scipy.special.gammaincinv(2n, 0.5)`:
    ///
    /// - n = 0.5 → 2.5 × 10⁻⁴
    /// - n = 1.0 → 4.2 × 10⁻⁵
    /// - n = 2.0 → 4.7 × 10⁻⁶
    /// - n = 4.0 → 5.4 × 10⁻⁷
    /// - n = 6.0 → 1.6 × 10⁻⁷
    ///
    /// Below n ≈ 0.36 the series degrades sharply; do not rely on it
    /// outside the typical galaxy range `n ∈ [0.5, 8]`.
    pub fn b_n(&self) -> f64 {
        let n = self.n;
        2.0 * n - 1.0 / 3.0
            + 4.0 / (405.0 * n)
            + 46.0 / (25_515.0 * n.powi(2))
            + 131.0 / (1_148_175.0 * n.powi(3))
            - 2_194_697.0 / (30_690_717_750.0 * n.powi(4))
    }

    /// Surface brightness at offset `(dx_arcsec, dy_arcsec)` from the
    /// galaxy centre, returned as the dimensionless ratio `I(r) / I_e`.
    ///
    /// `+dx_arcsec` is east, `+dy_arcsec` is north.
    /// `position_angle_deg` is degrees east of north — major-axis unit
    /// vector is `(sin_pa, cos_pa)`.
    ///
    /// Cross-validated against `astropy.modeling.functional_models.Sersic2D`:
    /// for n=4, r_eff=2.0, axis_ratio=0.6, PA=45°: SB(1.0, 0.5) = 2.461462
    /// and SB(2.0, 2.0) = 0.499511 (both within 1e-5).
    pub fn surface_brightness_at(&self, dx_arcsec: f64, dy_arcsec: f64) -> f64 {
        let bn = self.b_n();
        let a = self.theta_half_arcsec;
        let b = a * self.axis_ratio;
        let pa_rad = self.position_angle_deg.to_radians();
        let (sin_pa, cos_pa) = pa_rad.sin_cos();
        // Project (dx, dy) onto the rotated major / minor axes. The major
        // axis unit vector is (sin_pa, cos_pa); the minor axis is its
        // +90° rotation, (-cos_pa, sin_pa).
        let x_maj = dx_arcsec * sin_pa + dy_arcsec * cos_pa;
        let x_min = -dx_arcsec * cos_pa + dy_arcsec * sin_pa;
        let z = ((x_maj / a).powi(2) + (x_min / b).powi(2)).sqrt();
        (-bn * (z.powf(1.0 / self.n) - 1.0)).exp()
    }
}

/// Lanczos approximation to Γ(x), valid for x > 0. Coefficients are the
/// classical g=7, n=9 Spouge-style series; relative error is below
/// 1e-12 over `x ∈ [0.5, 100]`, well above the largest 2n we ever see
/// for catalog galaxies.
pub fn gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    // Coefficients reproduced from the canonical Lanczos g=7 derivation,
    // truncated to f64 precision (more digits would lose to rounding).
    const COEF: [f64; 9] = [
        0.999_999_999_999_810,
        676.520_368_121_885_2,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_1,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];
    // Reflection formula for x < 0.5 to extend domain — not needed for
    // our 2n use case where 2n ≥ 1, but cheap to keep.
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = COEF[0];
        for (i, c) in COEF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        let t = x + G + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

/// Radius (in arcsec along the major axis) at which the Sérsic SB drops
/// to `frac` of `I_e`. Solves `I(r) / I_e = frac` for r:
/// `r = θ_eff · ((-ln(frac)) / b_n + 1)^n`.
///
/// Used by `SersicSplat::footprint_pixels` to size the per-galaxy
/// rendering bounding box adaptively (a fixed multiplier of θ_eff would
/// undersize n=4 halos and oversize n=1 disks — see /tmp/galaxy_demo
/// renders for the failure mode).
pub fn radius_at_fraction(profile: &SersicProfile, frac: f64) -> f64 {
    let bn = profile.b_n();
    let raw = -(frac.ln()) / bn + 1.0;
    profile.theta_half_arcsec * raw.powf(profile.n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sersic(n: f64, axis_ratio: f64, pa: f64) -> SersicProfile {
        SersicProfile {
            theta_half_arcsec: 2.0,
            n,
            axis_ratio,
            position_angle_deg: pa,
        }
    }

    /// **Physics anchor**: scipy.special.gammaincinv(2n, 0.5) is the
    /// numerically-exact value of b_n. The Ciotti-Bertin series tracked
    /// against scipy's reference at five n values; tolerances ~2x above
    /// each measured residual lock the series form against accidental
    /// regression.
    #[test]
    fn b_n_matches_ciotti_bertin_series_against_scipy_reference() {
        let p_half = sersic(0.5, 1.0, 0.0);
        let p_one = sersic(1.0, 1.0, 0.0);
        let p_two = sersic(2.0, 1.0, 0.0);
        let p_four = sersic(4.0, 1.0, 0.0);
        let p_six = sersic(6.0, 1.0, 0.0);
        assert_abs_diff_eq!(p_half.b_n(), 0.6931471806, epsilon = 5e-4);
        assert_abs_diff_eq!(p_one.b_n(), 1.6783469900, epsilon = 1e-4);
        assert_abs_diff_eq!(p_two.b_n(), 3.6720607489, epsilon = 1e-5);
        assert_abs_diff_eq!(p_four.b_n(), 7.6692494425, epsilon = 1e-6);
        assert_abs_diff_eq!(p_six.b_n(), 11.6683631530, epsilon = 1e-6);
    }

    /// At the half-light radius along the major axis, by construction
    /// I(θ_half) / I_e = 1.
    #[test]
    fn surface_brightness_unity_at_half_light_radius_along_major_axis() {
        let p = sersic(4.0, 0.6, 0.0); // PA=0 → major axis along +y (north)
        assert_abs_diff_eq!(
            p.surface_brightness_at(0.0, p.theta_half_arcsec),
            1.0,
            epsilon = 1e-12
        );
    }

    /// At the centre, I(0) / I_e = exp(b_n) by definition.
    #[test]
    fn central_surface_brightness_equals_exp_b_n() {
        let p = sersic(4.0, 0.6, 45.0);
        assert_abs_diff_eq!(
            p.surface_brightness_at(0.0, 0.0),
            p.b_n().exp(),
            epsilon = 1e-9
        );
    }

    /// **Cross-validation against AstroPy**: `Sersic2D` model with
    /// `amplitude=1, r_eff=2.0, n=4, ellip=0.4, theta=(45°+90°) rad`.
    /// The +90° is the documented convention translation
    /// (AstroPy measures from +x; this evaluator uses east-of-north
    /// from +y). Reference values produced via astropy/scipy.
    #[test]
    fn surface_brightness_matches_astropy_sersic2d_reference() {
        let p = sersic(4.0, 0.6, 45.0);
        assert_abs_diff_eq!(p.surface_brightness_at(1.0, 0.5), 2.461462, epsilon = 1e-5);
        assert_abs_diff_eq!(p.surface_brightness_at(2.0, 2.0), 0.499511, epsilon = 1e-5);
    }

    /// Circular profile (axis_ratio = 1) has no preferred direction —
    /// SB depends only on √(dx² + dy²). Locks the rotation symmetry.
    #[test]
    fn circular_profile_is_axis_independent() {
        let p = sersic(2.5, 1.0, 37.0);
        let r = 1.3;
        let east = p.surface_brightness_at(r, 0.0);
        let north = p.surface_brightness_at(0.0, r);
        let diag = p.surface_brightness_at(r / 2.0_f64.sqrt(), r / 2.0_f64.sqrt());
        assert_abs_diff_eq!(east, north, epsilon = 1e-12);
        assert_abs_diff_eq!(east, diag, epsilon = 1e-12);
    }

    /// Rotating PA by 90° rotates the major axis east-of-north from +y
    /// into +x. Locks the PA convention.
    #[test]
    fn position_angle_90_rotates_major_axis_to_east() {
        let p = sersic(4.0, 0.5, 90.0);
        // Major axis now along +x; half-light along major axis = (θ_eff, 0).
        assert_abs_diff_eq!(
            p.surface_brightness_at(p.theta_half_arcsec, 0.0),
            1.0,
            epsilon = 1e-12
        );
        // Minor axis now along +y; half-light = (0, θ_eff · q).
        assert_abs_diff_eq!(
            p.surface_brightness_at(0.0, p.theta_half_arcsec * p.axis_ratio),
            1.0,
            epsilon = 1e-12
        );
    }

    /// **Physics anchor**: Γ(n) for integer n equals (n-1)!. Locks the
    /// Lanczos approximation against the discrete factorial sequence,
    /// which is the only ground-truth value that doesn't require an
    /// external reference library.
    #[test]
    fn gamma_matches_factorial_for_integer_arguments() {
        let factorials = [
            (1.0, 1.0),       // 0!
            (2.0, 1.0),       // 1!
            (3.0, 2.0),       // 2!
            (4.0, 6.0),       // 3!
            (5.0, 24.0),      // 4!
            (6.0, 120.0),     // 5!
            (10.0, 362880.0), // 9!
        ];
        for (x, expected) in factorials {
            let g = gamma(x);
            let rel = (g - expected).abs() / expected;
            assert!(
                rel < 1e-10,
                "Γ({x}) = {g}, expected {expected} (rel err {rel:.2e})"
            );
        }
    }

    /// Γ(0.5) = √π. Anchors the half-integer values that show up for n=0.25
    /// (2n=0.5) — outside our intended Sérsic range but the math should
    /// still be exact at this canonical value.
    #[test]
    fn gamma_half_equals_sqrt_pi() {
        let g = gamma(0.5);
        let expected = std::f64::consts::PI.sqrt();
        let rel = (g - expected).abs() / expected;
        assert!(rel < 1e-12, "Γ(0.5) = {g}, expected √π = {expected}");
    }

    /// `radius_at_fraction(p, 1.0) == θ_eff` because by definition
    /// SB(θ_eff) = I_e (i.e., the fraction at θ_eff is exactly 1.0).
    /// Closes the round-trip on the formula derivation.
    #[test]
    fn radius_at_fraction_one_equals_theta_eff() {
        let p = sersic(4.0, 0.6, 0.0);
        let r = radius_at_fraction(&p, 1.0);
        assert_abs_diff_eq!(r, p.theta_half_arcsec, epsilon = 1e-12);
    }

    /// `radius_at_fraction(p, frac)` and `surface_brightness_at` must
    /// be inverses of each other along the major axis. Tests this for
    /// n=1 and n=4 at frac=1e-2 and frac=1e-4 (the truncation budget
    /// SersicSplat uses).
    #[test]
    fn radius_at_fraction_inverts_surface_brightness() {
        for n in [1.0_f64, 4.0] {
            let p = sersic(n, 1.0, 0.0); // circular, PA=0 → major axis along +y
            for frac in [1e-2, 1e-4] {
                let r = radius_at_fraction(&p, frac);
                let sb = p.surface_brightness_at(0.0, r);
                let rel = (sb - frac).abs() / frac;
                assert!(
                    rel < 1e-9,
                    "round-trip n={n} frac={frac} r={r:.4} sb={sb:.6e} (rel err {rel:.2e})"
                );
            }
        }
    }
}

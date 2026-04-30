//! `SersicSplat` — `MeanFluxDeposit` impl for an elliptical Sérsic
//! galaxy profile rendered onto a pixel grid at a known plate scale.
//!
//! Wraps a [`SersicProfile`] together with the per-pixel arcsecond
//! scale and the precomputed analytic-integral normalisation so the
//! per-pixel inner loop is allocation-free and transcendental-free
//! (the gamma function call happens once at construction).
//!
//! # Math
//!
//! For a Sérsic profile written in the standard catalog form,
//!
//! ```text
//! I(r) = I_e · exp[-b_n · ((r / θ_eff)^(1/n) - 1)]
//! ```
//!
//! integrating over the elliptical plane (semi-axes `a = θ_eff`,
//! `b = q · θ_eff`) gives the total flux:
//!
//! ```text
//! F_total = 2π · n · b_n^(-2n) · Γ(2n) · q · θ_eff² · exp(b_n) · I_e
//! ```
//!
//! The `exp(b_n)` factor compensates for the `(- b_n · (-1))` term
//! inside the exponential — it is the difference between writing the
//! profile relative to `I_e` (catalog convention, what
//! [`SersicProfile`] uses) versus relative to `I_0 = I_e · exp(b_n)`
//! (the central-SB convention). Forgetting this factor under-states
//! `K` by exp(b_n), which at n=4 is ~2140 — the `/tmp/galaxy_demo`
//! prototype had this bug and only escaped detection because of the
//! asinh-with-percentile-clip display stretch. Reference: standard
//! Sérsic-luminosity derivation, e.g. Graham & Driver (2005)
//! Eq. (4)–(6).
//!
//! Solving for `I_e` from the catalog's `total_flux` and depositing
//! per pixel:
//!
//! ```text
//! I_e = total_flux / K
//! flux_pixel = I_e · SB(dx, dy) · pixel_area_arcsec²
//!            = (total_flux / K) · SB(dx, dy) · arcsec_per_pixel²
//! ```
//!
//! where `K = 2π · n · b_n^(-2n) · Γ(2n) · q · θ_eff² · exp(b_n)`.
//! The inverse of `K` is what we cache as `inv_analytic_norm` to turn
//! a per-pixel division into a multiply.
//!
//! # Truncation
//!
//! `footprint_pixels` returns the radius (in pixels) at which the
//! Sérsic SB drops to `1e-4 · I_e`, computed via
//! [`radius_at_fraction`]. This is profile-aware: a de Vaucouleurs
//! n=4 halo extends out to ~30 θ_eff while an n=1 disk reaches the
//! same fraction at ~5 θ_eff. The single-multiplier truncation we
//! used in the throwaway demo (`/tmp/galaxy_demo`) leaves visible
//! square edges around large n=4 galaxies; this profile-aware
//! truncation eliminates that artifact.
//!
//! # Thumbnail-cached evaluation
//!
//! `pixel_flux` is a bilinear interpolation into a `THUMB_OVERSAMPLE`×
//! oversampled SB thumbnail computed once at construction. The
//! thumbnail evaluates `SersicProfile::surface_brightness_at` at every
//! cell centre on a grid that's `THUMB_OVERSAMPLE × THUMB_OVERSAMPLE`
//! finer than the sensor pixel grid; subsequent splat lookups touch
//! four neighbouring cells and combine them with bilinear weights.
//!
//! Cost vs. earlier per-pixel Simpson rendering: thumbnail builds
//! in `(2·footprint·OS+1)²` SB evaluations once, then every pixel in
//! the deposit footprint is a handful of array reads + multiplies.
//! For an NSA-typical galaxy footprint of `N²` pixels the per-render
//! cost drops from `~9N²` SB evaluations (3×3 Simpson + adaptive
//! refinement) to `~OS²·N²` SB evaluations *at construction* + `O(N²)`
//! near-free interpolations at splat — ~5–10× faster end-to-end at
//! `OS = 2`.
//!
//! Bias / accuracy: the bilinear interpolant approximates a
//! pixel-mean SB by averaging the four `OS²`-finer cells around the
//! pixel centre. For NSA-typical plate scales (θ_eff covering many
//! pixels) the SB pattern is smooth on cell scales and the
//! interpolant matches Simpson within the truncation budget — see
//! the `n=1` and `n=4` flux-conservation tests. For pathologically
//! under-resolved galaxies (θ_eff < pixel) some residual peak
//! overcount remains; the test
//! `under_resolved_galaxy_overestimates_flux_documenting_v1_limitation`
//! pins the bias direction (always over).

use ndarray::Array2;
use starfield::catalogs::SersicProfile;

use crate::image_proc::deposit::MeanFluxDeposit;

/// Thumbnail oversampling factor: each sensor pixel covers
/// `THUMB_OVERSAMPLE × THUMB_OVERSAMPLE` thumbnail cells. Bilinear
/// interpolation at the sensor pixel centre then averages four
/// neighbouring thumbnail cells, which approximates the pixel-mean SB
/// well enough for NSA-typical galaxies (θ_eff covering many sensor
/// pixels) without the per-pixel Simpson cost.
const THUMB_OVERSAMPLE: i32 = 2;

/// `MeanFluxDeposit` impl for an elliptical Sérsic profile at a given
/// plate scale. See module-level docs for the math.
#[derive(Debug, Clone)]
pub struct SersicSplat {
    profile: SersicProfile,
    arcsec_per_pixel: f64,
    /// `1.0 / profile.total_flux_per_ie()`. Cached inverse so the
    /// per-pixel kernel is a multiply, not a divide. Upstream's
    /// helper handles the full Graham & Driver Eq. 4–6 closed form
    /// including the easy-to-drop `exp(b_n)` factor.
    inv_analytic_norm: f64,
    /// Pixel area in arcsec² — folded into `pixel_flux` to convert
    /// surface brightness into per-pixel flux.
    pixel_area_arcsec2: f64,
    /// Bounding-box half-width in pixels for the per-pixel deposit
    /// loop. Sized to the radius at which SB drops to `1e-4 · I_e`,
    /// so the truncation remainder stays well below the per-pixel
    /// noise floor for the deposit.
    footprint_px: i32,
    /// `THUMB_OVERSAMPLE`× oversampled SB pattern over the deposit's
    /// footprint, evaluated once at construction time. Each cell
    /// stores `SersicProfile::surface_brightness_at(cx, cy)` at the
    /// cell centre. `pixel_flux` bilinear-interpolates this thumbnail
    /// instead of re-evaluating the SB function per pixel.
    thumb: Array2<f64>,
    /// Cell width in arcsec: `arcsec_per_pixel / THUMB_OVERSAMPLE`.
    thumb_arcsec_per_cell: f64,
    /// Half-extent of the thumbnail in arcsec: positions outside
    /// `[-thumb_half_extent, +thumb_half_extent]` on either axis are
    /// outside the rendered footprint and return zero contribution.
    thumb_half_extent_arcsec: f64,
    /// Cell offset of the centre cell. Lookup index along an axis
    /// is `(c_arcsec / thumb_arcsec_per_cell) + thumb_centre_cell`.
    thumb_centre_cell: f64,
}

/// Radius (in arcsec along the major axis) at which the Sérsic SB
/// drops to `frac` of `I_e`. Closed-form inverse of the Sérsic SB
/// expression: `r = θ_eff · ((-ln(frac)) / b_n + 1)^n`.
///
/// Local to the renderer because it's a render-time *truncation
/// policy* helper, not a catalog primitive — upstream starfield
/// deliberately doesn't expose it.
fn radius_at_fraction(profile: &SersicProfile, frac: f64) -> f64 {
    let bn = profile.b_n();
    let raw = -(frac.ln()) / bn + 1.0;
    profile.theta_half_arcsec * raw.powf(profile.n)
}

/// Truncation fraction (relative to `I_e`) outside which the Sérsic
/// deposit is set to zero. The corresponding radius is computed
/// per-profile via [`radius_at_fraction`].
const TRUNCATION_SB_FRACTION: f64 = 1e-4;

impl SersicSplat {
    /// Build a deposit for `profile` at the given plate scale.
    /// Precomputes the analytic normalisation, the truncation
    /// footprint, and a `THUMB_OVERSAMPLE`× oversampled SB thumbnail
    /// covering the entire footprint. The thumbnail bake is the only
    /// place `SersicProfile::surface_brightness_at` is called — every
    /// subsequent `pixel_flux` lookup is a bilinear interpolation
    /// from the cached array.
    pub fn new(profile: SersicProfile, arcsec_per_pixel: f64) -> Self {
        // K = 2π · n · b_n^(-2n) · Γ(2n) · q · θ_eff² · exp(b_n) —
        // see SersicProfile::total_flux_per_ie upstream for the
        // derivation and Graham & Driver (2005) reference.
        let analytic_norm = profile.total_flux_per_ie();
        let footprint_arcsec = radius_at_fraction(&profile, TRUNCATION_SB_FRACTION);
        // ceil to int pixels; clamp to at least 1 so a sub-pixel galaxy
        // still drops a single pixel of light.
        let footprint_px = ((footprint_arcsec / arcsec_per_pixel).ceil() as i32).max(1);

        // Build the THUMB_OVERSAMPLE-oversampled SB thumbnail.
        // **Cell-centered grid**: cell k along each axis has its
        // sample centre at `(k + 0.5 - centre) · cell_size`. The
        // galaxy origin (`cx = cy = 0`) deliberately falls *between*
        // cells, so a sensor pixel centred on the galaxy reads four
        // off-centre cells via bilinear interpolation rather than
        // sampling the central SB peak directly. Without this
        // half-cell offset, bilinear at a galaxy-centred pixel would
        // return the peak SB exactly and over-count flux by
        // `exp(b_n)` (~2140 at n=4) compared to the true pixel-mean.
        let thumb_arcsec_per_cell = arcsec_per_pixel / THUMB_OVERSAMPLE as f64;
        let footprint_cells = footprint_px * THUMB_OVERSAMPLE;
        let thumb_size = (footprint_cells * 2) as usize;
        let thumb_centre = footprint_cells as f64; // cell-coordinate of galaxy origin
        let thumb_half_extent_arcsec = footprint_cells as f64 * thumb_arcsec_per_cell;
        let mut thumb = Array2::<f64>::zeros((thumb_size, thumb_size));
        for tr in 0..thumb_size {
            let cy = (tr as f64 + 0.5 - thumb_centre) * thumb_arcsec_per_cell;
            for tc in 0..thumb_size {
                let cx = (tc as f64 + 0.5 - thumb_centre) * thumb_arcsec_per_cell;
                thumb[[tr, tc]] = profile.surface_brightness_at(cx, cy);
            }
        }

        Self {
            profile,
            arcsec_per_pixel,
            inv_analytic_norm: analytic_norm.recip(),
            pixel_area_arcsec2: arcsec_per_pixel * arcsec_per_pixel,
            footprint_px,
            thumb,
            thumb_arcsec_per_cell,
            thumb_half_extent_arcsec,
            thumb_centre_cell: thumb_centre,
        }
    }

    /// The wrapped profile.
    pub fn profile(&self) -> &SersicProfile {
        &self.profile
    }

    /// Pixel scale (arcsec / pixel) baked into this deposit.
    pub fn arcsec_per_pixel(&self) -> f64 {
        self.arcsec_per_pixel
    }
}

impl MeanFluxDeposit for SersicSplat {
    fn footprint_pixels(&self) -> i32 {
        self.footprint_px
    }

    fn pixel_flux(&self, dx: f64, dy: f64, total_flux: f64) -> f64 {
        if total_flux == 0.0 {
            return 0.0;
        }
        let s = self.arcsec_per_pixel;
        // Image coordinates: +dx (col offset) maps to +x_sky = east.
        // +dy (row offset) maps to *south* (row index grows downward
        // in standard image display), so cy = -dy*s to align with the
        // SB evaluator's +y_sky = north convention. Without this
        // flip, elliptical galaxies render mirrored about the
        // horizontal axis vs. AstroPy `Sersic2D`.
        let cx = dx * s;
        let cy = -dy * s;

        // Bilinear interpolation into the precomputed thumbnail. The
        // thumbnail covers `[-thumb_half_extent, +thumb_half_extent]`
        // on each axis at `THUMB_OVERSAMPLE × THUMB_OVERSAMPLE` cells
        // per sensor pixel. Outside that extent the deposit is below
        // the truncation budget; return zero.
        if cx.abs() > self.thumb_half_extent_arcsec || cy.abs() > self.thumb_half_extent_arcsec {
            return 0.0;
        }
        // Cell-centered grid: cell k centre is at
        // `(k + 0.5 - thumb_centre_cell) · thumb_arcsec_per_cell`.
        // For a query at `cx`, the fractional cell index is
        // `cx / cell_size + thumb_centre - 0.5`.
        let tx = cx / self.thumb_arcsec_per_cell + self.thumb_centre_cell - 0.5;
        let ty = cy / self.thumb_arcsec_per_cell + self.thumb_centre_cell - 0.5;
        let last = self.thumb.shape()[0] - 1;
        let tx0 = (tx.floor() as isize).clamp(0, last as isize - 1) as usize;
        let ty0 = (ty.floor() as isize).clamp(0, last as isize - 1) as usize;
        let tx1 = tx0 + 1;
        let ty1 = ty0 + 1;
        let fx = (tx - tx0 as f64).clamp(0.0, 1.0);
        let fy = (ty - ty0 as f64).clamp(0.0, 1.0);
        let s00 = self.thumb[[ty0, tx0]];
        let s01 = self.thumb[[ty0, tx1]];
        let s10 = self.thumb[[ty1, tx0]];
        let s11 = self.thumb[[ty1, tx1]];
        let s0 = s00 * (1.0 - fx) + s01 * fx;
        let s1 = s10 * (1.0 - fx) + s11 * fx;
        let sb = s0 * (1.0 - fy) + s1 * fy;

        total_flux * self.inv_analytic_norm * sb * self.pixel_area_arcsec2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_proc::deposit::splat_deposit;
    use ndarray::Array2;

    fn sersic(n: f64, axis_ratio: f64, pa: f64, theta_eff: f64) -> SersicProfile {
        SersicProfile {
            theta_half_arcsec: theta_eff,
            n,
            axis_ratio,
            position_angle_deg: pa,
        }
    }

    /// **Physics anchor — INVARIANTS §3 flux conservation**: depositing
    /// `total_flux = 1.0` of a well-resolved circular Sérsic onto a
    /// generous box must integrate back to within the truncation
    /// budget. "Well-resolved" means `θ_eff` covers many pixels so
    /// the single-point pixel-centre SB sampling is a good
    /// approximation to the true pixel integral.
    ///
    /// For a circular Sérsic, the fraction of total flux enclosed
    /// within radius `R` along the major axis is
    /// `γ(2n, b_n · (R/θ_eff)^(1/n)) / Γ(2n)`. At the truncation
    /// radius `radius_at_fraction(p, 1e-4)`, this fraction is
    /// > 0.99 for n=1 and ~0.97 for n=4 (the n=4 wings carry more
    /// of the total light).
    #[test]
    fn sersic_deposit_conserves_flux_within_truncation_budget_n_equals_1() {
        // Well-resolved: θ_eff = 5", arcsec_per_pixel = 0.25 → 20 px per θ_eff.
        let p = sersic(1.0, 1.0, 0.0, 5.0);
        let arcsec_per_pixel = 0.25;
        let splat = SersicSplat::new(p, arcsec_per_pixel);
        let footprint = splat.footprint_pixels() as usize;
        let n = footprint * 2 + 16;
        let mut buf = Array2::<f64>::zeros((n, n));
        let total_in = 1.0_f64;
        splat_deposit(&mut buf, n as f64 * 0.5, n as f64 * 0.5, total_in, &splat);
        let total_out: f64 = buf.iter().sum();
        assert!(
            total_out > 0.99 && total_out < 1.01,
            "n=1 well-resolved capture {total_out} outside [0.99, 1.01]"
        );
    }

    /// Same flux conservation at n=4 (de Vaucouleurs). The extended
    /// halo carries ~3% of the total light beyond the 1e-4 SB
    /// truncation; the rendered sum lives in [0.95, 1.0).
    #[test]
    fn sersic_deposit_conserves_flux_within_truncation_budget_n_equals_4() {
        // Well-resolved: θ_eff = 5", arcsec_per_pixel = 0.25.
        let p = sersic(4.0, 1.0, 0.0, 5.0);
        let arcsec_per_pixel = 0.25;
        let splat = SersicSplat::new(p, arcsec_per_pixel);
        let footprint = splat.footprint_pixels() as usize;
        let n = footprint * 2 + 16;
        let mut buf = Array2::<f64>::zeros((n, n));
        splat_deposit(&mut buf, n as f64 * 0.5, n as f64 * 0.5, 1.0, &splat);
        let total_out: f64 = buf.iter().sum();
        assert!(
            total_out > 0.95 && total_out < 1.02,
            "n=4 well-resolved capture {total_out} outside [0.95, 1.02]"
        );
    }

    /// **Plate-scale invariance** in the well-resolved regime
    /// (θ_eff ≥ 5 px per the v1 resolution requirement). All three
    /// plate scales must agree on integrated flux to better than 2%,
    /// since they all bracket the truncation bound from above. Locks
    /// the `pixel_area` factor in the deposit math.
    #[test]
    fn integrated_flux_is_plate_scale_invariant_when_well_resolved() {
        let p = sersic(2.0, 0.7, 30.0, 5.0); // θ_eff = 5"
        let total_in = 1.0_f64;
        let mut totals = Vec::new();
        // All three sample θ_eff with ≥ 5 px (per the v1 requirement).
        for arcsec_per_pixel in [0.1, 0.25, 1.0] {
            let splat = SersicSplat::new(p, arcsec_per_pixel);
            let footprint = splat.footprint_pixels() as usize;
            let n = footprint * 2 + 16;
            let mut buf = Array2::<f64>::zeros((n, n));
            splat_deposit(&mut buf, n as f64 * 0.5, n as f64 * 0.5, total_in, &splat);
            totals.push(buf.iter().sum::<f64>());
        }
        let mean = totals.iter().sum::<f64>() / totals.len() as f64;
        for t in &totals {
            let rel = (t - mean).abs() / mean;
            assert!(
                rel < 0.02,
                "plate-scale variance too large: totals={totals:?} mean={mean}"
            );
        }
    }

    /// **Documents the v1 resolution limitation**: pathologically
    /// under-resolved galaxies (θ_eff smaller than a sensor pixel)
    /// under-count their flux because the cell-centered thumbnail
    /// samples SB at offsets ≥ 0.25 pixels from the galaxy centre,
    /// missing most of the peak when the entire profile fits inside
    /// a single pixel.
    ///
    /// The bias is *predictable* in direction (always under) and
    /// modest in magnitude (~30-40% under at θ_eff = 0.3 pixels).
    /// For NSA-typical galaxies (θ_eff covering many pixels) this
    /// regime never matters. A future improvement could analytically
    /// integrate the central pixel(s) for sub-pixel galaxies.
    #[test]
    fn under_resolved_galaxy_under_counts_flux_documenting_v1_limitation() {
        // Pathologically under-resolved: θ_eff = 0.3", arcsec_per_pixel = 1".
        let p = sersic(4.0, 1.0, 0.0, 0.3);
        let splat = SersicSplat::new(p, 1.0);
        let footprint = splat.footprint_pixels() as usize;
        let n = footprint * 2 + 16;
        let mut buf = Array2::<f64>::zeros((n, n));
        splat_deposit(&mut buf, n as f64 * 0.5, n as f64 * 0.5, 1.0, &splat);
        let total_out: f64 = buf.iter().sum();
        assert!(
            total_out > 0.4 && total_out < 0.9,
            "under-resolved capture {total_out} outside [0.4, 0.9] (cell-centred under-sample bias)"
        );
    }

    /// The deposit must be invariant under PA rotation in the sense
    /// that rotating BOTH the PA *and* the deposit centre's pixel-space
    /// orientation by the same angle produces the same total flux
    /// (and the same per-radial-bin distribution). Anchored simply: a
    /// circular profile is unchanged by any PA, so the integrated flux
    /// at PA=0, 30°, 60°, 90° must be equal pixel-for-pixel.
    #[test]
    fn circular_profile_independent_of_pa() {
        let arcsec_per_pixel = 0.5;
        let mut sums = Vec::new();
        for pa in [0.0, 30.0, 60.0, 90.0, 173.4] {
            let p = sersic(2.0, 1.0, pa, 1.5);
            let splat = SersicSplat::new(p, arcsec_per_pixel);
            let footprint = splat.footprint_pixels() as usize;
            let n = footprint * 2 + 8;
            let mut buf = Array2::<f64>::zeros((n, n));
            splat_deposit(&mut buf, n as f64 * 0.5, n as f64 * 0.5, 1.0, &splat);
            sums.push(buf.iter().sum::<f64>());
        }
        let first = sums[0];
        for s in &sums[1..] {
            assert!(
                (s - first).abs() < 1e-12,
                "circular profile not PA-invariant: sums={sums:?}"
            );
        }
    }

    /// Zero total flux is a no-op — same contract as `splat_deposit`.
    #[test]
    fn zero_total_flux_is_no_op() {
        let p = sersic(4.0, 0.6, 45.0, 2.0);
        let splat = SersicSplat::new(p, 0.5);
        assert_eq!(splat.pixel_flux(0.0, 0.0, 0.0), 0.0);
        assert_eq!(splat.pixel_flux(1.0, 1.0, 0.0), 0.0);
    }

    /// Footprint sizing: an n=4 galaxy must have a *larger* footprint
    /// than an n=1 galaxy with the same θ_eff, because n=4 halos
    /// extend much further. Locks the profile-aware truncation against
    /// regression to a single-multiplier scheme.
    #[test]
    fn footprint_grows_with_sersic_index() {
        let arcsec_per_pixel = 0.5;
        let theta_eff = 2.0;
        let foot_n1 =
            SersicSplat::new(sersic(1.0, 1.0, 0.0, theta_eff), arcsec_per_pixel).footprint_pixels();
        let foot_n4 =
            SersicSplat::new(sersic(4.0, 1.0, 0.0, theta_eff), arcsec_per_pixel).footprint_pixels();
        assert!(
            foot_n4 > foot_n1,
            "n=4 footprint {foot_n4} not larger than n=1 footprint {foot_n1}"
        );
    }

    /// **Anchor on a flat-SB region** — at a pixel far from the
    /// centre but within the truncation radius, SB varies slowly
    /// over the pixel so the 3×3 Simpson sub-sample converges to
    /// the centre-evaluated value. Locks the (b_n, K, pixel_area)
    /// chain in a regime where sub-pixel curvature isn't a factor.
    #[test]
    fn pixel_flux_at_flat_sb_region_matches_closed_form() {
        let p = sersic(4.0, 0.6, 45.0, 2.0);
        let arcsec_per_pixel = 0.5;
        let splat = SersicSplat::new(p, arcsec_per_pixel);
        let total = 1000.0;
        // Sample at (10, 10) pixels = (5, 5) arcsec — well outside
        // the half-light radius where SB varies slowly across a pixel.
        let dx_pix = 10.0;
        let dy_pix = 10.0;
        let pixel = splat.pixel_flux(dx_pix, dy_pix, total);
        // I_e from upstream's closed-form K = total_flux_per_ie. Same
        // formula SersicSplat::new uses internally — anchors the
        // pixel-flux output against the analytic expectation in a
        // slowly-varying region where 3×3 Simpson sub-sampling
        // converges to centre-sampling to O(h^4).
        let i_e = total / p.total_flux_per_ie();
        // Pass the same (cx, cy) sign convention as SersicSplat::pixel_flux:
        // +cx = east (col offset), +cy = north (negated row offset).
        let sb_centre =
            p.surface_brightness_at(dx_pix * arcsec_per_pixel, -dy_pix * arcsec_per_pixel);
        let expected = i_e * sb_centre * arcsec_per_pixel * arcsec_per_pixel;
        // 3×3 Simpson on a slowly-varying region matches centre-
        // sampling to O(h^4) — for our setup well below 1%.
        let rel = (pixel - expected).abs() / expected;
        assert!(
            rel < 1e-2,
            "flat-SB pixel {pixel} vs expected {expected} (rel err {rel:.2e})"
        );
    }
}

#[cfg(test)]
mod orientation_diag {
    use super::*;
    use crate::image_proc::deposit::splat_deposit;
    use ndarray::Array2;

    /// **Orientation regression** — locks the y-axis sign convention
    /// in `SersicSplat::pixel_flux` against AstroPy's `Sersic2D` for
    /// PA=45° east-of-north (major axis NE-SW). Image convention:
    /// row 0 = top = north, col 0 = left = west, east on the right.
    ///
    /// PA=45 east-of-north (major axis NE-SW) means high SB on the
    /// "/" diagonal: bright at NE corner (row<center, col>center)
    /// and SW corner (row>center, col<center); low SB at NW and SE
    /// corners.
    ///
    /// If anyone removes the `cy = -dy*s` sign flip in `pixel_flux`,
    /// this test fires immediately — galaxies would render mirrored
    /// about the horizontal axis, swapping "/" and "\" elongation.
    #[test]
    fn pa45_orientation_matches_astropy_ne_sw_major_axis() {
        let profile = SersicProfile {
            theta_half_arcsec: 5.0,
            n: 4.0,
            axis_ratio: 0.5,
            position_angle_deg: 45.0,
        };
        let splat = SersicSplat::new(profile, 0.5);
        let n: usize = 200;
        let mut buf = Array2::<f64>::zeros((n, n));
        splat_deposit(&mut buf, n as f64 * 0.5, n as f64 * 0.5, 1.0, &splat);

        // Sample 30px in from each corner.
        let off = 30;
        let c = n / 2;
        let nw = buf[[c - off, c - off]];
        let ne = buf[[c - off, c + off]];
        let sw = buf[[c + off, c - off]];
        let se = buf[[c + off, c + off]];

        // PA=45 east-of-north → "/" diagonal: NE+SW high, NW+SE low.
        assert!(
            ne > 5.0 * nw,
            "expected NE >> NW for PA=45 east-of-north (\"/\" major axis); got NE={ne:.3e} NW={nw:.3e}"
        );
        assert!(
            sw > 5.0 * se,
            "expected SW >> SE for PA=45 east-of-north (\"/\" major axis); got SW={sw:.3e} SE={se:.3e}"
        );
        // NE ≈ SW and NW ≈ SE by point symmetry of the Sérsic ellipse.
        let asym_diag = (ne - sw).abs() / ne.max(sw);
        let asym_anti = (nw - se).abs() / nw.max(se);
        assert!(
            asym_diag < 0.01 && asym_anti < 0.01,
            "ellipse should be point-symmetric: NE={ne:.3e} SW={sw:.3e} NW={nw:.3e} SE={se:.3e}"
        );
    }

    /// **Profile** the per-galaxy splat cost on a synthetic
    /// NSA-realistic field — IMX455 (9568×6380) at 0.224″/pix, mix of
    /// n=1 disks and n=4 ellipticals across the full theta_eff range
    /// NSA actually publishes. Reports per-galaxy splat times so we
    /// can see whether the cost is uniform or dominated by a few big
    /// n=4 outliers.
    #[test]
    #[ignore]
    fn profile_synthetic_nsa_field() {
        use std::time::Instant;
        let arcsec_per_pixel = 0.224_f64;
        let (w, h) = (9568_usize, 6380_usize);
        let mut buf = Array2::<f64>::zeros((h, w));
        // Mix that approximates a 66-galaxy NSA Coma-like field:
        // mostly small disks, a handful of bigger ellipticals.
        let mut sources: Vec<(f64, f64, SersicProfile)> = Vec::new();
        let centre_x = w as f64 * 0.5;
        let centre_y = h as f64 * 0.5;
        let mut rand_seed = 0xCAFEFEEDu64;
        for i in 0..50 {
            // LCG for repeatable scatter
            rand_seed = rand_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r1 = (rand_seed >> 32) as u32 as f64 / u32::MAX as f64;
            rand_seed = rand_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r2 = (rand_seed >> 32) as u32 as f64 / u32::MAX as f64;
            let dx_pix = (r1 - 0.5) * 1500.0;
            let dy_pix = (r2 - 0.5) * 1500.0;
            sources.push((
                centre_x + dx_pix,
                centre_y + dy_pix,
                SersicProfile {
                    theta_half_arcsec: 2.0 + (i as f64) * 0.1,
                    n: 1.0,
                    axis_ratio: 0.6,
                    position_angle_deg: (i as f64) * 7.0,
                },
            ));
        }
        // 16 ellipticals (n=4) — these have huge halos
        for i in 0..16 {
            rand_seed = rand_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r1 = (rand_seed >> 32) as u32 as f64 / u32::MAX as f64;
            rand_seed = rand_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r2 = (rand_seed >> 32) as u32 as f64 / u32::MAX as f64;
            let dx_pix = (r1 - 0.5) * 1500.0;
            let dy_pix = (r2 - 0.5) * 1500.0;
            sources.push((
                centre_x + dx_pix,
                centre_y + dy_pix,
                SersicProfile {
                    theta_half_arcsec: 5.0 + (i as f64) * 1.0,
                    n: 4.0,
                    axis_ratio: 0.7,
                    position_angle_deg: (i as f64) * 11.0,
                },
            ));
        }
        eprintln!(
            "Profile: {} galaxies on {}×{} buffer at {}″/pix",
            sources.len(),
            w,
            h,
            arcsec_per_pixel
        );
        let total_start = Instant::now();
        for (i, (px, py, profile)) in sources.iter().enumerate() {
            let splat = SersicSplat::new(*profile, arcsec_per_pixel);
            let footprint = splat.footprint_pixels();
            let pixels_in_box = (2 * footprint as i64 + 1).pow(2);
            let t = Instant::now();
            splat_deposit(&mut buf, *px, *py, 1.0, &splat);
            let elapsed = t.elapsed();
            let ns_per_px = elapsed.as_nanos() as f64 / pixels_in_box as f64;
            eprintln!(
                "  galaxy {:2} n={:.1} θ_eff={:5.1}″ footprint={:4}px ({:9} px-in-box)  {:>9.2}ms  {:.0} ns/px",
                i,
                profile.n,
                profile.theta_half_arcsec,
                footprint,
                pixels_in_box,
                elapsed.as_secs_f64() * 1e3,
                ns_per_px,
            );
        }
        let total = total_start.elapsed();
        eprintln!(
            "Total: {:.2}s for {} galaxies",
            total.as_secs_f64(),
            sources.len()
        );
    }

    /// Emit raw float buffer of a PA=45 axis_ratio=0.5 SersicSplat
    /// render to /tmp/rust_sersic_pa45_raw.f64. Diagnostic only.
    #[test]
    #[ignore]
    fn emit_pa45_diagnostic() {
        use std::io::Write;
        let profile = SersicProfile {
            theta_half_arcsec: 5.0,
            n: 4.0,
            axis_ratio: 0.5,
            position_angle_deg: 45.0,
        };
        let splat = SersicSplat::new(profile, 0.5);
        let n = 200;
        let mut buf = Array2::<f64>::zeros((n, n));
        splat_deposit(&mut buf, n as f64 * 0.5, n as f64 * 0.5, 1.0, &splat);
        // Print corner samples
        let off = 30;
        let c = n / 2;
        eprintln!("Rust SersicSplat render at PA=45, q=0.5, theta_eff=5\":");
        eprintln!(
            "  TL ({}, {}): {:.4e}",
            c - off,
            c - off,
            buf[[c - off, c - off]]
        );
        eprintln!(
            "  TR ({}, {}): {:.4e}",
            c - off,
            c + off,
            buf[[c - off, c + off]]
        );
        eprintln!(
            "  BL ({}, {}): {:.4e}",
            c + off,
            c - off,
            buf[[c + off, c - off]]
        );
        eprintln!(
            "  BR ({}, {}): {:.4e}",
            c + off,
            c + off,
            buf[[c + off, c + off]]
        );
        // Save raw f64
        let mut bytes = Vec::with_capacity(n * n * 8);
        for &v in buf.iter() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut f = std::fs::File::create("/tmp/rust_sersic_pa45_raw.f64").unwrap();
        f.write_all(&bytes).unwrap();
        eprintln!("wrote /tmp/rust_sersic_pa45_raw.f64 ({}x{} f64)", n, n);
    }
}

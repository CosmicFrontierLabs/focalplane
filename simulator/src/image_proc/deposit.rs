//! `MeanFluxDeposit` and `FrameSource` traits — the renderer-facing
//! contract that point sources (PSF-convolved stars) and extended
//! sources (Sérsic galaxies, future measured profiles) both implement.
//!
//! Both renderer paths — the static [`super::render::add_stars_to_image`]
//! and the motion-blur per-stamp deposit in
//! [`crate::sims::motion_blur::SensorAccumulator`] — used to carry
//! independent copies of the same bounding-box-loop and the same
//! per-pixel PSF call. This module collapses both into a single generic
//! [`splat_deposit`] helper plus a [`render_sources`] convenience that
//! iterates a source list.
//!
//! # Single-Poisson invariant
//!
//! Implementors must never sample noise inside `pixel_flux` — the
//! shot-noise sampling runs once at the end of the frame on the
//! combined mean-electron image. See `planning/INVARIANTS.md` §1 for
//! the architectural contract; this trait surface is what locks the
//! contract at the source-deposit level.

use std::time::Duration;

use ndarray::Array2;
use shared::image_proc::airy::PixelScaledAiryDisk;
use shared::units::Area;

/// Spatial flux distribution for one source at one deposit centre,
/// distributing `total_flux` over neighbouring pixels of a 2-D buffer.
///
/// PSF-convolved point sources implement this via Simpson's-rule
/// integration of an Airy disk (see the blanket impl below). Extended
/// sources (Sérsic galaxies in the upcoming PR 3) implement it via
/// direct surface-brightness evaluation × pixel area.
///
/// # Determinism
///
/// `pixel_flux` must be deterministic and side-effect-free — no RNG,
/// no caches with hidden state. The renderer relies on this to keep
/// per-tile RNG seeding reproducible. See INVARIANTS §5.
///
/// # Truncation
///
/// `footprint_pixels` describes the bounding-box half-width inside
/// which the deposit contributes meaningfully. Pixels outside it are
/// skipped. Implementors choose the threshold:
///
/// - PSF stars: `2 × first_zero` of the Airy disk (~99.99…% capture)
/// - Sérsic galaxies (PR 3): radius at which SB drops to `1e-4 · I_e`
pub trait MeanFluxDeposit {
    /// Half-width of the bounding box in pixels for the per-pixel loop.
    fn footprint_pixels(&self) -> i32;

    /// Mean flux contributed to the pixel whose centre sits at offset
    /// `(dx, dy)` *pixels* from the deposit centre, given `total_flux`
    /// integrated over the source for the integration window.
    ///
    /// Must return zero for `total_flux == 0` and must be free of
    /// side effects (no RNG, no shared state).
    fn pixel_flux(&self, dx: f64, dy: f64, total_flux: f64) -> f64;
}

/// One source ready to be splatted onto a single sensor for one
/// integration window. Generic over its deposit so the per-pixel inner
/// loop monomorphises (no vtable cost in the hot path).
///
/// Stars implement this with `Deposit = PixelScaledAiryDisk`. Galaxies
/// (PR 3) implement it with `Deposit = SersicSplat`. Adding a new
/// source kind (satellite trails, asteroids) means implementing this
/// trait + adding one `render_sources` call to each renderer.
pub trait FrameSource {
    /// Spatial deposit shape — typically the chromatic effective Airy
    /// disk for stars or the elliptical Sérsic profile for galaxies.
    type Deposit: MeanFluxDeposit;

    /// Sub-pixel position on the sensor: `(x, y)` where `0` is the
    /// corner and `(width-1, height-1)` is the opposite corner.
    fn position_pixels(&self) -> (f64, f64);

    /// Total mean electrons this source contributes during the
    /// integration window — the value `splat_deposit` distributes
    /// across the deposit's footprint.
    fn total_electrons(&self, dt: Duration, aperture: Area) -> f64;

    /// The spatial deposit (Airy disk for stars; SersicSplat for
    /// galaxies in PR 3).
    fn deposit(&self) -> &Self::Deposit;
}

/// Splat one source's mean-electron contribution onto `buf`. Iterates
/// pixels within `deposit.footprint_pixels()` of `(px, py)`, calls
/// `deposit.pixel_flux(dx, dy, total_flux)` per pixel, accumulates
/// into `buf`. Skips out-of-bounds pixels. Early-returns if
/// `total_flux == 0.0` so a stamp with no exposure budget is a no-op.
///
/// `buf` is indexed `[row, col] = [y, x]` to match `Array2<f64>`'s
/// `(rows, cols)` shape, which is the convention used throughout
/// focalplane's image pipeline.
pub fn splat_deposit<D: MeanFluxDeposit>(
    buf: &mut Array2<f64>,
    px: f64,
    py: f64,
    total_flux: f64,
    deposit: &D,
) {
    if total_flux == 0.0 {
        return;
    }
    let (height, width) = buf.dim();
    if height == 0 || width == 0 {
        return;
    }
    let footprint = deposit.footprint_pixels();
    let xc = px.round() as i32;
    let yc = py.round() as i32;
    let min_x = (xc - footprint).max(0);
    let max_x = (xc + footprint).min(width as i32 - 1);
    let min_y = (yc - footprint).max(0);
    let max_y = (yc + footprint).min(height as i32 - 1);
    for x in min_x..=max_x {
        for y in min_y..=max_y {
            let dx = x as f64 - px;
            let dy = y as f64 - py;
            buf[[y as usize, x as usize]] += deposit.pixel_flux(dx, dy, total_flux);
        }
    }
}

/// Generic per-frame render pass — splats every source onto `buf`.
/// Monomorphised per `S` so the inner deposit call inlines and `splat_deposit`
/// itself is monomorphised on the source's `Deposit`.
///
/// This is the helper both renderer paths call into. The static path
/// invokes it once per (sensor, kind-of-source); the motion-blur path
/// invokes it once per (sensor, kind-of-source, stamp).
pub fn render_sources<S: FrameSource>(
    buf: &mut Array2<f64>,
    sources: &[S],
    dt: Duration,
    aperture: Area,
) {
    for s in sources {
        let total = s.total_electrons(dt, aperture);
        let (px, py) = s.position_pixels();
        splat_deposit(buf, px, py, total, s.deposit());
    }
}

/// Blanket impl: a chromatic Airy disk is a deposit. Footprint is
/// `2 × first_zero` (matches the existing star-rendering convention),
/// per-pixel flux is Simpson's-rule integration of the Airy intensity
/// over the pixel.
impl MeanFluxDeposit for PixelScaledAiryDisk {
    fn footprint_pixels(&self) -> i32 {
        (self.first_zero().max(1.0) * 2.0).ceil() as i32
    }

    fn pixel_flux(&self, dx: f64, dy: f64, total_flux: f64) -> f64 {
        self.pixel_flux_simpson(dx, dy, total_flux)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::image_proc::airy::PixelScaledAiryDisk;
    use shared::units::{LengthExt, Wavelength};
    use std::cell::Cell;

    struct CountingDeposit {
        calls: Cell<usize>,
    }

    impl MeanFluxDeposit for CountingDeposit {
        fn footprint_pixels(&self) -> i32 {
            1_000_000
        }

        fn pixel_flux(&self, _dx: f64, _dy: f64, total_flux: f64) -> f64 {
            self.calls.set(self.calls.get() + 1);
            total_flux
        }
    }

    /// `splat_deposit` with `total_flux == 0` must be a no-op. Locks the
    /// "zero exposure budget = no work" early return — important for the
    /// motion-blur path where individual stamps may legitimately receive
    /// zero electrons (e.g. a star that briefly leaves the sensor footprint).
    #[test]
    fn zero_total_flux_is_a_noop() {
        let psf = PixelScaledAiryDisk::with_fwhm(1.0, Wavelength::from_nanometers(550.0));
        let mut buf = Array2::<f64>::zeros((50, 50));
        splat_deposit(&mut buf, 25.0, 25.0, 0.0, &psf);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    /// Pixels outside the buffer get silently skipped (no panic).
    /// Locks the bounds-check; the motion-blur path relies on this when
    /// a star straddles the sensor edge mid-stamp.
    #[test]
    fn out_of_bounds_centre_does_not_panic() {
        let psf = PixelScaledAiryDisk::with_fwhm(1.0, Wavelength::from_nanometers(550.0));
        let mut buf = Array2::<f64>::zeros((20, 20));
        // Centre wholly outside the buffer — every pixel in the footprint
        // is bounds-rejected, but the function must complete cleanly.
        splat_deposit(&mut buf, 1000.0, 1000.0, 100.0, &psf);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn sensor_spanning_footprint_only_visits_buffer_pixels() {
        let deposit = CountingDeposit {
            calls: Cell::new(0),
        };
        let mut buf = Array2::<f64>::zeros((5, 5));
        splat_deposit(&mut buf, 2.0, 2.0, 1.0, &deposit);
        assert_eq!(deposit.calls.get(), 25);
    }

    /// **Physics anchor**: depositing `T` electrons via an Airy disk into a
    /// generous box must integrate back to ≈ `T` (within the documented
    /// truncation budget). The first-zero × 2 box captures > 99% of the
    /// total flux for a well-sampled disk; the residual is genuine flux
    /// in higher-order rings beyond the box.
    ///
    /// This test fires if either:
    ///   (a) `pixel_flux_simpson` weights are wrong (over- or under-counts), or
    ///   (b) `footprint_pixels` is too tight (truncates more than ~1%).
    #[test]
    fn airy_deposit_conserves_flux_to_simpson_truncation_budget() {
        let psf = PixelScaledAiryDisk::with_fwhm(2.0, Wavelength::from_nanometers(550.0));
        // Buffer is large enough to contain the full first-zero-times-2 footprint
        // even with a sub-pixel offset; size is 2*footprint+8 each side.
        let footprint = psf.footprint_pixels() as usize;
        let n = footprint * 2 + 8;
        let mut buf = Array2::<f64>::zeros((n, n));
        let centre = (n as f64) * 0.5;
        let total_in = 1_000_000.0_f64;
        // Use an off-grid centre to exercise the sub-pixel Simpson path.
        splat_deposit(&mut buf, centre + 0.37, centre - 0.21, total_in, &psf);
        let total_out: f64 = buf.iter().sum();
        let captured = total_out / total_in;
        // The 99% lower bound is the documented Simpson-on-Airy capture
        // budget. Above it, accept a few-percent slop for the edge of
        // the bounding box where higher-order Airy rings live.
        assert!(
            captured > 0.99,
            "Airy capture {captured} < 99% (expected >0.99 for 2×first_zero box)"
        );
        // Upper sanity bound — capture cannot exceed 100% (Simpson rule
        // is an interpolant, not an extrapolant).
        assert!(captured <= 1.0 + 1e-9, "Airy capture {captured} > 100%");
    }

    /// Footprint must be at least 1 pixel — even a delta-function PSF
    /// covers its own pixel. Locks the `max(1.0)` clamp.
    #[test]
    fn footprint_is_at_least_one_pixel() {
        // Tiny FWHM forces first_zero below 1.0; the impl floors to 1.
        let psf = PixelScaledAiryDisk::with_fwhm(0.05, Wavelength::from_nanometers(550.0));
        assert!(psf.footprint_pixels() >= 1);
    }

    /// **INVARIANTS §2 lock**: the static-renderer entry
    /// (`add_stars_to_image`'s inner per-star deposit) and the
    /// motion-blur per-stamp entry (`SensorAccumulator::splat_psf`) must
    /// produce **byte-identical** buffers when given the same source +
    /// total_flux at the same position. After the PR 2 refactor both
    /// route through `splat_deposit`, so this test fires only if a
    /// future PR adds a fork that diverges them.
    ///
    /// The test exercises a sub-pixel offset and a non-integer
    /// `total_flux` so any rounding-order divergence between the two
    /// paths shows up.
    #[test]
    fn static_and_splat_psf_paths_are_byte_equal() {
        use crate::sims::motion_blur::SensorAccumulator;
        let psf = PixelScaledAiryDisk::with_fwhm(2.0, Wavelength::from_nanometers(550.0));
        let total_flux = 12_345.6789_f64;
        let px = 25.31;
        let py = 24.07;

        // Path A: direct splat_deposit (what add_stars_to_image runs through).
        let mut buf_a = Array2::<f64>::zeros((50, 50));
        splat_deposit(&mut buf_a, px, py, total_flux, &psf);

        // Path B: motion-blur SensorAccumulator (what splat_psf runs through).
        let mut acc = SensorAccumulator::zero(50, 50);
        acc.splat_psf(px, py, total_flux, &psf);
        let buf_b = acc.star_mean_electrons;

        // Byte-equality (not approx) — the two paths must agree to the bit.
        assert_eq!(
            buf_a.as_slice().unwrap(),
            buf_b.as_slice().unwrap(),
            "static and splat_psf paths must be byte-identical"
        );
    }
}

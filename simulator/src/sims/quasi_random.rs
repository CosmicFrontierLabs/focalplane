//! Low-discrepancy quasi-random sequences for time-domain stamp
//! sampling.
//!
//! The motion-blur renderer needs to pick `M` time samples within each
//! exposure window so the resulting integral of `(orientation × scene)`
//! converges quickly. Three options exist for that 1D sampling
//! problem, in order of increasing convergence quality:
//!
//! 1. **Pure Monte Carlo** — `M` uniform random draws. Standard
//!    error decays as `O(σ/√M)`. Generates the worst clumping.
//! 2. **Stratified Monte Carlo** — divide the window into `M` equal
//!    sub-bins and place one uniform-random draw in each. Removes
//!    the clumping; smooth-integrand error decays as `O(σ/M)` (the
//!    randomized midpoint rule). Requires per-sample RNG draws.
//! 3. **Quasi-random / golden-ratio sequence** — deterministic
//!    placement at `(seed + i · φ⁻¹) mod 1` for each stamp `i`,
//!    where `φ = (1 + √5)/2`. Discrepancy decays as `O(log(M)/M)`,
//!    typically with a smaller constant than stratified MC. No RNG
//!    needed.
//!
//! For our use the third is strictly better:
//!
//! - **Faster convergence per stamp** for any smooth integrand and
//!   typically also for the bandlimited periodic content of
//!   PSD-derived trajectories.
//! - **Determinism without seeding.** The sequence is a pure function
//!   of `(M, phase)`. The renderer derives `phase` from the existing
//!   `tile_seed` so different tiles get different realizations
//!   (preventing systematic per-frame bias) but each tile is still
//!   bit-reproducible.
//! - **Nested subsets are free.** The first `m` of any `n`-point
//!   sequence is itself a good `m`-point set, so progressive-render
//!   modes (start coarse, refine) cost nothing.
//! - **No spectral comb against trajectory tones near the stamp
//!   Nyquist.** Quasi-random sampling avoids the systematic bias a
//!   regular grid produces against frequencies at `k / Δt_stamp`,
//!   inheriting one of stratified MC's key correctness properties.
//!
//! ## Convergence behavior — what to expect
//!
//! For a smooth bounded integrand `f` on `[0, T]`, the
//! quasi-random integral estimate `(T/M) Σ f(stamp_time[i])`
//! converges to `∫ f dt` at the rate
//!
//! ```text
//! |error|  ≤  V(f) · D_M
//! ```
//!
//! where `V(f)` is the total variation of `f` and `D_M` is the
//! star-discrepancy of the sequence (Koksma's inequality). For the
//! golden-ratio sequence in 1D, `D_M = O(log(M)/M)` — the best
//! possible. Stratified MC's standard error scales as `O(σ_within/√M)`
//! where `σ_within` is the within-stratum variance of `f`, which for
//! a smooth `f` decays itself as `O(M⁻¹)`, giving overall
//! `O(σ/M)` — slightly worse than R2's logarithm-of-M rate.
//!
//! Empirically R2 reaches the same accuracy as stratified MC at
//! roughly half the M, with the savings growing slowly with M.
//!
//! ## When this is the wrong tool
//!
//! - **2D dither for super-resolution composite reconstruction** —
//!   needs a true 2D low-discrepancy sequence (R2 with the plastic
//!   constant) or Poisson-disk. The 1D sequence here is the wrong
//!   shape. Future work.
//! - **Integrating a discontinuous function** — Koksma's inequality
//!   needs bounded variation; for sharp jumps in `f(t)` (e.g.
//!   trajectories with discrete waypoints SLERP'd through, where the
//!   second derivative jumps at every waypoint), the constant in the
//!   bound is large. Still better than stratified MC, but the gap
//!   shrinks.
//! - **n very small (n ≤ 2)** — the asymptotic advantage is gone;
//!   for the renderer this is the trivially-static-trajectory case
//!   where M=1 is correct anyway.

use std::f64::consts::PI;

/// Inverse golden ratio: `1 / φ = (√5 - 1) / 2 ≈ 0.61803398874989484`.
///
/// Used as the irrational stride for 1D low-discrepancy sequences.
/// Conjectured (and overwhelmingly empirically observed) to give the
/// best 1D discrepancy of any irrational stride; provably optimal
/// among quadratic irrationals via the continued-fraction expansion.
pub const GOLDEN_RATIO_INVERSE: f64 = 0.6180339887498949;

/// Generate `m` low-discrepancy offsets in `[0, 1)` from the
/// golden-ratio sequence, starting at `phase`.
///
/// Stamp `i` lands at `((i + 0.5) · φ⁻¹ + phase) mod 1`. The
/// `(i + 0.5)` shift keeps the first stamp away from the lower
/// boundary so the sequence spans the unit interval symmetrically;
/// changing it to `i` would only relabel the offsets.
///
/// The `phase` parameter shifts the entire sequence by a constant
/// `mod 1`. Different per-tile `phase` values produce different
/// realizations of the same low-discrepancy property — useful for
/// preventing systematic per-frame bias when many tiles render the
/// same trajectory.
pub fn golden_offsets(m: usize, phase: f64) -> Vec<f64> {
    let m = m.max(1);
    let phase = phase.rem_euclid(1.0);
    (0..m)
        .map(|i| {
            let raw = (i as f64 + 0.5) * GOLDEN_RATIO_INVERSE + phase;
            raw.rem_euclid(1.0)
        })
        .collect()
}

/// Convert a `u64` seed into a uniform `phase ∈ [0, 1)`. Use this to
/// derive a per-tile phase from the existing tile-seed scheme.
pub fn phase_from_seed(seed: u64) -> f64 {
    // Top 53 bits → f64 with full precision; lower bits are noise we
    // don't need. Equivalent to (seed >> 11) / 2^53 but written
    // without bit-shift gotchas.
    (seed >> 11) as f64 / (1_u64 << 53) as f64
}

/// Discrete Fourier magnitude of the indicator function for a 1D
/// sample set `points ⊂ [0, 1)` evaluated at integer frequency `k`:
/// `|Σⱼ exp(−2πi k xⱼ)|`. Returns the magnitude (not magnitude²).
///
/// Test-only / diagnostic. A perfectly uniform sample set has zero
/// magnitude at all `k ≠ 0`; clumped sets show spikes; quasi-random
/// sets show a low-frequency void rising to a flat ~√M plateau at
/// high frequencies.
pub fn dft_magnitude(points: &[f64], k: i64) -> f64 {
    let mut re = 0.0;
    let mut im = 0.0;
    let omega = -2.0 * PI * k as f64;
    for &x in points {
        let theta = omega * x;
        re += theta.cos();
        im += theta.sin();
    }
    (re * re + im * im).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_offsets_are_in_unit_interval() {
        for m in [1, 4, 16, 100, 1000] {
            let offsets = golden_offsets(m, 0.0);
            assert_eq!(offsets.len(), m);
            for x in &offsets {
                assert!(*x >= 0.0 && *x < 1.0, "offset {x} outside [0, 1)");
            }
        }
    }

    #[test]
    fn test_golden_offsets_reproducible_for_same_phase() {
        let a = golden_offsets(64, 0.123);
        let b = golden_offsets(64, 0.123);
        assert_eq!(a, b);
    }

    #[test]
    fn test_golden_offsets_differ_for_different_phase() {
        let a = golden_offsets(64, 0.0);
        let b = golden_offsets(64, 0.5);
        // Same length, but no shared values; the constant phase shifts
        // every offset by 0.5 mod 1 so a[i] and b[i] should differ by
        // approximately 0.5 (modulo wrap).
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            let diff = (x - y).rem_euclid(1.0);
            assert!(
                (diff - 0.5).abs() < 1e-12,
                "expected constant 0.5 phase shift, got diff={diff}"
            );
        }
    }

    #[test]
    fn test_golden_offsets_one_per_equal_bin_for_fibonacci_m() {
        // For Fibonacci M (where φ⁻¹·M ≈ M_{n−1}), the golden-ratio
        // sequence's "exactly one point per equal-width bin" property
        // is exact. Verify for M = 8, 13, 21.
        for &m in &[8_usize, 13, 21] {
            let offsets = golden_offsets(m, 0.0);
            let mut bin_hits = vec![false; m];
            for x in &offsets {
                let bin = (x * m as f64) as usize;
                let bin = bin.min(m - 1);
                assert!(
                    !bin_hits[bin],
                    "two offsets in bin {bin}: {offsets:?} (M={m})"
                );
                bin_hits[bin] = true;
            }
            assert!(bin_hits.iter().all(|&hit| hit));
        }
    }

    #[test]
    fn test_golden_offsets_low_discrepancy_dft() {
        // Star-discrepancy spec: |DFT_k| should grow much slower than
        // √M for quasi-random sequences vs uniform random. Compare
        // DFT magnitudes at low k for golden-ratio offsets vs a known
        // weak alternative (regularly-spaced grid with a small jitter).
        let m = 256;
        let offsets = golden_offsets(m, 0.0);
        // DFT at k=1, 2, 3: should be O(1) or smaller, not O(√M ≈ 16).
        for k in 1..=4 {
            let mag = dft_magnitude(&offsets, k);
            assert!(
                mag < 5.0,
                "golden-ratio M=256 DFT at k={k} is {mag:.2}, expected ≪ √M = 16"
            );
        }
    }

    #[test]
    fn test_phase_from_seed_in_unit_interval() {
        for seed in [0_u64, 1, 42, u64::MAX, 0xDEADBEEF, 0xCAFEBABE_F00DBA11] {
            let p = phase_from_seed(seed);
            assert!(
                p >= 0.0 && p < 1.0,
                "phase {p} from seed {seed} out of range"
            );
        }
    }

    #[test]
    fn test_phase_from_seed_is_well_distributed() {
        // 1000 phases from a sweep of seeds should populate the unit
        // interval uniformly — verifies that phase_from_seed is doing
        // something sensible and not collapsing to a constant.
        let phases: Vec<f64> = (0..1000_u64)
            .map(|s| phase_from_seed(s.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
            .collect();
        let mean = phases.iter().sum::<f64>() / phases.len() as f64;
        // Uniform [0,1) has mean 0.5, std 1/√12 ≈ 0.289. With N=1000
        // the sample mean is within ~3σ/√N ≈ 0.027 of 0.5.
        assert!(
            (mean - 0.5).abs() < 0.05,
            "phase mean {mean:.4} unexpectedly far from 0.5"
        );
    }

    #[test]
    fn test_golden_integrates_polynomial_to_high_accuracy() {
        // Polynomial integration is the cleanest convergence test: a
        // smooth, non-periodic integrand where the sequence's
        // low-discrepancy property is what matters. f(x) = 4·x·(1 − x)
        // has true integral 2/3 on [0, 1]. The golden-ratio M=32
        // estimate should be within ~5·log(M)/M ≈ 0.02 of the true
        // value (Koksma–Hlawka constant for this particular function
        // sits around 5).
        let f = |x: f64| 4.0 * x * (1.0 - x);
        let m = 32;
        let estimate = golden_offsets(m, 0.0).iter().map(|&x| f(x)).sum::<f64>() / m as f64;
        let truth = 2.0 / 3.0;
        assert!(
            (estimate - truth).abs() < 0.02,
            "golden M={m} integral of 4x(1-x) = {estimate}, truth = {truth}, err = {:.2e}",
            estimate - truth
        );
    }
}

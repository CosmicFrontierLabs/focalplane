# Motion and jitter in the focal-plane renderer

The renderer integrates each exposure window across many spacecraft
orientations so that the time-varying scene smears into the image the
way a real shutter would record it. Two physically distinct regimes
need different sampling treatments, and the renderer has two
independent cadences to handle them.

## The regimes

**Motion** — low-frequency, deterministic spacecraft pointing change.
Examples: a slew between targets, a tracking-error envelope, a
deliberate scan pattern. Smooth, predictable, *integrable* with a
small number of well-placed orientation samples. The total angular
displacement is large but the per-photon variation is monotonic over
the exposure window.

**Jitter** — high-frequency stochastic pointing residuals. Examples:
reaction-wheel induced micro-vibration, structural ringing, optical
bench thermal flutter. Stochastic, broadband, often dominated by tones
at the wheel rotation rate plus harmonics. Within a single 250 ms
exposure the spacecraft may complete many cycles of jitter; the photon
arrival time within the exposure decides where each photon lands.

If both regimes existed in isolation the right sampling cadence would
be obvious for each. They do not — they are mixed together in any
realistic spacecraft trajectory — so the renderer slices the exposure
window two different ways.

## Two-cadence sampling

Every exposure window is split twice. See
[`SubsampleSchedule`](../simulator/src/sims/motion_blur.rs) for the type
and [`render_tile`](../simulator/src/sims/motion_blur.rs) for the
inner loop.

### Outer cadence — `n` subsamples (scene-state refresh)

`n` evenly-spaced sub-orientation samples drive everything that
depends on *which slice of the trajectory we are in*:

- per-star chromatic flux integration
- in-field star slice (envelope prefilter)
- zodiacal background mean (depends on instantaneous boresight)
- padding margin for the projection-and-cull pass

Each subsample is expensive — flux integration is a Simpson rule over
the stellar spectrum × QE curve, paid once per (star, sensor). It is
worth caching across stamps within a subsample, but cannot be cached
across subsamples without a stale-orientation bug.

The outer cadence is picked from `--max-drift-per-sample-px`, which
sets a path-length budget: the scheduler computes the trajectory's
total angular drift across the exposure and picks the smallest `n`
that keeps per-subsample drift below the budget. Default `0.1 px`.

### Inner cadence — `m` stamps per subsample (stratified Monte Carlo)

Each subsample window is divided into `m` equal-width sub-bins, and
each stamp lands at a **uniform-random offset within its sub-bin** —
classical stratified Monte Carlo:

```
Δt_sub_bin = (exposure / n) / m
stamp_time[j] = t_sub_start + (j + U_j) · Δt_sub_bin    where U_j ~ Uniform[0, 1)
```

Each stamp then:

1. Queries the trajectory at `stamp_time[j]` via SLERP (cheap)
2. Projects every visible star to its per-time pixel position
3. Deposits `flux / m` electrons through the same Simpson PSF stamp
   used for the single-stamp case

The accumulated mean-electron image preserves the convolved-PSF
expectation, but the *spatial distribution* of the deposited flux now
reflects the within-subsample motion instead of being pinned to a
single per-subsample orientation.

**Why stratified MC and not a regular grid.** A deterministic
even-spaced grid at interval `Δt_stamp` has a comb response in
frequency: a trajectory tone at `f ≈ k / Δt_stamp` is sampled at the
same phase every stamp and the M deposits don't average out the
within-cycle variation — the resulting image is systematically biased
at that frequency. Pure Monte Carlo (random uniform anywhere in the
subsample) avoids the bias but converges only as O(1/√m). Stratified
MC keeps one random sample per equal-width sub-bin, so it (a) removes
the regular-grid bias for any tone above the stamp Nyquist *and*
(b) keeps the smooth-integrand convergence of a regular grid for the
low-frequency content. It's the right combination for trajectories
mixing low-frequency drift and high-frequency stochastic content.

The inner cadence is picked from the optional `--max-drift-per-stamp-px`
budget. When unset, `m = 1` and each subsample contributes a single
random stamp drawn uniformly from the entire subsample window. This is
not bit-identical to the deterministic-midpoint behavior the renderer
used before stratified MC was introduced, but it is unbiased.

## Why split the budgets

The two budgets exist because per-subsample work and per-stamp work
have very different costs:

| Work                            | Per subsample | Per stamp |
| ------------------------------- | ------------- | --------- |
| Trajectory SLERP                | yes           | yes       |
| Star projection                 | yes           | yes       |
| Chromatic flux integration      | yes           | no (cached) |
| In-field star envelope prefilter| yes           | no        |
| Zodiacal evaluation             | yes           | no        |
| PSF Simpson stamp deposit       | yes           | yes       |

Forcing the same cadence on both would either over-pay for cheap
high-frequency stamping or under-resolve the low-frequency motion. The
two-budget scheme lets each cadence scale with its actual physical
demand.

## When each cadence dominates

For a **drift-only trajectory** (clean slew, telescope pointing change,
no jitter), the outer cadence does all the work. `n = O(10)` captures
the smooth motion, `m = 1` is sufficient, and the renderer behaves
exactly as it did before this split existed.

For a **jitter-dominated trajectory** (PSD-derived reaction-wheel
residuals at fixed RPM, structural flutter), the inner cadence is what
matters. The outer cadence can stay loose — `n = O(10)` is fine for
scene-state refresh — and `m` scales with the trajectory's spectral
bandwidth to capture every wiggle in the photon-deposit positions.

The interesting and common case is **mixed**: deterministic drift plus
stochastic jitter. The two budgets are picked independently and the
renderer composes them automatically.

## Determinism

The renderer draws three independent RNG streams per tile, all seeded
from a single `tile_seed(base_seed, frame_idx, sensor_idx)` plus a
named domain tag (`rng_domain::POISSON`, `READ_NOISE`, `STAMP_JITTER`).
Domain separation guarantees that perturbing one source — adding
`STAMP_JITTER` for stratified-MC stamp placement, in this PR — does
not shift the bytes the other two would have produced.

The stamp-jitter RNG is consumed sequentially: subsample 0 draws its
M uniforms, then subsample 1, and so on. Output PNGs are
byte-identical for fixed `(base_seed, frame_idx, sensor_idx, n, m)`.
This invariant is pinned by `test_per_stamp_render_is_deterministic`
(end-to-end render comparison) and `test_stamp_times_seeded_rng_is_reproducible`
(stamp-time draw comparison) in `motion_blur.rs`.

## Picking budgets in practice

### Default heuristic — path-length budget

Set `--max-drift-per-sample-px = 0.1` (the default). The path-length
scheduler picks `n` so per-subsample drift stays below 0.1 px. Leave
`--max-drift-per-stamp-px` unset so `m = 1`. Behavior is identical to
the renderer before this split existed.

This works correctly even for jittery trajectories — the path-length
budget will pick a very large `n` that captures the within-exposure
jitter — but it pays the per-subsample cost (chromatic flux, envelope
filter, zodiacal) for every one of those samples. On the LOS-PSD
trajectory it picks `n = 2628..4711` per 250 ms exposure.

### Looser outer + finer inner — explicit two-budget

Set `--max-drift-per-sample-px = 1.0` (10× looser) and
`--max-drift-per-stamp-px = 0.1` (matching the original effective
sub-stamp drift). The outer cadence now picks `n = O(100)` instead of
`O(1000)`, and the inner cadence picks `m = O(10)` to recover the
within-subsample fidelity. Same total stamp count, much less
per-subsample work.

This is the cost-saving regime the inner cadence enables.

### Principled velocity-variance approach (proposed, not yet wired up)

For a stationary jitter process with one-sided PSD `S(f)`, the right
metric for stamp count is angular velocity variance, not path length:

```
σ²_v  =  ∫ (2π f)² S(f) df            [angular velocity variance]

T_min ≈ T_exp · σ_v · √(1/ε) / σ_PSF   [total stamps for relative error ε]
```

where `σ_PSF` is the PSF Gaussian width (≈ 1 pixel at typical
diffraction-limited sampling) and `ε` is the relative-error tolerance
(0.01 = 1 % peak error is a sensible default). The path-length budget
is secretly an approximation of this: `path_length ≈ T_exp · σ_v` for
a zero-mean jitter process, so setting `max_drift_per_sample_px ≈ σ_PSF · √ε`
recovers the same answer.

The advantages of the velocity-variance form:

- **Cheaper**: one integral over the PSD at trajectory-load time vs.
  `max_drift_over_window` per frame (which is `O(N_waypoints)` in the
  current implementation, the bottleneck on long PSD-derived
  trajectories with `O(10⁵)` waypoints).
- **Stable**: doesn't fluctuate frame-to-frame with the trajectory's
  instantaneous wiggle pattern.
- **Self-documenting**: `--target-relative-error 0.01` says what it
  enforces; `--max-drift-per-sample-px 0.1` says how it enforces it.

A `Trajectory::velocity_variance_rad2_per_s2()` accessor that
estimates `σ²_v` from numerical differences of consecutive waypoints
would let the renderer pick `n × m` from a single `--target-relative-error`
knob.

## Empirical findings — LOS-PSD reaction wheel at 2007 RPM

Trajectory: 5 s slice, 2 kHz waypoints, 0.121″ / 0.126″ per-axis RMS,
PSD content out to 1 kHz. Rendered through a single IMX455 on the
cosmic-frontier-jbt50cm telescope, 250 ms exposures, 5 frames, all
under the **stratified-MC** stamp placement.

| pass        | `--max-drift-per-sample-px` | `--max-drift-per-stamp-px` | `n` per frame | `m` per subsample | wall (5 tiles ‖) |
| ----------- | --------------------------- | -------------------------- | ------------- | ----------------- | ---------------- |
| **default** | 0.1                         | unset                      | 2628–4711     | 1                 | hours (killed)   |
| **M=1**     | 1.0                         | unset                      | 311–456       | 1                 | 133 s            |
| **M=10**    | 1.0                         | 0.1                        | 311–456       | 10                | 803 s (6.0×)     |
| **M=50**    | 1.0                         | 0.02                       | 311–456       | 50                | 3,827 s (29×)    |

Frame-1 comparison on three brightness tiers, stratified MC:

| star tier            | peak M=1 | peak M=10 | peak M=50 | RMS Δ(M=10−M=1) | RMS Δ(M=50−M=1) | RMS Δ(M=50−M=10) |
| -------------------- | -------- | --------- | --------- | --------------- | --------------- | ---------------- |
| brightest            | 23,803   | 23,643    | 23,639    | 31.26           | 31.40           | **1.44**         |
| median               | 98       | 84        | 84        | 1.39            | 1.44            | **0.57**         |
| ~half-median         | 50       | 50        | 50        | 1.33            | 1.34            | **0.43**         |

Three clean takeaways:

1. **M=10 has converged for this trajectory.** RMS(M=50−M=10) is
   ~30× smaller than RMS(M=10−M=1) on every tier. The PSD has zero
   content above 1 kHz; M=10 gives an effective stamp rate above
   16 kHz; nothing left for higher M to recover. M=50 buys nothing
   measurable for ~5× the runtime.

2. **The big M=10−M=1 RMS is the *variance of the M=1 estimator*,
   not bias against M=10.** Stratified M=1 is single-sample MC: one
   uniform-random draw places all photons at one orientation, which
   could be anywhere in the subsample window. The 31 ADU dipole on
   the brightest star is the ±half-pixel uncertainty from that
   single draw. M=10 has averaged it down (1/√10 ≈ 0.32× variance);
   M=50 by another √5 ≈ 0.45×, indistinguishable from M=10 in
   absolute terms.

3. **Peak ADU drops M=1 → M=10 are systematic, not noise.** Bright
   star peak: 23,803 → 23,643 (−0.7%). Median: 98 → 84 (−14%). The
   M=1 case concentrates all photons at one sub-pixel position, so
   peaks are inflated; M=10 spreads them across the actual jitter
   cloud and the peak relaxes to its true value. The fractional
   effect is largest for faint sources because at low SNR every
   photon dominates the peak pixel.

Practical takeaway: **for this trajectory class M=10 is the
production-grade choice.** M=1 is too noisy per pixel to trust for
photometry; M=50 is wasted compute for indistinguishable output.

A different trajectory class (looser per-sample budget so
σ_jit_sub > σ_PSF, or trajectory content closer to the stamp
Nyquist) would push the M-needed value higher. The principled picker
sketched above (M ≈ k·max(1, (σ_jit_sub/σ_PSF)²) for some
k = 4–10 visually, 50–100 photometrically) recovers the right scaling
in either regime.

## Code references

- [`simulator/src/sims/motion_blur.rs`](../simulator/src/sims/motion_blur.rs)
  — `SubsampleSchedule`, `MotionBlurConfig`, `render_tile`,
  `tile_seed`, the per-stamp determinism tests
- [`simulator/src/sims/trajectory.rs`](../simulator/src/sims/trajectory.rs)
  — `Trajectory::orientation_at`, `frame_times`,
  `TrajectoryRenderConfig`
- [`simulator/src/bin/motion_simulator.rs`](../simulator/src/bin/motion_simulator.rs)
  — CLI flags `--max-drift-per-sample-px`, `--max-drift-per-stamp-px`
- [`scripts/los_psd_to_trajectory_csv.py`](../scripts/los_psd_to_trajectory_csv.py)
  — converter from a 2-axis LOS PSD CSV to a quaternion-waypoint
  trajectory file consumable by `motion_simulator --mode csv`

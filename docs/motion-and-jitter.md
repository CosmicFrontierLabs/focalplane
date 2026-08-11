# Motion and jitter in the focal-plane renderer

The renderer integrates each exposure window across many spacecraft
orientations so that the time-varying scene smears into the image the
way a real shutter would record it. Two physically distinct regimes
mix in any realistic trajectory, but the renderer handles them with a
single sampling cadence — they only differ in *how much* of it they
demand.

## The regimes

**Motion** — low-frequency, deterministic spacecraft pointing change.
Examples: a slew between targets, a tracking-error envelope, a
deliberate scan pattern. Smooth, predictable, with monotonic
per-photon variation over the exposure window.

**Jitter** — high-frequency stochastic pointing residuals. Examples:
reaction-wheel induced micro-vibration, structural ringing.
Stochastic, broadband, often dominated by tones at the wheel rotation
rate plus harmonics. Within a single 250 ms exposure the spacecraft
may complete many cycles of jitter; the photon arrival time within
the exposure decides where each photon lands.

## Single stamp loop, single budget

Every exposure window is sampled by `stamps_per_exposure` PSF stamps
placed via the **1D golden-ratio low-discrepancy sequence**:

```
φ⁻¹  =  (√5 − 1)/2  ≈  0.6180339887
stamp_time[j]  =  frame_start  +  ((j + 0.5)·φ⁻¹  +  phase) mod 1  ·  exposure
```

Each stamp:

1. Queries the trajectory at `stamp_time[j]` via SLERP (cheap)
2. Projects every visible star to its per-time pixel position
3. Deposits `flux / stamps_per_exposure` electrons through the
   Simpson PSF stamp

The accumulated mean-electron image is the discrete approximation of
the continuous time-integral of the per-photon scene under the
trajectory's instantaneous orientation density.

`stamps_per_exposure` is picked from `--max-drift-per-stamp-px`: the
scheduler computes the trajectory's total angular path length across
the exposure and chooses the smallest count that keeps per-stamp
drift below the budget. Default `0.1 px`. See
[`SubsampleSchedule`](../simulator/src/sims/motion_blur.rs) for the
type, [`render_tile`](../simulator/src/sims/motion_blur.rs) for the
loop, and [`quasi_random`](../simulator/src/sims/quasi_random.rs) for
the sequence implementation + convergence discussion.

**Why golden-ratio and not stratified MC or a regular grid.**

- **Regular grid** has a comb spectral response: trajectory content at
  `f ≈ k/Δt_stamp` is sampled at the same phase every stamp and the
  deposits don't average out the within-cycle variation, giving a
  systematic bias at that frequency.
- **Stratified Monte Carlo** (one uniform-random sample per equal sub-bin)
  removes the comb bias and gives `O(σ_within/√M)` convergence —
  decent, but requires per-stamp RNG draws and converges as `O(1/M)`
  for smooth integrands.
- **Golden-ratio sequence** is deterministic, has discrepancy
  `O(log(M)/M)` (best possible in 1D), no RNG draws needed, and the
  first `m` of any `n`-point sequence is itself a good `m`-point set
  (free progressive-render mode). For our use case it strictly
  dominates stratified MC.

`phase` is per-tile, derived deterministically from the tile seed so
different `(frame, sensor)` tiles get different realizations of the
same low-discrepancy property — preventing systematic per-frame bias
without sacrificing reproducibility.

## Envelope prefilter — the only "scene-state" knob

Before the stamp loop runs, the renderer prunes the catalog star list
down to the candidates that could land on a sensor *at any moment
during the exposure*. This is done with a single mid-frame projection
plus a padded focal-plane AABB:

```
padding_mm  =  PSF_extent_padding  +  peak_excursion_rad · focal_length
```

where `peak_excursion_rad` is the maximum angular distance any
trajectory waypoint in the exposure window deviates from the
mid-frame orientation. Computed by
[`Trajectory::peak_excursion_rad`](../simulator/src/sims/trajectory.rs)
in one linear pass over the in-window waypoints.

A single mid-frame projection per star plus a padded AABB is
correctness-equivalent for any star whose mid-frame position is within
`peak_excursion_rad` of the focal plane. Candidate stars that turn
out to be off-sensor at a given stamp time get filtered naturally by
`project_to_sensor` returning `None`.

Zodiacal mean and the candidate star list are frame-scoped, computed
once per `(frame, sensor)`; only the per-stamp orientation lookup +
projection + PSF deposit run inside the stamp loop.

## When the budget needs to scale up

For a **drift-only trajectory** (clean slew, telescope pointing change,
no jitter), `stamps_per_exposure = O(10)` captures the smooth motion
and the renderer is essentially free.

For a **jitter-dominated trajectory** (PSD-derived reaction-wheel
residuals at fixed RPM, structural flutter), the path length per
exposure is much larger and `stamps_per_exposure` scales with the
trajectory's spectral bandwidth — typically `O(10³–10⁴)` for wheel-
tone PSDs in 250 ms exposures.

The mixed case (deterministic drift + stochastic jitter) is the
common one; the path-length budget composes the two regimes
automatically.

## Determinism

The renderer draws two independent RNG streams per tile (Poisson
photon noise + Gaussian read noise) and one deterministic per-tile
phase for stamp placement. All three derive from a single per-tile
seed `tile_seed(base_seed, frame_idx, sensor_idx)` XOR'd with a
named domain tag (`POISSON_DOMAIN`, `READ_NOISE_DOMAIN`,
`STAMP_PHASE_DOMAIN`). Domain separation guarantees perturbing one
source cannot shift the bytes another would have produced.

Stamp times are deterministic from `(stamps_per_exposure, phase)` —
no per-stamp RNG draws — so output PNGs are byte-identical for
fixed `(base_seed, frame_idx, sensor_idx, stamps_per_exposure)`.
This invariant is pinned by `test_per_stamp_render_is_deterministic`
(end-to-end render comparison) and `test_stamp_times_seeded_rng_is_reproducible`
(stamp-time draw comparison) in `motion_blur.rs`.

## Picking the stamp count

Set `--max-drift-per-stamp-px = b` (default `0.1`). The scheduler
computes the trajectory's total angular path length over the exposure
and picks

```
stamps_per_exposure = ceil(total_drift_rad / (b · pixel_scale_rad))
```

Tighten `b` to capture higher-frequency content; loosen to spend less.
For the LOS-PSD trajectory at the default budget, that's
~2,600–4,700 stamps per 250 ms exposure.

### Principled velocity-variance picker (proposed, not yet wired up)

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
a zero-mean jitter process, so setting `max_drift_per_stamp_px ≈ σ_PSF · √ε`
recovers the same answer.

Advantages of the velocity-variance form over the current
path-length budget:

- **Cheaper**: one integral over the PSD at trajectory-load time vs.
  `max_drift_over_window` per frame (currently `O(N_waypoints)` per
  frame, the bottleneck on long PSD-derived trajectories with
  `O(10⁵)` waypoints).
- **Stable**: doesn't fluctuate frame-to-frame with the trajectory's
  instantaneous wiggle pattern.
- **Self-documenting**: `--target-relative-error 0.01` says what it
  enforces; `--max-drift-per-stamp-px 0.1` says how it enforces it.

A `Trajectory::velocity_variance_rad2_per_s2()` accessor that
estimates `σ²_v` from numerical differences of consecutive waypoints
would let the renderer pick `stamps_per_exposure` from a single
`--target-relative-error` knob.

## Empirical findings — LOS-PSD reaction wheel at 2007 RPM

Trajectory: 5 s slice, 2 kHz waypoints, 0.121″ / 0.126″ per-axis RMS,
PSD content out to 1 kHz. Rendered through a single IMX455 on the
cosmic-frontier-jbt50cm telescope, 250 ms exposures, 5 frames, all
under the **stratified-MC** stamp placement.

| pass        | `--max-drift-per-stamp-px` | total stamps | wall (5 tiles ‖) |
| ----------- | -------------------------- | ------------ | ---------------- |
| **default** | 0.1                        | 2628–4711    | hours (killed)   |
| **M=1**     | 1.0                        | 311–456      | 133 s            |
| **M=10**    | 0.1                        | 3110–4560    | 803 s (6.0×)     |
| **M=50**    | 0.02                       | 15550–22800  | 3,827 s (29×)    |

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
   not bias against M=10.** Stratified M=1 places all photons at one
   uniform-random orientation, which could be anywhere in the
   exposure. The 31 ADU dipole on the brightest star is the
   ±half-pixel uncertainty from that single draw. M=10 has averaged
   it down (1/√10 ≈ 0.32× variance); M=50 by another √5 ≈ 0.45×,
   indistinguishable from M=10 in absolute terms.

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

A trajectory with content closer to the stamp Nyquist would push the
required stamp count higher. The principled picker sketched above
(`stamps_per_exposure ≈ k · max(1, (σ_jit/σ_PSF)²)` for some
`k = 4–10` visually, `50–100` photometrically) recovers the right
scaling in either regime.

## Code references

- [`simulator/src/sims/motion_blur.rs`](../simulator/src/sims/motion_blur.rs)
  — `SubsampleSchedule`, `MotionBlurConfig`, `render_tile`,
  `tile_seed` + domain constants, the per-stamp determinism tests
- [`simulator/src/sims/quasi_random.rs`](../simulator/src/sims/quasi_random.rs)
  — golden-ratio sequence, per-tile phase derivation, convergence
  notes, spectral-discrepancy tests
- [`simulator/src/sims/trajectory.rs`](../simulator/src/sims/trajectory.rs)
  — `Trajectory::orientation_at`, `Trajectory::peak_excursion_rad`,
  `frame_times`,
  `TrajectoryRenderConfig`
- [`simulator/src/bin/motion_simulator.rs`](../simulator/src/bin/motion_simulator.rs)
  — CLI flag `--max-drift-per-stamp-px`
- [`scripts/los_psd_to_trajectory_csv.py`](../scripts/los_psd_to_trajectory_csv.py)
  — converter from a 2-axis LOS PSD CSV to a quaternion-waypoint
  trajectory file consumable by `motion_simulator --mode csv`

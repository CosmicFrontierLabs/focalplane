# Renderer invariants

Properties the rendering pipeline preserves for every source type (stars,
Sérsic galaxies, solar-system bodies). Code comments cite these as
`INVARIANTS §N`. Each one is pinned by at least one test; a change that
breaks the test breaks the invariant.

## §1 — Shot noise is sampled once, on means, after all deposits

**Statement**: Sources deposit *mean* electrons only. Poisson shot noise
is drawn after every deposit for the frame has landed, on the
accumulated mean image, never inside a deposit and never per motion
stamp.

**Why**: A deposit that sampled noise would break determinism (§5) and
the static/motion equivalence (§2), and per-stamp draws would multiply
RNG work by the stamp count. Statistically, a sum of *independent*
Poisson draws is itself Poisson with the summed mean, so splitting the
draw across independent components does not bias the noise; what does
is drawing on a fraction of the mean and rescaling (`N · Poisson(λ/N)`
has variance `N · λ`), or reusing one RNG stream for two draws so they
are correlated.

**Where**:

- Motion path: `sims/motion_blur.rs` `render_tile_array_roi`, one
  `apply_poisson_photon_noise` call on the combined tile mean, seeded
  `tile_seed ^ POISSON_DOMAIN`.
- Static path: `image_proc/render.rs` `Renderer::render_with_options`,
  which draws the star image and the zodiacal background as two
  independent components. Its star draw and sensor-noise draw share a
  seed; see issue #195.

**Test**: `scene_galaxy::tests::variance_equals_mean_poisson_identity`
(`#[ignore]`, slow): per-pixel variance over 200 seeded renders must
equal the per-pixel mean.

## §2 — Static and motion paths agree byte-for-byte in the degenerate case

**Statement**: Given the same source, flux and position, the static
renderer and the motion-blur renderer deposit byte-identical buffers.
Both route through `image_proc::deposit::splat_deposit`.

**Why**: It is the sharpest check that the two paths share their
projection, flux integration and deposit. Any fork between them shows up
as a byte difference rather than a subtle photometric drift.

**Tests**:

- `image_proc::deposit::tests::static_and_splat_psf_paths_are_byte_equal`
  (stars)
- `scene_galaxy::tests::galaxy_static_and_motion_paths_are_byte_equal`
  (galaxies)
- `sims::motion_blur::tests::test_render_one_frame_roi_full_sensor_is_byte_equal`
  (a full-sensor ROI matches the unwindowed render)

## §3 — Flux conservation within a documented truncation budget

**Statement**: The mean electrons a source deposits over its footprint
equal its expected total over the integration window, less a known
truncation fraction.

- PSF stars: the footprint is `2 × first_zero` of the Airy disk.
- Sérsic galaxies: the footprint extends to where surface brightness
  falls to `1e-4 · I_e`. `I_e` is normalised to the *full* analytic
  Sérsic integral over `[0, ∞)`, not to the truncated box, so the
  on-image total sits a known `1 - completeness` below the catalog
  flux instead of silently inflating the visible ellipse. Enclosed
  fractions exceed 0.99 for `n = 1` and reach about 0.97 for `n = 4`.

**Tests**: `image_proc::sersic_splat::tests::sersic_deposit_conserves_flux_within_truncation_budget_n_equals_1`
and `..._n_equals_4`, `integrated_flux_is_plate_scale_invariant_when_well_resolved`,
and `scene_galaxy::tests::end_to_end_flux_roundtrip_through_renderer`.

## §4 — Chromatic effective PSF is per source

**Statement**: Every source carries its own electron-weighted effective
PSF, `SourceFlux::disk`, computed by `photometry::photon_electron_fluxes`
from that source's spectrum and the sensor QE. The renderer never bakes
in a global FWHM.

**Why**: The Airy disk scales with wavelength, so sources of different
colour have different effective widths on the same sensor.

## §5 — Seeded renders are reproducible

**Statement**: For a fixed `(base_seed, frame_idx, sensor_idx)`, output
is bit-exact. Deposits are deterministic and side-effect free (see the
`MeanFluxDeposit` contract); all randomness flows from
`sims::motion_blur::tile_seed` through named sub-stream domains.

**Tests**: `sims::motion_blur::tests::test_tile_seed_is_deterministic_and_varies`,
`test_render_one_frame_is_deterministic`,
`test_per_stamp_render_is_deterministic` and
`test_frame_sensor_tile_parallelism_determinism`.

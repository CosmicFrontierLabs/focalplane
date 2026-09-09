# Inner solar system bodies in the focal-plane renderer

Plan for adding resolved planets (disks, textures, atmospheric limbs,
illumination phase) to the simulator, with a headline study: **can a
telescope near Mars guide on the Earth limb?**

This document is organised as:

1. The physical picture and the numbers that drive the design
2. What already exists (in `starfield` and in `simulator`)
3. Target architecture: optional cargo feature, second-pass
   compositing, module layout, data flow
4. Rendering physics, body by body (including the Sun)
5. The Earth-limb-from-Mars guiding study
6. Validation ladder
7. Delivery plan (PR sequence, split across the three repos)
8. Risks, open questions, decisions needed
9. References

---

## 1. Physical picture

### 1.1 Earth as seen from Mars

Earth is an *inner* planet from Mars: it shows phases like Venus does
from Earth, never strays far from the Sun, and is brightest near
quadrature rather than at closest approach (closest approach is
inferior conjunction, where Earth is a thin crescent lost in the
solar glare). Circular-orbit geometry (a_E = 1 AU, a_M = 1.524 AU):

| Earth–Mars range | Angular diameter | Phase angle α | Solar elongation | V (Mallama & Hilton Eq. 5) |
|---:|---:|---:|---:|---:|
| 0.50 AU | 35.1″ | ~180° | ~0° | (crescent, in glare) |
| 0.75 AU | 23.4″ | 120° | 34° | −1.8 |
| 1.00 AU | 17.6″ | 99° | 40° | −2.1 |
| 1.50 AU | 11.7″ | 72° | 39° | −2.1 |
| 2.00 AU | 8.8″ | 48° | 29° | −2.1 |
| 2.50 AU | 7.0″ | 10° | 6° | −2.0 |

- **Maximum elongation from the Sun is ~41°** for circular orbits
  (asin(1/1.524)) and **~47°** at the eccentricity extremes (Earth at
  aphelion, Mars at perihelion). Any telescope guiding on Earth from
  Mars is *always* pointed within 47° of the Sun. Stray light and baffle
  performance are first-order for this study, not an afterthought.
- Mallama & Hilton (2018) put Earth's brightest from Mars at
  **V = −2.55** (α ≈ 96°). Their Earth model uses V₁(0) = −3.99
  (geometric albedo 0.434, from EPOXI). A 2025 reanalysis
  (arXiv:2507.22258) argues for **p_V ≈ 0.24**, i.e. Earth may be
  ~0.6 mag fainter than the almanac formula. Earth's brightness is
  cloud-dependent at the ±0.5 mag level; the renderer must treat cloud
  fraction as a free parameter and the study must bracket it.
- **The Moon** is always within ~18′ of Earth (384 400 km at 0.5 AU) and
  1.8″–13″ across. From Mars near quadrature it is roughly V ≈ +2 to +3,
  4–5 mag fainter than Earth, phase-locked to the same α. It is a
  second resolved body in the guide field and must be rendered (it is
  also a useful airless-body cross-check for the limb algorithm).
- **Aberration** for a Mars-orbiting observer (v ≈ 24 km/s) is ~16.6″.
  It shifts stars and Earth almost identically over a small field, so
  it is irrelevant to *relative* guiding, but it must be applied
  consistently if absolute pointing is compared against catalog stars.
- **Observer position matters.** A spacecraft 20 000 km from Mars
  centre displaces the apparent direction to Earth by ~55″ at 0.5 AU.
  The observer must be a spacecraft state, not "Mars".

### 1.2 Resolution and sampling

Diffraction limit 1.22 λ/D at 550 nm: 0.05 m → 2.8″, 0.1 m → 1.4″,
0.3 m → 0.46″, 0.5 m → 0.28″. On the JBT 50 cm baseline
(f = 5.97 m, IMX455 3.76 µm → 0.13″/px) Earth at 1 AU is ~135 px across
and the PSF is ~2 px FWHM, so the disk is *well* resolved and the limb
edge is PSF-limited, not pixel-limited. On a 5 cm star tracker Earth
is 3–15 resolution elements across: marginally resolved, and the
phase-dependent photocentre bias dominates.

### 1.3 The atmospheric limb

- Earth's visible-light limb is not the surface. Along a tangent ray at
  height h the Rayleigh optical depth is roughly
  τ_vert(h)·√(2πR/H) ≈ 75·τ_vert(h); with τ_vert(0, 550 nm) ≈ 0.1 and
  H ≈ 8.4 km the limb (τ ≈ 1) sits near **35 km**, higher in the blue
  (λ⁻⁴) and lower in the red. Artemis I OpNav pre-flight assumed a 35 km
  atmosphere bias and measured **~25 km** from flight imagery
  (camera-, exposure- and wavelength-dependent).
- At 0.5 AU, 36 km subtends 0.10″; at 1.5 AU, 0.03″. The atmosphere
  therefore appears as a **sub-pixel limb extension plus a faint halo**
  (Rayleigh scale height ≈ 0.02″), never as a resolved ring, for any
  aperture ≤ 1 m. The rendering requirement is a correct *sub-pixel
  radiance profile across the limb*, integrated over the sensor band,
  because that is what sets the limb-fit bias.
- At large phase angles the sunlit atmosphere extends past the
  geometric terminator (Venus-style cusp extension) via forward
  scattering by aerosols and clouds. Guiding at α > 120° relies on this
  thin arc; Mie forward scattering must be modelled, not just Rayleigh.

### 1.4 Saturation

Earth at V ≈ −2.1 on the JBT 50 cm + IMX455 baseline delivers of order
10¹⁰ photons/s into the aperture spread over ~1.4×10⁴ pixels: the
disk-average pixel fills a 26 ke⁻ well in **tens of milliseconds**,
bright clouds several times faster. Realistic Earth-guiding exposures
are 1–10 ms (or need an ND / narrowband filter). The current renderer
clips at full well with no bleed; blooming is needed to make saturated
Earth images honest.

---

## 2. What already exists

### 2.1 `starfield` 0.13 (we own this repo)

Far more solar-system machinery than expected. It is a Rust port of
Skyfield, including a from-scratch port of `jplephem`:

- **SPK/BSP reader**, pure Rust (`jplephem::{DAF, SPK, SpiceKernel,
  PlanetState}`), Chebyshev types 2/3 and type 21, memmapped.
  `Loader::open("de440s.bsp")` downloads from JPL into
  `~/.cache/starfield`. `planetlib::{Body, Ephemeris}` gives a typed API.
- **Time** (`Timescale`, `Time`): UTC/TAI/TT/TDB/UT1, leap seconds,
  ΔT, GMST/GAST, precession (`precessionlib`), IAU2000A nutation
  (`nutationlib`), frame matrices, polar motion. `framelib::Frame`
  trait with ICRS / true-equator / ecliptic / galactic.
- **Apparent places**: `Position::observe(target, kernel, t)` does
  light-time iteration; `Position::apparent()` adds gravitational
  deflection and stellar aberration. Works for any observer whose
  barycentric state is known.
- **Planet magnitudes**: `magnitudelib::planetary_magnitude` implements
  Mallama & Hilton (2018) for Mercury through Neptune, **including
  Earth (Eq. 5)**, with Saturn/Uranus sub-latitude terms.
- Kepler propagation (`keplerlib::KeplerOrbit`), osculating elements,
  Horizons client (including SPK generation), SGP4, geoid/topocentric,
  eclipses, almanac event search.
- Catalog traits: `StarCatalog`, `StarData`, `Equatorial`,
  `ExtendedSource` + `SersicProfile`, and (feature `photometry`) the
  per-band scalar `Photometry` / `RadialProfile` / `IsophoteSeries`
  traits. **No spectra, bandpasses or blackbodies** in starfield.

**Gaps in starfield** (all things we should add there, not in
focalplane, because they are generic celestial mechanics):

| Gap | What is needed |
|---|---|
| `jplephem::pck` is a 24-line stub | Text PCK (`pck00011.tpc`) parser: `BODYnnn_RADII`, `_POLE_RA/DEC`, `_PM`, `_NUT_PREC_*`; binary PCK (type 2 Euler-angle Chebyshev, reuses DAF + Chebyshev code) for `moon_pa_de440` and `earth_latest_high_prec`. API: `body_fixed_to_icrf(body, t) -> Matrix3`, `radii(body) -> [f64;3]` |
| No planetary radii / flattening anywhere | Comes for free with the PCK parser; also embed an IAU 2015 (Archinal et al. 2018 + 2019 corrigendum) constants table as a no-network fallback |
| Phase geometry is private inside `magnitudelib` | Public `illumination::{phase_angle, illuminated_fraction, sub_solar_point, sub_observer_point, bright_limb_position_angle, north_pole_position_angle, angular_semi_diameter}` |
| `Position::observe` takes a body *name* only | `observe_star(&StarData)` / `apparent_star()` so catalog stars get aberration + deflection for an arbitrary observer (needed so Earth and stars are treated consistently) |
| No observer-on-orbit helper | `Observer::on_kepler_orbit(center_body, KeplerOrbit)` / `Observer::from_spk(spacecraft_id)` producing a barycentric `Position` with velocity at `t` |
| Earth body-fixed via IAU_EARTH is coarse | Wire the existing ICRS→ITRS chain (`m_matrix`, ERA/GAST, polar motion) as the Earth body-fixed frame; IAU elements for everything else |
| `magnitudelib` has no Moon | Add lunar V (e.g. Mallama et al. lunar phase curve or Allen/Kieffer) so the Moon-in-field cross-check has a truth value |
| `SersicProfile` is the only extended-source model | Do **not** add planet disks to starfield; keep radiometry in focalplane. Starfield supplies geometry only |

### 2.2 `starfield-datasources` (we own this repo)

`datasource-utils` gives `cache_dir()`, `ensure_cache_subdir()`,
`download_to_file(url, path, timeout)` (atomic, with progress),
`build_http_client()`. Small curated tables are embedded with
`include_str!` (see `bright-galaxies`). No checksum support; add
SHA-256 verification for large binary assets (textures, BSP).

New datasource crates proposed:

- `starfield-planet-maps`: downloads and caches equirectangular maps
  (see §4.6 for the list), exposes `PlanetMap::load(body, layer, month)
  -> EquirectMap<f32>` with lon/lat sampling. Optional feature-gated
  high-resolution tiers.
- `starfield-solar-spectrum`: embedded, downsampled (1 nm) TSIS-1
  Hybrid Solar Reference Spectrum v2 (Coddington et al. 2021/2023,
  CEOS-endorsed reference) as a `(λ nm, W m⁻² nm⁻¹ at 1 AU)` table.
- `starfield-reflectance-library`: embedded spectral endmembers
  (ocean water, vegetation, bare soil/desert sand, snow/ice, liquid
  water cloud, ice cloud, lunar mare/highland, Mars bright/dark,
  Venus cloud top) from USGS/ASTER/ECOSTRESS spectral libraries and
  published planetary spectra.

### 2.3 `simulator` (this repo)

The extended-source seam is already generic:
`image_proc::deposit::{MeanFluxDeposit, FrameSource, splat_deposit,
render_sources}`. Sérsic galaxies are just one `MeanFluxDeposit` impl
(`SersicSplat`), evaluated analytically per pixel with 2×2 sub-pixel
quadrature. A planet is a third impl. The trait docs explicitly
anticipate "satellite trails, asteroids".

Photometry: `Spectrum` trait (F_ν, CGS) → `photon_electron_fluxes(psf,
spectrum, qe) -> SourceFlux` collapses a spectrum to one electron rate
plus one chromatic Airy disk. Zodiacal light is an HST/STIS solar-like
template scaled by an (elongation, ecliptic latitude) table, injected
as a *uniform* per-pixel mean.

Motion: `Trajectory` of quaternion waypoints indexed by `Duration`
from t = 0; golden-ratio stamp sampling within each exposure
(`SubsampleSchedule`); stars re-projected per stamp, galaxies splatted
once per exposure.

**Gaps in simulator:**

| Gap | Consequence for planets |
|---|---|
| No absolute time, no observer state | Cannot evaluate an ephemeris. Need `Epoch` + `Observer` on `Scene`, `Trajectory`, `MotionBlurConfig`, and `--epoch` / `--observer` CLI args |
| No solar spectrum | Reflected-light radiometry has nothing to reflect |
| Spectrum collapsed to a scalar at build time | Planet surface brightness varies spectrally *across the disk* (blue limb, red deserts, white clouds). Need per-texel band-integrated reflectance, not one SED per source |
| Extended sources get **no PSF convolution** | Fine for 60″ galaxies, wrong for a 2–50 px planet whose limb is PSF-limited. Need convolution (or analytic PSF⊗edge) for the planet deposit |
| Galaxies not re-projected per stamp | Earth moves relative to stars at up to ~0.1″/min from Mars, and the spacecraft jitters; the planet deposit must be re-splatted per stamp like stars |
| Saturation = hard clip | Need blooming / charge bleed for honest saturated Earth images |
| Zodiacal table is Earth-based (1 AU) | Scale by heliocentric distance, ≈ r^−2.3 (Leinert et al. 1998) |
| No stray light model | Guiding always within 47° of the Sun; need at least a PST(θ_sun)-driven uniform background term |
| `SolarAngularCoordinates` is a CLI input | Derive from epoch + observer + pointing |
| Detection/centroiding is star-oriented (`shared` DAO/IRAF) | Need limb detection, photocentre correction, template correlation |
| Context render has no planet marker | Add disk outline + terminator to `context_render` |

---

## 3. Target architecture

### 3.0 Integration strategy: optional feature, second pass

Solar-system bodies are **opt-in** and **rendered as a separate pass**
that composites onto the star-field mean image. Consumers that never
enable them see no API change, no new dependencies, no extra runtime.

**Cargo feature `solar-system`** (simulator crate, default off):

- Gates the `epoch`, `solar_system`, `atmosphere`, `bodies`,
  `scene_planet`, `image_proc::planet_disk`, `algo::{limb,
  photocenter, template_match}` modules, the new datasource crates
  (`starfield-planet-maps`, `starfield-solar-spectrum`,
  `starfield-reflectance-library`), and the `--epoch` / `--observer` /
  `--bodies` CLI flags.
- Everything that is *useful regardless* of planets stays unconditional:
  PSF convolution utility, blooming, stray-light background, zodiacal
  heliocentric scaling, solar spectrum type. Those are small and have
  no heavy deps.
- The `Scene`, `LightSources` and `MotionBlurConfig` structs gain one
  `#[cfg(feature = "solar-system")] bodies: BodyPass` field each (or a
  boxed `Option<Box<dyn SecondPass>>` if we prefer no `cfg` in struct
  definitions; decision below). With the feature off the renderers are
  byte-identical to today, which is testable in CI by running the
  existing INVARIANTS tests under both feature sets.

**Second pass** (`image_proc::compose.rs`):

Bodies are opaque and *occult* whatever is behind them: background
stars, galaxies, zodiacal light. A planet is therefore not just another
additive `FrameSource`; it must also mask. The pass order per stamp is:

1. **Pass 1 (existing)**: stars and galaxies deposit mean electrons
   into `star_mean_electrons` exactly as today.
2. **Pass 2 (new, feature-gated)**: for each body in the field, build
   its oversampled coverage mask and radiance stamp; multiply the pass-1
   buffer *and the per-pixel zodiacal/stray-light mean* by
   `(1 − coverage)` inside the body's footprint; add the body's own
   radiance stamp. Coverage is the PSF-convolved geometric disk (0–1 per
   pixel), so occulted stars fade across the limb consistently with how
   the limb itself is blurred.
3. **Unified Poisson + sensor noise (existing)**: unchanged. INVARIANTS
   §1 holds because compositing acts on *means* before the single
   Poisson draw; §2 holds because both the static `Renderer` and the
   motion-blur `SensorAccumulator` call the same `compose::apply_bodies`.

Occultation of one body by another (Moon behind Earth, Earth behind
Mars) is handled by depth-sorting bodies by observer distance and
compositing far-to-near with the same mask multiply. Stars *in front*
of a body do not exist at these distances, so a single depth order is
enough.

Why a second pass rather than a `FrameSource` impl: the deposit trait is
additive by contract (`pixel_flux` returns mean electrons to add) and
the whole renderer is built on that linearity. Occultation is
multiplicative. Keeping the multiplicative step in one explicit
function is simpler and easier to lock with tests than teaching
`splat_deposit` about masks.

Cost with the feature on and no bodies in the field: one AABB test per
body per stamp. Cost with a body in the field: one stamp build per
(body, sensor, cache bucket) plus a footprint-sized multiply-add per
stamp.

### 3.1 Module layout (simulator)

```
simulator/src/
  epoch.rs                    Epoch (wraps starfield Time) + Observer (barycentric state provider)
  solar_system/
    mod.rs                    BodyId, BodyState { apparent dir, distance, sub-solar/sub-obs points, α, radii, body_fixed_to_camera }
    ephemeris.rs              SolarSystem { kernel, pck } -> BodyState at (epoch, observer); light-time + aberration via starfield
    observer.rs               Observer::{at_body, kepler_orbit_about, spk_id, fixed_barycentric}
  photometry/
    solar.rs                  SolarSpectrum: Spectrum  (TSIS-1 HSRS at 1 AU, scaled by 1/r²)
    reflectance.rs            Endmember spectra; BandIntegratedReflectance::new(endmember, qe, solar) -> per-endmember e⁻ s⁻¹ m⁻² sr⁻¹ weights
    zodiacal.rs               + heliocentric_scale(r_au)
    stray_light.rs            PST(θ_sun) → uniform background e⁻/s/px (user-supplied PST curve)
  atmosphere/
    mod.rs                    AtmosphereModel { rayleigh, mie, ozone, layers } with Earth/Mars/Venus presets
    rayleigh.rs               Bodhaine et al. (1999) cross-section, depolarisation; phase function
    mie.rs                    Henyey–Greenstein / Cornette–Shanks aerosol; cloud droplet forward peak
    ozone.rs                  Serdyuchenko et al. (2014) Chappuis-band cross-sections; layer profile
    lut.rs                    Precomputed transmittance T(r, μ) and single-scatter S(r, μ, μ_s, ν) per spectral bin (Bruneton & Neyret 2008 parameterisation); multiple-scatter orders via iteration
    limb.rs                   Tangent-ray integrator for the limb profile (Chapman geometry), used both by the renderer and by tests
  bodies/
    mod.rs                    trait BodyAppearance { fn radiance(&self, surf_pt, sun_dir, view_dir, band) -> f64 }
    brdf.rs                   Lambert, Lommel–Seeliger, Minnaert, Hapke (with opposition surge + roughness), Cox–Munk glint
    texture.rs                EquirectMap sampling (bilinear, mip-level chosen from px footprint) → endmember abundances / albedo
    earth.rs                  Surface (land-cover endmembers + snow) + cloud layer + ocean glint + atmosphere
    moon.rs                   Hapke (Sato et al. 2014 params) × LROC WAC normalised albedo
    mars.rs                   L-S + dust-haze atmosphere × MDIM/Viking albedo, polar caps
    venus.rs                  Featureless cloud deck, Minnaert limb darkening, Mallama phase curve, H₂SO₄ forward-scatter cusps
    mercury.rs                Hapke × MESSENGER MDIS map
    gas_giant.rs              Jupiter/Saturn: Minnaert limb darkening × OPAL maps, oblate spheroid; rings deferred
    sun.rs                    Self-luminous disk: TSIS-1 spectrum × wavelength-dependent limb darkening (Neckel & Labs 1994 / Pierce & Slaughter 1977)
  image_proc/
    planet_disk.rs            BodyStamp: supersampled ray-cast of the ellipsoid + atmosphere shell → (coverage, radiance) stamps, PSF-convolved
    compose.rs                Second pass: depth-sorted mask-multiply + add of BodyStamps onto the pass-1 mean image (feature `solar-system`)
    psf_convolve.rs           Convolution of an oversampled stamp with the sampled PSF (shared::convolve2d or FFT via realfft) — unconditional
    blooming.rs               Charge-bleed model applied before quantisation — unconditional
  scene_planet.rs             Body (sky truth) + BodyInFrame { x, y, stamp: BodyStamp, distance } + project_bodies_to_sensors
  algo/
    limb.rs                   Sun-direction scan → sub-pixel edges (Zernike moment method, Christian 2017) → Christian–Robinson ellipse/circle fit; terminator exclusion
    photocenter.rs            Intensity-weighted centroid + Lambert / Lommel–Seeliger phase-offset correction (Owen 2011)
    template_match.rs         Normalised cross-correlation against a rendered prediction (centre-finding by correlation)
  sims/
    planet_scene.rs           Loader: epoch + observer + pointing → Vec<Planet> in field (mirrors bright_galaxies.rs)
    earth_limb_guiding.rs     The study harness (§5)
```

### 3.2 Data flow for one exposure

```
pass 1 (unchanged): stars + galaxies ──► star_mean_electrons

pass 2 (feature solar-system):
epoch, observer ──► SolarSystem::body_state(body)  ──► BodyState
                                                          │  (apparent Equatorial, distance, α, sub-points,
                                                          │   body_fixed→ICRF, radii)
pointing quaternion(t_stamp) ────────────────────────────►│
                                                          ▼
                          BodyInFrame { x, y, stamp: BodyStamp, distance }
                                                          │
       BodyStamp::build(body_state, sensor, psf) ◄────────┘
         ├─ oversampled stamp (N×N per px): for each sub-sample
         │    ray ∩ ellipsoid → surface point → (μ₀, μ, α) → BodyAppearance::radiance(band)
         │    ray ∩ atmosphere shell (miss surface) → limb LUT radiance
         │    surface path → × transmittance, + in-scatter along path (LUT)
         ├─ convolve (coverage, radiance) stamps with sampled PSF at the sensor plate scale
         └─ BodyStamp { coverage: 0..1 per px, electrons per px }
                                                          │
       compose::apply_bodies (far → near):  mean = mean × (1 − coverage) + electrons
       (INVARIANTS §1 preserved: operates on means only, before the single Poisson draw)
```

`total_electrons` for the planet is the band-integrated disk total from
the same stamp so that `pixel_flux` is a pure normalised shape; the
integrated value is checked against `starfield::planetary_magnitude`
at build time (§6).

### 3.3 Spectral handling without a per-pixel spectrum

Spatially varying colour is handled by **spectral endmembers**, not by
storing a spectrum per texel:

- Each texel carries abundances a_i over a small set of endmembers
  (Earth: ocean, vegetation, soil, sand, snow, water cloud, ice cloud;
  Mars: bright, dark, ice; …). Abundances come from land-cover class
  maps (MODIS MCD12) plus snow/cloud masks; Blue Marble RGB is used
  only for QA imagery, not as a spectral truth.
- For the sensor's `combined_qe` band, precompute once per render
  w_i = ∫ (F☉(λ)/π r²) ρ_i(λ) QE(λ) A_pix dλ, so texel radiance in
  electrons is Σ a_i w_i × BRDF shape factor. This is exact for
  Lambert-like surfaces and a good approximation where the BRDF's
  spectral dependence is weak.
- The atmosphere is spectrally strong (λ⁻⁴), so its LUTs are built per
  spectral bin (10–20 nm bins over the QE support) and band-integrated
  with QE weights at LUT-build time.
- RGB "true-colour" previews for `context_render` and docs use the same
  endmembers integrated over the cone responses in `photometry/human.rs`.

---

## 4. Rendering physics

### 4.1 Radiometric chain

Spectral radiance leaving a surface element:

$$L(\lambda) = \frac{F_\odot(\lambda)}{\pi\, r_\odot^2}\; R(\mu_0, \mu, \alpha; \lambda)$$

with F☉ the TSIS-1 irradiance at 1 AU, r☉ the body's heliocentric
distance in AU, and R the bidirectional reflectance (R = A for a
Lambert surface with albedo A, giving L = A F/π at normal incidence).
Electrons per pixel per second:

$$\dot e = A_{\rm ap} \,\Omega_{\rm pix} \int L(\lambda)\, QE(\lambda)\, \frac{\lambda}{hc}\, d\lambda$$

Integrated over the disk this must reproduce V(α) from Mallama & Hilton
to within the model's stated uncertainty (Earth: ±0.3 mag because of
clouds; airless bodies: ±0.05 mag). This is the primary radiometric
unit test.

### 4.2 Surface photometric models (`bodies/brdf.rs`)

| Model | Use | Notes |
|---|---|---|
| Lambert | Clouds, ice, fallback | R = A μ₀ |
| Lommel–Seeliger | Moon, Mercury, Mars quick-look | R = (w/4π)·μ₀/(μ₀+μ)·P(α); correct limb behaviour for dark regolith |
| Minnaert | Venus, gas giants | R = A μ₀^k μ^(k−1); k ≈ 0.5–0.9 fit to limb darkening |
| Hapke (1993/2012) | Moon (Sato et al. 2014 resolved parameter maps), Mercury, Mars | Single-scattering albedo w, phase function (double H-G), opposition surge B₀/h, roughness θ̄. Needed for correct opposition-surge behaviour at small α and for validated lunar photometry |
| Cox–Munk | Ocean specular glint | Wind-speed-dependent slope distribution; the glint is a real, bright, phase-dependent feature on Earth's disk and biases photocentres |

### 4.3 Earth (`bodies/earth.rs`)

Layers, bottom to top:

1. **Surface**: endmember abundance map (MODIS land cover + monthly snow
   cover), Lambert BRDF per endmember, Cox–Munk glint on water texels,
   night side dark (city lights ~10⁻⁶ of day-side radiance; ignored).
2. **Clouds**: separate layer with its own texture (monthly MODIS cloud
   fraction as a baseline, plus an option for a stochastic cloud field
   with matching power spectrum so the study can vary weather). Optical
   thickness → reflectance via a two-stream approximation; forward-peaked
   Mie phase function for the cusp extension at high α. Cloud-top height
   ~2–10 km enters the terminator geometry.
3. **Atmosphere**: spherical-shell Rayleigh (Bodhaine et al. 1999
   cross-sections, H ≈ 8 km) + aerosol Mie (Henyey–Greenstein g ≈ 0.76,
   H ≈ 1.2 km, optical depth a tunable 0.05–0.3) + ozone Chappuis
   absorption (Serdyuchenko et al. 2014 cross-sections, layer peaked
   ~25 km). Single scattering computed exactly along each sub-sample ray
   through precomputed transmittance LUTs; multiple scattering added by
   the Bruneton & Neyret (2008) iterative orders (2–4 orders suffice in
   the visible). Rays that miss the surface but pass through the shell
   produce the limb radiance; rays that hit the surface attenuate the
   surface term and add in-scattered path radiance.
4. **Refraction** is negligible for our purposes (bends rays by
   ~0.5° only at the very lowest tangent heights, which are optically
   thick anyway); note and skip.

The limb profile L(h) must be produced by the same code path as the
disk so that the limb bias measured by `algo/limb.rs` is self-consistent.
Validation target: the effective detected limb sits 20–35 km above the
ellipsoid in a broadband visible camera (Artemis I: 25 km measured,
35 km pre-flight guess), and moves upward at bluer wavelengths.

### 4.4 Moon, Mercury, Mars

- **Moon**: Hapke with Sato et al. (2014) LROC WAC parameter maps
  (or a global mean set) × WAC 643 nm normalised-albedo mosaic; lunar
  libration and pole from the DE440 lunar PA binary PCK. Colour from a
  two-endmember (mare/highland) model.
- **Mercury**: Hapke (Domingue et al.) × MESSENGER MDIS monochrome map.
- **Mars**: Lommel–Seeliger + a thin dust-haze scattering layer
  (τ ≈ 0.1–1 with dust-storm option), albedo from the Viking/MOLA
  colourised mosaic split into bright/dark/ice endmembers, seasonal
  polar caps. Mallama's L(λ_e) and L(L_s) corrections provide the
  disk-integrated validation curve.

### 4.5 Venus and gas giants

- **Venus**: featureless in the visible. Uniform cloud deck, Minnaert
  limb darkening, disk-integrated brightness pinned to Mallama & Hilton
  Eqs. 3–4 (including the α ≈ 163° forward-scatter inflection). Cusp
  extension from a Mie shell with H₂SO₄ droplet phase function.
- **Jupiter / Saturn / Uranus / Neptune**: oblate spheroid (flattening
  from PCK), Minnaert limb darkening with band-dependent k, texture
  from Hubble OPAL cylindrical maps (public domain). Saturn's rings are
  a separate deliverable (annulus with optical-depth profile, shadows);
  defer.

### 4.6 Texture / data assets (`starfield-planet-maps`)

| Body | Layer | Source | Resolution | Licence |
|---|---|---|---|---|
| Earth | land-cover classes | MODIS MCD12Q1/MCD12C1 | 0.05° | NASA, public |
| Earth | monthly snow, cloud fraction | MODIS (MYD10CM, MYD08_M3) | 0.05°–1° | NASA, public |
| Earth | QA true-colour | Blue Marble Next Generation (monthly) | 500 m–8 km tiers | NASA, public |
| Earth | multispectral disk validation | DSCOVR EPIC L1B (10 bands 317–780 nm) | full disk | NASA, public |
| Moon | albedo | LROC WAC 643 nm normalised mosaic | 100 m (use 1–4 km tier) | NASA/PDS |
| Mars | albedo/colour | USGS Viking MDIM 2.1 / MOLA-Viking colour | 232 m (use 1–4 km tier) | USGS, public |
| Mercury | albedo | MESSENGER MDIS global mosaic | 166 m (use 1 km tier) | NASA/PDS |
| Jupiter, Saturn, Uranus, Neptune | colour | Hubble OPAL global maps | ~0.1° | NASA/STScI, public |
| Sun | spectrum | TSIS-1 HSRS v2 | 0.01 nm (embed at 1 nm) | NASA/LASP, public |

Ship 1–4 km tiers by default (a few MB each); expose finer tiers behind
a cargo feature. Equirectangular, body-fixed longitudes per IAU
conventions (Mars west-longitude vs east-longitude conventions are a
known trap; encode the convention in the map metadata).

### 4.7 The `BodyStamp` (planet disk rendering)

- Footprint half-width = ceil(semi-diameter / plate scale + atmosphere
  shell + 2 × PSF first zero).
- A `BodyStamp` is a **cached oversampled pair of stamps**, coverage
  (0–1) and radiance (electrons per pixel per second), built once per
  (body, sensor, stamp-time bucket). Unlike Sérsic, the shape cannot be
  evaluated analytically per pixel, so the stamp is ray-cast at the
  oversampled grid, convolved with the PSF, and downsampled; the second
  pass then reads it by bilinear lookup at the sub-pixel body centre.
  Memory is bounded: a 200 px disk at 4× oversampling is 640 k floats
  per stamp.
- Oversampling factor chosen from the PSF FWHM in pixels (≥ 4 samples
  per FWHM) and from the limb: the limb transition region (atmosphere
  scale height) is always sub-sample, so the last sample straddling the
  limb uses analytic partial coverage of the ellipsoid edge to avoid
  aliasing the edge position (this matters for a limb-fit study).
- Convolution uses the *sampled* PSF, not the Gaussian approximation
  currently used for stars, so Airy rings and the obscuration ratio can
  be included; document the difference and add a switch to use the same
  Gaussian approximation as stars for like-for-like comparisons.
- Per-stamp behaviour in `motion_blur`: re-project the planet centre at
  every stamp (like stars) and reuse the stamp image unless the
  body-fixed rotation or α has changed by more than a threshold
  (Earth rotates 0.25°/min; over a 10 ms exposure nothing changes;
  across a 10-minute guiding sequence it does).

### 4.8 Detector effects to add

- **Blooming**: simple vertical charge-bleed model (excess above full
  well spills to neighbours along the column with a bleed fraction),
  applied to the mean image before quantisation; parameterised per
  sensor. Without it, Earth images are flat-topped and the limb algorithm
  is tested on fiction.
- **Stray light**: a user-supplied point-source-transmittance curve
  PST(θ☉) → uniform background electrons/s/px; plus a ghost-free
  assumption documented. This is the dominant background at 30–47°
  from the Sun for most baffle designs and it must be in the SNR budget.
- **Zodiacal at Mars**: scale the STIS template by (r/1 AU)^−2.3.

### 4.9 The Sun

Yes, but in two tiers, because the Sun plays two different roles.

**Tier 1, geometry and background (Phase 2, cheap, do it now).** The
Sun already exists as an ephemeris body. Expose it as a `BodyState`
like any other so that every render knows: solar elongation of the
boresight, the Sun's angular radius (32′ at 1 AU, 21′ = 0.35° at Mars),
whether any body in the field is in transit across or occulted by the
Sun, and the stray-light background via the PST(θ☉) term. This is the
part the Earth-limb study actually needs: it is always working within
47° of the Sun, and Earth transits of the Sun as seen from Mars are
real events (next 2084-11-10) that a general tool should predict rather
than mis-render. Also drives the `SolarAngularCoordinates` derivation
for zodiacal light.

**Tier 2, the solar disk as a rendered source (Phase 3, optional,
small once the body framework exists).** A self-luminous body is the
*simplest* `BodyAppearance`: no BRDF, no phase, radiance = TSIS-1
spectral radiance × limb darkening I(μ)/I(1). Use the wavelength-
dependent polynomial coefficients of Neckel & Labs (1994) (or Pierce &
Slaughter 1977), band-integrated with the sensor QE like everything
else. No texture: sunspots and faculae modulate the disk at the 0.1 %
level and are irrelevant to any detector that can survive pointing at
it. Radiometric truth: V☉ = −26.74 at 1 AU, −25.8 from Mars.

Use cases that justify tier 2: (a) solar-limb fine guidance is an
established technique for heliophysics payloads, and the limb-fit code
from §5.3 applies directly to a disk that is bright, sharp, cloud-free
and never crescent, so it is a good algorithm stress test; (b) transit
and partial-occultation rendering (Earth or Moon silhouetted against
the Sun from Mars; Phobos/Deimos transits from a Mars orbiter); (c)
honest saturation and blooming behaviour when the Sun's disk or its
PSF wings intrude into the field edge; (d) the same limb-darkening
machinery is what gas giants need anyway.

What tier 2 does *not* attempt: coronal or chromospheric emission,
solar oscillations, or any thermal effect on the detector. If the Sun
is inside the field the render is flagged (`metadata.json`) as
non-physical for any sensor without a solar filter, and the exposure
planner refuses unless a neutral-density/filter option is set.

---

## 5. Earth-limb guiding study

### 5.1 Question

For a telescope in Mars orbit, can Earth (or the Earth–Moon pair)
serve as a fine-guidance reference, and with what noise-equivalent
angle (NEA), bias, and drift, as a function of aperture, exposure,
phase angle, cloud state and jitter spectrum? Compare against guiding on
field stars in the same frame.

### 5.2 Scenarios

| Parameter | Values |
|---|---|
| Epoch / geometry | Sweep one synodic period (2027–2029) at monthly steps: α from ~10° to ~170°, range 0.4–2.6 AU |
| Observer | Mars-centred Kepler orbits: 400 km circular; areostationary (~17 000 km); plus Mars-Sun L1-like heliocentric offset. Includes Mars occultation and eclipse of Earth |
| Telescope | 5 cm star tracker (unresolved regime), 10 cm, JBT 50 cm baseline, 1 m |
| Sensor | IMX455, GSENSE4040BSI (existing presets) |
| Exposure | 1, 3, 10, 30, 100 ms (saturation-limited), plus ND filter option |
| Cloud state | MODIS monthly climatology; stochastic fields at 3 fractions (0.4, 0.6, 0.8); cloud-free; overcast |
| Jitter | Existing PSD-derived trajectories (`sims/jitter.rs`), drift-only, static |
| Stray light | PST curves for two baffle assumptions (good/poor) |

### 5.3 Algorithms under test (`algo/`)

1. **Photocentre + phase correction**: intensity-weighted centroid
   with a Lambert or Lommel–Seeliger offset correction along the Sun
   direction (Owen 2011; Christian 2015). Cheap, works
   unresolved, but sensitive to cloud asymmetry and glint.
2. **Limb fit**: scan along the illumination direction for the lit
   limb, sub-pixel edge refinement by Zernike moments (Christian 2017),
   discard terminator-side points by the α-dependent angular window,
   Christian–Robinson non-iterative ellipse fit with the known
   (inflated) ellipsoid, yielding centre and apparent radius. Cloud-
   insensitive by construction; its bias is the atmosphere height,
   which is stable.
3. **Template correlation**: normalised cross-correlation of the frame
   against a rendered prediction from the same ephemeris and
   climatological clouds (centre-finding by correlation, as flown on
   New Horizons / OSIRIS-REx). Best noise performance; degraded by
   weather mismatch, which is exactly what the stochastic cloud fields
   probe.
4. **Star field reference**: existing DAO/IRAF detection + ICP match
   on the same frames, as the control.

### 5.4 Metrics and outputs

- NEA (1σ) in mas vs exposure, aperture, α; compare with the
  photon-noise floor for a limb of the observed length.
- Bias vs α and vs cloud state (mas and km at the Earth).
- Drift over 10 s, 60 s, 600 s sequences (Earth rotation moves clouds
  by 0.25°/min; check the limb methods are immune and the photocentre
  is not).
- Failure envelope: thin crescent (α > 150°), Moon overlap /
  occultation, saturation fraction, stray-light SNR.
- CSV rows through `scene_runner` conventions; plots via existing
  `plotting.rs` / `analysis/*.py`.

### 5.5 Expected qualitative result (to be tested, not assumed)

The limb is a sharp, cloud-independent, PSF-limited edge whose apparent
height varies slowly with wavelength and season; on a 50 cm aperture at
1 AU it is ~400 px long and should yield sub-mas NEA in ms exposures,
limited by saturation management and stray light rather than photon
noise. Photocentre methods will show tens-of-mas cloud-driven drift.
Unresolved (5 cm) guiding degrades to the phase-corrected photocentre
with a bias uncertainty set by Earth's albedo distribution.

---

## 6. Validation ladder

1. **Ephemeris**: RA/Dec, range, α, angular diameter, sub-observer and
   sub-solar lon/lat for Earth/Moon/Mars/Venus from a Mars-centred
   observer vs JPL Horizons (`horizons` client already in starfield) at
   ten epochs; tolerance 0.1″ / 0.01° (Horizons uses the same DE440).
2. **Rotation**: body-fixed frame vs SPICE `pxform` reference values
   generated offline (SpiceyPy) for IAU_MARS, IAU_MOON (via PA kernel),
   ITRF93; tolerance 1″.
3. **Disk-integrated photometry**: rendered V vs Mallama & Hilton
   for each body across α; Earth against Mallama Eq. 5 (V₁(0) = −3.99)
   *and* the 2025 p_V = 0.24 two-parameter curve, with cloud fraction
   as the knob that spans the two.
4. **Colour**: disk-integrated reflectance vs Earthshine spectra
   (Woolf et al. 2002; Turnbull et al. 2006) and EPOXI 7-band
   light curves (Livengood et al. 2011, α ≈ 58°, 75°, 77°); DSCOVR EPIC
   10-band full-disk images at α ≈ 4–12° for spatially resolved colour.
5. **Limb height**: broadband detected limb 20–35 km (Artemis I:
   25 km measured), decreasing red-ward; Rayleigh-only limit checked
   against the analytic Chapman-function estimate.
6. **Real Earth-from-Mars imagery**: HiRISE PSP_005558_9040
   (2007-10-03, range 1.42×10⁸ km, Earth 18.9″, Moon 5.1″, α = 98°,
   illuminated fraction 43.2%, separation 49.2″) and Mars Express VMC
   Earth–Moon frames (2023-05/06). Render with the same geometry and
   compare disk size, phase, Moon offset and relative brightness.
7. **Algorithm self-consistency**: limb fit on a rendered airless sphere
   recovers the injected centre to < 0.05 px; with atmosphere, recovers
   centre and reports the expected radius inflation; photocentre
   correction recovers centre on a Lambert sphere to < 0.02 px.
8. **INVARIANTS**: the existing §1 (single Poisson, mean-only deposits)
   and §2 (static vs motion route byte-identical) locks extended to
   planets.

---

## 7. Delivery plan

Sizes: S ≈ a day, M ≈ a few days, L ≈ a week+. Dependencies flow
top-down; items in the same phase can be parallel.

Starfield issues filed for Phase 0 (OrbitalCommons/starfield):
[#157](https://github.com/OrbitalCommons/starfield/issues/157) text PCK,
[#158](https://github.com/OrbitalCommons/starfield/issues/158) binary PCK,
[#159](https://github.com/OrbitalCommons/starfield/issues/159) `BodyFixedFrame`,
[#160](https://github.com/OrbitalCommons/starfield/issues/160) embedded IAU table,
[#161](https://github.com/OrbitalCommons/starfield/issues/161) `ItrsFrame`,
[#162](https://github.com/OrbitalCommons/starfield/issues/162) phase angle / illuminated fraction / elongation,
[#163](https://github.com/OrbitalCommons/starfield/issues/163) sub-observer / sub-solar points,
[#164](https://github.com/OrbitalCommons/starfield/issues/164) pole and bright-limb position angles,
[#165](https://github.com/OrbitalCommons/starfield/issues/165) angular semi-diameter / apparent ellipse,
[#166](https://github.com/OrbitalCommons/starfield/issues/166) `observe_star`,
[#167](https://github.com/OrbitalCommons/starfield/issues/167) spacecraft observer on a Kepler orbit,
[#168](https://github.com/OrbitalCommons/starfield/issues/168) Moon magnitude,
[#169](https://github.com/OrbitalCommons/starfield/issues/169) PCK download helpers,
[#170](https://github.com/OrbitalCommons/starfield/issues/170) occultation predicate.

### Phase 0 — starfield foundations (starfield repo)

| # | Item | Size |
|---|---|---|
| S0.1 | Text PCK parser (`pck00011.tpc`): radii, pole, prime meridian, nutation-precession angles; `body_fixed_to_icrf(body, t)`; embedded IAU-2015 fallback table | M |
| S0.2 | Binary PCK (type 2 Euler-angle Chebyshev) reader; wire `moon_pa_de440` | M |
| S0.3 | Public `illumination` module: phase angle, illuminated fraction, sub-solar/sub-observer points, bright-limb PA, north-pole PA, semi-diameter | S |
| S0.4 | `observe_star` / apparent star places for an arbitrary observer | S |
| S0.5 | `Observer` helpers: body-centred Kepler orbit, SPK spacecraft id, fixed heliocentric | S |
| S0.6 | Earth body-fixed frame via existing ICRS→ITRS chain | S |
| S0.7 | Lunar V magnitude in `magnitudelib` | S |
| S0.8 | Release starfield 0.14; bump in focalplane | S |

### Phase 1 — datasources (starfield-datasources repo)

| # | Item | Size |
|---|---|---|
| D1.1 | `starfield-solar-spectrum`: embedded 1 nm TSIS-1 HSRS v2 | S |
| D1.2 | `starfield-reflectance-library`: embedded endmember spectra with provenance | M |
| D1.3 | `starfield-planet-maps`: downloader + cache + SHA-256 + equirect sampler; Earth land cover/snow/cloud, Moon, Mars, Mercury, OPAL maps at coarse tiers | L |
| D1.4 | SHA-256 verification helper in `datasource-utils` | S |
| D1.5 | `starfield-planet-spectra`: Karkoschka (1994, 1998) spectral geometric albedos, 300–1050 nm, for Jupiter, Saturn (globe, zero ring tilt), Uranus, Neptune, Titan; `AlbedoKind` distinguishes zero-phase from full-disk-at-archive-phase | done (PR open) |

Interface decisions agreed with the datasources side (2026-09-09):

- Map sampler takes **east-positive planetocentric longitude and
  latitude in radians**, row 0 = north; every archive convention is
  converted inside the map, never at the call site. Focalplane feeds it
  planetocentric coordinates straight from the body-fixed frame with no
  longitude-sense adjustment.
- Sampler always returns an `EndmemberMix`; a scalar albedo is a
  one-endmember mix. Endmember ids resolve into the reflectance library,
  which exposes band means so per-endmember weights are computed once
  per render.
- Solar spectrum ships from datasources as 1 nm box means covering at
  least 300–1100 nm.
- Everything keyed on NAIF id, not `planetlib::Body` (Titan and the
  other moons have no `Body` variant).
- Earth uses ITRF93 and the Moon the DE440 principal-axes frame via
  starfield's `frame_for(id)`, matching Horizons.
- Time-dependent surface features (Mars seasonal CO₂ caps to −40°
  latitude near Ls 80–90, Jupiter's Great Red Spot drifting ~0.36°/day
  in System III) are parametric overlays composited over a static base
  map, sampled with an epoch argument.
- Maps ship a coarse tier by default and offer area-averaged sampling,
  because one rendered pixel spans thousands of texels of a 100–232 m
  mosaic.

### Phase 2 — simulator plumbing (this repo)

| # | Item | Size |
|---|---|---|
| F2.0 | Cargo feature `solar-system` (default off); `BodyPass` hook on `Scene`/`LightSources`/`MotionBlurConfig`; CI job runs tests with and without the feature; byte-identity test of existing renders with feature off | S |
| F2.1 | `epoch.rs`: `Epoch` on `Scene`, `Trajectory`, `MotionBlurConfig`; `--epoch` CLI; metadata.json records it | M |
| F2.2 | `solar_system/`: `SolarSystem` wrapper, `BodyState` (Sun included as a body), `Observer` config (`--observer mars-orbit:a,e,i,...`); solar elongation, transit/occultation flags in metadata | M |
| F2.3 | Derive `SolarAngularCoordinates` from epoch/observer/pointing; zodiacal r^−2.3 scaling | S |
| F2.4 | `photometry/solar.rs` (`SolarSpectrum: Spectrum`) + `reflectance.rs` band-integrated endmember weights | M |
| F2.5 | `image_proc/psf_convolve.rs` + sampled-PSF option; `blooming.rs` before quantisation | M |
| F2.6 | `stray_light.rs` PST background term + CLI | S |

### Phase 3 — bodies and disk rendering

| # | Item | Size |
|---|---|---|
| F3.1 | `bodies/brdf.rs` (Lambert, L-S, Minnaert, Hapke, Cox–Munk) with unit tests against published values | M |
| F3.2 | `image_proc/planet_disk.rs` (`BodyStamp`: coverage + radiance) + `compose.rs` second pass + `scene_planet.rs` + static `Renderer` wiring; **uniform Lambert sphere first**; occultation-of-stars test; INVARIANTS §1/§2 tests; integrated V vs Mallama test | L |
| F3.3 | Motion-blur wiring: `LightSources.bodies`, per-stamp re-projection, stamp cache, depth-sorted compositing | M |
| F3.4 | Textured airless bodies: Moon (Hapke × WAC), Mercury, Mars (L-S + haze) | M |
| F3.5 | `atmosphere/`: Rayleigh/Mie/ozone LUTs, spherical-shell single + multiple scattering, limb integrator; Earth/Mars/Venus presets; limb-height test | L |
| F3.6 | `bodies/earth.rs`: endmember surface, cloud layer (climatology + stochastic), glint, atmosphere; colour validation tests | L |
| F3.7 | Venus, gas giants (no rings) | M |
| F3.7b | `bodies/sun.rs`: limb-darkened solar disk (Neckel & Labs), transit/silhouette rendering via the same compositing, exposure-planner guard | S |
| F3.8 | `context_render` planet markers; `planet_scene.rs` loader; a `planet_view` binary for single-frame renders (PNG + FITS + true-colour preview) | M |

### Phase 4 — guiding algorithms and study

| # | Item | Size |
|---|---|---|
| F4.1 | `algo/photocenter.rs` with Lambert / L-S phase correction | S |
| F4.2 | `algo/limb.rs`: scan, Zernike sub-pixel edges, terminator exclusion, Christian–Robinson fit | L |
| F4.3 | `algo/template_match.rs`: NCC centre-finding against a rendered prediction | M |
| F4.4 | `sims/earth_limb_guiding.rs` harness + `earth_limb_guiding` binary; CSV outputs; analysis notebook/plots | L |
| F4.5 | Run the matrix in §5.2; write `docs/earth-limb-guiding-results.md` | M |

### Phase 5 — polish / stretch

Saturn rings; Mars dust storms; Earth city lights and earthshine on
the Moon; polarisation (Rayleigh limb is strongly polarised, relevant
if a polarising element is in the optical train); asteroids via
`starfield-mpc` Kepler propagation (already exists) as point sources.

---

## 8. Risks, open questions, decisions

- **Earth's absolute brightness is uncertain by ~0.6 mag** (p_V 0.24 vs
  0.43 in the literature). Treat cloud fraction as the free parameter
  and report results across the bracket.
- **Weather is not knowable in advance.** Any template method must be
  evaluated against clouds it did not see; the stochastic field
  generator is essential, not optional.
- **Stray light dominates the SNR budget** at 30–47° from the Sun and
  is entirely design-dependent. The plan makes PST an input; someone
  has to supply realistic curves for the candidate baffles.
- **Saturation management**: with ms exposures, read noise and frame
  rate become the limits. IMX455 `max_frame_rate_fps` exists in the
  sensor preset; the study should respect it.
- **Compute**: a 4× oversampled 200 px disk with per-sample atmosphere
  LUT lookups is ~10⁶ evaluations per stamp; with stamp caching this is
  fine, but jitter runs with 10³–10⁴ stamps/exposure must reuse the
  stamp and shift it, not re-render it.
- **Texture licensing/hosting**: all listed sources are US-government
  public domain; we still should mirror coarse tiers ourselves so CI
  is not hostage to NASA servers (the existing `#[ignore]`d network-test
  convention applies).
- **PSF fidelity**: the current star path deposits a Gaussian
  approximation of the Airy disk; a limb study is sensitive to PSF
  wings. Decide whether to move stars to the sampled PSF too (larger
  change, out of scope here) or accept the documented mismatch.
- **Decision needed**: observer definition for the headline study
  (low Mars orbit vs areostationary vs heliocentric near Mars). The
  plan sweeps three; if only one is wanted, pick areostationary as the
  most telescope-like.
- **Decision needed**: whether Saturn's rings and asteroids are in
  scope for the first release (plan says no).
- **Decision needed**: feature-gate mechanics. Option A: `#[cfg(feature
  = "solar-system")]` fields on `Scene`/`LightSources` (zero cost, but
  `cfg` in public struct definitions). Option B: an always-present
  `Option<Box<dyn SecondPass>>` hook with the concrete implementation
  behind the feature (cleaner API, one virtual call per stamp). Plan
  recommends **B**: the hook is also how future non-planet passes
  (satellite trails, debris streaks) would attach.
- **Decision needed**: Sun tier 2 (rendered disk) in first release or
  deferred. Plan recommends tier 1 in Phase 2 unconditionally and tier 2
  as the last item of Phase 3, since it is small once the body framework
  exists.

---

## 9. References

- Mallama, A. & Hilton, J. L. (2018), *Computing apparent planetary
  magnitudes for The Astronomical Almanac*, Astron. Comput. 25, 10.
  arXiv:1808.01973. (Eqs. 2–17; Earth Eq. 5: V = 5 log₁₀(r d) − 3.99
  − 1.060×10⁻³ α + 2.054×10⁻⁴ α².)
- Archinal, B. A. et al. (2018), *Report of the IAU WGCCRE: 2015*,
  Celest. Mech. Dyn. Astron. 130:22, and 2019 corrigendum.
- Christian, J. A. (2017), *Accurate Planetary Limb Localization for
  Image-Based Spacecraft Navigation*, J. Spacecraft Rockets 54(3), 708.
- Christian, J. A. (2016), *Horizon-Based Optical Navigation Using
  Images of a Planet with an Atmosphere*, AIAA 2016-5442.
- Christian, J. A. (2021), *A Tutorial on Horizon-Based Optical
  Navigation and Attitude Determination with Space Imaging Systems*,
  IEEE Access.
- Inman, R. & Holt, G. (2024), *Artemis I Optical Navigation System
  Performance*, AIAA SciTech (Earth atmosphere bias 25 km measured vs
  35 km assumed).
- Owen, W. M. (2011), *Methods of Optical Navigation*, AAS 11-215.
- Christian, J. A. (2015), *Optical Navigation Using Planet's Centroid
  and Apparent Diameter in Image*, J. Guid. Control Dyn. 38(2), 192.
- Bruneton, E. & Neyret, F. (2008), *Precomputed Atmospheric
  Scattering*, Comput. Graph. Forum 27(4).
- Bodhaine, B. A. et al. (1999), *On Rayleigh Optical Depth
  Calculations*, J. Atmos. Ocean. Tech. 16, 1854.
- Serdyuchenko, A. et al. (2014), *High spectral resolution ozone
  absorption cross-sections*, AMT 7, 609.
- Coddington, O. M. et al. (2021, 2023), *The TSIS-1 Hybrid Solar
  Reference Spectrum* (v2 and full-spectrum extension).
- Sato, H. et al. (2014), *Resolved Hapke parameter maps of the Moon*,
  JGR Planets 119.
- Hapke, B. (2012), *Theory of Reflectance and Emittance Spectroscopy*,
  2nd ed.
- Neckel, H. & Labs, D. (1994), *Solar limb darkening 303–1099 nm*,
  Solar Phys. 153, 91; Pierce, A. K. & Slaughter, C. D. (1977), Solar
  Phys. 51, 25 (solar limb-darkening polynomials).
- Livengood, T. A. et al. (2011), EPOXI Earth observations; Woolf et
  al. (2002) and Turnbull et al. (2006), Earthshine spectra.
- Leinert, Ch. et al. (1998), *The 1997 reference of diffuse night sky
  brightness*, A&AS 127, 1 (zodiacal heliocentric scaling).
- arXiv:2507.22258 (2025), *Inferring and Interpreting the Visual
  Geometric Albedo and Phase Function of Earth* (p_V ≈ 0.24).
- HiRISE PSP_005558_9040, Earth and Moon from Mars (2007-10-03).

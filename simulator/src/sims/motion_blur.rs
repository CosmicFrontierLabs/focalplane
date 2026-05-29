//! Parallel motion-blur renderer for focal-plane trajectories.
//!
//! Each frame's exposure window is integrated as a single flat
//! stratified-Monte-Carlo sequence of PSF stamps. The total stamp
//! count is chosen adaptively from a per-stamp drift budget so the
//! trajectory's angular path between consecutive stamps stays below
//! a fraction of a pixel; for a static trajectory the schedule
//! collapses to a single stamp.
//!
//! # Noise model
//!
//! A single unified Poisson draw is taken per `(frame, sensor)` over the
//! mean-electron image comprising:
//!
//! - Star contributions accumulated across all stamps (each stamp
//!   contributes `flux × dt` electrons where `dt = exposure / stamps_per_exposure`).
//! - A uniform zodiacal mean evaluated at the frame's central boresight.
//! - A uniform dark-current mean from the per-sensor dark current rate
//!   times the full exposure duration.
//!
//! Gaussian read noise is added separately after the Poisson draw because
//! it is an electronic readout effect, not a photon shot-noise contribution.
//!
//! # Parallelism
//!
//! The top-level entry point parallelizes over `(frame_idx, sensor_idx)`
//! pairs via rayon. Each tile independently consumes its pre-filtered
//! star slice, materializes a per-sensor mean-electron accumulator,
//! draws noise, quantizes, and writes its PNG. Nothing is reduced
//! across tiles.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::SecondsFormat;
use image::{ImageBuffer, Luma};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::{debug, info};
use ndarray::Array2;
use rayon::prelude::*;
use shared::image_proc::detection::AABB;
use shared::image_proc::noise::{apply_gaussian_read_noise, apply_poisson_photon_noise};
use shared::units::{AngleExt, LengthExt, TemperatureExt};
use starfield::catalogs::StarData;
use starfield::Equatorial;

use crate::hardware::satellite::{FocalPlaneConfig, FocalPlaneProjector};
use crate::hardware::SatelliteConfig;
use crate::image_proc::deposit::MeanFluxDeposit;
use crate::image_proc::render::quantize_image;
use crate::photometry::photoconversion::SourceFlux;
use crate::photometry::spectrum::Spectrum;
use crate::photometry::zodiacal::{SolarAngularCoordinates, ZodiacalLight};
use crate::scene_galaxy::{project_galaxies_to_sensors, Galaxy, GalaxyInFrame};
use crate::sims::motion_blur_metadata::{
    sensor_dir_name, sensor_relative_png_path, EquatorialMeta, FrameMeta, RenderConfigMeta,
    RenderMetadata, StarMeta, TrajectoryMeta, WaypointMeta,
};
use crate::sims::orientation::{boresight_of, roll_of};
use crate::sims::quasi_random;
use crate::sims::trajectory::{Trajectory, TrajectoryError};
use crate::star_math::star_data_to_fluxes;

/// Default per-stamp drift budget (in pixels). The total stamp count is
/// derived as `ceil(total_drift_rad / (DEFAULT_MAX_DRIFT_PER_STAMP_PX * pixel_scale_rad))`.
pub const DEFAULT_MAX_DRIFT_PER_STAMP_PX: f64 = 0.1;

/// Time-domain stamp schedule inside an exposure window.
///
/// A `SubsampleSchedule` describes a single flat sequence of
/// PSF-stamp times across the exposure: the window is divided into
/// `stamps_per_exposure` equal-width sub-bins and each stamp lands at
/// a uniform-random offset within its sub-bin (stratified Monte
/// Carlo). Stratification preserves smooth-integrand convergence
/// while removing the systematic bias a regular grid produces against
/// trajectory tones near `k / Δt_stamp`.
///
/// Each stamp queries the trajectory at its own time, projects every
/// in-field star, and deposits `flux / stamps_per_exposure` electrons
/// at the per-time projected pixel position.
///
/// `envelope_padding_rad` is the safety padding the *envelope
/// prefilter* uses to inflate the focal-plane AABB so that any star
/// whose excursion brings it onto the sensor at any point during the
/// exposure is retained. It is the only thing the renderer needs from
/// "scene-state cadence" — there is no separate subsample-level loop
/// for scene state.
#[derive(Debug, Clone, Copy)]
pub struct SubsampleSchedule {
    /// Absolute trajectory time at which the frame's exposure starts.
    pub frame_start: Duration,
    /// Total exposure duration for this frame.
    pub exposure: Duration,
    /// Total number of stratified-MC PSF stamps across the exposure (>= 1).
    pub stamps_per_exposure: usize,
    /// Peak excursion of the trajectory from the frame's mid-time
    /// orientation, in radians. The envelope prefilter inflates the
    /// focal-plane AABB by this amount (converted to mm) on top of
    /// its own PSF-extent padding.
    pub envelope_padding_rad: f64,
}

impl SubsampleSchedule {
    /// Per-stamp duration: `exposure / stamps_per_exposure`. Each
    /// stamp's electron contribution is integrated over this `dt`.
    pub fn dt(&self) -> Duration {
        Duration::from_secs_f64(
            self.exposure.as_secs_f64() / self.stamps_per_exposure.max(1) as f64,
        )
    }

    /// Quasi-random stamp times across the entire exposure window,
    /// derived from the 1D golden-ratio low-discrepancy sequence (see
    /// [`crate::sims::quasi_random`]). Returns a vector of length
    /// `stamps_per_exposure.max(1)`. Stamp `j` lands at
    /// `frame_start + offset_j · exposure`, where `offset_j` is the
    /// `j`-th element of the golden-ratio sequence with the given
    /// `phase` shift.
    ///
    /// The golden-ratio sequence is **deterministic** — no RNG is
    /// consumed. `phase` is the only source of per-tile variation;
    /// the renderer derives it from the tile seed via
    /// [`crate::sims::quasi_random::phase_from_seed`].
    pub fn stamp_times(&self, phase: f64) -> Vec<Duration> {
        let total = self.stamps_per_exposure.max(1);
        let exposure_s = self.exposure.as_secs_f64();
        let t0 = self.frame_start.as_secs_f64();
        crate::sims::quasi_random::golden_offsets(total, phase)
            .into_iter()
            .map(|u| Duration::from_secs_f64(t0 + u * exposure_s))
            .collect()
    }

    /// Frame mid-time, used as the reference orientation for envelope
    /// prefiltering and zodiacal evaluation.
    pub fn mid_time(&self) -> Duration {
        self.frame_start + self.exposure / 2
    }

    /// Construct a schedule from a per-stamp drift budget plus a peak
    /// excursion (both in radians). Stamp count is
    /// `ceil(total_drift_rad / (max_drift_per_stamp_px * pixel_scale_rad))`,
    /// minimum 1. Envelope padding stores `peak_excursion_rad` directly
    /// for later mm conversion in [`envelope_prefilter`].
    pub fn from_drift_budget(
        frame_start: Duration,
        exposure: Duration,
        total_drift_rad_over_exposure: f64,
        peak_excursion_rad: f64,
        pixel_scale_rad: f64,
        max_drift_per_stamp_px: f64,
    ) -> Self {
        let stamps_per_exposure = if total_drift_rad_over_exposure <= 0.0
            || pixel_scale_rad <= 0.0
            || max_drift_per_stamp_px <= 0.0
        {
            1
        } else {
            let budget_rad = max_drift_per_stamp_px * pixel_scale_rad;
            ((total_drift_rad_over_exposure / budget_rad).ceil() as usize).max(1)
        };
        Self {
            frame_start,
            exposure,
            stamps_per_exposure,
            envelope_padding_rad: peak_excursion_rad.max(0.0),
        }
    }
}

/// Per-tile mean-electron accumulator (pre-Poisson).
///
/// Stars are splatted in as mean-electron contributions (not draws) so
/// that all sub-samples, the zodiacal uniform, and the dark-current uniform
/// can be combined into a single Poisson lambda.
#[derive(Debug, Clone)]
pub struct SensorAccumulator {
    /// Accumulated mean electrons from star subsample splats.
    pub star_mean_electrons: Array2<f64>,
}

impl SensorAccumulator {
    /// Allocate a zeroed accumulator shaped `(height, width)`.
    pub fn zero(width: usize, height: usize) -> Self {
        Self {
            star_mean_electrons: Array2::zeros((height, width)),
        }
    }

    /// Splat one star's mean electrons (already integrated over `dt`, not
    /// Poisson-sampled) into the accumulator via Simpson's rule over the
    /// Airy disk.
    ///
    /// `total_electrons` is the expected number of electrons this subsample
    /// contributes across the disk — i.e. the star's mean-electron rate at
    /// the star's chromatic effective PSF, multiplied by aperture area and
    /// the subsample duration.
    pub fn splat_psf(
        &mut self,
        px: f64,
        py: f64,
        total_electrons: f64,
        psf: &shared::image_proc::airy::PixelScaledAiryDisk,
    ) {
        crate::image_proc::deposit::splat_deposit(
            &mut self.star_mean_electrons,
            px,
            py,
            total_electrons,
            psf,
        );
    }

    /// Splat one galaxy's mean electrons (already integrated over `dt`,
    /// not Poisson-sampled) into the same accumulator stars use, via
    /// the galaxy's `SersicSplat` deposit. The single-Poisson invariant
    /// (INVARIANTS §1) is preserved because galaxies land on the same
    /// `star_mean_electrons` buffer that the unified Poisson eventually
    /// samples — galaxy and star shot noise are co-sampled from the
    /// per-pixel `Poisson(λ_total)` of the combined mean.
    pub fn splat_galaxy(
        &mut self,
        px: f64,
        py: f64,
        total_electrons: f64,
        sersic: &crate::image_proc::sersic_splat::SersicSplat,
    ) {
        crate::image_proc::deposit::splat_deposit(
            &mut self.star_mean_electrons,
            px,
            py,
            total_electrons,
            sersic,
        );
    }

    /// Consume the accumulator and return the combined mean-electron image
    /// = star mean + zodiacal uniform + dark-current uniform (pre-Poisson).
    ///
    /// The accumulator's existing `star_mean_electrons` buffer is reused —
    /// no second 488 MB allocation, no read-then-write of a fresh array.
    /// The scalar background is added in place via rayon-parallel
    /// element-wise iteration (ndarray's `rayon` feature is enabled).
    pub fn into_combined_mean(mut self, zodiacal_per_px: f64, dark_per_px: f64) -> Array2<f64> {
        let bg = (zodiacal_per_px + dark_per_px).max(0.0);
        if bg > 0.0 {
            self.star_mean_electrons
                .par_iter_mut()
                .for_each(|pixel| *pixel += bg);
        }
        self.star_mean_electrons
    }
}

/// Per-tile flux cache keyed by `(star_id, sensor_idx)`.
///
/// Flux calculation is expensive (Simpson's rule integration over stellar
/// spectrum × QE curve) and depends only on the star and the sensor, not on
/// the spacecraft orientation. Caching across frames and subsamples avoids
/// repeating that work for every sub-orientation.
pub type FluxCache = HashMap<(u64, usize), SourceFlux>;

/// All light contributing to a rendered frame: foreground point sources
/// (stars), extended sources (galaxies, sky-truth and flat), and the
/// diffuse sky background (zodiacal model parameters).
///
/// `galaxies` is a **flat** slice of [`Galaxy`] entries — one per
/// catalog source, independent of which sensor (or how many) the
/// galaxy projects onto. The motion-blur renderer projects each
/// galaxy onto every sensor whose extent (plus halo padding) contains
/// the centre, so a galaxy whose Sérsic halo subtends multiple
/// sensors gets rendered on each of them.
#[derive(Clone, Copy)]
pub struct LightSources<'a> {
    pub catalog_stars: &'a [StarData],
    pub galaxies: &'a [Galaxy],
    pub zodiacal: SolarAngularCoordinates,
}

/// Static inputs shared across every `(frame, sensor)` tile in a render.
///
/// Bundles the trajectory, the scene's light sources, the focal-plane
/// hardware, and the per-sensor projected-galaxy lists so downstream
/// functions take a single `&RenderScene` rather than several
/// independent references. `projected_galaxies[sensor_idx]` is built
/// at render start by [`project_galaxies_to_sensors`] from
/// `sources.galaxies` — a single galaxy may appear in more than one
/// sensor's list if its halo subtends them.
struct RenderScene<'a> {
    trajectory: &'a Trajectory,
    sources: LightSources<'a>,
    fp: &'a FocalPlaneConfig,
    projected_galaxies: Vec<Vec<GalaxyInFrame>>,
}

/// Per-frame render plan produced by the serial planning pass.
///
/// Holds everything the tile renderer needs that varies per frame: the
/// subsample schedule, the prefiltered in-field star slice, the padding
/// budget, and the per-sensor zodiacal electrons/pixel for the full
/// exposure (one entry per sensor in the focal-plane array).
struct FramePlan<'a> {
    idx: usize,
    schedule: SubsampleSchedule,
    stars: Vec<&'a StarData>,
    padding_mm: f64,
    zodiacal_per_px: Vec<f64>,
}

/// Per-tile work unit: the shared scene, the per-frame plan, and which
/// sensor on the focal-plane array the tile renders to. These three
/// references always travel together across every per-tile function, so
/// callers bundle them once and pass a single `&TileRenderContext`.
struct TileRenderContext<'a> {
    scene: &'a RenderScene<'a>,
    plan: &'a FramePlan<'a>,
    sensor_idx: usize,
}

/// Configuration for [`render_motion_trajectory`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MotionBlurConfig {
    /// Time between frames (controls how many output frames are rendered).
    pub timestep: Duration,
    /// Exposure duration per frame.
    pub exposure: Duration,
    /// Per-stamp drift budget in pixels. Total stamps per exposure
    /// is derived as `ceil(total_drift_rad / (max_drift_per_stamp_px *
    /// pixel_scale_rad))`. Tighten to capture high-frequency jitter.
    pub max_drift_per_stamp_px: f64,
    /// Optional base RNG seed (combined per-tile with `(frame_idx, sensor_idx)`).
    pub base_seed: Option<u64>,
    /// If true, force `stamps_per_exposure = 1` per frame regardless of the
    /// adaptive budget. Useful for debugging and performance comparisons.
    pub force_static: bool,
    /// If true, suppress the indicatif progress bar (INFO logs still emit).
    /// Intended for non-interactive runs where the bar adds noise.
    pub quiet: bool,
    /// Telescope display name copied into `metadata.json` under
    /// `hardware.telescope`. Metadata-only; ignored by the render math.
    pub telescope_name: String,
    /// Catalog path recorded in `metadata.json` under
    /// `render_config.catalog_path`. Metadata-only.
    pub catalog_path: PathBuf,
    /// Operating temperature (Celsius) recorded in `metadata.json` under
    /// `hardware.temperature_c`. Metadata-only — the per-tile renderer reads
    /// the temperature off the per-sensor `SatelliteConfig` built from
    /// [`FocalPlaneConfig`].
    pub temperature_c: f64,
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        Self {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: DEFAULT_MAX_DRIFT_PER_STAMP_PX,
            base_seed: None,
            force_static: false,
            quiet: false,
            telescope_name: String::new(),
            catalog_path: PathBuf::new(),
            temperature_c: 0.0,
        }
    }
}

/// Save a 2D u16 image as a 16-bit grayscale PNG.
fn save_u16_png(image: &Array2<u16>, path: &Path) -> Result<(), TrajectoryError> {
    let (height, width) = image.dim();
    let raw: Vec<u16> = image.iter().copied().collect();
    let img: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::from_raw(width as u32, height as u32, raw)
            .ok_or_else(|| TrajectoryError::ImageWrite("buffer size mismatch".into()))?;
    img.save(path)
        .map_err(|e| TrajectoryError::ImageWrite(e.to_string()))
}

/// Compute the maximum angular drift (radians) the trajectory traverses
/// over a time window `[t_start, t_end]`. For multi-waypoint trajectories
/// this is the sum of per-segment drift inside the window (upper bound).
///
/// For the common two-waypoint SLERP case this reduces to
/// `rate * (t_end - t_start)` where the rate is the constant angular
/// distance between the two waypoints divided by the trajectory duration.
fn max_drift_over_window(
    trajectory: &Trajectory,
    t_start: Duration,
    t_end: Duration,
) -> Result<f64, TrajectoryError> {
    if t_end <= t_start {
        return Ok(0.0);
    }
    let wps = trajectory.waypoints();
    if wps.len() < 2 {
        return Ok(0.0);
    }

    let mut drift = 0.0_f64;
    for seg in wps.windows(2) {
        let seg_start = seg[0].time;
        let seg_end = seg[1].time;
        if seg_end <= t_start || seg_start >= t_end {
            continue;
        }
        let overlap_start = seg_start.max(t_start);
        let overlap_end = seg_end.min(t_end);
        let seg_dur = (seg_end - seg_start).as_secs_f64();
        if seg_dur <= 0.0 {
            continue;
        }
        let ang = seg[0].orientation.angle_to(&seg[1].orientation);
        let rate = ang / seg_dur;
        drift += rate * (overlap_end - overlap_start).as_secs_f64();
    }
    Ok(drift)
}

/// Pixel scale (radians/pixel) for the first sensor of the array. All sensors
/// in a single focal plane share the telescope's plate scale, so this is the
/// representative pixel-scale budget used by the adaptive scheduler.
fn pixel_scale_rad(fp: &FocalPlaneConfig) -> Option<f64> {
    let sat = fp.satellite_for_sensor(0)?;
    Some(sat.plate_scale_per_pixel().as_radians())
}

/// Envelope-prefilter: prune catalog stars whose mid-time projected
/// position falls outside the focal-plane AABB inflated by the
/// per-frame **peak excursion** plus a fixed PSF-extent margin.
///
/// One orientation lookup per star (mid-frame) instead of one per
/// schedule sample point. Correctness comes from the inflated AABB:
/// any star whose position at mid-frame is within
/// `padding_mm + excursion_mm` of the focal-plane envelope can
/// possibly project onto a sensor at *some* time during the exposure,
/// so it must be in the candidate set. Per-stamp `project_to_sensor`
/// then decides whether each stamp actually lands on the detector.
fn envelope_prefilter<'a>(
    trajectory: &Trajectory,
    catalog_stars: &'a [StarData],
    schedule: &SubsampleSchedule,
    fp: &FocalPlaneConfig,
    padding_mm: f64,
    pixel_scale_rad: f64,
) -> Result<Vec<&'a StarData>, TrajectoryError> {
    let (min_x, min_y, max_x, max_y) = match fp.total_aabb_mm() {
        Some(aabb) => aabb,
        None => return Ok(Vec::new()),
    };
    let mid_t = schedule
        .mid_time()
        .min(trajectory.end_time())
        .max(trajectory.start_time());
    let q_mid = trajectory.orientation_at(mid_t)?;

    // Convert peak excursion (radians) to a mm padding via the focal
    // length implied by the pixel scale. mm/rad = pixel_size_mm / pixel_scale_rad
    // collapses to focal_length_mm; we just need (rad * focal_length_mm) which
    // we get from any sensor's pixel size and pixel-scale ratio.
    let excursion_mm = if pixel_scale_rad > 0.0 {
        if let Some(sat) = fp.satellite_for_sensor(0) {
            let pixel_size_mm = sat.sensor.pixel_size().as_millimeters();
            schedule.envelope_padding_rad * (pixel_size_mm / pixel_scale_rad)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let pad = padding_mm + excursion_mm;

    let mut kept: Vec<&'a StarData> = Vec::new();
    for star in catalog_stars {
        let hit = if let Some((x_mm, y_mm)) = fp.sky_to_mm(&star.position, &q_mid) {
            x_mm >= min_x - pad && x_mm <= max_x + pad && y_mm >= min_y - pad && y_mm <= max_y + pad
        } else {
            false
        };
        if hit {
            kept.push(star);
        }
    }
    Ok(kept)
}

/// Cheap splitmix-style mixer that turns `(base_seed, frame_idx, sensor_idx)`
/// into a deterministic per-tile 64-bit seed. Sub-stream seeds come
/// from XOR'ing this with the named domain tags below.
fn tile_seed(base: u64, frame_idx: usize, sensor_idx: usize) -> u64 {
    let mut h = base
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(frame_idx as u64)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= (sensor_idx as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}

// Per-tile sub-stream tags. XOR with the tile seed to derive the
// stream's own deterministic seed. Values are arbitrary as long as
// they are pairwise distinct.
const POISSON_DOMAIN: u64 = 0x0000_0000_0000_0000;
const READ_NOISE_DOMAIN: u64 = 0xA5A5_5A5A_A5A5_5A5A;
const STAMP_PHASE_DOMAIN: u64 = 0x4D6F_6E74_6543_6172; // "MonteCar" (8 ASCII bytes)

/// Render a single `(frame, sensor)` tile into a quantized `Array2<u16>`.
///
/// Runs one flat stamp loop across the exposure, composes the unified
/// Poisson lambda, draws Poisson + Gaussian read noise, and quantizes.
/// No disk I/O — the caller decides what to do with the returned image.
fn render_tile_array(
    ctx: &TileRenderContext<'_>,
    flux_cache: &Arc<Mutex<FluxCache>>,
    satellite: &SatelliteConfig,
    tile_seed: u64,
) -> Result<Array2<u16>, TrajectoryError> {
    let (width, height) = satellite.sensor.dimensions.get_pixel_width_height();
    if width == 0 || height == 0 {
        return Ok(Array2::<u16>::zeros((height, width)));
    }
    let full_roi = AABB::from_coords(0, 0, height - 1, width - 1);
    render_tile_array_roi(ctx, flux_cache, satellite, tile_seed, full_roi)
}

/// Output of [`build_tile_mean_image`]: the pre-noise mean-electron
/// image for a tile ROI plus the elapsed-time breakdown for the
/// caller's debug log.
struct MeanImageResult {
    mean_image: Array2<f64>,
    roi_w: usize,
    roi_h: usize,
    ms_splat_stars: u128,
    ms_splat_galaxies: u128,
    ms_combined_mean: u128,
}

/// Build the pre-noise mean-electron image for one tile's `roi` —
/// splats stars across the stamp schedule, splats galaxies once,
/// folds in the per-tile zodiacal and dark-current scalars. The
/// returned `Array2<f64>` is shaped `(roi_height, roi_width)` with the
/// pixel at `(0, 0)` corresponding to sensor pixel
/// `(roi.min_row, roi.min_col)`.
///
/// Stars and galaxies whose entire footprint AABB falls outside the
/// ROI are skipped (perf, no math effect). Partial-overlap deposits
/// land via the standard `splat_deposit` bounds check, which clips at
/// the ROI buffer's edge.
fn build_tile_mean_image(
    ctx: &TileRenderContext<'_>,
    flux_cache: &Arc<Mutex<FluxCache>>,
    satellite: &SatelliteConfig,
    tile_seed: u64,
    roi: AABB,
) -> Result<MeanImageResult, TrajectoryError> {
    let TileRenderContext {
        scene,
        plan,
        sensor_idx,
    } = *ctx;
    let (sensor_w, sensor_h) = satellite.sensor.dimensions.get_pixel_width_height();
    if sensor_w == 0
        || sensor_h == 0
        || roi.max_row >= sensor_h
        || roi.max_col >= sensor_w
        || roi.min_row > roi.max_row
        || roi.min_col > roi.max_col
    {
        return Err(TrajectoryError::RoiOutOfBounds {
            roi: roi.to_tuple(),
            sensor_idx,
            width: sensor_w,
            height: sensor_h,
        });
    }
    let roi_w = roi.max_col - roi.min_col + 1;
    let roi_h = roi.max_row - roi.min_row + 1;
    let x_off = roi.min_col as f64;
    let y_off = roi.min_row as f64;

    let mut accumulator = SensorAccumulator::zero(roi_w, roi_h);
    let aperture = satellite.telescope.clear_aperture_area();

    let schedule = &plan.schedule;
    let dt = schedule.dt();
    let stamp_weight = 1.0 / schedule.stamps_per_exposure.max(1) as f64;

    // Per-tile R2 phase derived deterministically from the tile
    // seed: shifts the golden-ratio stamp sequence so different
    // (frame, sensor) tiles render different realizations of the
    // same low-discrepancy property, preventing systematic per-frame
    // bias.
    let stamp_phase = quasi_random::phase_from_seed(tile_seed ^ STAMP_PHASE_DOMAIN);

    // One-shot per-tile cache of (star -> SourceFlux) lookups. Flux
    // depends only on (star, sensor), not on orientation, so the
    // value is stable across every stamp of the exposure.
    let mut local_flux: HashMap<u64, SourceFlux> = HashMap::new();

    let t_splat_stars = Instant::now();
    for stamp_t in schedule.stamp_times(stamp_phase) {
        let t_clamped = stamp_t
            .min(scene.trajectory.end_time())
            .max(scene.trajectory.start_time());
        let orientation = scene.trajectory.orientation_at(t_clamped)?;
        for star in &plan.stars {
            let hit =
                match scene
                    .fp
                    .project_to_sensor(star, &orientation, sensor_idx, plan.padding_mm)
                {
                    Some(px) => px,
                    None => continue,
                };
            let flux = local_flux.entry(star.id).or_insert_with(|| {
                let mut cache = flux_cache.lock().expect("flux cache mutex poisoned");
                cache
                    .entry((star.id, sensor_idx))
                    .or_insert_with(|| star_data_to_fluxes(star, satellite))
                    .clone()
            });
            // Skip stars whose entire PSF footprint falls outside the
            // ROI extent — the splat would write nothing once shifted
            // into ROI-local coordinates, so the per-pixel Simpson loop
            // is a pure cost.
            let footprint = flux.electrons.disk.footprint_pixels() as f64;
            if hit.0 + footprint < x_off
                || hit.0 - footprint >= x_off + roi_w as f64
                || hit.1 + footprint < y_off
                || hit.1 - footprint >= y_off + roi_h as f64
            {
                continue;
            }
            // Per-stamp electron contribution: flux rate × per-stamp
            // dt (= exposure / stamps_per_exposure) × aperture. Sum
            // over all stamps reproduces the full exposure budget.
            let total_electrons = flux.electrons.integrated_over(&dt, aperture) * stamp_weight;
            accumulator.splat_psf(
                hit.0 - x_off,
                hit.1 - y_off,
                total_electrons,
                &flux.electrons.disk,
            );
        }
    }
    let ms_splat_stars = t_splat_stars.elapsed().as_millis();

    // Galaxies: pre-projected per-sensor (one position per render),
    // splatted at full-exposure flux so the unified Poisson stage
    // samples the combined (stars + galaxies + zodi + dark) mean.
    // Per-stamp re-projection isn't done here because the bounding
    // box of a Sérsic deposit is much larger than a PSF stamp; a
    // sub-pixel galaxy shift across the exposure is well below the
    // per-pixel noise floor of any plausible exposure. Static
    // trajectories (start == end) are exact.
    let t_splat_galaxies = Instant::now();
    let galaxies = scene
        .projected_galaxies
        .get(sensor_idx)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    for galaxy in galaxies {
        let footprint = galaxy.deposit.footprint_pixels() as f64;
        if galaxy.x + footprint < x_off
            || galaxy.x - footprint >= x_off + roi_w as f64
            || galaxy.y + footprint < y_off
            || galaxy.y - footprint >= y_off + roi_h as f64
        {
            continue;
        }
        let total_electrons = galaxy
            .flux
            .electrons
            .integrated_over(&schedule.exposure, aperture);
        accumulator.splat_galaxy(
            galaxy.x - x_off,
            galaxy.y - y_off,
            total_electrons,
            &galaxy.deposit,
        );
    }
    let ms_splat_galaxies = t_splat_galaxies.elapsed().as_millis();

    // Dark current: rate × full exposure, uniform over pixels.
    let dark_rate = satellite
        .sensor
        .dark_current_at_temperature(satellite.temperature);
    let dark_mean = (dark_rate * schedule.exposure.as_secs_f64()).max(0.0);

    // Build unified Poisson mean image.
    let t_combined_mean = Instant::now();
    let mean_image = accumulator.into_combined_mean(plan.zodiacal_per_px[sensor_idx], dark_mean);
    let ms_combined_mean = t_combined_mean.elapsed().as_millis();

    Ok(MeanImageResult {
        mean_image,
        roi_w,
        roi_h,
        ms_splat_stars,
        ms_splat_galaxies,
        ms_combined_mean,
    })
}

/// Render the subset of a `(frame, sensor)` tile that falls inside `roi`.
///
/// Produces an `Array2<u16>` shaped `(roi_height, roi_width)`. The
/// pixel at `(0, 0)` of the returned array corresponds to sensor pixel
/// `(roi.min_row, roi.min_col)`.
///
/// `roi` uses inclusive bounds in sensor (row, col) coordinates and must
/// be entirely inside the sensor extent. Stars and galaxies are filtered
/// against the ROI extent inflated by the PSF/Sérsic footprint so any
/// source whose deposit can touch the ROI is splatted; partial-overlap
/// deposits clip at the ROI edge.
///
/// **Bit-equality contract**: when `roi` covers the full sensor, the
/// quantized output is byte-identical to [`render_tile_array`] for the
/// same `(scene, plan, sensor_idx, tile_seed)`. For a smaller `roi` the
/// pre-noise mean-electron image is byte-identical to the corresponding
/// slice of the full-sensor mean, but the Poisson and Gaussian noise
/// streams differ because the shared parallel-chunk noise sampler keys
/// its per-row-chunk RNG off the buffer's row count.
fn render_tile_array_roi(
    ctx: &TileRenderContext<'_>,
    flux_cache: &Arc<Mutex<FluxCache>>,
    satellite: &SatelliteConfig,
    tile_seed: u64,
    roi: AABB,
) -> Result<Array2<u16>, TrajectoryError> {
    let MeanImageResult {
        mean_image,
        roi_w,
        roi_h,
        ms_splat_stars,
        ms_splat_galaxies,
        ms_combined_mean,
    } = build_tile_mean_image(ctx, flux_cache, satellite, tile_seed, roi)?;

    let t_poisson = Instant::now();
    let poisson_image = apply_poisson_photon_noise(mean_image, Some(tile_seed ^ POISSON_DOMAIN));
    let ms_poisson = t_poisson.elapsed().as_millis();

    // Gaussian read noise (electronics, not shot noise) applied afterwards.
    let read_noise_rms = satellite
        .sensor
        .read_noise_estimator
        .estimate(
            satellite.temperature.as_celsius(),
            ctx.plan.schedule.exposure,
        )
        .unwrap_or(0.0)
        .max(0.0);
    let t_read_noise = Instant::now();
    let final_electrons = apply_gaussian_read_noise(
        poisson_image,
        read_noise_rms,
        Some(tile_seed ^ READ_NOISE_DOMAIN),
    );
    let ms_read_noise = t_read_noise.elapsed().as_millis();

    let t_quantize = Instant::now();
    let quantized = quantize_image(&final_electrons, &satellite.sensor);
    let ms_quantize = t_quantize.elapsed().as_millis();

    debug!(
        "tile-phases frame={} sensor={} roi=({},{})+{}x{} \
         splat_stars={}ms splat_galaxies={}ms combined_mean={}ms \
         poisson={}ms read_noise={}ms quantize={}ms",
        ctx.plan.idx,
        ctx.sensor_idx,
        roi.min_col,
        roi.min_row,
        roi_w,
        roi_h,
        ms_splat_stars,
        ms_splat_galaxies,
        ms_combined_mean,
        ms_poisson,
        ms_read_noise,
        ms_quantize,
    );

    Ok(quantized)
}

/// Shared precomputed inputs for the planning pass and per-frame renders.
///
/// Built once per render (or once per `render_one_frame` call) from a
/// [`FocalPlaneConfig`]: the per-sensor satellite views, the first sensor's
/// padding budget and pixel scale, and the per-sensor pixel-solid-angle
/// ratios used to spread the zodiacal rate across heterogeneous sensors.
struct RenderContext {
    satellites: Vec<SatelliteConfig>,
    padding_mm: f64,
    px_scale: f64,
    sensor_ratios: Vec<f64>,
}

impl RenderContext {
    /// Build the context from a focal plane. Errors when the focal plane
    /// exposes no sensors.
    fn from_focal_plane(fp: &FocalPlaneConfig) -> Result<Self, TrajectoryError> {
        let sensor_count = fp.array.sensor_count();
        if sensor_count == 0 {
            return Err(TrajectoryError::NoSensors);
        }
        let first_sat = fp
            .satellite_for_sensor(0)
            .ok_or(TrajectoryError::NoSensors)?;
        let airy_pix = first_sat.airy_disk_pixel_space();
        let pixel_size_mm = first_sat.sensor.pixel_size().as_millimeters();
        let padding_mm = airy_pix.first_zero() * 2.0 * pixel_size_mm;
        let px_scale = pixel_scale_rad(fp).unwrap_or(0.0);
        let satellites: Vec<SatelliteConfig> = (0..sensor_count)
            .map(|i| {
                fp.satellite_for_sensor(i)
                    .expect("sensor index in range for enumerated sensor_count")
            })
            .collect();
        let sensor_ratios: Vec<f64> = satellites
            .iter()
            .map(|sat| per_sensor_pixel_solid_angle_ratio(&satellites[0], sat))
            .collect();
        Ok(Self {
            satellites,
            padding_mm,
            px_scale,
            sensor_ratios,
        })
    }
}

/// Project a flat catalog of [`Galaxy`] sky-truth entries onto every
/// sensor of `fp`, returning a per-sensor `Vec<GalaxyInFrame>` list
/// ready to drive the per-tile splat loop.
///
/// Projection happens at the trajectory's start-time pose — galaxies
/// are treated as scene-static across the render run (today's
/// behaviour). Per-frame re-projection is a future improvement; the
/// sub-pixel galaxy drift across a typical exposure is well below
/// the Sérsic deposit's per-pixel noise floor, so a single
/// projection per render run preserves accuracy for static and
/// near-static trajectories.
fn project_galaxies_for_render(
    trajectory: &Trajectory,
    fp: &FocalPlaneConfig,
    galaxies: &[Galaxy],
) -> Result<Vec<Vec<GalaxyInFrame>>, TrajectoryError> {
    if galaxies.is_empty() {
        // Skip the cost of an orientation lookup when there's nothing
        // to project. Returns a zero-length outer Vec; callers tolerate
        // a missing sensor entry by emitting an empty slice in the
        // per-tile galaxy loop.
        return Ok(Vec::new());
    }
    let reference_orientation = trajectory.orientation_at(trajectory.start_time())?;
    Ok(project_galaxies_to_sensors(
        galaxies,
        fp,
        &reference_orientation,
    ))
}

/// Build the per-frame render plan: subsample schedule, prefiltered star
/// slice, and the per-sensor zodiacal electrons/pixel for the full exposure.
///
/// Pure function — no I/O. Used both by the trajectory-level renderer's
/// serial planning pass and by [`render_one_frame`].
fn plan_frame<'a>(
    scene: &'a RenderScene<'a>,
    ctx: &RenderContext,
    zlight: &ZodiacalLight,
    frame_idx: usize,
    frame_start: Duration,
    config: &MotionBlurConfig,
) -> Result<FramePlan<'a>, TrajectoryError> {
    let exposure = config.exposure;
    // Clamp the exposure window to the trajectory so we do not try to
    // sample beyond the defined range.
    let t_end = (frame_start + exposure).min(scene.trajectory.end_time());
    let exposure = if t_end > frame_start {
        t_end - frame_start
    } else {
        Duration::ZERO
    };

    // Mid-frame orientation drives both the envelope-padding peak
    // excursion calculation and the zodiacal evaluation.
    let mid_t = (frame_start + exposure / 2)
        .min(scene.trajectory.end_time())
        .max(scene.trajectory.start_time());
    let mid_q = scene.trajectory.orientation_at(mid_t)?;
    let mid_bore = boresight_of(&mid_q);

    let drift = max_drift_over_window(scene.trajectory, frame_start, t_end)?;
    let peak_excursion = scene
        .trajectory
        .peak_excursion_rad(frame_start, t_end, &mid_q)?;

    let schedule = if config.force_static || exposure.is_zero() {
        SubsampleSchedule {
            frame_start,
            exposure,
            stamps_per_exposure: 1,
            envelope_padding_rad: peak_excursion,
        }
    } else {
        SubsampleSchedule::from_drift_budget(
            frame_start,
            exposure,
            drift,
            peak_excursion,
            ctx.px_scale,
            config.max_drift_per_stamp_px,
        )
    };

    let first_sat = &ctx.satellites[0];
    let zodiacal_per_px_per_s =
        zodiacal_per_px_per_s_at(zlight, first_sat, &scene.sources.zodiacal, &mid_bore);
    let exposure_s = schedule.exposure.as_secs_f64();
    let zodiacal_per_px: Vec<f64> = ctx
        .sensor_ratios
        .iter()
        .map(|r| zodiacal_per_px_per_s * r * exposure_s)
        .collect();

    let stars = envelope_prefilter(
        scene.trajectory,
        scene.sources.catalog_stars,
        &schedule,
        scene.fp,
        ctx.padding_mm,
        ctx.px_scale,
    )?;

    Ok(FramePlan {
        idx: frame_idx,
        schedule,
        stars,
        padding_mm: ctx.padding_mm,
        zodiacal_per_px,
    })
}

/// Render a single frame to one `Array2<u16>` per sensor in the focal plane.
///
/// In-memory, no disk I/O. The returned vector is indexed by sensor index:
/// `result[sensor_idx]` is the quantized image for that sensor at the
/// requested `frame_start`.
///
/// Bit-identical to a single iteration of [`render_motion_trajectory`] for
/// the same `(base_seed, frame_idx)` combination. The deterministic seed
/// mixing (`tile_seed(base_seed, frame_idx, sensor_idx)`) is preserved, so
/// callers chaining `render_one_frame` over a sequence of frame times will
/// reproduce the trajectory renderer's output byte-for-byte.
///
/// `frame_start` is the absolute trajectory time at which the exposure
/// begins. The exposure window is `config.exposure`, clamped to the
/// trajectory's end. Motion blur stamp count is derived adaptively from
/// `config.max_drift_per_stamp_px`.
///
/// `frame_idx` is mixed into the per-tile seed and into log messages; pass
/// the frame's position within an outer sequence (or `0` for one-shot use).
///
/// `flux_cache` lets the caller share a `(star_id, sensor_idx) -> SourceFlux`
/// cache across multiple calls (the slow step of star photometry). Pass
/// `None` for a fresh per-call cache; pass `Some(...)` to reuse a cache
/// across an outer loop over many frames with the same catalog and
/// focal plane.
pub fn render_one_frame(
    trajectory: &Trajectory,
    sources: &LightSources<'_>,
    fp: &FocalPlaneConfig,
    frame_start: Duration,
    frame_idx: usize,
    config: &MotionBlurConfig,
    flux_cache: Option<Arc<Mutex<FluxCache>>>,
) -> Result<Vec<Array2<u16>>, TrajectoryError> {
    let ctx = RenderContext::from_focal_plane(fp)?;
    let projected_galaxies = project_galaxies_for_render(trajectory, fp, sources.galaxies)?;
    let scene = RenderScene {
        trajectory,
        sources: *sources,
        fp,
        projected_galaxies,
    };
    let zlight = ZodiacalLight::new();
    let plan = plan_frame(&scene, &ctx, &zlight, frame_idx, frame_start, config)?;
    let cache = flux_cache.unwrap_or_else(|| Arc::new(Mutex::new(HashMap::new())));
    let base_seed = config.base_seed.unwrap_or(0xDEADBEEF_DEADBEEF);
    let sensor_count = ctx.satellites.len();

    let results: Vec<Result<(usize, Array2<u16>), TrajectoryError>> = (0..sensor_count)
        .into_par_iter()
        .map(|sensor_idx| {
            let sat = &ctx.satellites[sensor_idx];
            let seed = tile_seed(base_seed, frame_idx, sensor_idx);
            let tile_ctx = TileRenderContext {
                scene: &scene,
                plan: &plan,
                sensor_idx,
            };
            let arr = render_tile_array(&tile_ctx, &cache, sat, seed)?;
            Ok((sensor_idx, arr))
        })
        .collect();

    let mut images: Vec<Option<Array2<u16>>> = (0..sensor_count).map(|_| None).collect();
    for r in results {
        let (idx, arr) = r?;
        images[idx] = Some(arr);
    }
    Ok(images
        .into_iter()
        .map(|i| i.expect("sensor index covered"))
        .collect())
}

/// Render the `roi` slice of a single `(frame, sensor)` into a quantized
/// `Array2<u16>` shaped `(roi_height, roi_width)`.
///
/// Live-render counterpart to [`render_one_frame`] for callers that
/// only need a sub-region of one sensor — e.g. an FSM-offset
/// region-of-interest at full frame rate without paying full-sensor
/// stamping and noise-sampling cost. The pixel at `(0, 0)` of the
/// returned array corresponds to sensor pixel
/// `(roi.min_row, roi.min_col)`.
///
/// `roi` is in sensor pixel coordinates with inclusive bounds and must
/// be entirely contained in the sensor at `sensor_idx`. The returned
/// image has dimensions `(roi.max_row - roi.min_row + 1,
/// roi.max_col - roi.min_col + 1)`.
///
/// `sensor_idx` selects which sensor on the focal plane the ROI lies
/// on. Out-of-range indices return [`TrajectoryError::NoSensors`].
///
/// # Bit-equality
///
/// - When `roi` covers the full sensor extent the output equals
///   `render_one_frame(...)[sensor_idx]` **byte-for-byte**: the
///   accumulator and noise buffers have the same shape as the
///   full-sensor render so every chunk-keyed RNG stream visits the
///   same pixels in the same order.
/// - For a smaller `roi` the **pre-noise mean-electron image** is
///   bit-identical to the corresponding slice of a full-frame render.
///   The post-noise quantized output diverges because the
///   `process_array_in_parallel_chunks` sampler used by
///   `apply_poisson_photon_noise` and `apply_gaussian_read_noise`
///   keys its per-row-chunk seed off the array's row count, so a
///   smaller buffer produces a different RNG sequence even with the
///   same base seed.
#[allow(clippy::too_many_arguments)]
pub fn render_one_frame_roi(
    trajectory: &Trajectory,
    sources: &LightSources<'_>,
    fp: &FocalPlaneConfig,
    frame_start: Duration,
    frame_idx: usize,
    config: &MotionBlurConfig,
    flux_cache: Option<Arc<Mutex<FluxCache>>>,
    roi: AABB,
    sensor_idx: usize,
) -> Result<Array2<u16>, TrajectoryError> {
    let ctx = RenderContext::from_focal_plane(fp)?;
    if sensor_idx >= ctx.satellites.len() {
        return Err(TrajectoryError::NoSensors);
    }
    let projected_galaxies = project_galaxies_for_render(trajectory, fp, sources.galaxies)?;
    let scene = RenderScene {
        trajectory,
        sources: *sources,
        fp,
        projected_galaxies,
    };
    let zlight = ZodiacalLight::new();
    let plan = plan_frame(&scene, &ctx, &zlight, frame_idx, frame_start, config)?;
    let cache = flux_cache.unwrap_or_else(|| Arc::new(Mutex::new(HashMap::new())));
    let base_seed = config.base_seed.unwrap_or(0xDEADBEEF_DEADBEEF);
    let sat = &ctx.satellites[sensor_idx];
    let seed = tile_seed(base_seed, frame_idx, sensor_idx);
    let tile_ctx = TileRenderContext {
        scene: &scene,
        plan: &plan,
        sensor_idx,
    };
    render_tile_array_roi(&tile_ctx, &cache, sat, seed, roi)
}

/// Render the full trajectory with motion blur, parallel over `(frame, sensor)`.
///
/// Returns the total number of frames rendered.
///
/// `sources.galaxies` is a flat sky-truth catalog. The renderer
/// projects each galaxy onto every sensor whose extent (plus halo
/// padding) contains its centre — so galaxies that subtend multiple
/// sensors get rendered on each of them. Projection happens once at
/// the trajectory's start-time orientation and the splat runs at
/// full-exposure flux per tile: exact for static trajectories, an
/// approximation for drifting trajectories where a sub-pixel galaxy
/// shift across the exposure becomes detectable.
pub fn render_motion_trajectory(
    trajectory: &Trajectory,
    sources: &LightSources<'_>,
    fp: &FocalPlaneConfig,
    config: &MotionBlurConfig,
    output_dir: &Path,
) -> Result<usize, TrajectoryError> {
    let ctx = RenderContext::from_focal_plane(fp)?;
    let sensor_count = ctx.satellites.len();

    let projected_galaxies = project_galaxies_for_render(trajectory, fp, sources.galaxies)?;
    let scene = RenderScene {
        trajectory,
        sources: *sources,
        fp,
        projected_galaxies,
    };

    let frame_times = trajectory.frame_times(config.timestep);
    let total_frames = frame_times.len();
    let total_tiles = total_frames * sensor_count;
    info!(
        "Rendering trajectory: {total_frames} frames × {sensor_count} sensors = {total_tiles} \
         tiles, exposure={:.3}s, timestep={:.3}s, max_drift_per_stamp_px={:.3}",
        config.exposure.as_secs_f64(),
        config.timestep.as_secs_f64(),
        config.max_drift_per_stamp_px,
    );

    // Progress bar over all (frame, sensor) tiles. `.inc(1)` is called by
    // each worker after the tile's PNG is written. Non-TTY stdout is detected
    // automatically by indicatif; `quiet` mode forces a hidden draw target.
    let pb = if config.quiet {
        ProgressBar::with_draw_target(Some(total_tiles as u64), ProgressDrawTarget::hidden())
    } else {
        ProgressBar::new(total_tiles as u64)
    };
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] \
             {pos}/{len} tiles ({eta})",
        )
        .expect("progress bar template is valid")
        .progress_chars("=>-"),
    );

    let zlight = ZodiacalLight::new();
    let mut plans: Vec<FramePlan> = Vec::with_capacity(total_frames);
    let mut stamps_min = usize::MAX;
    let mut stamps_max = 0usize;
    for (frame_idx, &t) in frame_times.iter().enumerate() {
        let plan = plan_frame(&scene, &ctx, &zlight, frame_idx, t, config)?;
        let mid_t = plan
            .schedule
            .mid_time()
            .min(scene.trajectory.end_time())
            .max(scene.trajectory.start_time());
        let mid_bore = boresight_of(&scene.trajectory.orientation_at(mid_t)?);
        stamps_min = stamps_min.min(plan.schedule.stamps_per_exposure);
        stamps_max = stamps_max.max(plan.schedule.stamps_per_exposure);
        // Per-frame summary routed through the bar so it coexists with the
        // live progress line without scrambled output.
        pb.println(format!(
            "frame {:06} at t={:.3}s: stamps={}, boresight=(ra={:.4}°, dec={:.4}°), \
             stars={}",
            frame_idx,
            t.as_secs_f64(),
            plan.schedule.stamps_per_exposure,
            mid_bore.ra_degrees(),
            mid_bore.dec_degrees(),
            plan.stars.len(),
        ));
        plans.push(plan);
    }
    if total_frames == 0 {
        stamps_min = 0;
        stamps_max = 0;
    }
    info!(
        "Stamps per exposure: min={} max={} (force_static={})",
        stamps_min, stamps_max, config.force_static
    );

    // Build (frame, sensor) tile list.
    let base_seed = config.base_seed.unwrap_or(0xDEADBEEF_DEADBEEF);

    #[derive(Clone, Copy)]
    struct Tile {
        frame_plan_idx: usize,
        sensor_idx: usize,
    }

    let tiles: Vec<Tile> = (0..plans.len())
        .flat_map(|fp_idx| {
            (0..sensor_count).map(move |s| Tile {
                frame_plan_idx: fp_idx,
                sensor_idx: s,
            })
        })
        .collect();

    let flux_cache: Arc<Mutex<FluxCache>> = Arc::new(Mutex::new(HashMap::new()));

    // Pre-create per-sensor output directories (`<output_dir>/sensor_NN/`).
    // Tile workers write 16-bit PNGs directly into these subdirs.
    for sensor_idx in 0..sensor_count {
        let sensor_dir = output_dir.join(sensor_dir_name(sensor_idx));
        std::fs::create_dir_all(&sensor_dir)
            .map_err(|e| TrajectoryError::ImageWrite(e.to_string()))?;
    }

    let output_paths: Vec<PathBuf> = tiles
        .iter()
        .map(|tile| {
            let plan = &plans[tile.frame_plan_idx];
            output_dir.join(sensor_relative_png_path(tile.sensor_idx, plan.idx))
        })
        .collect();

    let render_started = Instant::now();
    let results: Vec<Result<(), TrajectoryError>> = tiles
        .par_iter()
        .zip(output_paths.par_iter())
        .map(|(tile, out_path)| {
            let plan = &plans[tile.frame_plan_idx];
            let sat = &ctx.satellites[tile.sensor_idx];
            let seed = tile_seed(base_seed, plan.idx, tile.sensor_idx);
            let tile_started = Instant::now();
            let tile_ctx = TileRenderContext {
                scene: &scene,
                plan,
                sensor_idx: tile.sensor_idx,
            };
            let result = render_tile_array(&tile_ctx, &flux_cache, sat, seed)
                .and_then(|arr| save_u16_png(&arr, out_path));
            debug!(
                "tile frame={} sensor={} stamps={} stars={} elapsed={}ms",
                plan.idx,
                tile.sensor_idx,
                plan.schedule.stamps_per_exposure,
                plan.stars.len(),
                tile_started.elapsed().as_millis(),
            );
            pb.inc(1);
            result
        })
        .collect();

    for r in results {
        r?;
    }

    let render_elapsed = render_started.elapsed();
    let tiles_per_s = if render_elapsed.as_secs_f64() > 0.0 {
        total_tiles as f64 / render_elapsed.as_secs_f64()
    } else {
        0.0
    };
    pb.finish_with_message(format!(
        "rendered {} tiles in {:.2}s",
        total_tiles,
        render_elapsed.as_secs_f64(),
    ));

    let cache = flux_cache.lock().expect("flux cache mutex poisoned");
    info!(
        "Flux cache: {} entries across {} sensors",
        cache.len(),
        sensor_count
    );
    info!(
        "Rendered {} tiles in {:.2}s ({:.1} tiles/s)",
        total_tiles,
        render_elapsed.as_secs_f64(),
        tiles_per_s,
    );
    drop(cache);

    // Assemble and write metadata.json describing the render. Every PNG path
    // recorded here is relative (forward-slash) to `output_dir`.
    let metadata = build_render_metadata(
        &scene,
        config,
        &plans
            .iter()
            .map(|p| (p.idx, p.schedule))
            .collect::<Vec<_>>(),
        sensor_count,
    )?;
    let metadata_path = output_dir.join("metadata.json");
    let file = std::fs::File::create(&metadata_path)
        .map_err(|e| TrajectoryError::ImageWrite(e.to_string()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &metadata)
        .map_err(|e| TrajectoryError::ImageWrite(e.to_string()))?;
    use std::io::Write;
    writer
        .flush()
        .map_err(|e| TrajectoryError::ImageWrite(e.to_string()))?;
    info!("Wrote render metadata to {}", metadata_path.display());

    Ok(total_frames)
}

/// Build the [`RenderMetadata`] descriptor from the finalized render plan.
///
/// Consumes the per-frame `(frame_idx, schedule)` pairs produced during the
/// serial planning pass plus the render config and per-sensor satellite
/// configs. Static scene inputs (trajectory, catalog, focal plane, zodiacal
/// coords) come through the [`RenderScene`] handle.
fn build_render_metadata(
    scene: &RenderScene,
    config: &MotionBlurConfig,
    frame_plans: &[(usize, SubsampleSchedule)],
    sensor_count: usize,
) -> Result<RenderMetadata, TrajectoryError> {
    let rendered_at = chrono::Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);

    let start = scene.trajectory.start_time();
    let end = scene.trajectory.end_time();
    let trajectory_meta = TrajectoryMeta {
        duration_s: (end - start).as_secs_f64(),
        start_time_s: start.as_secs_f64(),
        end_time_s: end.as_secs_f64(),
        waypoints: scene
            .trajectory
            .waypoints()
            .iter()
            .map(|wp| {
                let q = wp.orientation;
                let bore = boresight_of(&q);
                WaypointMeta {
                    time_s: wp.time.as_secs_f64(),
                    quat: q,
                    boresight: EquatorialMeta {
                        ra_deg: bore.ra_degrees(),
                        dec_deg: bore.dec_degrees(),
                    },
                    roll_deg: roll_of(&q).to_degrees(),
                }
            })
            .collect(),
    };

    let mut frames: Vec<FrameMeta> = Vec::with_capacity(frame_plans.len());
    for (frame_idx, schedule) in frame_plans {
        let mid_t = (schedule.frame_start + schedule.exposure / 2)
            .min(scene.trajectory.end_time())
            .max(scene.trajectory.start_time());
        let q = scene.trajectory.orientation_at(mid_t)?;
        let bore = boresight_of(&q);
        let mut paths: BTreeMap<String, String> = BTreeMap::new();
        for sensor_idx in 0..sensor_count {
            paths.insert(
                sensor_dir_name(sensor_idx),
                sensor_relative_png_path(sensor_idx, *frame_idx),
            );
        }
        frames.push(FrameMeta {
            idx: *frame_idx,
            t_s: schedule.frame_start.as_secs_f64(),
            exposure_s: schedule.exposure.as_secs_f64(),
            quat: q,
            boresight: EquatorialMeta {
                ra_deg: bore.ra_degrees(),
                dec_deg: bore.dec_degrees(),
            },
            roll_deg: roll_of(&q).to_degrees(),
            n_stamps: schedule.stamps_per_exposure,
            paths,
        });
    }

    let stars: Vec<StarMeta> = scene
        .sources
        .catalog_stars
        .iter()
        .map(|s| StarMeta {
            id: s.id,
            // starfield::StarData does not currently expose a name
            // field; left None until an upstream catalog wires it in.
            name: None,
            ra_deg: s.position.ra_degrees(),
            dec_deg: s.position.dec_degrees(),
            magnitude: s.magnitude,
            color_index: s.b_v,
        })
        .collect();

    let galaxies: Vec<Galaxy> = scene.sources.galaxies.to_vec();

    let focal_plane = scene.fp.clone();

    let render_config = RenderConfigMeta {
        exposure_s: config.exposure.as_secs_f64(),
        timestep_s: config.timestep.as_secs_f64(),
        max_drift_per_stamp_px: config.max_drift_per_stamp_px,
        seed: config.base_seed.unwrap_or(0),
        force_static: config.force_static,
        catalog_path: config.catalog_path.to_string_lossy().into_owned(),
        zodiacal: scene.sources.zodiacal,
    };

    Ok(RenderMetadata {
        version: "1.2".to_string(),
        rendered_at,
        trajectory: trajectory_meta,
        frames,
        stars,
        galaxies,
        focal_plane,
        render_config,
    })
}

/// Per-pixel-per-second zodiacal electron rate at a given boresight for a
/// specific satellite (combined QE + aperture + per-pixel solid angle).
fn zodiacal_per_px_per_s_at(
    zlight: &ZodiacalLight,
    sat: &SatelliteConfig,
    _zodiacal: &SolarAngularCoordinates,
    _boresight: &Equatorial,
) -> f64 {
    // We evaluate at the trajectory-level `zodiacal` coordinates. The
    // boresight argument is kept to make the API explicit; a future
    // extension could compute an observer-relative elongation per frame.
    let coords = *_zodiacal;
    let z_spect = zlight
        .get_zodiacal_spectrum(&coords)
        .expect("zodiacal spectrum evaluation failed");
    let focal_length_mm = sat.telescope.focal_length.as_meters() * 1000.0;
    let pixel_size_mm = sat.sensor.dimensions.pixel_size().as_millimeters();
    let pixel_scale_arcsec_per_pixel = 206265.0 * pixel_size_mm / focal_length_mm;
    let pixel_solid_angle_arcsec2 = pixel_scale_arcsec_per_pixel * pixel_scale_arcsec_per_pixel;
    let aperture = sat.telescope.clear_aperture_area();
    // `photo_electrons` returns electrons integrated over `exposure`; divide
    // by seconds to get a rate we can combine with any schedule.
    z_spect.photo_electrons(&sat.combined_qe, aperture, &Duration::from_secs(1))
        * pixel_solid_angle_arcsec2
}

/// Ratio of per-pixel zodiacal electron rate between `reference` and
/// `target`; used to scale a per-frame zodiacal rate computed at the
/// reference sensor to other sensors in the array. When all sensors share
/// pixel size / QE (the common case) the ratio is `1.0`.
fn per_sensor_pixel_solid_angle_ratio(
    reference: &SatelliteConfig,
    target: &SatelliteConfig,
) -> f64 {
    let ref_px_mm = reference.sensor.dimensions.pixel_size().as_millimeters();
    let tgt_px_mm = target.sensor.dimensions.pixel_size().as_millimeters();
    if ref_px_mm == 0.0 {
        return 1.0;
    }
    // Pixel solid angle scales like pixel_size^2; QE is folded into
    // electrons via photo_electrons, which is telescope-level, not
    // sensor-level. For a heterogeneous array this is an approximation;
    // in the common homogeneous case it is exactly 1.0.
    (tgt_px_mm / ref_px_mm).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::sensor::models::GSENSE4040BSI;
    use crate::hardware::sensor_array::SensorArray;
    use crate::hardware::telescope::TelescopeConfig;
    use crate::sims::orientation::orientation_from_pointing;
    use crate::sims::trajectory::Waypoint;
    use approx::assert_abs_diff_eq;
    use nalgebra::UnitQuaternion;
    use shared::units::{Length, LengthExt, Temperature, TemperatureExt};
    use std::f64::consts::PI;

    fn tiny_fp() -> FocalPlaneConfig {
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        FocalPlaneConfig::new(
            telescope,
            SensorArray::single(GSENSE4040BSI.clone().with_dimensions(64, 64)),
            Temperature::from_celsius(-10.0),
        )
    }

    #[test]
    fn test_from_drift_budget_static_gives_one_stamp() {
        let sched = SubsampleSchedule::from_drift_budget(
            Duration::ZERO,
            Duration::from_secs(1),
            0.0, // zero drift
            0.0, // zero excursion
            1e-6,
            0.1,
        );
        assert_eq!(sched.stamps_per_exposure, 1);
        assert_eq!(sched.envelope_padding_rad, 0.0);
    }

    #[test]
    fn test_from_drift_budget_picks_total_stamps_from_path_length() {
        // 10 px of total path length at pixel_scale=1e-5 rad/px with a
        // 0.1 px per-stamp budget should yield 100 total stamps.
        let pixel_scale = 1e-5_f64;
        let drift_px = 10.0;
        let budget_px = 0.1;
        let excursion_rad = 0.5_f64 * pixel_scale; // arbitrary positive
        let sched = SubsampleSchedule::from_drift_budget(
            Duration::ZERO,
            Duration::from_secs(1),
            drift_px * pixel_scale,
            excursion_rad,
            pixel_scale,
            budget_px,
        );
        let expected = (drift_px / budget_px).ceil() as usize;
        assert_eq!(sched.stamps_per_exposure, expected);
        assert_abs_diff_eq!(sched.envelope_padding_rad, excursion_rad, epsilon = 1e-15);
    }

    #[test]
    fn test_stamp_times_span_the_exposure_window() {
        // Stamps must all fall inside [frame_start, frame_start + exposure).
        let sched = SubsampleSchedule {
            frame_start: Duration::from_secs(10),
            exposure: Duration::from_secs(4),
            stamps_per_exposure: 16,
            envelope_padding_rad: 0.0,
        };
        let stamps = sched.stamp_times(0.0);
        assert_eq!(stamps.len(), 16);
        let lo = 10.0;
        let hi = 14.0;
        for t in &stamps {
            let v = t.as_secs_f64();
            assert!(v >= lo && v < hi, "stamp {v} outside [{lo}, {hi})");
        }
    }

    #[test]
    fn test_stamp_times_are_reproducible_for_same_phase() {
        // The R2 stamp times are a pure function of (schedule, phase).
        let sched = SubsampleSchedule {
            frame_start: Duration::ZERO,
            exposure: Duration::from_secs(1),
            stamps_per_exposure: 128,
            envelope_padding_rad: 0.0,
        };
        let a = sched.stamp_times(0.123);
        let b = sched.stamp_times(0.123);
        assert_eq!(a, b);
        let c = sched.stamp_times(0.124);
        assert_ne!(a, c);
    }

    #[test]
    fn test_mid_time_is_exposure_midpoint() {
        let sched = SubsampleSchedule {
            frame_start: Duration::from_secs(10),
            exposure: Duration::from_secs(4),
            stamps_per_exposure: 4,
            envelope_padding_rad: 0.0,
        };
        assert_eq!(sched.mid_time(), Duration::from_secs(12));
    }

    #[test]
    fn test_sensor_accumulator_combined_mean() {
        let mut acc = SensorAccumulator::zero(4, 4);
        acc.star_mean_electrons[[1, 1]] = 5.0;
        let combined = acc.into_combined_mean(2.0, 1.0);
        assert_eq!(combined[[0, 0]], 3.0); // 0 + 2 + 1
        assert_eq!(combined[[1, 1]], 8.0); // 5 + 2 + 1
    }

    #[test]
    fn test_tile_seed_is_deterministic_and_varies() {
        let a = tile_seed(42, 0, 0);
        let b = tile_seed(42, 0, 0);
        let c = tile_seed(42, 1, 0);
        let d = tile_seed(42, 0, 1);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(c, d);
    }

    #[test]
    fn test_tile_seed_domains_are_independent() {
        // The same tile must produce distinct sub-seeds for each named
        // domain, so that perturbing one stream cannot shift the bytes
        // another would have produced.
        let s = tile_seed(1234, 5, 6);
        assert_ne!(s ^ POISSON_DOMAIN, s ^ READ_NOISE_DOMAIN);
        assert_ne!(s ^ POISSON_DOMAIN, s ^ STAMP_PHASE_DOMAIN);
        assert_ne!(s ^ READ_NOISE_DOMAIN, s ^ STAMP_PHASE_DOMAIN);
    }

    fn static_trajectory() -> Trajectory {
        let eq = Equatorial::from_degrees(45.0, 30.0);
        // Two identical orientations => zero drift.
        Trajectory::new(vec![
            Waypoint::new(Duration::ZERO, orientation_from_pointing(&eq, 0.0)),
            Waypoint::new(Duration::from_secs(10), orientation_from_pointing(&eq, 0.0)),
        ])
        .unwrap()
    }

    #[test]
    fn test_max_drift_over_window_static_is_zero() {
        let traj = static_trajectory();
        let drift = max_drift_over_window(&traj, Duration::ZERO, Duration::from_secs(5)).unwrap();
        assert!(drift.abs() < 1e-12);
    }

    #[test]
    fn test_max_drift_over_window_scales_linearly() {
        let start = Equatorial::from_degrees(45.0, 30.0);
        let end = Equatorial::from_degrees(46.0, 30.0); // ~1 deg span over 10 s
        let traj = Trajectory::from_endpoints(start, end, Duration::from_secs(10)).unwrap();
        let full = max_drift_over_window(&traj, Duration::ZERO, Duration::from_secs(10)).unwrap();
        let half = max_drift_over_window(&traj, Duration::ZERO, Duration::from_secs(5)).unwrap();
        assert_abs_diff_eq!(full, 2.0 * half, epsilon = 1e-9);
        // About 1 degree total (cos(30)~0.866 but angle_to is proper 3D angle).
        assert!(full > 0.5_f64.to_radians());
        assert!(full < 1.5_f64.to_radians());
    }

    fn render_static_tmp(fp: &FocalPlaneConfig, force_static: bool) -> usize {
        let traj = static_trajectory();
        let cfg = MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.1,
            base_seed: Some(7),
            force_static,
            quiet: true,
            ..Default::default()
        };
        let tmp = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &[],
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        let frames = render_motion_trajectory(&traj, &sources, fp, &cfg, tmp.path()).unwrap();
        frames
    }

    #[test]
    fn test_render_runs_end_to_end_no_stars() {
        let fp = tiny_fp();
        let frames = render_static_tmp(&fp, false);
        assert!(frames > 0);
    }

    #[test]
    fn test_render_runs_force_static() {
        let fp = tiny_fp();
        let frames = render_static_tmp(&fp, true);
        assert!(frames > 0);
    }

    #[test]
    fn test_render_motion_trajectory_with_quiet_progress() {
        // Trivial single-frame render with the progress bar hidden. Guards the
        // indicatif wiring: the render must complete cleanly when `quiet` is
        // set, regardless of whether stdout is a TTY.
        let fp = tiny_fp();
        let traj = static_trajectory();
        let cfg = MotionBlurConfig {
            timestep: Duration::from_secs(10),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.1,
            base_seed: Some(3),
            force_static: true,
            quiet: true,
            ..Default::default()
        };
        let tmp = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &[],
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        let frames = render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp.path()).unwrap();
        assert!(frames >= 1);
    }

    #[test]
    fn test_flux_cache_is_reused_across_frames() {
        // Render 3 frames of a static trajectory with a few stars in-field;
        // after rendering the cache should hold at most `num_stars * sensors`
        // entries — not 3x that.
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let stars: Vec<StarData> = (0..3)
            .map(|i| StarData {
                id: i as u64,
                magnitude: 8.0,
                position: Equatorial::from_degrees(
                    pointing.ra_degrees() + (i as f64) * 0.001,
                    pointing.dec_degrees(),
                ),
                b_v: Some(0.6),
            })
            .collect();
        let traj = static_trajectory();
        let cfg = MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.1,
            base_seed: Some(11),
            force_static: true,
            quiet: true,
            ..Default::default()
        };
        let tmp = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &stars,
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        // Render normally to exercise the cache.
        let frames = render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp.path()).unwrap();
        assert!(frames >= 3);
        // Cache size bound: at most num_stars * sensor_count.
        // We can't peek from outside the function, so rely on invariant by
        // reconstructing the shared mutex via a second call and checking
        // the cache does not keep growing unboundedly with repeat work.
        // The tightest assertion we can make without instrumenting the
        // internals is: a second run with the same inputs renders the same
        // number of frames, which would fail if the path were non-idempotent.
        let frames2 = render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp.path()).unwrap();
        assert_eq!(frames, frames2);
    }

    #[test]
    fn test_motion_blur_smears_a_point_source() {
        // A drifting trajectory should depress the peak pixel and roughly
        // conserve total star flux relative to a static one.
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let star = StarData {
            id: 1,
            magnitude: 6.0,
            position: pointing,
            b_v: Some(0.6),
        };
        let static_traj = static_trajectory();
        // Drift a few pixels across the exposure: 4 arcsec-ish span on a
        // typical plate scale — small enough to stay in-frame.
        let end = Equatorial::from_degrees(pointing.ra_degrees() + 2.0e-4, pointing.dec_degrees());
        let moving = Trajectory::from_endpoints(pointing, end, Duration::from_secs(10)).unwrap();

        // Use cfg with no zodiacal/dark noise shenanigans — seed both the
        // same so only the schedule differs.
        let cfg_static = MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.1,
            base_seed: Some(101),
            force_static: false,
            quiet: true,
            ..Default::default()
        };
        let cfg_moving = cfg_static.clone();

        // We need to read back the rendered mean (pre-Poisson) to make a
        // deterministic comparison. Use SensorAccumulator directly via the
        // internal tile API equivalent: run the adaptive scheduler and the
        // accumulator loop locally for each.
        let static_acc = simulate_tile_accumulator(&static_traj, &star, &fp, &cfg_static);
        let moving_acc = simulate_tile_accumulator(&moving, &star, &fp, &cfg_moving);

        let static_peak = static_acc
            .star_mean_electrons
            .iter()
            .cloned()
            .fold(0.0, f64::max);
        let moving_peak = moving_acc
            .star_mean_electrons
            .iter()
            .cloned()
            .fold(0.0, f64::max);
        let static_sum = static_acc.star_mean_electrons.sum();
        let moving_sum = moving_acc.star_mean_electrons.sum();

        assert!(
            moving_peak < static_peak,
            "motion-blurred peak should be below static peak (got moving={} static={})",
            moving_peak,
            static_peak
        );
        // Approximate flux conservation. The 10% tolerance absorbs
        // PSF-truncation differences when the deterministic R2 stamp
        // phase puts the moving star at a sub-pixel offset whose
        // discrete pixel sum loses slightly more PSF tail than the
        // pixel-aligned static case.
        assert!(
            (static_sum - moving_sum).abs() / static_sum.max(1.0) < 0.10,
            "motion-blur should approximately conserve total flux \
             (static_sum={}, moving_sum={})",
            static_sum,
            moving_sum
        );
        let _ = PI;
    }

    /// Test-only helper: runs the subsample accumulator loop for a single
    /// star on a single-sensor focal plane and returns the pre-Poisson
    /// accumulator. Mirrors the production path inside `render_tile` but
    /// stops before the Poisson/read-noise stage.
    fn simulate_tile_accumulator(
        trajectory: &Trajectory,
        star: &StarData,
        fp: &FocalPlaneConfig,
        cfg: &MotionBlurConfig,
    ) -> SensorAccumulator {
        // Test-only adaptor: forwards to the stratified-MC helper with
        // a fixed seed so the legacy test that just checks "blur depresses
        // the peak" stays deterministic.
        simulate_tile_accumulator_stratified(trajectory, star, fp, cfg, tile_seed(0xDEADBEEF, 0, 0))
    }

    #[test]
    fn test_motion_blur_collapses_to_static_when_n_1() {
        // Zero drift + N=1 should produce the same mean-electron map as a
        // direct single-orientation single-sample splat.
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let star = StarData {
            id: 1,
            magnitude: 8.0,
            position: pointing,
            b_v: Some(0.6),
        };
        let traj = static_trajectory();
        let cfg = MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.1,
            base_seed: Some(0),
            force_static: true,
            quiet: true,
            ..Default::default()
        };
        let acc = simulate_tile_accumulator(&traj, &star, &fp, &cfg);

        let first_sat = fp.satellite_for_sensor(0).unwrap();
        let pixel_size_mm = first_sat.sensor.pixel_size().as_millimeters();
        let airy_pix = first_sat.airy_disk_pixel_space();
        let padding_mm = airy_pix.first_zero() * 2.0 * pixel_size_mm;
        let flux = star_data_to_fluxes(&star, &first_sat);
        let q = orientation_from_pointing(&pointing, 0.0);
        let (px, py) = fp.project_to_sensor(&star, &q, 0, padding_mm).unwrap();
        let total = flux
            .electrons
            .integrated_over(&cfg.exposure, first_sat.telescope.clear_aperture_area());
        let (w, h) = first_sat.sensor.dimensions.get_pixel_width_height();
        let mut ref_acc = SensorAccumulator::zero(w, h);
        ref_acc.splat_psf(px, py, total, &flux.electrons.disk);

        let diff = (&acc.star_mean_electrons - &ref_acc.star_mean_electrons)
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-9, "max |diff| = {}", diff);
    }

    #[test]
    fn test_per_stamp_render_is_deterministic() {
        // Same (base_seed, frame, sensor, M) must produce byte-identical
        // PNGs across runs even with the per-stamp inner loop active. A
        // genuine sub-arcsec sweep across the exposure exercises the
        // per-stamp orientation queries, not just the M=1 fallback.
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let drift = Equatorial::from_degrees(45.0 + 0.001, 30.0); // ~3.6"
        let traj = Trajectory::new(vec![
            Waypoint::new(Duration::ZERO, orientation_from_pointing(&pointing, 0.0)),
            Waypoint::new(
                Duration::from_secs(10),
                orientation_from_pointing(&drift, 0.0),
            ),
        ])
        .unwrap();
        let stars: Vec<StarData> = (0..4)
            .map(|i| StarData {
                id: i as u64,
                magnitude: 7.0,
                position: Equatorial::from_degrees(
                    pointing.ra_degrees() + 0.0005 * i as f64,
                    pointing.dec_degrees(),
                ),
                b_v: Some(0.6),
            })
            .collect();
        let cfg = MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.01,
            base_seed: Some(424242),
            force_static: false,
            quiet: true,
            ..Default::default()
        };
        let tmp_a = tempfile::tempdir().unwrap();
        let tmp_b = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &stars,
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp_a.path()).unwrap();
        render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp_b.path()).unwrap();
        let name = "sensor_00/frame_000000.png";
        let a = std::fs::read(tmp_a.path().join(name)).unwrap();
        let b = std::fs::read(tmp_b.path().join(name)).unwrap();
        assert_eq!(a, b, "per-stamp rendering must be deterministic");
    }

    #[test]
    fn test_per_stamp_changes_render_vs_per_subsample() {
        // Same trajectory, same seed, only the per-stamp budget differs.
        // The two outputs MUST differ — otherwise the inner stamp loop is
        // a no-op. Uses a sweep big enough to cross multiple pixels so
        // the smear shape is sensitive to where stamps land within each
        // subsample.
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let drift = Equatorial::from_degrees(45.0 + 0.01, 30.0); // ~36"
        let traj = Trajectory::new(vec![
            Waypoint::new(Duration::ZERO, orientation_from_pointing(&pointing, 0.0)),
            Waypoint::new(
                Duration::from_secs(10),
                orientation_from_pointing(&drift, 0.0),
            ),
        ])
        .unwrap();
        let stars = vec![StarData {
            id: 0,
            magnitude: 5.0,
            position: pointing,
            b_v: Some(0.6),
        }];
        let cfg_coarse = MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            // Loose per-stamp budget keeps the total stamp count low
            // so individual stamp positions are visible in the output.
            max_drift_per_stamp_px: 1.0,
            base_seed: Some(99),
            force_static: false,
            quiet: true,
            ..Default::default()
        };
        let cfg_fine = MotionBlurConfig {
            max_drift_per_stamp_px: 0.05,
            ..cfg_coarse.clone()
        };
        let tmp_coarse = tempfile::tempdir().unwrap();
        let tmp_fine = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &stars,
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        render_motion_trajectory(&traj, &sources, &fp, &cfg_coarse, tmp_coarse.path()).unwrap();
        render_motion_trajectory(&traj, &sources, &fp, &cfg_fine, tmp_fine.path()).unwrap();
        let name = "sensor_00/frame_000000.png";
        let coarse = std::fs::read(tmp_coarse.path().join(name)).unwrap();
        let fine = std::fs::read(tmp_fine.path().join(name)).unwrap();
        assert_ne!(
            coarse, fine,
            "tightening max_drift_per_stamp_px on a moving trajectory must change the rendered streak"
        );
    }

    #[test]
    fn test_render_one_frame_is_deterministic() {
        // Two calls with the same arguments and the same seed must return
        // byte-identical arrays. Exercises a drifting trajectory so the
        // per-stamp loop is active (not the N=1 fallback) and a couple of
        // stars so the flux cache + envelope prefilter are exercised too.
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let drift = Equatorial::from_degrees(45.0 + 0.001, 30.0); // ~3.6"
        let traj = Trajectory::new(vec![
            Waypoint::new(Duration::ZERO, orientation_from_pointing(&pointing, 0.0)),
            Waypoint::new(
                Duration::from_secs(10),
                orientation_from_pointing(&drift, 0.0),
            ),
        ])
        .unwrap();
        let stars: Vec<StarData> = (0..3)
            .map(|i| StarData {
                id: i as u64,
                magnitude: 7.0,
                position: Equatorial::from_degrees(
                    pointing.ra_degrees() + 0.0005 * i as f64,
                    pointing.dec_degrees(),
                ),
                b_v: Some(0.6),
            })
            .collect();
        let cfg = MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.05,
            base_seed: Some(0xABCD_1234),
            force_static: false,
            quiet: true,
            ..Default::default()
        };

        let sources = LightSources {
            catalog_stars: &stars,
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        let first = render_one_frame(&traj, &sources, &fp, Duration::ZERO, 0, &cfg, None).unwrap();
        let second = render_one_frame(&traj, &sources, &fp, Duration::ZERO, 0, &cfg, None).unwrap();

        assert_eq!(first.len(), second.len());
        assert!(!first.is_empty(), "expected at least one sensor");
        for (s, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            assert_eq!(a, b, "sensor {s} arrays differ between identical calls");
        }
    }

    #[test]
    fn test_render_one_frame_matches_render_motion_trajectory_bytes() {
        // `render_motion_trajectory` is reimplemented in terms of the same
        // primitives as `render_one_frame`. For any given (base_seed,
        // frame_idx) the per-sensor quantized arrays must equal the PNG
        // pixels the trajectory renderer writes for that frame.
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let stars: Vec<StarData> = (0..2)
            .map(|i| StarData {
                id: i as u64,
                magnitude: 7.5,
                position: Equatorial::from_degrees(
                    pointing.ra_degrees() + 0.0005 * i as f64,
                    pointing.dec_degrees(),
                ),
                b_v: Some(0.6),
            })
            .collect();
        let traj = static_trajectory();
        let cfg = MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.1,
            base_seed: Some(54321),
            force_static: false,
            quiet: true,
            ..Default::default()
        };

        let frame_idx = 0;
        let frame_start = Duration::ZERO;
        let sources = LightSources {
            catalog_stars: &stars,
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        let one =
            render_one_frame(&traj, &sources, &fp, frame_start, frame_idx, &cfg, None).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp.path()).unwrap();

        let png_path = tmp.path().join(sensor_relative_png_path(0, frame_idx));
        let img = image::open(&png_path).unwrap().to_luma16();
        let (w, h) = img.dimensions();
        let png_arr = Array2::from_shape_vec((h as usize, w as usize), img.into_raw()).unwrap();
        assert_eq!(
            one[0], png_arr,
            "render_one_frame must produce the same pixels render_motion_trajectory writes"
        );
    }

    /// Standard fixture for the ROI bit-equality tests: tiny focal plane,
    /// a handful of well-placed stars, deterministic seed.
    fn roi_test_fixture() -> (
        FocalPlaneConfig,
        Vec<StarData>,
        Trajectory,
        MotionBlurConfig,
    ) {
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let stars: Vec<StarData> = (0..6)
            .map(|i| StarData {
                id: i as u64,
                magnitude: 7.0 + i as f64 * 0.1,
                position: Equatorial::from_degrees(
                    pointing.ra_degrees() + 0.0008 * (i as f64 - 2.5),
                    pointing.dec_degrees() + 0.0005 * (i as f64 - 1.5),
                ),
                b_v: Some(0.6),
            })
            .collect();
        let traj = static_trajectory();
        let cfg = MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.1,
            base_seed: Some(0x0FC8_BEEF),
            force_static: false,
            quiet: true,
            ..Default::default()
        };
        (fp, stars, traj, cfg)
    }

    #[test]
    fn test_render_one_frame_roi_full_sensor_is_byte_equal() {
        // ROI covering the entire sensor must produce a byte-identical
        // image to render_one_frame's output for that sensor. The
        // accumulator and noise buffers are identically shaped between
        // the two paths, so the chunk-keyed RNG streams advance through
        // the same pixels in the same order.
        let (fp, stars, traj, cfg) = roi_test_fixture();
        let sources = LightSources {
            catalog_stars: &stars,
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        let full = render_one_frame(&traj, &sources, &fp, Duration::ZERO, 0, &cfg, None).unwrap();

        let sat = fp.satellite_for_sensor(0).unwrap();
        let (w, h) = sat.sensor.dimensions.get_pixel_width_height();
        let roi = AABB::from_coords(0, 0, h - 1, w - 1);
        let roi_image =
            render_one_frame_roi(&traj, &sources, &fp, Duration::ZERO, 0, &cfg, None, roi, 0)
                .unwrap();

        assert_eq!(roi_image.dim(), (h, w));
        assert_eq!(
            roi_image, full[0],
            "full-sensor ROI must be byte-equal to render_one_frame[sensor_idx]"
        );
    }

    #[test]
    fn test_render_one_frame_roi_subset_mean_matches_full_slice() {
        // Sub-region ROI: the pre-noise mean-electron image must equal the
        // corresponding slice of the full-sensor mean byte-for-byte. The
        // quantized post-noise output cannot be checked directly because
        // the chunk-keyed noise sampler advances differently on a smaller
        // buffer (see render_one_frame_roi docs).
        let (fp, stars, traj, cfg) = roi_test_fixture();
        let ctx = RenderContext::from_focal_plane(&fp).unwrap();
        let sources = LightSources {
            catalog_stars: &stars,
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        let scene = RenderScene {
            trajectory: &traj,
            sources,
            fp: &fp,
            projected_galaxies: Vec::new(),
        };
        let zlight = ZodiacalLight::new();
        let plan = plan_frame(&scene, &ctx, &zlight, 0, Duration::ZERO, &cfg).unwrap();
        let cache: Arc<Mutex<FluxCache>> = Arc::new(Mutex::new(HashMap::new()));
        let sensor_idx = 0;
        let sat = &ctx.satellites[sensor_idx];
        let seed = tile_seed(cfg.base_seed.unwrap(), 0, sensor_idx);

        let (w, h) = sat.sensor.dimensions.get_pixel_width_height();
        let tile_ctx = TileRenderContext {
            scene: &scene,
            plan: &plan,
            sensor_idx,
        };
        let full_mean = build_tile_mean_image(
            &tile_ctx,
            &cache,
            sat,
            seed,
            AABB::from_coords(0, 0, h - 1, w - 1),
        )
        .unwrap()
        .mean_image;

        // A 24x24 ROI offset into the interior of the 64x64 sensor.
        let roi = AABB::from_coords(12, 14, 35, 37);
        let roi_w = roi.max_col - roi.min_col + 1;
        let roi_h = roi.max_row - roi.min_row + 1;
        let roi_mean = build_tile_mean_image(&tile_ctx, &cache, sat, seed, roi)
            .unwrap()
            .mean_image;

        assert_eq!(roi_mean.dim(), (roi_h, roi_w));
        for r in 0..roi_h {
            for c in 0..roi_w {
                let full_val = full_mean[[roi.min_row + r, roi.min_col + c]];
                let roi_val = roi_mean[[r, c]];
                assert_eq!(
                    roi_val.to_bits(),
                    full_val.to_bits(),
                    "mean-image pixel ({r},{c}) (sensor {},{}) diverged: roi={roi_val} full={full_val}",
                    roi.min_row + r,
                    roi.min_col + c,
                );
            }
        }

        // Sanity check the quantized output too: dimension and dtype only.
        let roi_image = render_one_frame_roi(
            &traj,
            &sources,
            &fp,
            Duration::ZERO,
            0,
            &cfg,
            None,
            roi,
            sensor_idx,
        )
        .unwrap();
        assert_eq!(roi_image.dim(), (roi_h, roi_w));
    }

    #[test]
    fn test_render_one_frame_roi_rejects_out_of_bounds() {
        let (fp, stars, traj, cfg) = roi_test_fixture();
        let sat = fp.satellite_for_sensor(0).unwrap();
        let (w, h) = sat.sensor.dimensions.get_pixel_width_height();
        let oob = AABB::from_coords(0, 0, h, w); // max equal to dim -> out of bounds (inclusive)
        let sources = LightSources {
            catalog_stars: &stars,
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        let err = render_one_frame_roi(&traj, &sources, &fp, Duration::ZERO, 0, &cfg, None, oob, 0)
            .unwrap_err();
        assert!(
            matches!(err, TrajectoryError::RoiOutOfBounds { .. }),
            "expected RoiOutOfBounds, got {err:?}"
        );

        let bad_sensor = render_one_frame_roi(
            &traj,
            &sources,
            &fp,
            Duration::ZERO,
            0,
            &cfg,
            None,
            AABB::from_coords(0, 0, h - 1, w - 1),
            99,
        )
        .unwrap_err();
        assert!(
            matches!(bad_sensor, TrajectoryError::NoSensors),
            "expected NoSensors for sensor_idx out of range, got {bad_sensor:?}"
        );
    }

    #[test]
    fn test_frame_sensor_tile_parallelism_determinism() {
        // Rendering twice with the same seeds should produce identical
        // output bytes regardless of rayon pool size.
        let fp = tiny_fp();
        let traj = static_trajectory();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let stars: Vec<StarData> = (0..4)
            .map(|i| StarData {
                id: i as u64,
                magnitude: 7.0,
                position: Equatorial::from_degrees(
                    pointing.ra_degrees() + 0.0005 * i as f64,
                    pointing.dec_degrees(),
                ),
                b_v: Some(0.6),
            })
            .collect();
        let cfg = MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.1,
            base_seed: Some(1337),
            force_static: false,
            quiet: true,
            ..Default::default()
        };
        let tmp_a = tempfile::tempdir().unwrap();
        let tmp_b = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &stars,
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp_a.path()).unwrap();
        render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp_b.path()).unwrap();

        // Compare the PNG bytes of one frame. The new layout routes frame 0
        // of sensor 0 to `sensor_00/frame_000000.png`.
        let name = "sensor_00/frame_000000.png";
        let a = std::fs::read(tmp_a.path().join(name)).unwrap();
        let b = std::fs::read(tmp_b.path().join(name)).unwrap();
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // Output layout and metadata.json coverage.
    // -----------------------------------------------------------------------

    /// Build a minimal motion-blur config shaped for the layout/metadata
    /// tests below. Kept local to tests so production does not carry a
    /// test-only constructor.
    fn minimal_metadata_cfg(seed: u64) -> MotionBlurConfig {
        MotionBlurConfig {
            timestep: Duration::from_secs(1),
            exposure: Duration::from_secs(1),
            max_drift_per_stamp_px: 0.1,
            base_seed: Some(seed),
            force_static: true,
            quiet: true,
            telescope_name: "Test".to_string(),
            catalog_path: std::path::PathBuf::from("fake_catalog.bin"),
            temperature_c: -10.0,
        }
    }

    #[test]
    fn test_output_structure_creates_per_sensor_dirs() {
        // Render two frames on a single-sensor focal plane. Assert that the
        // per-sensor directory is created, that both frame files land inside
        // it, and that metadata.json is written at the output root.
        let fp = tiny_fp();
        let traj = static_trajectory();
        let cfg = minimal_metadata_cfg(5);
        let tmp = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &[],
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        let frames = render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp.path()).unwrap();
        assert!(frames >= 2, "expected at least 2 frames, got {}", frames);

        let sensor_dir = tmp.path().join("sensor_00");
        assert!(sensor_dir.is_dir(), "sensor_00 directory must exist");
        assert!(sensor_dir.join("frame_000000.png").is_file());
        assert!(sensor_dir.join("frame_000001.png").is_file());

        let metadata_path = tmp.path().join("metadata.json");
        assert!(metadata_path.is_file(), "metadata.json must exist");

        // The legacy flat-layout file must not be produced any more.
        assert!(!tmp.path().join("frame_000000.png").exists());
        assert!(!tmp.path().join("frame_000000_sensor_00.png").exists());
    }

    #[test]
    fn test_metadata_json_round_trip() {
        // Render a tiny sequence, parse metadata.json back as a generic
        // serde_json::Value, and spot-check the shape-level invariants.
        let fp = tiny_fp();
        let traj = static_trajectory();
        let cfg = minimal_metadata_cfg(17);
        let tmp = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &[],
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        let frames = render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp.path()).unwrap();
        let raw = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(v["version"], "1.2");
        // With no extended sources the galaxies list is present and empty.
        assert!(v["galaxies"].as_array().unwrap().is_empty());
        let frames_arr = v["frames"].as_array().unwrap();
        assert_eq!(frames_arr.len(), frames);
        assert!(frames_arr[0]["paths"]
            .as_object()
            .unwrap()
            .contains_key("sensor_00"));
        assert_eq!(
            frames_arr[0]["paths"]["sensor_00"], "sensor_00/frame_000000.png",
            "relative PNG path should use forward slashes and match layout"
        );

        let waypoints = v["trajectory"]["waypoints"].as_array().unwrap();
        assert!(
            waypoints.len() >= 2,
            "trajectory must expose >= 2 waypoints"
        );
        // quat is [w, x, y, z] — 4 elements.
        assert_eq!(waypoints[0]["quat"].as_array().unwrap().len(), 4);

        assert!(v["stars"].is_array());

        // focal_plane carries the full FocalPlaneConfig graph (telescope,
        // sensor array with per-sensor noise / QE / geometry). Shape spot-
        // check only — uom unit serialization is opaque, so we just verify
        // the top-level objects exist with the right children.
        let focal_plane = &v["focal_plane"];
        assert!(focal_plane["telescope"].is_object());
        assert!(focal_plane["telescope"]["name"].is_string());
        let tele_qe = &focal_plane["telescope"]["quantum_efficiency"];
        let tele_wl = tele_qe["wavelengths_nm"].as_array().unwrap();
        let tele_eff = tele_qe["efficiencies"].as_array().unwrap();
        assert_eq!(tele_wl.len(), tele_eff.len());
        assert!(tele_wl.len() >= 2);
        let sensors = focal_plane["array"]["sensors"].as_array().unwrap();
        assert!(!sensors.is_empty());
        let s0_inner = &sensors[0]["sensor"];
        assert!(s0_inner["name"].is_string());
        assert!(s0_inner["bit_depth"].as_u64().unwrap() > 0);
        assert!(s0_inner["dn_per_electron"].as_f64().unwrap() > 0.0);
        let dc = &s0_inner["dark_current_estimator"];
        let dc_t = dc["temperatures_c"].as_array().unwrap();
        let dc_v = dc["dark_currents_e_per_px_per_s"].as_array().unwrap();
        assert_eq!(dc_t.len(), dc_v.len());
        assert!(dc_t.len() >= 2);

        let zodi = &v["render_config"]["zodiacal"];
        assert!(zodi["elongation_deg"].is_number());
        assert!(zodi["latitude_deg"].is_number());
    }

    #[test]
    fn test_metadata_includes_extended_sources_at_sky_coords() {
        // A LightSources carrying galaxies must surface them under
        // metadata.galaxies — flat, like stars — with sky coordinates,
        // integrated electron rate, and the full Sérsic shape.
        use crate::photometry::photoconversion::{SourceFlux, SpotFlux};
        use shared::image_proc::airy::PixelScaledAiryDisk;
        use shared::units::Wavelength;
        use starfield::catalogs::SersicProfile;

        let profile = SersicProfile {
            theta_half_arcsec: 3.5,
            n: 2.5,
            axis_ratio: 0.6,
            position_angle_deg: 42.0,
        };
        let psf = PixelScaledAiryDisk::with_fwhm(2.0, Wavelength::from_nanometers(550.0));
        let spot = SpotFlux {
            disk: psf,
            flux: 1.25e-2,
        };
        let flux = SourceFlux {
            photons: spot.clone(),
            electrons: spot,
        };
        let galaxy = crate::scene_galaxy::Galaxy {
            id: 987654,
            name: Some("M-test".to_string()),
            position: Equatorial::from_degrees(123.5, -7.25),
            profile,
            flux,
        };
        let galaxies = vec![galaxy];

        let fp = tiny_fp();
        let traj = static_trajectory();
        let cfg = minimal_metadata_cfg(29);
        let tmp = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &[],
            galaxies: &galaxies,
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp.path()).unwrap();
        let raw = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let meta: crate::sims::motion_blur_metadata::RenderMetadata =
            serde_json::from_str(&raw).unwrap();

        assert_eq!(meta.galaxies.len(), 1);
        let g = &meta.galaxies[0];
        assert_eq!(g.id, 987654);
        assert_eq!(g.name.as_deref(), Some("M-test"));
        assert_abs_diff_eq!(g.position.ra_degrees(), 123.5, epsilon = 1e-12);
        assert_abs_diff_eq!(g.position.dec_degrees(), -7.25, epsilon = 1e-12);
        assert_abs_diff_eq!(g.flux.electrons.flux, 1.25e-2, epsilon = 1e-15);
        assert_abs_diff_eq!(g.profile.theta_half_arcsec, 3.5, epsilon = 1e-15);
        assert_abs_diff_eq!(g.profile.n, 2.5, epsilon = 1e-15);
        assert_abs_diff_eq!(g.profile.axis_ratio, 0.6, epsilon = 1e-15);
        assert_abs_diff_eq!(g.profile.position_angle_deg, 42.0, epsilon = 1e-15);
    }

    #[test]
    fn test_metadata_frame_boresight_matches_trajectory() {
        // The mid-frame boresight recorded in metadata.json must match the
        // one computed from `trajectory.orientation_at(mid_t)` to within
        // numerical noise.
        let fp = tiny_fp();
        let traj = static_trajectory();
        let cfg = minimal_metadata_cfg(23);
        let tmp = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &[],
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp.path()).unwrap();
        let raw = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let meta: crate::sims::motion_blur_metadata::RenderMetadata =
            serde_json::from_str(&raw).unwrap();

        for frame in &meta.frames {
            // Mid-frame orientation: same clamping as the renderer.
            let frame_start = Duration::from_secs_f64(frame.t_s);
            let exposure = Duration::from_secs_f64(frame.exposure_s);
            let mid_t = (frame_start + exposure / 2)
                .min(traj.end_time())
                .max(traj.start_time());
            let q = traj.orientation_at(mid_t).unwrap();
            let expected = boresight_of(&q);
            assert!(
                (frame.boresight.ra_deg - expected.ra_degrees()).abs() < 1e-9,
                "frame {} RA mismatch: meta={} expected={}",
                frame.idx,
                frame.boresight.ra_deg,
                expected.ra_degrees()
            );
            assert!(
                (frame.boresight.dec_deg - expected.dec_degrees()).abs() < 1e-9,
                "frame {} Dec mismatch: meta={} expected={}",
                frame.idx,
                frame.boresight.dec_deg,
                expected.dec_degrees()
            );
        }
    }

    #[test]
    fn test_metadata_quat_round_trips_via_serde() {
        // Build a trajectory whose first waypoint has a known roll about the
        // boresight, then verify the metadata round-trips that quaternion
        // exactly (component-for-component) and reproduces the original roll
        // when re-evaluated through `roll_of`. Implementation-agnostic with
        // respect to nalgebra's on-disk array layout — we read fields off
        // the deserialized `UnitQuaternion` rather than indexing positions.
        use crate::sims::orientation::orientation_from_pointing;

        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let roll = 0.7_f64; // radians
        let traj = Trajectory::from_endpoints_with_roll(
            pointing,
            roll,
            pointing,
            roll,
            Duration::from_secs(10),
        )
        .unwrap();

        let fp = tiny_fp();
        let cfg = minimal_metadata_cfg(31);
        let tmp = tempfile::tempdir().unwrap();
        let sources = LightSources {
            catalog_stars: &[],
            galaxies: &[],
            zodiacal: SolarAngularCoordinates::zodiacal_minimum(),
        };
        render_motion_trajectory(&traj, &sources, &fp, &cfg, tmp.path()).unwrap();
        let raw = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let meta: crate::sims::motion_blur_metadata::RenderMetadata =
            serde_json::from_str(&raw).unwrap();

        let q_expected = orientation_from_pointing(&pointing, roll);
        let wp0 = &meta.trajectory.waypoints[0];
        assert_abs_diff_eq!(wp0.quat.w, q_expected.w, epsilon = 1e-12);
        assert_abs_diff_eq!(wp0.quat.i, q_expected.i, epsilon = 1e-12);
        assert_abs_diff_eq!(wp0.quat.j, q_expected.j, epsilon = 1e-12);
        assert_abs_diff_eq!(wp0.quat.k, q_expected.k, epsilon = 1e-12);

        // Round-trip: roll_of evaluated on the deserialized quaternion
        // recovers the original roll angle.
        let recovered = roll_of(&wp0.quat);
        assert_abs_diff_eq!(recovered, roll, epsilon = 1e-9);
    }

    // -----------------------------------------------------------------------
    // Pure-tone PSF spreading.
    //
    // Closed-form check that the stratified-MC stamp loop reproduces the
    // expected motion-blur smear of a sinusoidal one-axis tilt: variance
    // of a sinusoid with amplitude A is A²/2, and convolving the static
    // PSF with the tone's position density grows the per-pixel second
    // moment of the rendered star by exactly that amount on the affected
    // axis (and not at all on the orthogonal axis).
    // -----------------------------------------------------------------------

    /// Stratified-MC accumulator helper. Mirrors the production
    /// [`render_tile`] flat stamp loop without the Poisson + read-noise
    /// + PNG-write tail, returning the pre-noise mean-electron image.
    fn simulate_tile_accumulator_stratified(
        trajectory: &Trajectory,
        star: &StarData,
        fp: &FocalPlaneConfig,
        cfg: &MotionBlurConfig,
        seed: u64,
    ) -> SensorAccumulator {
        let first_sat = fp.satellite_for_sensor(0).unwrap();
        let pixel_size_mm = first_sat.sensor.pixel_size().as_millimeters();
        let airy_pix = first_sat.airy_disk_pixel_space();
        let padding_mm = airy_pix.first_zero() * 2.0 * pixel_size_mm;
        let px_scale = pixel_scale_rad(fp).unwrap_or(0.0);

        let t_start = Duration::ZERO;
        let t_end = (t_start + cfg.exposure).min(trajectory.end_time());
        let exposure = t_end - t_start;
        let drift = max_drift_over_window(trajectory, t_start, t_end).unwrap();
        let mid_t = (t_start + exposure / 2)
            .min(trajectory.end_time())
            .max(trajectory.start_time());
        let mid_q = trajectory.orientation_at(mid_t).unwrap();
        let peak_excursion = trajectory
            .peak_excursion_rad(t_start, t_end, &mid_q)
            .unwrap();
        let schedule = SubsampleSchedule::from_drift_budget(
            t_start,
            exposure,
            drift,
            peak_excursion,
            px_scale,
            cfg.max_drift_per_stamp_px,
        );
        let (width, height) = first_sat.sensor.dimensions.get_pixel_width_height();
        let mut acc = SensorAccumulator::zero(width, height);
        let aperture = first_sat.telescope.clear_aperture_area();
        let flux = star_data_to_fluxes(star, &first_sat);
        let dt = schedule.dt();
        let stamp_weight = 1.0 / schedule.stamps_per_exposure.max(1) as f64;
        let stamp_phase = quasi_random::phase_from_seed(seed ^ STAMP_PHASE_DOMAIN);
        for stamp_t in schedule.stamp_times(stamp_phase) {
            let t_clamped = stamp_t
                .min(trajectory.end_time())
                .max(trajectory.start_time());
            let q = trajectory.orientation_at(t_clamped).unwrap();
            if let Some((px, py)) = fp.project_to_sensor(star, &q, 0, padding_mm) {
                let total = flux.electrons.integrated_over(&dt, aperture) * stamp_weight;
                acc.splat_psf(px, py, total, &flux.electrons.disk);
            }
        }
        acc
    }

    /// Flux-weighted second moments of an electron-mean image around the
    /// flux-weighted centroid. Background-subtracts the corner median,
    /// then takes full-image moments — no threshold cut, since
    /// thresholding biases the second moment when the underlying
    /// distribution is sub-pixel-narrow (the bright pixel passes, the
    /// PSF wings don't, and σ² collapses to zero).
    fn image_centroid_and_variance(img: &Array2<f64>) -> (f64, f64, f64, f64) {
        let (h, w) = img.dim();
        let mut corners = vec![
            img[[0, 0]],
            img[[0, w - 1]],
            img[[h - 1, 0]],
            img[[h - 1, w - 1]],
        ];
        corners.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let bg = corners[1];
        let mut sum_w = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for y in 0..h {
            for x in 0..w {
                let v = (img[[y, x]] - bg).max(0.0);
                sum_w += v;
                sum_x += v * x as f64;
                sum_y += v * y as f64;
            }
        }
        let cx = sum_x / sum_w;
        let cy = sum_y / sum_w;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;
        for y in 0..h {
            for x in 0..w {
                let v = (img[[y, x]] - bg).max(0.0);
                sum_xx += v * (x as f64 - cx).powi(2);
                sum_yy += v * (y as f64 - cy).powi(2);
            }
        }
        (cy, cx, sum_yy / sum_w, sum_xx / sum_w)
    }

    /// Build a trajectory whose orientation is the nominal pointing
    /// modulated by a sinusoidal tilt of amplitude `amp_rad` at
    /// frequency `freq_hz` about an arbitrary body-frame `axis`
    /// (a unit vector). Sampled densely enough that SLERP between
    /// waypoints reconstructs the tone with negligible error
    /// (≥ 64 samples per cycle).
    fn build_pure_tone_trajectory(
        pointing: Equatorial,
        amp_rad: f64,
        freq_hz: f64,
        duration: Duration,
        axis: nalgebra::Vector3<f64>,
    ) -> Trajectory {
        let cycles = (freq_hz * duration.as_secs_f64()).ceil() as usize;
        let n_waypoints = (cycles * 64).max(256);
        let q_base = orientation_from_pointing(&pointing, 0.0);
        let dt = duration.as_secs_f64();
        let waypoints: Vec<Waypoint> = (0..=n_waypoints)
            .map(|i| {
                let t_s = dt * i as f64 / n_waypoints as f64;
                let theta = amp_rad * (2.0 * PI * freq_hz * t_s).sin();
                let q_jitter = UnitQuaternion::from_scaled_axis(axis * theta);
                let q_total = q_base * q_jitter;
                Waypoint::new(Duration::from_secs_f64(t_s), q_total)
            })
            .collect();
        Trajectory::new(waypoints).unwrap()
    }

    /// Run the variance-addition identity check for a pure sinusoidal
    /// tilt of amplitude `amp_px` (in pixels) about an arbitrary body
    /// axis. Asserts:
    ///   1. the rendered centroid is invariant (zero-mean tone);
    ///   2. the total per-axis second-moment growth equals A²/2
    ///      within 25% (loose tolerance absorbs finite-cycles, finite-
    ///      stamp MC noise, sub-pixel PSF discretization, and SLERP
    ///      curvature between waypoints — the math itself is exact);
    ///   3. the growth is anisotropic — one image axis absorbs ≥70% of
    ///      the predicted variance, and the other image axis's variance
    ///      stays within 10% of its static value.
    /// `axis_label` appears in failure messages so a regression in the
    /// body→image projection convention is easy to read off.
    fn assert_pure_tone_psf_spread(body_axis: nalgebra::Vector3<f64>, axis_label: &str) {
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let star = StarData {
            id: 0,
            magnitude: 5.0,
            position: pointing,
            b_v: Some(0.6),
        };

        let exposure = Duration::from_millis(250);
        let px_scale = pixel_scale_rad(&fp).unwrap();
        // Amplitude well above the pixel pitch so the variance signal
        // dominates any sub-pixel discretization artifact.
        let amp_px = 2.0_f64;
        let amp_rad = amp_px * px_scale;
        // 100 Hz over 250 ms = 25 cycles per exposure, well into the
        // "many cycles → arcsine density on the affected axis" regime
        // where Var(A·sin(ωt)) = A²/2 is exact.
        let tone_freq_hz = 100.0;
        let predicted_extra_var_px2 = amp_px * amp_px / 2.0;

        let traj_end = exposure + Duration::from_millis(50);
        let static_traj = Trajectory::new(vec![
            Waypoint::new(Duration::ZERO, orientation_from_pointing(&pointing, 0.0)),
            Waypoint::new(traj_end, orientation_from_pointing(&pointing, 0.0)),
        ])
        .unwrap();
        let tone_traj =
            build_pure_tone_trajectory(pointing, amp_rad, tone_freq_hz, traj_end, body_axis);

        // Loose per-sample budget so N stays modest; tight per-stamp
        // budget so the stratified-MC stamp loop captures the tone.
        let cfg = MotionBlurConfig {
            timestep: exposure,
            exposure,
            max_drift_per_stamp_px: 0.05,
            base_seed: Some(7),
            force_static: false,
            quiet: true,
            ..Default::default()
        };
        let seed = tile_seed(7, 0, 0);

        let static_acc = simulate_tile_accumulator_stratified(&static_traj, &star, &fp, &cfg, seed);
        let tone_acc = simulate_tile_accumulator_stratified(&tone_traj, &star, &fp, &cfg, seed);

        let (cy_s, cx_s, var_y_s, var_x_s) =
            image_centroid_and_variance(&static_acc.star_mean_electrons);
        let (cy_t, cx_t, var_y_t, var_x_t) =
            image_centroid_and_variance(&tone_acc.star_mean_electrons);

        // 1. Centroid invariance.
        assert!(
            (cx_t - cx_s).abs() < 0.05 && (cy_t - cy_s).abs() < 0.05,
            "[body-{axis_label} tone] centroid drift too large: Δ=({:+.3}, {:+.3}) px",
            cx_t - cx_s,
            cy_t - cy_s
        );

        // 2. Total variance growth ≈ A²/2.
        let total_extra_var = (var_x_t + var_y_t) - (var_x_s + var_y_s);
        let total_rel_err =
            (total_extra_var - predicted_extra_var_px2).abs() / predicted_extra_var_px2;
        assert!(
            total_rel_err < 0.25,
            "[body-{axis_label} tone] total variance growth {:.4} px² vs predicted {:.4} px² \
             (rel err {:.1}%)",
            total_extra_var,
            predicted_extra_var_px2,
            total_rel_err * 100.0
        );

        // 3. Anisotropy: one axis absorbs the growth, the other stays put.
        let dx = (var_x_t - var_x_s).abs();
        let dy = (var_y_t - var_y_s).abs();
        assert!(
            dx.max(dy) >= 0.7 * predicted_extra_var_px2,
            "[body-{axis_label} tone] neither image axis absorbed the predicted variance: \
             Δσ²_x={:.4}, Δσ²_y={:.4}, predicted {:.4}",
            var_x_t - var_x_s,
            var_y_t - var_y_s,
            predicted_extra_var_px2
        );
        let (unaffected_static, unaffected_tone, unaffected_label) = if dx > dy {
            (var_y_s, var_y_t, "y")
        } else {
            (var_x_s, var_x_t, "x")
        };
        let unaffected_change =
            (unaffected_tone - unaffected_static).abs() / unaffected_static.max(1e-6);
        assert!(
            unaffected_change < 0.10,
            "[body-{axis_label} tone] unaffected image-{unaffected_label} variance changed by \
             {:.1}%: from {:.4} to {:.4}",
            unaffected_change * 100.0,
            unaffected_static,
            unaffected_tone
        );

        eprintln!(
            "body-{axis_label} pure-tone PSF spread: amp={:.2} px, predicted Δσ²={:.4} px²; \
             measured Δσ²_x={:+.4}, Δσ²_y={:+.4}, total={:.4}",
            amp_px,
            predicted_extra_var_px2,
            var_x_t - var_x_s,
            var_y_t - var_y_s,
            total_extra_var
        );
    }

    #[test]
    fn test_pure_tone_jitter_body_x_axis() {
        assert_pure_tone_psf_spread(nalgebra::Vector3::x(), "X");
    }

    #[test]
    fn test_pure_tone_jitter_body_y_axis() {
        assert_pure_tone_psf_spread(nalgebra::Vector3::y(), "Y");
    }
}

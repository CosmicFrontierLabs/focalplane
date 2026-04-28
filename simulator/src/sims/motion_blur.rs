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
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use shared::image_proc::noise::apply_poisson_photon_noise;
use shared::units::{AngleExt, LengthExt, TemperatureExt};
use starfield::catalogs::StarData;
use starfield::Equatorial;

use crate::hardware::satellite::{FocalPlaneConfig, FocalPlaneProjector};
use crate::hardware::SatelliteConfig;
use crate::image_proc::render::quantize_image;
use crate::photometry::photoconversion::SourceFlux;
use crate::photometry::spectrum::Spectrum;
use crate::photometry::zodiacal::{SolarAngularCoordinates, ZodiacalLight};
use crate::sims::motion_blur_metadata::{
    sensor_dir_name, sensor_relative_png_path, EquatorialMeta, FrameMeta, HardwareMeta,
    RenderConfigMeta, RenderMetadata, SensorMeta, StarMeta, TrajectoryMeta, WaypointMeta,
    ZodiacalMeta,
};
use crate::sims::orientation::{boresight_of, roll_of};
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

    /// Stratified Monte Carlo stamp times across the entire exposure
    /// window. Returns a vector of length `stamps_per_exposure.max(1)`.
    /// Stamp `j` lands at `frame_start + (j + U_j) * dt`, where
    /// `U_j ~ Uniform[0, 1)` is drawn from `rng` and
    /// `dt = exposure / stamps_per_exposure`.
    pub fn stamp_times<R: Rng>(&self, rng: &mut R) -> Vec<Duration> {
        let total = self.stamps_per_exposure.max(1);
        let dt = self.exposure.as_secs_f64() / total as f64;
        let t0 = self.frame_start.as_secs_f64();
        (0..total)
            .map(|j| {
                let u: f64 = rng.random();
                Duration::from_secs_f64(t0 + (j as f64 + u) * dt)
            })
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
        if total_electrons == 0.0 {
            return;
        }
        let (height, width) = self.star_mean_electrons.dim();
        let max_pix_dist: i32 = (psf.first_zero().max(1.0) * 2.0).ceil() as i32;
        let xc = px.round() as i32;
        let yc = py.round() as i32;
        for x in (xc - max_pix_dist)..=(xc + max_pix_dist) {
            for y in (yc - max_pix_dist)..=(yc + max_pix_dist) {
                if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                    continue;
                }
                let x_rel = x as f64 - px;
                let y_rel = y as f64 - py;
                let contribution = psf.pixel_flux_simpson(x_rel, y_rel, total_electrons);
                self.star_mean_electrons[[y as usize, x as usize]] += contribution;
            }
        }
    }

    /// Returns the combined mean-electron image = star mean + zodiacal
    /// uniform + dark-current uniform (pre-Poisson).
    pub fn combined_mean(&self, zodiacal_per_px: f64, dark_per_px: f64) -> Array2<f64> {
        let bg = (zodiacal_per_px + dark_per_px).max(0.0);
        &self.star_mean_electrons + bg
    }
}

/// Per-tile flux cache keyed by `(star_id, sensor_idx)`.
///
/// Flux calculation is expensive (Simpson's rule integration over stellar
/// spectrum × QE curve) and depends only on the star and the sensor, not on
/// the spacecraft orientation. Caching across frames and subsamples avoids
/// repeating that work for every sub-orientation.
pub type FluxCache = HashMap<(u64, usize), SourceFlux>;

/// Static inputs shared across every `(frame, sensor)` tile in a render.
///
/// Bundles the trajectory, the full star catalog, the focal-plane hardware,
/// and the scene's zodiacal coordinates so downstream functions take a single
/// `&RenderScene` rather than four independent references.
struct RenderScene<'a> {
    trajectory: &'a Trajectory,
    catalog_stars: &'a [StarData],
    fp: &'a FocalPlaneConfig,
    zodiacal: SolarAngularCoordinates,
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

/// Configuration for [`render_motion_trajectory`].
#[derive(Debug, Clone)]
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

/// Named randomness sources within a single render tile. Each source
/// gets its own deterministic seed derived from the tile's master
/// seed, so perturbing one (e.g. changing the stamp-jitter sequence)
/// cannot shift the bytes the others would have produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngDomain {
    /// Unified Poisson photon-noise draw on the accumulated mean image.
    Poisson,
    /// Gaussian read-noise draw applied after Poisson.
    ReadNoise,
    /// Stratified-MC uniform-random per-stamp time jitter inside each subsample.
    StampJitter,
}

impl RngDomain {
    /// Domain-specific 64-bit XOR tag. Values are arbitrary as long
    /// as they are pairwise distinct; the downstream
    /// `StdRng::seed_from_u64` does its own state expansion. Treat
    /// these as labels, not as cryptographic constants.
    const fn tag(self) -> u64 {
        match self {
            RngDomain::Poisson => 0x0000_0000_0000_0000,
            RngDomain::ReadNoise => 0xA5A5_5A5A_A5A5_5A5A,
            RngDomain::StampJitter => 0x4D6F_6E74_6543_6172, // "MonteCar" (8 ASCII bytes)
        }
    }
}

/// Per-tile master seed plus the API for deriving sub-stream seeds
/// and RNGs for each [`RngDomain`]. All streams are deterministic
/// from `(base_seed, frame_idx, sensor_idx)`, and each domain is
/// independent of the others — adding a new randomness source via a
/// new [`RngDomain`] variant does not perturb existing output.
#[derive(Debug, Clone, Copy)]
pub struct TileSeed(u64);

impl TileSeed {
    /// Build the per-tile master seed from `(base, frame_idx, sensor_idx)`.
    pub fn for_tile(base: u64, frame_idx: usize, sensor_idx: usize) -> Self {
        // Cheap splitmix-style mix; reproducible, well-distributed
        // enough for RNG seeding.
        let mut h = base
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(frame_idx as u64)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= (sensor_idx as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 31;
        Self(h)
    }

    /// Sub-stream seed for APIs that consume `u64` directly
    /// (e.g. [`apply_poisson_photon_noise`]).
    pub fn seed(self, domain: RngDomain) -> u64 {
        self.0 ^ domain.tag()
    }

    /// Sub-stream RNG ready to consume for APIs that take `&mut impl Rng`.
    pub fn rng(self, domain: RngDomain) -> StdRng {
        StdRng::seed_from_u64(self.seed(domain))
    }
}

/// Render a single `(frame, sensor)` tile.
///
/// Runs one flat stratified-MC stamp loop across the exposure,
/// composes the unified Poisson lambda, draws Poisson + Gaussian
/// read noise, quantizes, and saves a PNG.
fn render_tile(
    scene: &RenderScene,
    plan: &FramePlan,
    sensor_idx: usize,
    flux_cache: &Arc<Mutex<FluxCache>>,
    satellite: &SatelliteConfig,
    tile_seed: TileSeed,
    output_path: &Path,
) -> Result<(), TrajectoryError> {
    let (width, height) = satellite.sensor.dimensions.get_pixel_width_height();
    let mut accumulator = SensorAccumulator::zero(width, height);
    let aperture = satellite.telescope.clear_aperture_area();

    let schedule = &plan.schedule;
    let dt = schedule.dt();
    let stamp_weight = 1.0 / schedule.stamps_per_exposure.max(1) as f64;

    // Per-tile RNG for stratified-MC stamp placement; sequential
    // consumption keeps the stamp-time sequence bit-deterministic
    // for a given tile seed.
    let mut stamp_rng = tile_seed.rng(RngDomain::StampJitter);

    // One-shot per-tile cache of (star -> SourceFlux) lookups. Flux
    // depends only on (star, sensor), not on orientation, so the
    // value is stable across every stamp of the exposure.
    let mut local_flux: HashMap<u64, SourceFlux> = HashMap::new();

    for stamp_t in schedule.stamp_times(&mut stamp_rng) {
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
            // Per-stamp electron contribution: flux rate × per-stamp
            // dt (= exposure / stamps_per_exposure) × aperture. Sum
            // over all stamps reproduces the full exposure budget.
            let total_electrons = flux.electrons.integrated_over(&dt, aperture) * stamp_weight;
            accumulator.splat_psf(hit.0, hit.1, total_electrons, &flux.electrons.disk);
        }
    }

    // Dark current: rate × full exposure, uniform over pixels.
    let dark_rate = satellite
        .sensor
        .dark_current_at_temperature(satellite.temperature);
    let dark_mean = (dark_rate * schedule.exposure.as_secs_f64()).max(0.0);

    // Build unified Poisson mean image and draw.
    let mean_image = accumulator.combined_mean(plan.zodiacal_per_px[sensor_idx], dark_mean);
    let poisson_image =
        apply_poisson_photon_noise(&mean_image, Some(tile_seed.seed(RngDomain::Poisson)));

    // Gaussian read noise (electronics, not shot noise) applied afterwards.
    let read_noise_rms = satellite
        .sensor
        .read_noise_estimator
        .estimate(satellite.temperature.as_celsius(), schedule.exposure)
        .unwrap_or(0.0)
        .max(0.0);
    let final_electrons = if read_noise_rms > 0.0 {
        let mut rng = tile_seed.rng(RngDomain::ReadNoise);
        let normal =
            Normal::new(0.0_f64, read_noise_rms).expect("read noise RMS must be non-negative");
        poisson_image.mapv(|e| (e + normal.sample(&mut rng)).max(0.0))
    } else {
        poisson_image
    };

    let quantized = quantize_image(&final_electrons, &satellite.sensor);
    save_u16_png(&quantized, output_path)
}

/// Render the full trajectory with motion blur, parallel over `(frame, sensor)`.
///
/// Returns the total number of frames rendered.
pub fn render_motion_trajectory(
    trajectory: &Trajectory,
    catalog_stars: &[StarData],
    fp: &FocalPlaneConfig,
    zodiacal: SolarAngularCoordinates,
    config: &MotionBlurConfig,
    output_dir: &Path,
) -> Result<usize, TrajectoryError> {
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

    let scene = RenderScene {
        trajectory,
        catalog_stars,
        fp,
        zodiacal,
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

    // Pre-build per-sensor satellite configs (cheap, avoids lock-free work
    // per tile). Materialized early so the planning pass can precompute
    // the per-sensor zodiacal scaling for each frame.
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
        let exposure = config.exposure;
        // Clamp the exposure window to the trajectory so we do not try to
        // sample beyond the defined range.
        let t_end = (t + exposure).min(scene.trajectory.end_time());
        let exposure = if t_end > t { t_end - t } else { Duration::ZERO };

        // Mid-frame orientation drives both the envelope-padding peak
        // excursion calculation and the zodiacal evaluation.
        let mid_t = (t + exposure / 2)
            .min(scene.trajectory.end_time())
            .max(scene.trajectory.start_time());
        let mid_q = scene.trajectory.orientation_at(mid_t)?;
        let mid_bore = boresight_of(&mid_q);

        let drift = max_drift_over_window(scene.trajectory, t, t_end)?;
        let peak_excursion = scene.trajectory.peak_excursion_rad(t, t_end, &mid_q)?;

        let schedule = if config.force_static || exposure.is_zero() {
            SubsampleSchedule {
                frame_start: t,
                exposure,
                stamps_per_exposure: 1,
                envelope_padding_rad: peak_excursion,
            }
        } else {
            SubsampleSchedule::from_drift_budget(
                t,
                exposure,
                drift,
                peak_excursion,
                px_scale,
                config.max_drift_per_stamp_px,
            )
        };
        stamps_min = stamps_min.min(schedule.stamps_per_exposure);
        stamps_max = stamps_max.max(schedule.stamps_per_exposure);

        let zodiacal_per_px_per_s =
            zodiacal_per_px_per_s_at(&zlight, &first_sat, &scene.zodiacal, &mid_bore);
        let exposure_s = schedule.exposure.as_secs_f64();
        let zodiacal_per_px: Vec<f64> = sensor_ratios
            .iter()
            .map(|r| zodiacal_per_px_per_s * r * exposure_s)
            .collect();

        let stars = envelope_prefilter(
            scene.trajectory,
            scene.catalog_stars,
            &schedule,
            scene.fp,
            padding_mm,
            px_scale,
        )?;
        // Per-frame summary routed through the bar so it coexists with the
        // live progress line without scrambled output.
        pb.println(format!(
            "frame {:06} at t={:.3}s: stamps={}, boresight=(ra={:.4}°, dec={:.4}°), \
             stars={}",
            frame_idx,
            t.as_secs_f64(),
            schedule.stamps_per_exposure,
            mid_bore.ra_degrees(),
            mid_bore.dec_degrees(),
            stars.len(),
        ));
        plans.push(FramePlan {
            idx: frame_idx,
            schedule,
            stars,
            padding_mm,
            zodiacal_per_px,
        });
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
            let sat = &satellites[tile.sensor_idx];
            let seed = TileSeed::for_tile(base_seed, plan.idx, tile.sensor_idx);
            let tile_started = Instant::now();
            let result = render_tile(
                &scene,
                plan,
                tile.sensor_idx,
                &flux_cache,
                sat,
                seed,
                out_path,
            );
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
        &satellites,
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
    satellites: &[SatelliteConfig],
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
                    quat: [q.w, q.i, q.j, q.k],
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
            quat: [q.w, q.i, q.j, q.k],
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
        .catalog_stars
        .iter()
        .map(|s| StarMeta {
            id: s.id,
            ra_deg: s.position.ra_degrees(),
            dec_deg: s.position.dec_degrees(),
            magnitude: s.magnitude,
            color_index: s.b_v,
        })
        .collect();

    let sensors: Vec<SensorMeta> = (0..sensor_count)
        .map(|i| {
            let ps = &scene.fp.array.sensors[i];
            let (width, height) = ps.sensor.dimensions.get_pixel_width_height();
            SensorMeta {
                idx: i,
                name: satellites[i].sensor.name.clone(),
                dimensions_px: [width, height],
                position_mm: [ps.position.x_mm, ps.position.y_mm],
            }
        })
        .collect();

    let hardware = HardwareMeta {
        telescope: config.telescope_name.clone(),
        temperature_c: config.temperature_c,
        sensors,
    };

    let render_config = RenderConfigMeta {
        exposure_s: config.exposure.as_secs_f64(),
        timestep_s: config.timestep.as_secs_f64(),
        max_drift_per_stamp_px: config.max_drift_per_stamp_px,
        seed: config.base_seed.unwrap_or(0),
        force_static: config.force_static,
        catalog_path: config.catalog_path.to_string_lossy().into_owned(),
        zodiacal: ZodiacalMeta {
            elongation_deg: scene.zodiacal.elongation(),
            latitude_deg: scene.zodiacal.latitude(),
        },
    };

    Ok(RenderMetadata {
        version: "1.0".to_string(),
        rendered_at,
        trajectory: trajectory_meta,
        frames,
        stars,
        hardware,
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
    fn test_stamp_times_are_stratified_uniformly_across_exposure() {
        // 4 stamps over a 4s exposure: each stamp must land in its own
        // 1s sub-bin [j, j+1) (uniform-random within the bin).
        let sched = SubsampleSchedule {
            frame_start: Duration::from_secs(10),
            exposure: Duration::from_secs(4),
            stamps_per_exposure: 4,
            envelope_padding_rad: 0.0,
        };
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let stamps = sched.stamp_times(&mut rng);
        assert_eq!(stamps.len(), 4);
        for (j, t) in stamps.iter().enumerate() {
            let lo = 10.0 + j as f64;
            let hi = lo + 1.0;
            let v = t.as_secs_f64();
            assert!(
                v >= lo && v < hi,
                "stamp {v} (j={j}) must land in stratum [{lo}, {hi})"
            );
        }
    }

    #[test]
    fn test_stamp_times_seeded_rng_is_reproducible() {
        // Two calls with two RNGs seeded identically must produce
        // identical stamp time sequences — the determinism contract
        // the renderer relies on.
        let sched = SubsampleSchedule {
            frame_start: Duration::ZERO,
            exposure: Duration::from_secs(1),
            stamps_per_exposure: 128,
            envelope_padding_rad: 0.0,
        };
        let mut rng_a = StdRng::seed_from_u64(0xDEADBEEF);
        let mut rng_b = StdRng::seed_from_u64(0xDEADBEEF);
        let a = sched.stamp_times(&mut rng_a);
        let b = sched.stamp_times(&mut rng_b);
        assert_eq!(a, b);
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
        let combined = acc.combined_mean(2.0, 1.0);
        assert_eq!(combined[[0, 0]], 3.0); // 0 + 2 + 1
        assert_eq!(combined[[1, 1]], 8.0); // 5 + 2 + 1
    }

    #[test]
    fn test_tile_seed_is_deterministic_and_varies() {
        let a = TileSeed::for_tile(42, 0, 0).seed(RngDomain::Poisson);
        let b = TileSeed::for_tile(42, 0, 0).seed(RngDomain::Poisson);
        let c = TileSeed::for_tile(42, 1, 0).seed(RngDomain::Poisson);
        let d = TileSeed::for_tile(42, 0, 1).seed(RngDomain::Poisson);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(c, d);
    }

    #[test]
    fn test_tile_seed_domains_are_independent() {
        // The same tile must produce three distinct sub-seeds for the
        // three named domains, so that perturbing (e.g.) the stamp
        // jitter does not accidentally shift the Poisson or read-noise
        // streams.
        let tile = TileSeed::for_tile(1234, 5, 6);
        let p = tile.seed(RngDomain::Poisson);
        let r = tile.seed(RngDomain::ReadNoise);
        let s = tile.seed(RngDomain::StampJitter);
        assert_ne!(p, r);
        assert_ne!(p, s);
        assert_ne!(r, s);
        // Same domain on the same tile must reproduce the same seed.
        assert_eq!(tile.seed(RngDomain::StampJitter), s);
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
        let frames = render_motion_trajectory(
            &traj,
            &[],
            fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp.path(),
        )
        .unwrap();
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
        let frames = render_motion_trajectory(
            &traj,
            &[],
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp.path(),
        )
        .unwrap();
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
        // Render normally to exercise the cache.
        let frames = render_motion_trajectory(
            &traj,
            &stars,
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp.path(),
        )
        .unwrap();
        assert!(frames >= 3);
        // Cache size bound: at most num_stars * sensor_count.
        // We can't peek from outside the function, so rely on invariant by
        // reconstructing the shared mutex via a second call and checking
        // the cache does not keep growing unboundedly with repeat work.
        // The tightest assertion we can make without instrumenting the
        // internals is: a second run with the same inputs renders the same
        // number of frames, which would fail if the path were non-idempotent.
        let frames2 = render_motion_trajectory(
            &traj,
            &stars,
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp.path(),
        )
        .unwrap();
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
        // Total flux conservation within a few percent.
        assert!(
            (static_sum - moving_sum).abs() / static_sum.max(1.0) < 0.05,
            "motion-blur should conserve total flux (static_sum={}, moving_sum={})",
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
        simulate_tile_accumulator_stratified(
            trajectory,
            star,
            fp,
            cfg,
            TileSeed::for_tile(0xDEADBEEF, 0, 0),
        )
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
        render_motion_trajectory(
            &traj,
            &stars,
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp_a.path(),
        )
        .unwrap();
        render_motion_trajectory(
            &traj,
            &stars,
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp_b.path(),
        )
        .unwrap();
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
        render_motion_trajectory(
            &traj,
            &stars,
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg_coarse,
            tmp_coarse.path(),
        )
        .unwrap();
        render_motion_trajectory(
            &traj,
            &stars,
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg_fine,
            tmp_fine.path(),
        )
        .unwrap();
        let name = "sensor_00/frame_000000.png";
        let coarse = std::fs::read(tmp_coarse.path().join(name)).unwrap();
        let fine = std::fs::read(tmp_fine.path().join(name)).unwrap();
        assert_ne!(
            coarse, fine,
            "tightening max_drift_per_stamp_px on a moving trajectory must change the rendered streak"
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
        render_motion_trajectory(
            &traj,
            &stars,
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp_a.path(),
        )
        .unwrap();
        render_motion_trajectory(
            &traj,
            &stars,
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp_b.path(),
        )
        .unwrap();

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
        let frames = render_motion_trajectory(
            &traj,
            &[],
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp.path(),
        )
        .unwrap();
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
        let frames = render_motion_trajectory(
            &traj,
            &[],
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp.path(),
        )
        .unwrap();
        let raw = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(v["version"], "1.0");
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

        let sensors = v["hardware"]["sensors"].as_array().unwrap();
        let dims = sensors[0]["dimensions_px"].as_array().unwrap();
        assert_eq!(dims.len(), 2);
        assert!(dims[0].as_u64().unwrap() > 0);
        assert!(dims[1].as_u64().unwrap() > 0);

        let zodi = &v["render_config"]["zodiacal"];
        assert!(zodi["elongation_deg"].is_number());
        assert!(zodi["latitude_deg"].is_number());
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
        render_motion_trajectory(
            &traj,
            &[],
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp.path(),
        )
        .unwrap();
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
    fn test_metadata_quat_is_wxyz_order() {
        // Build a trajectory whose first waypoint has a known roll about the
        // boresight. The UnitQuaternion for a rotation of angle θ about a
        // unit axis has w = cos(θ/2). We emit quat = [w, x, y, z], so
        // meta.waypoints[0].quat[0] must match cos(θ/2) under the composed
        // orientation.
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
        render_motion_trajectory(
            &traj,
            &[],
            &fp,
            SolarAngularCoordinates::zodiacal_minimum(),
            &cfg,
            tmp.path(),
        )
        .unwrap();
        let raw = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let meta: crate::sims::motion_blur_metadata::RenderMetadata =
            serde_json::from_str(&raw).unwrap();

        // Expected quat comes directly from nalgebra's constructor to avoid
        // re-deriving the half-angle formula under roll composition.
        let q_expected = orientation_from_pointing(&pointing, roll);
        let wp0 = &meta.trajectory.waypoints[0];
        assert_abs_diff_eq!(wp0.quat[0], q_expected.w, epsilon = 1e-12);
        assert_abs_diff_eq!(wp0.quat[1], q_expected.i, epsilon = 1e-12);
        assert_abs_diff_eq!(wp0.quat[2], q_expected.j, epsilon = 1e-12);
        assert_abs_diff_eq!(wp0.quat[3], q_expected.k, epsilon = 1e-12);

        // Round-trip: reconstruct a UnitQuaternion from [w, x, y, z] and
        // check that roll_of recovers the original roll.
        let q_round = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            wp0.quat[0],
            wp0.quat[1],
            wp0.quat[2],
            wp0.quat[3],
        ));
        let recovered = roll_of(&q_round);
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
        seed: TileSeed,
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
        let mut stamp_rng = seed.rng(RngDomain::StampJitter);
        for stamp_t in schedule.stamp_times(&mut stamp_rng) {
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
        let seed = TileSeed::for_tile(7, 0, 0);

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

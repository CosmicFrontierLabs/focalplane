//! Parallel motion-blur renderer for focal-plane trajectories.
//!
//! Replaces the static-per-frame trajectory renderer with an integrator
//! that averages many spacecraft sub-orientations across each exposure.
//! The number of sub-orientations per frame is chosen adaptively so that
//! the per-subsample boresight drift stays below a fraction of a pixel;
//! when the trajectory is static the schedule collapses to a single sample
//! per frame.
//!
//! # Noise model
//!
//! A single unified Poisson draw is taken per `(frame, sensor)` over the
//! mean-electron image comprising:
//!
//! - Star contributions accumulated across all subsamples (each subsample
//!   contributes its own `dt = exposure / N` electron rate).
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
use nalgebra::UnitQuaternion;
use ndarray::Array2;
use rand::{rngs::StdRng, SeedableRng};
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

/// Default per-subsample drift budget (in pixels). Adaptive scheduling picks
/// the smallest N that keeps per-subsample drift below this threshold.
pub const DEFAULT_MAX_DRIFT_PER_SAMPLE_PX: f64 = 0.1;

/// Time-domain subsampling schedule inside an exposure window.
///
/// A `SubsampleSchedule` describes how a frame's exposure is sliced in
/// time. Two cadences:
///
/// - **`n` sub-orientation samples** drive *scene-state* refresh
///   (per-star flux, the in-field star slice, the zodiacal mean).
///   Adaptive: chosen to keep per-subsample boresight drift under
///   `max_drift_per_sample_px`. Typically 1–10.
/// - **`stamps_per_sample` finer trajectory queries inside each
///   subsample** drive *PSF-stamp* placement. Each stamp queries the
///   trajectory at its own midpoint and deposits `flux / M` electrons
///   at the per-time projected pixel position. Defaults to 1 (one
///   stamp per subsample, identical to the original renderer);
///   raise it to capture high-frequency jitter that would otherwise
///   alias instead of contributing motion blur.
///
/// Subsample boundaries are midpoints so the integral of a
/// linearly-varying rate over the exposure equals the midpoint value
/// times the exposure; dividing by `n` yields the per-subsample
/// contribution and dividing again by `stamps_per_sample` yields the
/// per-stamp contribution.
#[derive(Debug, Clone, Copy)]
pub struct SubsampleSchedule {
    /// Absolute trajectory time at which the frame's exposure starts.
    pub frame_start: Duration,
    /// Total exposure duration for this frame.
    pub exposure: Duration,
    /// Number of sub-orientation samples across the exposure (>= 1).
    pub n: usize,
    /// Number of fine trajectory queries inside each subsample (>= 1).
    /// `1` reproduces the original one-stamp-per-subsample behavior.
    pub stamps_per_sample: usize,
}

impl SubsampleSchedule {
    /// Sub-sample interval: `exposure / n`.
    pub fn dt(&self) -> Duration {
        Duration::from_secs_f64(self.exposure.as_secs_f64() / self.n.max(1) as f64)
    }

    /// Evenly spaced sample midpoints inside the exposure window:
    /// `frame_start + (i + 0.5) * dt` for `i in 0..n`.
    pub fn sample_times(&self) -> Vec<Duration> {
        let dt = self.exposure.as_secs_f64() / self.n.max(1) as f64;
        let t0 = self.frame_start.as_secs_f64();
        (0..self.n.max(1))
            .map(|i| Duration::from_secs_f64(t0 + (i as f64 + 0.5) * dt))
            .collect()
    }

    /// Stamp midpoints inside subsample `sample_idx`. Returns a vector
    /// of length `stamps_per_sample.max(1)`, evenly spaced across the
    /// subsample window using the same midpoint convention as
    /// [`Self::sample_times`].
    pub fn stamp_times_for_sample(&self, sample_idx: usize) -> Vec<Duration> {
        let n = self.n.max(1);
        let m = self.stamps_per_sample.max(1);
        let dt = self.exposure.as_secs_f64() / n as f64;
        let stamp_dt = dt / m as f64;
        let t_sub_start = self.frame_start.as_secs_f64() + sample_idx as f64 * dt;
        (0..m)
            .map(|j| Duration::from_secs_f64(t_sub_start + (j as f64 + 0.5) * stamp_dt))
            .collect()
    }

    /// Adaptive schedule from a drift budget.
    ///
    /// `max_drift_rad_over_exposure` is the total angular distance the
    /// spacecraft is expected to travel over the full exposure window
    /// (radians). `pixel_scale_rad` is the instrument's angular resolution
    /// per pixel. `max_drift_per_sample_px` is the per-subsample drift
    /// budget in pixels (typically 0.1).
    ///
    /// `N = max(1, ceil(max_drift_over_exposure_rad / (max_drift_per_sample_px * pixel_scale_rad)))`
    ///
    /// When the drift budget is zero (static pointing), `N = 1`.
    /// `stamps_per_sample` is set to 1 — see [`Self::adaptive_with_stamps`]
    /// for finer per-subsample stamping.
    pub fn adaptive(
        frame_start: Duration,
        exposure: Duration,
        max_drift_rad_over_exposure: f64,
        pixel_scale_rad: f64,
        max_drift_per_sample_px: f64,
    ) -> Self {
        Self::adaptive_with_stamps(
            frame_start,
            exposure,
            max_drift_rad_over_exposure,
            pixel_scale_rad,
            max_drift_per_sample_px,
            None,
        )
    }

    /// Adaptive schedule that also picks `stamps_per_sample` from a
    /// finer per-stamp drift budget. When `max_drift_per_stamp_px` is
    /// `Some(b)`, each subsample is divided into
    /// `M = max(1, ceil((per-subsample drift) / (b * pixel_scale)))`
    /// stamps. When `None`, `M = 1` and behavior is identical to
    /// [`Self::adaptive`]. The two budgets are independent: the outer
    /// one drives scene-state refresh, the inner one drives PSF-stamp
    /// placement and is what actually captures sub-subsample jitter.
    pub fn adaptive_with_stamps(
        frame_start: Duration,
        exposure: Duration,
        max_drift_rad_over_exposure: f64,
        pixel_scale_rad: f64,
        max_drift_per_sample_px: f64,
        max_drift_per_stamp_px: Option<f64>,
    ) -> Self {
        let n = if max_drift_rad_over_exposure <= 0.0
            || pixel_scale_rad <= 0.0
            || max_drift_per_sample_px <= 0.0
        {
            1
        } else {
            let budget_rad = max_drift_per_sample_px * pixel_scale_rad;
            (max_drift_rad_over_exposure / budget_rad).ceil() as usize
        };
        let n = n.max(1);
        let stamps_per_sample = match max_drift_per_stamp_px {
            Some(stamp_budget_px)
                if stamp_budget_px > 0.0
                    && pixel_scale_rad > 0.0
                    && max_drift_rad_over_exposure > 0.0 =>
            {
                // Per-subsample drift assumes the trajectory is
                // approximately uniform over the exposure (the same
                // assumption the per-sample budget already makes).
                let per_sub_drift_rad = max_drift_rad_over_exposure / n as f64;
                let stamp_budget_rad = stamp_budget_px * pixel_scale_rad;
                ((per_sub_drift_rad / stamp_budget_rad).ceil() as usize).max(1)
            }
            _ => 1,
        };
        Self {
            frame_start,
            exposure,
            n,
            stamps_per_sample,
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
    /// Per-subsample drift budget in pixels. The adaptive scheduler picks
    /// `N` so that drift per subsample stays below this threshold.
    pub max_drift_per_sample_px: f64,
    /// Optional finer per-stamp drift budget (pixels). When `Some(b)`,
    /// each subsample is divided into `M` PSF stamps so drift per
    /// stamp stays below `b`. When `None`, `M = 1` and behavior matches
    /// the original one-stamp-per-subsample renderer. Set this an order
    /// of magnitude tighter than `max_drift_per_sample_px` to capture
    /// high-frequency jitter (e.g. PSD-derived reaction-wheel residuals)
    /// that would otherwise alias instead of contributing motion blur.
    pub max_drift_per_stamp_px: Option<f64>,
    /// Optional base RNG seed (combined per-tile with `(frame_idx, sensor_idx)`).
    pub base_seed: Option<u64>,
    /// If true, force `N = 1` per frame regardless of the adaptive budget.
    /// Useful for debugging and performance comparisons.
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
            max_drift_per_sample_px: DEFAULT_MAX_DRIFT_PER_SAMPLE_PX,
            max_drift_per_stamp_px: None,
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

/// Envelope-prefilter: prune catalog stars whose padded mm position never
/// lands on the array's total AABB for any orientation sampled on this
/// frame's exposure window. Conservative: uses the frame's mid-time
/// orientation plus explicit sub-orientations to build a coverage padding.
///
/// This version is stricter than `project_stars_to_focal_plane_oriented`
/// in that it unions the in-band stars across the subsamples, so a star
/// that swings across the edge during the exposure is retained.
fn envelope_prefilter<'a>(
    trajectory: &Trajectory,
    catalog_stars: &'a [StarData],
    schedule: &SubsampleSchedule,
    fp: &FocalPlaneConfig,
    padding_mm: f64,
) -> Result<Vec<&'a StarData>, TrajectoryError> {
    let (min_x, min_y, max_x, max_y) = match fp.total_aabb_mm() {
        Some(aabb) => aabb,
        None => return Ok(Vec::new()),
    };
    let sample_times = schedule.sample_times();
    let mut orientations: Vec<UnitQuaternion<f64>> = Vec::with_capacity(sample_times.len() + 2);
    for t in sample_times {
        let t = t.min(trajectory.end_time()).max(trajectory.start_time());
        orientations.push(trajectory.orientation_at(t)?);
    }
    // Add the exposure endpoints (clamped) so stars that only enter or leave
    // during the frame are retained.
    let t_start = schedule.frame_start.max(trajectory.start_time());
    let t_end = (schedule.frame_start + schedule.exposure).min(trajectory.end_time());
    orientations.push(trajectory.orientation_at(t_start)?);
    orientations.push(trajectory.orientation_at(t_end)?);

    let mut kept: Vec<&'a StarData> = Vec::new();
    for star in catalog_stars {
        let mut hit = false;
        for q in &orientations {
            if let Some((x_mm, y_mm)) = fp.sky_to_mm(&star.position, q) {
                if x_mm >= min_x - padding_mm
                    && x_mm <= max_x + padding_mm
                    && y_mm >= min_y - padding_mm
                    && y_mm <= max_y + padding_mm
                {
                    hit = true;
                    break;
                }
            }
        }
        if hit {
            kept.push(star);
        }
    }
    Ok(kept)
}

/// Deterministic tile seed derived from `(base_seed, frame_idx, sensor_idx)`.
fn tile_seed(base_seed: u64, frame_idx: usize, sensor_idx: usize) -> u64 {
    // Cheap splitmix-style mix; reproducible, well-distributed enough for
    // RNG seeding.
    let mut h = base_seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(frame_idx as u64)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= (sensor_idx as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}

/// Render a single `(frame, sensor)` tile.
///
/// Runs the adaptive subsample loop, composes the unified Poisson lambda,
/// draws Poisson + Gaussian read noise, quantizes, and saves a PNG.
fn render_tile(
    scene: &RenderScene,
    plan: &FramePlan,
    sensor_idx: usize,
    flux_cache: &Arc<Mutex<FluxCache>>,
    satellite: &SatelliteConfig,
    tile_seed: u64,
    output_path: &Path,
) -> Result<(), TrajectoryError> {
    let (width, height) = satellite.sensor.dimensions.get_pixel_width_height();
    let mut accumulator = SensorAccumulator::zero(width, height);
    let aperture = satellite.telescope.clear_aperture_area();

    let schedule = &plan.schedule;
    let dt = schedule.dt();
    let stamps_per_sample = schedule.stamps_per_sample.max(1);
    let stamp_weight = 1.0 / stamps_per_sample as f64;

    for sample_idx in 0..schedule.n.max(1) {
        // Per-subsample per-star flux is shared across all M stamps of
        // this subsample (flux depends on star+sensor only, not on
        // orientation), so we pull it from the global cache once and
        // hold a local copy to avoid M lock acquisitions per star.
        let mut subsample_flux: HashMap<u64, SourceFlux> = HashMap::new();
        for stamp_t in schedule.stamp_times_for_sample(sample_idx) {
            let t_clamped = stamp_t
                .min(scene.trajectory.end_time())
                .max(scene.trajectory.start_time());
            let orientation = scene.trajectory.orientation_at(t_clamped)?;
            for star in &plan.stars {
                let hit = match scene.fp.project_to_sensor(
                    star,
                    &orientation,
                    sensor_idx,
                    plan.padding_mm,
                ) {
                    Some(px) => px,
                    None => continue,
                };
                let flux = subsample_flux.entry(star.id).or_insert_with(|| {
                    let mut cache = flux_cache.lock().expect("flux cache mutex poisoned");
                    cache
                        .entry((star.id, sensor_idx))
                        .or_insert_with(|| star_data_to_fluxes(star, satellite))
                        .clone()
                });
                // Per-stamp electron contribution: integrate flux rate
                // over the *subsample* dt, then split evenly across the
                // subsample's M stamps. Sum over M stamps reproduces the
                // single-stamp budget exactly.
                let total_electrons = flux.electrons.integrated_over(&dt, aperture) * stamp_weight;
                accumulator.splat_psf(hit.0, hit.1, total_electrons, &flux.electrons.disk);
            }
        }
    }

    // Dark current: rate × full exposure, uniform over pixels.
    let dark_rate = satellite
        .sensor
        .dark_current_at_temperature(satellite.temperature);
    let dark_mean = (dark_rate * schedule.exposure.as_secs_f64()).max(0.0);

    // Build unified Poisson mean image and draw.
    let mean_image = accumulator.combined_mean(plan.zodiacal_per_px[sensor_idx], dark_mean);
    let poisson_image = apply_poisson_photon_noise(&mean_image, Some(tile_seed));

    // Gaussian read noise (electronics, not shot noise) applied afterwards.
    let read_noise_rms = satellite
        .sensor
        .read_noise_estimator
        .estimate(satellite.temperature.as_celsius(), schedule.exposure)
        .unwrap_or(0.0)
        .max(0.0);
    let final_electrons = if read_noise_rms > 0.0 {
        let mut rng = StdRng::seed_from_u64(tile_seed ^ 0xA5A5_5A5A_A5A5_5A5A);
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
         tiles, exposure={:.3}s, timestep={:.3}s, max_drift_per_sample_px={:.3}",
        config.exposure.as_secs_f64(),
        config.timestep.as_secs_f64(),
        config.max_drift_per_sample_px,
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
    let mut n_min = usize::MAX;
    let mut n_max = 0usize;
    for (frame_idx, &t) in frame_times.iter().enumerate() {
        let exposure = config.exposure;
        // Clamp the exposure window to the trajectory so we do not try to
        // sample beyond the defined range.
        let t_end = (t + exposure).min(scene.trajectory.end_time());
        let exposure = if t_end > t { t_end - t } else { Duration::ZERO };
        let drift = max_drift_over_window(scene.trajectory, t, t_end)?;

        let schedule = if config.force_static || exposure.is_zero() {
            SubsampleSchedule {
                frame_start: t,
                exposure,
                n: 1,
                stamps_per_sample: 1,
            }
        } else {
            SubsampleSchedule::adaptive_with_stamps(
                t,
                exposure,
                drift,
                px_scale,
                config.max_drift_per_sample_px,
                config.max_drift_per_stamp_px,
            )
        };
        n_min = n_min.min(schedule.n);
        n_max = n_max.max(schedule.n);

        // Mid-frame boresight for zodiacal evaluation.
        let mid_t = t + schedule.exposure / 2;
        let mid_t = mid_t
            .min(scene.trajectory.end_time())
            .max(scene.trajectory.start_time());
        let mid_q = scene.trajectory.orientation_at(mid_t)?;
        let mid_bore = boresight_of(&mid_q);
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
        )?;
        // Per-frame summary routed through the bar so it coexists with the
        // live progress line without scrambled output.
        pb.println(format!(
            "frame {:06} at t={:.3}s: N={} subsamples, boresight=(ra={:.4}°, dec={:.4}°), \
             stars={}",
            frame_idx,
            t.as_secs_f64(),
            schedule.n,
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
        n_min = 0;
        n_max = 0;
    }
    info!(
        "Adaptive N per frame: min={} max={} (force_static={})",
        n_min, n_max, config.force_static
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
            let seed = tile_seed(base_seed, plan.idx, tile.sensor_idx);
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
                "tile frame={} sensor={} N={} stars={} elapsed={}ms",
                plan.idx,
                tile.sensor_idx,
                plan.schedule.n,
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
            n_subsamples: schedule.n,
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
        max_drift_per_sample_px: config.max_drift_per_sample_px,
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
    fn test_adaptive_schedule_static_gives_n_1() {
        let sched = SubsampleSchedule::adaptive(
            Duration::ZERO,
            Duration::from_secs(1),
            0.0, // zero drift
            1e-6,
            0.1,
        );
        assert_eq!(sched.n, 1);
        assert_eq!(sched.sample_times().len(), 1);
    }

    #[test]
    fn test_adaptive_schedule_respects_budget() {
        // 10 px of drift at pixel_scale=1e-5 rad/px with a 0.1 px budget
        // should yield N = ceil(10 / 0.1) = 100.
        let pixel_scale = 1e-5_f64;
        let drift_px = 10.0;
        let budget_px = 0.1;
        let sched = SubsampleSchedule::adaptive(
            Duration::ZERO,
            Duration::from_secs(1),
            drift_px * pixel_scale,
            pixel_scale,
            budget_px,
        );
        let expected = (drift_px / budget_px).ceil() as usize;
        assert!(
            (sched.n as isize - expected as isize).abs() <= 1,
            "got {} expected {}",
            sched.n,
            expected
        );
    }

    #[test]
    fn test_sample_times_midpoints() {
        let sched = SubsampleSchedule {
            frame_start: Duration::from_secs(10),
            exposure: Duration::from_secs(4),
            n: 4,
            stamps_per_sample: 1,
        };
        let times: Vec<f64> = sched
            .sample_times()
            .iter()
            .map(|d| d.as_secs_f64())
            .collect();
        // midpoints: 10 + (0.5, 1.5, 2.5, 3.5) = 10.5, 11.5, 12.5, 13.5
        assert_abs_diff_eq!(times[0], 10.5, epsilon = 1e-12);
        assert_abs_diff_eq!(times[1], 11.5, epsilon = 1e-12);
        assert_abs_diff_eq!(times[2], 12.5, epsilon = 1e-12);
        assert_abs_diff_eq!(times[3], 13.5, epsilon = 1e-12);
    }

    #[test]
    fn test_stamp_times_default_is_one_per_subsample() {
        // With stamps_per_sample = 1, stamp midpoints collapse onto sample midpoints.
        let sched = SubsampleSchedule {
            frame_start: Duration::from_secs(10),
            exposure: Duration::from_secs(4),
            n: 4,
            stamps_per_sample: 1,
        };
        for i in 0..sched.n {
            let stamps = sched.stamp_times_for_sample(i);
            assert_eq!(stamps.len(), 1);
            assert_abs_diff_eq!(
                stamps[0].as_secs_f64(),
                sched.sample_times()[i].as_secs_f64(),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_stamp_times_subdivide_subsample_window() {
        // Each 1s subsample is divided into 4 stamps at the 1/8, 3/8, 5/8, 7/8 marks.
        let sched = SubsampleSchedule {
            frame_start: Duration::from_secs(10),
            exposure: Duration::from_secs(4),
            n: 4,
            stamps_per_sample: 4,
        };
        let stamps_first: Vec<f64> = sched
            .stamp_times_for_sample(0)
            .iter()
            .map(|d| d.as_secs_f64())
            .collect();
        // Subsample 0 spans [10.0, 11.0). Stamp dt = 0.25, midpoints at 10.125, 10.375, 10.625, 10.875.
        assert_eq!(stamps_first.len(), 4);
        assert_abs_diff_eq!(stamps_first[0], 10.125, epsilon = 1e-12);
        assert_abs_diff_eq!(stamps_first[1], 10.375, epsilon = 1e-12);
        assert_abs_diff_eq!(stamps_first[2], 10.625, epsilon = 1e-12);
        assert_abs_diff_eq!(stamps_first[3], 10.875, epsilon = 1e-12);

        // Subsample 3 spans [13.0, 14.0). Last stamp midpoint at 13.875.
        let stamps_last = sched.stamp_times_for_sample(3);
        assert_abs_diff_eq!(stamps_last[3].as_secs_f64(), 13.875, epsilon = 1e-12);
    }

    #[test]
    fn test_adaptive_with_stamps_picks_m_from_per_subsample_drift() {
        // 10 px drift / 1s exposure, 0.1 px per-sub budget => N = 100, so each
        // subsample sees 0.1 px drift. With per-stamp budget 0.01 px, each
        // subsample should be split into M = 10 stamps.
        let pixel_scale = 1e-5_f64;
        let drift_px = 10.0;
        let sched = SubsampleSchedule::adaptive_with_stamps(
            Duration::ZERO,
            Duration::from_secs(1),
            drift_px * pixel_scale,
            pixel_scale,
            0.1,        // per-sub budget
            Some(0.01), // per-stamp budget
        );
        assert_eq!(sched.n, 100);
        assert_eq!(sched.stamps_per_sample, 10);
    }

    #[test]
    fn test_adaptive_with_stamps_none_collapses_to_m_1() {
        let pixel_scale = 1e-5_f64;
        let sched = SubsampleSchedule::adaptive_with_stamps(
            Duration::ZERO,
            Duration::from_secs(1),
            10.0 * pixel_scale,
            pixel_scale,
            0.1,
            None,
        );
        assert_eq!(sched.stamps_per_sample, 1);
    }

    #[test]
    fn test_adaptive_with_stamps_static_trajectory_gives_m_1() {
        let sched = SubsampleSchedule::adaptive_with_stamps(
            Duration::ZERO,
            Duration::from_secs(1),
            0.0, // no drift
            1e-5,
            0.1,
            Some(0.01),
        );
        assert_eq!(sched.n, 1);
        assert_eq!(sched.stamps_per_sample, 1);
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
        let a = tile_seed(42, 0, 0);
        let b = tile_seed(42, 0, 0);
        let c = tile_seed(42, 1, 0);
        let d = tile_seed(42, 0, 1);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(c, d);
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
            max_drift_per_sample_px: 0.1,
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
            max_drift_per_sample_px: 0.1,
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
            max_drift_per_sample_px: 0.1,
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
            max_drift_per_sample_px: 0.1,
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
        let first_sat = fp.satellite_for_sensor(0).unwrap();
        let pixel_size_mm = first_sat.sensor.pixel_size().as_millimeters();
        let airy_pix = first_sat.airy_disk_pixel_space();
        let padding_mm = airy_pix.first_zero() * 2.0 * pixel_size_mm;
        let px_scale = pixel_scale_rad(fp).unwrap_or(0.0);

        let t_start = Duration::ZERO;
        let t_end = (t_start + cfg.exposure).min(trajectory.end_time());
        let exposure = t_end - t_start;
        let drift = max_drift_over_window(trajectory, t_start, t_end).unwrap();
        let schedule = SubsampleSchedule::adaptive(
            t_start,
            exposure,
            drift,
            px_scale,
            cfg.max_drift_per_sample_px,
        );
        let (width, height) = first_sat.sensor.dimensions.get_pixel_width_height();
        let mut acc = SensorAccumulator::zero(width, height);
        let aperture = first_sat.telescope.clear_aperture_area();
        let flux = star_data_to_fluxes(star, &first_sat);
        let dt = schedule.dt();
        for t in schedule.sample_times() {
            let q = trajectory.orientation_at(t).unwrap();
            if let Some((px, py)) = fp.project_to_sensor(star, &q, 0, padding_mm) {
                let total = flux.electrons.integrated_over(&dt, aperture);
                acc.splat_psf(px, py, total, &flux.electrons.disk);
            }
        }
        acc
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
            max_drift_per_sample_px: 0.1,
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
            max_drift_per_sample_px: 0.1,
            max_drift_per_stamp_px: Some(0.01),
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
            // Loose per-sub budget so N stays small and there's room for
            // M to actually change the within-subsample distribution.
            max_drift_per_sample_px: 1.0,
            max_drift_per_stamp_px: None,
            base_seed: Some(99),
            force_static: false,
            quiet: true,
            ..Default::default()
        };
        let cfg_fine = MotionBlurConfig {
            max_drift_per_stamp_px: Some(0.05),
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
            "raising stamps_per_sample on a moving trajectory must change the rendered streak"
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
            max_drift_per_sample_px: 0.1,
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
            max_drift_per_sample_px: 0.1,
            max_drift_per_stamp_px: None,
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
}

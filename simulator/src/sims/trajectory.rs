use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use nalgebra::UnitQuaternion;
use starfield::catalogs::{StarCatalog, StarData};
use starfield::Equatorial;
use thiserror::Error;

use crate::hardware::satellite::FocalPlaneConfig;
use crate::image_proc::render::{StarInFocalPlane, StarInFrame};
use crate::photometry::photoconversion::SourceFlux;
use crate::sims::motion_blur::{
    render_motion_trajectory, LightSources, MotionBlurConfig, DEFAULT_MAX_DRIFT_PER_STAMP_PX,
};
use crate::sims::orientation::{boresight_of, orientation_from_pointing};
use crate::star_math::star_data_to_fluxes;

pub use crate::sims::motion_blur::render_one_frame;

#[derive(Debug, Error)]
pub enum TrajectoryError {
    #[error("trajectory must have at least 2 waypoints, got {0}")]
    TooFewWaypoints(usize),

    #[error("waypoint times must be strictly monotonically increasing (index {index}: {earlier:?} >= {later:?})")]
    NonMonotonicTimes {
        index: usize,
        earlier: Duration,
        later: Duration,
    },

    #[error("time {0:?} is outside trajectory range [{1:?}, {2:?}]")]
    TimeOutOfRange(Duration, Duration, Duration),

    #[error("period {period:?} is shorter than trajectory span {span:?}")]
    PeriodTooShort { period: Duration, span: Duration },

    #[error("focal plane has no sensors")]
    NoSensors,

    #[error("ROI {roi:?} out of bounds for sensor {sensor_idx} ({width}x{height})")]
    RoiOutOfBounds {
        roi: (usize, usize, usize, usize),
        sensor_idx: usize,
        width: usize,
        height: usize,
    },

    #[error("image write error: {0}")]
    ImageWrite(String),
}

#[derive(Debug, Clone)]
pub struct Waypoint {
    pub time: Duration,
    pub orientation: UnitQuaternion<f64>,
}

impl Waypoint {
    /// Construct a waypoint from an explicit orientation.
    pub fn new(time: Duration, orientation: UnitQuaternion<f64>) -> Self {
        Self { time, orientation }
    }

    /// Construct a waypoint from a pointing with zero roll.
    pub fn from_pointing(time: Duration, pointing: Equatorial) -> Self {
        Self {
            time,
            orientation: orientation_from_pointing(&pointing, 0.0),
        }
    }

    /// Construct a waypoint from a pointing and an explicit roll angle (radians).
    pub fn from_pointing_and_roll(time: Duration, pointing: Equatorial, roll_rad: f64) -> Self {
        Self {
            time,
            orientation: orientation_from_pointing(&pointing, roll_rad),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Trajectory {
    waypoints: Vec<Waypoint>,
    /// When `Some`, calls to [`Trajectory::orientation_at`] treat `t` modulo
    /// `period`. The implicit wrap segment from `end_time()` to
    /// `start_time() + period` SLERPs from the last stored waypoint's
    /// orientation back to the first stored waypoint's orientation, so
    /// `orientation_at(start) == orientation_at(start + period)` by
    /// construction.
    period: Option<Duration>,
}

impl Trajectory {
    pub fn new(waypoints: Vec<Waypoint>) -> Result<Self, TrajectoryError> {
        if waypoints.len() < 2 {
            return Err(TrajectoryError::TooFewWaypoints(waypoints.len()));
        }
        for i in 1..waypoints.len() {
            if waypoints[i].time <= waypoints[i - 1].time {
                return Err(TrajectoryError::NonMonotonicTimes {
                    index: i,
                    earlier: waypoints[i - 1].time,
                    later: waypoints[i].time,
                });
            }
        }
        Ok(Self {
            waypoints,
            period: None,
        })
    }

    /// Build a two-waypoint trajectory between two pointings, both with zero roll.
    pub fn from_endpoints(
        start: Equatorial,
        end: Equatorial,
        duration: Duration,
    ) -> Result<Self, TrajectoryError> {
        Self::new(vec![
            Waypoint::from_pointing(Duration::ZERO, start),
            Waypoint::from_pointing(duration, end),
        ])
    }

    /// Build a two-waypoint trajectory between two pointings with explicit
    /// start/end roll angles (radians).
    pub fn from_endpoints_with_roll(
        start: Equatorial,
        start_roll_rad: f64,
        end: Equatorial,
        end_roll_rad: f64,
        duration: Duration,
    ) -> Result<Self, TrajectoryError> {
        Self::new(vec![
            Waypoint::from_pointing_and_roll(Duration::ZERO, start, start_roll_rad),
            Waypoint::from_pointing_and_roll(duration, end, end_roll_rad),
        ])
    }

    pub fn start_time(&self) -> Duration {
        self.waypoints.first().unwrap().time
    }

    pub fn end_time(&self) -> Duration {
        self.waypoints.last().unwrap().time
    }

    pub fn waypoints(&self) -> &[Waypoint] {
        &self.waypoints
    }

    /// Trajectory span = `end_time() - start_time()`. Convenience.
    pub fn duration(&self) -> Duration {
        self.end_time() - self.start_time()
    }

    /// Period after which the trajectory repeats, if marked periodic via
    /// [`Trajectory::with_period`] or [`Trajectory::looped`]. `None` for
    /// the default open-ended form.
    pub fn period(&self) -> Option<Duration> {
        self.period
    }

    /// Mark the trajectory as periodic with the given period.
    ///
    /// `period` must be at least the trajectory's span
    /// (`end_time() - start_time()`). Any excess
    /// (`period - span`) becomes the implicit wrap segment over which
    /// [`orientation_at`] SLERPs from the last waypoint back to the
    /// first. Setting `period` exactly equal to the span produces a
    /// zero-length wrap and is only appropriate when the first and last
    /// waypoint orientations already agree.
    pub fn with_period(mut self, period: Duration) -> Result<Self, TrajectoryError> {
        let span = self.duration();
        if period < span {
            return Err(TrajectoryError::PeriodTooShort { period, span });
        }
        self.period = Some(period);
        Ok(self)
    }

    /// Convenience: mark the trajectory periodic, picking the period so
    /// the implicit wrap segment has the same duration as the
    /// trajectory's first stored segment.
    ///
    /// Equivalent to
    /// `self.with_period(self.duration() + (waypoints[1].time - waypoints[0].time))`,
    /// which guarantees the wrap segment is non-degenerate and visually
    /// matches the local sampling cadence at the seam. The looping
    /// semantics live entirely inside [`orientation_at`] — no
    /// post-processed smoothing or pre-baked extra waypoints are
    /// introduced.
    pub fn looped(self) -> Self {
        let first_dt = self.waypoints[1].time - self.waypoints[0].time;
        let period = self.duration() + first_dt;
        self.with_period(period)
            .expect("looped(): period == span + first_dt > span by construction")
    }

    /// Interpolate the spacecraft orientation at a given time using
    /// quaternion SLERP between bracketing waypoints.
    ///
    /// If [`with_period`] / [`looped`] has been called, `t` is taken
    /// modulo the period; out-of-range times are then never an error.
    /// When the wrapped time falls in the implicit wrap segment between
    /// the last stored waypoint and `start_time() + period`, the result
    /// is a SLERP from the last waypoint's orientation back to the
    /// first waypoint's orientation.
    pub fn orientation_at(&self, t: Duration) -> Result<UnitQuaternion<f64>, TrajectoryError> {
        let start = self.start_time();
        let end = self.end_time();

        let t_in_span = if let Some(period) = self.period {
            let period_s = period.as_secs_f64();
            if period_s <= 0.0 {
                return Ok(self.waypoints[0].orientation);
            }
            let rel = (t.as_secs_f64() - start.as_secs_f64()).rem_euclid(period_s);
            let span_s = self.duration().as_secs_f64();
            if rel <= span_s {
                start + Duration::from_secs_f64(rel)
            } else {
                // Wrap segment: SLERP from last waypoint to first.
                let wrap_dur_s = period_s - span_s;
                let frac = (rel - span_s) / wrap_dur_s;
                let last = self.waypoints.last().unwrap().orientation;
                let first = self.waypoints[0].orientation;
                return Ok(last.slerp(&first, frac));
            }
        } else {
            if t < start || t > end {
                return Err(TrajectoryError::TimeOutOfRange(t, start, end));
            }
            t
        };

        let seg_idx = self
            .waypoints
            .windows(2)
            .position(|w| t_in_span >= w[0].time && t_in_span <= w[1].time)
            .unwrap_or(self.waypoints.len() - 2);

        let w0 = &self.waypoints[seg_idx];
        let w1 = &self.waypoints[seg_idx + 1];

        let seg_duration = (w1.time - w0.time).as_secs_f64();
        if seg_duration == 0.0 {
            return Ok(w0.orientation);
        }
        let frac = (t_in_span - w0.time).as_secs_f64() / seg_duration;

        Ok(w0.orientation.slerp(&w1.orientation, frac))
    }

    /// Interpolate the boresight pointing at a given time.
    ///
    /// Convenience wrapper around [`Trajectory::orientation_at`] that
    /// extracts the boresight direction from the interpolated orientation.
    pub fn pointing_at(&self, t: Duration) -> Result<Equatorial, TrajectoryError> {
        let q = self.orientation_at(t)?;
        Ok(boresight_of(&q))
    }

    /// Maximum angular distance (radians) between any waypoint
    /// orientation in `[t_start, t_end]` and a chosen `reference`
    /// orientation. Scans every waypoint inside the window plus the
    /// two SLERP-interpolated boundary orientations; returns the
    /// largest deviation.
    ///
    /// This is the *peak excursion* of the trajectory relative to the
    /// reference, and is the right padding budget for an envelope
    /// prefilter that only projects stars at the reference orientation
    /// — it tells you how far the focal-plane AABB must be inflated to
    /// catch any star whose excursion brings it onto the sensor at
    /// some moment during the window.
    ///
    /// Distinct from [`Self::pointing_at`] etc.: returns a *bound on
    /// motion* over the window, not an instantaneous orientation.
    pub fn peak_excursion_rad(
        &self,
        t_start: Duration,
        t_end: Duration,
        reference: &UnitQuaternion<f64>,
    ) -> Result<f64, TrajectoryError> {
        if t_end <= t_start {
            return Ok(0.0);
        }
        let lo = t_start.max(self.start_time());
        let hi = t_end.min(self.end_time());
        if hi <= lo {
            return Ok(0.0);
        }
        let mut peak = 0.0_f64;
        let q_lo = self.orientation_at(lo)?;
        let q_hi = self.orientation_at(hi)?;
        peak = peak.max(reference.angle_to(&q_lo));
        peak = peak.max(reference.angle_to(&q_hi));
        for wp in &self.waypoints {
            if wp.time <= lo {
                continue;
            }
            if wp.time >= hi {
                break;
            }
            peak = peak.max(reference.angle_to(&wp.orientation));
        }
        Ok(peak)
    }

    /// Generate evenly spaced frame times from start to end. Each
    /// emitted time is a frame *start* — the renderer integrates from
    /// `t` forward up to `t + exposure`, clamped to `end_time()`. We
    /// stop strictly before `end_time()` so we never emit a frame
    /// whose entire exposure window collapses to zero against the
    /// upper clamp (that produced a black tail frame in every render).
    pub fn frame_times(&self, timestep: Duration) -> Vec<Duration> {
        let mut times = Vec::new();
        let mut t = self.start_time();
        let end = self.end_time();
        while t < end {
            times.push(t);
            t += timestep;
        }
        times
    }
}

/// Convert Equatorial (ra, dec in radians) to unit 3D vector.
fn equatorial_to_xyz(eq: &Equatorial) -> [f64; 3] {
    let cos_dec = eq.dec.cos();
    [cos_dec * eq.ra.cos(), cos_dec * eq.ra.sin(), eq.dec.sin()]
}

/// Convert unit 3D vector back to Equatorial.
fn xyz_to_equatorial(xyz: [f64; 3]) -> Equatorial {
    let [x, y, z] = xyz;
    let ra = y.atan2(x);
    let dec = z.atan2((x * x + y * y).sqrt());
    // Normalize RA to [0, 2pi)
    let ra = if ra < 0.0 {
        ra + 2.0 * std::f64::consts::PI
    } else {
        ra
    };
    Equatorial::new(ra, dec)
}

/// Angular distance in degrees between two Equatorial pointings (haversine).
fn angular_distance_deg(a: &Equatorial, b: &Equatorial) -> f64 {
    let va = equatorial_to_xyz(a);
    let vb = equatorial_to_xyz(b);
    let dot = (va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]).clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

/// Compute the FOV envelope for a trajectory: a single (center, diameter_deg) that
/// encompasses all pointings plus the base instrument FOV.
pub fn fov_envelope(trajectory: &Trajectory, base_fov_deg: f64) -> (Equatorial, f64) {
    // Average unit vectors to find centroid
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    let n = trajectory.waypoints.len() as f64;
    let pointings: Vec<Equatorial> = trajectory
        .waypoints
        .iter()
        .map(|wp| boresight_of(&wp.orientation))
        .collect();
    for p in &pointings {
        let [x, y, z] = equatorial_to_xyz(p);
        cx += x;
        cy += y;
        cz += z;
    }
    cx /= n;
    cy /= n;
    cz /= n;

    // Normalize
    let mag = (cx * cx + cy * cy + cz * cz).sqrt();
    if mag < 1e-15 {
        // Degenerate: pointings span a half-sphere. Use first waypoint as center.
        let center = pointings[0];
        let max_dist = pointings
            .iter()
            .map(|p| angular_distance_deg(&center, p))
            .fold(0.0f64, f64::max);
        return (center, 2.0 * max_dist + base_fov_deg);
    }

    let center = xyz_to_equatorial([cx / mag, cy / mag, cz / mag]);
    let max_dist = pointings
        .iter()
        .map(|p| angular_distance_deg(&center, p))
        .fold(0.0f64, f64::max);

    (center, 2.0 * max_dist + base_fov_deg)
}

/// Prefetch all catalog stars that fall inside the given FOV envelope.
///
/// `envelope` is the `(center, diameter_deg)` pair returned by
/// [`fov_envelope`]: the first element is the envelope's pointing centre
/// and the second is its total angular diameter in degrees (so the
/// included region is a cone of half-angle `diameter_deg / 2` around the
/// centre).
///
/// The returned `Vec<StarData>` owns its rows and outlives the catalog
/// reference, so callers (notably `tracking-test-bench`'s `scene-camera`
/// crate) can wrap it in an `Arc` once per `SceneCamera::open()` and
/// reuse it across thousands of render calls without re-reading the
/// catalog or holding a borrow against it.
///
/// `StarCatalog` is not object-safe (associated `Star` type, `impl
/// Iterator` return positions) so this function takes a generic catalog
/// reference rather than the `&dyn StarCatalog` form sketched in the
/// originating issue.
pub fn prefetch_catalog_for_envelope<C: StarCatalog + ?Sized>(
    catalog: &C,
    envelope: (Equatorial, f64),
) -> Vec<StarData> {
    let (center, diameter_deg) = envelope;
    catalog.stars_in_field(center.ra_degrees(), center.dec_degrees(), diameter_deg)
}

/// Route focal-plane stars to sensors with a flux cache to avoid redundant computation.
///
/// Identical to `route_stars_to_sensors` in render.rs but caches `star_data_to_fluxes()`
/// results keyed by `(star_id, sensor_index)`. Retained as a public helper for
/// external consumers that still want the single-orientation scene-style
/// routing; the new motion-blur renderer uses its own cache-carrying tile
/// worker instead.
pub fn route_stars_to_sensors_cached(
    fp_stars: &[StarInFocalPlane],
    focal_plane: &FocalPlaneConfig,
    padding_mm: f64,
    flux_cache: &mut HashMap<(u64, usize), SourceFlux>,
) -> Vec<Vec<StarInFrame>> {
    let sensor_count = focal_plane.array.sensor_count();
    let mut per_sensor_stars: Vec<Vec<StarInFrame>> =
        (0..sensor_count).map(|_| Vec::new()).collect();

    let satellites: Vec<_> = (0..sensor_count)
        .filter_map(|i| focal_plane.satellite_for_sensor(i))
        .collect();

    for fp_star in fp_stars {
        let hits = focal_plane
            .array
            .mm_to_pixels_padded(fp_star.x_mm, fp_star.y_mm, padding_mm);
        for (pixel_x, pixel_y, sensor_idx) in hits {
            let key = (fp_star.star.id, sensor_idx);
            let flux = flux_cache
                .entry(key)
                .or_insert_with(|| star_data_to_fluxes(&fp_star.star, &satellites[sensor_idx]))
                .clone();
            per_sensor_stars[sensor_idx].push(StarInFrame {
                x: pixel_x,
                y: pixel_y,
                spot: flux,
                star: fp_star.star,
            });
        }
    }

    for stars in &mut per_sensor_stars {
        stars.sort_by(|a, b| {
            a.spot
                .electrons
                .flux
                .partial_cmp(&b.spot.electrons.flux)
                .unwrap()
        });
    }

    per_sensor_stars
}

/// Configuration for rendering a trajectory sequence.
///
/// Thin compatibility wrapper over [`MotionBlurConfig`]. Existing call sites
/// continue to use this type; internally `render_trajectory` delegates to
/// the motion-blur path with the adaptive subsample scheduler.
pub struct TrajectoryRenderConfig<'a> {
    /// Trajectory defining the pointing over time.
    pub trajectory: &'a Trajectory,
    /// Time between frames.
    pub timestep: Duration,
    /// Exposure duration per frame.
    pub exposure: Duration,
    /// Focal plane hardware configuration.
    pub focal_plane: &'a FocalPlaneConfig,
    /// All light contributing to the rendered frames: foreground stars,
    /// pre-projected per-sensor galaxies, and diffuse zodiacal background.
    /// See [`LightSources`] and `crate::sims::nsa_galaxies` for the
    /// builder that produces the galaxy shape from an NSA FITS file.
    pub sources: LightSources<'a>,
    /// Directory to write 16-bit PNG output frames.
    pub output_dir: &'a Path,
    /// Optional RNG seed for reproducible noise.
    pub base_seed: Option<u64>,
    /// Optional override for the per-stamp drift budget (pixels). Total
    /// stamps per exposure is derived from this and the trajectory's
    /// per-frame angular path length. Defaults to
    /// [`crate::sims::motion_blur::DEFAULT_MAX_DRIFT_PER_STAMP_PX`].
    pub max_drift_per_stamp_px: Option<f64>,
    /// Force a single stamp per frame regardless of drift. Useful for
    /// debugging.
    pub force_static: bool,
    /// Suppress the indicatif progress bar during rendering.
    pub quiet: bool,
    /// Telescope display name recorded in `metadata.json`. Metadata-only.
    pub telescope_name: String,
    /// Catalog path recorded in `metadata.json`. Metadata-only.
    pub catalog_path: std::path::PathBuf,
    /// Operating temperature (Celsius) recorded in `metadata.json`.
    /// Metadata-only.
    pub temperature_c: f64,
}

/// Render a sequence of frames along a trajectory, writing 16-bit PNG files.
///
/// Delegates to [`render_motion_trajectory`], which integrates an adaptive
/// number of sub-orientations across each frame's exposure. When the drift
/// budget yields `N = 1` (static or near-static trajectory), the output is
/// equivalent to a single-orientation render, except that the photon, zodi,
/// and dark-current means share a single unified Poisson draw rather than
/// three separate ones.
///
/// Returns the total number of frames rendered.
pub fn render_trajectory(config: &TrajectoryRenderConfig) -> Result<usize, TrajectoryError> {
    let motion_cfg = MotionBlurConfig {
        timestep: config.timestep,
        exposure: config.exposure,
        max_drift_per_stamp_px: config
            .max_drift_per_stamp_px
            .unwrap_or(DEFAULT_MAX_DRIFT_PER_STAMP_PX),
        base_seed: config.base_seed,
        force_static: config.force_static,
        quiet: config.quiet,
        telescope_name: config.telescope_name.clone(),
        catalog_path: config.catalog_path.clone(),
        temperature_c: config.temperature_c,
    };
    render_motion_trajectory(
        config.trajectory,
        &config.sources,
        config.focal_plane,
        &motion_cfg,
        config.output_dir,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sims::orientation::roll_of;
    use approx::assert_abs_diff_eq;
    use std::time::Duration;

    fn make_pointing(ra_deg: f64, dec_deg: f64) -> Equatorial {
        Equatorial::from_degrees(ra_deg, dec_deg)
    }

    #[test]
    fn test_trajectory_requires_two_waypoints() {
        let result = Trajectory::new(vec![Waypoint::from_pointing(
            Duration::ZERO,
            make_pointing(0.0, 0.0),
        )]);
        assert!(result.is_err());
    }

    #[test]
    fn test_trajectory_rejects_non_monotonic() {
        let result = Trajectory::new(vec![
            Waypoint::from_pointing(Duration::from_secs(5), make_pointing(0.0, 0.0)),
            Waypoint::from_pointing(Duration::from_secs(3), make_pointing(10.0, 10.0)),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pointing_at_returns_endpoints() {
        let traj = Trajectory::from_endpoints(
            make_pointing(10.0, 20.0),
            make_pointing(20.0, 30.0),
            Duration::from_secs(10),
        )
        .unwrap();

        let start = traj.pointing_at(Duration::ZERO).unwrap();
        assert_abs_diff_eq!(start.ra_degrees(), 10.0, epsilon = 1e-9);
        assert_abs_diff_eq!(start.dec_degrees(), 20.0, epsilon = 1e-9);

        let end = traj.pointing_at(Duration::from_secs(10)).unwrap();
        assert_abs_diff_eq!(end.ra_degrees(), 20.0, epsilon = 1e-9);
        assert_abs_diff_eq!(end.dec_degrees(), 30.0, epsilon = 1e-9);
    }

    #[test]
    fn test_pointing_at_midpoint_is_between() {
        let traj = Trajectory::from_endpoints(
            make_pointing(10.0, 20.0),
            make_pointing(20.0, 20.0),
            Duration::from_secs(10),
        )
        .unwrap();
        let mid = traj.pointing_at(Duration::from_secs(5)).unwrap();

        assert!(mid.ra_degrees() > 10.0 && mid.ra_degrees() < 20.0);
        assert_abs_diff_eq!(mid.dec_degrees(), 20.0, epsilon = 0.5);
    }

    #[test]
    fn test_pointing_at_rejects_out_of_range() {
        let traj = Trajectory::from_endpoints(
            make_pointing(10.0, 20.0),
            make_pointing(20.0, 30.0),
            Duration::from_secs(10),
        )
        .unwrap();

        assert!(traj.pointing_at(Duration::from_secs(11)).is_err());
    }

    #[test]
    fn test_frame_times() {
        let traj = Trajectory::from_endpoints(
            make_pointing(0.0, 0.0),
            make_pointing(10.0, 0.0),
            Duration::from_secs(10),
        )
        .unwrap();

        // end_time is excluded so we never emit a frame whose exposure
        // window collapses against the upper clamp.
        let times = traj.frame_times(Duration::from_secs(2));
        assert_eq!(times.len(), 5); // 0, 2, 4, 6, 8 (10 is end_time, excluded)
        assert_eq!(times[0], Duration::ZERO);
        assert_eq!(times[4], Duration::from_secs(8));
    }

    #[test]
    fn test_frame_times_excludes_end() {
        // Regression: a 10s trajectory at 1s timestep used to return 11
        // entries (0..=10) with the 10s entry rendering as a black tile
        // because (end - end) = 0 exposure.
        let traj = Trajectory::from_endpoints(
            make_pointing(0.0, 0.0),
            make_pointing(10.0, 0.0),
            Duration::from_secs(10),
        )
        .unwrap();
        let times = traj.frame_times(Duration::from_secs(1));
        assert_eq!(times.len(), 10);
        assert_eq!(*times.last().unwrap(), Duration::from_secs(9));
        assert!(times.iter().all(|t| *t < traj.end_time()));
    }

    #[test]
    fn test_peak_excursion_static_is_zero() {
        let p = make_pointing(45.0, 0.0);
        let traj = Trajectory::from_endpoints(p, p, Duration::from_secs(10)).unwrap();
        let q_mid = traj.orientation_at(Duration::from_secs(5)).unwrap();
        let exc = traj
            .peak_excursion_rad(Duration::ZERO, Duration::from_secs(10), &q_mid)
            .unwrap();
        assert!(
            exc < 1e-12,
            "static trajectory peak excursion = {exc} (expected ~0)"
        );
    }

    #[test]
    fn test_peak_excursion_linear_is_half_total_drift() {
        // Linear sweep from RA 0 to RA 1 deg over 10 s. The mid-time
        // orientation is at RA 0.5 deg, so the maximum deviation from
        // mid-time is at the endpoints — half the total angular sweep.
        let traj = Trajectory::from_endpoints(
            make_pointing(0.0, 0.0),
            make_pointing(1.0, 0.0),
            Duration::from_secs(10),
        )
        .unwrap();
        let q_mid = traj.orientation_at(Duration::from_secs(5)).unwrap();
        let exc = traj
            .peak_excursion_rad(Duration::ZERO, Duration::from_secs(10), &q_mid)
            .unwrap();
        let expected = 0.5_f64.to_radians();
        assert_abs_diff_eq!(exc, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_peak_excursion_clamps_to_trajectory_window() {
        let traj = Trajectory::from_endpoints(
            make_pointing(0.0, 0.0),
            make_pointing(1.0, 0.0),
            Duration::from_secs(10),
        )
        .unwrap();
        let q_mid = traj.orientation_at(Duration::from_secs(5)).unwrap();
        // Window beyond end: returns 0 because hi-lo collapses
        let exc = traj
            .peak_excursion_rad(Duration::from_secs(20), Duration::from_secs(30), &q_mid)
            .unwrap();
        assert_eq!(exc, 0.0);
    }

    #[test]
    fn test_fov_envelope_covers_trajectory() {
        let traj = Trajectory::from_endpoints(
            make_pointing(10.0, 0.0),
            make_pointing(20.0, 0.0),
            Duration::from_secs(10),
        )
        .unwrap();

        let base_fov = 1.0;
        let (center, diameter) = fov_envelope(&traj, base_fov);

        // Center should be near RA=15, Dec=0
        assert_abs_diff_eq!(center.ra_degrees(), 15.0, epsilon = 1.0);
        assert!(center.dec_degrees().abs() < 1.0);

        // Diameter should cover the 10 degree span plus the base FOV
        assert!(diameter >= 10.0 + base_fov);
    }

    #[test]
    fn test_angular_distance() {
        let a = make_pointing(0.0, 0.0);
        let b = make_pointing(90.0, 0.0);
        let dist = angular_distance_deg(&a, &b);
        assert_abs_diff_eq!(dist, 90.0, epsilon = 0.01);
    }

    #[test]
    fn test_angular_distance_same_point() {
        let a = make_pointing(45.0, 30.0);
        let dist = angular_distance_deg(&a, &a);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_slerp_with_roll_midpoint() {
        // Hold the pointing fixed so SLERP reduces to a pure roll rotation
        // about the boresight; the midpoint roll must then be exactly
        // halfway between the endpoint rolls.
        let pointing = make_pointing(10.0, 20.0);
        let start_roll: f64 = 0.2;
        let end_roll: f64 = 0.8;

        let traj = Trajectory::from_endpoints_with_roll(
            pointing,
            start_roll,
            pointing,
            end_roll,
            Duration::from_secs(10),
        )
        .unwrap();

        // Start/end orientations preserve their constructed rolls.
        let q0 = traj.orientation_at(Duration::ZERO).unwrap();
        assert_abs_diff_eq!(roll_of(&q0), start_roll, epsilon = 1e-9);
        let q1 = traj.orientation_at(Duration::from_secs(10)).unwrap();
        assert_abs_diff_eq!(roll_of(&q1), end_roll, epsilon = 1e-9);

        // Midpoint roll is exactly halfway.
        let qm = traj.orientation_at(Duration::from_secs(5)).unwrap();
        let expected_roll = 0.5 * (start_roll + end_roll);
        assert_abs_diff_eq!(roll_of(&qm), expected_roll, epsilon = 1e-9);
    }

    /// Build a five-waypoint trajectory whose first and last orientations
    /// differ by a known angle. Used by the periodicity tests below.
    fn nonperiodic_test_trajectory() -> Trajectory {
        let mut wps = Vec::new();
        for i in 0..5 {
            let t = Duration::from_secs_f64(i as f64 * 0.25);
            let pointing = make_pointing(10.0 + 4.0 * i as f64, 20.0);
            wps.push(Waypoint::from_pointing(t, pointing));
        }
        Trajectory::new(wps).unwrap()
    }

    #[test]
    fn with_period_rejects_period_shorter_than_span() {
        let traj = nonperiodic_test_trajectory();
        let span = traj.duration();
        let too_short = span - Duration::from_millis(1);
        let err = traj.with_period(too_short).expect_err("must reject");
        match err {
            TrajectoryError::PeriodTooShort { period, span: s } => {
                assert_eq!(period, too_short);
                assert_eq!(s, span);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn looped_makes_pose_at_seam_match_pose_at_start() {
        let traj = nonperiodic_test_trajectory().looped();
        let period = traj.period().expect("looped sets a period");

        let q_start = traj.orientation_at(Duration::ZERO).unwrap();
        let q_seam = traj.orientation_at(period).unwrap();

        // After one full period we should be back at the start orientation
        // bit-exactly (modulo float wobble): both calls land on the same
        // SLERP endpoint by construction.
        assert_abs_diff_eq!(q_start.angle_to(&q_seam), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn periodic_trajectory_is_periodic_across_many_cycles() {
        let traj = nonperiodic_test_trajectory().looped();
        let period = traj.period().unwrap();
        let period_s = period.as_secs_f64();

        // Sample at a fractional offset inside the period.
        let t0 = Duration::from_secs_f64(0.37 * period_s);
        let q0 = traj.orientation_at(t0).unwrap();

        // The same fractional offset in any future cycle must agree.
        for cycle in 1..=7 {
            let t_cycle = Duration::from_secs_f64(cycle as f64 * period_s + 0.37 * period_s);
            let q_cycle = traj.orientation_at(t_cycle).unwrap();
            let drift = q0.angle_to(&q_cycle);
            assert!(
                drift.abs() < 1e-12,
                "cycle {cycle}: angular drift {drift:e} exceeds 1e-12 rad"
            );
        }
    }

    #[test]
    fn periodic_trajectory_wrap_segment_lands_on_first_waypoint() {
        // Within the wrap segment, SLERP goes from last waypoint to first.
        // At the wrap segment's end (i.e. t == period), pose must equal
        // the first waypoint's orientation.
        let traj = nonperiodic_test_trajectory().looped();
        let period = traj.period().unwrap();
        let first = traj.waypoints()[0].orientation;
        let q_end = traj.orientation_at(period).unwrap();
        assert_abs_diff_eq!(first.angle_to(&q_end), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn periodic_trajectory_wrap_segment_lands_on_last_waypoint() {
        // At the start of the wrap segment (i.e. t == end_time), the
        // SLERP frac is 0 so pose must equal the last stored waypoint.
        let traj = nonperiodic_test_trajectory().looped();
        let last = traj.waypoints().last().unwrap().orientation;
        let q_at_end = traj.orientation_at(traj.end_time()).unwrap();
        assert_abs_diff_eq!(last.angle_to(&q_at_end), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn non_periodic_trajectory_still_rejects_out_of_range() {
        let traj = nonperiodic_test_trajectory();
        let err = traj
            .orientation_at(traj.end_time() + Duration::from_millis(1))
            .expect_err("non-periodic must error on out-of-range");
        assert!(matches!(err, TrajectoryError::TimeOutOfRange(_, _, _)));
    }

    /// In-memory catalog used by the prefetch test: holds a fixed list of
    /// `StarData` rows and implements just the trait surface the default
    /// `stars_in_field` filter needs (`filter_star_data` is the hot path).
    struct PrefetchTestCatalog {
        stars: Vec<StarData>,
    }

    impl StarCatalog for PrefetchTestCatalog {
        type Star = StarData;

        fn get_star(&self, id: usize) -> Option<&StarData> {
            self.stars.get(id)
        }

        fn stars(&self) -> impl Iterator<Item = &StarData> {
            self.stars.iter()
        }

        fn len(&self) -> usize {
            self.stars.len()
        }

        fn filter<F>(&self, predicate: F) -> Vec<&StarData>
        where
            F: Fn(&StarData) -> bool,
        {
            self.stars.iter().filter(|s| predicate(s)).collect()
        }

        fn star_data(&self) -> impl Iterator<Item = StarData> + '_ {
            self.stars.iter().cloned()
        }

        fn filter_star_data<F>(&self, predicate: F) -> Vec<StarData>
        where
            F: Fn(&StarData) -> bool,
        {
            self.stars
                .iter()
                .filter(|s| predicate(s))
                .cloned()
                .collect()
        }
    }

    #[test]
    fn prefetch_catalog_for_envelope_is_nonempty_and_bounded() {
        // Catalog with a mix of inside-envelope and well-outside-envelope
        // stars. The envelope is a 0.5-degree-diameter cone around
        // (RA=10°, Dec=20°), so the half-angle bound is 0.25°.
        let center = Equatorial::from_degrees(10.0, 20.0);
        let envelope_diameter_deg = 0.5;
        let envelope = (center, envelope_diameter_deg);
        let half_angle_deg = envelope_diameter_deg / 2.0;

        let catalog = PrefetchTestCatalog {
            stars: vec![
                // Inside the cone: dead-centre and a near-edge offset.
                StarData::new(1, 10.0, 20.0, 5.0, Some(0.5)),
                StarData::new(2, 10.1, 20.1, 6.0, Some(0.6)),
                // Well outside the cone (degrees away in both axes).
                StarData::new(3, 50.0, -30.0, 7.0, Some(0.8)),
                StarData::new(4, 200.0, 60.0, 8.0, Some(1.0)),
            ],
        };

        let prefetched = prefetch_catalog_for_envelope(&catalog, envelope);

        assert!(
            !prefetched.is_empty(),
            "prefetch should return at least one star for an envelope that contains catalog rows"
        );

        // Every returned star must be within the envelope's half-angle
        // of the centre. Tolerance covers the cosine-distance arithmetic
        // used by `stars_in_field`.
        let tol_deg = 1e-9;
        for star in &prefetched {
            let dist = angular_distance_deg(&center, &star.position);
            assert!(
                dist <= half_angle_deg + tol_deg,
                "star {} at ({:.6}°, {:.6}°) is {:.6}° from centre, exceeds half-angle {:.6}°",
                star.id,
                star.position.ra_degrees(),
                star.position.dec_degrees(),
                dist,
                half_angle_deg,
            );
        }
    }
}

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use nalgebra::UnitQuaternion;
use starfield::catalogs::StarData;
use starfield::Equatorial;
use thiserror::Error;

use crate::hardware::satellite::FocalPlaneConfig;
use crate::image_proc::render::{StarInFocalPlane, StarInFrame};
use crate::photometry::photoconversion::SourceFlux;
use crate::photometry::zodiacal::SolarAngularCoordinates;
use crate::sims::motion_blur::{
    render_motion_trajectory, MotionBlurConfig, DEFAULT_MAX_DRIFT_PER_STAMP_PX,
};
use crate::sims::orientation::{boresight_of, orientation_from_pointing};
use crate::star_math::star_data_to_fluxes;

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

    #[error("focal plane has no sensors")]
    NoSensors,

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
        Ok(Self { waypoints })
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

    /// Interpolate the spacecraft orientation at a given time using
    /// quaternion SLERP between bracketing waypoints.
    pub fn orientation_at(&self, t: Duration) -> Result<UnitQuaternion<f64>, TrajectoryError> {
        let start = self.start_time();
        let end = self.end_time();
        if t < start || t > end {
            return Err(TrajectoryError::TimeOutOfRange(t, start, end));
        }

        let seg_idx = self
            .waypoints
            .windows(2)
            .position(|w| t >= w[0].time && t <= w[1].time)
            .unwrap_or(self.waypoints.len() - 2);

        let w0 = &self.waypoints[seg_idx];
        let w1 = &self.waypoints[seg_idx + 1];

        let seg_duration = (w1.time - w0.time).as_secs_f64();
        if seg_duration == 0.0 {
            return Ok(w0.orientation);
        }
        let frac = (t - w0.time).as_secs_f64() / seg_duration;

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

    /// Return a phase-continuous version of this trajectory by SLERP-blending
    /// the raw end orientation into the first `smoothing_window` of the trajectory.
    ///
    /// The returned trajectory satisfies `orientation_at(start_time)` ==
    /// `orientation_at(end_time)` exactly: the seam is bit-identical, so a
    /// consumer that wraps phase as `t mod duration` sees no attitude jump
    /// at the wrap point.
    ///
    /// The blend is one-sided at the start: for `t in [start, start +
    /// smoothing_window]`, the orientation is `slerp(q_end, raw(t), s(u))`
    /// where `u = t / smoothing_window` and `s(u) = 3u² - 2u³` is a
    /// smoothstep with `s'(0) = s'(1) = 0`. In the continuous limit, the
    /// zero slope at `u = 1` makes the blended trajectory's angular
    /// velocity at the join point match the raw trajectory's velocity
    /// there, and the zero slope at `u = 0` makes the blended trajectory
    /// have zero angular velocity at the seam itself. When the wrap
    /// brings us back from `end_time` to `start_time`, the raw
    /// trajectory's end velocity is not preserved across the seam, but
    /// the resulting jerk (rather than the previous pose jump) is far
    /// less visible in rendered frames. The output is a piecewise-SLERP
    /// approximation of the continuous blend, so small local velocity
    /// jumps remain between adjacent sample segments — see
    /// `MIN_BLEND_SAMPLES` below.
    ///
    /// After `start + smoothing_window`, the trajectory follows the raw
    /// waypoints unchanged. The end waypoint is preserved so the wrap
    /// target itself is exact.
    ///
    /// If `smoothing_window >= duration`, it is clamped to the trajectory's
    /// duration (the entire trajectory becomes the blend region). If
    /// `smoothing_window == 0`, the result is the input trajectory with
    /// its first waypoint orientation replaced by `q_end` — bit-exact at
    /// the seam but with a pose discontinuity immediately after; this
    /// degenerate case is supported for API completeness.
    ///
    /// At least `MIN_BLEND_SAMPLES` evenly spaced sample points are
    /// inserted across the blend window so the SLERP between consecutive
    /// waypoints in the output closely tracks the underlying smoothstep
    /// curve even when the input has only sparse waypoints (e.g.
    /// `from_endpoints`).
    pub fn looped(self, smoothing_window: Duration) -> Trajectory {
        const MIN_BLEND_SAMPLES: usize = 16;

        let start = self.start_time();
        let end = self.end_time();
        let total = end - start;
        let q_end = self
            .orientation_at(end)
            .expect("end_time is always in range");

        if smoothing_window.is_zero() {
            let mut wps = self.waypoints;
            wps[0].orientation = q_end;
            return Trajectory { waypoints: wps };
        }

        let window = smoothing_window.min(total);
        let window_end_time = start + window;
        let q_window_end = self
            .orientation_at(window_end_time)
            .expect("window_end_time is within trajectory range");

        let blended = |t: Duration| -> UnitQuaternion<f64> {
            let raw = self
                .orientation_at(t)
                .expect("t is within trajectory range");
            let u = (t - start).as_secs_f64() / window.as_secs_f64();
            let s = smoothstep(u);
            q_end.slerp(&raw, s)
        };

        let mut new_waypoints: Vec<Waypoint> = Vec::new();
        new_waypoints.push(Waypoint::new(start, q_end));

        let mut sample_times: Vec<Duration> = Vec::new();
        for wp in &self.waypoints {
            if wp.time > start && wp.time < window_end_time {
                sample_times.push(wp.time);
            }
        }
        let window_secs = window.as_secs_f64();
        for i in 1..MIN_BLEND_SAMPLES {
            let frac = i as f64 / MIN_BLEND_SAMPLES as f64;
            sample_times.push(start + Duration::from_secs_f64(window_secs * frac));
        }
        sample_times.sort();
        sample_times.dedup();

        for t in sample_times {
            if t <= start || t >= window_end_time {
                continue;
            }
            new_waypoints.push(Waypoint::new(t, blended(t)));
        }

        new_waypoints.push(Waypoint::new(window_end_time, q_window_end));

        for wp in &self.waypoints {
            if wp.time > window_end_time {
                new_waypoints.push(wp.clone());
            }
        }

        if new_waypoints.last().map(|w| w.time < end).unwrap_or(false) {
            new_waypoints.push(Waypoint::new(end, q_end));
        }

        Trajectory {
            waypoints: new_waypoints,
        }
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

/// Cubic Hermite smoothstep: `3u² - 2u³` on `[0, 1]`, clamped outside.
/// Has zero derivative at both endpoints, so it produces a C1 join when
/// used to blend between two parameterised curves.
fn smoothstep(u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
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
    /// Pre-fetched catalog stars covering the full trajectory envelope.
    pub catalog_stars: &'a [StarData],
    /// Per-sensor pre-projected galaxies (`Scene::with_galaxies` shape).
    /// Defaults to an empty slice — callers that don't render galaxies
    /// pass `&[]` and the inner vec slot is filled with empties at the
    /// motion-blur layer. See `crate::sims::nsa_galaxies` for the
    /// builder that produces this shape from an NSA FITS file.
    pub per_sensor_galaxies: &'a [Vec<crate::scene_galaxy::GalaxyInFrame>],
    /// Solar angular coordinates for zodiacal background.
    pub zodiacal: SolarAngularCoordinates,
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
        config.catalog_stars,
        config.per_sensor_galaxies,
        config.focal_plane,
        config.zodiacal,
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

    /// Sample angular velocity (rad/s) at `t` via a centred finite
    /// difference over `±dt`, clamped to the trajectory range.
    fn angular_velocity_at(traj: &Trajectory, t: Duration, dt: Duration) -> f64 {
        let start = traj.start_time();
        let end = traj.end_time();
        let t_lo = if t > start + dt { t - dt } else { start };
        let t_hi = if t + dt < end { t + dt } else { end };
        let q_lo = traj.orientation_at(t_lo).unwrap();
        let q_hi = traj.orientation_at(t_hi).unwrap();
        let span = (t_hi - t_lo).as_secs_f64();
        if span <= 0.0 {
            return 0.0;
        }
        q_lo.angle_to(&q_hi) / span
    }

    #[test]
    fn test_looped_seam_is_exact() {
        // Non-periodic linear sweep: pose at t=0 (RA 10) is far from
        // pose at t=10s (RA 20). After looping, the seam must match.
        let traj = Trajectory::from_endpoints(
            make_pointing(10.0, 20.0),
            make_pointing(20.0, 30.0),
            Duration::from_secs(10),
        )
        .unwrap();

        let looped = traj.looped(Duration::from_secs(2));

        let q_start = looped.orientation_at(Duration::ZERO).unwrap();
        let q_end = looped.orientation_at(Duration::from_secs(10)).unwrap();
        let seam_angle = q_start.angle_to(&q_end);
        assert!(
            seam_angle < 1e-9,
            "seam angle = {seam_angle} rad (expected < 1e-9)"
        );
    }

    #[test]
    fn test_looped_preserves_end_orientation() {
        // The end waypoint defines the wrap target — looped() must not
        // perturb it.
        let traj = Trajectory::from_endpoints(
            make_pointing(10.0, 20.0),
            make_pointing(20.0, 30.0),
            Duration::from_secs(10),
        )
        .unwrap();
        let raw_end = traj.orientation_at(Duration::from_secs(10)).unwrap();

        let looped = traj.clone().looped(Duration::from_secs(2));
        let looped_end = looped.orientation_at(Duration::from_secs(10)).unwrap();
        assert!(raw_end.angle_to(&looped_end) < 1e-12);
    }

    #[test]
    fn test_looped_outside_blend_window_matches_raw() {
        // After t = smoothing_window, the looped trajectory must follow
        // the raw orientations exactly.
        let traj = Trajectory::from_endpoints(
            make_pointing(10.0, 20.0),
            make_pointing(20.0, 30.0),
            Duration::from_secs(10),
        )
        .unwrap();
        let raw_at_5 = traj.orientation_at(Duration::from_secs(5)).unwrap();
        let raw_at_8 = traj.orientation_at(Duration::from_secs(8)).unwrap();

        let looped = traj.looped(Duration::from_secs(2));
        let looped_at_5 = looped.orientation_at(Duration::from_secs(5)).unwrap();
        let looped_at_8 = looped.orientation_at(Duration::from_secs(8)).unwrap();

        assert!(raw_at_5.angle_to(&looped_at_5) < 1e-9);
        assert!(raw_at_8.angle_to(&looped_at_8) < 1e-9);
    }

    #[test]
    fn test_looped_blend_region_bounded_angular_velocity() {
        // For a trajectory that is already approximately periodic
        // (start ~= end), the blend region should not introduce wild
        // angular velocity — it should stay within a small overshoot
        // factor of the raw peak angular velocity.
        //
        // Trajectory: RA 10 → RA 10.05 → RA 10 (we route through an
        // intermediate to make it non-trivial, with start ≈ end so the
        // looped gap is tiny). Raw peak angular velocity is around
        // 0.05 deg over 5 s = 0.01 deg/s = 1.75e-4 rad/s.
        let waypoints = vec![
            Waypoint::from_pointing(Duration::ZERO, make_pointing(10.0, 0.0)),
            Waypoint::from_pointing(Duration::from_secs(5), make_pointing(10.05, 0.0)),
            Waypoint::from_pointing(Duration::from_secs(10), make_pointing(10.0, 0.0)),
        ];
        let traj = Trajectory::new(waypoints).unwrap();

        // Measure raw trajectory's peak angular velocity by sampling.
        let dt = Duration::from_millis(50);
        let mut raw_peak: f64 = 0.0;
        for i in 0..=200 {
            let t = Duration::from_secs_f64(10.0 * i as f64 / 200.0);
            raw_peak = raw_peak.max(angular_velocity_at(&traj, t, dt));
        }

        let window = Duration::from_secs(2);
        let looped = traj.clone().looped(window);

        // Sample angular velocity across the blend region. Allow a 5×
        // overshoot factor to absorb the tiny seam-closing motion plus
        // smoothstep's peak slope of 1.5 vs raw's slower segments.
        let bound = 5.0 * raw_peak;
        let samples = 200;
        let window_secs = window.as_secs_f64();
        for i in 0..=samples {
            let frac = i as f64 / samples as f64;
            let t = Duration::from_secs_f64(window_secs * frac);
            let omega = angular_velocity_at(&looped, t, dt);
            assert!(
                omega <= bound,
                "blend angular velocity {omega} rad/s exceeds bound {bound} rad/s (raw peak = {raw_peak}) at t={t:?}"
            );
        }
    }

    #[test]
    fn test_looped_angularly_continuous_across_blend() {
        // No large pose discontinuity inside the blend region: between
        // any two adjacent sample times, the angular separation should
        // be small (no orientation jumps).
        let waypoints = vec![
            Waypoint::from_pointing(Duration::ZERO, make_pointing(10.0, 0.0)),
            Waypoint::from_pointing(Duration::from_secs(5), make_pointing(10.05, 0.0)),
            Waypoint::from_pointing(Duration::from_secs(10), make_pointing(10.0, 0.0)),
        ];
        let traj = Trajectory::new(waypoints).unwrap();

        let window = Duration::from_secs(2);
        let looped = traj.looped(window);

        let window_secs = window.as_secs_f64();
        let samples = 500;
        let step_secs = window_secs / samples as f64;
        let mut prev = looped.orientation_at(Duration::ZERO).unwrap();
        // No pair of orientations separated by `step_secs` should differ
        // by more than this generous bound (well under one arcminute).
        let max_step_rad: f64 = 1e-3;
        for i in 1..=samples {
            let t = Duration::from_secs_f64(i as f64 * step_secs);
            let q = looped.orientation_at(t).unwrap();
            let step = prev.angle_to(&q);
            assert!(
                step < max_step_rad,
                "pose jump of {step} rad between adjacent samples at t={t:?}"
            );
            prev = q;
        }
    }

    #[test]
    fn test_looped_with_multi_waypoint_trajectory() {
        // Verify the looped helper handles a trajectory with many
        // existing waypoints across the blend window: the originals in
        // (start, window_end) are blended and additional samples are
        // inserted to keep the SLERP between consecutive waypoints
        // close to the smoothstep curve.
        let mut waypoints = Vec::new();
        for i in 0..=10 {
            let frac = i as f64 / 10.0;
            let t = Duration::from_secs_f64(10.0 * frac);
            let ra = 10.0 + 10.0 * frac;
            waypoints.push(Waypoint::from_pointing(t, make_pointing(ra, 0.0)));
        }
        let traj = Trajectory::new(waypoints).unwrap();

        let looped = traj.looped(Duration::from_secs(3));

        // Seam closes exactly.
        let q_start = looped.orientation_at(Duration::ZERO).unwrap();
        let q_end = looped.orientation_at(Duration::from_secs(10)).unwrap();
        assert!(q_start.angle_to(&q_end) < 1e-9);

        // Post-blend behaviour matches raw.
        let raw_at_7 = Trajectory::from_endpoints(
            make_pointing(10.0, 0.0),
            make_pointing(20.0, 0.0),
            Duration::from_secs(10),
        )
        .unwrap()
        .orientation_at(Duration::from_secs(7))
        .unwrap();
        let looped_at_7 = looped.orientation_at(Duration::from_secs(7)).unwrap();
        // Allow more tolerance here since the dense waypoints introduce
        // small SLERP-segment differences vs the two-endpoint reference.
        assert!(raw_at_7.angle_to(&looped_at_7) < 1e-2);
    }

    #[test]
    fn test_looped_window_clamped_to_duration() {
        // smoothing_window > duration is clamped to duration.
        let traj = Trajectory::from_endpoints(
            make_pointing(10.0, 20.0),
            make_pointing(20.0, 30.0),
            Duration::from_secs(10),
        )
        .unwrap();

        let looped = traj.looped(Duration::from_secs(100));
        let q_start = looped.orientation_at(Duration::ZERO).unwrap();
        let q_end = looped.orientation_at(Duration::from_secs(10)).unwrap();
        assert!(q_start.angle_to(&q_end) < 1e-9);
    }

    #[test]
    fn test_looped_zero_window_snaps_endpoints() {
        // Degenerate case: zero-window blend just snaps the start
        // orientation to the end orientation, with no smoothing.
        let traj = Trajectory::from_endpoints(
            make_pointing(10.0, 20.0),
            make_pointing(20.0, 30.0),
            Duration::from_secs(10),
        )
        .unwrap();

        let looped = traj.looped(Duration::ZERO);
        let q_start = looped.orientation_at(Duration::ZERO).unwrap();
        let q_end = looped.orientation_at(Duration::from_secs(10)).unwrap();
        assert!(q_start.angle_to(&q_end) < 1e-12);
    }

    #[test]
    fn test_smoothstep_endpoints_and_midpoint() {
        assert_abs_diff_eq!(smoothstep(0.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(smoothstep(1.0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(smoothstep(0.5), 0.5, epsilon = 1e-12);
        // Clamps outside [0, 1].
        assert_abs_diff_eq!(smoothstep(-0.5), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(smoothstep(1.5), 1.0, epsilon = 1e-12);
    }
}

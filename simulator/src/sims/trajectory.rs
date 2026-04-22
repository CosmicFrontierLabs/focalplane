use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use image::{ImageBuffer, Luma};
use log::info;
use nalgebra::UnitQuaternion;
use starfield::catalogs::StarData;
use starfield::Equatorial;
use thiserror::Error;

use crate::hardware::satellite::FocalPlaneConfig;
use crate::image_proc::render::{
    project_stars_to_focal_plane_oriented, StarInFocalPlane, StarInFrame,
};
use crate::photometry::photoconversion::SourceFlux;
use crate::photometry::zodiacal::SolarAngularCoordinates;
use crate::scene::Scene;
use crate::sims::orientation::{boresight_of, orientation_from_pointing};
use crate::star_math::star_data_to_fluxes;
use shared::units::LengthExt;

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

    /// Generate evenly spaced frame times from start to end (inclusive of start).
    pub fn frame_times(&self, timestep: Duration) -> Vec<Duration> {
        let mut times = Vec::new();
        let mut t = self.start_time();
        let end = self.end_time();
        while t <= end {
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

/// Route focal-plane stars to sensors with a flux cache to avoid redundant computation.
///
/// Identical to `route_stars_to_sensors` in render.rs but caches `star_data_to_fluxes()`
/// results keyed by (star_id, sensor_index). For trajectory rendering where the same
/// stars appear across many frames on the same sensors, this eliminates repeated
/// Simpson's rule integration.
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

/// Save a single-sensor u16 image as 16-bit grayscale PNG.
fn save_u16_png(image: &ndarray::Array2<u16>, path: &Path) -> Result<(), TrajectoryError> {
    let (height, width) = image.dim();
    let raw: Vec<u16> = image.iter().copied().collect();
    let img: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::from_raw(width as u32, height as u32, raw)
            .ok_or_else(|| TrajectoryError::ImageWrite("buffer size mismatch".into()))?;
    img.save(path)
        .map_err(|e| TrajectoryError::ImageWrite(e.to_string()))
}

/// Configuration for rendering a trajectory sequence.
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
    /// Solar angular coordinates for zodiacal background.
    pub zodiacal: SolarAngularCoordinates,
    /// Directory to write 16-bit PNG output frames.
    pub output_dir: &'a Path,
    /// Optional RNG seed for reproducible noise.
    pub base_seed: Option<u64>,
}

/// Render a sequence of frames along a trajectory, writing 16-bit PNG files.
///
/// Queries the star catalog conceptually "once" (caller provides pre-fetched stars),
/// then for each frame along the trajectory: re-projects, routes to sensors, renders,
/// and writes 16-bit PNG output.
///
/// Returns the total number of frames rendered.
pub fn render_trajectory(config: &TrajectoryRenderConfig) -> Result<usize, TrajectoryError> {
    let TrajectoryRenderConfig {
        trajectory,
        timestep,
        exposure,
        focal_plane,
        catalog_stars,
        zodiacal,
        output_dir,
        base_seed,
    } = config;
    let sensor_count = focal_plane.array.sensor_count();
    if sensor_count == 0 {
        return Err(TrajectoryError::NoSensors);
    }

    let first_sat = focal_plane
        .satellite_for_sensor(0)
        .ok_or(TrajectoryError::NoSensors)?;
    let airy_pix = first_sat.airy_disk_pixel_space();
    let pixel_size_mm = first_sat.sensor.pixel_size().as_millimeters();
    let padding_mm = airy_pix.first_zero() * 2.0 * pixel_size_mm;

    let star_refs: Vec<&StarData> = catalog_stars.iter().collect();
    let frame_times = trajectory.frame_times(*timestep);
    let total_frames = frame_times.len();

    info!("Rendering {total_frames} frames across {sensor_count} sensors");

    let mut flux_cache: HashMap<(u64, usize), SourceFlux> = HashMap::new();
    let single_sensor = sensor_count == 1;

    for (frame_idx, &t) in frame_times.iter().enumerate() {
        let orientation = trajectory.orientation_at(t)?;
        let pointing = boresight_of(&orientation);

        let fp_stars = project_stars_to_focal_plane_oriented(
            &star_refs,
            &orientation,
            focal_plane,
            padding_mm,
        );
        let per_sensor_stars =
            route_stars_to_sensors_cached(&fp_stars, focal_plane, padding_mm, &mut flux_cache);

        let scene = Scene::from_stars(
            (*focal_plane).clone(),
            per_sensor_stars,
            pointing,
            *zodiacal,
        );

        let seed = base_seed.map(|s| s + frame_idx as u64);
        let results = scene.render_with_seed(exposure, seed);

        for (sensor_idx, result) in results.iter().enumerate() {
            let filename = if single_sensor {
                format!("frame_{frame_idx:06}.png")
            } else {
                format!("frame_{frame_idx:06}_sensor_{sensor_idx:02}.png")
            };
            let path = output_dir.join(&filename);
            save_u16_png(&result.quantized_image, &path)?;
        }

        if (frame_idx + 1) % 10 == 0 || frame_idx + 1 == total_frames {
            info!(
                "Rendered frame {}/{} (t={:.3}s)",
                frame_idx + 1,
                total_frames,
                t.as_secs_f64()
            );
        }
    }

    info!(
        "Flux cache: {} entries ({} cache-eligible lookups saved)",
        flux_cache.len(),
        flux_cache.len()
    );

    Ok(total_frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sims::orientation::roll_of;
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
        assert!((start.ra_degrees() - 10.0).abs() < 1e-9);
        assert!((start.dec_degrees() - 20.0).abs() < 1e-9);

        let end = traj.pointing_at(Duration::from_secs(10)).unwrap();
        assert!((end.ra_degrees() - 20.0).abs() < 1e-9);
        assert!((end.dec_degrees() - 30.0).abs() < 1e-9);
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
        assert!((mid.dec_degrees() - 20.0).abs() < 0.5);
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

        let times = traj.frame_times(Duration::from_secs(2));
        assert_eq!(times.len(), 6); // 0, 2, 4, 6, 8, 10
        assert_eq!(times[0], Duration::ZERO);
        assert_eq!(times[5], Duration::from_secs(10));
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
        assert!((center.ra_degrees() - 15.0).abs() < 1.0);
        assert!(center.dec_degrees().abs() < 1.0);

        // Diameter should cover the 10 degree span plus the base FOV
        assert!(diameter >= 10.0 + base_fov);
    }

    #[test]
    fn test_angular_distance() {
        let a = make_pointing(0.0, 0.0);
        let b = make_pointing(90.0, 0.0);
        let dist = angular_distance_deg(&a, &b);
        assert!((dist - 90.0).abs() < 0.01);
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
        assert!((roll_of(&q0) - start_roll).abs() < 1e-9);
        let q1 = traj.orientation_at(Duration::from_secs(10)).unwrap();
        assert!((roll_of(&q1) - end_roll).abs() < 1e-9);

        // Midpoint roll is exactly halfway.
        let qm = traj.orientation_at(Duration::from_secs(5)).unwrap();
        let expected_roll = 0.5 * (start_roll + end_roll);
        assert!(
            (roll_of(&qm) - expected_roll).abs() < 1e-9,
            "midpoint roll {:.9} != expected {:.9}",
            roll_of(&qm),
            expected_roll
        );
    }
}

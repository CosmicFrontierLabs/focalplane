//! Render-descriptor schema emitted alongside the motion-blur PNG sequence.
//!
//! The motion-blur renderer writes one `metadata.json` per output directory
//! describing the trajectory, per-frame orientation/exposure, the set of
//! catalog stars queried for the run, the hardware configuration, and the
//! render knobs. The JSON shape is stable and versioned via
//! [`RenderMetadata::version`] so downstream consumers can match on it.
//!
//! # File layout assumption
//!
//! The renderer writes PNG files under per-sensor subdirectories:
//!
//! ```text
//! <output_dir>/
//! ├── metadata.json
//! ├── sensor_00/
//! │   ├── frame_000000.png
//! │   └── frame_000001.png
//! └── sensor_01/
//!     └── ...
//! ```
//!
//! The `paths` map on each [`FrameMeta`] stores relative forward-slash paths
//! of the form `"sensor_00/frame_000000.png"` — portable across platforms.

use std::collections::BTreeMap;

use nalgebra::UnitQuaternion;
use serde::{Deserialize, Serialize};

/// Root descriptor written as `metadata.json` at the output-directory root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderMetadata {
    /// Schema version for this descriptor. Currently always `"1.0"`.
    pub version: String,
    /// ISO-8601 UTC timestamp at which the render finished building metadata.
    pub rendered_at: String,
    /// Trajectory-level summary (duration and waypoints).
    pub trajectory: TrajectoryMeta,
    /// One entry per rendered frame.
    pub frames: Vec<FrameMeta>,
    /// Catalog stars covering the trajectory envelope, recorded verbatim.
    pub stars: Vec<StarMeta>,
    /// Telescope + sensor array hardware summary.
    pub hardware: HardwareMeta,
    /// Render knobs (exposure, timestep, seed, etc.).
    pub render_config: RenderConfigMeta,
}

/// Summary of the trajectory: duration and its waypoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryMeta {
    /// End time minus start time, in seconds.
    pub duration_s: f64,
    /// Trajectory start time (seconds from trajectory origin).
    pub start_time_s: f64,
    /// Trajectory end time (seconds from trajectory origin).
    pub end_time_s: f64,
    /// Ordered list of waypoints, matching the input trajectory.
    pub waypoints: Vec<WaypointMeta>,
}

/// Single trajectory waypoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaypointMeta {
    /// Waypoint time (seconds from trajectory origin).
    pub time_s: f64,
    /// Orientation quaternion. Serializes as a 4-element JSON array in
    /// nalgebra's native `[i, j, k, w]` order (imaginary parts first,
    /// real part last) — same shape as `Quaternion::coords`. Consumers
    /// reconstructing via `Quaternion::new(w, i, j, k)` must pull `w`
    /// from index 3, not 0.
    pub quat: UnitQuaternion<f64>,
    /// Boresight pointing derived from `quat`.
    pub boresight: EquatorialMeta,
    /// Roll angle derived from `quat`, in degrees.
    pub roll_deg: f64,
}

/// Equatorial coordinate pair in degrees.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EquatorialMeta {
    /// Right ascension in degrees.
    pub ra_deg: f64,
    /// Declination in degrees.
    pub dec_deg: f64,
}

/// Per-frame metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMeta {
    /// Frame index (0-based).
    pub idx: usize,
    /// Frame start time (seconds from trajectory origin).
    pub t_s: f64,
    /// Exposure duration for this frame, in seconds.
    pub exposure_s: f64,
    /// Mid-frame orientation quaternion. Serializes as a 4-element
    /// JSON array in nalgebra's native `[i, j, k, w]` order (imaginary
    /// parts first, real part last).
    pub quat: UnitQuaternion<f64>,
    /// Mid-frame boresight pointing derived from `quat`.
    pub boresight: EquatorialMeta,
    /// Mid-frame roll angle derived from `quat`, in degrees.
    pub roll_deg: f64,
    /// Total number of stratified-MC PSF stamps deposited across this exposure.
    pub n_stamps: usize,
    /// Map from `"sensor_NN"` (zero-padded sensor index) to the relative
    /// forward-slash PNG path, e.g. `"sensor_00/frame_000000.png"`.
    pub paths: BTreeMap<String, String>,
}

/// Catalog star recorded verbatim from the trajectory envelope query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarMeta {
    /// Catalog star identifier.
    pub id: u64,
    /// Right ascension in degrees.
    pub ra_deg: f64,
    /// Declination in degrees.
    pub dec_deg: f64,
    /// Apparent magnitude (catalog-defined band).
    pub magnitude: f64,
    /// Optional color index (`B - V`, a.k.a. `bp_rp` for Gaia-style catalogs).
    pub color_index: Option<f64>,
}

/// Hardware summary: telescope + per-sensor layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMeta {
    /// Telescope name.
    pub telescope: String,
    /// Operating temperature in Celsius.
    pub temperature_c: f64,
    /// Per-sensor entries in array-index order.
    pub sensors: Vec<SensorMeta>,
}

/// Per-sensor layout entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorMeta {
    /// Sensor index in the array.
    pub idx: usize,
    /// Sensor model name.
    pub name: String,
    /// Sensor dimensions in pixels, `[width, height]`.
    pub dimensions_px: [usize; 2],
    /// Sensor center position in the focal plane, `[x_mm, y_mm]`.
    pub position_mm: [f64; 2],
}

/// Render knobs captured for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderConfigMeta {
    /// Exposure duration per frame, in seconds.
    pub exposure_s: f64,
    /// Time between successive frame start times, in seconds.
    pub timestep_s: f64,
    /// Per-stamp drift budget in pixels.
    pub max_drift_per_stamp_px: f64,
    /// Base RNG seed used to derive per-tile seeds.
    pub seed: u64,
    /// If true, the adaptive scheduler was bypassed (a single PSF stamp per frame).
    pub force_static: bool,
    /// Catalog path the run was configured with.
    pub catalog_path: String,
    /// Solar angular coordinates used for zodiacal-light evaluation.
    pub zodiacal: ZodiacalMeta,
}

/// Solar angular coordinate pair in degrees.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ZodiacalMeta {
    /// Solar elongation in degrees.
    pub elongation_deg: f64,
    /// Ecliptic latitude in degrees.
    pub latitude_deg: f64,
}

/// Relative PNG path for a given sensor index, using forward slashes.
///
/// Matches the on-disk layout `sensor_NN/frame_NNNNNN.png`.
pub fn sensor_relative_png_path(sensor_idx: usize, frame_idx: usize) -> String {
    format!(
        "{}/{}",
        sensor_dir_name(sensor_idx),
        frame_file_name(frame_idx),
    )
}

/// Zero-padded `"sensor_NN"` directory name.
pub fn sensor_dir_name(sensor_idx: usize) -> String {
    format!("sensor_{sensor_idx:02}")
}

/// Zero-padded `"frame_NNNNNN.png"` file name (no sensor suffix).
pub fn frame_file_name(frame_idx: usize) -> String {
    format!("frame_{frame_idx:06}.png")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_dir_name_is_zero_padded() {
        assert_eq!(sensor_dir_name(0), "sensor_00");
        assert_eq!(sensor_dir_name(7), "sensor_07");
        assert_eq!(sensor_dir_name(42), "sensor_42");
    }

    #[test]
    fn test_frame_file_name_is_zero_padded() {
        assert_eq!(frame_file_name(0), "frame_000000.png");
        assert_eq!(frame_file_name(12345), "frame_012345.png");
    }

    #[test]
    fn test_sensor_relative_png_path_uses_forward_slash() {
        let p = sensor_relative_png_path(3, 100);
        assert_eq!(p, "sensor_03/frame_000100.png");
        assert!(p.contains('/'));
    }

    #[test]
    fn test_render_metadata_round_trip_via_serde() {
        let meta = RenderMetadata {
            version: "1.0".to_string(),
            rendered_at: "2026-04-22T00:00:00Z".to_string(),
            trajectory: TrajectoryMeta {
                duration_s: 10.0,
                start_time_s: 0.0,
                end_time_s: 10.0,
                waypoints: vec![WaypointMeta {
                    time_s: 0.0,
                    quat: UnitQuaternion::identity(),
                    boresight: EquatorialMeta {
                        ra_deg: 0.0,
                        dec_deg: 0.0,
                    },
                    roll_deg: 0.0,
                }],
            },
            frames: Vec::new(),
            stars: Vec::new(),
            hardware: HardwareMeta {
                telescope: "Test".to_string(),
                temperature_c: -10.0,
                sensors: Vec::new(),
            },
            render_config: RenderConfigMeta {
                exposure_s: 1.0,
                timestep_s: 1.0,
                max_drift_per_stamp_px: 0.1,
                seed: 42,
                force_static: false,
                catalog_path: "catalog.bin".to_string(),
                zodiacal: ZodiacalMeta {
                    elongation_deg: 90.0,
                    latitude_deg: 45.0,
                },
            },
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: RenderMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, "1.0");
        assert_eq!(parsed.hardware.telescope, "Test");
    }
}

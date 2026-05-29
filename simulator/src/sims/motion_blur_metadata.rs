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
use starfield::catalogs::StarData;

use crate::hardware::satellite::FocalPlaneConfig;
use crate::photometry::zodiacal::SolarAngularCoordinates;
use crate::scene_galaxy::Galaxy;
use crate::sims::motion_blur::MotionBlurConfig;
use crate::sims::trajectory::Trajectory;

/// Root descriptor written as `metadata.json` at the output-directory root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderMetadata {
    /// Schema version for this descriptor. Currently always `"1.2"`.
    pub version: String,
    /// ISO-8601 UTC timestamp at which the render finished building metadata.
    pub rendered_at: String,
    /// Trajectory carrying the waypoints used by the render.
    pub trajectory: Trajectory,
    /// One entry per rendered frame.
    pub frames: Vec<FrameMeta>,
    /// Catalog stars covering the trajectory envelope, recorded verbatim.
    pub stars: Vec<StarData>,
    /// Extended (Sérsic) sources covering the trajectory envelope,
    /// recorded verbatim from the catalog query.
    pub galaxies: Vec<Galaxy>,
    /// Full focal-plane optical + detector configuration.
    pub focal_plane: FocalPlaneConfig,
    /// Solar angular coordinates used for zodiacal-light evaluation.
    pub zodiacal: SolarAngularCoordinates,
    /// Render knobs (exposure, timestep, seed, etc.) — the full
    /// [`MotionBlurConfig`] passed to the renderer.
    pub render_config: MotionBlurConfig,
}

/// Per-frame metadata. Composite of render-schedule fields
/// (`idx`, `n_stamps`), the mid-frame pose, and the file-system layout
/// (`paths`) — no single production type covers all of these together.
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
    /// Mid-frame boresight right ascension, in degrees (derived from `quat`).
    pub boresight_ra_deg: f64,
    /// Mid-frame boresight declination, in degrees (derived from `quat`).
    pub boresight_dec_deg: f64,
    /// Mid-frame roll angle derived from `quat`, in degrees.
    pub roll_deg: f64,
    /// Total number of stratified-MC PSF stamps deposited across this exposure.
    pub n_stamps: usize,
    /// Map from `"sensor_NN"` (zero-padded sensor index) to the relative
    /// forward-slash PNG path, e.g. `"sensor_00/frame_000000.png"`.
    pub paths: BTreeMap<String, String>,
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
        use crate::hardware::sensor::models::GSENSE4040BSI;
        use crate::hardware::sensor_array::SensorArray;
        use crate::hardware::telescope::TelescopeConfig;
        use crate::photometry::photoconversion::{SourceFlux, SpotFlux};
        use crate::sims::trajectory::Waypoint;
        use shared::image_proc::airy::PixelScaledAiryDisk;
        use shared::units::{Length, LengthExt, Temperature, TemperatureExt, Wavelength};
        use starfield::catalogs::SersicProfile;
        use starfield::Equatorial;
        use std::path::PathBuf;
        use std::time::Duration;

        let zodiacal_for_test =
            SolarAngularCoordinates::new(90.0, 45.0).expect("test zodiacal coords should be valid");

        let psf = PixelScaledAiryDisk::with_fwhm(2.0, Wavelength::from_nanometers(550.0));
        let spot = SpotFlux {
            disk: psf,
            flux: 7.0e-3,
        };
        let galaxies = vec![Galaxy {
            id: 12345,
            name: Some("NGC test".to_string()),
            position: Equatorial::from_degrees(187.25, 12.5),
            profile: SersicProfile {
                theta_half_arcsec: 4.0,
                n: 1.5,
                axis_ratio: 0.7,
                position_angle_deg: 30.0,
            },
            flux: SourceFlux {
                photons: spot.clone(),
                electrons: spot,
            },
        }];
        let trajectory = Trajectory::new(vec![
            Waypoint::new(Duration::ZERO, UnitQuaternion::identity()),
            Waypoint::new(Duration::from_secs(10), UnitQuaternion::identity()),
        ])
        .expect("test trajectory should be valid");
        let meta = RenderMetadata {
            version: "1.2".to_string(),
            rendered_at: "2026-04-22T00:00:00Z".to_string(),
            trajectory,
            frames: Vec::new(),
            stars: Vec::new(),
            galaxies,
            focal_plane: FocalPlaneConfig::new(
                TelescopeConfig::new(
                    "Test",
                    Length::from_meters(0.5),
                    Length::from_meters(2.5),
                    0.8,
                ),
                SensorArray::single(GSENSE4040BSI.clone().with_dimensions(64, 64)),
                Temperature::from_celsius(-10.0),
            ),
            zodiacal: zodiacal_for_test,
            render_config: MotionBlurConfig {
                timestep: Duration::from_secs(1),
                exposure: Duration::from_secs(1),
                max_drift_per_stamp_px: 0.1,
                base_seed: Some(42),
                force_static: false,
                quiet: true,
                telescope_name: "Test".to_string(),
                catalog_path: PathBuf::from("catalog.bin"),
                temperature_c: -10.0,
            },
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: RenderMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, "1.2");
        assert_eq!(parsed.focal_plane.telescope.name, "Test");
        assert!((parsed.focal_plane.telescope.aperture.as_meters() - 0.5).abs() < 1e-12);
        assert_eq!(parsed.render_config.base_seed, Some(42));
        assert_eq!(parsed.trajectory.waypoints().len(), 2);
        let g = &parsed.galaxies[0];
        assert_eq!(g.id, 12345);
        assert_eq!(g.name.as_deref(), Some("NGC test"));
        assert!((g.position.ra_degrees() - 187.25).abs() < 1e-9);
        assert!((g.position.dec_degrees() - 12.5).abs() < 1e-9);
        assert_eq!(g.profile.n, 1.5);
        assert_eq!(g.profile.axis_ratio, 0.7);
        assert_eq!(g.flux.electrons.flux, 7.0e-3);
    }
}

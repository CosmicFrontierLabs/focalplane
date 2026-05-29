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

use crate::photometry::quantum_efficiency::QuantumEfficiency;
use crate::photometry::zodiacal::SolarAngularCoordinates;

/// Root descriptor written as `metadata.json` at the output-directory root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderMetadata {
    /// Schema version for this descriptor. Currently always `"1.2"`.
    pub version: String,
    /// ISO-8601 UTC timestamp at which the render finished building metadata.
    pub rendered_at: String,
    /// Trajectory-level summary (duration and waypoints).
    pub trajectory: TrajectoryMeta,
    /// One entry per rendered frame.
    pub frames: Vec<FrameMeta>,
    /// Catalog stars covering the trajectory envelope, recorded verbatim.
    pub stars: Vec<StarMeta>,
    /// Extended (Sérsic) sources covering the trajectory envelope,
    /// recorded verbatim from the catalog query. Stored flat — symmetric
    /// with [`RenderMetadata::stars`] — and de-duplicated by `id` if the
    /// same source was routed onto more than one sensor.
    pub galaxies: Vec<GalaxyMeta>,
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
    /// reconstructing via `Quaternion::new(w, i, j, k)` must remember
    /// to pull `w` from index 3, not 0.
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
    /// Human-readable catalog name when available (e.g. HIP designation,
    /// variable-star name). `None` for catalogs that only carry numeric
    /// identifiers.
    pub name: Option<String>,
    /// Right ascension in degrees.
    pub ra_deg: f64,
    /// Declination in degrees.
    pub dec_deg: f64,
    /// Apparent magnitude (catalog-defined band).
    pub magnitude: f64,
    /// Optional color index (`B - V`, a.k.a. `bp_rp` for Gaia-style catalogs).
    pub color_index: Option<f64>,
}

/// Extended (Sérsic) catalog source.
///
/// Mirrors [`StarMeta`] for sky-position semantics: `(ra_deg, dec_deg)`
/// is the catalog-truth centre, independent of which sensor (if any)
/// the source projects onto under any given trajectory pose. Pixel-frame
/// information is intentionally **not** recorded here — consumers can
/// reproject through the same trajectory + focal plane the renderer
/// used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalaxyMeta {
    /// Catalog galaxy identifier (e.g. NSAID for NSA, hashed name for
    /// the bright-galaxy catalog).
    pub id: u64,
    /// Human-readable catalog name when available (e.g. NGC/Messier
    /// designation). `None` for catalogs that only carry numeric
    /// identifiers (e.g. NSAID-keyed entries).
    pub name: Option<String>,
    /// Right ascension of the galaxy centre, in degrees (J2000).
    pub ra_deg: f64,
    /// Declination of the galaxy centre, in degrees (J2000).
    pub dec_deg: f64,
    /// Integrated photoelectron flux rate at the entrance aperture,
    /// in electrons per second per square centimetre. Equivalent to
    /// `SourceFlux::electrons.flux`.
    pub electrons_per_s_per_cm2: f64,
    /// Elliptical Sérsic profile parameters describing the extended
    /// surface-brightness shape.
    pub sersic: SersicMeta,
}

/// Elliptical Sérsic profile parameters for a [`GalaxyMeta`].
///
/// Mirrors `starfield::catalogs::SersicProfile` so downstream consumers
/// can reconstruct the surface-brightness model without depending on
/// `starfield`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SersicMeta {
    /// Half-light radius along the major axis, in arcseconds.
    pub theta_half_arcsec: f64,
    /// Sérsic index n (dimensionless, typically ~0.5 to ~6).
    pub n: f64,
    /// Axis ratio b/a, where b ≤ a (`1.0` is circular).
    pub axis_ratio: f64,
    /// Position angle of the major axis, degrees east of north (J2000).
    pub position_angle_deg: f64,
}

/// Hardware dump: telescope optics + per-sensor configuration.
///
/// Carries enough of the `FocalPlaneConfig` to let downstream consumers
/// reproduce pixel-to-sky projection and electron-to-DN conversion
/// without re-reading the source library. Curves (QE, dark current,
/// read noise) are tabulated verbatim from the renderer's interpolators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMeta {
    /// Telescope optical configuration.
    pub telescope: TelescopeMeta,
    /// Operating temperature in Celsius (shared across all sensors).
    pub temperature_c: f64,
    /// Per-sensor entries in array-index order.
    pub sensors: Vec<SensorMeta>,
}

/// Telescope optics and combined-system parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelescopeMeta {
    /// Telescope model name.
    pub name: String,
    /// Clear-aperture diameter in meters.
    pub aperture_m: f64,
    /// Effective focal length in meters (including optical train).
    pub focal_length_m: f64,
    /// Convenience: focal length divided by aperture.
    pub f_number: f64,
    /// Central obscuration ratio (0.0 - 1.0, fraction of aperture
    /// radius blocked by the secondary).
    pub obscuration_ratio: f64,
    /// Reference wavelength for diffraction-limited calculations, in nm.
    pub corrected_to_nm: f64,
    /// Wavelength-dependent quantum efficiency of the optical train
    /// (mirror reflectivity × lens transmission × etc.). Serializes
    /// as `{wavelengths_nm: [...], efficiencies: [...]}`.
    pub quantum_efficiency: QuantumEfficiency,
}

/// Per-sensor entry — geometry, conversion gains, and noise curves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorMeta {
    /// Sensor index in the array.
    pub idx: usize,
    /// Sensor model name.
    pub name: String,
    /// Sensor dimensions in pixels, `[width, height]`.
    pub dimensions_px: [usize; 2],
    /// Square-pixel physical pitch, in micrometers.
    pub pixel_pitch_um: f64,
    /// Sensor centre position on the focal plane, `[x_mm, y_mm]`,
    /// measured from the optical axis.
    pub position_mm: [f64; 2],
    /// ADC bit depth (8, 12, 14, 16 typical).
    pub bit_depth: u8,
    /// Gain in DN per electron.
    pub dn_per_electron: f64,
    /// Full-well capacity in electrons (saturation limit).
    pub max_well_depth_e: f64,
    /// Detector quantum efficiency curve (sensor only, optics excluded).
    pub quantum_efficiency: QuantumEfficiency,
    /// Combined telescope × sensor QE, pre-computed by `SatelliteConfig`.
    /// Equivalent to `QuantumEfficiency::product(telescope_qe, sensor_qe)`.
    pub combined_qe: QuantumEfficiency,
    /// Tabulated dark current vs. temperature.
    pub dark_current: DarkCurrentMeta,
    /// Tabulated read noise vs. (frame rate, temperature).
    pub read_noise: ReadNoiseMeta,
}

/// Tabulated dark current as a function of detector temperature.
/// Exponential interpolation between samples (the renderer's model
/// follows the standard 8 °C doubling rule).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkCurrentMeta {
    /// Temperature samples in degrees Celsius, ascending.
    pub temperatures_c: Vec<f64>,
    /// Dark current values in electrons per pixel per second,
    /// aligned with `temperatures_c`.
    pub dark_currents_e_per_px_per_s: Vec<f64>,
}

/// Tabulated read noise surface over (frame rate, temperature).
/// Bilinear interpolation between samples. Matches the underlying
/// `BilinearInterpolator` axis convention exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadNoiseMeta {
    /// X-axis grid: frame rates in Hz, ascending.
    pub frame_rates_hz: Vec<f64>,
    /// Y-axis grid: temperatures in degrees Celsius, ascending.
    pub temperatures_c: Vec<f64>,
    /// Noise surface in electrons RMS, indexed as
    /// `noise_e_rms[temperature_idx][frame_rate_idx]` (rows align
    /// with `temperatures_c`, columns with `frame_rates_hz`).
    pub noise_e_rms: Vec<Vec<f64>>,
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
    /// Serializes as `{elongation_deg, latitude_deg}`.
    pub zodiacal: SolarAngularCoordinates,
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
        let galaxies = vec![GalaxyMeta {
            id: 12345,
            name: Some("NGC test".to_string()),
            ra_deg: 187.25,
            dec_deg: 12.5,
            electrons_per_s_per_cm2: 7.0e-3,
            sersic: SersicMeta {
                theta_half_arcsec: 4.0,
                n: 1.5,
                axis_ratio: 0.7,
                position_angle_deg: 30.0,
            },
        }];
        let passband =
            QuantumEfficiency::from_table(vec![400.0, 550.0, 700.0], vec![0.0, 0.7, 0.0])
                .expect("test passband should be valid");
        let meta = RenderMetadata {
            version: "1.2".to_string(),
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
            galaxies,
            hardware: HardwareMeta {
                telescope: TelescopeMeta {
                    name: "Test".to_string(),
                    aperture_m: 0.5,
                    focal_length_m: 6.0,
                    f_number: 12.0,
                    obscuration_ratio: 0.3,
                    corrected_to_nm: 550.0,
                    quantum_efficiency: passband.clone(),
                },
                temperature_c: -10.0,
                sensors: vec![SensorMeta {
                    idx: 0,
                    name: "TestSensor".to_string(),
                    dimensions_px: [64, 64],
                    pixel_pitch_um: 3.76,
                    position_mm: [0.0, 0.0],
                    bit_depth: 16,
                    dn_per_electron: 0.5,
                    max_well_depth_e: 51000.0,
                    quantum_efficiency: passband.clone(),
                    combined_qe: passband.clone(),
                    dark_current: DarkCurrentMeta {
                        temperatures_c: vec![-20.0, 0.0, 20.0],
                        dark_currents_e_per_px_per_s: vec![0.01, 0.1, 1.0],
                    },
                    read_noise: ReadNoiseMeta {
                        frame_rates_hz: vec![5.0, 1000.0],
                        temperatures_c: vec![-20.0, 20.0],
                        noise_e_rms: vec![vec![1.2, 1.5], vec![1.4, 1.7]],
                    },
                }],
            },
            render_config: RenderConfigMeta {
                exposure_s: 1.0,
                timestep_s: 1.0,
                max_drift_per_stamp_px: 0.1,
                seed: 42,
                force_static: false,
                catalog_path: "catalog.bin".to_string(),
                zodiacal: SolarAngularCoordinates::new(90.0, 45.0)
                    .expect("test zodiacal coords should be valid"),
            },
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: RenderMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, "1.2");
        assert_eq!(parsed.hardware.telescope.name, "Test");
        assert_eq!(parsed.hardware.telescope.aperture_m, 0.5);
        let g = &parsed.galaxies[0];
        assert_eq!(g.id, 12345);
        assert_eq!(g.name.as_deref(), Some("NGC test"));
        assert_eq!(g.ra_deg, 187.25);
        assert_eq!(g.dec_deg, 12.5);
        assert_eq!(g.sersic.n, 1.5);
        assert_eq!(g.sersic.axis_ratio, 0.7);
        // Curve shapes round-trip.
        let s = &parsed.hardware.sensors[0];
        assert_eq!(s.pixel_pitch_um, 3.76);
        assert_eq!(
            s.quantum_efficiency.wavelengths_nm(),
            &[400.0_f64, 550.0, 700.0][..]
        );
        assert_eq!(s.dark_current.dark_currents_e_per_px_per_s.len(), 3);
        assert_eq!(s.read_noise.noise_e_rms[0], vec![1.2, 1.5]);
    }
}

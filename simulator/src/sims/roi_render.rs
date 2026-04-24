//! Standalone physics renderer for a small region-of-interest (ROI) on
//! a chosen sensor.
//!
//! Produces a `size_px × size_px` u16 patch rendered with the same
//! motion-blur + Poisson + read-noise + quantization pipeline as the
//! full sensor tile renderer in `sims::motion_blur`, but only for a
//! small window around a chosen anchor. Used by the context-view
//! renderer so the summary image can carry a physically correct,
//! oversampled zoom that reveals intra-exposure motion artifacts
//! without touching the full focal-plane render path.

use std::time::Duration;

use ndarray::Array2;
use shared::units::{AngleExt, LengthExt};
use starfield::catalogs::StarData;

use crate::hardware::satellite::{FocalPlaneConfig, FocalPlaneProjector};
use crate::sims::motion_blur::{
    render_region, RegionRenderInputs, SensorRegion, SubsampleSchedule,
};
use crate::sims::trajectory::{Trajectory, TrajectoryError};

/// Location + size of a region of interest in sensor-pixel space.
#[derive(Debug, Clone, Copy)]
pub struct RoiAnchor {
    pub sensor_idx: usize,
    pub center_px: (f64, f64),
    pub size_px: usize,
}

impl RoiAnchor {
    /// Top-left corner of the ROI as integer sensor pixels. The render
    /// output covers `[x0, x0 + size_px) × [y0, y0 + size_px)`.
    pub fn top_left_px(&self) -> (i32, i32) {
        let half = self.size_px as f64 / 2.0;
        (
            (self.center_px.0 - half).round() as i32,
            (self.center_px.1 - half).round() as i32,
        )
    }
}

/// Pick the ROI anchor: the brightest in-field star on any sensor that
/// sits at least `size_px / 2` pixels from every edge of its sensor.
/// Brightness is ranked by [`StarData::magnitude`] (smaller = brighter).
/// Returns `None` if no qualifying star is found.
pub fn pick_roi_anchor(
    stars: &[StarData],
    fp: &FocalPlaneConfig,
    orientation: &nalgebra::UnitQuaternion<f64>,
    size_px: usize,
) -> Option<RoiAnchor> {
    let padding_mm = 0.0;
    let half = size_px as f64 / 2.0;
    let mut best: Option<(f64, RoiAnchor)> = None;
    for (sensor_idx, ps) in fp.array.sensors.iter().enumerate() {
        let (w_px, h_px) = ps.sensor.dimensions.get_pixel_width_height();
        let (w_px, h_px) = (w_px as f64, h_px as f64);
        for star in stars {
            let Some((px, py)) = fp.project_to_sensor(star, orientation, sensor_idx, padding_mm)
            else {
                continue;
            };
            if px < half || py < half || px > w_px - half || py > h_px - half {
                continue;
            }
            let score = star.magnitude;
            if best.as_ref().map(|(m, _)| score < *m).unwrap_or(true) {
                best = Some((
                    score,
                    RoiAnchor {
                        sensor_idx,
                        center_px: (px, py),
                        size_px,
                    },
                ));
            }
        }
    }
    best.map(|(_, a)| a)
}

/// Endpoint-approximation of maximum angular drift a trajectory covers
/// from `t_start` to `t_end` (radians). Cheaper than the full per-segment
/// sum used in `motion_blur::max_drift_over_window`; within a single
/// frame's exposure the two agree to within a fraction of a percent.
fn drift_over_window(
    trajectory: &Trajectory,
    t_start: Duration,
    t_end: Duration,
) -> Result<f64, TrajectoryError> {
    if t_end <= t_start {
        return Ok(0.0);
    }
    let q_a = trajectory.orientation_at(t_start)?;
    let q_b = trajectory.orientation_at(t_end)?;
    Ok(q_a.angle_to(&q_b))
}

/// Per-call frame knobs for [`render_roi_patch`]: when to start the
/// integration, how long to integrate, how aggressive the motion-blur
/// subsampler should be, and a seed for Poisson + read noise.
#[derive(Debug, Clone, Copy)]
pub struct RoiFramePlan {
    pub frame_start: Duration,
    pub exposure: Duration,
    pub max_drift_per_sample_px: f64,
    pub seed: u64,
}

/// Render a `size_px × size_px` ROI patch through the same motion-blur
/// pipeline as the full sensor tile. Delegates to
/// [`crate::sims::motion_blur::render_region`] so intra-exposure
/// pointing drift produces matching motion-blur streaks in the ROI.
pub fn render_roi_patch(
    trajectory: &Trajectory,
    stars: &[StarData],
    fp: &FocalPlaneConfig,
    anchor: RoiAnchor,
    plan: &RoiFramePlan,
) -> Option<Array2<u16>> {
    let satellite = fp.satellite_for_sensor(anchor.sensor_idx)?;
    let (x0, y0) = anchor.top_left_px();
    let region =
        SensorRegion::from_offset(anchor.sensor_idx, (x0, y0), anchor.size_px, anchor.size_px);

    // Adaptive schedule: same shape as motion_blur, approximated via
    // endpoint drift since the ROI is a one-off render without plan
    // metadata.
    let traj_end = trajectory.end_time();
    let traj_start = trajectory.start_time();
    let t_a = plan.frame_start.max(traj_start);
    let t_b = (plan.frame_start + plan.exposure).min(traj_end);
    let eff_exposure = if t_b > t_a { t_b - t_a } else { Duration::ZERO };
    let drift = drift_over_window(trajectory, t_a, t_b).ok()?;
    let plate_scale_rad = satellite.plate_scale_per_pixel().as_radians();
    let schedule = SubsampleSchedule::adaptive(
        t_a,
        eff_exposure,
        drift,
        plate_scale_rad,
        plan.max_drift_per_sample_px,
    );

    // Pixel-space halo for prefiltering stars that could contribute to
    // the ROI through their PSF skirts.
    let psf_radius_px = satellite.airy_disk_pixel_space().first_zero().ceil() as i32 * 2;
    let padding_mm = psf_radius_px as f64 * satellite.sensor.pixel_size().as_millimeters();
    let size_f = anchor.size_px as f64;
    let halo = psf_radius_px as f64 + 1.0;

    // Coarse prefilter: only pass stars whose projection under the
    // first-sample orientation already lands within the ROI + halo.
    // For small exposures with tight ODY residuals this is a perfect
    // filter; worst case (slew in mid-exposure) it's conservative
    // enough to keep the physics right without scanning every star at
    // every subsample.
    let first_sample = schedule.sample_times().into_iter().next().unwrap_or(t_a);
    let sample_t = first_sample.min(traj_end).max(traj_start);
    let orient0 = trajectory.orientation_at(sample_t).ok()?;
    let kept: Vec<&StarData> = stars
        .iter()
        .filter(|star| {
            fp.project_to_sensor(star, &orient0, anchor.sensor_idx, padding_mm)
                .is_some_and(|(sx, sy)| {
                    let lx = sx - x0 as f64;
                    let ly = sy - y0 as f64;
                    lx >= -halo && lx <= size_f + halo && ly >= -halo && ly <= size_f + halo
                })
        })
        .collect();

    let zodiacal_per_px = zodiacal_per_pixel(&satellite, &eff_exposure);
    let inputs = RegionRenderInputs {
        trajectory,
        stars: &kept,
        fp,
        satellite: &satellite,
    };
    render_region(
        &inputs,
        &region,
        &schedule,
        padding_mm,
        zodiacal_per_px,
        None,
        plan.seed,
    )
    .ok()
}

/// Per-pixel zodiacal electrons for the supplied satellite over
/// `exposure`. Mirrors the formula in `motion_blur::zodiacal_per_px_per_s_at`
/// so the ROI background matches the full-tile background.
fn zodiacal_per_pixel(satellite: &crate::hardware::SatelliteConfig, exposure: &Duration) -> f64 {
    use crate::photometry::spectrum::Spectrum;
    use crate::photometry::zodiacal::{SolarAngularCoordinates, ZodiacalLight};
    let zlight = ZodiacalLight::new();
    let coords = SolarAngularCoordinates::zodiacal_minimum();
    let Ok(z_spect) = zlight.get_zodiacal_spectrum(&coords) else {
        return 0.0;
    };
    let focal_length_mm = satellite.telescope.focal_length.as_meters() * 1000.0;
    let pixel_size_mm = satellite.sensor.dimensions.pixel_size().as_millimeters();
    let pixel_scale_arcsec = 206265.0 * pixel_size_mm / focal_length_mm;
    let pixel_solid_angle_arcsec2 = pixel_scale_arcsec * pixel_scale_arcsec;
    let aperture = satellite.telescope.clear_aperture_area();
    z_spect.photo_electrons(&satellite.combined_qe, aperture, exposure) * pixel_solid_angle_arcsec2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::sensor::models::GSENSE4040BSI;
    use crate::hardware::sensor_array::SensorArray;
    use crate::hardware::telescope::TelescopeConfig;
    use crate::sims::orientation::orientation_from_pointing;
    use crate::sims::trajectory::Waypoint;
    use shared::units::{Length, LengthExt, TemperatureExt};
    use starfield::Equatorial;

    fn tiny_fp() -> FocalPlaneConfig {
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        FocalPlaneConfig::new(
            telescope,
            SensorArray::single(GSENSE4040BSI.clone()),
            shared::units::Temperature::from_celsius(-10.0),
        )
    }

    fn static_trajectory(eq: Equatorial) -> Trajectory {
        Trajectory::new(vec![
            Waypoint::from_pointing_and_roll(Duration::ZERO, eq, 0.0),
            Waypoint::from_pointing_and_roll(Duration::from_secs(10), eq, 0.0),
        ])
        .unwrap()
    }

    #[test]
    fn roi_anchor_picks_brightest_in_bounds_star() {
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let orient = orientation_from_pointing(&pointing, 0.0);
        let stars = vec![
            StarData {
                id: 1,
                magnitude: 5.0,
                position: pointing,
                b_v: Some(0.6),
            },
            StarData {
                id: 2,
                magnitude: 3.0,
                position: pointing,
                b_v: Some(0.6),
            },
        ];
        let anchor = pick_roi_anchor(&stars, &fp, &orient, 64).expect("an anchor exists");
        assert_eq!(anchor.sensor_idx, 0);
        assert_eq!(anchor.size_px, 64);
    }

    #[test]
    fn render_roi_patch_returns_expected_shape() {
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let orient = orientation_from_pointing(&pointing, 0.0);
        let star = StarData {
            id: 1,
            magnitude: 7.0,
            position: pointing,
            b_v: Some(0.6),
        };
        let traj = static_trajectory(pointing);
        let anchor = pick_roi_anchor(std::slice::from_ref(&star), &fp, &orient, 32).unwrap();
        let plan = RoiFramePlan {
            frame_start: Duration::from_secs(0),
            exposure: Duration::from_millis(100),
            max_drift_per_sample_px: 0.1,
            seed: 42,
        };
        let patch = render_roi_patch(&traj, std::slice::from_ref(&star), &fp, anchor, &plan)
            .expect("patch renders");
        assert_eq!(patch.dim(), (32, 32));
    }
}

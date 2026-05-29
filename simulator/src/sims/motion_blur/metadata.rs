//! Assembly of the per-render `metadata.json` document.
//!
//! Turns a completed [`RenderScene`] plus its per-frame schedules into the
//! [`RenderMetadata`] serialized alongside the rendered PNGs.

use std::collections::BTreeMap;

use chrono::SecondsFormat;

use crate::hardware::SatelliteConfig;
use crate::scene_galaxy::Galaxy;
use crate::sims::motion_blur_metadata::{
    sensor_dir_name, sensor_relative_png_path, EquatorialMeta, FrameMeta, HardwareMeta,
    RenderConfigMeta, RenderMetadata, SensorMeta, StarMeta, TrajectoryMeta, WaypointMeta,
    ZodiacalMeta,
};
use crate::sims::orientation::{boresight_of, roll_of};
use crate::sims::trajectory::TrajectoryError;

use super::render::{MotionBlurConfig, RenderScene};
use super::schedule::SubsampleSchedule;

pub(super) fn build_render_metadata(
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
            elongation_deg: scene.sources.zodiacal.elongation(),
            latitude_deg: scene.sources.zodiacal.latitude(),
        },
    };

    Ok(RenderMetadata {
        version: "1.1".to_string(),
        rendered_at,
        trajectory: trajectory_meta,
        frames,
        stars,
        galaxies,
        hardware,
        render_config,
    })
}

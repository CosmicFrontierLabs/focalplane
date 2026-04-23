use clap::{Parser, ValueEnum};
use log::{info, warn};
use rayon::prelude::*;
use simulator::hardware::satellite::FocalPlaneConfig;
use simulator::hardware::sensor_array::{SensorArray, SPENCER_ARRAY_PLAN};
use simulator::shared_args::{DurationArg, SensorModel, SharedSimulationArgs};
use simulator::sims::context_render::{render_context_frame, ContextRenderConfig};
use simulator::sims::trajectory::{
    fov_envelope, render_trajectory, Trajectory, TrajectoryRenderConfig, Waypoint,
};
use simulator::star_math::field_diameter_for_array;
use simulator::units::{AngleExt, LengthExt, TemperatureExt};
use starfield::catalogs::StarCatalog;
use starfield::Equatorial;
use std::path::Path;
use std::time::Instant;

/// Focal-plane array layout. `Single` is the default (one sensor at
/// boresight); `Spencer` is the hard-coded four-IMX455 mosaic defined
/// in `hardware::sensor_array::SPENCER_ARRAY_PLAN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ArrayFormat {
    /// Single sensor centered on the telescope boresight. Uses `--sensor`.
    Single,
    /// Four-IMX455 mosaic. `--sensor` is ignored when this is selected.
    Spencer,
}

/// Parse a "WxH" size string into a `(width, height)` pair.
fn parse_size(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected format 'WxH' (e.g. '3840x2160'), got '{s}'"))?;
    let width: u32 = w
        .trim()
        .parse()
        .map_err(|e| format!("invalid width '{w}': {e}"))?;
    let height: u32 = h
        .trim()
        .parse()
        .map_err(|e| format!("invalid height '{h}': {e}"))?;
    if width == 0 || height == 0 {
        return Err("width and height must be > 0".into());
    }
    Ok((width, height))
}

/// Parse coordinates string in format "ra,dec" (degrees)
fn parse_ra_dec(s: &str) -> Result<Equatorial, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err("Coordinates must be in format 'ra,dec' (degrees)".to_string());
    }
    let ra = parts[0]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid RA value".to_string())?;
    let dec = parts[1]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid Dec value".to_string())?;
    if !(0.0..360.0).contains(&ra) {
        return Err("RA must be in range [0, 360) degrees".to_string());
    }
    if !(-90.0..=90.0).contains(&dec) {
        return Err("Dec must be in range [-90, 90] degrees".to_string());
    }
    Ok(Equatorial::from_degrees(ra, dec))
}

#[derive(Parser, Debug)]
#[command(
    name = "Motion Simulator",
    about = "Render a sequence of 16-bit PNG frames along a trajectory",
    long_about = "Renders a moving image stream for a focal plane sensor array.\n\n\
        The telescope sweeps from a start pointing to an end pointing over \
        a specified duration. The star catalog is queried once with a broad \
        field of view encompassing the entire trajectory, then each frame \
        is rendered by re-projecting cached stars at the interpolated pointing.\n\n\
        Output layout under --output-dir:\n  \
        - metadata.json at the root describing the run,\n  \
        - one sensor_NN/ subdirectory per sensor, each containing \
          frame_NNNNNN.png files (16-bit grayscale PNG)."
)]
struct Args {
    #[command(flatten)]
    shared: SharedSimulationArgs,

    #[arg(
        long,
        default_value_t = SensorModel::Imx455,
        help = "Sensor model for the focal plane (ignored when --array-format is not 'single')"
    )]
    sensor: SensorModel,

    #[arg(
        long,
        value_enum,
        default_value_t = ArrayFormat::Single,
        help = "Focal-plane array layout"
    )]
    array_format: ArrayFormat,

    #[arg(
        long,
        value_parser = parse_ra_dec,
        help = "Start pointing in 'ra,dec' degrees (e.g., '56.75,24.12')"
    )]
    start: Equatorial,

    #[arg(
        long,
        value_parser = parse_ra_dec,
        help = "End pointing in 'ra,dec' degrees (e.g., '57.0,24.5')"
    )]
    end: Equatorial,

    #[arg(
        long,
        default_value = "10s",
        help = "Total trajectory duration (e.g., '10s', '1m')"
    )]
    duration: DurationArg,

    #[arg(
        long,
        default_value = "0",
        allow_hyphen_values = true,
        help = "Roll angle (degrees) at the start waypoint"
    )]
    start_roll: f64,

    #[arg(
        long,
        default_value = "0",
        allow_hyphen_values = true,
        help = "Roll angle (degrees) at the end waypoint"
    )]
    end_roll: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        allow_hyphen_values = true,
        help = "Extra full roll rotations (in addition to end_roll - start_roll). \
                Nonzero values automatically subdivide the trajectory into \
                multiple waypoints so the roll sweeps through smoothly rather \
                than collapsing to the shortest quaternion path."
    )]
    roll_turns: f64,

    #[arg(
        long,
        default_value = "1s",
        help = "Time between frames (e.g., '1s', '500ms')"
    )]
    timestep: DurationArg,

    #[arg(
        long,
        default_value = "trajectory_output",
        help = "Output directory for PNG frame sequence"
    )]
    output_dir: String,

    #[arg(long, default_value_t = 42, help = "Random seed for noise generation")]
    seed: u64,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Per-subsample drift budget in pixels for adaptive motion-blur scheduling"
    )]
    max_drift_per_sample_px: f64,

    #[arg(
        long,
        default_value_t = false,
        help = "Force N=1 subsample per frame (disables motion blur; for debugging)"
    )]
    force_static: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Suppress the indicatif progress bar; INFO logs still emit"
    )]
    quiet: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Also render a context-view PNG per frame (under <output-dir>/context/)"
    )]
    context_view: bool,

    #[arg(
        long,
        value_parser = parse_size,
        default_value = "3840x2160",
        help = "Context view image dimensions, formatted 'WxH' in pixels"
    )]
    context_size: (u32, u32),

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Per-edge padding fraction for the context view (0.05 = 5% on each side)"
    )]
    context_padding: f64,

    #[arg(
        long,
        default_value_t = false,
        help = "Skip the per-sensor render and only emit context-view frames \
                (implies --context-view)"
    )]
    context_only: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();
    let wallclock = Instant::now();

    let telescope = args.shared.telescope.to_config();

    info!(
        "Telescope: {} (aperture={:.3}m, f/{:.1})",
        telescope.name,
        telescope.aperture.as_meters(),
        telescope.f_number()
    );

    let array = match args.array_format {
        ArrayFormat::Single => {
            let sensor = args.sensor.to_config();
            info!("Array: single-sensor, boresight-centered ({})", sensor.name);
            SensorArray::single(sensor.clone())
        }
        ArrayFormat::Spencer => {
            if !matches!(args.sensor, SensorModel::Imx455) {
                warn!(
                    "--array-format=spencer is IMX455-only; ignoring --sensor={:?}",
                    args.sensor
                );
            }
            info!("Array: Spencer (4x IMX455)");
            SPENCER_ARRAY_PLAN.clone()
        }
    };
    let focal_plane = FocalPlaneConfig::new(
        telescope.clone(),
        array,
        simulator::units::Temperature::from_celsius(args.shared.temperature),
    );

    let total_roll_deg = (args.end_roll - args.start_roll) + 360.0 * args.roll_turns;
    let trajectory = if args.roll_turns.abs() > f64::EPSILON {
        // Multi-waypoint trajectory: subdivide so each SLERP segment sweeps
        // at most 90° of roll. Necessary because a two-waypoint trajectory
        // cannot represent a > 180° quaternion path — SLERP collapses to
        // the shortest arc, losing whole rotations.
        let segments = ((total_roll_deg.abs() / 90.0).ceil() as usize).max(1);
        let duration_s = args.duration.0.as_secs_f64();
        let waypoints: Vec<Waypoint> = (0..=segments)
            .map(|i| {
                let frac = i as f64 / segments as f64;
                let t = std::time::Duration::from_secs_f64(duration_s * frac);
                let ra = args.start.ra_degrees()
                    + frac * (args.end.ra_degrees() - args.start.ra_degrees());
                let dec = args.start.dec_degrees()
                    + frac * (args.end.dec_degrees() - args.start.dec_degrees());
                let eq = Equatorial::from_degrees(ra, dec);
                let roll_deg = args.start_roll + frac * total_roll_deg;
                Waypoint::from_pointing_and_roll(t, eq, roll_deg.to_radians())
            })
            .collect();
        Trajectory::new(waypoints)?
    } else if args.start_roll == 0.0 && args.end_roll == 0.0 {
        Trajectory::from_endpoints(args.start, args.end, args.duration.0)?
    } else {
        Trajectory::from_endpoints_with_roll(
            args.start,
            args.start_roll.to_radians(),
            args.end,
            args.end_roll.to_radians(),
            args.duration.0,
        )?
    };

    let base_fov_deg = field_diameter_for_array(&focal_plane)
        .map(|a| a.as_degrees())
        .ok_or("Empty focal plane")?;

    let (envelope_center, envelope_diameter) = fov_envelope(&trajectory, base_fov_deg);

    info!(
        "Trajectory: RA {:.4}° Dec {:.4}° → RA {:.4}° Dec {:.4}° over {:.1}s",
        args.start.ra_degrees(),
        args.start.dec_degrees(),
        args.end.ra_degrees(),
        args.end.dec_degrees(),
        args.duration.0.as_secs_f64()
    );
    info!(
        "Roll: start {:.3}° → end {:.3}°",
        args.start_roll, args.end_roll
    );
    info!(
        "FOV envelope: center RA {:.4}° Dec {:.4}°, diameter {:.4}°",
        envelope_center.ra_degrees(),
        envelope_center.dec_degrees(),
        envelope_diameter
    );

    info!("Loading catalog...");
    let catalog = args.shared.load_catalog()?;

    let stars = catalog.stars_in_field(
        envelope_center.ra_degrees(),
        envelope_center.dec_degrees(),
        envelope_diameter,
    );
    info!("Cached {} stars for trajectory envelope", stars.len());

    let output_path = Path::new(&args.output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path)?;
    }

    if !args.context_only {
        let render_config = TrajectoryRenderConfig {
            trajectory: &trajectory,
            timestep: args.timestep.0,
            exposure: args.shared.exposure.0,
            focal_plane: &focal_plane,
            catalog_stars: &stars,
            zodiacal: args.shared.coordinates,
            output_dir: output_path,
            base_seed: Some(args.seed),
            max_drift_per_sample_px: Some(args.max_drift_per_sample_px),
            force_static: args.force_static,
            quiet: args.quiet,
            telescope_name: telescope.name.clone(),
            catalog_path: args.shared.catalog.clone(),
            temperature_c: args.shared.temperature,
        };
        let frame_count = render_trajectory(&render_config)?;

        let elapsed = wallclock.elapsed();
        info!(
            "Rendered {} frames in {:.2}s ({:.1} fps)",
            frame_count,
            elapsed.as_secs_f64(),
            frame_count as f64 / elapsed.as_secs_f64()
        );
        info!(
            "Output: {} (metadata.json + sensor_NN/frame_NNNNNN.png layout)",
            args.output_dir
        );
    } else {
        info!("--context-only set; skipping sensor tile render");
    }

    if args.context_view || args.context_only {
        let context_dir = output_path.join("context");
        std::fs::create_dir_all(&context_dir)?;
        let (ctx_w, ctx_h) = args.context_size;
        let ctx_cfg = ContextRenderConfig {
            width: ctx_w,
            height: ctx_h,
            padding_fraction: args.context_padding,
            ..Default::default()
        };
        let frame_times = trajectory.frame_times(args.timestep.0);
        info!(
            "Rendering {} context-view frames at {}x{} into {}/",
            frame_times.len(),
            ctx_cfg.width,
            ctx_cfg.height,
            context_dir.display()
        );
        let context_started = Instant::now();
        let results: Vec<Result<(), String>> = frame_times
            .par_iter()
            .enumerate()
            .map(|(idx, t)| -> Result<(), String> {
                let orientation = trajectory.orientation_at(*t).map_err(|e| e.to_string())?;
                let path = context_dir.join(format!("frame_{idx:06}.png"));
                render_context_frame(&orientation, &stars, &focal_plane, &ctx_cfg, &path)
                    .map_err(|e| e.to_string())
            })
            .collect();
        for r in results {
            r.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        }
        info!(
            "Context views rendered in {:.2}s",
            context_started.elapsed().as_secs_f64()
        );
    }

    Ok(())
}

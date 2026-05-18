use clap::{Parser, ValueEnum};
use log::{info, warn};
use rayon::prelude::*;
use simulator::hardware::satellite::FocalPlaneConfig;
use simulator::hardware::sensor_array::{SensorArray, SPENCER_ARRAY_PLAN};
use simulator::shared_args::{DurationArg, SensorModel, SharedSimulationArgs};
use simulator::sims::context_render::{render_context_frame, ContextRenderConfig};
use simulator::sims::jitter::build_pink_trajectory;
use simulator::sims::trajectory::{
    fov_envelope, prefetch_catalog_for_envelope, render_trajectory, Trajectory,
    TrajectoryRenderConfig, Waypoint,
};
use simulator::star_math::field_diameter_for_array;
use simulator::units::{AngleExt, LengthExt, TemperatureExt};
use starfield::catalogs::StarCatalog;
use starfield::Equatorial;
use starfield_gaia::{Dr3, LazyLoadingCatalog};
use std::path::Path;
use std::time::Instant;

/// Build a circular-sweep trajectory of radius `radius_deg` around
/// `pointing`, tracing `turns` full loops over `duration` at
/// `waypoints_per_turn` waypoints per turn. RA offsets are scaled by
/// `1/cos(dec)` so the motion is geometric on the sky.
fn build_circle_trajectory(
    pointing: Equatorial,
    radius_deg: f64,
    turns: f64,
    waypoints_per_turn: usize,
    duration: std::time::Duration,
    start_roll_deg: f64,
    total_roll_deg: f64,
) -> Result<Trajectory, Box<dyn std::error::Error>> {
    let turns = turns.max(f64::EPSILON);
    let segments = ((waypoints_per_turn as f64 * turns).ceil() as usize).max(8);
    let duration_s = duration.as_secs_f64();
    let ra0 = pointing.ra_degrees();
    let dec0 = pointing.dec_degrees();
    let cos_dec0 = dec0.to_radians().cos().max(1e-6);
    let waypoints: Vec<Waypoint> = (0..=segments)
        .map(|i| {
            let frac = i as f64 / segments as f64;
            let t = std::time::Duration::from_secs_f64(duration_s * frac);
            let theta = 2.0 * std::f64::consts::PI * turns * frac;
            let ra = ra0 + (radius_deg * theta.cos()) / cos_dec0;
            let dec = dec0 + radius_deg * theta.sin();
            let eq = Equatorial::from_degrees(ra, dec);
            let roll_deg = start_roll_deg + frac * total_roll_deg;
            Waypoint::from_pointing_and_roll(t, eq, roll_deg.to_radians())
        })
        .collect();
    Ok(Trajectory::new(waypoints)?)
}

/// Build a linear-sweep trajectory from `start` to `end` over
/// `duration`. When `total_roll_deg.abs() > 0`, subdivides into
/// multiple waypoints so SLERP sweeps through every 90° of roll rather
/// than collapsing to the shortest-arc quaternion path.
fn build_line_trajectory(
    start: Equatorial,
    end: Equatorial,
    duration: std::time::Duration,
    start_roll_deg: f64,
    end_roll_deg: f64,
    total_roll_deg: f64,
) -> Result<Trajectory, Box<dyn std::error::Error>> {
    if total_roll_deg.abs() > f64::EPSILON {
        let segments = ((total_roll_deg.abs() / 90.0).ceil() as usize).max(1);
        let duration_s = duration.as_secs_f64();
        let waypoints: Vec<Waypoint> = (0..=segments)
            .map(|i| {
                let frac = i as f64 / segments as f64;
                let t = std::time::Duration::from_secs_f64(duration_s * frac);
                let ra = start.ra_degrees() + frac * (end.ra_degrees() - start.ra_degrees());
                let dec = start.dec_degrees() + frac * (end.dec_degrees() - start.dec_degrees());
                let eq = Equatorial::from_degrees(ra, dec);
                let roll_deg = start_roll_deg + frac * total_roll_deg;
                Waypoint::from_pointing_and_roll(t, eq, roll_deg.to_radians())
            })
            .collect();
        return Ok(Trajectory::new(waypoints)?);
    }
    if start_roll_deg == 0.0 && end_roll_deg == 0.0 {
        Ok(Trajectory::from_endpoints(start, end, duration)?)
    } else {
        Ok(Trajectory::from_endpoints_with_roll(
            start,
            start_roll_deg.to_radians(),
            end,
            end_roll_deg.to_radians(),
            duration,
        )?)
    }
}

/// Load a trajectory from a simple CSV of absolute orientations.
/// Expected columns (by header, case-sensitive): `time_s`, `qw`, `qx`,
/// `qy`, `qz`. Extra columns are ignored. Quaternions are normalized on
/// load; non-monotonic `time_s` is rejected.
fn build_csv_trajectory(path: &std::path::Path) -> Result<Trajectory, Box<dyn std::error::Error>> {
    use nalgebra::{Quaternion, UnitQuaternion};
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
    let mut lines = BufReader::new(file).lines();

    let header = lines
        .next()
        .ok_or_else(|| format!("{} is empty", path.display()))?
        .map_err(|e| e.to_string())?;
    let columns: Vec<&str> = header.split(',').map(str::trim).collect();
    let idx = |name: &str| -> Result<usize, String> {
        columns
            .iter()
            .position(|c| *c == name)
            .ok_or_else(|| format!("{} is missing column `{name}`", path.display()))
    };
    let (i_t, i_qw, i_qx, i_qy, i_qz) = (
        idx("time_s")?,
        idx("qw")?,
        idx("qx")?,
        idx("qy")?,
        idx("qz")?,
    );

    let mut waypoints: Vec<Waypoint> = Vec::new();
    let mut last_t = f64::NEG_INFINITY;
    for (line_no, line) in lines.enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        let parse = |i: usize, name: &str| -> Result<f64, String> {
            cells
                .get(i)
                .ok_or_else(|| format!("line {}: missing `{name}` cell", line_no + 2))?
                .parse::<f64>()
                .map_err(|e| format!("line {}: bad `{name}` value: {e}", line_no + 2))
        };
        let t_s = parse(i_t, "time_s")?;
        if t_s <= last_t {
            return Err(format!(
                "{}: `time_s` must be strictly increasing (line {} has {} ≤ previous {})",
                path.display(),
                line_no + 2,
                t_s,
                last_t,
            )
            .into());
        }
        last_t = t_s;

        let q = UnitQuaternion::from_quaternion(Quaternion::new(
            parse(i_qw, "qw")?,
            parse(i_qx, "qx")?,
            parse(i_qy, "qy")?,
            parse(i_qz, "qz")?,
        ));
        waypoints.push(Waypoint::new(
            std::time::Duration::from_secs_f64(t_s.max(0.0)),
            q,
        ));
    }

    if waypoints.len() < 2 {
        return Err(format!(
            "{}: need at least 2 rows, found {}",
            path.display(),
            waypoints.len()
        )
        .into());
    }
    Ok(Trajectory::new(waypoints)?)
}

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

/// CLI-facing value enum for `--mode`. Must be unit variants because
/// clap's `ValueEnum` only supports that shape; see [`TrajectoryKind`]
/// for the resolved, data-carrying counterpart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TrajectoryMode {
    /// Linear sweep from `--start` to `--end`.
    Line,
    /// Circle in RA/Dec around `--start`; uses `--circle-radius-deg`,
    /// `--circle-turns`, `--circle-waypoints`.
    Circle,
    /// Pink-spectrum pointing residual around `--start`; uses
    /// `--pink-mean-arcsec`, `--pink-waypoints`, `--pink-seed`.
    Pink,
    /// Trajectory loaded from a CSV of (time_s, qw, qx, qy, qz) rows
    /// via `--trajectory-csv`. `--start` is not required; the CSV
    /// supplies absolute orientations.
    Csv,
}

/// Resolved trajectory description with mode-specific parameters
/// already pulled out of the flat CLI [`Args`]. Produced by
/// [`TrajectoryKind::from_args`] and consumed by [`TrajectoryKind::build`]
/// so the dispatch in `main` has a single `match`.
#[derive(Debug, Clone)]
enum TrajectoryKind {
    Line {
        start: Equatorial,
        end: Equatorial,
    },
    Circle {
        start: Equatorial,
        radius_deg: f64,
        turns: f64,
        waypoints_per_turn: usize,
    },
    Pink {
        start: Equatorial,
        mean_arcsec: f64,
        waypoints: usize,
        seed: u64,
    },
    Csv {
        path: std::path::PathBuf,
    },
}

impl TrajectoryKind {
    /// Pull the mode-specific params out of the flat `Args`. clap's
    /// `required_if_eq` guarantees the relevant `Option`s are `Some`
    /// by the time we get here, so the `expect` messages just describe
    /// the invariant rather than running in practice.
    fn from_args(args: &Args) -> Self {
        let start = || {
            args.start.expect(
                "clap required_unless_present guarantees --start for line/circle/pink modes",
            )
        };
        match args.mode {
            TrajectoryMode::Line => TrajectoryKind::Line {
                start: start(),
                end: args
                    .end
                    .expect("clap required_if_eq guarantees --end when --mode=line"),
            },
            TrajectoryMode::Circle => TrajectoryKind::Circle {
                start: start(),
                radius_deg: args.circle_radius_deg.expect(
                    "clap required_if_eq guarantees --circle-radius-deg when --mode=circle",
                ),
                turns: args.circle_turns,
                waypoints_per_turn: args.circle_waypoints,
            },
            TrajectoryMode::Pink => TrajectoryKind::Pink {
                start: start(),
                mean_arcsec: args
                    .pink_mean_arcsec
                    .expect("clap required_if_eq guarantees --pink-mean-arcsec when --mode=pink"),
                waypoints: args.pink_waypoints,
                seed: args.pink_seed,
            },
            TrajectoryMode::Csv => TrajectoryKind::Csv {
                path: args
                    .trajectory_csv
                    .clone()
                    .expect("clap required_if_eq guarantees --trajectory-csv when --mode=csv"),
            },
        }
    }

    /// Dispatch to the matching builder.
    fn build(
        &self,
        duration: std::time::Duration,
        start_roll_deg: f64,
        end_roll_deg: f64,
        total_roll_deg: f64,
    ) -> Result<Trajectory, Box<dyn std::error::Error>> {
        match self {
            TrajectoryKind::Line { start, end } => build_line_trajectory(
                *start,
                *end,
                duration,
                start_roll_deg,
                end_roll_deg,
                total_roll_deg,
            ),
            TrajectoryKind::Circle {
                start,
                radius_deg,
                turns,
                waypoints_per_turn,
            } => build_circle_trajectory(
                *start,
                *radius_deg,
                *turns,
                *waypoints_per_turn,
                duration,
                start_roll_deg,
                total_roll_deg,
            ),
            TrajectoryKind::Pink {
                start,
                mean_arcsec,
                waypoints,
                seed,
            } => build_pink_trajectory(
                *start,
                *mean_arcsec,
                *waypoints,
                *seed,
                duration,
                start_roll_deg,
                total_roll_deg,
            )
            .map_err(Into::into),
            TrajectoryKind::Csv { path } => build_csv_trajectory(path),
        }
    }
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
        value_enum,
        help = "Trajectory shape: line | circle | pink (no default — clap's \
                `required_if_eq` won't fire on defaulted values, so the \
                caller must pick explicitly)"
    )]
    mode: TrajectoryMode,

    #[arg(
        long,
        value_parser = parse_ra_dec,
        required_unless_present = "trajectory_csv",
        help = "Start pointing in 'ra,dec' degrees (e.g., '56.75,24.12'); \
                not required when --mode=csv"
    )]
    start: Option<Equatorial>,

    #[arg(
        long,
        value_parser = parse_ra_dec,
        required_if_eq("mode", "line"),
        help = "End pointing in 'ra,dec' degrees (required when --mode=line)"
    )]
    end: Option<Equatorial>,

    #[arg(
        long,
        required_if_eq("mode", "circle"),
        help = "Circle radius in degrees (required when --mode=circle)"
    )]
    circle_radius_deg: Option<f64>,

    #[arg(
        long,
        default_value_t = 256,
        help = "Number of waypoints to subdivide the circle into (one period = \
                --circle-turns full loops)"
    )]
    circle_waypoints: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Number of full circle loops over the trajectory duration"
    )]
    circle_turns: f64,

    #[arg(
        long,
        required_if_eq("mode", "pink"),
        help = "Mean pointing-residual magnitude in arcseconds (required when --mode=pink)"
    )]
    pink_mean_arcsec: Option<f64>,

    #[arg(
        long,
        default_value_t = 1024,
        help = "Waypoint count for pink-noise trajectory (higher = more high-frequency content captured)"
    )]
    pink_waypoints: usize,

    #[arg(long, default_value_t = 0, help = "RNG seed for pink-noise trajectory")]
    pink_seed: u64,

    #[arg(
        long,
        required_if_eq("mode", "csv"),
        help = "Path to a CSV file with columns time_s, qw, qx, qy, qz \
                (required when --mode=csv)"
    )]
    trajectory_csv: Option<std::path::PathBuf>,

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
        help = "Per-stamp drift budget in pixels. The renderer schedules \
                ceil(per-frame angular path length / (this * pixel_scale)) \
                stratified-MC PSF stamps per exposure. Tighten to capture \
                high-frequency jitter (e.g. PSD-derived reaction-wheel \
                residuals)."
    )]
    max_drift_per_stamp_px: f64,

    #[arg(
        long,
        default_value_t = false,
        help = "Force a single PSF stamp per frame (disables motion blur; for debugging)"
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

    #[arg(
        long,
        help = "Path to NSA FITS file for galaxy rendering. If unset and \
                --galaxies is on, defaults to ~/.cache/starfield/nsa/nsa_v0_1_2.fits \
                (downloaded on demand via the NYU mirror if absent)."
    )]
    galaxy_catalog: Option<std::path::PathBuf>,

    #[arg(
        long,
        default_value_t = false,
        help = "Render NSA galaxies in the field. Default off — turns on \
                NSA loading + Sersic deposit splatting through the standard \
                renderer pathway."
    )]
    galaxies: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Load stars from a Gaia DR3 healpix-sharded excerpt directory \
                (per-source BP-RP color → B-V via the linear fit, plus \
                Hipparcos bright-star augmentation). Overrides --catalog when \
                set."
    )]
    gaia_dr3: bool,

    #[arg(
        long,
        help = "Directory holding the healpix-sharded DR3 excerpt. Defaults \
                to ~/.cache/starfield/gaia-excerpts/dr3-mag20/."
    )]
    gaia_excerpt_dir: Option<std::path::PathBuf>,

    #[arg(
        long,
        default_value_t = 19.0,
        help = "Magnitude cutoff for Gaia DR3 cone loader (G band). Default \
                matches the historical to_mag_19.bin cutoff."
    )]
    gaia_mag_limit: f64,
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
    let kind = TrajectoryKind::from_args(&args);
    let trajectory = kind.build(
        args.duration.0,
        args.start_roll,
        args.end_roll,
        total_roll_deg,
    )?;

    let base_fov_deg = field_diameter_for_array(&focal_plane)
        .map(|a| a.as_degrees())
        .ok_or("Empty focal plane")?;

    let (envelope_center, envelope_diameter) = fov_envelope(&trajectory, base_fov_deg);

    match &kind {
        TrajectoryKind::Line { start, end } => {
            info!(
                "Trajectory: line RA {:.4}° Dec {:.4}° → RA {:.4}° Dec {:.4}° over {:.1}s",
                start.ra_degrees(),
                start.dec_degrees(),
                end.ra_degrees(),
                end.dec_degrees(),
                args.duration.0.as_secs_f64()
            );
        }
        TrajectoryKind::Circle {
            start,
            radius_deg,
            turns,
            ..
        } => {
            info!(
                "Trajectory: circle centered on RA {:.4}° Dec {:.4}°, radius {:.4}°, \
                 {:.2} turns over {:.1}s ({} waypoints)",
                start.ra_degrees(),
                start.dec_degrees(),
                radius_deg,
                turns,
                args.duration.0.as_secs_f64(),
                trajectory.waypoints().len()
            );
        }
        TrajectoryKind::Pink {
            start,
            mean_arcsec,
            seed,
            ..
        } => {
            info!(
                "Trajectory: pink-noise residuals around RA {:.4}° Dec {:.4}°, \
                 mean |offset|={:.2}″, {} waypoints over {:.1}s (seed={})",
                start.ra_degrees(),
                start.dec_degrees(),
                mean_arcsec,
                trajectory.waypoints().len(),
                args.duration.0.as_secs_f64(),
                seed,
            );
        }
        TrajectoryKind::Csv { path } => {
            info!(
                "Trajectory: loaded {} waypoints from {} spanning {:.3}s",
                trajectory.waypoints().len(),
                path.display(),
                trajectory.end_time().as_secs_f64() - trajectory.start_time().as_secs_f64(),
            );
        }
    }
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

    let stars: Vec<starfield::catalogs::StarData> = if args.gaia_dr3 {
        let excerpt_dir = args
            .gaia_excerpt_dir
            .clone()
            .unwrap_or_else(simulator::sims::gaia_dr3::default_excerpt_dir);
        info!(
            "Loading Gaia DR3 cone from {} (mag <= {:.2}, augmented with Hipparcos bright \
             supplement)...",
            excerpt_dir.display(),
            args.gaia_mag_limit
        );
        let lazy = LazyLoadingCatalog::<Dr3>::open(&excerpt_dir)?;
        let (cat, _augmented) = simulator::sims::gaia_dr3::materialize_cone_augmented(
            &lazy,
            envelope_center,
            envelope_diameter * 0.5,
            args.gaia_mag_limit,
        )?;
        cat.star_data().collect()
    } else {
        info!("Loading catalog...");
        let catalog = args.shared.load_catalog_with_progress(args.quiet)?;
        prefetch_catalog_for_envelope(&catalog, (envelope_center, envelope_diameter))
    };
    info!("Cached {} stars for trajectory envelope", stars.len());

    // NSA galaxies (optional). Pre-projected once at the envelope
    // centre — the per-tile renderer uses the same per-sensor list
    // for every frame. See `sims::nsa_galaxies` for the loader
    // documentation. `galaxies_in_field` is a flat sky-position list
    // for the context-view ellipse overlay (it covers galaxies that
    // land in sensor gaps too).
    let (per_sensor_galaxies, galaxies_in_field) = if args.galaxies {
        let path = args
            .galaxy_catalog
            .clone()
            .unwrap_or_else(simulator::sims::nsa_galaxies::default_nsa_path);
        let cfg = simulator::sims::nsa_galaxies::GalaxyLoaderConfig::default();
        // Use the envelope diameter / 2 as a generous FOV radius
        // around the trajectory centre. The loader applies its own
        // padding on top.
        let per_sensor = simulator::sims::nsa_galaxies::load_and_route_nsa_galaxies(
            &path,
            &envelope_center,
            &focal_plane,
            envelope_diameter * 0.5,
            &cfg,
        )?;
        let in_field = simulator::sims::nsa_galaxies::load_galaxies_in_fov(
            &path,
            &envelope_center,
            envelope_diameter * 0.5,
            &cfg,
        )?;
        (per_sensor, in_field)
    } else {
        (
            vec![Vec::new(); focal_plane.array.sensor_count()],
            Vec::new(),
        )
    };

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
            per_sensor_galaxies: &per_sensor_galaxies,
            zodiacal: args.shared.coordinates,
            output_dir: output_path,
            base_seed: Some(args.seed),
            max_drift_per_stamp_px: Some(args.max_drift_per_stamp_px),
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
                render_context_frame(
                    &orientation,
                    &stars,
                    &galaxies_in_field,
                    &focal_plane,
                    &ctx_cfg,
                    &path,
                )
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use clap::Parser;

    /// Shared baseline args with a pinned trajectory mode. Every test
    /// supplies its own `--mode ...` override because `Args` has no
    /// default for `--mode`.
    fn base(mode: &'static str) -> Vec<&'static str> {
        vec![
            "motion_simulator",
            "--start",
            "213.39,-55.86",
            "--duration",
            "10s",
            "--catalog",
            "fake.bin",
            "--output-dir",
            "/tmp/nope",
            "--mode",
            mode,
        ]
    }

    #[test]
    fn mode_is_required() {
        let argv = [
            "motion_simulator",
            "--start",
            "0,0",
            "--duration",
            "1s",
            "--catalog",
            "fake.bin",
            "--output-dir",
            "/tmp/x",
        ];
        let err = Args::try_parse_from(argv).expect_err("missing --mode must fail");
        assert!(
            err.to_string().contains("--mode"),
            "clap error should mention --mode, got: {err}"
        );
    }

    #[test]
    fn line_mode_requires_end() {
        let err = Args::try_parse_from(base("line")).expect_err("missing --end must fail");
        assert!(
            err.to_string().contains("--end"),
            "clap error should mention --end, got: {err}"
        );
    }

    #[test]
    fn line_mode_with_end_parses() {
        let mut argv = base("line");
        argv.extend(["--end", "213.50,-55.70"]);
        let args = Args::try_parse_from(argv).expect("line + --end parses");
        assert_eq!(args.mode, TrajectoryMode::Line);
        assert_relative_eq!(args.end.unwrap().ra_degrees(), 213.50, epsilon = 1e-9);
    }

    #[test]
    fn circle_mode_requires_radius() {
        let err = Args::try_parse_from(base("circle"))
            .expect_err("missing --circle-radius-deg must fail");
        assert!(
            err.to_string().contains("circle-radius-deg"),
            "clap error should mention --circle-radius-deg, got: {err}"
        );
    }

    #[test]
    fn circle_mode_with_radius_parses_and_ignores_end() {
        let mut argv = base("circle");
        argv.extend(["--circle-radius-deg", "0.1"]);
        let args = Args::try_parse_from(argv).expect("circle + radius parses");
        assert_eq!(args.mode, TrajectoryMode::Circle);
        assert_relative_eq!(args.circle_radius_deg.unwrap(), 0.1, epsilon = 1e-9);
        assert!(args.end.is_none(), "--end not required for circle mode");
    }

    #[test]
    fn pink_mode_requires_mean_arcsec() {
        let err =
            Args::try_parse_from(base("pink")).expect_err("missing --pink-mean-arcsec must fail");
        assert!(
            err.to_string().contains("pink-mean-arcsec"),
            "clap error should mention --pink-mean-arcsec, got: {err}"
        );
    }

    #[test]
    fn pink_mode_with_mean_arcsec_parses() {
        let mut argv = base("pink");
        argv.extend(["--pink-mean-arcsec", "6.0", "--pink-seed", "42"]);
        let args = Args::try_parse_from(argv).expect("pink + mean parses");
        assert_eq!(args.mode, TrajectoryMode::Pink);
        assert_relative_eq!(args.pink_mean_arcsec.unwrap(), 6.0, epsilon = 1e-9);
        assert_eq!(args.pink_seed, 42);
        assert!(args.end.is_none(), "--end not required for pink mode");
    }

    #[test]
    fn trajectory_kind_from_args_matches_mode() {
        // Line.
        let mut argv = base("line");
        argv.extend(["--end", "213.50,-55.70"]);
        let args = Args::try_parse_from(argv).unwrap();
        assert!(matches!(
            TrajectoryKind::from_args(&args),
            TrajectoryKind::Line { .. }
        ));

        // Circle.
        let mut argv = base("circle");
        argv.extend(["--circle-radius-deg", "0.2"]);
        let args = Args::try_parse_from(argv).unwrap();
        match TrajectoryKind::from_args(&args) {
            TrajectoryKind::Circle { radius_deg, .. } => {
                assert_relative_eq!(radius_deg, 0.2, epsilon = 1e-9);
            }
            other => panic!("expected Circle, got {other:?}"),
        }

        // Pink.
        let mut argv = base("pink");
        argv.extend(["--pink-mean-arcsec", "6.0"]);
        let args = Args::try_parse_from(argv).unwrap();
        match TrajectoryKind::from_args(&args) {
            TrajectoryKind::Pink { mean_arcsec, .. } => {
                assert_relative_eq!(mean_arcsec, 6.0, epsilon = 1e-9);
            }
            other => panic!("expected Pink, got {other:?}"),
        }
    }

    #[test]
    fn csv_mode_requires_trajectory_csv() {
        let argv = [
            "motion_simulator",
            "--mode",
            "csv",
            "--duration",
            "1s",
            "--catalog",
            "fake.bin",
            "--output-dir",
            "/tmp/x",
        ];
        let err = Args::try_parse_from(argv).expect_err("missing --trajectory-csv must fail");
        assert!(
            err.to_string().contains("trajectory-csv"),
            "clap error should mention --trajectory-csv, got: {err}"
        );
    }

    #[test]
    fn csv_mode_does_not_require_start() {
        let argv = [
            "motion_simulator",
            "--mode",
            "csv",
            "--trajectory-csv",
            "/tmp/nope.csv",
            "--duration",
            "1s",
            "--catalog",
            "fake.bin",
            "--output-dir",
            "/tmp/x",
        ];
        let args = Args::try_parse_from(argv).expect("csv + path parses without --start");
        assert_eq!(args.mode, TrajectoryMode::Csv);
        assert!(args.start.is_none());
        assert!(args.trajectory_csv.is_some());
    }

    #[test]
    fn build_csv_trajectory_loads_three_waypoints() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            tmp.path(),
            "time_s,qw,qx,qy,qz\n\
             0.0,1.0,0.0,0.0,0.0\n\
             0.5,0.99999,0.004,0.001,0.0\n\
             1.0,0.99998,0.006,0.002,0.0\n",
        )
        .unwrap();
        let traj = build_csv_trajectory(tmp.path()).unwrap();
        assert_eq!(traj.waypoints().len(), 3);
        assert_relative_eq!(traj.end_time().as_secs_f64(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn build_csv_trajectory_rejects_nonmonotonic_time() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            tmp.path(),
            "time_s,qw,qx,qy,qz\n\
             0.0,1.0,0.0,0.0,0.0\n\
             1.0,1.0,0.0,0.0,0.0\n\
             0.5,1.0,0.0,0.0,0.0\n",
        )
        .unwrap();
        let err = build_csv_trajectory(tmp.path()).unwrap_err();
        assert!(
            err.to_string().contains("strictly increasing"),
            "expected monotonicity error, got: {err}"
        );
    }
}

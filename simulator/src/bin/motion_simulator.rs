use clap::{Parser, ValueEnum};
use log::{info, warn};
use simulator::hardware::satellite::FocalPlaneConfig;
use simulator::hardware::sensor_array::{SensorArray, SPENCER_ARRAY_PLAN};
use simulator::shared_args::{DurationArg, SensorModel, SharedSimulationArgs};
use simulator::sims::trajectory::{
    fov_envelope, render_trajectory, Trajectory, TrajectoryRenderConfig,
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

    let trajectory = if args.start_roll == 0.0 && args.end_roll == 0.0 {
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

    Ok(())
}

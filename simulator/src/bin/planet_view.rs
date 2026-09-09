//! Render solar-system bodies as seen from a spacecraft at one epoch.
//!
//! Quick-look tool for the solar-system second pass: points the telescope
//! at a target body, composites the requested bodies onto an empty star
//! field, and writes a 16-bit PNG plus an 8-bit stretched preview.
//!
//! ```text
//! cargo run --features solar-system --bin planet_view -- \
//!     --epoch 2007-10-03T05:30:00Z --observer mars --bodies earth,moon \
//!     --telescope cosmic-frontier-jbt50cm --exposure-s 0.001 --out earth_from_mars
//! ```
//!
//! Surface models are quick-look defaults: Lambert spheres whose albedo
//! reproduces each planet's V geometric albedo (Mallama & Hilton 2018,
//! `A = 3p/2`), and the lunar-average Hapke set for the Moon.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use image::{ImageBuffer, Luma};
use ndarray::Array2;
use shared::units::{Temperature, TemperatureExt};
use starfield::catalogs::StarData;

use simulator::bodies::brdf::{Brdf, Hapke, Lambert};
use simulator::body_pass::{BodyPass, SceneBody};
use simulator::epoch::Epoch;
use simulator::hardware::satellite::{FocalPlaneConfig, FocalPlaneProjector, SatelliteConfig};
use simulator::photometry::zodiacal::SolarAngularCoordinates;
use simulator::scene::Scene;
use simulator::shared_args::{SensorModel, TelescopeModel};
use simulator::sims::orientation::orientation_from_pointing;
use simulator::solar_system::{BodyId, BodyState, Observer, SolarSystem};

/// Smallest solar elongation with zodiacal-light table coverage.
const MIN_ZODIACAL_ELONGATION_DEG: f64 = 15.0;

#[derive(Parser, Debug)]
#[command(about = "Render solar-system bodies from a spacecraft at one epoch")]
struct Args {
    /// Exposure start, RFC 3339 UTC or `JD 2461558.5` (TDB).
    #[arg(long)]
    epoch: String,

    /// Body whose centre the observer sits at.
    #[arg(long, default_value = "mars")]
    observer: String,

    /// Comma-separated bodies to render; the first is the pointing target.
    #[arg(long, default_value = "earth,moon")]
    bodies: String,

    #[arg(long, value_enum, default_value_t = TelescopeModel::CosmicFrontierJbt50cm)]
    telescope: TelescopeModel,

    #[arg(long, value_enum, default_value_t = SensorModel::Imx455)]
    sensor: SensorModel,

    /// Sensor window rendered around the target, pixels (square).
    #[arg(long, default_value_t = 512)]
    window_px: usize,

    /// Exposure time in seconds.
    #[arg(long, default_value_t = 0.001)]
    exposure_s: f64,

    /// Sensor temperature, °C.
    #[arg(long, default_value_t = -10.0)]
    temperature_c: f64,

    /// Sub-samples per pixel edge when rasterising disks.
    #[arg(long, default_value_t = 4)]
    oversampling: usize,

    /// RNG seed for the noise draws.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Output path stem; writes `<stem>.png` (16-bit) and `<stem>_preview.png`.
    #[arg(long, default_value = "planet_view")]
    out: PathBuf,
}

/// Quick-look reflectance law for a body.
fn default_brdf(body: BodyId) -> Arc<dyn Brdf> {
    let lambert = |geometric_albedo: f64| -> Arc<dyn Brdf> {
        Arc::new(Lambert {
            albedo: (1.5 * geometric_albedo).min(1.0),
        })
    };
    match body {
        BodyId::Moon => Arc::new(Hapke::lunar_average()),
        BodyId::Mercury => lambert(0.142),
        BodyId::Venus => lambert(0.689),
        BodyId::Earth => lambert(0.434),
        BodyId::Mars => lambert(0.170),
        BodyId::Jupiter => lambert(0.538),
        BodyId::Saturn => lambert(0.499),
        BodyId::Uranus => lambert(0.488),
        BodyId::Neptune => lambert(0.442),
        BodyId::Sun => lambert(1.0),
    }
}

fn parse_body(name: &str) -> Result<BodyId, String> {
    BodyId::parse(name).ok_or_else(|| format!("unknown body {name:?}"))
}

fn describe(state: &BodyState, elongation_deg: f64) -> String {
    let ill = &state.illumination;
    format!(
        "{:<8} RA {:9.4}° Dec {:+8.4}°  range {:.4} AU  diam {:7.3}″  phase {:6.2}°  lit {:5.1}%  \
         limb PA {:6.1}°  elong {:5.1}°  V {}",
        state.body,
        state.direction.ra_degrees(),
        state.direction.dec_degrees(),
        state.distance_au,
        state.angular_diameter_arcsec(),
        ill.phase_angle.to_degrees(),
        100.0 * ill.illuminated_fraction,
        ill.bright_limb_position_angle.to_degrees(),
        elongation_deg,
        state
            .v_magnitude
            .map_or_else(|| "  n/a".to_string(), |v| format!("{v:+5.2}")),
    ) + &format!(
        "\n{:<8} sub-observer lon {:7.2}°E lat {:+6.2}°  sub-solar {}  pole PA {:6.1}°",
        "",
        state.sub_observer.lon_rad.to_degrees(),
        state.sub_observer.lat_rad.to_degrees(),
        state.sub_solar.map_or_else(
            || "n/a".to_string(),
            |p| format!(
                "lon {:7.2}°E lat {:+6.2}°",
                p.lon_rad.to_degrees(),
                p.lat_rad.to_degrees()
            )
        ),
        state.north_pole_position_angle.to_degrees(),
    )
}

fn save_u16(image: &Array2<u16>, path: &PathBuf) -> Result<(), String> {
    let (h, w) = image.dim();
    let raw: Vec<u16> = image.iter().copied().collect();
    ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(w as u32, h as u32, raw)
        .ok_or_else(|| "buffer size mismatch".to_string())?
        .save(path)
        .map_err(|e| e.to_string())
}

/// Asinh stretch between the 1st percentile and the maximum, to 8 bits.
fn save_preview(image: &Array2<f64>, path: &PathBuf) -> Result<(), String> {
    let (h, w) = image.dim();
    let mut sorted: Vec<f64> = image.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.is_empty() {
        return Err("empty image".into());
    }
    let floor = sorted[sorted.len() / 100];
    let ceiling = *sorted.last().unwrap();
    let span = (ceiling - floor).max(1e-9);
    let scale = 1000.0;
    let norm = |v: f64| ((v - floor).max(0.0) / span * scale).asinh() / scale.asinh();
    let raw: Vec<u8> = image
        .iter()
        .map(|&v| (norm(v) * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(w as u32, h as u32, raw)
        .ok_or_else(|| "buffer size mismatch".to_string())?
        .save(path)
        .map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args = Args::parse();

    let epoch = Epoch::parse(&args.epoch).map_err(|e| e.to_string())?;
    let observer = Observer::BodyCenter(parse_body(&args.observer)?);
    let bodies: Vec<BodyId> = args
        .bodies
        .split(',')
        .map(parse_body)
        .collect::<Result<_, _>>()?;
    let target = *bodies.first().ok_or("at least one body is required")?;

    let system = Arc::new(SolarSystem::new().map_err(|e| e.to_string())?);
    let satellite = SatelliteConfig::new(
        args.telescope.to_config().clone(),
        args.sensor
            .to_config()
            .with_dimensions(args.window_px, args.window_px),
        Temperature::from_celsius(args.temperature_c),
    );
    let focal_plane = FocalPlaneConfig::from_satellite(&satellite);

    println!(
        "{} on {} ({:.3}″/px, window {} px), epoch {}",
        satellite.telescope.name,
        satellite.sensor.name,
        satellite.plate_scale_arcsec_per_pixel(),
        args.window_px,
        epoch
    );

    let mut states = Vec::new();
    for &body in &bodies {
        let state = system
            .body_state(body, &observer, &epoch)
            .map_err(|e| e.to_string())?;
        let elongation = system
            .solar_elongation(&state.direction, &observer, &epoch)
            .map_err(|e| e.to_string())?
            .to_degrees();
        println!("{}", describe(&state, elongation));
        states.push((state, elongation));
    }
    let (target_state, target_elongation) = &states[0];
    let pointing = target_state.direction;

    let scene_bodies: Vec<SceneBody> = bodies
        .iter()
        .map(|&id| SceneBody {
            id,
            brdf: default_brdf(id),
        })
        .collect();
    let pass = BodyPass::new(Arc::clone(&system), observer, scene_bodies)
        .with_oversampling(args.oversampling);

    // The zodiacal table has no data inside the solar exclusion zone;
    // inside it the true background is stray light, which is not
    // modelled, so the nearest tabulated elongation is used.
    let zodiacal_elongation = target_elongation.clamp(MIN_ZODIACAL_ELONGATION_DEG, 180.0);
    let zodiacal =
        SolarAngularCoordinates::new(zodiacal_elongation, 0.0).map_err(|e| e.to_string())?;
    let scene = Scene::from_catalog(focal_plane, Vec::new(), pointing, zodiacal)
        .with_second_pass(Arc::new(pass), Some(epoch));

    let exposure = std::time::Duration::from_secs_f64(args.exposure_s);
    let result = scene.render_with_seed(&exposure, Some(args.seed)).remove(0);

    let well = satellite.sensor.max_well_depth_e;
    let saturated = result.star_image.iter().filter(|&&e| e >= well).count();
    let total = result.star_image.sum();
    println!(
        "body electrons {:.3e} (peak {:.3e} e⁻/px, {} px at full well {:.0} e⁻)",
        total,
        result.star_image.iter().cloned().fold(0.0, f64::max),
        saturated,
        well
    );
    println!(
        "{} is {} px across on this sensor",
        target,
        (target_state.angular_diameter_arcsec() / satellite.plate_scale_arcsec_per_pixel()).round()
    );
    // Geometric centre of each body on the sensor, through the same
    // projector the second pass uses (pixel index = pixel centre).
    let orientation = orientation_from_pointing(&pointing, 0.0);
    for (state, _) in &states {
        let probe = StarData::with_position(0, state.direction, 0.0, None);
        match scene
            .focal_plane
            .project_to_sensor(&probe, &orientation, 0, 0.0)
        {
            Some((x, y)) => println!("{} centre px {x:.4} {y:.4}", state.body),
            None => println!("{} centre px off-sensor", state.body),
        }
    }

    let png = args.out.with_extension("png");
    let preview = PathBuf::from(format!("{}_preview.png", args.out.display()));
    save_u16(&result.quantized_image, &png)?;
    let electrons = result.mean_electron_image();
    save_preview(&electrons, &preview)?;
    println!("wrote {} and {}", png.display(), preview.display());
    Ok(())
}

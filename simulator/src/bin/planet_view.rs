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
use std::sync::{Arc, LazyLock};

use ab_glyph::{FontRef, PxScale};
use clap::Parser;
use image::{ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_circle_mut, draw_text_mut};
use ndarray::Array2;
use serde_json::json;
use shared::units::{AreaExt, LengthExt, Temperature, TemperatureExt};
use starfield::catalogs::{StarCatalog, StarData};
use starfield_gaia::{Dr3, LazyLoadingCatalog};

use starfield_planet_maps::{earth_tier, mars_tier, AbundanceTier};
use starfield_reflectance_library::ReflectanceLibrary;

use simulator::atmosphere::RayleighAtmosphere;
use simulator::bodies::brdf::{Hapke, Lambert};
use simulator::bodies::surface::{SurfaceModel, TexturedSurfaceModel};
use simulator::body_pass::{BodyPass, SceneBody};
use simulator::epoch::Epoch;
use simulator::hardware::satellite::{FocalPlaneConfig, FocalPlaneProjector, SatelliteConfig};
use simulator::image_proc::render::quantize_image;
use simulator::photometry::zodiacal::SolarAngularCoordinates;
use simulator::scene::Scene;
use simulator::shared_args::{SensorModel, TelescopeModel};
use simulator::sims::orientation::orientation_from_pointing;
use simulator::solar_system::{
    BodyId, BodyState, Observer, SiteStatus, SiteView, SolarSystem, SurfaceSite,
};

/// Smallest solar elongation with zodiacal-light table coverage.
const MIN_ZODIACAL_ELONGATION_DEG: f64 = 15.0;

static FONT_DATA: &[u8] = include_bytes!("../../assets/fonts/DejaVuSansMono-Bold.ttf");
static FONT: LazyLock<FontRef<'static>> =
    LazyLock::new(|| FontRef::try_from_slice(FONT_DATA).expect("bundled font parses"));

/// Overlay colour for surface sites, distinct from the red/green centre
/// markers used in downstream videos.
const SITE_COLOUR: Rgb<u8> = Rgb([0, 220, 255]);

/// A surface site projected onto the sensor, ready to draw.
struct SiteMarker {
    site: SurfaceSite,
    view: SiteView,
    pixel: Option<(f64, f64)>,
}

/// Parse `Name:lat_deg:lon_east_deg:height_m`.
fn parse_site(spec: &str) -> Result<SurfaceSite, String> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 4 {
        return Err(format!(
            "site {spec:?} must be Name:lat_deg:lon_east_deg:height_m"
        ));
    }
    let num = |s: &str, what: &str| {
        s.trim()
            .parse::<f64>()
            .map_err(|_| format!("site {spec:?}: bad {what} {s:?}"))
    };
    Ok(SurfaceSite {
        name: parts[0].trim().to_string(),
        latitude_deg: num(parts[1], "latitude")?,
        longitude_east_deg: num(parts[2], "longitude")?,
        height_m: num(parts[3], "height")?,
    })
}

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

    /// Render grey Lambert/Hapke spheres instead of composition tiers.
    #[arg(long, default_value_t = false)]
    untextured: bool,

    /// Leave out Earth's Rayleigh atmosphere (limb glow, twilight,
    /// two-way extinction of the surface).
    #[arg(long, default_value_t = false)]
    no_atmosphere: bool,

    /// Surface site on the target body to mark, as
    /// `Name:lat_deg:lon_east_deg:height_m` (geodetic). Repeatable.
    #[arg(long = "site")]
    sites: Vec<String>,

    /// Qualifier appended to site labels.
    #[arg(long, default_value = "likely receiving site")]
    site_qualifier: String,

    /// Render background stars from the Gaia DR3 excerpt around the
    /// pointing (occulted by bodies in front of them).
    #[arg(long, default_value_t = false)]
    stars: bool,

    /// Faintest Gaia G magnitude to load when `--stars` is set.
    #[arg(long, default_value_t = 18.0)]
    star_mag_limit: f64,

    /// HEALPix-sharded Gaia DR3 excerpt directory; defaults to the
    /// starfield cache.
    #[arg(long)]
    gaia_dir: Option<PathBuf>,

    /// Write the pre-noise mean image (no shot, read or dark noise) as
    /// the 16-bit output instead of a noisy realisation.
    #[arg(long, default_value_t = false)]
    noiseless: bool,

    /// Skip the PSF blur on body stamps (stars keep theirs), isolating
    /// the PSF's effect on the limb.
    #[arg(long, default_value_t = false)]
    no_psf: bool,

    /// Fixed preview stretch ceiling in electrons per pixel, so a series
    /// of frames shares one brightness scale. Default: each frame's own
    /// maximum.
    #[arg(long)]
    stretch_peak_e: Option<f64>,

    /// Output path stem; writes `<stem>.png` (16-bit) and `<stem>_preview.png`.
    #[arg(long, default_value = "planet_view")]
    out: PathBuf,
}

/// Quick-look grey reflectance law for a body: a Lambert sphere whose
/// albedo reproduces the V geometric albedo, or the lunar Hapke set.
fn grey_surface(body: BodyId) -> Arc<dyn SurfaceModel> {
    let lambert = |geometric_albedo: f64| -> Arc<dyn SurfaceModel> {
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

/// Textured surface where a composition tier exists (Earth, Mars),
/// grey otherwise.
fn default_surface(
    body: BodyId,
    library: &Arc<ReflectanceLibrary>,
    untextured: bool,
) -> Result<Arc<dyn SurfaceModel>, String> {
    if untextured {
        return Ok(grey_surface(body));
    }
    let tier: Option<(AbundanceTier, &str)> = match body {
        BodyId::Earth => Some((
            earth_tier().map_err(|e| e.to_string())?,
            "Earth MCD12C1 composition 0.25°",
        )),
        BodyId::Mars => Some((
            mars_tier().map_err(|e| e.to_string())?,
            "Mars Viking/MDIM albedo 0.1° (uncalibrated contrast)",
        )),
        _ => None,
    };
    Ok(match tier {
        Some((tier, label)) => Arc::new(TexturedSurfaceModel::new(
            Arc::new(tier),
            Arc::new(Lambert { albedo: 1.0 }),
            Arc::clone(library),
            label,
        )),
        None => grey_surface(body),
    })
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

/// Asinh stretch between the 1st percentile and `ceiling` (default the
/// frame maximum), to 8 bits: `v ↦ asinh(1000 · (v − floor)/(ceiling −
/// floor)) / asinh(1000)`.
fn stretch_to_u8(image: &Array2<f64>, ceiling: Option<f64>) -> Result<Vec<u8>, String> {
    let mut sorted: Vec<f64> = image.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.is_empty() {
        return Err("empty image".into());
    }
    let floor = sorted[sorted.len() / 100];
    let ceiling = ceiling.unwrap_or(*sorted.last().unwrap());
    let span = (ceiling - floor).max(1e-9);
    let scale = 1000.0;
    let norm = |v: f64| ((v - floor).max(0.0) / span * scale).asinh() / scale.asinh();
    Ok(image
        .iter()
        .map(|&v| (norm(v) * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect())
}

/// Stretched 8-bit preview; greyscale without sites, RGB with cyan site
/// markers and labels when sites are given.
fn save_preview(
    image: &Array2<f64>,
    path: &PathBuf,
    markers: &[SiteMarker],
    qualifier: &str,
    stretch_ceiling: Option<f64>,
) -> Result<(), String> {
    let (h, w) = image.dim();
    let raw = stretch_to_u8(image, stretch_ceiling)?;
    if markers.is_empty() {
        return ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(w as u32, h as u32, raw)
            .ok_or_else(|| "buffer size mismatch".to_string())?
            .save(path)
            .map_err(|e| e.to_string());
    }
    let mut rgb = RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let v = raw[(y as usize) * w + x as usize];
        Rgb([v, v, v])
    });
    let font_px = ((h as f32) / 32.0).clamp(10.0, 18.0);
    let font_scale = PxScale::from(font_px);
    // Full label with status in the caption band; only the name at the
    // marker, so a long qualifier never runs over the disk.
    let mut caption_y = 4;
    for marker in markers {
        let status = marker.view.status();
        let caption = match (status, marker.pixel) {
            (SiteStatus::FarSide, _) => {
                format!("{} ({qualifier}): not visible, far side", marker.site.name)
            }
            (_, None) => format!("{} ({qualifier}): {status}, off sensor", marker.site.name),
            (_, Some((px, py))) => {
                let (cx, cy) = (px.round() as i32, py.round() as i32);
                draw_filled_circle_mut(&mut rgb, (cx, cy), 3, SITE_COLOUR);
                draw_hollow_circle_mut(&mut rgb, (cx, cy), 7, SITE_COLOUR);
                let name = &marker.site.name;
                let text_w = (name.chars().count() as f32 * font_px * 0.62) as i32;
                let tx = if cx + 12 + text_w < w as i32 {
                    cx + 12
                } else {
                    (cx - 12 - text_w).max(0)
                };
                let ty = (cy - font_px as i32 / 2).clamp(0, (h as i32 - font_px as i32).max(0));
                draw_text_mut(&mut rgb, SITE_COLOUR, tx, ty, font_scale, &*FONT, name);
                format!("{name} ({qualifier}): {status}")
            }
        };
        draw_text_mut(
            &mut rgb,
            SITE_COLOUR,
            4,
            caption_y,
            font_scale,
            &*FONT,
            &caption,
        );
        caption_y += font_px as i32 + 2;
    }
    rgb.save(path).map_err(|e| e.to_string())
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

    let library = Arc::new(ReflectanceLibrary::load_embedded().map_err(|e| e.to_string())?);
    let mut scene_bodies = Vec::with_capacity(bodies.len());
    for &id in &bodies {
        let surface = default_surface(id, &library, args.untextured)?;
        println!("{id} surface: {}", surface.label());
        let mut body = SceneBody::new(id, surface);
        if id == BodyId::Earth && !args.no_atmosphere {
            let air = RayleighAtmosphere::earth(id.equatorial_radius_km());
            println!(
                "{id} atmosphere: Rayleigh single scattering, H {:.1} km, top {:.0} km, \
                 τ(550 nm) {:.4}",
                air.scale_height_km,
                air.top_km,
                air.vertical_optical_depth(550.0, 0.0)
            );
            body = body.with_atmosphere(air);
        }
        scene_bodies.push(body);
    }
    let pass = BodyPass::new(Arc::clone(&system), observer.clone(), scene_bodies)
        .with_oversampling(args.oversampling)
        .with_psf_blur(!args.no_psf);
    let solar_rate = pass
        .solar_electron_rate(&satellite)
        .map_err(|e| e.to_string())?;
    println!(
        "solar photo-electron rate through the aperture at 1 AU: {solar_rate:.4e} e⁻/s \
         (TSIS-1 HSRS × combined QE)"
    );

    // The zodiacal table has no data inside the solar exclusion zone;
    // inside it the true background is stray light, which is not
    // modelled, so the nearest tabulated elongation is used.
    let zodiacal_elongation = target_elongation.clamp(MIN_ZODIACAL_ELONGATION_DEG, 180.0);
    let zodiacal =
        SolarAngularCoordinates::new(zodiacal_elongation, 0.0).map_err(|e| e.to_string())?;

    // Background stars: a Gaia cone covering the window's half-diagonal
    // plus a PSF margin, so every star that can deposit light is loaded.
    let stars: Vec<StarData> = if args.stars {
        let dir = args
            .gaia_dir
            .clone()
            .unwrap_or_else(simulator::sims::gaia_dr3::default_excerpt_dir);
        let lazy = LazyLoadingCatalog::<Dr3>::open(&dir).map_err(|e| e.to_string())?;
        let half_diag_deg = (args.window_px as f64) * satellite.plate_scale_arcsec_per_pixel()
            / 3600.0
            * std::f64::consts::FRAC_1_SQRT_2
            + 0.005;
        let (catalog, _) = simulator::sims::gaia_dr3::materialize_cone_augmented(
            &lazy,
            pointing,
            half_diag_deg,
            args.star_mag_limit,
        )
        .map_err(|e| e.to_string())?;
        let stars: Vec<StarData> = catalog.star_data().collect();
        println!(
            "stars: {} Gaia DR3 sources to G {:.1} within {:.3}° of the pointing",
            stars.len(),
            args.star_mag_limit,
            half_diag_deg
        );
        stars
    } else {
        Vec::new()
    };
    let star_count = stars.len();

    let scene = Scene::from_catalog(focal_plane, stars, pointing, zodiacal)
        .with_second_pass(Arc::new(pass), Some(epoch.clone()));

    let exposure = std::time::Duration::from_secs_f64(args.exposure_s);
    let mut result = scene
        .render_with_options(&exposure, !args.noiseless, Some(args.seed))
        .remove(0);
    if args.noiseless {
        // Pre-noise mean of sky + bodies + zodiacal light; no read or
        // dark noise either.
        let mean = &result.star_image + &result.zodiacal_image;
        result.quantized_image = quantize_image(&mean, &satellite.sensor);
        result.sensor_noise_image.fill(0.0);
    }

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

    // Surface sites on the target: same light-time-corrected orientation
    // and projector as the disk, so marker and texture agree.
    let mut markers = Vec::new();
    for spec in &args.sites {
        let site = parse_site(spec)?;
        let view = target_state.view_site(&site);
        let probe = StarData::with_position(0, view.direction, 0.0, None);
        let pixel = scene
            .focal_plane
            .project_to_sensor(&probe, &orientation, 0, 0.0);
        let px_text = pixel.map_or_else(
            || "off-sensor".to_string(),
            |(x, y)| format!("{x:.4} {y:.4}"),
        );
        println!(
            "site {} lat {:+.6} lon {:+.6}E h {:.0} m: {}; px {px_text}; offset E {:+.4}\" N {:+.4}\"; \
             emission_cos {:+.4}; incidence_cos {}",
            site.name,
            site.latitude_deg,
            site.longitude_east_deg,
            site.height_m,
            view.status(),
            view.sky_offset_rad.0.to_degrees() * 3600.0,
            view.sky_offset_rad.1.to_degrees() * 3600.0,
            view.emission_cosine,
            view.incidence_cosine
                .map_or_else(|| "n/a".to_string(), |c| format!("{c:+.4}")),
        );
        markers.push(SiteMarker { site, view, pixel });
    }

    let png = args.out.with_extension("png");
    let preview = PathBuf::from(format!("{}_preview.png", args.out.display()));
    save_u16(&result.quantized_image, &png)?;
    let electrons = result.mean_electron_image();
    save_preview(
        &electrons,
        &preview,
        &markers,
        &args.site_qualifier,
        args.stretch_peak_e,
    )?;

    // Per-frame metadata: everything needed to reproduce and to read
    // the image quantitatively.
    let metadata_path = PathBuf::from(format!("{}.json", args.out.display()));
    let git_commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let body_json = |state: &BodyState, elongation_deg: f64| {
        let ill = &state.illumination;
        json!({
            "body": state.body.name(),
            "naif_id": state.body.naif_id(),
            "apparent_ra_deg": state.direction.ra_degrees(),
            "apparent_dec_deg": state.direction.dec_degrees(),
            "range_au": state.distance_au,
            "range_km": state.distance_km(),
            "light_time_s": state.light_time_s,
            "heliocentric_distance_au": ill.heliocentric_distance_au,
            "angular_diameter_arcsec": state.angular_diameter_arcsec(),
            "phase_angle_deg": ill.phase_angle.to_degrees(),
            "illuminated_fraction": ill.illuminated_fraction,
            "bright_limb_position_angle_deg": ill.bright_limb_position_angle.to_degrees(),
            "north_pole_position_angle_deg": state.north_pole_position_angle.to_degrees(),
            "solar_elongation_deg": elongation_deg,
            "v_magnitude_mallama_hilton_2018": state.v_magnitude,
            "sub_observer_lon_east_deg": state.sub_observer.lon_rad.to_degrees(),
            "sub_observer_lat_planetocentric_deg": state.sub_observer.lat_rad.to_degrees(),
            "sub_solar_lon_east_deg": state.sub_solar.map(|p| p.lon_rad.to_degrees()),
            "sub_solar_lat_planetocentric_deg": state.sub_solar.map(|p| p.lat_rad.to_degrees()),
            "equatorial_radius_km": state.body.equatorial_radius_km(),
            "polar_radius_km": state.body.polar_radius_km(),
            "electrons_per_sr_per_unit_reflectance": args.exposure_s * solar_rate
                / ill.heliocentric_distance_au.powi(2),
        })
    };
    let sites_json: Vec<_> = markers
        .iter()
        .map(|m| {
            json!({
                "name": m.site.name,
                "qualifier": args.site_qualifier,
                "latitude_geodetic_deg": m.site.latitude_deg,
                "longitude_east_deg": m.site.longitude_east_deg,
                "height_m": m.site.height_m,
                "status": m.view.status().to_string(),
                "visible_geometric": m.view.visible,
                "emission_cosine": m.view.emission_cosine,
                "incidence_cosine": m.view.incidence_cosine,
                "sky_offset_east_arcsec": m.view.sky_offset_rad.0.to_degrees() * 3600.0,
                "sky_offset_north_arcsec": m.view.sky_offset_rad.1.to_degrees() * 3600.0,
                "pixel_x": m.pixel.map(|p| p.0),
                "pixel_y": m.pixel.map(|p| p.1),
                "meaning": "apparent imaged site at the light-time-corrected epoch; not a transmit point-ahead or arrival-time aimpoint; visible means on the facing hemisphere, not link availability",
            })
        })
        .collect();
    let metadata = json!({
        "generator": {
            "crate": "focalplane simulator",
            "version": env!("CARGO_PKG_VERSION"),
            "binary": "planet_view",
            "git_commit": git_commit,
            "command_line": std::env::args().collect::<Vec<_>>(),
        },
        "epoch": {
            "utc": epoch.to_string(),
            "jd_tdb": epoch.jd_tdb(),
        },
        "observer": format!("{observer:?}"),
        "pointing_icrf": {
            "ra_deg": pointing.ra_degrees(),
            "dec_deg": pointing.dec_degrees(),
            "roll_deg": 0.0,
            "target": target.name(),
        },
        "instrument": {
            "telescope": satellite.telescope.name,
            "aperture_m": satellite.telescope.aperture.as_meters(),
            "focal_length_m": satellite.telescope.focal_length.as_meters(),
            "obscuration_ratio_linear": satellite.telescope.obscuration_ratio,
            "clear_aperture_cm2": satellite.telescope.clear_aperture_area().as_square_centimeters(),
            "sensor": satellite.sensor.name,
            "pixel_um": satellite.sensor.pixel_size().as_micrometers(),
            "plate_scale_arcsec_per_px": satellite.plate_scale_arcsec_per_pixel(),
            "window_px": args.window_px,
            "exposure_s": args.exposure_s,
            "temperature_c": args.temperature_c,
            "full_well_e": well,
            "dark_current_e_per_s_per_px": satellite.sensor.dark_current_at_temperature(satellite.temperature),
            "psf_model": if args.no_psf { "none on bodies (delta); stars Gaussian-approximated Airy" } else { "Gaussian approximation of the Airy core at the reference wavelength, 3x3 Simpson per pixel" },
            "psf_fwhm_px": satellite.airy_disk_pixel_space().fwhm(),
            "image_parity": "north up, east left (sky parity); pixel index = pixel centre",
        },
        "models": {
            "bodies": bodies.iter().map(|b| b.name()).collect::<Vec<_>>(),
            "textured": !args.untextured,
            "earth_atmosphere": if args.no_atmosphere || target != BodyId::Earth && !bodies.contains(&BodyId::Earth) { json!(null) } else {
                let air = RayleighAtmosphere::earth(BodyId::Earth.equatorial_radius_km());
                json!({"type": "Rayleigh single scattering", "scale_height_km": air.scale_height_km, "top_km": air.top_km, "tau_vertical_550nm": air.vertical_optical_depth(550.0, 0.0), "spectral_bins": 8, "multiple_scattering": false, "aerosols": false, "ozone": false, "clouds": false, "refraction": false})
            },
            "noise": if args.noiseless { "none (pre-noise mean image)" } else { "Poisson shot noise on bodies+stars and on zodiacal light, Gaussian read noise, dark current" },
            "stray_light": "not modelled",
            "ocean_glint": "not modelled",
            "clouds": "not modelled",
            "body_motion_blur": "not modelled (body evaluated at mid-exposure orientation)",
            "oversampling_per_pixel_edge": args.oversampling,
            "stars": if args.stars { json!({"catalog": "Gaia DR3 excerpt + Hipparcos bright supplement", "mag_limit_g": args.star_mag_limit, "count_in_cone": star_count}) } else { json!(null) },
        },
        "radiometry": {
            "solar_spectrum": simulator::photometry::solar::HSRS_PROVENANCE,
            "solar_electron_rate_1au_e_per_s": solar_rate,
            "chain": "F_sun(lambda) [W m^-2 nm^-1 @1 AU] -> photons (lambda/hc) -> electrons (combined QE, 1 nm bins) -> Edot_sun; body radiance per unit solar irradiance r [sr^-1] = BRDF x texel band albedo x T_sun T_view + Rayleigh path radiance; pixel e- = sum_subsamples r * (T * Edot_sun / d_sun^2) * Omega_sub",
            "scene_electrons_total": total,
            "scene_electrons_peak_per_px": result.star_image.iter().cloned().fold(0.0, f64::max),
            "pixels_at_full_well": saturated,
            "zodiacal_elongation_used_deg": zodiacal_elongation,
        },
        "bodies": states.iter().map(|(s, e)| body_json(s, *e)).collect::<Vec<_>>(),
        "sites": sites_json,
        "outputs": {
            "image_16bit_dn": png.display().to_string(),
            "preview_8bit": preview.display().to_string(),
            "preview_stretch": format!("asinh(1000 x) / asinh(1000), floor = 1st percentile, ceiling = {}", args.stretch_peak_e.map_or("frame max".to_string(), |c| format!("{c} e-/px"))),
            "dn_per_electron": satellite.sensor.dn_per_electron,
            "black_level_dn": satellite.sensor.black_level_dn,
        },
    });
    std::fs::write(
        &metadata_path,
        serde_json::to_string_pretty(&metadata).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "wrote {}, {} and {}",
        png.display(),
        preview.display(),
        metadata_path.display()
    );
    Ok(())
}

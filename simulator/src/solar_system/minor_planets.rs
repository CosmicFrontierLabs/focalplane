//! Minor planets as unresolved point sources.
//!
//! Every numbered and provisionally designated asteroid in the Minor
//! Planet Center's MPCORB catalogue can be placed on the sky for any
//! observer the ephemeris knows about, with a V magnitude from the IAU
//! H-G system. The result is a [`StarData`] per object, so asteroids go
//! through the same first-pass point-source photometry and PSF as the
//! stars, occulted by the resolved bodies of the second pass like any
//! other background source.
//!
//! # Method
//!
//! - Orbits: two-body propagation of the MPCORB osculating elements
//!   (`starfield-mpc` → `starfield::keplerlib`). No planetary
//!   perturbations, so positions drift from the true ones by a few
//!   arcseconds per year on the main belt and faster for near-Earth
//!   objects at close approach; the MPC refreshes the elements daily and
//!   the catalogue epoch is recorded in each sighting.
//! - Geometry: barycentric position = Sun's barycentric position plus
//!   the heliocentric two-body position, evaluated at the emission epoch
//!   found by two light-time iterations; first-order observer aberration
//!   `d′ ∝ d + v_obs / c`, the same correction starfield applies to
//!   stars and planets. Gravitational deflection is not applied (< 1 mas
//!   away from the Sun's limb).
//! - Photometry: `V = H + 5 log10(r Δ) − 2.5 log10[(1−G) Φ₁ + G Φ₂]`
//!   (Bowell et al. 1989), G = 0.15 where the catalogue has none.
//!   Reflected sunlight is treated as a solar-colour point source
//!   (`B−V` = [`MINOR_PLANET_B_V`]); S-type bodies are slightly redder,
//!   C-type slightly bluer, at the few-percent level in a broadband CMOS
//!   band.
//! - Not modelled: rotational light curves, resolved disks (Ceres reaches
//!   0.8″, six JBT pixels, at best), comets' comae, satellites of
//!   asteroids.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use nalgebra::Vector3;
use rayon::prelude::*;
use starfield::catalogs::StarData;
use starfield::constants::C_AUDAY;
use starfield::jplephem_ext::SpiceKernelExt;
use starfield::keplerlib::KeplerOrbit;
use starfield::time::Timescale;
use starfield::Equatorial;
use starfield_mpc::{parse_mpcorb_line, MpcOrbRecord};
use thiserror::Error;

use super::{Observer, SolarSystem, SolarSystemError, SECONDS_PER_DAY};
use crate::epoch::Epoch;

/// Colour assigned to every minor planet: the Sun's B−V plus the mean
/// reddening of the main belt.
pub const MINOR_PLANET_B_V: f64 = 0.75;

/// Where the MPC publishes the full catalogue.
pub const MPCORB_URL: &str = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT";

/// Errors from catalogue loading.
#[derive(Debug, Error)]
pub enum MinorPlanetError {
    /// Reading the catalogue file.
    #[error("reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Fetching the catalogue from the MPC.
    #[error("downloading MPCORB.DAT: {0}")]
    Download(String),
    /// A catalogue file with no parseable records.
    #[error("{path} contains no MPCORB records")]
    Empty { path: PathBuf },
    /// Ephemeris evaluation for the observer or the Sun.
    #[error(transparent)]
    Ephemeris(#[from] SolarSystemError),
}

/// One MPCORB row reduced to what placing and weighing the body needs.
#[derive(Clone, Debug)]
pub struct MinorPlanetElements {
    /// Packed MPC designation (`00001`, `K14A00A`, …).
    pub designation: String,
    /// Human-readable designation (`(1) Ceres`, `2014 AA`).
    pub name: String,
    /// Absolute magnitude H, if the catalogue has one.
    pub h: Option<f64>,
    /// Slope parameter G, if the catalogue has one.
    pub g: Option<f64>,
    /// The osculating orbit, ecliptic elements already rotated to the ICRF.
    orbit: KeplerOrbit,
    /// Osculating epoch, TT Julian date.
    pub epoch_tt: f64,
    /// The full record for anything else a caller wants (arc, RMS, …).
    pub record: MpcOrbRecord,
}

/// An MPCORB catalogue in memory.
#[derive(Debug)]
pub struct MinorPlanetCatalog {
    elements: Vec<MinorPlanetElements>,
    path: PathBuf,
    skipped_unusable: usize,
}

impl MinorPlanetCatalog {
    /// `~/.cache/starfield/mpcorb/MPCORB.DAT`.
    pub fn default_path() -> PathBuf {
        starfield_datasource_utils::cache_dir()
            .join("mpcorb")
            .join("MPCORB.DAT")
    }

    /// Load the cached catalogue, downloading it from the MPC first when
    /// the cache is empty (about 320 MB, 1.5 million rows).
    pub fn load_default() -> Result<Self, MinorPlanetError> {
        let path = Self::default_path();
        if !starfield_datasource_utils::file_exists_and_not_empty(&path) {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir).map_err(|source| MinorPlanetError::Io {
                    path: dir.to_path_buf(),
                    source,
                })?;
            }
            starfield_datasource_utils::download_to_file(MPCORB_URL, &path, 900)
                .map_err(|e| MinorPlanetError::Download(e.to_string()))?;
        }
        Self::from_file(&path)
    }

    /// Parse an MPCORB-format file (the full catalogue or any subset).
    pub fn from_file(path: &Path) -> Result<Self, MinorPlanetError> {
        let text = std::fs::read_to_string(path).map_err(|source| MinorPlanetError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let ts = Timescale::default();
        let lines: Vec<&str> = text.lines().collect();
        let parsed: Vec<Option<MinorPlanetElements>> = lines
            .par_iter()
            .filter_map(|line| parse_mpcorb_line(line))
            .map(|record| MinorPlanetElements::from_record(record, &ts))
            .collect();
        let skipped_unusable = parsed.iter().filter(|e| e.is_none()).count();
        let elements: Vec<MinorPlanetElements> = parsed.into_iter().flatten().collect();
        if elements.is_empty() {
            return Err(MinorPlanetError::Empty {
                path: path.to_path_buf(),
            });
        }
        Ok(Self {
            elements,
            path: path.to_path_buf(),
            skipped_unusable,
        })
    }

    /// A catalogue from records already in hand (tests, hand-picked sets).
    pub fn from_records(records: Vec<MpcOrbRecord>) -> Self {
        let ts = Timescale::default();
        let parsed: Vec<Option<MinorPlanetElements>> = records
            .into_iter()
            .map(|r| MinorPlanetElements::from_record(r, &ts))
            .collect();
        let skipped_unusable = parsed.iter().filter(|e| e.is_none()).count();
        Self {
            elements: parsed.into_iter().flatten().collect(),
            path: PathBuf::from("<records>"),
            skipped_unusable,
        }
    }

    /// Number of bodies with usable elements.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// True when no body has usable elements.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Rows dropped for an unparseable epoch or elements that describe
    /// no bound orbit.
    pub fn skipped_unusable(&self) -> usize {
        self.skipped_unusable
    }

    /// File the catalogue came from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// All bodies.
    pub fn elements(&self) -> &[MinorPlanetElements] {
        &self.elements
    }

    /// Look a body up by packed or readable designation (case-insensitive
    /// on the readable form, so `ceres` finds `(1) Ceres`).
    pub fn find(&self, designation: &str) -> Option<&MinorPlanetElements> {
        let lower = designation.trim().to_ascii_lowercase();
        self.elements.iter().find(|e| {
            e.designation.eq_ignore_ascii_case(designation.trim())
                || e.name.to_ascii_lowercase() == lower
                || e.name.to_ascii_lowercase().ends_with(&format!(") {lower}"))
        })
    }

    /// Osculating epochs present, as (earliest, latest) TT Julian dates.
    pub fn epoch_range_tt(&self) -> Option<(f64, f64)> {
        let mut it = self.elements.iter().map(|e| e.epoch_tt);
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), t| (lo.min(t), hi.max(t))))
    }
}

impl MinorPlanetElements {
    fn from_record(record: MpcOrbRecord, ts: &Timescale) -> Option<Self> {
        let orbit = record.to_kepler_orbit(ts)?;
        // Elements that describe no bound orbit (e ≥ 1 with a > 0, a ≤ 0,
        // NaN fields) give a non-finite state; screen them here so every
        // stored orbit propagates.
        if !orbit.is_finite() {
            return None;
        }
        Some(Self {
            designation: record.designation.clone(),
            name: record.readable_designation.clone(),
            h: record.h_magnitude,
            g: record.g_slope,
            epoch_tt: orbit.epoch_tt,
            orbit,
            record,
        })
    }

    /// Heliocentric ICRF position (AU) and velocity (AU/day) at `tt_jd`.
    fn heliocentric_at(&self, epoch: &Epoch) -> (Vector3<f64>, Vector3<f64>) {
        let p = self.orbit.at(epoch.time());
        (p.position, p.velocity)
    }

    /// Stable 64-bit id for [`StarData`], from the packed designation.
    pub fn star_id(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.designation.hash(&mut hasher);
        // Keep clear of Gaia source ids, which are < 2^63.
        hasher.finish() | (1 << 63)
    }
}

/// A minor planet placed on the sky for one observer and epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct MinorPlanetSighting {
    /// Packed MPC designation.
    pub designation: String,
    /// Readable designation.
    pub name: String,
    /// Apparent ICRF direction (light time and observer aberration applied).
    pub direction: Equatorial,
    /// Direction with light time only, no aberration.
    pub astrometric_direction: Equatorial,
    /// V magnitude from the H-G system.
    pub v_magnitude: f64,
    /// Heliocentric distance at emission, AU.
    pub heliocentric_au: f64,
    /// Observer–body distance, AU.
    pub range_au: f64,
    /// One-way light time, seconds.
    pub light_time_s: f64,
    /// Sun–body–observer angle, radians.
    pub phase_angle: f64,
    /// Apparent sky motion relative to the stars, arcseconds per hour.
    pub sky_motion_arcsec_per_hour: f64,
    /// Absolute magnitude used.
    pub h: f64,
    /// Slope parameter used.
    pub g: f64,
    /// Osculating epoch of the elements, TT Julian date.
    pub elements_epoch_tt: f64,
}

impl MinorPlanetSighting {
    /// The body as a first-pass point source.
    pub fn star_data(&self, id: u64) -> StarData {
        StarData::with_position(id, self.direction, self.v_magnitude, Some(MINOR_PLANET_B_V))
    }
}

fn to_equatorial(v: &Vector3<f64>) -> Equatorial {
    let ra = v.y.atan2(v.x).rem_euclid(std::f64::consts::TAU);
    let dec = (v.z / v.norm()).clamp(-1.0, 1.0).asin();
    Equatorial::new(ra, dec)
}

fn unit_vector(direction: &Equatorial) -> Vector3<f64> {
    let (ra, dec) = (direction.ra, direction.dec);
    Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
}

/// Barycentric state of the observer and the Sun at one epoch, the two
/// things every body in a query shares.
struct QueryFrame {
    observer_position_au: Vector3<f64>,
    observer_velocity_au_day: Vector3<f64>,
    sun_position_au: Vector3<f64>,
}

impl SolarSystem {
    /// Every catalogue body within `radius` of `centre` and brighter than
    /// `mag_limit` (V) as seen by `observer` at `epoch`, brightest first.
    ///
    /// The whole catalogue is propagated on every call (about a second for
    /// the full MPCORB on a many-core machine); callers rendering a series
    /// of epochs should keep the catalogue loaded and call this per frame.
    pub fn minor_planets_in_cone(
        &self,
        catalog: &MinorPlanetCatalog,
        observer: &Observer,
        epoch: &Epoch,
        centre: &Equatorial,
        radius: f64,
        mag_limit: f64,
    ) -> Result<Vec<MinorPlanetSighting>, MinorPlanetError> {
        let frame = {
            let mut kernel = self.kernel.lock().expect("ephemeris kernel mutex poisoned");
            let obs = observer.barycentric(&mut kernel, epoch)?;
            let sun = kernel
                .at("sun", epoch.time())
                .map_err(SolarSystemError::from)?;
            QueryFrame {
                observer_position_au: obs.position,
                observer_velocity_au_day: obs.velocity,
                sun_position_au: sun.position,
            }
        };
        let centre_vec = unit_vector(centre);
        let cos_radius = radius.cos();

        let mut sightings: Vec<MinorPlanetSighting> = catalog
            .elements
            .par_iter()
            .filter_map(|body| {
                let h = body.h?;
                let sighting = sight(body, h, &frame, epoch);
                // Cone test on the astrometric direction; the aberration
                // shift is far smaller than any PSF margin a caller adds.
                let d = unit_vector(&sighting.astrometric_direction);
                if d.dot(&centre_vec) < cos_radius || sighting.v_magnitude > mag_limit {
                    return None;
                }
                Some(sighting)
            })
            .collect();
        sightings.sort_by(|a, b| a.v_magnitude.total_cmp(&b.v_magnitude));
        Ok(sightings)
    }

    /// One named body, wherever it is on the sky.
    pub fn minor_planet(
        &self,
        body: &MinorPlanetElements,
        observer: &Observer,
        epoch: &Epoch,
    ) -> Result<Option<MinorPlanetSighting>, MinorPlanetError> {
        let Some(h) = body.h else {
            return Ok(None);
        };
        let frame = {
            let mut kernel = self.kernel.lock().expect("ephemeris kernel mutex poisoned");
            let obs = observer.barycentric(&mut kernel, epoch)?;
            let sun = kernel
                .at("sun", epoch.time())
                .map_err(SolarSystemError::from)?;
            QueryFrame {
                observer_position_au: obs.position,
                observer_velocity_au_day: obs.velocity,
                sun_position_au: sun.position,
            }
        };
        Ok(Some(sight(body, h, &frame, epoch)))
    }
}

/// Geometric (light-time-corrected) observer→body vector in AU and the
/// body's heliocentric position at emission.
fn light_time_corrected(
    body: &MinorPlanetElements,
    frame: &QueryFrame,
    epoch: &Epoch,
) -> (Vector3<f64>, Vector3<f64>, f64) {
    let mut emission = epoch.clone();
    let mut helio = Vector3::zeros();
    let mut rel = Vector3::zeros();
    let mut light_time_days = 0.0;
    for _ in 0..3 {
        helio = body.heliocentric_at(&emission).0;
        rel = frame.sun_position_au + helio - frame.observer_position_au;
        light_time_days = rel.norm() / C_AUDAY;
        emission = epoch.offset_secs(-light_time_days * SECONDS_PER_DAY);
    }
    (rel, helio, light_time_days)
}

fn sight(
    body: &MinorPlanetElements,
    h: f64,
    frame: &QueryFrame,
    epoch: &Epoch,
) -> MinorPlanetSighting {
    let (rel, helio, light_time_days) = light_time_corrected(body, frame, epoch);
    let range_au = rel.norm();
    let geometric = rel / range_au;
    let beta = frame.observer_velocity_au_day / C_AUDAY;
    let apparent = (geometric + beta).normalize();

    let heliocentric_au = helio.norm();
    // Sun–body–observer angle at the body.
    let to_sun = -helio;
    let to_observer = -rel;
    let phase_angle = to_sun.angle(&to_observer);
    let g = body.g.unwrap_or(0.15);
    let v_magnitude = hg_apparent_magnitude(h, g, heliocentric_au, range_au, phase_angle);

    // Sky motion from the geometric direction one hour later; the
    // observer is held fixed, so this is motion relative to the stars as
    // that observer sees them, parallax included.
    let later = epoch.offset_secs(3600.0);
    let (rel_later, _, _) = light_time_corrected(body, frame, &later);
    let sky_motion_arcsec_per_hour = geometric.angle(&rel_later.normalize()).to_degrees() * 3600.0;

    MinorPlanetSighting {
        designation: body.designation.clone(),
        name: body.name.clone(),
        direction: to_equatorial(&apparent),
        astrometric_direction: to_equatorial(&geometric),
        v_magnitude,
        heliocentric_au,
        range_au,
        light_time_s: light_time_days * SECONDS_PER_DAY,
        phase_angle,
        sky_motion_arcsec_per_hour,
        h,
        g,
        elements_epoch_tt: body.epoch_tt,
    }
}

/// IAU H-G apparent magnitude (Bowell et al. 1989).
pub fn hg_apparent_magnitude(h: f64, g: f64, r_au: f64, delta_au: f64, phase_angle: f64) -> f64 {
    let tan_half = (phase_angle / 2.0).tan();
    let phi1 = (-3.332 * tan_half.powf(0.631)).exp();
    let phi2 = (-1.862 * tan_half.powf(1.218)).exp();
    h + 5.0 * (r_au * delta_au).log10() - 2.5 * ((1.0 - g) * phi1 + g * phi2).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solar_system::BodyId;
    use approx::assert_abs_diff_eq;

    /// MPCORB row for (1) Ceres, epoch 2026-06-09 (K2669), from the
    /// 2026-09-14 catalogue.
    const CERES: &str = "00001    3.34  0.15 K2669 274.41935   73.29420   80.24863   10.58803  0.0796923  0.21430445   2.7655526  0 MPO980521  7297 126 1801-2026 0.83 M-v 30k Veres      4000      (1) Ceres              20260103";

    fn ceres() -> MinorPlanetCatalog {
        MinorPlanetCatalog::from_records(vec![parse_mpcorb_line(CERES).unwrap()])
    }

    /// JPL Horizons, Ceres from the geocentre at 2026-09-14 00:00 UTC:
    /// APmag 8.825 with r = 2.686777 AU, Δ = 2.857616 AU, phase 20.614°.
    #[test]
    fn hg_magnitude_matches_horizons_for_ceres() {
        let v = hg_apparent_magnitude(3.34, 0.15, 2.686_777, 2.857_616, 20.614_f64.to_radians());
        eprintln!("residual ceres_v_from_earth={v:.3} horizons=8.825");
        assert_abs_diff_eq!(v, 8.825, epsilon = 0.1);
    }

    #[test]
    fn catalogue_parses_and_finds_by_name() {
        let cat = ceres();
        assert_eq!(cat.len(), 1);
        let body = cat.find("Ceres").unwrap();
        assert_eq!(body.designation, "00001");
        assert_eq!(body.h, Some(3.34));
        assert!(cat.find("00001").is_some());
        assert!(cat.find("Vesta").is_none());
        // 2026-06-09 TT.
        assert_abs_diff_eq!(body.epoch_tt, 2_461_200.5, epsilon = 1e-6);
    }

    #[test]
    fn star_ids_avoid_gaia_range() {
        let cat = ceres();
        assert!(cat.elements()[0].star_id() >= 1 << 63);
    }

    /// Horizons astrometric ICRF positions of Ceres at 2026-09-14 00:00 UTC:
    /// from the geocentre RA 101.706521° Dec +22.985889°, Δ 2.857616 AU,
    /// V 8.825; from the Mars centre RA 84.301025° Dec +22.170577°,
    /// Δ 1.156803 AU, phase 4.7162°, V 6.223. Two-body propagation from
    /// the 2026-06-09 elements should land within a few arcseconds. Needs
    /// the DE440s kernel.
    #[test]
    #[ignore]
    fn ceres_matches_horizons_from_earth_and_mars() {
        let system = SolarSystem::new().unwrap();
        let cat = ceres();
        let body = cat.find("Ceres").unwrap();
        let epoch = Epoch::parse("2026-09-14T00:00:00Z").unwrap();
        for (observer, ra, dec, delta, v, phase_deg) in [
            (
                Observer::BodyCenter(BodyId::EARTH),
                101.706_521,
                22.985_889,
                2.857_616,
                8.825,
                20.614,
            ),
            (
                Observer::BodyCenter(BodyId::MARS),
                84.301_025,
                22.170_577,
                1.156_803,
                6.223,
                4.7162,
            ),
        ] {
            let s = system
                .minor_planet(body, &observer, &epoch)
                .unwrap()
                .unwrap();
            let horizons = Equatorial::from_degrees(ra, dec);
            let sep_arcsec = s
                .astrometric_direction
                .angular_distance(&horizons)
                .to_degrees()
                * 3600.0;
            let aberration_arcsec = s
                .direction
                .angular_distance(&s.astrometric_direction)
                .to_degrees()
                * 3600.0;
            eprintln!(
                "residual ceres_{:?}_astrometric_vs_horizons_arcsec={sep_arcsec:.2} aberration_arcsec={aberration_arcsec:.2} v={:.3} horizons_v={v} range={:.6} phase_deg={:.4} motion_arcsec_per_hour={:.2}",
                observer, s.v_magnitude, s.range_au, s.phase_angle.to_degrees(), s.sky_motion_arcsec_per_hour
            );
            assert!(sep_arcsec < 10.0, "{sep_arcsec}″ from Horizons");
            assert!(
                (5.0..30.0).contains(&aberration_arcsec),
                "aberration {aberration_arcsec}″"
            );
            assert_abs_diff_eq!(s.range_au, delta, epsilon = 2e-4);
            assert_abs_diff_eq!(s.phase_angle.to_degrees(), phase_deg, epsilon = 0.05);
            assert_abs_diff_eq!(s.v_magnitude, v, epsilon = 0.1);
        }
    }

    /// Cone query: Ceres is found in a 1° cone around its own position
    /// and not in one 5° away; the magnitude cut works.
    #[test]
    #[ignore]
    fn cone_query_finds_and_excludes() {
        let system = SolarSystem::new().unwrap();
        let cat = ceres();
        let epoch = Epoch::parse("2026-09-14T00:00:00Z").unwrap();
        let mars = Observer::BodyCenter(BodyId::MARS);
        let here = Equatorial::from_degrees(84.301_025, 22.170_577);
        let found = system
            .minor_planets_in_cone(&cat, &mars, &epoch, &here, 1.0_f64.to_radians(), 20.0)
            .unwrap();
        assert_eq!(found.len(), 1);
        let star = found[0].star_data(cat.elements()[0].star_id());
        assert_abs_diff_eq!(star.magnitude, found[0].v_magnitude);
        let away = Equatorial::from_degrees(89.301_025, 22.170_577);
        assert!(system
            .minor_planets_in_cone(&cat, &mars, &epoch, &away, 1.0_f64.to_radians(), 20.0)
            .unwrap()
            .is_empty());
        assert!(system
            .minor_planets_in_cone(&cat, &mars, &epoch, &here, 1.0_f64.to_radians(), 5.0)
            .unwrap()
            .is_empty());
    }
}

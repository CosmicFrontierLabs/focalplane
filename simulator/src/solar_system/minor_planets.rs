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
//! - Catalogue: `starfield::jpl::mpc::MpcorbCatalog`, which resolves
//!   MPCORB.DAT through the starfield datastore and screens out rows
//!   with no propagatable orbit.
//! - Orbits: two-body propagation of the MPCORB osculating elements
//!   (`starfield::keplerlib`). No planetary
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

use nalgebra::Vector3;
use rayon::prelude::*;
use starfield::catalogs::StarData;
use starfield::constants::{C_AUDAY, DAY_S};
use starfield::jpl::mpc::{MinorPlanet, MpcorbCatalog};
use starfield::jplephem_ext::SpiceKernelExt;
use starfield::magnitudelib::small_body::apparent_magnitude_with_phase;
use starfield::Equatorial;
use thiserror::Error;

use super::{direction_of, unit_vector, Observer, SolarSystem, SolarSystemError};
use crate::epoch::Epoch;

/// Colour assigned to every minor planet: the Sun's B−V plus the mean
/// reddening of the main belt.
pub const MINOR_PLANET_B_V: f64 = 0.75;

/// Errors from minor-planet queries.
#[derive(Debug, Error)]
pub enum MinorPlanetError {
    /// Ephemeris evaluation for the observer or the Sun.
    #[error(transparent)]
    Ephemeris(#[from] SolarSystemError),
}

/// A [`StarData`] id for a catalogue body.
pub trait MinorPlanetStarId {
    /// Stable 64-bit id from the packed designation, kept clear of Gaia
    /// source ids (which are below 2^63).
    fn star_id(&self) -> u64;
}

impl MinorPlanetStarId for MinorPlanet {
    fn star_id(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.designation().hash(&mut hasher);
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
        catalog: &MpcorbCatalog,
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
            .bodies()
            .par_iter()
            .filter_map(|body| {
                let h = body.h()?;
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
        body: &MinorPlanet,
        observer: &Observer,
        epoch: &Epoch,
    ) -> Result<Option<MinorPlanetSighting>, MinorPlanetError> {
        let Some(h) = body.h() else {
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
    body: &MinorPlanet,
    frame: &QueryFrame,
    epoch: &Epoch,
) -> (Vector3<f64>, Vector3<f64>, f64) {
    let mut emission = epoch.clone();
    let mut helio = Vector3::zeros();
    let mut rel = Vector3::zeros();
    let mut light_time_days = 0.0;
    for _ in 0..3 {
        helio = body.heliocentric_at(emission.time()).position;
        rel = frame.sun_position_au + helio - frame.observer_position_au;
        light_time_days = rel.norm() / C_AUDAY;
        emission = epoch.offset_secs(-light_time_days * DAY_S);
    }
    (rel, helio, light_time_days)
}

fn sight(body: &MinorPlanet, h: f64, frame: &QueryFrame, epoch: &Epoch) -> MinorPlanetSighting {
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
    let g = body.g().unwrap_or(0.15);
    let v_magnitude = apparent_magnitude_with_phase(h, heliocentric_au, range_au, phase_angle, g);

    // Sky motion from the geometric direction one hour later; the
    // observer is held fixed, so this is motion relative to the stars as
    // that observer sees them, parallax included.
    let later = epoch.offset_secs(3600.0);
    let (rel_later, _, _) = light_time_corrected(body, frame, &later);
    let sky_motion_arcsec_per_hour = geometric.angle(&rel_later.normalize()).to_degrees() * 3600.0;

    MinorPlanetSighting {
        designation: body.designation().to_string(),
        name: body.name().to_string(),
        direction: direction_of(&apparent),
        astrometric_direction: direction_of(&geometric),
        v_magnitude,
        heliocentric_au,
        range_au,
        light_time_s: light_time_days * DAY_S,
        phase_angle,
        sky_motion_arcsec_per_hour,
        h,
        g,
        elements_epoch_tt: body.epoch_tt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solar_system::BodyId;
    use approx::assert_abs_diff_eq;
    use starfield::jpl::mpc::parse_mpcorb_line;

    /// MPCORB row for (1) Ceres, epoch 2026-06-09 (K2669), from the
    /// 2026-09-14 catalogue.
    const CERES: &str = "00001    3.34  0.15 K2669 274.41935   73.29420   80.24863   10.58803  0.0796923  0.21430445   2.7655526  0 MPO980521  7297 126 1801-2026 0.83 M-v 30k Veres      4000      (1) Ceres              20260103";

    fn ceres() -> MpcorbCatalog {
        MpcorbCatalog::from_records(vec![parse_mpcorb_line(CERES).unwrap()])
    }

    #[test]
    fn catalogue_parses_and_finds_by_name() {
        let cat = ceres();
        assert_eq!(cat.len(), 1);
        let body = cat.find("Ceres").unwrap();
        assert_eq!(body.designation(), "00001");
        assert_eq!(body.h(), Some(3.34));
        assert!(cat.find("00001").is_some());
        assert!(cat.find("Vesta").is_none());
        // 2026-06-09 TT.
        assert_abs_diff_eq!(body.epoch_tt(), 2_461_200.5, epsilon = 1e-6);
    }

    #[test]
    fn star_ids_avoid_gaia_range() {
        let cat = ceres();
        assert!(cat.bodies()[0].star_id() >= 1 << 63);
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
        let star = found[0].star_data(cat.bodies()[0].star_id());
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

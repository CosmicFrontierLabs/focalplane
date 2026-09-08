//! Where solar-system bodies are, as seen from the observing spacecraft.
//!
//! This module turns an [`Epoch`], an [`Observer`] and a [`BodyId`] into
//! a [`BodyState`]: the apparent direction of the body (light-time,
//! gravitational deflection and stellar aberration applied through
//! starfield's `observe`/`apparent` chain), its distance and angular
//! size, and the illumination geometry a renderer needs (phase angle,
//! illuminated fraction, Sun direction, bright-limb position angle).
//!
//! Ephemerides come from a JPL SPK kernel read by starfield's pure-Rust
//! `jplephem` port; the default is `de440s.bsp` (1849–2150), fetched into
//! `~/.cache/starfield` on first use.
//!
//! Body radii and kernel names are tabulated here from the IAU WGCCRE
//! 2015 report (Archinal et al. 2018). Body orientation (pole, prime
//! meridian) is deliberately absent: it arrives with starfield's PCK
//! support and will be consumed from there.

use std::fmt;
use std::sync::Mutex;

use nalgebra::Vector3;
use starfield::jplephem::{JplephemError, SpiceKernel};
use starfield::jplephem_ext::SpiceKernelExt;
use starfield::magnitudelib::planetary_magnitude;
use starfield::positions::Position;
use starfield::{Equatorial, Loader, StarfieldError};
use thiserror::Error;

use crate::epoch::Epoch;

/// Astronomical unit in kilometres (IAU 2012).
pub const AU_KM: f64 = 149_597_870.7;

/// Seconds per day.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// Default planetary ephemeris kernel.
pub const DEFAULT_KERNEL: &str = "de440s.bsp";

/// Errors from ephemeris evaluation.
#[derive(Debug, Error)]
pub enum SolarSystemError {
    /// Kernel download or open failure.
    #[error("loading ephemeris kernel: {0}")]
    Load(#[from] StarfieldError),
    /// Kernel evaluation failure (body not in kernel, time out of range).
    #[error("evaluating ephemeris: {0}")]
    Ephemeris(#[from] JplephemError),
}

/// Bodies with tabulated radii and kernel names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BodyId {
    Sun,
    Mercury,
    Venus,
    Earth,
    Moon,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
}

impl BodyId {
    /// Every tabulated body.
    pub const ALL: [BodyId; 10] = [
        BodyId::Sun,
        BodyId::Mercury,
        BodyId::Venus,
        BodyId::Earth,
        BodyId::Moon,
        BodyId::Mars,
        BodyId::Jupiter,
        BodyId::Saturn,
        BodyId::Uranus,
        BodyId::Neptune,
    ];

    /// NAIF integer id.
    pub fn naif_id(self) -> i32 {
        match self {
            BodyId::Sun => 10,
            BodyId::Mercury => 199,
            BodyId::Venus => 299,
            BodyId::Earth => 399,
            BodyId::Moon => 301,
            BodyId::Mars => 499,
            BodyId::Jupiter => 599,
            BodyId::Saturn => 699,
            BodyId::Uranus => 799,
            BodyId::Neptune => 899,
        }
    }

    /// Name used for SPK lookups. The planetary DE kernels carry Earth,
    /// the Moon and the Sun as bodies and every other planet as a system
    /// barycentre. Mercury and Venus have no moons, so their barycentre
    /// is their centre; Mars's barycentre sits within a kilometre of its
    /// centre; the giant planets' barycentre offsets (up to ~0.07 solar
    /// radii for Jupiter) are far below the pointing precision of any
    /// focal-plane simulation.
    pub fn spice_name(self) -> &'static str {
        match self {
            BodyId::Sun => "sun",
            BodyId::Mercury => "mercury barycenter",
            BodyId::Venus => "venus barycenter",
            BodyId::Earth => "earth",
            BodyId::Moon => "moon",
            BodyId::Mars => "mars barycenter",
            BodyId::Jupiter => "jupiter barycenter",
            BodyId::Saturn => "saturn barycenter",
            BodyId::Uranus => "uranus barycenter",
            BodyId::Neptune => "neptune barycenter",
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            BodyId::Sun => "Sun",
            BodyId::Mercury => "Mercury",
            BodyId::Venus => "Venus",
            BodyId::Earth => "Earth",
            BodyId::Moon => "Moon",
            BodyId::Mars => "Mars",
            BodyId::Jupiter => "Jupiter",
            BodyId::Saturn => "Saturn",
            BodyId::Uranus => "Uranus",
            BodyId::Neptune => "Neptune",
        }
    }

    /// Equatorial radius in km (IAU WGCCRE 2015; Sun per IAU 2015
    /// nominal value).
    pub fn equatorial_radius_km(self) -> f64 {
        match self {
            BodyId::Sun => 695_700.0,
            BodyId::Mercury => 2_440.53,
            BodyId::Venus => 6_051.8,
            BodyId::Earth => 6378.1366,
            BodyId::Moon => 1_737.4,
            BodyId::Mars => 3_396.19,
            BodyId::Jupiter => 71_492.0,
            BodyId::Saturn => 60_268.0,
            BodyId::Uranus => 25_559.0,
            BodyId::Neptune => 24_764.0,
        }
    }

    /// Polar radius in km (IAU WGCCRE 2015).
    pub fn polar_radius_km(self) -> f64 {
        match self {
            BodyId::Sun => 695_700.0,
            BodyId::Mercury => 2_438.26,
            BodyId::Venus => 6_051.8,
            BodyId::Earth => 6356.7519,
            BodyId::Moon => 1_737.4,
            BodyId::Mars => 3_376.20,
            BodyId::Jupiter => 66_854.0,
            BodyId::Saturn => 54_364.0,
            BodyId::Uranus => 24_973.0,
            BodyId::Neptune => 24_341.0,
        }
    }

    /// Flattening `(a − c) / a`.
    pub fn flattening(self) -> f64 {
        let a = self.equatorial_radius_km();
        (a - self.polar_radius_km()) / a
    }

    /// True for the one self-luminous body.
    pub fn is_sun(self) -> bool {
        matches!(self, BodyId::Sun)
    }

    /// Parse a case-insensitive body name.
    pub fn parse(name: &str) -> Option<Self> {
        let lower = name.trim().to_ascii_lowercase();
        BodyId::ALL
            .into_iter()
            .find(|b| b.name().to_ascii_lowercase() == lower)
    }
}

impl fmt::Display for BodyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Where the telescope is.
#[derive(Clone, Debug, PartialEq)]
pub enum Observer {
    /// At the centre of a body (a good enough stand-in for a low orbit
    /// when only angular sizes and phases are needed).
    BodyCenter(BodyId),
    /// Fixed offset from a body centre in the ICRF frame. Position in
    /// km, velocity in km/s.
    OffsetFromBody {
        body: BodyId,
        position_km: Vector3<f64>,
        velocity_km_s: Vector3<f64>,
    },
    /// Explicit solar-system-barycentric state: position in AU,
    /// velocity in AU/day.
    Barycentric {
        position_au: Vector3<f64>,
        velocity_au_day: Vector3<f64>,
    },
}

impl Observer {
    /// Barycentric state of the observer at `epoch`.
    fn barycentric(
        &self,
        kernel: &mut SpiceKernel,
        epoch: &Epoch,
    ) -> Result<Position, SolarSystemError> {
        match self {
            Observer::BodyCenter(body) => Ok(kernel.at(body.spice_name(), epoch.time())?),
            Observer::OffsetFromBody {
                body,
                position_km,
                velocity_km_s,
            } => {
                let center = kernel.at(body.spice_name(), epoch.time())?;
                Ok(Position::barycentric(
                    center.position + position_km / AU_KM,
                    center.velocity + velocity_km_s * (SECONDS_PER_DAY / AU_KM),
                    body.naif_id(),
                ))
            }
            Observer::Barycentric {
                position_au,
                velocity_au_day,
            } => Ok(Position::barycentric(*position_au, *velocity_au_day, 0)),
        }
    }
}

/// Illumination geometry of a body from three barycentric positions.
///
/// All directions are ICRF unit vectors rooted at the body.
#[derive(Clone, Debug, PartialEq)]
pub struct IlluminationGeometry {
    /// Sun–body–observer angle, radians.
    pub phase_angle: f64,
    /// `(1 + cos α) / 2`.
    pub illuminated_fraction: f64,
    /// Unit vector from the body toward the Sun.
    pub sun_direction: Vector3<f64>,
    /// Unit vector from the body toward the observer.
    pub observer_direction: Vector3<f64>,
    /// Body–Sun distance in AU.
    pub heliocentric_distance_au: f64,
    /// Position angle of the bright limb's midpoint on the observer's
    /// sky, radians east of celestial (ICRF) north. `0` when the Sun
    /// lies along the line of sight.
    pub bright_limb_position_angle: f64,
}

impl IlluminationGeometry {
    /// Geometry from barycentric positions (any consistent unit) of the
    /// observer, the body (at its light-time-corrected epoch) and the
    /// Sun.
    pub fn from_barycentric(observer: Vector3<f64>, body: Vector3<f64>, sun: Vector3<f64>) -> Self {
        let to_sun = sun - body;
        let to_observer = observer - body;
        let heliocentric_distance_au = to_sun.norm();
        let sun_direction = to_sun / heliocentric_distance_au;
        let observer_direction = to_observer.normalize();
        let cos_alpha = sun_direction.dot(&observer_direction).clamp(-1.0, 1.0);
        let phase_angle = cos_alpha.acos();

        // Sky-plane basis at the body's apparent direction.
        let line_of_sight = -observer_direction;
        let (east, north) = sky_basis(&line_of_sight);
        let sun_in_sky = sun_direction - line_of_sight * sun_direction.dot(&line_of_sight);
        let bright_limb_position_angle = if sun_in_sky.norm() < 1e-12 {
            0.0
        } else {
            sun_in_sky.dot(&east).atan2(sun_in_sky.dot(&north))
        };

        Self {
            phase_angle,
            illuminated_fraction: 0.5 * (1.0 + cos_alpha),
            sun_direction,
            observer_direction,
            heliocentric_distance_au,
            bright_limb_position_angle,
        }
    }
}

/// Local east and north unit vectors on the sky at a line-of-sight
/// direction in the ICRF frame.
fn sky_basis(line_of_sight: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let pole = Vector3::z();
    let mut east = pole.cross(line_of_sight);
    if east.norm() < 1e-12 {
        east = Vector3::x();
    }
    let east = east.normalize();
    let north = line_of_sight.cross(&east).normalize();
    (east, north)
}

/// A body as seen from the observer at one epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyState {
    /// Which body.
    pub body: BodyId,
    /// Apparent ICRF direction (light-time, deflection and aberration
    /// applied), radians.
    pub direction: Equatorial,
    /// Observer–body distance at the light-time-corrected epoch, AU.
    pub distance_au: f64,
    /// One-way light time, seconds.
    pub light_time_s: f64,
    /// Angular semi-diameter of the equatorial radius, radians.
    pub angular_semi_diameter: f64,
    /// Sun and observer directions, phase, limb orientation.
    pub illumination: IlluminationGeometry,
    /// Apparent V magnitude from Mallama & Hilton (2018) when starfield
    /// has a formula for the body; `None` for the Sun and Moon.
    pub v_magnitude: Option<f64>,
}

impl BodyState {
    /// Angular diameter in arcseconds.
    pub fn angular_diameter_arcsec(&self) -> f64 {
        2.0 * self.angular_semi_diameter.to_degrees() * 3600.0
    }

    /// Observer–body distance in km.
    pub fn distance_km(&self) -> f64 {
        self.distance_au * AU_KM
    }
}

/// Ephemeris evaluator shared by every body in a scene.
///
/// Wraps the SPK kernel in a mutex so it can sit behind a `Send + Sync`
/// second pass; kernel evaluation is microseconds, far below any render
/// cost.
pub struct SolarSystem {
    kernel: Mutex<SpiceKernel>,
}

impl fmt::Debug for SolarSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SolarSystem { kernel: SpiceKernel }")
    }
}

impl SolarSystem {
    /// Load the default DE440s kernel, downloading it if absent.
    pub fn new() -> Result<Self, SolarSystemError> {
        Self::with_kernel(DEFAULT_KERNEL)
    }

    /// Load a named JPL kernel (e.g. `de440.bsp`), downloading if absent.
    pub fn with_kernel(filename: &str) -> Result<Self, SolarSystemError> {
        Ok(Self::from_kernel(Loader::new().open(filename)?))
    }

    /// Wrap an already-open kernel.
    pub fn from_kernel(kernel: SpiceKernel) -> Self {
        Self {
            kernel: Mutex::new(kernel),
        }
    }

    /// Apparent state of `body` from `observer` at `epoch`.
    pub fn body_state(
        &self,
        body: BodyId,
        observer: &Observer,
        epoch: &Epoch,
    ) -> Result<BodyState, SolarSystemError> {
        let mut kernel = self.kernel.lock().expect("ephemeris kernel mutex poisoned");
        let observer_bary = observer.barycentric(&mut kernel, epoch)?;
        let astrometric = observer_bary.observe(body.spice_name(), &mut kernel, epoch.time())?;
        let apparent = astrometric.apparent(&mut kernel, epoch.time())?;
        let (ra_hours, dec_degrees, _) = apparent.radec(None);
        let direction = Equatorial::from_degrees(ra_hours * 15.0, dec_degrees);

        let body_bary = observer_bary.position + astrometric.position;
        let emission = epoch.offset_secs(-astrometric.light_time * SECONDS_PER_DAY);
        let sun_bary = if body.is_sun() {
            body_bary
        } else {
            kernel
                .at(BodyId::Sun.spice_name(), emission.time())?
                .position
        };
        let illumination =
            IlluminationGeometry::from_barycentric(observer_bary.position, body_bary, sun_bary);

        let distance_au = astrometric.distance();
        let angular_semi_diameter = (body.equatorial_radius_km() / (distance_au * AU_KM))
            .clamp(-1.0, 1.0)
            .asin();
        let v_magnitude = planetary_magnitude(&astrometric, epoch.time()).ok();

        Ok(BodyState {
            body,
            direction,
            distance_au,
            light_time_s: astrometric.light_time * SECONDS_PER_DAY,
            angular_semi_diameter,
            illumination,
            v_magnitude,
        })
    }

    /// Angle between a sky direction and the apparent Sun, radians.
    pub fn solar_elongation(
        &self,
        direction: &Equatorial,
        observer: &Observer,
        epoch: &Epoch,
    ) -> Result<f64, SolarSystemError> {
        let sun = self.body_state(BodyId::Sun, observer, epoch)?;
        Ok(direction.angular_distance(&sun.direction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn body_table_is_consistent() {
        for body in BodyId::ALL {
            assert!(body.polar_radius_km() <= body.equatorial_radius_km());
            assert!(body.flattening() >= 0.0);
            assert_eq!(BodyId::parse(body.name()), Some(body));
            assert_eq!(BodyId::parse(&body.name().to_uppercase()), Some(body));
        }
        assert_abs_diff_eq!(BodyId::Jupiter.flattening(), 0.0649, epsilon = 1e-4);
        assert_abs_diff_eq!(BodyId::Earth.flattening(), 1.0 / 298.257, epsilon = 1e-5);
        assert_eq!(BodyId::parse("Phobos"), None);
        assert_eq!(BodyId::Mars.naif_id(), 499);
    }

    #[test]
    fn phase_angle_is_zero_at_opposition_and_pi_at_conjunction() {
        let sun = Vector3::zeros();
        let observer = Vector3::new(1.5, 0.0, 0.0);
        // Body beyond the observer on the anti-Sun line: fully lit.
        let opposition =
            IlluminationGeometry::from_barycentric(observer, Vector3::new(2.5, 0.0, 0.0), sun);
        assert_abs_diff_eq!(opposition.phase_angle, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(opposition.illuminated_fraction, 1.0, epsilon = 1e-12);
        // Body between observer and Sun: night side toward us.
        let conjunction =
            IlluminationGeometry::from_barycentric(observer, Vector3::new(1.0, 0.0, 0.0), sun);
        assert_abs_diff_eq!(conjunction.phase_angle, PI, epsilon = 1e-12);
        assert_abs_diff_eq!(conjunction.illuminated_fraction, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn phase_angle_at_quadrature_like_geometry() {
        // Observer at 1.5 AU on x, body at 1 AU on y, Sun at origin:
        // angle at the body between (0,−1,0) and (1.5,−1,0).
        let geom = IlluminationGeometry::from_barycentric(
            Vector3::new(1.5, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::zeros(),
        );
        assert_abs_diff_eq!(geom.phase_angle, (1.5_f64).atan(), epsilon = 1e-12);
        assert_abs_diff_eq!(geom.heliocentric_distance_au, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn bright_limb_points_toward_the_sun_on_the_sky() {
        // Observer at origin, body on +x (RA 0, Dec 0). Sun displaced
        // toward +z (north) → PA 0; toward +y (east) → PA 90°.
        let observer = Vector3::zeros();
        let body = Vector3::new(1.0, 0.0, 0.0);
        let north =
            IlluminationGeometry::from_barycentric(observer, body, Vector3::new(1.0, 0.0, 1.0));
        assert_abs_diff_eq!(north.bright_limb_position_angle, 0.0, epsilon = 1e-12);
        let east =
            IlluminationGeometry::from_barycentric(observer, body, Vector3::new(1.0, 1.0, 0.0));
        assert_abs_diff_eq!(east.bright_limb_position_angle, FRAC_PI_2, epsilon = 1e-12);
        let south =
            IlluminationGeometry::from_barycentric(observer, body, Vector3::new(1.0, 0.0, -1.0));
        assert_abs_diff_eq!(south.bright_limb_position_angle.abs(), PI, epsilon = 1e-12);
    }

    #[test]
    fn body_state_angular_helpers() {
        let state = BodyState {
            body: BodyId::Earth,
            direction: Equatorial::from_degrees(0.0, 0.0),
            distance_au: 1.0,
            light_time_s: 499.0,
            angular_semi_diameter: (BodyId::Earth.equatorial_radius_km() / AU_KM).asin(),
            illumination: IlluminationGeometry::from_barycentric(
                Vector3::zeros(),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(1.0, 1.0, 0.0),
            ),
            v_magnitude: None,
        };
        // Earth's equatorial radius subtends 8.794″ at 1 AU (the solar
        // parallax), so its diameter is 17.59″.
        assert_abs_diff_eq!(state.angular_diameter_arcsec(), 17.588, epsilon = 0.01);
        assert_abs_diff_eq!(state.distance_km(), AU_KM, epsilon = 1e-6);
    }

    /// HiRISE PSP_005558_9040 (2007-10-03): range 142 million km (so
    /// Earth's 12 756 km diameter subtends 18.5″ and the Moon's 3 475 km
    /// subtends 5.0″), phase angle 98°, illuminated fraction 43.2 %,
    /// Earth–Moon separation 32″–49″ through the day. Needs the DE440s
    /// kernel (network on first run).
    #[test]
    #[ignore]
    fn earth_from_mars_matches_hirise_release_geometry() {
        let system = SolarSystem::new().unwrap();
        let epoch = Epoch::parse("2007-10-03T05:30:00Z").unwrap();
        let mars = Observer::BodyCenter(BodyId::Mars);
        let earth = system.body_state(BodyId::Earth, &mars, &epoch).unwrap();
        let moon = system.body_state(BodyId::Moon, &mars, &epoch).unwrap();

        assert_abs_diff_eq!(earth.distance_km(), 1.42e8, epsilon = 1.5e6);
        assert_abs_diff_eq!(earth.angular_diameter_arcsec(), 18.5, epsilon = 0.3);
        assert_abs_diff_eq!(moon.angular_diameter_arcsec(), 5.05, epsilon = 0.1);
        assert_abs_diff_eq!(
            earth.illumination.phase_angle.to_degrees(),
            98.0,
            epsilon = 1.0
        );
        assert_abs_diff_eq!(
            earth.illumination.illuminated_fraction,
            0.432,
            epsilon = 0.01
        );
        let separation_arcsec = earth
            .direction
            .angular_distance(&moon.direction)
            .to_degrees()
            * 3600.0;
        assert!(
            (30.0..70.0).contains(&separation_arcsec),
            "Earth–Moon separation {separation_arcsec:.1}″"
        );
        let v = earth.v_magnitude.expect("Mallama & Hilton covers Earth");
        assert!((-3.0..-1.5).contains(&v), "Earth V from Mars = {v:.2}");
        assert!(moon.v_magnitude.is_none());

        let elongation = system
            .solar_elongation(&earth.direction, &mars, &epoch)
            .unwrap()
            .to_degrees();
        // asin(a_E / a_M) is 41° for circular orbits; eccentricity lets
        // Earth reach ~47.4° (Earth at aphelion, Mars at perihelion).
        assert!(
            elongation < 47.5,
            "Earth elongation from Mars {elongation:.1}° exceeds the orbital maximum"
        );
    }
}

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
//! Body radii come from starfield's embedded IAU WGCCRE 2015 table
//! (Archinal et al. 2018); body-fixed frames from starfield's
//! `PlanetaryConstants::frame_for` (ITRS for Earth, the DE440 principal
//! axes for the Moon when a PA kernel is loaded, IAU rotational elements
//! otherwise). Kernel names are the only thing tabulated locally.

use std::fmt;
use std::sync::Mutex;

use nalgebra::{Matrix3, Vector3};
use starfield::framelib::Frame;
use starfield::jplephem::{JplephemError, SpiceKernel};
use starfield::jplephem_ext::SpiceKernelExt;
use starfield::magnitudelib::planetary_magnitude;
use starfield::planetarylib::subpoint::SubPoint;
use starfield::planetarylib::PlanetaryConstants;
use starfield::planetlib::Body;
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
    /// Kernel download or open failure, or a geometry helper that could
    /// not place the Sun or find a body frame.
    #[error("starfield: {0}")]
    Starfield(#[from] StarfieldError),
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

    /// The starfield body carrying the embedded IAU constants.
    pub fn body(self) -> Body {
        match self {
            BodyId::Sun => Body::Sun,
            BodyId::Mercury => Body::Mercury,
            BodyId::Venus => Body::Venus,
            BodyId::Earth => Body::Earth,
            BodyId::Moon => Body::Moon,
            BodyId::Mars => Body::Mars,
            BodyId::Jupiter => Body::Jupiter,
            BodyId::Saturn => Body::Saturn,
            BodyId::Uranus => Body::Uranus,
            BodyId::Neptune => Body::Neptune,
        }
    }

    /// Triaxial radii `[a, b, c]` in km from starfield's IAU 2015 table.
    pub fn radii_km(self) -> [f64; 3] {
        self.body().radii_km()
    }

    /// Equatorial radius in km.
    pub fn equatorial_radius_km(self) -> f64 {
        self.radii_km()[0]
    }

    /// Polar radius in km.
    pub fn polar_radius_km(self) -> f64 {
        self.radii_km()[2]
    }

    /// Flattening `(a − c) / a`.
    pub fn flattening(self) -> f64 {
        self.body().flattening()
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

impl IlluminationGeometry {
    /// Sun direction in the body's sky frame `(east, north, toward
    /// observer)`; the frame the stamp rasteriser lights the disk in.
    pub fn sun_direction_sky_frame(&self) -> Vector3<f64> {
        let toward_observer = self.observer_direction;
        let (east, north) = sky_basis(&-toward_observer);
        Vector3::new(
            self.sun_direction.dot(&east),
            self.sun_direction.dot(&north),
            self.sun_direction.dot(&toward_observer),
        )
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
    /// Apparent V magnitude from Mallama & Hilton (2018), or the lunar
    /// phase curve for the Moon; `None` for the Sun.
    pub v_magnitude: Option<f64>,
    /// Body-fixed point under the observer, planetocentric, longitude
    /// east-positive in `[0, 2π)`: what the disk centre shows.
    pub sub_observer: SubPoint,
    /// Body-fixed point under the Sun, planetocentric; `None` for the Sun.
    pub sub_solar: Option<SubPoint>,
    /// Position angle of the body's north pole on the sky, radians east
    /// of celestial north.
    pub north_pole_position_angle: f64,
    /// Projected ellipse of the oblate body: `(semi_major_rad,
    /// semi_minor_rad, position_angle_rad)`.
    pub apparent_ellipse: (f64, f64, f64),
    /// Rotation from the body's sky frame `(east, north, toward observer)`
    /// into body-fixed coordinates at the light-time-corrected epoch. A
    /// surface normal expressed in the sky frame becomes the body-fixed
    /// direction a texture is sampled at.
    pub sky_to_body_fixed: Matrix3<f64>,
}

/// A fixed point on a body's surface, in the body's geodetic
/// (planetographic) coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceSite {
    /// Label for logs and overlays.
    pub name: String,
    /// Geodetic latitude, degrees, north positive.
    pub latitude_deg: f64,
    /// Longitude, degrees, east positive.
    pub longitude_east_deg: f64,
    /// Height above the reference ellipsoid, metres.
    pub height_m: f64,
}

impl SurfaceSite {
    /// Position of the site relative to the body centre in body-fixed
    /// coordinates, km, on the oblate ellipsoid with equatorial radius
    /// `a` and polar radius `c`.
    pub fn body_fixed_km(&self, radii_km: [f64; 3]) -> Vector3<f64> {
        let a = radii_km[0];
        let c = radii_km[2];
        let e2 = 1.0 - (c * c) / (a * a);
        let lat = self.latitude_deg.to_radians();
        let lon = self.longitude_east_deg.to_radians();
        let (sin_lat, cos_lat) = lat.sin_cos();
        let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        let h = self.height_m / 1000.0;
        Vector3::new(
            (n + h) * cos_lat * lon.cos(),
            (n + h) * cos_lat * lon.sin(),
            (n * (1.0 - e2) + h) * sin_lat,
        )
    }

    /// Outward geodetic surface normal in body-fixed coordinates.
    pub fn body_fixed_normal(&self) -> Vector3<f64> {
        let lat = self.latitude_deg.to_radians();
        let lon = self.longitude_east_deg.to_radians();
        Vector3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin())
    }
}

/// Where a [`SurfaceSite`] appears against its body's disk.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteView {
    /// Angular offset of the site from the body centre on the sky,
    /// radians: `(east, north)`.
    pub sky_offset_rad: (f64, f64),
    /// Apparent direction of the site (body centre direction plus the
    /// small-angle offset).
    pub direction: Equatorial,
    /// Cosine of the emission angle at the site (surface normal against
    /// the direction to the observer). Positive means the site faces
    /// the observer.
    pub emission_cosine: f64,
    /// Cosine of the solar incidence angle at the site; `None` for the
    /// Sun itself.
    pub incidence_cosine: Option<f64>,
    /// True when the site is on the observer-facing hemisphere and
    /// therefore imaged (lit or not).
    pub visible: bool,
}

/// How a site presents to the observer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteStatus {
    /// On the facing hemisphere and sunlit: imaged on the bright disk.
    VisibleDay,
    /// On the facing hemisphere but past the terminator: imaged on the
    /// dark disk (in silhouette against nothing; not a photometric
    /// feature).
    VisibleNight,
    /// On the far hemisphere: occulted by the body itself.
    FarSide,
}

impl SiteView {
    /// Classify the site: far side, or visible by day or by night.
    pub fn status(&self) -> SiteStatus {
        if !self.visible {
            SiteStatus::FarSide
        } else if self.incidence_cosine.is_none_or(|c| c > 0.0) {
            SiteStatus::VisibleDay
        } else {
            SiteStatus::VisibleNight
        }
    }
}

impl fmt::Display for SiteStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SiteStatus::VisibleDay => "visible, day side",
            SiteStatus::VisibleNight => "visible, night side",
            SiteStatus::FarSide => "not visible, far side",
        })
    }
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

    /// Project a surface site onto the sky with the same light-time-
    /// corrected orientation the disk is rendered with. The offset is
    /// the site's body-fixed position rotated into the sky frame and
    /// divided by the distance (small-angle; the disk is always well
    /// under a degree across).
    pub fn view_site(&self, site: &SurfaceSite) -> SiteView {
        let radii = self.body.radii_km();
        // sky_to_body_fixed is orthonormal, so its transpose maps
        // body-fixed vectors into the sky frame.
        let body_to_sky = self.sky_to_body_fixed.transpose();
        let position_sky = body_to_sky * site.body_fixed_km(radii);
        let normal_sky = body_to_sky * site.body_fixed_normal();
        let distance_km = self.distance_km();
        let east = position_sky.x / distance_km;
        let north = position_sky.y / distance_km;
        // Toward-observer axis is +z of the sky frame.
        let emission_cosine = normal_sky.z;
        let incidence_cosine = if self.body.is_sun() {
            None
        } else {
            Some(normal_sky.dot(&self.illumination.sun_direction_sky_frame()))
        };
        let cos_dec = self.direction.dec.cos().max(1e-12);
        let direction = Equatorial::new(
            self.direction.ra + east / cos_dec,
            self.direction.dec + north,
        );
        SiteView {
            sky_offset_rad: (east, north),
            direction,
            emission_cosine,
            incidence_cosine,
            visible: emission_cosine > 0.0,
        }
    }
}

/// Where to aim a beam so it reaches a moving target: the transmit
/// counterpart of an apparent position.
#[derive(Clone, Debug, PartialEq)]
pub struct TransmitAimpoint {
    /// When the photons arrive at the target.
    pub arrival_epoch: Epoch,
    /// Forward one-way light time, seconds.
    pub light_time_s: f64,
    /// Transmitter-at-emission to target-at-arrival distance, AU.
    pub distance_au: f64,
    /// ICRF direction from the transmitter to where the target will be
    /// at arrival, before aberration.
    pub geometric_direction: Equatorial,
    /// Direction to aim the beam in the transmitter's own frame, after
    /// removing the transmitter's velocity aberration.
    pub aim_direction: Equatorial,
    /// Angle between `geometric_direction` and `aim_direction`, radians
    /// (≈ |v_tx × d̂| / c).
    pub aberration_angle: f64,
    /// Transmitter barycentric velocity over c, ICRF components.
    pub transmitter_velocity_over_c: Vector3<f64>,
}

/// Direction to aim so that, after the transmitter's own velocity
/// aberration, the beam travels along `geometric_direction` in the
/// barycentric frame: `d − β` normalised, the inverse of the receive
/// correction `d + β` that starfield applies to observed light.
pub fn aim_before_aberration(
    geometric_direction: Vector3<f64>,
    beta: Vector3<f64>,
) -> Vector3<f64> {
    (geometric_direction - beta).normalize()
}

impl TransmitAimpoint {
    /// Point-ahead angle, radians: how far the aim direction lies from a
    /// received apparent direction of the same target.
    pub fn point_ahead_from(&self, received_apparent: &Equatorial) -> f64 {
        self.aim_direction.angular_distance(received_apparent)
    }
}

/// Ephemeris evaluator shared by every body in a scene.
///
/// Wraps the SPK kernel in a mutex so it can sit behind a `Send + Sync`
/// second pass; kernel evaluation is microseconds, far below any render
/// cost.
pub struct SolarSystem {
    kernel: Mutex<SpiceKernel>,
    /// Body orientation source: text and binary PCK kernels read so far,
    /// falling back to the embedded IAU 2015 table.
    constants: PlanetaryConstants,
}

impl fmt::Debug for SolarSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SolarSystem { kernel: SpiceKernel, constants: PlanetaryConstants }")
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

    /// Wrap an already-open kernel, with body orientation from the
    /// embedded IAU 2015 table.
    pub fn from_kernel(kernel: SpiceKernel) -> Self {
        Self {
            kernel: Mutex::new(kernel),
            constants: PlanetaryConstants::new(),
        }
    }

    /// Replace the planetary constants (for example after reading
    /// `pck00011.tpc` or a lunar principal-axes kernel into them).
    pub fn with_constants(mut self, constants: PlanetaryConstants) -> Self {
        self.constants = constants;
        self
    }

    /// Body-fixed frame for `body`: ITRS for Earth, the lunar principal
    /// axes when a PA kernel is loaded, IAU rotational elements otherwise.
    pub fn frame_for(&self, body: BodyId) -> Result<Box<dyn Frame>, SolarSystemError> {
        Ok(self.constants.frame_for(body.naif_id())?)
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
        let radii = body.radii_km();
        let angular_semi_diameter = astrometric.angular_semi_diameter(radii);
        let v_magnitude = planetary_magnitude(&astrometric, epoch.time()).ok();

        let frame = self.constants.frame_for(body.naif_id())?;
        let t = epoch.time();
        let sub_observer = astrometric
            .sub_observer_point(frame.as_ref(), radii, t)
            .to_planetocentric(radii);
        let sub_solar = if body.is_sun() {
            None
        } else {
            Some(
                astrometric
                    .sub_solar_point(frame.as_ref(), radii, &mut kernel, t)?
                    .to_planetocentric(radii),
            )
        };
        let north_pole_position_angle = astrometric.north_pole_position_angle(frame.as_ref(), t);
        let apparent_ellipse = astrometric.apparent_ellipse(frame.as_ref(), radii, t);

        // Body-fixed orientation when the light left the body, composed
        // with the same sky basis the illumination geometry uses.
        let icrf_to_body_fixed = frame.rotation_at(emission.time());
        let toward_observer = illumination.observer_direction;
        let (east, north) = sky_basis(&-toward_observer);
        let sky_to_icrf = Matrix3::from_columns(&[east, north, toward_observer]);
        let sky_to_body_fixed = icrf_to_body_fixed * sky_to_icrf;

        Ok(BodyState {
            body,
            direction,
            distance_au,
            light_time_s: astrometric.light_time * SECONDS_PER_DAY,
            angular_semi_diameter,
            illumination,
            v_magnitude,
            sub_observer,
            sub_solar,
            north_pole_position_angle,
            apparent_ellipse,
            sky_to_body_fixed,
        })
    }

    /// Where a transmitter at `observer` must aim at `epoch` so that its
    /// photons arrive at `target` (optionally at a fixed `site` on it).
    ///
    /// This is the *transmit* problem, the mirror of what
    /// [`SolarSystem::body_state`] solves for reception:
    ///
    /// 1. Solve the forward light time: `t_arr = t + |r_target(t_arr) −
    ///    r_tx(t)| / c`, iterated to convergence, with the target (and
    ///    the site, rotated with the body frame) taken at the arrival
    ///    epoch.
    /// 2. The geometric aim direction is the unit vector from the
    ///    transmitter at `t` to the target at `t_arr` in the ICRF.
    /// 3. Transmitter aberration: a beam launched along `d′` in the
    ///    transmitter's frame propagates along `d ∝ d′ + v/c` in the
    ///    barycentric frame, so to send photons along `d` the beam must
    ///    be aimed along `d′ ∝ d − v/c` (first order).
    ///
    /// The point-ahead angle is the angle between this aim direction and
    /// the received apparent direction of the same target, and is of
    /// order `2 v_rel / c`: tens of arcseconds between Earth and Mars.
    pub fn transmit_aimpoint(
        &self,
        target: BodyId,
        site: Option<&SurfaceSite>,
        observer: &Observer,
        epoch: &Epoch,
    ) -> Result<TransmitAimpoint, SolarSystemError> {
        let mut kernel = self.kernel.lock().expect("ephemeris kernel mutex poisoned");
        let tx = observer.barycentric(&mut kernel, epoch)?;
        let radii = target.radii_km();
        let frame = self.constants.frame_for(target.naif_id())?;

        // Site position in the ICRF at an arrival epoch, or the body
        // centre when no site is given.
        let target_position =
            |kernel: &mut SpiceKernel, arrival: &Epoch| -> Result<Vector3<f64>, SolarSystemError> {
                let centre = kernel.at(target.spice_name(), arrival.time())?.position;
                Ok(match site {
                    Some(site) => {
                        let body_fixed_km = site.body_fixed_km(radii);
                        let icrf_km = frame.rotation_at(arrival.time()).transpose() * body_fixed_km;
                        centre + icrf_km / AU_KM
                    }
                    None => centre,
                })
            };

        let mut light_time_days = 0.0;
        let mut arrival = epoch.clone();
        let mut geometric = Vector3::zeros();
        for _ in 0..8 {
            let r_target = target_position(&mut kernel, &arrival)?;
            geometric = r_target - tx.position;
            light_time_days = geometric.norm() / starfield::constants::C_AUDAY;
            arrival = epoch.offset_secs(light_time_days * SECONDS_PER_DAY);
        }
        let distance_au = geometric.norm();
        let geometric_direction = geometric / distance_au;
        // tx.velocity is AU/day; v/c in the same units.
        let beta = tx.velocity / starfield::constants::C_AUDAY;
        let aim_direction = aim_before_aberration(geometric_direction, beta);

        let to_equatorial = |v: Vector3<f64>| {
            let ra = v.y.atan2(v.x).rem_euclid(std::f64::consts::TAU);
            let dec = (v.z / v.norm()).clamp(-1.0, 1.0).asin();
            Equatorial::new(ra, dec)
        };

        Ok(TransmitAimpoint {
            arrival_epoch: arrival,
            light_time_s: light_time_days * SECONDS_PER_DAY,
            distance_au,
            geometric_direction: to_equatorial(geometric_direction),
            aim_direction: to_equatorial(aim_direction),
            aberration_angle: geometric_direction.angle(&aim_direction),
            transmitter_velocity_over_c: beta,
        })
    }

    /// Run `f` with the ephemeris kernel locked; for cross-checks that
    /// need starfield's `Position` helpers directly.
    pub fn with_ephemeris<R>(&self, f: impl FnOnce(&mut SpiceKernel) -> R) -> R {
        let mut kernel = self.kernel.lock().expect("ephemeris kernel mutex poisoned");
        f(&mut kernel)
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
    use approx::{assert_abs_diff_eq, assert_relative_eq};
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
            sub_observer: SubPoint {
                lon_rad: 0.0,
                lat_rad: 0.0,
                planetocentric: true,
            },
            sub_solar: None,
            north_pole_position_angle: 0.0,
            apparent_ellipse: (0.0, 0.0, 0.0),
            sky_to_body_fixed: Matrix3::identity(),
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
        // The Moon is ~4–5 mag fainter than Earth at the same phase.
        let moon_v = moon.v_magnitude.expect("starfield covers the Moon");
        assert!(
            (1.0..4.5).contains(&moon_v),
            "Moon V from Mars = {moon_v:.2} (Earth {v:.2})"
        );

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

        // Body-fixed geometry from the embedded IAU frame: real, finite,
        // planetocentric, and the Earth sub-points are on opposite sides
        // of the terminator for a 98° phase.
        assert!(earth.sub_observer.planetocentric);
        assert!((0.0..std::f64::consts::TAU).contains(&earth.sub_observer.lon_rad));
        assert!(earth.sub_observer.lat_rad.abs() <= std::f64::consts::FRAC_PI_2);
        let sub_solar = earth.sub_solar.expect("Earth has a sub-solar point");
        let dlon = (earth.sub_observer.lon_rad - sub_solar.lon_rad)
            .rem_euclid(std::f64::consts::TAU)
            .to_degrees();
        let dlon = dlon.min(360.0 - dlon);
        assert!(
            (80.0..115.0).contains(&dlon),
            "sub-point longitude gap {dlon:.1}°"
        );
        assert!(earth.north_pole_position_angle.is_finite());
        // The sky-frame "toward observer" axis maps to the sub-observer
        // direction in body-fixed coordinates, and the matrix is a
        // rotation.
        let disk_centre = earth.sky_to_body_fixed * Vector3::z();
        let lon = disk_centre
            .y
            .atan2(disk_centre.x)
            .rem_euclid(std::f64::consts::TAU);
        let lat = disk_centre.z.asin();
        let dlon_centre = (lon - earth.sub_observer.lon_rad)
            .rem_euclid(std::f64::consts::TAU)
            .min(
                std::f64::consts::TAU
                    - (lon - earth.sub_observer.lon_rad).rem_euclid(std::f64::consts::TAU),
            );
        assert_abs_diff_eq!(dlon_centre, 0.0, epsilon = 2e-4);
        assert_abs_diff_eq!(lat, earth.sub_observer.lat_rad, epsilon = 2e-4);
        // Orthonormal; the sky basis (east, north, toward observer) is
        // left-handed, so the determinant is −1, which is a valid change
        // of basis rather than a mirrored texture.
        assert_abs_diff_eq!(
            earth.sky_to_body_fixed.determinant().abs(),
            1.0,
            epsilon = 1e-9
        );
        let east_bf = earth.sky_to_body_fixed * Vector3::x();
        let north_bf = earth.sky_to_body_fixed * Vector3::y();
        assert_abs_diff_eq!(east_bf.dot(&north_bf), 0.0, epsilon = 1e-9);
        // Seen from outside, a body's east longitude increases toward
        // sky *west* (the Moon's Mare Crisium limb is IAU east and sky
        // west), so the sky-east axis projects negatively onto the local
        // east tangent at the sub-observer point. A mirrored texture
        // would flip this sign.
        let local_east = Vector3::new(-disk_centre.y, disk_centre.x, 0.0).normalize();
        assert!(
            east_bf.dot(&local_east) < 0.0,
            "sky east should map to decreasing body longitude"
        );

        // Palomar (MPC 675, Caltech almanac) at this epoch: the site is
        // 117° of longitude from the sub-observer point, so it is on the
        // far side; a site placed at the sub-observer point is at disk
        // centre facing us; one 60° east of it lies toward sky west and
        // inside the disk.
        let palomar = SurfaceSite {
            name: "Palomar".into(),
            latitude_deg: 33.356_667,
            longitude_east_deg: -116.8625,
            height_m: 1706.0,
        };
        let view = earth.view_site(&palomar);
        assert_eq!(view.status(), SiteStatus::FarSide);
        let centre = SurfaceSite {
            name: "sub-observer".into(),
            latitude_deg: earth
                .sub_observer
                .to_planetographic(BodyId::Earth.radii_km())
                .lat_rad
                .to_degrees(),
            longitude_east_deg: earth.sub_observer.lon_rad.to_degrees(),
            height_m: 0.0,
        };
        let cv = earth.view_site(&centre);
        assert!(
            cv.sky_offset_rad.0.hypot(cv.sky_offset_rad.1) < 1e-4 * earth.angular_semi_diameter
        );
        assert!(cv.emission_cosine > 0.9999);
        let east_of_centre = SurfaceSite {
            longitude_east_deg: centre.longitude_east_deg + 60.0,
            ..centre.clone()
        };
        let ev = earth.view_site(&east_of_centre);
        assert!(ev.visible);
        assert!(
            ev.sky_offset_rad.0 < 0.0,
            "east longitude lies toward sky west"
        );
        assert!(ev.sky_offset_rad.0.hypot(ev.sky_offset_rad.1) < earth.angular_semi_diameter);
        let (major, minor, _) = earth.apparent_ellipse;
        assert_abs_diff_eq!(major, earth.angular_semi_diameter, epsilon = 1e-9);
        assert!(minor <= major && minor > 0.99 * major);

        // Local illumination geometry agrees with starfield's helpers.
        let mars = Observer::BodyCenter(BodyId::Mars);
        system.with_ephemeris(|kernel| {
            let observer_bary = mars.barycentric(kernel, &epoch).unwrap();
            let astrometric = observer_bary
                .observe(BodyId::Earth.spice_name(), kernel, epoch.time())
                .unwrap();
            let upstream_phase = astrometric.phase_angle(kernel, epoch.time()).unwrap();
            assert_abs_diff_eq!(
                earth.illumination.phase_angle,
                upstream_phase,
                epsilon = 1e-6
            );
            let upstream_lit = astrometric
                .illuminated_fraction(kernel, epoch.time())
                .unwrap();
            assert_abs_diff_eq!(
                earth.illumination.illuminated_fraction,
                upstream_lit,
                epsilon = 1e-6
            );
            let upstream_limb = astrometric
                .bright_limb_position_angle(kernel, epoch.time())
                .unwrap();
            let dpa = (earth.illumination.bright_limb_position_angle - upstream_limb)
                .rem_euclid(std::f64::consts::TAU);
            let dpa = dpa.min(std::f64::consts::TAU - dpa);
            assert_abs_diff_eq!(dpa, 0.0, epsilon = 1e-6);
        });
    }

    /// A body whose body-fixed frame coincides with the sky frame: +x
    /// east, +y north, +z toward the observer.
    fn aligned_state(distance_au: f64) -> BodyState {
        BodyState {
            body: BodyId::Earth,
            direction: Equatorial::from_degrees(0.0, 0.0),
            distance_au,
            light_time_s: 0.0,
            angular_semi_diameter: (BodyId::Earth.equatorial_radius_km() / (distance_au * AU_KM))
                .asin(),
            illumination: IlluminationGeometry::from_barycentric(
                Vector3::zeros(),
                Vector3::new(distance_au, 0.0, 0.0),
                // Sun off to the sky-east side of the body.
                Vector3::new(distance_au, 1.0, 0.0),
            ),
            v_magnitude: None,
            sub_observer: SubPoint {
                lon_rad: 0.0,
                lat_rad: std::f64::consts::FRAC_PI_2,
                planetocentric: true,
            },
            sub_solar: None,
            north_pole_position_angle: 0.0,
            apparent_ellipse: (0.0, 0.0, 0.0),
            sky_to_body_fixed: Matrix3::identity(),
        }
    }

    #[test]
    fn site_at_the_sub_observer_point_sits_at_disk_centre_facing_us() {
        let state = aligned_state(1.0);
        // Body +z is toward the observer, so the north pole is the
        // sub-observer point in this aligned frame.
        let pole = SurfaceSite {
            name: "pole".into(),
            latitude_deg: 90.0,
            longitude_east_deg: 0.0,
            height_m: 0.0,
        };
        let view = state.view_site(&pole);
        assert_abs_diff_eq!(view.sky_offset_rad.0, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(view.sky_offset_rad.1, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(view.emission_cosine, 1.0, epsilon = 1e-12);
        assert!(view.visible);
    }

    #[test]
    fn equatorial_sites_land_on_the_limb_at_the_semi_diameter() {
        let state = aligned_state(1.0);
        // Body +x (lon 0, lat 0) is sky east: on the east limb, edge-on.
        let east_limb = SurfaceSite {
            name: "east".into(),
            latitude_deg: 0.0,
            longitude_east_deg: 0.0,
            height_m: 0.0,
        };
        let view = state.view_site(&east_limb);
        assert_relative_eq!(
            view.sky_offset_rad.0,
            state.angular_semi_diameter,
            max_relative = 1e-6
        );
        assert_abs_diff_eq!(view.sky_offset_rad.1, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(view.emission_cosine, 0.0, epsilon = 1e-12);
        assert!(
            !view.visible,
            "a limb point is not on the facing hemisphere"
        );
        // It faces the Sun, which sits to the sky east.
        assert!(view.incidence_cosine.unwrap() > 0.9);
        // The far side (−z) is hidden.
        let far = SurfaceSite {
            name: "far".into(),
            latitude_deg: -90.0,
            longitude_east_deg: 0.0,
            height_m: 0.0,
        };
        assert!(!state.view_site(&far).visible);
        assert_abs_diff_eq!(state.view_site(&far).emission_cosine, -1.0, epsilon = 1e-12);
        assert_eq!(state.view_site(&far).status(), SiteStatus::FarSide);
    }

    #[test]
    fn night_side_but_facing_is_visible_not_far_side() {
        let state = aligned_state(1.0);
        // Sun is to the sky east (+x). A facing-hemisphere site tilted
        // toward sky west (lon 180°, lat 45°) faces us but not the Sun.
        let dusk = SurfaceSite {
            name: "dusk".into(),
            latitude_deg: 45.0,
            longitude_east_deg: 180.0,
            height_m: 0.0,
        };
        let view = state.view_site(&dusk);
        assert!(view.visible);
        assert!(view.incidence_cosine.unwrap() < 0.0);
        assert_eq!(view.status(), SiteStatus::VisibleNight);
        // Its sky offset is toward the west: negative east component.
        assert!(view.sky_offset_rad.0 < 0.0);
        // Same latitude on the Sun-facing side is a day-side site.
        let dawn = SurfaceSite {
            longitude_east_deg: 0.0,
            ..dusk.clone()
        };
        assert_eq!(state.view_site(&dawn).status(), SiteStatus::VisibleDay);
    }

    #[test]
    fn site_height_and_flattening_enter_the_geodetic_position() {
        let radii = BodyId::Earth.radii_km();
        let sea_level = SurfaceSite {
            name: "equator".into(),
            latitude_deg: 0.0,
            longitude_east_deg: 0.0,
            height_m: 0.0,
        };
        assert_relative_eq!(
            sea_level.body_fixed_km(radii).norm(),
            radii[0],
            max_relative = 1e-12
        );
        let raised = SurfaceSite {
            height_m: 1000.0,
            ..sea_level.clone()
        };
        assert_relative_eq!(
            raised.body_fixed_km(radii).norm(),
            radii[0] + 1.0,
            max_relative = 1e-12
        );
        let pole = SurfaceSite {
            name: "pole".into(),
            latitude_deg: 90.0,
            longitude_east_deg: 0.0,
            height_m: 0.0,
        };
        assert_relative_eq!(
            pole.body_fixed_km(radii).norm(),
            radii[2],
            max_relative = 1e-9
        );
    }

    /// Transmit solve from Mars to Earth at the HiRISE epoch: forward
    /// light time matches the backward one, the aim leads the received
    /// direction by tens of arcseconds, and the aberration term alone is
    /// the transmitter speed over c. Needs the DE440s kernel.
    #[test]
    #[ignore]
    fn transmit_aimpoint_leads_the_received_direction() {
        let system = SolarSystem::new().unwrap();
        let epoch = Epoch::parse("2007-10-03T16:30:00Z").unwrap();
        let mars = Observer::BodyCenter(BodyId::Mars);
        let received = system.body_state(BodyId::Earth, &mars, &epoch).unwrap();
        let aim = system
            .transmit_aimpoint(BodyId::Earth, None, &mars, &epoch)
            .unwrap();

        // Forward and backward light times differ only by the target's
        // motion over the light time (v_E · lt / c ≈ 0.05 s), so they
        // agree to well under a second.
        assert_abs_diff_eq!(aim.light_time_s, received.light_time_s, epsilon = 0.5);
        assert_abs_diff_eq!(
            aim.arrival_epoch.jd_tdb(),
            epoch.jd_tdb() + aim.light_time_s / SECONDS_PER_DAY,
            epsilon = 1e-9
        );

        // Transmitter aberration is the transverse velocity over c:
        // |β × d̂|, at most v_Mars / c ≈ 16.5″.
        let beta = aim.transmitter_velocity_over_c;
        let speed_arcsec = beta.norm().to_degrees() * 3600.0;
        assert!((14.0..18.0).contains(&speed_arcsec), "v/c {speed_arcsec}");
        let (ra, dec) = (aim.geometric_direction.ra, aim.geometric_direction.dec);
        let d = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
        // Exact angle of d − β from d: tan θ = |β × d̂| / (1 − β · d̂).
        let expected_aberration = beta.cross(&d).norm().atan2(1.0 - beta.dot(&d));
        assert_relative_eq!(
            aim.aberration_angle,
            expected_aberration,
            max_relative = 1e-6
        );
        eprintln!(
            "residual transmit_aberration_arcsec={:.4} v_over_c_arcsec={speed_arcsec:.4} \
             aberration_minus_transverse_rad={:.3e}",
            aim.aberration_angle.to_degrees() * 3600.0,
            aim.aberration_angle - expected_aberration,
        );

        // Site rotation happens at the arrival epoch: Palomar's ICRF
        // position rotates by ω_E · lt · cos(lat) between emission and
        // arrival, ≈ 1.9° · cos 33° for a 475 s light time.
        let palomar = SurfaceSite {
            name: "Palomar".into(),
            latitude_deg: 33.356_667,
            longitude_east_deg: -116.8625,
            height_m: 1706.0,
        };
        let frame = system.frame_for(BodyId::Earth).unwrap();
        let body_fixed = palomar.body_fixed_km(BodyId::Earth.radii_km());
        let at_emission = frame.rotation_at(epoch.time()).transpose() * body_fixed;
        let at_arrival = frame.rotation_at(aim.arrival_epoch.time()).transpose() * body_fixed;
        let rotated = at_emission.angle(&at_arrival);
        let sidereal_rate = std::f64::consts::TAU / 86_164.090_5;
        // The position vector sweeps a circle at its geocentric latitude,
        // 0.18° below the geodetic one on the ellipsoid.
        let cos_geocentric_lat = body_fixed.xy().norm() / body_fixed.norm();
        let expected_rotation = sidereal_rate * aim.light_time_s * cos_geocentric_lat;
        eprintln!(
            "residual site_rotation_over_light_time_deg={:.5} expected={:.5}",
            rotated.to_degrees(),
            expected_rotation.to_degrees()
        );
        assert_relative_eq!(rotated, expected_rotation, max_relative = 1e-3);

        // Point-ahead between received apparent and transmit aim: the
        // relative velocity term, tens of arcseconds, never degrees.
        let point_ahead_arcsec = aim.point_ahead_from(&received.direction).to_degrees() * 3600.0;
        assert!(
            (10.0..80.0).contains(&point_ahead_arcsec),
            "{point_ahead_arcsec}"
        );

        // Aiming at Palomar rather than Earth's centre moves the aim by
        // no more than the apparent Earth radius.
        let site_aim = system
            .transmit_aimpoint(BodyId::Earth, Some(&palomar), &mars, &epoch)
            .unwrap();
        let shift = site_aim.aim_direction.angular_distance(&aim.aim_direction);
        assert!(
            shift <= received.angular_semi_diameter * 1.001,
            "site shift {shift}"
        );
        assert!(shift > 0.0);
        eprintln!(
            "residual point_ahead_arcsec={point_ahead_arcsec:.4} \
             site_shift_arcsec={:.4} earth_semi_diameter_arcsec={:.4}",
            shift.to_degrees() * 3600.0,
            received.angular_semi_diameter.to_degrees() * 3600.0
        );
    }

    /// The aberration sign, with no kernel: a transmitter moving along
    /// +x must aim behind the geometric direction (toward −x), and the
    /// receive correction `d′ + β` must undo it to first order.
    #[test]
    fn transmit_aim_lags_the_transmitter_velocity() {
        let d = Vector3::new(0.0, 1.0, 0.0);
        let beta = Vector3::new(1e-4, 0.0, 0.0);
        let aim = aim_before_aberration(d, beta);
        assert!(aim.x < 0.0, "aim {aim}");
        assert_abs_diff_eq!(aim.angle(&d), 1e-4, epsilon = 1e-12);
        let recovered = (aim + beta).normalize();
        assert_abs_diff_eq!(recovered.angle(&d), 0.0, epsilon = 1e-8);
        // Velocity along the line of sight changes nothing.
        let along = aim_before_aberration(d, Vector3::new(0.0, 1e-4, 0.0));
        assert_abs_diff_eq!(along.angle(&d), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn frames_resolve_from_the_embedded_table_without_kernels() {
        let constants = PlanetaryConstants::new();
        for body in BodyId::ALL {
            assert!(
                constants.frame_for(body.naif_id()).is_ok(),
                "{body} has no frame on a fresh PlanetaryConstants"
            );
        }
        assert_abs_diff_eq!(BodyId::Mars.equatorial_radius_km(), 3396.19, epsilon = 1e-9);
        assert_abs_diff_eq!(BodyId::Mars.polar_radius_km(), 3376.20, epsilon = 1e-9);
        assert_abs_diff_eq!(BodyId::Sun.equatorial_radius_km(), 695_700.0, epsilon = 1.0);
    }
}

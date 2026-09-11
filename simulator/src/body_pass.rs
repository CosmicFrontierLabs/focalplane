//! The solar-system second pass: bodies from an ephemeris, composited
//! onto the star field.
//!
//! [`BodyPass`] implements [`SecondPass`]. For every configured body it
//! asks the [`SolarSystem`] where the body is at the exposure's epoch,
//! projects that direction through the focal plane at the exposure's
//! middle orientation sample, rasterises a [`BodyStamp`] with the body's
//! reflectance law, and returns the stamps ordered far-to-near.
//!
//! # Solar irradiance
//!
//! Reflected-light radiometry needs the photo-electron rate the aperture
//! collects from the Sun at 1 AU through the sensor band. It comes from
//! the TSIS-1 reference spectrum ([`crate::photometry::solar`]), loaded
//! once per pass; a sensor band outside the archive coverage is a
//! compositing error, not a dark image.
//!
//! # Limits
//!
//! - Bodies are evaluated at one orientation sample per exposure (the
//!   middle one). Pointing jitter therefore blurs the stars but not the
//!   body; the body's own motion relative to the stars within an
//!   exposure is negligible.
//! - The Sun is not rendered; configuring it is an error.
//! - Disks are spheres of the equatorial radius; oblateness is not yet
//!   rasterised although the apparent ellipse is known.
//! - Surfaces are bound to the sensor band per composite call; the
//!   per-endmember band weights are a few thousand multiplies per body.

use std::sync::{Arc, OnceLock};

use nalgebra::{Matrix2, UnitQuaternion, Vector2};
use shared::units::{AngleExt, LengthExt};
use starfield::catalogs::StarData;
use starfield::Equatorial;

use crate::atmosphere::{AtmosphereRadiance, BoundRayleigh, RayleighAtmosphere};
use crate::bodies::brdf::Brdf;
use crate::bodies::surface::{SensorBand, SurfaceModel};
use crate::hardware::satellite::{FocalPlaneConfig, FocalPlaneProjector, SatelliteConfig};
use crate::image_proc::body_stamp::{BodyStamp, StampGeometry, DEFAULT_OVERSAMPLING};
use crate::image_proc::compose::{ComposeError, Composite, PassContext, SecondPass};
use crate::photometry::solar::TsisSolarSpectrum;
use crate::solar_system::{BodyId, BodyState, Observer, SolarSystem};

/// Spectral bins for the atmosphere's band integration.
const ATMOSPHERE_BINS: usize = 8;
/// Angular step used to measure the projection Jacobian, radians (≈2″).
const JACOBIAN_STEP_RAD: f64 = 1e-5;

/// One body to render and how its surface reflects.
#[derive(Clone)]
pub struct SceneBody {
    /// Which body.
    pub id: BodyId,
    /// Surface description, bound to each sensor band at render time.
    pub surface: Arc<dyn SurfaceModel>,
    /// Optional molecular atmosphere, bound to each sensor band at
    /// render time.
    pub atmosphere: Option<RayleighAtmosphere>,
}

impl SceneBody {
    /// A body with any surface model and no atmosphere.
    pub fn new(id: BodyId, surface: Arc<dyn SurfaceModel>) -> Self {
        Self {
            id,
            surface,
            atmosphere: None,
        }
    }

    /// A body whose whole surface follows one reflectance law.
    pub fn brdf<B: Brdf + Clone + std::fmt::Debug + 'static>(id: BodyId, law: B) -> Self {
        Self::new(id, Arc::new(law))
    }

    /// Wrap the body in a Rayleigh atmosphere.
    pub fn with_atmosphere(mut self, atmosphere: RayleighAtmosphere) -> Self {
        self.atmosphere = Some(atmosphere);
        self
    }
}

impl std::fmt::Debug for SceneBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SceneBody({}, {})", self.id, self.surface.label())
    }
}

/// Where body states come from.
#[derive(Debug)]
enum StateSource {
    /// Live ephemeris evaluation.
    Ephemeris {
        system: Arc<SolarSystem>,
        observer: Observer,
    },
    /// Pre-computed states, one per configured body, in the same order.
    /// For tests and synthetic scenes.
    Fixed(Vec<BodyState>),
}

/// Second pass rendering solar-system bodies.
#[derive(Debug)]
pub struct BodyPass {
    bodies: Vec<SceneBody>,
    source: StateSource,
    oversampling: usize,
    psf_blur: bool,
    solar: OnceLock<TsisSolarSpectrum>,
}

impl BodyPass {
    /// Bodies whose positions come from `system` as seen by `observer`.
    pub fn new(system: Arc<SolarSystem>, observer: Observer, bodies: Vec<SceneBody>) -> Self {
        Self {
            bodies,
            source: StateSource::Ephemeris { system, observer },
            oversampling: DEFAULT_OVERSAMPLING,
            psf_blur: true,
            solar: OnceLock::new(),
        }
    }

    /// Bodies at fixed, caller-supplied states (`states[i]` describes
    /// `bodies[i]`). The pass ignores the epoch.
    pub fn with_fixed_states(bodies: Vec<SceneBody>, states: Vec<BodyState>) -> Self {
        assert_eq!(bodies.len(), states.len(), "one state per body");
        Self {
            bodies,
            source: StateSource::Fixed(states),
            oversampling: DEFAULT_OVERSAMPLING,
            psf_blur: true,
            solar: OnceLock::new(),
        }
    }

    /// Sub-samples per pixel edge used when rasterising disks.
    pub fn with_oversampling(mut self, oversampling: usize) -> Self {
        self.oversampling = oversampling.max(2);
        self
    }

    /// Whether body stamps are blurred with the sensor PSF. Off renders
    /// the geometric disk at sub-sample resolution, for isolating the
    /// PSF's effect on the limb; stars keep their PSF regardless.
    pub fn with_psf_blur(mut self, psf_blur: bool) -> Self {
        self.psf_blur = psf_blur;
        self
    }

    /// The TSIS-1 solar spectrum, loaded on first use.
    fn solar(&self) -> Result<&TsisSolarSpectrum, ComposeError> {
        if let Some(sun) = self.solar.get() {
            return Ok(sun);
        }
        let sun = TsisSolarSpectrum::load().map_err(|e| ComposeError::Failed(e.to_string()))?;
        let _ = self.solar.set(sun);
        Ok(self.solar.get().expect("just set"))
    }

    /// Photo-electron rate (e⁻/s) the aperture collects from the Sun at
    /// 1 AU through the sensor band. Errors if the band leaves the TSIS-1
    /// coverage.
    pub fn solar_electron_rate(&self, satellite: &SatelliteConfig) -> Result<f64, ComposeError> {
        self.solar()?
            .photo_electron_rate(
                &satellite.combined_qe,
                satellite.telescope.clear_aperture_area(),
            )
            .map_err(|e| ComposeError::Failed(e.to_string()))
    }

    fn states(&self, ctx: &PassContext<'_>) -> Result<Vec<BodyState>, ComposeError> {
        match &self.source {
            StateSource::Fixed(states) => Ok(states.clone()),
            StateSource::Ephemeris { system, observer } => {
                let epoch = ctx.epoch.as_ref().ok_or(ComposeError::MissingEpoch)?;
                let sample = middle_sample(ctx)?;
                let at = epoch.offset(sample.offset);
                self.bodies
                    .iter()
                    .map(|b| {
                        system
                            .body_state(b.id, observer, &at)
                            .map_err(|e| ComposeError::Failed(e.to_string()))
                    })
                    .collect()
            }
        }
    }
}

/// The orientation sample nearest the middle of the exposure.
fn middle_sample<'a>(
    ctx: &'a PassContext<'_>,
) -> Result<&'a crate::image_proc::compose::OrientationSample, ComposeError> {
    ctx.samples
        .get(ctx.samples.len() / 2)
        .ok_or_else(|| ComposeError::Failed("no orientation samples".into()))
}

/// Project a sky direction onto `sensor_idx`, allowing `padding_mm`
/// of overhang.
fn project(
    fp: &FocalPlaneConfig,
    direction: &Equatorial,
    orientation: &UnitQuaternion<f64>,
    sensor_idx: usize,
    padding_mm: f64,
) -> Option<(f64, f64)> {
    let probe = StarData::with_position(0, *direction, 0.0, None);
    fp.project_to_sensor(&probe, orientation, sensor_idx, padding_mm)
}

/// Pixel position of `direction` plus the local Jacobian
/// `d(px) / d(east, north)` measured by finite differences.
fn project_with_jacobian(
    fp: &FocalPlaneConfig,
    direction: &Equatorial,
    orientation: &UnitQuaternion<f64>,
    sensor_idx: usize,
    padding_mm: f64,
) -> Option<((f64, f64), Matrix2<f64>)> {
    let centre = project(fp, direction, orientation, sensor_idx, padding_mm)?;
    let cos_dec = direction.dec.cos().max(1e-9);
    let east = Equatorial::new(direction.ra + JACOBIAN_STEP_RAD / cos_dec, direction.dec);
    let north = Equatorial::new(direction.ra, direction.dec + JACOBIAN_STEP_RAD);
    // Generous padding: the offset points sit ~2″ away and must project.
    let east_px = project(fp, &east, orientation, sensor_idx, padding_mm + 1.0)?;
    let north_px = project(fp, &north, orientation, sensor_idx, padding_mm + 1.0)?;
    let de = Vector2::new(east_px.0 - centre.0, east_px.1 - centre.1) / JACOBIAN_STEP_RAD;
    let dn = Vector2::new(north_px.0 - centre.0, north_px.1 - centre.1) / JACOBIAN_STEP_RAD;
    Some((centre, Matrix2::from_columns(&[de, dn])))
}

impl SecondPass for BodyPass {
    fn composite(&self, ctx: &PassContext<'_>) -> Result<Composite, ComposeError> {
        if let Some(sun) = self.bodies.iter().find(|b| b.id.is_sun()) {
            return Err(ComposeError::Failed(format!(
                "{} is not renderable yet: self-luminous bodies are not implemented",
                sun.id
            )));
        }
        let states = self.states(ctx)?;
        let orientation = middle_sample(ctx)?.orientation;
        let satellite = ctx.satellite;
        let psf = if self.psf_blur {
            satellite.airy_disk_pixel_space()
        } else {
            // A PSF far narrower than a pixel: the sampled kernel collapses
            // to a single central weight.
            shared::image_proc::airy::PixelScaledAiryDisk::with_fwhm(
                1e-3,
                satellite.telescope.corrected_to,
            )
        };
        let pixel_mm = satellite.sensor.pixel_size().as_millimeters();
        let exposure_s = ctx.exposure.as_secs_f64();
        let solar_rate = self.solar_electron_rate(satellite)?;
        let band = SensorBand {
            qe: &satellite.combined_qe,
            solar: self.solar()?,
        };

        let mut layers: Vec<(f64, BodyStamp)> = Vec::new();
        for (body, state) in self.bodies.iter().zip(states.iter()) {
            let surface = body
                .surface
                .bind(&band)
                .map_err(|e| ComposeError::Failed(format!("{}: {e}", body.id)))?;
            let rad_per_px = ctx
                .focal_plane
                .plate_scale_rad_per_px(ctx.sensor_idx)
                .unwrap_or(satellite.plate_scale_per_pixel().as_radians());
            let radius_px = state.angular_semi_diameter / rad_per_px;
            let padding_mm = (radius_px + 2.0 * psf.first_zero() + 2.0) * pixel_mm;
            let Some((centre, jacobian)) = project_with_jacobian(
                ctx.focal_plane,
                &state.direction,
                &orientation,
                ctx.sensor_idx,
                padding_mm,
            ) else {
                continue;
            };
            let illumination = &state.illumination;
            let sun_direction_sky = sun_in_sky_frame(state);
            let geometry = StampGeometry {
                center_px: centre,
                jacobian,
                semi_diameter_rad: state.angular_semi_diameter,
                sun_direction_sky,
                phase_angle: illumination.phase_angle,
                electrons_per_sr: exposure_s * solar_rate
                    / illumination.heliocentric_distance_au.powi(2),
                sky_to_body_fixed: state.sky_to_body_fixed,
                radius_km: body.id.equatorial_radius_km(),
            };
            let bound_air = body
                .atmosphere
                .as_ref()
                .map(|atm| {
                    BoundRayleigh::bind(
                        atm.clone(),
                        &satellite.combined_qe,
                        self.solar()?,
                        ATMOSPHERE_BINS,
                    )
                    .map_err(|e| ComposeError::Failed(format!("{}: {e}", body.id)))
                })
                .transpose()?;
            let air: Option<&dyn AtmosphereRadiance> =
                bound_air.as_ref().map(|b| b as &dyn AtmosphereRadiance);
            let stamp = BodyStamp::build(&geometry, surface.as_ref(), air, &psf, self.oversampling);
            let no_data = surface.no_data_samples();
            if no_data > 0 {
                log::warn!(
                    "{}: {no_data} sub-samples fell on unmapped terrain and rendered dark",
                    body.id
                );
            }
            layers.push((state.distance_au, stamp));
        }

        // Far to near so nearer bodies mask farther ones.
        layers.sort_by(|a, b| b.0.total_cmp(&a.0));
        Ok(Composite {
            layers: layers
                .into_iter()
                .map(|(_, stamp)| stamp.into_composite(ctx.roi_origin))
                .collect(),
        })
    }
}

/// Sun direction in the body's `(east, north, toward observer)` frame.
fn sun_in_sky_frame(state: &BodyState) -> nalgebra::Vector3<f64> {
    let ill = &state.illumination;
    let toward_observer = ill.observer_direction;
    let line_of_sight = -toward_observer;
    let pole = nalgebra::Vector3::z();
    let mut east = pole.cross(&line_of_sight);
    if east.norm() < 1e-12 {
        east = nalgebra::Vector3::x();
    }
    let east = east.normalize();
    let north = line_of_sight.cross(&east).normalize();
    nalgebra::Vector3::new(
        ill.sun_direction.dot(&east),
        ill.sun_direction.dot(&north),
        ill.sun_direction.dot(&toward_observer),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bodies::brdf::{disk_integrated_reflectance, Lambert};
    use crate::hardware::sensor::models::IMX455;
    use crate::hardware::telescope::models::SMALL_50MM;
    use crate::image_proc::render::SecondPassBinding;
    use crate::photometry::zodiacal::SolarAngularCoordinates;
    use crate::scene::Scene;
    use crate::sims::orientation::orientation_from_pointing;
    use crate::solar_system::{IlluminationGeometry, AU_KM};
    use approx::assert_relative_eq;
    use nalgebra::Vector3;
    use shared::units::{Temperature, TemperatureExt};
    use starfield::planetarylib::subpoint::SubPoint;
    use std::time::Duration;

    fn satellite() -> SatelliteConfig {
        SatelliteConfig::new(
            SMALL_50MM.clone(),
            IMX455.with_dimensions(256, 256),
            Temperature::from_celsius(-10.0),
        )
    }

    /// A Lambert "Earth" at 1 AU, 60° phase, placed exactly at the
    /// pointing direction.
    fn fixed_earth(direction: Equatorial) -> BodyState {
        let observer = Vector3::new(1.5, 0.0, 0.0);
        let body = Vector3::new(0.5, 0.0, 0.0) + Vector3::new(0.0, 0.0, 0.0);
        // Sun placed so the phase angle is 60°: at the body, observer
        // direction is +x; put the Sun at 60° from it in the x–y plane.
        let sun = body
            + Vector3::new(
                60.0_f64.to_radians().cos(),
                60.0_f64.to_radians().sin(),
                0.0,
            );
        BodyState {
            body: BodyId::Earth,
            direction,
            distance_au: 1.0,
            light_time_s: 499.0,
            angular_semi_diameter: BodyId::Earth.equatorial_radius_km() / AU_KM,
            illumination: IlluminationGeometry::from_barycentric(observer, body, sun),
            v_magnitude: None,
            sub_observer: SubPoint {
                lon_rad: 0.0,
                lat_rad: 0.0,
                planetocentric: true,
            },
            sub_solar: None,
            north_pole_position_angle: 0.0,
            apparent_ellipse: (0.0, 0.0, 0.0),
            sky_to_body_fixed: nalgebra::Matrix3::identity(),
        }
    }

    #[test]
    fn solar_electron_rate_is_enormous_and_finite() {
        let pass = BodyPass::with_fixed_states(Vec::new(), Vec::new());
        let rate = pass.solar_electron_rate(&satellite()).unwrap();
        // 1361 W/m² at ~3.6×10⁻¹⁹ J per visible photon is ~4×10²¹
        // photons/s/m²; a 5 cm aperture (2×10⁻³ m²) at ~50 % band
        // efficiency collects a few ×10¹⁸ electrons/s.
        assert!(rate > 3e17 && rate < 1e19, "rate = {rate:e}");
    }

    #[test]
    fn fixed_body_renders_with_expected_total_electrons() {
        let sat = satellite();
        let fp = FocalPlaneConfig::from_satellite(&sat);
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let state = fixed_earth(pointing);
        let albedo = 0.3;
        let pass = BodyPass::with_fixed_states(
            vec![SceneBody::brdf(BodyId::Earth, Lambert { albedo })],
            vec![state.clone()],
        );

        let scene = Scene::from_catalog(
            fp,
            Vec::new(),
            pointing,
            SolarAngularCoordinates::zodiacal_minimum(),
        )
        .with_second_pass(Arc::new(pass), None);

        let exposure = Duration::from_millis(10);
        let renderer = crate::image_proc::render::Renderer::from_stars(
            &[],
            scene.focal_plane.satellite_for_sensor(0).unwrap(),
        )
        .with_second_pass(SecondPassBinding {
            pass: scene.second_pass.clone().unwrap(),
            orientation: orientation_from_pointing(&pointing, 0.0),
            epoch: None,
            sensor_idx: 0,
            focal_plane: scene.focal_plane.clone(),
        });
        let result = renderer.render_with_options(
            &exposure,
            &SolarAngularCoordinates::zodiacal_minimum(),
            false,
            Some(1),
        );

        let solar_rate = BodyPass::with_fixed_states(Vec::new(), Vec::new())
            .solar_electron_rate(&sat)
            .unwrap();
        let expected = exposure.as_secs_f64() * solar_rate
            / state.illumination.heliocentric_distance_au.powi(2)
            * state.angular_semi_diameter.powi(2)
            * disk_integrated_reflectance(&Lambert { albedo }, state.illumination.phase_angle, 300);
        let total = result.star_image.sum();
        assert_relative_eq!(total, expected, max_relative = 0.02);
        // Zodiacal light is hidden where the body is.
        let (w, h) = sat.sensor.dimensions.get_pixel_width_height();
        let centre = result.zodiacal_image[[h / 2, w / 2]];
        let corner = result.zodiacal_image[[2, 2]];
        assert!(centre < 1e-6 * corner, "centre {centre} corner {corner}");
    }

    #[test]
    fn sun_is_rejected() {
        let sat = satellite();
        let fp = FocalPlaneConfig::from_satellite(&sat);
        let pass = BodyPass::with_fixed_states(
            vec![SceneBody::brdf(BodyId::Sun, Lambert { albedo: 1.0 })],
            vec![fixed_earth(Equatorial::from_degrees(0.0, 0.0))],
        );
        let samples = [crate::image_proc::compose::OrientationSample {
            offset: Duration::ZERO,
            orientation: UnitQuaternion::identity(),
        }];
        let ctx = PassContext {
            focal_plane: &fp,
            satellite: &sat,
            sensor_idx: 0,
            roi_origin: (0, 0),
            roi_size: (256, 256),
            exposure: Duration::from_millis(10),
            epoch: None,
            samples: &samples,
        };
        assert!(matches!(pass.composite(&ctx), Err(ComposeError::Failed(_))));
    }

    /// Earth and Moon from Mars on the HiRISE date, rendered end to end,
    /// plus the epoch requirement of the ephemeris source. Needs the
    /// DE440s kernel (network on first run).
    #[test]
    #[ignore]
    fn earth_and_moon_from_mars_render() {
        let sat = satellite();
        let fp = FocalPlaneConfig::from_satellite(&sat);
        let system = Arc::new(SolarSystem::new().unwrap());
        let epoch = crate::epoch::Epoch::parse("2007-10-03T05:30:00Z").unwrap();
        let observer = Observer::BodyCenter(BodyId::Mars);
        let earth = system.body_state(BodyId::Earth, &observer, &epoch).unwrap();

        let no_epoch = BodyPass::new(Arc::clone(&system), observer.clone(), Vec::new());
        let samples = [crate::image_proc::compose::OrientationSample {
            offset: Duration::ZERO,
            orientation: UnitQuaternion::identity(),
        }];
        let ctx = PassContext {
            focal_plane: &fp,
            satellite: &sat,
            sensor_idx: 0,
            roi_origin: (0, 0),
            roi_size: (256, 256),
            exposure: Duration::from_millis(10),
            epoch: None,
            samples: &samples,
        };
        assert!(matches!(
            no_epoch.composite(&ctx),
            Err(ComposeError::MissingEpoch)
        ));

        let pass = BodyPass::new(
            system,
            observer,
            vec![
                SceneBody::new(
                    BodyId::Earth,
                    Arc::new(crate::bodies::surface::TexturedSurfaceModel::new(
                        Arc::new(starfield_planet_maps::earth_tier().unwrap()),
                        Arc::new(Lambert { albedo: 1.0 }),
                        Arc::new(
                            starfield_reflectance_library::ReflectanceLibrary::load_embedded()
                                .unwrap(),
                        ),
                        "Earth MCD12C1 0.25°",
                    )),
                ),
                SceneBody::brdf(BodyId::Moon, crate::bodies::brdf::Hapke::lunar_average()),
            ],
        );
        let scene = Scene::from_catalog(
            fp,
            Vec::new(),
            earth.direction,
            SolarAngularCoordinates::zodiacal_minimum(),
        )
        .with_second_pass(Arc::new(pass), Some(epoch));
        let result = scene
            .render_with_seed(&Duration::from_millis(1), Some(1))
            .remove(0);
        let total = result.star_image.sum();
        assert!(total.is_finite() && total > 0.0, "total {total}");
        eprintln!("Earth+Moon from Mars, 1 ms on 5 cm: {total:.3e} e⁻");
    }
}

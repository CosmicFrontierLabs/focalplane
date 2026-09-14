//! Provenance record written beside every rendered solar-system frame.
//!
//! One [`FrameMetadata`] per image: what was rendered (epoch, observer,
//! pointing, instrument), with which models and which ones were left
//! out, the radiometric totals, and the geometry of every body, surface
//! site and minor planet in the frame. It is plain serde data, so the
//! same struct reads a sidecar back for indexing or comparison.
//!
//! Field names are the JSON keys; downstream tooling indexes on them,
//! so renaming one is a format change.

use serde::{Deserialize, Serialize};

use super::minor_planets::MinorPlanetSighting;
use super::{BodyState, SiteView, SurfaceSite};

/// Everything written to `<out>.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrameMetadata {
    pub generator: Generator,
    pub epoch: EpochRecord,
    /// Debug rendering of the [`super::Observer`].
    pub observer: String,
    pub pointing_icrf: Pointing,
    pub instrument: Instrument,
    pub models: Models,
    pub radiometry: Radiometry,
    pub bodies: Vec<BodyRecord>,
    pub sites: Vec<SiteRecord>,
    pub minor_planets: Vec<MinorPlanetRecord>,
    pub outputs: Outputs,
}

/// Which program, at which source state, with which arguments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Generator {
    #[serde(rename = "crate")]
    pub crate_name: String,
    pub version: String,
    pub binary: String,
    /// `git rev-parse HEAD` of the source tree the binary was built
    /// from, when available.
    pub git_commit: Option<String>,
    pub command_line: Vec<String>,
}

/// Exposure start.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpochRecord {
    pub utc: String,
    pub jd_tdb: f64,
}

/// Boresight in the ICRF.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pointing {
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub roll_deg: f64,
    /// Body the boresight is on.
    pub target: String,
}

/// Telescope, sensor and exposure settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Instrument {
    pub telescope: String,
    pub aperture_m: f64,
    pub focal_length_m: f64,
    pub obscuration_ratio_linear: f64,
    pub clear_aperture_cm2: f64,
    pub sensor: String,
    pub pixel_um: f64,
    pub plate_scale_arcsec_per_px: f64,
    pub window_px: usize,
    pub exposure_s: f64,
    pub temperature_c: f64,
    pub full_well_e: f64,
    pub dark_current_e_per_s_per_px: f64,
    pub psf_model: String,
    pub psf_fwhm_px: f64,
    pub image_parity: String,
}

/// What was modelled and what was not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Models {
    pub bodies: Vec<String>,
    pub textured: bool,
    pub earth_atmosphere: Option<AtmosphereModel>,
    pub noise: String,
    pub stray_light: String,
    pub ocean_glint: String,
    pub clouds: String,
    pub body_motion_blur: String,
    pub oversampling_per_pixel_edge: usize,
    pub stars: Option<StarsModel>,
    pub minor_planets: Option<MinorPlanetsModel>,
}

/// Atmosphere parameters and the effects it leaves out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AtmosphereModel {
    #[serde(rename = "type")]
    pub kind: String,
    pub scale_height_km: f64,
    pub top_km: f64,
    pub tau_vertical_550nm: f64,
    pub spectral_bins: usize,
    pub multiple_scattering: bool,
    pub aerosols: bool,
    pub ozone: bool,
    pub clouds: bool,
    pub refraction: bool,
}

/// Background star catalogue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StarsModel {
    pub catalog: String,
    pub mag_limit_g: f64,
    pub count_in_cone: usize,
}

/// Minor-planet catalogue and how it was used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinorPlanetsModel {
    pub catalog: String,
    pub catalog_file: String,
    pub catalog_bodies: usize,
    pub elements_epoch_tt_jd_range: Option<(f64, f64)>,
    pub mag_limit_v: f64,
    pub count_in_cone: usize,
    pub propagation: String,
    pub photometry: String,
    pub not_modelled: Vec<String>,
}

/// Radiometric chain and scene totals (pre-clipping means).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Radiometry {
    pub solar_spectrum: String,
    pub solar_electron_rate_1au_e_per_s: f64,
    pub chain: String,
    pub scene_electrons_total: f64,
    pub scene_electrons_peak_per_px: f64,
    pub pixels_at_full_well: usize,
    pub zodiacal_elongation_used_deg: f64,
}

/// One resolved body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyRecord {
    pub body: String,
    pub naif_id: i32,
    pub apparent_ra_deg: f64,
    pub apparent_dec_deg: f64,
    pub range_au: f64,
    pub range_km: f64,
    pub light_time_s: f64,
    pub heliocentric_distance_au: f64,
    pub angular_diameter_arcsec: f64,
    pub phase_angle_deg: f64,
    pub illuminated_fraction: f64,
    pub bright_limb_position_angle_deg: f64,
    pub north_pole_position_angle_deg: f64,
    pub solar_elongation_deg: f64,
    pub v_magnitude_mallama_hilton_2018: Option<f64>,
    pub sub_observer_lon_east_deg: f64,
    pub sub_observer_lat_planetocentric_deg: f64,
    pub sub_solar_lon_east_deg: Option<f64>,
    pub sub_solar_lat_planetocentric_deg: Option<f64>,
    pub equatorial_radius_km: f64,
    pub polar_radius_km: f64,
    /// `exposure × solar electron rate at 1 AU / d☉²`: electrons a
    /// steradian of unit-reflectance surface would give.
    pub electrons_per_sr_per_unit_reflectance: f64,
}

impl BodyRecord {
    /// Record for `state` seen at `solar_elongation_deg`, with
    /// `electrons_per_sr_per_unit_reflectance` already computed by the
    /// renderer.
    pub fn from_state(
        state: &BodyState,
        solar_elongation_deg: f64,
        electrons_per_sr_per_unit_reflectance: f64,
    ) -> Self {
        let ill = &state.illumination;
        Self {
            body: state.body.name().to_string(),
            naif_id: state.body.naif_id(),
            apparent_ra_deg: state.direction.ra_degrees(),
            apparent_dec_deg: state.direction.dec_degrees(),
            range_au: state.distance_au,
            range_km: state.distance_km(),
            light_time_s: state.light_time_s,
            heliocentric_distance_au: ill.heliocentric_distance_au,
            angular_diameter_arcsec: state.angular_diameter_arcsec(),
            phase_angle_deg: ill.phase_angle.to_degrees(),
            illuminated_fraction: ill.illuminated_fraction,
            bright_limb_position_angle_deg: ill.bright_limb_position_angle.to_degrees(),
            north_pole_position_angle_deg: state.north_pole_position_angle.to_degrees(),
            solar_elongation_deg,
            v_magnitude_mallama_hilton_2018: state.v_magnitude,
            sub_observer_lon_east_deg: state.sub_observer.lon_rad.to_degrees(),
            sub_observer_lat_planetocentric_deg: state.sub_observer.lat_rad.to_degrees(),
            sub_solar_lon_east_deg: state.sub_solar.map(|p| p.lon_rad.to_degrees()),
            sub_solar_lat_planetocentric_deg: state.sub_solar.map(|p| p.lat_rad.to_degrees()),
            equatorial_radius_km: state.body.equatorial_radius_km(),
            polar_radius_km: state.body.polar_radius_km(),
            electrons_per_sr_per_unit_reflectance,
        }
    }
}

/// What a site marker means, stated in every record so the JSON cannot
/// be misread as a transmit aimpoint.
pub const SITE_MEANING: &str = "apparent imaged site at the light-time-corrected epoch; \
    not a transmit point-ahead or arrival-time aimpoint; visible means on the facing \
    hemisphere, not link availability";

/// One surface site projected onto the frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteRecord {
    pub name: String,
    pub qualifier: String,
    pub latitude_geodetic_deg: f64,
    pub longitude_east_deg: f64,
    pub height_m: f64,
    pub status: String,
    pub visible_geometric: bool,
    pub emission_cosine: f64,
    pub incidence_cosine: Option<f64>,
    pub sky_offset_east_arcsec: f64,
    pub sky_offset_north_arcsec: f64,
    pub pixel_x: Option<f64>,
    pub pixel_y: Option<f64>,
    pub meaning: String,
}

impl SiteRecord {
    /// Record for `site` as seen in `view`, at `pixel` if on the sensor.
    pub fn new(
        site: &SurfaceSite,
        view: &SiteView,
        qualifier: &str,
        pixel: Option<(f64, f64)>,
    ) -> Self {
        Self {
            name: site.name.clone(),
            qualifier: qualifier.to_string(),
            latitude_geodetic_deg: site.latitude_deg,
            longitude_east_deg: site.longitude_east_deg,
            height_m: site.height_m,
            status: view.status().to_string(),
            visible_geometric: view.visible,
            emission_cosine: view.emission_cosine,
            incidence_cosine: view.incidence_cosine,
            sky_offset_east_arcsec: view.sky_offset_rad.0.to_degrees() * 3600.0,
            sky_offset_north_arcsec: view.sky_offset_rad.1.to_degrees() * 3600.0,
            pixel_x: pixel.map(|p| p.0),
            pixel_y: pixel.map(|p| p.1),
            meaning: SITE_MEANING.to_string(),
        }
    }
}

/// One minor planet in the frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinorPlanetRecord {
    pub designation: String,
    pub name: String,
    pub apparent_ra_deg: f64,
    pub apparent_dec_deg: f64,
    pub astrometric_ra_deg: f64,
    pub astrometric_dec_deg: f64,
    pub pixel_x: Option<f64>,
    pub pixel_y: Option<f64>,
    pub v_magnitude: f64,
    pub h: f64,
    pub g: f64,
    pub range_au: f64,
    pub heliocentric_au: f64,
    pub light_time_s: f64,
    pub phase_angle_deg: f64,
    pub sky_motion_arcsec_per_hour: f64,
    pub elements_epoch_tt_jd: f64,
}

impl MinorPlanetRecord {
    /// Record for a sighting, at `pixel` if on the sensor.
    pub fn new(sighting: &MinorPlanetSighting, pixel: Option<(f64, f64)>) -> Self {
        Self {
            designation: sighting.designation.clone(),
            name: sighting.name.clone(),
            apparent_ra_deg: sighting.direction.ra_degrees(),
            apparent_dec_deg: sighting.direction.dec_degrees(),
            astrometric_ra_deg: sighting.astrometric_direction.ra_degrees(),
            astrometric_dec_deg: sighting.astrometric_direction.dec_degrees(),
            pixel_x: pixel.map(|p| p.0),
            pixel_y: pixel.map(|p| p.1),
            v_magnitude: sighting.v_magnitude,
            h: sighting.h,
            g: sighting.g,
            range_au: sighting.range_au,
            heliocentric_au: sighting.heliocentric_au,
            light_time_s: sighting.light_time_s,
            phase_angle_deg: sighting.phase_angle.to_degrees(),
            sky_motion_arcsec_per_hour: sighting.sky_motion_arcsec_per_hour,
            elements_epoch_tt_jd: sighting.elements_epoch_tt,
        }
    }
}

/// Files written and how to read them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outputs {
    pub image_16bit_dn: String,
    pub preview_8bit: String,
    pub preview_stretch: String,
    pub dn_per_electron: f64,
    pub black_level_dn: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json_with_stable_keys() {
        let meta = FrameMetadata {
            generator: Generator {
                crate_name: "focalplane simulator".into(),
                version: "0.2.0".into(),
                binary: "planet_view".into(),
                git_commit: None,
                command_line: vec!["planet_view".into()],
            },
            epoch: EpochRecord {
                utc: "2027-06-01T23:00:00.000Z".into(),
                jd_tdb: 2_461_558.458_3,
            },
            observer: "BodyCenter(Mars)".into(),
            pointing_icrf: Pointing {
                ra_deg: 1.0,
                dec_deg: 2.0,
                roll_deg: 0.0,
                target: "Earth".into(),
            },
            instrument: Instrument {
                telescope: "JBT".into(),
                aperture_m: 0.485,
                focal_length_m: 5.987,
                obscuration_ratio_linear: 0.35,
                clear_aperture_cm2: 1621.0,
                sensor: "IMX455".into(),
                pixel_um: 3.76,
                plate_scale_arcsec_per_px: 0.1295,
                window_px: 512,
                exposure_s: 0.001,
                temperature_c: -10.0,
                full_well_e: 26000.0,
                dark_current_e_per_s_per_px: 0.0046,
                psf_model: "gaussian".into(),
                psf_fwhm_px: 1.85,
                image_parity: "north up, east left".into(),
            },
            models: Models {
                bodies: vec!["Earth".into()],
                textured: true,
                earth_atmosphere: None,
                noise: "none".into(),
                stray_light: "not modelled".into(),
                ocean_glint: "not modelled".into(),
                clouds: "not modelled".into(),
                body_motion_blur: "not modelled".into(),
                oversampling_per_pixel_edge: 4,
                stars: None,
                minor_planets: None,
            },
            radiometry: Radiometry {
                solar_spectrum: "TSIS-1".into(),
                solar_electron_rate_1au_e_per_s: 1.6e20,
                chain: "chain".into(),
                scene_electrons_total: 1.0,
                scene_electrons_peak_per_px: 1.0,
                pixels_at_full_well: 0,
                zodiacal_elongation_used_deg: 40.0,
            },
            bodies: Vec::new(),
            sites: Vec::new(),
            minor_planets: Vec::new(),
            outputs: Outputs {
                image_16bit_dn: "a.png".into(),
                preview_8bit: "a_preview.png".into(),
                preview_stretch: "asinh".into(),
                dn_per_electron: 2.5,
                black_level_dn: 0,
            },
        };
        let text = serde_json::to_string_pretty(&meta).unwrap();
        assert!(text.contains("\"crate\": \"focalplane simulator\""));
        assert!(text.contains("\"earth_atmosphere\": null"));
        let back: FrameMetadata = serde_json::from_str(&text).unwrap();
        assert_eq!(back, meta);
    }
}

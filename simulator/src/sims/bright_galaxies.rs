//! Bright-galaxy supplement → `GalaxyInFrame` routing.
//!
//! Loads the embedded `starfield-bright-galaxies` supplement (~45
//! naked-eye / wide-FOV galaxies — M31, M33, the Magellanic Clouds,
//! M51, M81/82, M101, etc. — that NSA explicitly excludes), filters
//! to the field of view around a pointing, builds per-galaxy Sersic
//! deposits + blackbody-approximated flux objects, and routes them to
//! the per-sensor `GalaxyInFrame` lists that `Scene::with_galaxies`
//! consumes.
//!
//! Differences vs. the NSA loader (`sims::nsa_galaxies`):
//!
//! - **Catalog is embedded** — no FITS file, no download. One call to
//!   `BrightGalaxyCatalog::load_embedded()` returns the whole
//!   supplement (small enough to ignore as a perf concern).
//! - **Spectrum is approximated.** Bright-galaxies entries carry only
//!   integrated `mag_v`, not per-band SDSS fluxes. We approximate the
//!   spectrum with a galaxy-typical `B-V = 0.85` blackbody scaled to
//!   `mag_v` (interpreted as Gaia G — within a tenth of a mag for
//!   spiral galaxies, fine for visualisation).
//! - **In-cone filter is centre + extent.** Several entries (M31,
//!   LMC, M33, …) span degrees on the sky, so a cone whose centre
//!   sits outside the galaxy can still have its outer envelope reach
//!   in. `BrightGalaxyCatalog::in_cone_extended` admits a galaxy when
//!   its envelope, truncated at the renderer's surface-brightness
//!   fraction, overlaps the field.

use log::info;
use starfield::catalogs::bright_galaxies::{BrightGalaxy, BrightGalaxyCatalog};
use starfield::catalogs::gaia::Cone;
use starfield::catalogs::ExtendedSource;
use starfield::framelib::attitude::attitude_from_pointing;
use starfield::Equatorial;

use crate::hardware::satellite::{FocalPlaneConfig, FocalPlaneProjector};
use crate::image_proc::sersic_splat::{SersicSplat, TRUNCATION_SB_FRACTION};
use crate::photometry::photoconversion::{photon_electron_fluxes, SourceFlux};
use crate::photometry::BlackbodyStellarSpectrum;
use crate::scene_galaxy::GalaxyInFrame;
use crate::sims::nsa_galaxies::GalaxyInField;

/// B-V proxy used for the blackbody spectrum approximation. 0.85 is a
/// rough integrated colour for an Sb spiral; ellipticals are ~1.0,
/// late-type spirals are ~0.6. The choice doesn't affect the centroid
/// or spatial extent — only the per-band electron rate split.
const DEFAULT_BV: f64 = 0.85;

/// Configuration for the bright-galaxies loader.
#[derive(Debug, Clone)]
pub struct BrightGalaxyLoaderConfig {
    /// Surface-brightness fraction of `I_e` at which each galaxy's
    /// envelope is truncated for the field-of-view overlap test.
    /// Defaults to the renderer's own truncation, so every galaxy whose
    /// rendered footprint can reach the field is admitted.
    pub sb_fraction: f64,
}

impl Default for BrightGalaxyLoaderConfig {
    fn default() -> Self {
        Self {
            sb_fraction: TRUNCATION_SB_FRACTION,
        }
    }
}

/// Galaxies whose truncated envelope overlaps the `fov_radius_deg` cone
/// about `pointing`, in catalog-name order.
fn galaxies_in_field<'a>(
    cat: &'a BrightGalaxyCatalog,
    pointing: &Equatorial,
    fov_radius_deg: f64,
    config: &BrightGalaxyLoaderConfig,
) -> Vec<&'a BrightGalaxy> {
    let cone = Cone::from_degrees(
        pointing.ra_degrees(),
        pointing.dec_degrees(),
        fov_radius_deg,
    );
    let mut in_field = cat.in_cone_extended(&cone, config.sb_fraction);
    in_field.sort_by(|a, b| a.name.cmp(&b.name));
    in_field
}

/// Load the embedded supplement, filter to the FOV around `pointing`,
/// and project to per-sensor `GalaxyInFrame` lists. Mirrors
/// [`crate::sims::nsa_galaxies::load_and_route_nsa_galaxies`].
pub fn load_and_route_bright_galaxies(
    pointing: &Equatorial,
    fp: &FocalPlaneConfig,
    fov_radius_deg: f64,
    config: &BrightGalaxyLoaderConfig,
) -> Result<Vec<Vec<GalaxyInFrame>>, Box<dyn std::error::Error>> {
    let cat = BrightGalaxyCatalog::load_embedded()?;
    info!("Bright-galaxies supplement loaded: {} entries", cat.len());

    let in_field = galaxies_in_field(&cat, pointing, fov_radius_deg, config);
    info!(
        "{} bright galaxies reach the {:.2}° cone around ({:.4}, {:.4})",
        in_field.len(),
        fov_radius_deg,
        pointing.ra_degrees(),
        pointing.dec_degrees()
    );

    let orientation = attitude_from_pointing(pointing, 0.0);
    let n_sensors = fp.array.sensor_count();
    let mut per_sensor: Vec<Vec<GalaxyInFrame>> = vec![Vec::new(); n_sensors];

    for (sensor_idx, sensor_galaxies) in per_sensor.iter_mut().enumerate() {
        let sat = match fp.satellite_for_sensor(sensor_idx) {
            Some(s) => s,
            None => continue,
        };
        let plate_scale_arcsec_per_px = sat.plate_scale_arcsec_per_pixel();
        let reference_disk = sat.airy_disk_pixel_space();
        let qe = &sat.sensor.quantum_efficiency;

        for entry in &in_field {
            let pos = Equatorial::from_degrees(entry.ra_deg, entry.dec_deg);
            let (px, py) = match fp.project_to_sensor(
                &starfield::catalogs::StarData::with_position(0, pos, 0.0, None),
                &orientation,
                sensor_idx,
                /* padding_mm */ 0.0,
            ) {
                Some(p) => p,
                None => continue,
            };
            let profile = match entry.sersic_profile() {
                Some(p) => p,
                None => continue,
            };
            let spectrum =
                BlackbodyStellarSpectrum::from_gaia_bv_magnitude(DEFAULT_BV, entry.mag_v as f64);
            let flux: SourceFlux = photon_electron_fluxes(&reference_disk, &spectrum, qe);
            let deposit = SersicSplat::new(profile, plate_scale_arcsec_per_px);
            sensor_galaxies.push(GalaxyInFrame {
                x: px,
                y: py,
                position: pos,
                id: hash_name(&entry.name),
                name: Some(entry.name.clone()),
                flux,
                deposit,
            });
            info!(
                "sensor {}: routed bright galaxy {} (mag_v={:.2}, theta_eff={:.1}\")",
                sensor_idx, entry.name, entry.mag_v, entry.radius_sersic_arcsec
            );
        }
    }

    Ok(per_sensor)
}

/// Stable u64 hash for the catalog id.
///
/// BrightGalaxy is keyed by name, not numeric id, but `GalaxyInFrame::id`
/// is `u64`. FNV-1a keeps the id stable across Rust releases and process
/// invocations.
fn hash_name(name: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    name.bytes().fold(FNV_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    })
}

/// Sky-position + ellipse view of the bright-galaxies in `fov_radius_deg`
/// around `pointing`, for the context-view dotted-ellipse overlay.
/// Mirrors [`crate::sims::nsa_galaxies::load_galaxies_in_fov`] — the
/// returned `GalaxyInField` records share the same shape so the
/// renderer can flatten both into a single overlay list.
pub fn load_bright_galaxies_in_fov(
    pointing: &Equatorial,
    fov_radius_deg: f64,
    config: &BrightGalaxyLoaderConfig,
) -> Result<Vec<GalaxyInField>, Box<dyn std::error::Error>> {
    let cat = BrightGalaxyCatalog::load_embedded()?;
    let out: Vec<GalaxyInField> = galaxies_in_field(&cat, pointing, fov_radius_deg, config)
        .into_iter()
        .filter_map(|g| {
            let profile = g.sersic_profile()?;
            Some(GalaxyInField {
                position: Equatorial::from_degrees(g.ra_deg, g.dec_deg),
                theta_half_arcsec: profile.theta_half_arcsec,
                axis_ratio: profile.axis_ratio,
                position_angle_deg: profile.position_angle_deg,
            })
        })
        .collect();
    info!(
        "{} bright galaxies for context overlay reach the {:.2}° cone around ({:.4}, {:.4})",
        out.len(),
        fov_radius_deg,
        pointing.ra_degrees(),
        pointing.dec_degrees()
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_name_is_stable_for_catalog_ids() {
        assert_eq!(hash_name("M31"), 0x1d0f_2419_b444_9500);
        assert_ne!(hash_name("M31"), hash_name("M33"));
    }

    #[test]
    fn field_admits_galaxies_by_envelope_not_centre() {
        let cat = BrightGalaxyCatalog::load_embedded().unwrap();
        let m31 = cat.get("M31").unwrap();
        let reach_deg = m31
            .sersic_profile()
            .unwrap()
            .radius_at_sb_fraction(TRUNCATION_SB_FRACTION)
            / 3600.0;
        let config = BrightGalaxyLoaderConfig::default();
        let names = |dec_offset_deg: f64| -> Vec<String> {
            let pointing = Equatorial::from_degrees(m31.ra_deg, m31.dec_deg + dec_offset_deg);
            galaxies_in_field(&cat, &pointing, 0.1, &config)
                .into_iter()
                .map(|g| g.name.clone())
                .collect()
        };
        // Centre outside the 0.1 deg field, envelope inside it.
        assert!(names(0.5 * reach_deg).contains(&"M31".to_string()));
        assert!(!names(reach_deg + 1.0).contains(&"M31".to_string()));
    }

    #[test]
    fn embedded_catalog_loads_for_bright_galaxy_routing() {
        let catalog = BrightGalaxyCatalog::load_embedded().expect("embedded catalog loads");

        assert!(!catalog.is_empty());
        assert!(catalog.get("M31").is_some());
    }
}

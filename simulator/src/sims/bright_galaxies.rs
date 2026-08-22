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
//!   in. We use a generous `fov_pad_deg` here (default 3°) so the
//!   centre-only test still catches them; once
//!   `BrightGalaxyCatalog::in_cone_extended` lands upstream
//!   (OrbitalCommons/starfield-datasources#54) we'll switch to it.

use log::info;
use starfield::Equatorial;
use starfield_bright_galaxies::{BrightGalaxy, BrightGalaxyCatalog};

use crate::hardware::satellite::{FocalPlaneConfig, FocalPlaneProjector};
use crate::image_proc::sersic_splat::SersicSplat;
use crate::photometry::photoconversion::{photon_electron_fluxes, SourceFlux};
use crate::photometry::BlackbodyStellarSpectrum;
use crate::scene_galaxy::GalaxyInFrame;
use crate::sims::nsa_galaxies::GalaxyInField;
use crate::sims::orientation::orientation_from_pointing;

/// B-V proxy used for the blackbody spectrum approximation. 0.85 is a
/// rough integrated colour for an Sb spiral; ellipticals are ~1.0,
/// late-type spirals are ~0.6. The choice doesn't affect the centroid
/// or spatial extent — only the per-band electron rate split.
const DEFAULT_BV: f64 = 0.85;

/// Configuration for the bright-galaxies loader.
#[derive(Debug, Clone)]
pub struct BrightGalaxyLoaderConfig {
    /// Pad the field-of-view filter by this many degrees so the
    /// centre-only `in_cone` test still catches galaxies whose outer
    /// envelope reaches into the cone (M31's truncated envelope
    /// extends ~2–3° from the centre at typical truncation budgets).
    pub fov_pad_deg: f64,
}

impl Default for BrightGalaxyLoaderConfig {
    fn default() -> Self {
        Self { fov_pad_deg: 3.0 }
    }
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

    let cos_dec0 = pointing.dec_degrees().to_radians().cos();
    let half_box_deg = fov_radius_deg + config.fov_pad_deg;
    let in_field: Vec<&BrightGalaxy> = cat
        .iter()
        .filter(|g| {
            let dra = (g.ra_deg - pointing.ra_degrees()) * cos_dec0;
            let ddec = g.dec_deg - pointing.dec_degrees();
            dra.abs() < half_box_deg && ddec.abs() < half_box_deg
        })
        .collect();
    info!(
        "{} bright galaxies in {:.2}° box around ({:.4}, {:.4})",
        in_field.len(),
        half_box_deg * 2.0,
        pointing.ra_degrees(),
        pointing.dec_degrees()
    );

    let orientation = orientation_from_pointing(pointing, 0.0);
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
            let profile =
                match <BrightGalaxy as starfield::catalogs::ExtendedSource>::sersic_profile(entry) {
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
    let cos_dec0 = pointing.dec_degrees().to_radians().cos();
    let half_box_deg = fov_radius_deg + config.fov_pad_deg;
    let out: Vec<GalaxyInField> = cat
        .iter()
        .filter(|g| {
            let dra = (g.ra_deg - pointing.ra_degrees()) * cos_dec0;
            let ddec = g.dec_deg - pointing.dec_degrees();
            dra.abs() < half_box_deg && ddec.abs() < half_box_deg
        })
        .map(|g| GalaxyInField {
            position: Equatorial::from_degrees(g.ra_deg, g.dec_deg),
            theta_half_arcsec: g.radius_sersic_arcsec as f64,
            // Bright-galaxies stores ellipticity = 1 - b/a; flip back
            // for the renderer's axis_ratio convention.
            axis_ratio: 1.0 - g.ellipticity_sersic as f64,
            position_angle_deg: g.pa_sersic_deg as f64,
        })
        .collect();
    info!(
        "{} bright galaxies for context overlay in {:.2}° box around ({:.4}, {:.4})",
        out.len(),
        half_box_deg * 2.0,
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
    fn embedded_catalog_loads_for_bright_galaxy_routing() {
        let catalog = BrightGalaxyCatalog::load_embedded().expect("embedded catalog loads");

        assert!(!catalog.is_empty());
        assert!(catalog.get("M31").is_some());
    }
}

//! NASA-Sloan Atlas (NSA) → flat [`Galaxy`] catalog routing.
//!
//! Loads an NSA FITS file (default location: `~/.cache/starfield/nsa/nsa_v0_1_2.fits`,
//! downloaded on demand via `starfield_nsa::download_nsa()` if absent),
//! filters to the field of view around a pointing, builds per-galaxy
//! Sérsic profiles + SDSS-spectrum flux objects, and returns them as
//! a flat `Vec<Galaxy>`. Per-sensor projection (with halo padding for
//! galaxies that subtend multiple sensors) is the motion-blur
//! renderer's responsibility.
//!
//! v1 surface area:
//!
//! - Always uses the *Sérsic-fit* fluxes (`NsaEntry::sersic_flux`),
//!   not the NMGY aperture fluxes — Sérsic-fit pairs naturally with
//!   the Sérsic SB profile so the rendered ellipse conserves the
//!   model flux. NMGY is the right choice for measured-radial-profile
//!   rendering when that lands as a follow-up.
//! - Filters out pathological fits at routing time:
//!   `n ∈ [0.6, 6.5]` and `theta_eff < 60″` — the NSA fitter
//!   occasionally blows up on bright stars or LSB galaxies, producing
//!   `theta_eff = 200″` n=0.5 entries that would dominate the
//!   render box and the per-pixel adaptive Simpson cost.
//! - Per-galaxy flux caching: the spectrum→`SourceFlux` integration
//!   is the expensive step, so it's done once per galaxy and reused
//!   across sensors.

use std::path::{Path, PathBuf};

use log::info;
use starfield::catalogs::{SersicProfile, StarCatalog};
use starfield::Equatorial;
use starfield_datasource_utils::cache_dir;
use starfield_nsa::{NsaCatalog, NsaEntry};

use crate::hardware::satellite::FocalPlaneConfig;
use crate::image_proc::sersic_splat::truncation_radius_arcsec;
use crate::photometry::photoconversion::{photon_electron_fluxes, SourceFlux};
use crate::photometry::{SDSSSpectrum, SdssBand};
use crate::scene_galaxy::Galaxy;

/// Default rejection threshold on `theta_half_arcsec`. NSA's typical
/// galaxies sit well below 30″; entries above 60″ are almost always
/// fit failures (bright stars, shredded LSBs).
pub const DEFAULT_MAX_THETA_EFF_ARCSEC: f64 = 60.0;

/// Default rejection range on Sérsic index. The NSA fitter clamps to
/// `n ∈ [0.5, 6]` but values right at the edges are usually
/// pathological — pick a slightly tighter bound that excludes the
/// rail-stuck blow-ups without losing real low-/high-n galaxies.
pub const DEFAULT_MIN_N: f64 = 0.6;
pub const DEFAULT_MAX_N: f64 = 6.5;

/// True if `entry`'s Sérsic fit is structurally valid and within
/// the supplied photometric/morphological rails.
pub fn is_well_fit(entry: &NsaEntry, min_n: f64, max_n: f64, max_theta_eff: f64) -> bool {
    let th = entry.sersic_th50 as f64;
    let n = entry.sersic_n as f64;
    let ba = entry.sersic_ba as f64;
    let phi = entry.sersic_phi as f64;
    th.is_finite()
        && th > 0.0
        && th < max_theta_eff
        && n.is_finite()
        && (min_n..=max_n).contains(&n)
        && ba.is_finite()
        && ba > 0.05
        && ba <= 1.0
        && phi.is_finite()
}

/// Build a `SersicProfile` from an `NsaEntry`'s structural fields.
/// **Mirrors the eventual `impl ExtendedSource for NsaEntry`**
/// (starfield-datasources #38) — when that lands upstream, this
/// function collapses to `entry.sersic_profile().unwrap()`.
fn nsa_to_sersic_profile(entry: &NsaEntry) -> SersicProfile {
    SersicProfile {
        theta_half_arcsec: entry.sersic_th50 as f64,
        n: entry.sersic_n as f64,
        axis_ratio: entry.sersic_ba as f64,
        position_angle_deg: entry.sersic_phi as f64,
    }
}

/// Build an `SDSSSpectrum` from an `NsaEntry`'s 5-band Sérsic-fit
/// fluxes (NSA's in-memory layout is 7-slot, with FUV/NUV padded to
/// zero on v0_1_2 files).
fn nsa_to_sdss_spectrum(entry: &NsaEntry) -> SDSSSpectrum {
    SDSSSpectrum::from_band_slice(&[
        (SdssBand::Fuv, entry.sersic_flux[0] as f64),
        (SdssBand::Nuv, entry.sersic_flux[1] as f64),
        (SdssBand::U, entry.sersic_flux[2] as f64),
        (SdssBand::G, entry.sersic_flux[3] as f64),
        (SdssBand::R, entry.sersic_flux[4] as f64),
        (SdssBand::I, entry.sersic_flux[5] as f64),
        (SdssBand::Z, entry.sersic_flux[6] as f64),
    ])
}

/// Configuration knobs for the loader. Defaults match NSA-typical
/// galaxies; expose as CLI flags if the renderer ever needs to render
/// a non-NSA catalog with different fit characteristics.
#[derive(Debug, Clone)]
pub struct GalaxyLoaderConfig {
    /// Reject entries with `theta_half_arcsec >= max_theta_eff`.
    pub max_theta_eff: f64,
    /// Reject entries with Sérsic index outside `[min_n, max_n]`.
    pub min_n: f64,
    pub max_n: f64,
}

impl Default for GalaxyLoaderConfig {
    fn default() -> Self {
        Self {
            max_theta_eff: DEFAULT_MAX_THETA_EFF_ARCSEC,
            min_n: DEFAULT_MIN_N,
            max_n: DEFAULT_MAX_N,
        }
    }
}

impl GalaxyLoaderConfig {
    /// Largest possible rendered major-axis radius admitted by these
    /// morphology rails, in degrees.
    pub fn max_visual_radius_deg(&self) -> f64 {
        let profile = SersicProfile {
            theta_half_arcsec: self.max_theta_eff,
            n: self.max_n,
            axis_ratio: 1.0,
            position_angle_deg: 0.0,
        };
        truncation_radius_arcsec(&profile) / 3600.0
    }
}

fn profile_overlaps_fov(
    profile: &SersicProfile,
    position: &Equatorial,
    pointing: &Equatorial,
    fov_radius_deg: f64,
) -> bool {
    let visual_radius_deg = truncation_radius_arcsec(profile) / 3600.0;
    pointing.angular_distance(position).to_degrees() <= fov_radius_deg + visual_radius_deg
}

/// Default cache path for the NSA FITS file:
/// `~/.cache/starfield/nsa/nsa_v0_1_2.fits`.
pub fn default_nsa_path() -> PathBuf {
    cache_dir().join("nsa").join("nsa_v0_1_2.fits")
}

/// Lightweight galaxy descriptor: sky position + Sérsic ellipse
/// parameters. Carries only what the context-view ellipse renderer
/// needs (no fluxes, no `SersicSplat`), so the per-galaxy cost stays
/// flat at a handful of `f64`s for the in-FOV catalog scan.
#[derive(Debug, Clone, Copy)]
pub struct GalaxyInField {
    pub position: Equatorial,
    pub theta_half_arcsec: f64,
    pub axis_ratio: f64,
    pub position_angle_deg: f64,
}

/// Load NSA from `path` (downloading if absent), filter to the angular
/// search region around `pointing`, drop pathological fits, clip each
/// candidate by its rendered radius, and return `GalaxyInField` entries.
///
/// Mirrors the in-FOV pre-filter in [`load_and_route_nsa_galaxies`] but
/// stops short of per-sensor projection — the context-view renderer
/// wants every galaxy in the focal-plane envelope, including ones that
/// land in a sensor gap, so it can mark them with a dotted ellipse.
pub fn load_galaxies_in_fov(
    path: &Path,
    pointing: &Equatorial,
    fov_radius_deg: f64,
    config: &GalaxyLoaderConfig,
) -> Result<Vec<GalaxyInField>, Box<dyn std::error::Error>> {
    let path = if path.exists() {
        path.to_path_buf()
    } else {
        info!("NSA FITS missing at {}; downloading...", path.display());
        starfield_nsa::download_nsa()?
    };
    let cat = NsaCatalog::from_fits_file(&path)?;
    let cos_dec0 = pointing.dec_degrees().to_radians().cos();
    let half_box_deg = fov_radius_deg + config.max_visual_radius_deg();
    let out: Vec<GalaxyInField> = cat
        .stars()
        .filter(|e| {
            let dra = (e.ra - pointing.ra_degrees()) * cos_dec0;
            let ddec = e.dec - pointing.dec_degrees();
            dra.abs() < half_box_deg && ddec.abs() < half_box_deg
        })
        .filter(|e| is_well_fit(e, config.min_n, config.max_n, config.max_theta_eff))
        .filter(|e| {
            let profile = nsa_to_sersic_profile(e);
            let position = Equatorial::from_degrees(e.ra, e.dec);
            profile_overlaps_fov(&profile, &position, pointing, fov_radius_deg)
        })
        .map(|e| GalaxyInField {
            position: Equatorial::from_degrees(e.ra, e.dec),
            theta_half_arcsec: e.sersic_th50 as f64,
            axis_ratio: e.sersic_ba as f64,
            position_angle_deg: e.sersic_phi as f64,
        })
        .collect();
    info!(
        "{} NSA galaxies in {:.2}° box around ({:.4}, {:.4}) for context overlay",
        out.len(),
        half_box_deg * 2.0,
        pointing.ra_degrees(),
        pointing.dec_degrees()
    );
    Ok(out)
}

/// Load NSA from `path` (downloading via the NYU mirror if absent),
/// search out to the largest admitted rendered radius, drop pathological
/// fits, clip each candidate by its own radius, and return a flat
/// `Vec<Galaxy>` of sky-truth catalog entries.
///
/// Per-sensor projection (including the multi-sensor halo case) is
/// the motion-blur renderer's job — see
/// [`crate::scene_galaxy::project_galaxies_to_sensors`].
///
/// `flux` is computed using the array's reference satellite (sensor
/// 0). For the homogeneous arrays currently in production this is
/// exactly the same flux every sensor would compute itself; for a
/// hypothetical heterogeneous array, per-sensor flux would belong
/// in a render-time cache analogous to stars' `FluxCache`.
pub fn load_and_route_nsa_galaxies(
    path: &Path,
    pointing: &Equatorial,
    fp: &FocalPlaneConfig,
    fov_radius_deg: f64,
    config: &GalaxyLoaderConfig,
) -> Result<Vec<Galaxy>, Box<dyn std::error::Error>> {
    info!(
        "loading NSA from {} (download on demand if absent)",
        path.display()
    );
    let path = if path.exists() {
        path.to_path_buf()
    } else {
        info!("NSA FITS missing at {}; downloading...", path.display());
        starfield_nsa::download_nsa()?
    };
    let cat = NsaCatalog::from_fits_file(&path)?;
    info!(
        "NSA loaded: {} galaxies, version {:?}",
        cat.len(),
        cat.version()
    );

    // Conservative broad-phase box reaches as far as the largest
    // profile admitted by the morphology rails. The following exact
    // angular test uses each candidate's own rendered radius.
    let cos_dec0 = pointing.dec_degrees().to_radians().cos();
    let half_box_deg = fov_radius_deg + config.max_visual_radius_deg();
    let in_field: Vec<&NsaEntry> = cat
        .stars()
        .filter(|e| {
            let dra = (e.ra - pointing.ra_degrees()) * cos_dec0;
            let ddec = e.dec - pointing.dec_degrees();
            dra.abs() < half_box_deg && ddec.abs() < half_box_deg
        })
        .filter(|e| is_well_fit(e, config.min_n, config.max_n, config.max_theta_eff))
        .filter(|e| {
            let profile = nsa_to_sersic_profile(e);
            let position = Equatorial::from_degrees(e.ra, e.dec);
            profile_overlaps_fov(&profile, &position, pointing, fov_radius_deg)
        })
        .collect();
    info!(
        "{} NSA galaxies in {:.2}° box around ({:.4}, {:.4}) after fit and extent filters",
        in_field.len(),
        half_box_deg * 2.0,
        pointing.ra_degrees(),
        pointing.dec_degrees()
    );

    // Flux computation uses the array's reference sensor (sensor 0) QE.
    // Homogeneous arrays — the current production case — share QE across
    // all sensors, so this is exact for every sensor the galaxy may
    // land on. For heterogeneous arrays this becomes a small per-sensor
    // approximation; the proper fix is render-time per-sensor flux
    // computation (analogous to stars' FluxCache) and is left for a
    // future PR.
    let reference_sat = fp
        .satellite_for_sensor(0)
        .ok_or("focal plane has no sensors")?;
    let reference_disk = reference_sat.airy_disk_pixel_space();
    let qe = &reference_sat.sensor.quantum_efficiency;

    let galaxies: Vec<Galaxy> = in_field
        .iter()
        .map(|entry| {
            let position = Equatorial::from_degrees(entry.ra, entry.dec);
            let profile = nsa_to_sersic_profile(entry);
            let spectrum = nsa_to_sdss_spectrum(entry);
            let flux: SourceFlux = photon_electron_fluxes(&reference_disk, &spectrum, qe);
            Galaxy {
                id: entry.nsaid as u64,
                name: None,
                position,
                profile,
                flux,
            }
        })
        .collect();
    info!("{} NSA galaxies prepared (flat, sky-truth)", galaxies.len());
    Ok(galaxies)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(theta_half_arcsec: f64, n: f64) -> SersicProfile {
        SersicProfile {
            theta_half_arcsec,
            n,
            axis_ratio: 0.7,
            position_angle_deg: 20.0,
        }
    }

    #[test]
    fn search_bound_covers_largest_admitted_profile() {
        let config = GalaxyLoaderConfig::default();
        let largest = profile(config.max_theta_eff, config.max_n);
        let actual_deg = truncation_radius_arcsec(&largest) / 3600.0;
        assert!(config.max_visual_radius_deg() >= actual_deg);
        assert!(config.max_visual_radius_deg() > 0.5);
    }

    #[test]
    fn off_sensor_centre_is_kept_when_halo_overlaps_fov() {
        let pointing = Equatorial::from_degrees(10.0, 0.0);
        let galaxy_profile = profile(30.0, 4.0);
        let visual_radius_deg = truncation_radius_arcsec(&galaxy_profile) / 3600.0;
        let fov_radius_deg = 0.2;
        let overlapping =
            Equatorial::from_degrees(10.0 + fov_radius_deg + visual_radius_deg * 0.5, 0.0);
        let separated =
            Equatorial::from_degrees(10.0 + fov_radius_deg + visual_radius_deg + 0.01, 0.0);

        assert!(profile_overlaps_fov(
            &galaxy_profile,
            &overlapping,
            &pointing,
            fov_radius_deg,
        ));
        assert!(!profile_overlaps_fov(
            &galaxy_profile,
            &separated,
            &pointing,
            fov_radius_deg,
        ));
    }
}

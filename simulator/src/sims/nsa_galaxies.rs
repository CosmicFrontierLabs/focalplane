//! NASA-Sloan Atlas (NSA) → `GalaxyInFrame` routing.
//!
//! Loads an NSA FITS file (default location: `~/.cache/starfield/nsa/nsa_v0_1_2.fits`,
//! downloaded on demand via `starfield_nsa::download_nsa()` if absent),
//! filters to the field of view around a pointing, builds per-galaxy
//! Sérsic deposits + SDSS-spectrum flux objects, and routes them to
//! the per-sensor `GalaxyInFrame` lists that `Scene::with_galaxies`
//! consumes.
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

use crate::hardware::satellite::{FocalPlaneConfig, FocalPlaneProjector};
use crate::image_proc::sersic_splat::SersicSplat;
use crate::photometry::photoconversion::{photon_electron_fluxes, SourceFlux};
use crate::photometry::{SDSSSpectrum, SdssBand};
use crate::scene_galaxy::GalaxyInFrame;
use crate::sims::orientation::orientation_from_pointing;

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
    /// Pad the field-of-view filter by this many degrees so a galaxy
    /// whose centre falls just outside the sensor footprint but whose
    /// outer envelope reaches in is still rendered.
    pub fov_pad_deg: f64,
}

impl Default for GalaxyLoaderConfig {
    fn default() -> Self {
        Self {
            max_theta_eff: DEFAULT_MAX_THETA_EFF_ARCSEC,
            min_n: DEFAULT_MIN_N,
            max_n: DEFAULT_MAX_N,
            fov_pad_deg: 0.5,
        }
    }
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
/// box around `pointing` plus `config.fov_pad_deg`, drop pathological
/// fits, and return a flat list of `GalaxyInField` entries.
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
    let half_box_deg = fov_radius_deg + config.fov_pad_deg;
    let out: Vec<GalaxyInField> = cat
        .stars()
        .filter(|e| {
            let dra = (e.ra - pointing.ra_degrees()) * cos_dec0;
            let ddec = e.dec - pointing.dec_degrees();
            dra.abs() < half_box_deg && ddec.abs() < half_box_deg
        })
        .filter(|e| is_well_fit(e, config.min_n, config.max_n, config.max_theta_eff))
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
/// filter to the field of view around `pointing` plus `fov_pad_deg`
/// of slack, drop pathological fits, and project to per-sensor
/// `GalaxyInFrame` lists.
///
/// The returned `Vec<Vec<GalaxyInFrame>>` matches the indexing of
/// `Scene::per_sensor_stars` and is the `Vec` that
/// `Scene::with_galaxies` expects.
pub fn load_and_route_nsa_galaxies(
    path: &Path,
    pointing: &Equatorial,
    fp: &FocalPlaneConfig,
    fov_radius_deg: f64,
    config: &GalaxyLoaderConfig,
) -> Result<Vec<Vec<GalaxyInFrame>>, Box<dyn std::error::Error>> {
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

    // Conservative angular pre-filter on the catalog before per-sensor
    // projection. fov_radius_deg + fov_pad_deg padding catches any
    // galaxy whose centre lies within reach of the array.
    let cos_dec0 = pointing.dec_degrees().to_radians().cos();
    let half_box_deg = fov_radius_deg + config.fov_pad_deg;
    let in_field: Vec<&NsaEntry> = cat
        .stars()
        .filter(|e| {
            let dra = (e.ra - pointing.ra_degrees()) * cos_dec0;
            let ddec = e.dec - pointing.dec_degrees();
            dra.abs() < half_box_deg && ddec.abs() < half_box_deg
        })
        .filter(|e| is_well_fit(e, config.min_n, config.max_n, config.max_theta_eff))
        .collect();
    info!(
        "{} NSA galaxies in {:.2}° box around ({:.4}, {:.4}) after well-fit filter",
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
            let pos = Equatorial::from_degrees(entry.ra, entry.dec);
            let (px, py) = match fp.project_to_sensor(
                &starfield::catalogs::StarData::with_position(0, pos, 0.0, None),
                &orientation,
                sensor_idx,
                /* padding_mm */ 0.0,
            ) {
                Some(p) => p,
                None => continue,
            };
            let profile = nsa_to_sersic_profile(entry);
            let spectrum = nsa_to_sdss_spectrum(entry);
            let flux: SourceFlux = photon_electron_fluxes(&reference_disk, &spectrum, qe);
            let deposit = SersicSplat::new(profile, plate_scale_arcsec_per_px);
            sensor_galaxies.push(GalaxyInFrame {
                x: px,
                y: py,
                id: entry.nsaid as u64,
                flux,
                deposit,
            });
        }
        info!(
            "sensor {}: {} galaxies routed",
            sensor_idx,
            sensor_galaxies.len()
        );
    }

    Ok(per_sensor)
}

//! Gaia DR3 catalog source for the renderer.
//!
//! Loads `Dr3Catalog` (from `starfield-gaia`) restricted to a circular
//! sky region around the trajectory pointing, augments with the embedded
//! Hipparcos bright-star supplement, and yields `StarData` rows the rest
//! of the focalplane pipeline already consumes.
//!
//! Compared to the older `MinimalCatalog`-from-`to_mag_19.bin` path,
//! every rendered star carries:
//!
//! - per-source DR3 photometry (`phot_g_mean_mag`)
//! - per-source `B-V` color derived from `BP-RP` via `Dr3Entry::b_v()`
//!   (linear fit calibrated on the Hipparcos cross-match), so
//!   `BlackbodyStellarSpectrum::from_gaia_bv_magnitude` gets a real
//!   color rather than the `DEFAULT_BV` fallback
//! - the bright-star Hipparcos supplement injected via
//!   `Dr3Catalog::augment_missing` so naked-eye stars (G < ~3) that
//!   were dropped from the published Gaia catalog still render

use std::path::PathBuf;

use log::info;
use starfield::Equatorial;
use starfield_datasource_utils::cache_dir;
use starfield_gaia::{Cone, Dr3Catalog};

/// Default location of the healpix-sharded DR3 mag-20 excerpt.
pub fn default_excerpt_dir() -> PathBuf {
    cache_dir().join("gaia-excerpts").join("dr3-mag20")
}

/// Load `Dr3Catalog` entries within `radius_deg` of `centre`, brighter
/// than `mag_limit`, from a healpix-sharded excerpt directory.
///
/// Forwards to `Dr3Catalog::from_excerpt_dir_for_cone`, which uses the
/// directory's manifest to identify the HEALPix level, computes the
/// cone-coverage cells via cdshealpix, parses only those shard files,
/// and post-filters rows by exact great-circle distance (HEALPix
/// covering is conservative; boundary cells are trimmed back to the
/// requested cone).
pub fn load_dr3_in_cone(
    excerpt_dir: &std::path::Path,
    centre: Equatorial,
    radius_deg: f64,
    mag_limit: f64,
) -> Result<Dr3Catalog, Box<dyn std::error::Error>> {
    info!(
        "load_dr3_in_cone(dir={}, ra={:.4}, dec={:.4}, radius={:.4}°, mag_limit={:.2})",
        excerpt_dir.display(),
        centre.ra_degrees(),
        centre.dec_degrees(),
        radius_deg,
        mag_limit
    );
    let cone = Cone::from_degrees(centre.ra_degrees(), centre.dec_degrees(), radius_deg);
    Ok(Dr3Catalog::from_excerpt_dir_for_cone(
        excerpt_dir,
        cone,
        mag_limit,
    )?)
}

/// Load a Gaia DR3 catalog cone + augment with the embedded Hipparcos
/// bright-star supplement. Returns the augmented catalog ready for
/// `StarCatalog::star_data()` iteration. The `n_added` count is the
/// number of bright supplement rows actually inserted (subject to the
/// same `mag_limit` gate).
pub fn load_dr3_cone_augmented(
    excerpt_dir: &std::path::Path,
    centre: Equatorial,
    radius_deg: f64,
    mag_limit: f64,
) -> Result<(Dr3Catalog, usize), Box<dyn std::error::Error>> {
    let mut cat = load_dr3_in_cone(excerpt_dir, centre, radius_deg, mag_limit)?;
    let n_added = cat.augment_missing(mag_limit)?;
    info!(
        "Gaia DR3 catalog: cone-loaded + augmented with {n_added} Hipparcos \
         bright-star supplement rows (mag_limit = {mag_limit:.2})"
    );
    Ok((cat, n_added))
}

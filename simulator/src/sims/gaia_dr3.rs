//! Gaia DR3 catalog source for the renderer.
//!
//! Holds a `LazyLoadingCatalog<Dr3>` (from `starfield-gaia`) for the
//! lifetime of a simulator run. The lazy handle reads only the excerpt
//! directory's manifest at construction; each cone query opens just the
//! HEALPix shards that intersect the cone, post-filters by exact
//! great-circle distance, and augments with the embedded Hipparcos
//! bright-star supplement so naked-eye stars dropped from the published
//! Gaia catalog still render.
//!
//! Every rendered star carries:
//!
//! - per-source DR3 photometry (`phot_g_mean_mag`)
//! - per-source `B-V` color derived from `BP-RP` via `Dr3Entry::b_v()`
//!   (linear fit calibrated on the Hipparcos cross-match), so
//!   `BlackbodyStellarSpectrum::from_gaia_bv_magnitude` gets a real
//!   color rather than the `DEFAULT_BV` fallback
//! - the bright-star Hipparcos supplement injected via
//!   `Dr3Catalog::augment_missing` (G < ~3 stars dropped from Gaia)

use std::path::{Path, PathBuf};

use log::info;
use starfield::Equatorial;
use starfield_datasource_utils::cache_dir;
use starfield_gaia::{Cone, Dr3, Dr3Catalog, GaiaCatalog, LazyLoadingCatalog};

/// Default location of the healpix-sharded DR3 mag-20 excerpt.
pub fn default_excerpt_dir() -> PathBuf {
    cache_dir().join("gaia-excerpts").join("dr3-mag20")
}

/// Open a lazy view over a HEALPix-sharded DR3 excerpt directory. Reads
/// only the manifest; no shard files are touched until a cone is
/// requested. Errors out if the directory is not a one-file-per-cell
/// HEALPix excerpt for the DR3 release.
pub fn open_dr3_excerpt(
    excerpt_dir: &Path,
) -> Result<LazyLoadingCatalog<Dr3>, Box<dyn std::error::Error>> {
    info!(
        "Opening Gaia DR3 excerpt at {} (lazy; manifest only)",
        excerpt_dir.display()
    );
    Ok(LazyLoadingCatalog::<Dr3>::open(excerpt_dir)?)
}

/// Materialize a cone from a lazy DR3 excerpt and augment it with the
/// embedded Hipparcos bright-star supplement. Returns the augmented
/// catalog ready for `StarCatalog::star_data()` iteration; the second
/// element is the supplement-row count actually inserted (subject to
/// `mag_limit`).
pub fn materialize_cone_augmented(
    lazy: &LazyLoadingCatalog<Dr3>,
    centre: Equatorial,
    radius_deg: f64,
    mag_limit: f64,
) -> Result<(Dr3Catalog, usize), Box<dyn std::error::Error>> {
    info!(
        "Materializing Gaia DR3 cone (ra={:.4}, dec={:.4}, radius={:.4}°, mag_limit={:.2})",
        centre.ra_degrees(),
        centre.dec_degrees(),
        radius_deg,
        mag_limit
    );
    let cone = Cone::from_degrees(centre.ra_degrees(), centre.dec_degrees(), radius_deg);
    let mem = lazy.materialize_cone(cone, mag_limit)?;
    let mut cat = Dr3Catalog(mem);
    let n_added = cat.augment_missing(mag_limit)?;
    info!(
        "Gaia DR3 catalog: cone-loaded + augmented with {n_added} Hipparcos \
         bright-star supplement rows (mag_limit = {mag_limit:.2})"
    );
    Ok((cat, n_added))
}

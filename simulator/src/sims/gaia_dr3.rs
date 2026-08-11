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

use std::path::PathBuf;

use log::info;
use starfield::Equatorial;
use starfield_datasource_utils::cache_dir;
use starfield_gaia::{Cone, Dr3, Dr3Catalog, GaiaCatalog, LazyLoadingCatalog};

/// Default location of the healpix-sharded DR3 mag-20 excerpt.
pub fn default_excerpt_dir() -> PathBuf {
    cache_dir().join("gaia-excerpts").join("dr3-mag20")
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
    // Hipparcos supplement injection, restricted to the cone. The
    // upstream `Dr3Catalog::augment_missing` inserts every supplement
    // row whose `fitted_g_mag` is brighter than `mag_limit` regardless
    // of sky position, which dumps ~15 k stars sky-wide for a typical
    // mag-19 limit; the loop below uses the same parser + entry
    // converter but adds the cone-containment test before each insert.
    let supplement = starfield_gaia::dr3::supplement::parse_embedded_supplement()?;
    let mut n_added = 0usize;
    for row in &supplement {
        if row.fitted_g_mag > mag_limit {
            continue;
        }
        if !cone.contains_radec_deg(row.ra, row.dec) {
            continue;
        }
        cat.insert(starfield_gaia::dr3::supplement::supplement_row_to_entry(
            row,
        ));
        n_added += 1;
    }
    info!(
        "Gaia DR3 catalog: cone-loaded + augmented with {n_added} Hipparcos \
         bright-star supplement rows in cone (mag_limit = {mag_limit:.2})"
    );
    Ok((cat, n_added))
}

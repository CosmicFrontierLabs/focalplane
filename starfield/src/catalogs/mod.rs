//! Star catalogs module
//!
//! This module provides functionality for loading and using star catalogs,
//! including efficient binary formats for optimized storage and loading.

pub mod binary_catalog;
mod gaia;
pub mod hipparcos;
pub mod synthetic;

pub use binary_catalog::{BinaryCatalog, MinimalStar};
pub use gaia::{GaiaCatalog, GaiaEntry};
pub use hipparcos::{HipparcosCatalog, HipparcosEntry};
pub use synthetic::{
    create_fov_catalog, create_synthetic_catalog, MagnitudeDistribution, SpatialDistribution,
    SyntheticCatalogConfig,
};

/// Trait for accessing star position data
pub trait StarPosition {
    /// Get star right ascension in degrees
    fn ra(&self) -> f64;

    /// Get star declination in degrees
    fn dec(&self) -> f64;
}

/// Common star properties that all catalog entries must provide
/// This represents the minimal set of properties required for rendering and calculations
#[derive(Debug, Clone, Copy)]
pub struct StarData {
    /// Star identifier
    pub id: u64,
    /// Right ascension in degrees
    pub ra: f64,
    /// Declination in degrees
    pub dec: f64,
    /// Apparent magnitude (lower is brighter)
    pub magnitude: f64,
    /// Optional B-V color index for rendering
    pub b_v: Option<f64>,
}

impl StarData {
    /// Create a new minimal star data structure
    pub fn new(id: u64, ra: f64, dec: f64, magnitude: f64, b_v: Option<f64>) -> Self {
        Self {
            id,
            ra,
            dec,
            magnitude,
            b_v,
        }
    }
}

impl StarPosition for StarData {
    fn ra(&self) -> f64 {
        self.ra
    }

    fn dec(&self) -> f64 {
        self.dec
    }
}

/// Generic trait for all star catalogs
pub trait StarCatalog {
    /// Star entry type for this catalog
    type Star;

    /// Get a star by its identifier
    fn get_star(&self, id: usize) -> Option<&Self::Star>;

    /// Get all stars in the catalog
    fn stars(&self) -> impl Iterator<Item = &Self::Star>;

    /// Get the number of stars in the catalog
    fn len(&self) -> usize;

    /// Check if the catalog is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Filter stars based on a predicate
    fn filter<F>(&self, predicate: F) -> Vec<&Self::Star>
    where
        F: Fn(&Self::Star) -> bool;

    /// Get stars as a unified StarData format
    /// This allows consistent handling of stars from different catalog types
    fn star_data(&self) -> impl Iterator<Item = StarData> + '_;

    /// Filter stars and return them in the standard format
    fn filter_star_data<F>(&self, predicate: F) -> Vec<StarData>
    where
        F: Fn(&StarData) -> bool;

    /// Get stars brighter than a specified magnitude in the standard format
    fn brighter_than(&self, magnitude: f64) -> Vec<StarData> {
        self.filter_star_data(|star| star.magnitude <= magnitude)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalogs::binary_catalog::{BinaryCatalog, MinimalStar};

    /// Test the StarCatalog trait with a simple binary catalog
    #[test]
    fn test_star_data() {
        // Create a binary catalog
        let stars = vec![
            MinimalStar::new(1, 100.0, 10.0, -1.5), // Sirius-like
            MinimalStar::new(2, 50.0, -20.0, 0.5),  // Canopus-like
            MinimalStar::new(3, 150.0, 30.0, 1.2),  // Bright
            MinimalStar::new(4, 200.0, -45.0, 3.7), // Medium
            MinimalStar::new(5, 250.0, 60.0, 5.9),  // Dim
        ];

        let catalog = BinaryCatalog::from_stars(stars, "Test catalog");

        // Test star_data iterator
        let star_data: Vec<StarData> = catalog.star_data().collect();
        assert_eq!(star_data.len(), 5);

        // Test brighter_than
        let bright_stars = catalog.brighter_than(1.0);
        assert_eq!(bright_stars.len(), 2);

        // Verify the brightest star
        let brightest = star_data
            .iter()
            .min_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap())
            .unwrap();
        assert_eq!(brightest.magnitude, -1.5);
    }
}

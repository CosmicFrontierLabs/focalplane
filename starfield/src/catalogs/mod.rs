//! Star catalogs module
//!
//! This module provides functionality for loading and using star catalogs.

mod gaia;
pub mod hipparcos;

pub use gaia::{GaiaCatalog, GaiaEntry};
pub use hipparcos::{HipparcosCatalog, HipparcosEntry};

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
}

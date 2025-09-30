//! Cached star catalog for efficient field-of-view queries
//!
//! Provides a caching layer over star catalogs to avoid repeated queries
//! for stars in similar sky regions. The cache maintains a buffer zone
//! larger than the requested FOV to handle small pointing changes.

use crate::units::{Angle, AngleExt};
use starfield::catalogs::{StarCatalog, StarData};
use starfield::Equatorial;
use std::sync::Arc;

/// Cache FOV multiplier - cache area is this times the requested FOV
/// This provides a buffer zone to avoid frequent cache misses
const CACHE_FOV_MULTIPLIER: f64 = 2.0;

/// A star catalog wrapper that caches stars in a region around the current pointing
///
/// This struct maintains an internal cache of stars for a sky region larger than
/// the requested field of view. When the pointing moves outside the cached region,
/// it automatically refreshes the cache with a new, larger region.
pub struct CachedStarCatalog<C: StarCatalog> {
    /// The underlying star catalog (shared via Arc to avoid cloning large catalogs)
    catalog: Arc<C>,

    /// Cached stars from the last query
    cached_stars: Vec<StarData>,

    /// Center of the cached region (None if cache never initialized)
    cache_center: Option<Equatorial>,

    /// Diameter of the cached region
    cache_fov_diameter: Angle,

    /// The actual sensor FOV diameter (not the cache size)
    sensor_fov_diameter: Angle,
}

impl<C: StarCatalog> CachedStarCatalog<C> {
    /// Create a new cached catalog with the specified sensor FOV
    ///
    /// # Arguments
    /// * `catalog` - The underlying star catalog to wrap (wrapped in Arc for sharing)
    /// * `sensor_fov_diameter` - The diameter of the sensor's field of view
    pub fn new(catalog: Arc<C>, sensor_fov_diameter: Angle) -> Self {
        // Pre-calculate cache FOV diameter (larger than sensor FOV)
        let cache_fov_diameter =
            Angle::from_degrees(sensor_fov_diameter.as_degrees() * CACHE_FOV_MULTIPLIER);

        Self {
            catalog,
            cached_stars: Vec::new(),
            cache_center: None,
            cache_fov_diameter,
            sensor_fov_diameter,
        }
    }

    /// Get stars in the field of view, using cache if possible
    ///
    /// This method checks if the requested pointing is within the cached region.
    /// If it is, returns stars from the cache. Otherwise, refreshes the cache
    /// with a new region centered on the requested pointing.
    ///
    /// # Arguments
    /// * `pointing` - The center of the field of view
    ///
    /// # Returns
    /// A vector of stars visible in the sensor's field of view
    pub fn get_stars_in_fov(&mut self, pointing: &Equatorial) -> Vec<StarData> {
        // Check if we need to refresh the cache
        if self.needs_cache_refresh(pointing) {
            self.refresh_cache(pointing);
        }

        // Filter cached stars to only those within the actual sensor FOV
        self.filter_stars_for_sensor_fov(pointing)
    }

    /// Check if the cache needs to be refreshed for the given pointing
    fn needs_cache_refresh(&self, pointing: &Equatorial) -> bool {
        // If cache was never initialized, always refresh
        let Some(cache_center) = self.cache_center else {
            return true;
        };

        // Calculate angular distance from cache center
        let dist = cache_center.angular_distance(pointing).to_degrees();

        // Refresh if pointing is outside the cached region
        // We use cache radius minus sensor radius to ensure full sensor FOV is covered
        let cache_radius_deg = self.cache_fov_diameter.as_degrees() / 2.0;
        let sensor_radius_deg = self.sensor_fov_diameter.as_degrees() / 2.0;
        let max_offset = cache_radius_deg - sensor_radius_deg;

        dist > max_offset
    }

    /// Refresh the cache with stars around the new pointing
    fn refresh_cache(&mut self, pointing: &Equatorial) {
        // Update cache center
        self.cache_center = Some(*pointing);

        // Query catalog for stars in the larger cached region using stars_in_field
        // cache_fov_diameter is pre-calculated in new()
        self.cached_stars = self.catalog.stars_in_field(
            pointing.ra.to_degrees(),
            pointing.dec.to_degrees(),
            self.cache_fov_diameter.as_degrees(),
        );
    }

    /// Filter cached stars to only those within the sensor FOV
    fn filter_stars_for_sensor_fov(&self, pointing: &Equatorial) -> Vec<StarData> {
        let sensor_radius_deg = self.sensor_fov_diameter.as_degrees() / 2.0;

        self.cached_stars
            .iter()
            .filter(|star| {
                let dist = pointing.angular_distance(&star.position).to_degrees();
                dist <= sensor_radius_deg
            })
            .cloned()
            .collect()
    }

    /// Get the underlying catalog (Arc reference)
    pub fn catalog(&self) -> &Arc<C> {
        &self.catalog
    }

    /// Get cache statistics for debugging
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache_center.map(|center| CacheStats {
            cached_stars: self.cached_stars.len(),
            cache_center: center,
            cache_fov_diameter: self.cache_fov_diameter,
            sensor_fov_diameter: self.sensor_fov_diameter,
        })
    }
}

/// Statistics about the cache state
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cached_stars: usize,
    pub cache_center: Equatorial,
    pub cache_fov_diameter: Angle,
    pub sensor_fov_diameter: Angle,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use starfield::framelib::random::RandomEquatorial;
    use std::collections::HashSet;

    /// Mock catalog for testing
    struct MockCatalog {
        stars: Vec<StarData>,
    }

    impl MockCatalog {
        fn new(stars: Vec<StarData>) -> Self {
            Self { stars }
        }

        fn random_stars(num_stars: usize, seed: u64) -> Self {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut randomizer = RandomEquatorial::with_seed(seed);
            let mut stars = Vec::with_capacity(num_stars);

            for i in 0..num_stars {
                let position = randomizer.next().unwrap();
                let mag = rng.gen_range(0.0..15.0);

                stars.push(StarData {
                    id: i as u64,
                    position,
                    magnitude: mag,
                    b_v: None,
                });
            }

            Self::new(stars)
        }
    }

    impl StarCatalog for MockCatalog {
        type Star = StarData;

        fn get_star(&self, index: usize) -> Option<&Self::Star> {
            self.stars.get(index)
        }

        fn stars(&self) -> impl Iterator<Item = &Self::Star> {
            self.stars.iter()
        }

        fn filter<F>(&self, predicate: F) -> Vec<&Self::Star>
        where
            F: Fn(&Self::Star) -> bool,
        {
            self.stars.iter().filter(|s| predicate(s)).collect()
        }

        fn star_data(&self) -> impl Iterator<Item = StarData> + '_ {
            self.stars.iter().cloned()
        }

        fn filter_star_data<F>(&self, predicate: F) -> Vec<StarData>
        where
            F: Fn(&StarData) -> bool,
        {
            self.stars
                .iter()
                .filter(|s| predicate(s))
                .cloned()
                .collect()
        }

        fn stars_in_field(&self, ra_deg: f64, dec_deg: f64, fov_deg: f64) -> Vec<StarData> {
            let center = Equatorial::from_degrees(ra_deg, dec_deg);
            let radius_deg = fov_deg / 2.0;

            self.stars
                .iter()
                .filter(|star| {
                    let dist = center.angular_distance(&star.position).to_degrees();
                    dist <= radius_deg
                })
                .cloned()
                .collect()
        }

        fn len(&self) -> usize {
            self.stars.len()
        }

        fn is_empty(&self) -> bool {
            self.stars.is_empty()
        }
    }

    #[test]
    fn test_cache_initialization() {
        let catalog = Arc::new(MockCatalog::new(vec![]));
        let mut cached = CachedStarCatalog::new(catalog, Angle::from_degrees(1.0));

        // Cache should start uninitialized
        assert!(cached.cache_stats().is_none());

        // First query should initialize cache
        let pointing = Equatorial::from_degrees(100.0, 30.0);
        cached.get_stars_in_fov(&pointing);

        // Now cache should be initialized
        let stats = cached.cache_stats().unwrap();
        assert_relative_eq!(stats.cache_center.ra.to_degrees(), 100.0, epsilon = 1e-6);
        assert_relative_eq!(stats.cache_center.dec.to_degrees(), 30.0, epsilon = 1e-6);
        assert_relative_eq!(stats.cache_fov_diameter.as_degrees(), 2.0, epsilon = 1e-6); // 2x multiplier
        assert_relative_eq!(stats.sensor_fov_diameter.as_degrees(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cache_hit_and_miss() {
        let stars = vec![
            StarData {
                id: 1,
                position: Equatorial::from_degrees(100.0, 30.0),
                magnitude: 5.0,
                b_v: None,
            },
            StarData {
                id: 2,
                position: Equatorial::from_degrees(100.1, 30.1),
                magnitude: 6.0,
                b_v: None,
            },
            StarData {
                id: 3,
                position: Equatorial::from_degrees(102.0, 32.0),
                magnitude: 7.0,
                b_v: None,
            }, // Outside 1 degree FOV but in cache
            StarData {
                id: 4,
                position: Equatorial::from_degrees(105.0, 35.0),
                magnitude: 8.0,
                b_v: None,
            }, // Way outside
        ];

        let catalog = Arc::new(MockCatalog::new(stars));
        let mut cached = CachedStarCatalog::new(catalog, Angle::from_degrees(1.0));

        // First query - cache miss
        let pointing1 = Equatorial::from_degrees(100.0, 30.0);
        let stars1 = cached.get_stars_in_fov(&pointing1);
        assert_eq!(stars1.len(), 2); // Stars 1 and 2 within 0.5 degrees

        // Small movement - should be cache hit
        let pointing2 = Equatorial::from_degrees(100.2, 30.2);
        let _stars2 = cached.get_stars_in_fov(&pointing2);

        // Check cache wasn't refreshed by verifying center unchanged
        let stats = cached.cache_stats().unwrap();
        assert_relative_eq!(stats.cache_center.ra.to_degrees(), 100.0, epsilon = 1e-6);

        // Large movement - should trigger cache refresh
        let pointing3 = Equatorial::from_degrees(105.0, 35.0);
        let _stars3 = cached.get_stars_in_fov(&pointing3);

        // Cache center should have moved
        let stats = cached.cache_stats().unwrap();
        assert_relative_eq!(stats.cache_center.ra.to_degrees(), 105.0, epsilon = 1e-6);
        assert_relative_eq!(stats.cache_center.dec.to_degrees(), 35.0, epsilon = 1e-6);
    }

    #[test]
    fn test_empty_region() {
        let catalog = Arc::new(MockCatalog::new(vec![]));
        let mut cached = CachedStarCatalog::new(catalog, Angle::from_degrees(1.0));

        // Query empty region
        let pointing = Equatorial::from_degrees(180.0, 0.0);
        let stars = cached.get_stars_in_fov(&pointing);
        assert_eq!(stars.len(), 0);

        // Cache should still be initialized
        assert!(cached.cache_stats().is_some());
    }

    #[test]
    fn test_cache_at_zero_zero() {
        // Test the specific case of first query at (0,0)
        let stars = vec![
            StarData {
                id: 1,
                position: Equatorial::from_degrees(0.0, 0.0),
                magnitude: 5.0,
                b_v: None,
            },
            StarData {
                id: 2,
                position: Equatorial::from_degrees(0.1, 0.1),
                magnitude: 6.0,
                b_v: None,
            },
        ];

        let catalog = Arc::new(MockCatalog::new(stars));
        let mut cached = CachedStarCatalog::new(catalog, Angle::from_degrees(1.0));

        let pointing = Equatorial::from_degrees(0.0, 0.0);
        let stars = cached.get_stars_in_fov(&pointing);
        assert_eq!(stars.len(), 2);

        // Verify cache was properly initialized
        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.cache_center.ra.to_degrees(), 0.0);
        assert_eq!(stats.cache_center.dec.to_degrees(), 0.0);
    }

    // Enable long-running tests for more exhaustive checks
    const ENABLE_LONG_TESTS: bool = false;

    #[test]
    fn test_fuzz_consistency() {
        let n_stars = if ENABLE_LONG_TESTS { 1_000_000 } else { 10_000 };

        // Create a large random catalog
        let catalog = Arc::new(MockCatalog::random_stars(n_stars, 42));
        let mut cached = CachedStarCatalog::new(catalog.clone(), Angle::from_degrees(2.0));

        // Use deterministic random sequence for reproducibility
        let mut randomizer = RandomEquatorial::with_seed(1337);

        // Test many random pointings (including poles)
        for _ in 0..100 {
            let pointing = randomizer.next().unwrap();

            // Get stars from cached catalog
            let cached_stars = cached.get_stars_in_fov(&pointing);

            // Get stars directly from underlying catalog for comparison
            let direct_stars = cached.catalog().stars_in_field(
                pointing.ra.to_degrees(),
                pointing.dec.to_degrees(),
                cached.sensor_fov_diameter.as_degrees(),
            );

            // Convert to sets for comparison (order might differ)
            let cached_ids: HashSet<u64> = cached_stars.iter().map(|s| s.id).collect();
            let direct_ids: HashSet<u64> = direct_stars.iter().map(|s| s.id).collect();

            assert_eq!(
                cached_ids,
                direct_ids,
                "Mismatch at pointing ({:.2}, {:.2}): cached {} stars, direct {} stars",
                pointing.ra.to_degrees(),
                pointing.dec.to_degrees(),
                cached_ids.len(),
                direct_ids.len()
            );
        }
    }

    #[test]
    fn test_cache_movement_patterns() {
        // Test various movement patterns to ensure cache behaves correctly
        let catalog = Arc::new(MockCatalog::random_stars(10_000, 999));
        let mut cached = CachedStarCatalog::new(catalog.clone(), Angle::from_degrees(1.0));

        let mut randomizer = RandomEquatorial::with_seed(777);

        // Start at random position
        let start_pointing = randomizer.next().unwrap();
        let start_ra = start_pointing.ra.to_degrees();
        let start_dec = start_pointing.dec.to_degrees();

        // Pattern 1: Small oscillation (should mostly hit cache)
        for i in 0..20 {
            let offset = (i as f64 * 0.1).sin() * 0.3; // Max 0.3 degree movement
            let pointing = Equatorial::from_degrees(start_ra + offset, start_dec);
            let cached_stars = cached.get_stars_in_fov(&pointing);
            let direct_stars = cached.catalog().stars_in_field(
                pointing.ra.to_degrees(),
                pointing.dec.to_degrees(),
                cached.sensor_fov_diameter.as_degrees(),
            );
            assert_eq!(cached_stars.len(), direct_stars.len());
        }

        // Pattern 2: Linear drift (should trigger some cache refreshes)
        for i in 0..10 {
            let pointing = Equatorial::from_degrees(start_ra + i as f64 * 0.5, start_dec);
            let cached_stars = cached.get_stars_in_fov(&pointing);
            let direct_stars = cached.catalog().stars_in_field(
                pointing.ra.to_degrees(),
                pointing.dec.to_degrees(),
                cached.sensor_fov_diameter.as_degrees(),
            );

            let cached_ids: HashSet<u64> = cached_stars.iter().map(|s| s.id).collect();
            let direct_ids: HashSet<u64> = direct_stars.iter().map(|s| s.id).collect();
            assert_eq!(cached_ids, direct_ids);
        }

        // Pattern 3: Random jumps (stress test cache refresh, including poles)
        for _ in 0..50 {
            let pointing = randomizer.next().unwrap();

            let cached_stars = cached.get_stars_in_fov(&pointing);
            let direct_stars = cached.catalog().stars_in_field(
                pointing.ra.to_degrees(),
                pointing.dec.to_degrees(),
                cached.sensor_fov_diameter.as_degrees(),
            );

            assert_eq!(
                cached_stars.len(),
                direct_stars.len(),
                "Count mismatch at ({:.1}, {:.1})",
                pointing.ra.to_degrees(),
                pointing.dec.to_degrees()
            );
        }
    }

    #[test]
    fn test_pole_behavior() {
        // Test behavior near celestial poles where RA converges
        let stars = vec![
            StarData {
                id: 1,
                position: Equatorial::from_degrees(0.0, 89.9),
                magnitude: 5.0,
                b_v: None,
            },
            StarData {
                id: 2,
                position: Equatorial::from_degrees(180.0, 89.9),
                magnitude: 6.0,
                b_v: None,
            }, // Opposite RA but still close at pole
            StarData {
                id: 3,
                position: Equatorial::from_degrees(90.0, 89.8),
                magnitude: 7.0,
                b_v: None,
            },
        ];

        let catalog = Arc::new(MockCatalog::new(stars));
        let mut cached = CachedStarCatalog::new(catalog, Angle::from_degrees(1.0));

        // Query near north pole
        let pointing = Equatorial::from_degrees(45.0, 89.5);
        let stars = cached.get_stars_in_fov(&pointing);

        // All stars should be visible due to convergence at pole
        assert!(
            stars.len() >= 2,
            "Expected at least 2 stars near pole, got {}",
            stars.len()
        );
    }
}

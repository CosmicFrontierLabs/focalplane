//! Bad pixel map for sensor defect correction.
//!
//! Provides a structured way to store and load pixel defect information for sensors.
//! Bad pixel maps can be generated from dark frame analysis or loaded from files.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Bad pixel map for a specific camera sensor.
///
/// Simple structure containing just the bad pixel coordinates, camera info, and timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BadPixelMap {
    /// Sensor model name
    pub sensor_model: String,

    /// Camera serial number
    pub camera_serial: String,

    /// Timestamp when the map was generated (Unix epoch seconds)
    pub timestamp: u64,

    /// List of bad pixel coordinates (x, y)
    pub pixels: Vec<(usize, usize)>,
}

impl BadPixelMap {
    /// Create a new bad pixel map
    pub fn new(sensor_model: String, camera_serial: String, timestamp: u64) -> Self {
        Self {
            sensor_model,
            camera_serial,
            timestamp,
            pixels: Vec::new(),
        }
    }

    /// Add a bad pixel to the map
    pub fn add_pixel(&mut self, x: usize, y: usize) {
        self.pixels.push((x, y));
    }

    /// Get total number of bad pixels
    pub fn num_bad_pixels(&self) -> usize {
        self.pixels.len()
    }

    /// Get bad pixels as a set of coordinates for fast lookup
    pub fn as_coordinate_set(&self) -> HashSet<(usize, usize)> {
        self.pixels.iter().copied().collect()
    }

    /// Check if a pixel is marked as bad
    pub fn is_bad_pixel(&self, x: usize, y: usize) -> bool {
        self.pixels.contains(&(x, y))
    }

    /// Save to JSON file
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load from JSON file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Calculate Euclidean distance from given pixel to nearest bad pixel.
    ///
    /// Returns the distance in pixels, or None if there are no bad pixels in the map.
    pub fn distance_to_nearest_bad_pixel(&self, x: usize, y: usize) -> Option<f64> {
        if self.pixels.is_empty() {
            return None;
        }

        let min_distance = self
            .pixels
            .iter()
            .map(|(bad_x, bad_y)| {
                let dx = x as f64 - *bad_x as f64;
                let dy = y as f64 - *bad_y as f64;
                (dx * dx + dy * dy).sqrt()
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        Some(min_distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bad_pixel_map_creation() {
        let mut map = BadPixelMap::new("TestCam".to_string(), "12345".to_string(), 1704067200);

        map.add_pixel(100, 200);
        map.add_pixel(150, 250);

        assert_eq!(map.num_bad_pixels(), 2);
        assert!(map.is_bad_pixel(100, 200));
        assert!(!map.is_bad_pixel(100, 201));
    }

    #[test]
    fn test_coordinate_set() {
        let mut map = BadPixelMap::new("TestCam".to_string(), "12345".to_string(), 1704067200);

        map.add_pixel(10, 20);
        map.add_pixel(30, 40);

        let coords = map.as_coordinate_set();
        assert!(coords.contains(&(10, 20)));
        assert!(coords.contains(&(30, 40)));
        assert!(!coords.contains(&(50, 60)));
    }

    #[test]
    fn test_distance_to_nearest_bad_pixel() {
        let mut map = BadPixelMap::new("TestCam".to_string(), "12345".to_string(), 1704067200);

        map.add_pixel(10, 10);
        map.add_pixel(100, 100);

        assert_eq!(map.distance_to_nearest_bad_pixel(10, 10), Some(0.0));
        assert!((map.distance_to_nearest_bad_pixel(13, 14).unwrap() - 5.0).abs() < 0.01);

        let empty_map = BadPixelMap::new("TestCam".to_string(), "12345".to_string(), 1704067200);
        assert_eq!(empty_map.distance_to_nearest_bad_pixel(10, 10), None);
    }
}

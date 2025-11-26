//! Configuration storage for camera calibration data.
//!
//! Provides centralized storage for camera-specific configuration like bad pixel maps.
//! All config is stored in ~/.cf_config/ by default.

use crate::bad_pixel_map::BadPixelMap;
use crate::optical_alignment::OpticalAlignment;
use std::path::{Path, PathBuf};

/// Configuration storage manager for camera calibration data.
///
/// Manages loading and saving of camera-specific configuration files
/// from a centralized directory (defaults to ~/.cf_config/).
#[derive(Debug, Clone)]
pub struct ConfigStorage {
    /// Root directory for all configuration (e.g., ~/.cf_config)
    root_path: PathBuf,
}

impl ConfigStorage {
    /// Create a new config storage with default path (~/.cf_config)
    pub fn new() -> std::io::Result<Self> {
        let home = std::env::var("HOME")
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::NotFound, "HOME not set"))?;
        let root_path = PathBuf::from(home).join(".cf_config");
        Ok(Self { root_path })
    }

    /// Create a new config storage with custom root path
    pub fn with_path(root_path: PathBuf) -> Self {
        Self { root_path }
    }

    /// Get the root configuration path
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }

    /// Get the bad pixel maps directory path
    fn bad_pixel_maps_dir(&self) -> PathBuf {
        self.root_path.join("bad_pixel_maps")
    }

    /// Generate filename for a bad pixel map given model and serial number
    fn bad_pixel_map_filename(&self, model: &str, serial: &str) -> PathBuf {
        assert!(
            !model.contains('-'),
            "Model name cannot contain dash character"
        );
        assert!(
            !serial.contains('-'),
            "Serial number cannot contain dash character"
        );

        let model_safe = model.replace(' ', "_");
        let filename = format!("{model_safe}-{serial}.json");
        self.bad_pixel_maps_dir().join(filename)
    }

    /// Get bad pixel map for a given camera model and serial number.
    ///
    /// Returns None if no bad pixel map exists for this camera.
    /// Returns Some(Err) if the file exists but cannot be loaded.
    pub fn get_bad_pixel_map(
        &self,
        model: &str,
        serial: &str,
    ) -> Option<Result<BadPixelMap, std::io::Error>> {
        let path = self.bad_pixel_map_filename(model, serial);

        if !path.exists() {
            return None;
        }

        Some(BadPixelMap::load_from_file(&path))
    }

    /// Save a bad pixel map for a given camera.
    ///
    /// Creates the bad_pixel_maps directory if it doesn't exist.
    /// Returns the path where the map was saved.
    pub fn save_bad_pixel_map(&self, map: &BadPixelMap) -> std::io::Result<PathBuf> {
        let dir = self.bad_pixel_maps_dir();
        std::fs::create_dir_all(&dir)?;

        let path = self.bad_pixel_map_filename(&map.sensor_model, &map.camera_serial);
        map.save_to_file(&path)?;
        Ok(path)
    }

    /// List all bad pixel maps available in storage.
    ///
    /// Returns a list of (model, serial) pairs for which bad pixel maps exist.
    pub fn list_bad_pixel_maps(&self) -> std::io::Result<Vec<(String, String)>> {
        let dir = self.bad_pixel_maps_dir();

        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut maps = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some((model, serial)) = filename.split_once('-') {
                        let model_restored = model.replace('_', " ");
                        maps.push((model_restored, serial.to_string()));
                    }
                }
            }
        }

        Ok(maps)
    }

    /// Delete a bad pixel map for a given camera.
    ///
    /// Returns Ok(true) if the file was deleted, Ok(false) if it didn't exist.
    pub fn delete_bad_pixel_map(&self, model: &str, serial: &str) -> std::io::Result<bool> {
        let path = self.bad_pixel_map_filename(model, serial);

        if !path.exists() {
            return Ok(false);
        }

        std::fs::remove_file(path)?;
        Ok(true)
    }

    // =========================================================================
    // Optical Alignment
    // =========================================================================

    /// Get the optical alignment file path
    fn optical_alignment_path(&self) -> PathBuf {
        self.root_path.join("optical_alignment.json")
    }

    /// Get the optical alignment calibration.
    ///
    /// Returns None if no calibration exists.
    /// Returns Some(Err) if the file exists but cannot be loaded.
    pub fn get_optical_alignment(&self) -> Option<Result<OpticalAlignment, std::io::Error>> {
        let path = self.optical_alignment_path();

        if !path.exists() {
            return None;
        }

        Some(OpticalAlignment::load_from_file(&path))
    }

    /// Save the optical alignment calibration.
    ///
    /// Creates the config directory if it doesn't exist.
    /// Returns the path where the calibration was saved.
    pub fn save_optical_alignment(&self, alignment: &OpticalAlignment) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(&self.root_path)?;

        let path = self.optical_alignment_path();
        alignment.save_to_file(&path)?;
        Ok(path)
    }

    /// Delete the optical alignment calibration.
    ///
    /// Returns Ok(true) if the file was deleted, Ok(false) if it didn't exist.
    pub fn delete_optical_alignment(&self) -> std::io::Result<bool> {
        let path = self.optical_alignment_path();

        if !path.exists() {
            return Ok(false);
        }

        std::fs::remove_file(path)?;
        Ok(true)
    }
}

impl Default for ConfigStorage {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self::with_path(PathBuf::from(".cf_config")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn create_test_storage() -> ConfigStorage {
        let temp_dir = std::env::temp_dir().join(format!(
            "cf_config_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        ConfigStorage::with_path(temp_dir)
    }

    #[test]
    fn test_config_storage_creation() {
        let storage = create_test_storage();
        assert!(storage.root_path().to_str().is_some());
    }

    #[test]
    fn test_bad_pixel_map_filename() {
        let storage = create_test_storage();
        let path = storage.bad_pixel_map_filename("IMX455", "sn006");

        assert!(path.to_str().unwrap().contains("bad_pixel_maps"));
        assert!(path.to_str().unwrap().ends_with("IMX455-sn006.json"));
    }

    #[test]
    fn test_save_and_load_bad_pixel_map() {
        let storage = create_test_storage();

        let mut map = BadPixelMap::new("TestSensor".to_string(), "TEST001".to_string(), 1704067200);
        map.add_pixel(10, 20);
        map.add_pixel(30, 40);

        storage.save_bad_pixel_map(&map).unwrap();

        let loaded = storage
            .get_bad_pixel_map("TestSensor", "TEST001")
            .expect("Map should exist")
            .expect("Map should load successfully");

        assert_eq!(loaded.sensor_model, "TestSensor");
        assert_eq!(loaded.camera_serial, "TEST001");
        assert_eq!(loaded.num_bad_pixels(), 2);
        assert!(loaded.is_bad_pixel(10, 20));
        assert!(loaded.is_bad_pixel(30, 40));

        std::fs::remove_dir_all(storage.root_path()).ok();
    }

    #[test]
    fn test_get_nonexistent_map() {
        let storage = create_test_storage();
        let result = storage.get_bad_pixel_map("Nonexistent", "SN999");
        assert!(result.is_none());
    }

    #[test]
    fn test_list_bad_pixel_maps() {
        let storage = create_test_storage();

        let map1 = BadPixelMap::new("Sensor1".to_string(), "SN001".to_string(), 1704067200);
        let map2 = BadPixelMap::new("Sensor2".to_string(), "SN002".to_string(), 1704067200);

        storage.save_bad_pixel_map(&map1).unwrap();
        storage.save_bad_pixel_map(&map2).unwrap();

        let mut maps = storage.list_bad_pixel_maps().unwrap();
        maps.sort();

        assert_eq!(maps.len(), 2);
        assert!(maps.contains(&("Sensor1".to_string(), "SN001".to_string())));
        assert!(maps.contains(&("Sensor2".to_string(), "SN002".to_string())));

        std::fs::remove_dir_all(storage.root_path()).ok();
    }

    #[test]
    fn test_delete_bad_pixel_map() {
        let storage = create_test_storage();

        let map = BadPixelMap::new("TestSensor".to_string(), "TESTDEL".to_string(), 1704067200);
        storage.save_bad_pixel_map(&map).unwrap();

        assert!(storage.get_bad_pixel_map("TestSensor", "TESTDEL").is_some());

        let deleted = storage
            .delete_bad_pixel_map("TestSensor", "TESTDEL")
            .unwrap();
        assert!(deleted);

        assert!(storage.get_bad_pixel_map("TestSensor", "TESTDEL").is_none());

        let deleted_again = storage
            .delete_bad_pixel_map("TestSensor", "TESTDEL")
            .unwrap();
        assert!(!deleted_again);

        std::fs::remove_dir_all(storage.root_path()).ok();
    }

    #[test]
    fn test_save_and_load_optical_alignment() {
        let storage = create_test_storage();

        let alignment = OpticalAlignment::new(1.0, 0.1, -0.1, 1.0, 100.0, 200.0, 500);

        let path = storage.save_optical_alignment(&alignment).unwrap();
        assert!(path.exists());

        let loaded = storage
            .get_optical_alignment()
            .expect("Alignment should exist")
            .expect("Alignment should load successfully");

        assert!((loaded.a - 1.0).abs() < 1e-10);
        assert!((loaded.b - 0.1).abs() < 1e-10);
        assert!((loaded.tx - 100.0).abs() < 1e-10);
        assert_eq!(loaded.num_points, 500);

        std::fs::remove_dir_all(storage.root_path()).ok();
    }

    #[test]
    fn test_get_nonexistent_optical_alignment() {
        let storage = create_test_storage();
        let result = storage.get_optical_alignment();
        assert!(result.is_none());
    }

    #[test]
    fn test_delete_optical_alignment() {
        let storage = create_test_storage();

        let alignment = OpticalAlignment::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 100);
        storage.save_optical_alignment(&alignment).unwrap();

        assert!(storage.get_optical_alignment().is_some());

        let deleted = storage.delete_optical_alignment().unwrap();
        assert!(deleted);

        assert!(storage.get_optical_alignment().is_none());

        let deleted_again = storage.delete_optical_alignment().unwrap();
        assert!(!deleted_again);

        std::fs::remove_dir_all(storage.root_path()).ok();
    }
}

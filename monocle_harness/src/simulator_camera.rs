//! Simulator-backed camera implementation for testing
//!
//! Provides a CameraInterface implementation that uses the telescope simulator
//! to generate realistic frames for testing the monocle subsystem. This camera
//! simulates realistic star fields, sensor noise, and supports configurable
//! pointing motion profiles for testing tracking algorithms.

use crate::motion_profiles::PointingMotion;
use ndarray::Array2;
use shared::cached_star_catalog::CachedStarCatalog;
use shared::camera_interface::{
    AABBExt, CameraConfig, CameraError, CameraInterface, CameraResult, FrameMetadata,
    SensorGeometry, Timestamp,
};
use shared::image_proc::detection::AABB;
use shared::units::TemperatureExt;
use simulator::hardware::{SatelliteConfig, TelescopeConfig};
use simulator::image_proc::render::StarInFrame;
use simulator::photometry::zodiacal::SolarAngularCoordinates;
use simulator::scene::Scene;
use simulator::star_math::field_diameter;
use simulator::units::LengthExt;
use starfield::catalogs::binary_catalog::BinaryCatalog;
use starfield::Equatorial;
use std::sync::Arc;
use std::time::Duration;

/// Simulator-backed camera for testing FGS and tracking algorithms.
///
/// Combines a star catalog, telescope/sensor configuration, and pointing motion
/// to generate realistic frames for testing.
pub struct SimulatorCamera {
    /// Camera configuration
    config: CameraConfig,
    /// Satellite configuration (telescope + sensor)
    satellite: SatelliteConfig,
    /// Cached star catalog for efficient repeated queries
    cached_catalog: CachedStarCatalog<BinaryCatalog>,
    /// Pointing motion profile
    pointing_motion: Box<dyn PointingMotion>,
    /// Elapsed time since start
    elapsed_time: Duration,
    /// Current ROI
    roi: Option<AABB>,
    /// Current exposure duration
    exposure: Duration,
    /// Frame counter
    frame_number: u64,
    /// RNG seed for noise generation
    noise_seed: u64,
    /// Stars rendered in the last frame (with pixel positions)
    last_rendered_stars: Vec<StarInFrame>,
}

impl SimulatorCamera {
    /// Create a new simulator camera with a star catalog and pointing motion.
    /// Uses default exposure of 100ms and initializes cached star catalog.
    pub fn new(
        satellite: SatelliteConfig,
        catalog: Arc<BinaryCatalog>,
        pointing_motion: Box<dyn PointingMotion>,
    ) -> Self {
        let sensor = &satellite.sensor;
        let (width, height) = sensor.dimensions.get_pixel_width_height();

        let config = CameraConfig {
            width,
            height,
            exposure: Duration::from_millis(100),
            bit_depth: sensor.bit_depth,
        };

        // Calculate FOV for cache size
        let fov_diameter = field_diameter(&satellite.telescope, &satellite.sensor);

        // Create cached catalog
        let cached_catalog = CachedStarCatalog::new(catalog, fov_diameter);

        Self {
            config,
            satellite,
            cached_catalog,
            pointing_motion,
            elapsed_time: Duration::ZERO,
            roi: None,
            exposure: Duration::from_millis(100),
            frame_number: 0,
            noise_seed: 42,
            last_rendered_stars: Vec::new(),
        }
    }

    /// Create with a specific telescope and sensor configuration.
    /// Convenience method that builds a SatelliteConfig from individual components.
    pub fn with_config(
        telescope: TelescopeConfig,
        sensor: simulator::SensorConfig,
        temperature_c: f64,
        catalog: Arc<BinaryCatalog>,
        pointing_motion: Box<dyn PointingMotion>,
    ) -> Self {
        use simulator::units::{Temperature, TemperatureExt};
        let satellite =
            SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(temperature_c));

        Self::new(satellite, catalog, pointing_motion)
    }

    /// Get the satellite configuration.
    /// Returns the combined telescope, sensor, and temperature settings.
    pub fn satellite_config(&self) -> &SatelliteConfig {
        &self.satellite
    }

    /// Get the current pointing based on elapsed time and motion profile.
    /// The pointing changes over time according to the configured PointingMotion.
    pub fn pointing(&self) -> Equatorial {
        self.pointing_motion.get_pointing(self.elapsed_time)
    }

    /// Reset elapsed time to start.
    /// Useful for restarting a test sequence from the beginning.
    pub fn reset_time(&mut self) {
        self.elapsed_time = Duration::ZERO;
    }

    /// Set a new pointing motion profile.
    /// Replaces the current motion and resets elapsed time to zero.
    pub fn set_pointing_motion(&mut self, motion: Box<dyn PointingMotion>) {
        self.pointing_motion = motion;
        self.elapsed_time = Duration::ZERO;
    }

    /// Get the stars that were rendered in the last frame.
    /// Returns empty vector if no frame has been captured yet.
    /// Each star contains pixel position (x, y) and original catalog data.
    pub fn get_last_rendered_stars(&self) -> &[StarInFrame] {
        &self.last_rendered_stars
    }

    /// Find the closest rendered star to a given pixel position.
    /// Returns None if no stars were rendered or position is too far from any star.
    pub fn find_closest_star(&self, x: f64, y: f64, max_distance: f64) -> Option<&StarInFrame> {
        let mut min_distance = f64::MAX;
        let mut closest_star = None;

        for star in &self.last_rendered_stars {
            let dx = star.x - x;
            let dy = star.y - y;
            let distance = (dx * dx + dy * dy).sqrt();

            if distance < min_distance && distance <= max_distance {
                min_distance = distance;
                closest_star = Some(star);
            }
        }

        closest_star
    }

    /// Generate a frame using the simulator.
    /// Queries stars from cached catalog and renders them with realistic noise.
    fn generate_frame(&mut self) -> CameraResult<Array2<u16>> {
        let pointing = self.pointing();

        // Get stars using cached catalog
        let stars = self.cached_catalog.get_stars_in_fov(&pointing);

        // Use minimum zodiacal light for best visibility
        let solar_angles = SolarAngularCoordinates::zodiacal_minimum();

        // Create scene
        let scene = Scene::from_catalog(self.satellite.clone(), stars, pointing, solar_angles);

        // Generate frame
        let result = scene.render_with_seed(&self.exposure, Some(self.noise_seed));

        // Store rendered stars for later matching with detections
        self.last_rendered_stars = result.rendered_stars.clone();

        // Increment seed for next frame to get different noise
        self.noise_seed = self.noise_seed.wrapping_add(1);

        Ok(result.quantized_image)
    }
}

impl CameraInterface for SimulatorCamera {
    fn check_roi_size(&self, _width: usize, _height: usize) -> CameraResult<()> {
        Ok(())
    }

    /// Set a region of interest for frame capture.
    /// Returns error if ROI extends beyond sensor dimensions.
    fn set_roi(&mut self, roi: AABB) -> CameraResult<()> {
        roi.validate_for_sensor(self.config.width, self.config.height)?;
        self.roi = Some(roi);
        Ok(())
    }

    /// Clear the region of interest, returning to full frame capture.
    fn clear_roi(&mut self) -> CameraResult<()> {
        self.roi = None;
        Ok(())
    }

    /// Set the exposure duration for frame capture.
    /// Must be positive, returns error for zero duration.
    fn set_exposure(&mut self, exposure: Duration) -> CameraResult<()> {
        if exposure.is_zero() {
            return Err(CameraError::ConfigError(
                "Exposure time must be positive".to_string(),
            ));
        }
        self.exposure = exposure;
        Ok(())
    }

    /// Get the current exposure duration.
    fn get_exposure(&self) -> Duration {
        self.exposure
    }

    /// Get the camera configuration (width, height, exposure).
    fn get_config(&self) -> &CameraConfig {
        &self.config
    }

    /// Get sensor geometry from satellite configuration
    fn geometry(&self) -> SensorGeometry {
        SensorGeometry {
            width: self.config.width,
            height: self.config.height,
            pixel_size_microns: self.satellite.sensor.pixel_size().as_micrometers(),
        }
    }

    /// Check if camera is ready. Always returns true for simulator.
    fn is_ready(&self) -> bool {
        true // Always ready
    }

    /// Get the current region of interest if set.
    fn get_roi(&self) -> Option<AABB> {
        self.roi
    }

    fn stream(
        &mut self,
        callback: &mut dyn FnMut(&Array2<u16>, &FrameMetadata) -> bool,
    ) -> CameraResult<()> {
        loop {
            // Generate full frame
            let mut frame = self.generate_frame()?;

            // Apply ROI if set
            if let Some(roi) = &self.roi {
                // Clamp ROI to actual frame bounds to avoid panics
                let (height, width) = frame.dim();

                // Check if ROI is completely outside frame bounds
                if roi.min_row >= height || roi.min_col >= width {
                    // Return a small empty frame rather than error
                    self.frame_number += 1;
                    self.elapsed_time += self.exposure;

                    let mut temperatures = std::collections::HashMap::new();
                    temperatures.insert(
                        "sensor".to_string(),
                        self.satellite.temperature.as_celsius(),
                    );

                    let metadata = FrameMetadata {
                        frame_number: self.frame_number,
                        exposure: self.exposure,
                        timestamp: Timestamp::from_duration(self.elapsed_time),
                        pointing: Some(self.pointing()),
                        roi: self.roi,
                        temperatures,
                    };

                    let roi_height = roi.max_row - roi.min_row + 1;
                    let roi_width = roi.max_col - roi.min_col + 1;
                    let empty_frame = Array2::zeros((roi_height, roi_width));

                    if !callback(&empty_frame, &metadata) {
                        break;
                    }
                    continue;
                }

                // Clamp ROI to frame bounds
                let clamped_roi = AABB {
                    min_row: roi.min_row.min(height - 1),
                    min_col: roi.min_col.min(width - 1),
                    max_row: roi.max_row.min(height - 1),
                    max_col: roi.max_col.min(width - 1),
                };

                frame = clamped_roi.extract_from_frame(&frame.view());
            }

            self.frame_number += 1;

            let mut temperatures = std::collections::HashMap::new();
            temperatures.insert(
                "sensor".to_string(),
                self.satellite.temperature.as_celsius(),
            );

            let metadata = FrameMetadata {
                frame_number: self.frame_number,
                exposure: self.exposure,
                timestamp: Timestamp::from_duration(self.elapsed_time),
                pointing: Some(self.pointing()),
                roi: self.roi,
                temperatures,
            };

            // Increment elapsed time by exposure duration
            self.elapsed_time += self.exposure;

            if !callback(&frame, &metadata) {
                break;
            }
        }
        Ok(())
    }

    fn saturation_value(&self) -> f64 {
        self.satellite.sensor.saturating_reading()
    }

    fn name(&self) -> &str {
        "SimulatorCamera"
    }

    fn get_bit_depth(&self) -> u8 {
        self.config.bit_depth
    }

    fn set_bit_depth(&mut self, bit_depth: u8) -> CameraResult<()> {
        self.config.bit_depth = bit_depth;
        Ok(())
    }

    fn get_serial(&self) -> String {
        "SIM-00000".to_string()
    }

    fn get_gain(&self) -> f64 {
        0.0
    }

    fn set_gain(&mut self, _gain: f64) -> CameraResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::create_jbt_hwk_camera;

    fn create_test_camera() -> SimulatorCamera {
        create_jbt_hwk_camera()
    }

    #[test]
    fn test_camera_creation() {
        let camera = create_test_camera();
        assert_eq!(camera.get_exposure(), Duration::from_millis(100));
        assert!(camera.is_ready()); // Always ready now
                                    // Camera has default pointing at (0, 0)
        let pointing = camera.pointing();
        assert_eq!(pointing, Equatorial::from_degrees(0.0, 0.0));

        // Verify sensor dimensions are 512x512 (JBT/HWK test config)
        let config = camera.get_config();
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
    }

    #[test]
    fn test_pointing_and_roi() {
        let mut camera = create_test_camera();

        // Check default pointing
        let pointing = camera.pointing();
        assert!(camera.is_ready());
        assert_eq!(pointing, Equatorial::from_degrees(0.0, 0.0));

        // Set valid ROI
        let roi = AABB {
            min_row: 100,
            min_col: 100,
            max_row: 500,
            max_col: 500,
        };
        assert!(camera.set_roi(roi.clone()).is_ok());
        assert_eq!(camera.get_roi(), Some(roi));

        // Clear ROI
        assert!(camera.clear_roi().is_ok());
        assert!(camera.get_roi().is_none());
    }

    #[test]
    fn test_invalid_roi() {
        let mut camera = create_test_camera();
        let config = camera.get_config();

        // ROI beyond sensor bounds
        let roi = AABB {
            min_row: 0,
            min_col: 0,
            max_row: 100,
            max_col: config.width + 100,
        };
        assert!(camera.set_roi(roi).is_err());
    }

    #[test]
    fn test_exposure_setting() {
        let mut camera = create_test_camera();

        // Valid exposure
        assert!(camera.set_exposure(Duration::from_millis(200)).is_ok());
        assert_eq!(camera.get_exposure(), Duration::from_millis(200));

        // Invalid exposure (zero duration)
        assert!(camera.set_exposure(Duration::ZERO).is_err());
        assert_eq!(camera.get_exposure(), Duration::from_millis(200)); // Unchanged
    }

    #[test]
    fn test_timestamp_monotonicity_and_progression() {
        let mut camera = create_test_camera();

        // Set a specific exposure time
        let exposure = Duration::from_millis(50);
        camera
            .set_exposure(exposure)
            .expect("Failed to set exposure");

        // Capture first frame and get initial timestamp
        let (_, metadata1) = camera
            .capture_frame()
            .expect("Failed to capture first frame");
        let timestamp1 = metadata1.timestamp;

        // First frame should have timestamp at 0 (start of capture)
        assert_eq!(
            timestamp1.to_duration(),
            Duration::ZERO,
            "First frame should have timestamp of 0"
        );

        // Capture multiple frames and verify timestamps
        let mut prev_timestamp = timestamp1;
        for i in 2..=10 {
            let (_, metadata) = camera
                .capture_frame()
                .expect(&format!("Failed to capture frame {}", i));
            let current_timestamp = metadata.timestamp;

            // Verify monotonicity - timestamps should always increase
            assert!(
                current_timestamp.to_duration() > prev_timestamp.to_duration(),
                "Timestamp {} ({:?}) should be greater than previous ({:?})",
                i,
                current_timestamp,
                prev_timestamp
            );

            // Verify progression - Frame 1 is at t=0, Frame 2 is at t=exposure, etc.
            let expected_duration = exposure * (i - 1) as u32;
            assert_eq!(
                current_timestamp.to_duration(),
                expected_duration,
                "Frame {} timestamp should be exactly {} ms",
                i,
                expected_duration.as_millis()
            );

            // Verify the difference between consecutive timestamps
            let time_diff = current_timestamp.to_duration() - prev_timestamp.to_duration();
            assert_eq!(
                time_diff, exposure,
                "Time difference between frames should be exactly one exposure duration"
            );

            prev_timestamp = current_timestamp;
        }
    }

    #[test]
    fn test_frame_dimensions() {
        let mut camera = create_test_camera();

        // Test full frame capture
        let (full_frame, metadata) = camera
            .capture_frame()
            .expect("Failed to capture full frame");
        assert_eq!(
            full_frame.shape(),
            &[512, 512],
            "Full frame should be 512x512"
        );
        assert_eq!(
            metadata.roi, None,
            "Full frame should have no ROI in metadata"
        );

        // Test ROI capture with various sizes
        let test_rois = vec![
            // Small ROI
            AABB {
                min_row: 100,
                min_col: 100,
                max_row: 199,
                max_col: 199,
            },
            // Medium ROI
            AABB {
                min_row: 200,
                min_col: 200,
                max_row: 399,
                max_col: 399,
            },
            // Asymmetric ROI
            AABB {
                min_row: 50,
                min_col: 100,
                max_row: 249,
                max_col: 399,
            },
            // Edge-to-edge ROI (full width, partial height)
            AABB {
                min_row: 100,
                min_col: 0,
                max_row: 299,
                max_col: 511,
            },
        ];

        for roi in test_rois {
            let expected_height = (roi.max_row - roi.min_row + 1) as usize;
            let expected_width = (roi.max_col - roi.min_col + 1) as usize;

            camera.set_roi(roi.clone()).expect("Failed to set ROI");
            let (roi_frame, metadata) =
                camera.capture_frame().expect("Failed to capture ROI frame");

            assert_eq!(
                roi_frame.shape(),
                &[expected_height, expected_width],
                "ROI frame dimensions mismatch for ROI {:?}",
                roi
            );
            assert_eq!(
                metadata.roi,
                Some(roi.clone()),
                "ROI metadata should match the set ROI"
            );
        }

        // Test clearing ROI returns to full frame
        camera.clear_roi().expect("Failed to clear ROI");
        let (cleared_frame, metadata) = camera
            .capture_frame()
            .expect("Failed to capture after clearing ROI");
        assert_eq!(
            cleared_frame.shape(),
            &[512, 512],
            "Should return to full frame after clearing ROI"
        );
        assert_eq!(
            metadata.roi, None,
            "Metadata should have no ROI after clearing"
        );
    }

    #[test]
    fn test_roi_with_star_tracking() {
        // This test simulates what FGS does: detect a star, set ROI around it, capture frames
        let mut camera = create_test_camera();

        // Also test with 256x256 sensor like in failing tests
        test_roi_tracking_for_camera(&mut camera, 512, "512x512 camera");

        // Create 256x256 camera like in runner tests
        use crate::motion_profiles::StaticPointing;
        use simulator::hardware::{SatelliteConfig, TelescopeConfig};
        use simulator::units::{Length, LengthExt, Temperature, TemperatureExt};
        use starfield::catalogs::binary_catalog::{BinaryCatalog, MinimalStar};

        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.3),
            Length::from_meters(3.0), // f/10 telescope
            0.9,
        );
        let sensor = simulator::hardware::sensor::models::IMX455
            .clone()
            .with_dimensions(256, 256);
        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(0.0));

        // Same catalog as runner tests
        let catalog = Arc::new(BinaryCatalog::from_stars(
            vec![
                MinimalStar::new(1, 0.0, 0.0, 5.0),
                MinimalStar::new(2, 0.01, 0.0, 6.0),
                MinimalStar::new(3, 0.0, 0.01, 7.0),
            ],
            "Test catalog",
        ));
        let motion = Box::new(StaticPointing::new(0.0, 0.0));
        let mut camera256 = SimulatorCamera::new(satellite, catalog, motion);

        test_roi_tracking_for_camera(&mut camera256, 256, "256x256 camera");
    }

    fn test_roi_tracking_for_camera(camera: &mut SimulatorCamera, size: usize, label: &str) {
        println!("\nTesting ROI tracking for {}", label);

        // First capture full frame
        let (frame1, _) = camera
            .capture_frame()
            .expect("Failed to capture initial frame");
        assert_eq!(frame1.shape(), &[size, size]);

        // Find brightest pixel (simulating star detection)
        let mut max_val = 0u16;
        let mut max_pos = (0, 0);
        for ((row, col), &val) in frame1.indexed_iter() {
            if val > max_val {
                max_val = val;
                max_pos = (row, col);
            }
        }

        println!(
            "Brightest pixel at ({}, {}) with value {}",
            max_pos.0, max_pos.1, max_val
        );

        // Set ROI around brightest pixel (32x32 box like FGS does)
        let roi_size = 32;
        let roi = AABB {
            min_row: max_pos.0.saturating_sub(roi_size / 2),
            min_col: max_pos.1.saturating_sub(roi_size / 2),
            max_row: (max_pos.0 + roi_size / 2).min(size - 1),
            max_col: (max_pos.1 + roi_size / 2).min(size - 1),
        };

        println!("Setting ROI: {:?}", roi);
        camera.set_roi(roi.clone()).expect("Failed to set ROI");

        // Capture multiple frames with ROI
        for i in 0..5 {
            let result = camera.capture_frame();
            match result {
                Ok((roi_frame, metadata)) => {
                    println!("Frame {} captured with shape {:?}", i, roi_frame.shape());
                    assert_eq!(metadata.roi, Some(roi), "ROI should be in metadata");

                    // Check dimensions match expected ROI size
                    let expected_height = (roi.max_row - roi.min_row + 1) as usize;
                    let expected_width = (roi.max_col - roi.min_col + 1) as usize;

                    // Could be smaller if ROI was clamped or out of bounds
                    assert!(roi_frame.shape()[0] <= expected_height);
                    assert!(roi_frame.shape()[1] <= expected_width);
                }
                Err(e) => {
                    panic!("Frame {} capture failed: {}", i, e);
                }
            }
        }
    }

    #[test]
    fn test_subsequent_frames_different() {
        let mut camera = create_test_camera();

        // Capture multiple frames
        let (frame1, metadata1) = camera.capture_frame().expect("Failed to capture frame 1");
        let (frame2, metadata2) = camera.capture_frame().expect("Failed to capture frame 2");
        let (frame3, metadata3) = camera.capture_frame().expect("Failed to capture frame 3");

        // Verify frame numbers increment
        assert_eq!(metadata1.frame_number + 1, metadata2.frame_number);
        assert_eq!(metadata2.frame_number + 1, metadata3.frame_number);

        // Verify frames have different noise patterns
        // We compare the frames statistically rather than pixel-by-pixel
        // since they should have the same stars but different noise

        // Calculate sum of absolute differences
        let diff_1_2: f64 = frame1
            .iter()
            .zip(frame2.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).abs() as f64)
            .sum();

        let diff_2_3: f64 = frame2
            .iter()
            .zip(frame3.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).abs() as f64)
            .sum();

        // There should be substantial differences due to noise
        let total_pixels = (512 * 512) as f64;
        let avg_diff_1_2 = diff_1_2 / total_pixels;
        let avg_diff_2_3 = diff_2_3 / total_pixels;

        assert!(
            avg_diff_1_2 > 0.1,
            "Frames 1 and 2 should differ (avg diff: {})",
            avg_diff_1_2
        );
        assert!(
            avg_diff_2_3 > 0.1,
            "Frames 2 and 3 should differ (avg diff: {})",
            avg_diff_2_3
        );

        // Also test with ROI to ensure noise changes there too
        let roi = AABB {
            min_row: 200,
            min_col: 200,
            max_row: 399,
            max_col: 399,
        };
        camera.set_roi(roi).expect("Failed to set ROI");

        let (roi_frame1, _) = camera
            .capture_frame()
            .expect("Failed to capture ROI frame 1");
        let (roi_frame2, _) = camera
            .capture_frame()
            .expect("Failed to capture ROI frame 2");

        let roi_diff: f64 = roi_frame1
            .iter()
            .zip(roi_frame2.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).abs() as f64)
            .sum();

        let roi_pixels = (200 * 200) as f64;
        let avg_roi_diff = roi_diff / roi_pixels;

        assert!(
            avg_roi_diff > 0.1,
            "ROI frames should also differ (avg diff: {})",
            avg_roi_diff
        );
    }
}

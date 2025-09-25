//! Simulator-backed camera implementation for testing
//!
//! Provides a CameraInterface implementation that uses the telescope simulator
//! to generate realistic frames for testing the monocle subsystem.

use ndarray::Array2;
use shared::camera_interface::{
    AABBExt, CameraConfig, CameraError, CameraInterface, CameraResult, FrameMetadata,
};
use shared::image_proc::detection::AABB;
use shared::units::TemperatureExt;
use simulator::hardware::{SatelliteConfig, TelescopeConfig};
use simulator::photometry::zodiacal::SolarAngularCoordinates;
use simulator::scene::Scene;
use starfield::catalogs::binary_catalog::BinaryCatalog;
use starfield::catalogs::StarData;
use starfield::Equatorial;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Default pointing direction (RA=0, Dec=0)
const DEFAULT_POINTING: Equatorial = Equatorial { ra: 0.0, dec: 0.0 };

/// Camera state for continuous capture mode
struct ContinuousCaptureState {
    is_active: bool,
    latest_frame: Option<(Array2<u16>, FrameMetadata)>,
    frame_counter: u64,
}

/// Simulator-backed camera for testing
pub struct SimulatorCamera {
    /// Camera configuration
    config: CameraConfig,
    /// Satellite configuration (telescope + sensor)
    satellite: SatelliteConfig,
    /// Star catalog
    catalog: Option<BinaryCatalog>,
    /// Current pointing (required)
    pointing: Equatorial,
    /// Current ROI
    roi: Option<AABB>,
    /// Current exposure duration
    exposure: Duration,
    /// Frame counter
    frame_number: u64,
    /// RNG seed for noise generation
    noise_seed: u64,
    /// Continuous capture state
    continuous_state: Arc<Mutex<ContinuousCaptureState>>,
}

impl SimulatorCamera {
    /// Create a new simulator camera with a star catalog (default pointing at RA=0, Dec=0)
    pub fn new(satellite: SatelliteConfig, catalog: BinaryCatalog) -> Self {
        let sensor = &satellite.sensor;
        let (width, height) = sensor.dimensions.get_pixel_width_height();

        let config = CameraConfig {
            width,
            height,
            exposure: Duration::from_millis(100),
        };

        Self {
            config,
            satellite,
            catalog: Some(catalog),
            pointing: DEFAULT_POINTING,
            roi: None,
            exposure: Duration::from_millis(100),
            frame_number: 0,
            noise_seed: 42,
            continuous_state: Arc::new(Mutex::new(ContinuousCaptureState {
                is_active: false,
                latest_frame: None,
                frame_counter: 0,
            })),
        }
    }

    /// Create with a specific telescope and sensor configuration
    pub fn with_config(
        telescope: TelescopeConfig,
        sensor: simulator::SensorConfig,
        temperature_c: f64,
        catalog: BinaryCatalog,
    ) -> Self {
        use simulator::units::{Temperature, TemperatureExt};
        let satellite =
            SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(temperature_c));

        Self::new(satellite, catalog)
    }

    /// Get the satellite configuration
    pub fn satellite_config(&self) -> &SatelliteConfig {
        &self.satellite
    }

    /// Get the current pointing
    pub fn pointing(&self) -> Equatorial {
        self.pointing
    }

    /// Set the pointing (convenience method that returns Result)
    pub fn set_pointing(&mut self, pointing: Equatorial) -> CameraResult<()> {
        self.pointing = pointing;
        Ok(())
    }

    /// Generate a frame using the simulator
    fn generate_frame(&mut self) -> CameraResult<Array2<u16>> {
        let pointing = self.pointing;

        let catalog = self
            .catalog
            .as_ref()
            .ok_or_else(|| CameraError::CaptureError("No catalog loaded".to_string()))?;

        // Get stars from catalog and convert to StarData
        let stars: Vec<StarData> = catalog
            .stars()
            .iter()
            .map(|s| {
                StarData::new(
                    s.id,
                    s.position.ra.to_degrees(),
                    s.position.dec.to_degrees(),
                    s.magnitude,
                    None,
                )
            })
            .collect();

        // Use minimum zodiacal light for best visibility
        let solar_angles = SolarAngularCoordinates::zodiacal_minimum();

        // Create scene
        let scene = Scene::from_catalog(self.satellite.clone(), stars, pointing, solar_angles);

        // Generate frame
        let result = scene.render_with_seed(&self.exposure, Some(self.noise_seed));

        // Increment seed for next frame to get different noise
        self.noise_seed = self.noise_seed.wrapping_add(1);

        Ok(result.quantized_image)
    }
}

impl CameraInterface for SimulatorCamera {
    fn set_roi(&mut self, roi: AABB) -> CameraResult<()> {
        roi.validate_for_sensor(self.config.width, self.config.height)?;
        self.roi = Some(roi);
        Ok(())
    }

    fn clear_roi(&mut self) -> CameraResult<()> {
        self.roi = None;
        Ok(())
    }

    fn capture_frame(&mut self) -> CameraResult<(Array2<u16>, FrameMetadata)> {
        // Generate full frame
        let mut frame = self.generate_frame()?;

        // Apply ROI if set
        if let Some(roi) = &self.roi {
            frame = roi.extract_from_frame(&frame.view());
        }

        self.frame_number += 1;

        let metadata = FrameMetadata {
            frame_number: self.frame_number,
            exposure: self.exposure,
            timestamp: SystemTime::now(),
            pointing: Some(self.pointing),
            roi: self.roi,
            temperature_c: self.satellite.temperature.as_celsius(),
        };

        Ok((frame, metadata))
    }

    fn set_exposure(&mut self, exposure: Duration) -> CameraResult<()> {
        if exposure.is_zero() {
            return Err(CameraError::ConfigError(
                "Exposure time must be positive".to_string(),
            ));
        }
        self.exposure = exposure;
        Ok(())
    }

    fn get_exposure(&self) -> Duration {
        self.exposure
    }

    fn get_config(&self) -> &CameraConfig {
        &self.config
    }

    fn is_ready(&self) -> bool {
        true // Always ready
    }

    fn get_roi(&self) -> Option<AABB> {
        self.roi
    }

    fn start_continuous_capture(&mut self) -> CameraResult<()> {
        let mut state = self.continuous_state.lock().unwrap();
        state.is_active = true;
        state.frame_counter = 0;
        Ok(())
    }

    fn stop_continuous_capture(&mut self) -> CameraResult<()> {
        let mut state = self.continuous_state.lock().unwrap();
        state.is_active = false;
        state.latest_frame = None;
        Ok(())
    }

    fn get_latest_frame(&mut self) -> Option<(Array2<u16>, FrameMetadata)> {
        // Check if capturing is active first
        let is_active = {
            let state = self.continuous_state.lock().unwrap();
            state.is_active
        }; // Lock is dropped here

        if !is_active {
            return None;
        }

        // In real hardware this would check for new frames
        // For simulator, we generate a new frame each call
        match self.capture_frame() {
            Ok((frame, metadata)) => {
                // Update state after capture
                let mut state = self.continuous_state.lock().unwrap();
                state.frame_counter += 1;
                state.latest_frame = Some((frame.clone(), metadata.clone()));
                Some((frame, metadata))
            }
            Err(_) => None,
        }
    }

    fn is_capturing(&self) -> bool {
        self.continuous_state.lock().unwrap().is_active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simulator::hardware::sensor::models as sensor_models;
    use simulator::units::{Length, LengthExt};
    use starfield::catalogs::binary_catalog::MinimalStar;

    fn create_test_catalog() -> BinaryCatalog {
        // Create a simple test catalog with a few stars
        let stars = vec![
            MinimalStar::new(1, 83.0, -5.0, 6.0),
            MinimalStar::new(2, 83.5, -5.5, 7.0),
            MinimalStar::new(3, 84.0, -5.0, 8.0),
        ];
        BinaryCatalog::from_stars(stars, "Test catalog")
    }

    fn create_test_camera() -> SimulatorCamera {
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_millimeters(80.0),
            Length::from_millimeters(400.0),
            0.9,
        );
        // Create a 1024x1024 sensor based on IMX455 characteristics
        let sensor = sensor_models::IMX455.clone().with_dimensions(1024, 1024);
        let catalog = create_test_catalog();
        let pointing = Equatorial::from_degrees(83.0, -5.0);

        SimulatorCamera::with_config(telescope, sensor, 0.0, catalog)
    }

    #[test]
    fn test_camera_creation() {
        let camera = create_test_camera();
        assert_eq!(camera.get_exposure(), Duration::from_millis(100));
        assert!(camera.is_ready()); // Always ready now
                                    // Camera has default pointing at (0, 0)
        assert_eq!(camera.pointing(), DEFAULT_POINTING);

        // Verify sensor dimensions are 1024x1024
        let config = camera.get_config();
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 1024);
    }

    #[test]
    fn test_pointing_and_roi() {
        let mut camera = create_test_camera();

        // Set pointing
        let pointing = Equatorial::from_degrees(0.0, 0.0);
        assert!(camera.set_pointing(pointing).is_ok());
        assert!(camera.is_ready());
        assert_eq!(camera.pointing(), pointing);

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
    fn test_continuous_capture_state() {
        let mut camera = create_test_camera();

        assert!(!camera.is_capturing());

        assert!(camera.start_continuous_capture().is_ok());
        assert!(camera.is_capturing());

        assert!(camera.stop_continuous_capture().is_ok());
        assert!(!camera.is_capturing());
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
            &[1024, 1024],
            "Full frame should be 1024x1024"
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
                max_row: 599,
                max_col: 599,
            },
            // Asymmetric ROI
            AABB {
                min_row: 50,
                min_col: 100,
                max_row: 249,
                max_col: 899,
            },
            // Edge-to-edge ROI (full width, partial height)
            AABB {
                min_row: 100,
                min_col: 0,
                max_row: 299,
                max_col: 1023,
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
            &[1024, 1024],
            "Should return to full frame after clearing ROI"
        );
        assert_eq!(
            metadata.roi, None,
            "Metadata should have no ROI after clearing"
        );
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
        let total_pixels = (1024 * 1024) as f64;
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
            min_row: 400,
            min_col: 400,
            max_row: 599,
            max_col: 599,
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

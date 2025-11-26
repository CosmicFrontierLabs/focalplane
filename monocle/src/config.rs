use serde::{Deserialize, Serialize};
use shared::bad_pixel_map::BadPixelMap;

/// Guide star filtering criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuideStarFilters {
    /// Detection threshold multiplier for noise level (e.g., 5.0 for 5-sigma)
    pub detection_threshold_sigma: f64,
    /// Minimum SNR for guide star selection
    pub snr_min: f64,
    /// Star diameter range in pixels (min, max)
    pub diameter_range: (f64, f64),
    /// Maximum aspect ratio for star shape
    pub aspect_ratio_max: f64,
    /// Pixel saturation value for sensor (DN)
    pub saturation_value: f64,
    /// Radius in pixels to search for saturated pixels
    pub saturation_search_radius: f64,
    /// Minimum distance from image edge in pixels
    pub minimum_edge_distance: f64,
    /// Bad pixel map for filtering sources near defective pixels
    pub bad_pixel_map: BadPixelMap,
    /// Minimum distance from bad pixels in pixels
    pub minimum_bad_pixel_distance: f64,
}

/// Configuration for the Fine Guidance System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FgsConfig {
    /// Number of frames to average during acquisition
    pub acquisition_frames: usize,
    /// Guide star filtering criteria
    pub filters: GuideStarFilters,
    /// ROI size around each guide star (pixels)
    pub roi_size: usize,
    /// Maximum reacquisition attempts before recalibration
    pub max_reacquisition_attempts: usize,
    /// Multiplier for FWHM to determine centroid computation radius (e.g., 5.0 for 5x FWHM)
    pub centroid_radius_multiplier: f64,
    /// FWHM measure in pixels
    pub fwhm: f64,
    /// Minimum SNR for tracking - tracking stops if SNR drops below this threshold
    pub snr_dropout_threshold: f64,
}

impl FgsConfig {
    /// Validate configuration parameters
    ///
    /// Checks that configuration values are internally consistent, particularly
    /// that minimum_edge_distance is large enough to accommodate the roi_size.
    /// If a camera is provided, also validates ROI size against camera-specific
    /// hardware constraints.
    ///
    /// # Arguments
    /// * `camera` - Optional camera reference to check hardware-specific ROI constraints
    ///
    /// # Returns
    /// * `Ok(())` if configuration is valid
    /// * `Err(String)` with details if configuration is invalid
    pub fn validate(
        &self,
        camera: Option<&dyn shared::camera_interface::CameraInterface>,
    ) -> Result<(), String> {
        let roi_half = self.roi_size as f64 / 2.0;

        if self.filters.minimum_edge_distance <= roi_half {
            return Err(format!(
                "minimum_edge_distance ({:.1}) must be greater than roi_size/2 ({:.1}) to ensure ROI never extends beyond image boundaries",
                self.filters.minimum_edge_distance,
                roi_half
            ));
        }

        if self.roi_size == 0 {
            return Err("roi_size must be greater than 0".to_string());
        }

        if self.acquisition_frames == 0 {
            return Err("acquisition_frames must be greater than 0".to_string());
        }

        // Validate against camera-specific ROI constraints if camera provided
        if let Some(cam) = camera {
            cam.check_roi_size(shared::image_size::PixelShape::with_width_height(
                self.roi_size,
                self.roi_size,
            ))
            .map_err(|e| format!("ROI size validation failed: {e}"))?;
        }

        Ok(())
    }
}

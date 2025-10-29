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
}

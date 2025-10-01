use serde::{Deserialize, Serialize};

/// Centroid computation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CentroidMethod {
    /// Intensity-weighted center of mass
    CenterOfMass,
    /// Gaussian PSF fitting
    GaussianFit,
    /// Quadratic interpolation
    QuadraticInterpolation,
}

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
}

impl Default for GuideStarFilters {
    fn default() -> Self {
        Self {
            detection_threshold_sigma: 5.0,
            snr_min: 20.0,
            diameter_range: (2.0, 20.0),
            aspect_ratio_max: 2.5,
            saturation_value: 4000.0,
            saturation_search_radius: 3.0,
            minimum_edge_distance: 10.0,
        }
    }
}

/// Configuration for the Fine Guidance System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FgsConfig {
    /// Number of frames to average during acquisition
    pub acquisition_frames: usize,
    /// Guide star filtering criteria
    pub filters: GuideStarFilters,
    /// Maximum number of guide stars to track
    pub max_guide_stars: usize,
    /// ROI size around each guide star (pixels)
    pub roi_size: usize,
    /// Maximum reacquisition attempts before recalibration
    pub max_reacquisition_attempts: usize,
    /// Centroid computation method
    pub centroid_method: CentroidMethod,
    /// Multiplier for FWHM to determine centroid computation radius (e.g., 5.0 for 5x FWHM)
    pub centroid_radius_multiplier: f64,
}

impl Default for FgsConfig {
    fn default() -> Self {
        Self {
            acquisition_frames: 10,
            filters: GuideStarFilters::default(),
            max_guide_stars: 3,
            roi_size: 64,
            max_reacquisition_attempts: 5,
            centroid_method: CentroidMethod::CenterOfMass,
            centroid_radius_multiplier: 5.0,
        }
    }
}

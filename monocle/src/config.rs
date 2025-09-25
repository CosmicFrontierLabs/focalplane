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

/// Configuration for the Fine Guidance System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FgsConfig {
    /// Number of frames to average during acquisition
    pub acquisition_frames: usize,
    /// Minimum SNR for guide star selection
    pub min_guide_star_snr: f64,
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
    /// Detection threshold multiplier for noise level (e.g., 5.0 for 5-sigma)
    pub detection_threshold_sigma: f64,
}

impl Default for FgsConfig {
    fn default() -> Self {
        Self {
            acquisition_frames: 10,
            min_guide_star_snr: 20.0,
            max_guide_stars: 3,
            roi_size: 64,
            max_reacquisition_attempts: 5,
            centroid_method: CentroidMethod::CenterOfMass,
            centroid_radius_multiplier: 5.0,
            detection_threshold_sigma: 5.0,
        }
    }
}

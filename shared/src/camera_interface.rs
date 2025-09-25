//! Camera abstraction layer for monocle testing infrastructure
//!
//! Provides a unified interface for camera operations that can be backed by
//! either the simulator (for testing) or actual hardware (for production).

use crate::image_proc::detection::AABB;
use ndarray::{Array2, ArrayView2};
use starfield::Equatorial;
use std::error::Error;
use std::fmt;
use std::time::Duration;

/// Error type for camera operations
#[derive(Debug)]
pub enum CameraError {
    /// Hardware communication error
    HardwareError(String),
    /// Invalid region of interest
    InvalidROI(String),
    /// Pointing error
    PointingError(String),
    /// Frame capture error
    CaptureError(String),
    /// Configuration error
    ConfigError(String),
}

impl fmt::Display for CameraError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CameraError::HardwareError(msg) => write!(f, "Hardware error: {msg}"),
            CameraError::InvalidROI(msg) => write!(f, "Invalid ROI: {msg}"),
            CameraError::PointingError(msg) => write!(f, "Pointing error: {msg}"),
            CameraError::CaptureError(msg) => write!(f, "Capture error: {msg}"),
            CameraError::ConfigError(msg) => write!(f, "Configuration error: {msg}"),
        }
    }
}

impl Error for CameraError {}

/// Result type for camera operations
pub type CameraResult<T> = Result<T, CameraError>;

/// Camera metadata returned with each frame
#[derive(Debug, Clone)]
pub struct FrameMetadata {
    /// Frame sequence number
    pub frame_number: u64,
    /// Exposure duration
    pub exposure: Duration,
    /// Timestamp when frame was captured
    pub timestamp: std::time::SystemTime,
    /// Current telescope pointing
    pub pointing: Option<Equatorial>,
    /// Current ROI if set
    pub roi: Option<AABB>,
    /// Sensor temperature in Celsius
    pub temperature_c: f64,
}

/// Configuration for camera initialization
#[derive(Debug, Clone)]
pub struct CameraConfig {
    /// Sensor width in pixels
    pub width: usize,
    /// Sensor height in pixels
    pub height: usize,
    /// Exposure duration
    pub exposure: Duration,
}

/// Trait for unified camera interface
///
/// This trait abstracts camera operations to allow testing with the simulator
/// and deployment with real hardware using the same interface.
pub trait CameraInterface: Send + Sync {
    /// Set region of interest for readout
    ///
    /// # Arguments
    /// * `roi` - Axis-aligned bounding box defining the ROI
    ///
    /// # Returns
    /// * `Ok(())` on successful ROI configuration
    /// * `Err(CameraError)` if ROI is invalid or unsupported
    fn set_roi(&mut self, roi: AABB) -> CameraResult<()>;

    /// Clear ROI and return to full-frame readout
    fn clear_roi(&mut self) -> CameraResult<()>;

    /// Capture a single frame
    ///
    /// # Returns
    /// * `Ok((frame, metadata))` containing the image data and metadata
    /// * `Err(CameraError)` if capture fails
    fn capture_frame(&mut self) -> CameraResult<(Array2<u16>, FrameMetadata)>;

    /// Set exposure time
    ///
    /// # Arguments
    /// * `exposure` - Exposure duration
    fn set_exposure(&mut self, exposure: Duration) -> CameraResult<()>;

    /// Get current exposure duration
    fn get_exposure(&self) -> Duration;

    /// Get camera configuration
    fn get_config(&self) -> &CameraConfig;

    /// Check if camera is ready to capture
    fn is_ready(&self) -> bool;

    /// Get current ROI (if set)
    fn get_roi(&self) -> Option<AABB>;

    /// Start continuous capture mode
    ///
    /// In this mode, the camera continuously captures frames that can be
    /// retrieved with `get_latest_frame()`. This is useful for live tracking.
    fn start_continuous_capture(&mut self) -> CameraResult<()>;

    /// Stop continuous capture mode
    fn stop_continuous_capture(&mut self) -> CameraResult<()>;

    /// Get the latest frame from continuous capture
    ///
    /// Returns None if no new frame is available since last call
    fn get_latest_frame(&mut self) -> Option<(Array2<u16>, FrameMetadata)>;

    /// Check if camera is in continuous capture mode
    fn is_capturing(&self) -> bool;
}

/// Helper functions for working with ROIs
pub trait AABBExt {
    /// Validate that ROI fits within sensor dimensions
    fn validate_for_sensor(&self, width: usize, height: usize) -> CameraResult<()>;

    /// Extract ROI from full frame
    fn extract_from_frame(&self, frame: &ArrayView2<u16>) -> Array2<u16>;
}

impl AABBExt for AABB {
    fn validate_for_sensor(&self, width: usize, height: usize) -> CameraResult<()> {
        if self.min_col >= width || self.min_row >= height {
            return Err(CameraError::InvalidROI(format!(
                "ROI starts beyond sensor bounds ({width}x{height})"
            )));
        }

        if self.max_col >= width || self.max_row >= height {
            return Err(CameraError::InvalidROI(format!(
                "ROI extends beyond sensor bounds ({width}x{height})"
            )));
        }

        if self.width() == 0 || self.height() == 0 {
            return Err(CameraError::InvalidROI(
                "ROI has zero dimensions".to_string(),
            ));
        }

        Ok(())
    }

    fn extract_from_frame(&self, frame: &ArrayView2<u16>) -> Array2<u16> {
        frame
            .slice(ndarray::s![
                self.min_row..=self.max_row,
                self.min_col..=self.max_col
            ])
            .to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roi_validation() {
        // Valid ROI
        let roi = AABB {
            min_col: 10,
            min_row: 10,
            max_col: 100,
            max_row: 100,
        };
        assert!(roi.validate_for_sensor(200, 200).is_ok());

        // ROI extends beyond sensor
        let roi = AABB {
            min_col: 10,
            min_row: 10,
            max_col: 250,
            max_row: 100,
        };
        assert!(roi.validate_for_sensor(200, 200).is_err());

        // ROI starts beyond sensor
        let roi = AABB {
            min_col: 201,
            min_row: 10,
            max_col: 250,
            max_row: 100,
        };
        assert!(roi.validate_for_sensor(200, 200).is_err());
    }

    #[test]
    fn test_roi_extraction() {
        let mut frame = Array2::<u16>::zeros((100, 100));

        // Mark a region
        for i in 20..30 {
            for j in 20..30 {
                frame[[i, j]] = 1000;
            }
        }

        let roi = AABB {
            min_col: 20,
            min_row: 20,
            max_col: 29,
            max_row: 29,
        };

        let extracted = roi.extract_from_frame(&frame.view());
        assert_eq!(extracted.shape(), &[10, 10]);
        assert_eq!(extracted[[0, 0]], 1000);
        assert_eq!(extracted[[9, 9]], 1000);
    }
}

//! Camera abstraction layer for monocle testing infrastructure
//!
//! Provides a unified interface for camera operations that can be backed by
//! either the simulator (for testing) or actual hardware (for production).

use crate::image_proc::detection::AABB;
use ndarray::{Array2, ArrayView2};
use starfield::Equatorial;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::time::Duration;

/// Timestamp structure aligned with V4L2 format
/// Represents time as seconds and nanoseconds since an epoch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timestamp {
    /// Seconds component
    pub seconds: u64,
    /// Nanoseconds component (0-999,999,999)
    pub nanos: u64,
}

impl Timestamp {
    /// Create a new timestamp
    pub fn new(seconds: u64, nanos: u64) -> Self {
        Self { seconds, nanos }
    }

    /// Create a timestamp from a Duration since epoch
    pub fn from_duration(duration: Duration) -> Self {
        let total_nanos = duration.as_nanos();
        let seconds = (total_nanos / 1_000_000_000) as u64;
        let nanos = (total_nanos % 1_000_000_000) as u64;
        Self { seconds, nanos }
    }

    /// Convert to Duration
    pub fn to_duration(&self) -> Duration {
        Duration::new(self.seconds, self.nanos as u32)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{:09}", self.seconds, self.nanos)
    }
}

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
    /// Values are intended to align with V4L2 timestamp format and expectations
    pub timestamp: Timestamp,
    /// Current telescope pointing
    pub pointing: Option<Equatorial>,
    /// Current ROI if set
    pub roi: Option<AABB>,
    /// Temperature readings from various sensors (e.g., "sensor", "fpga", "pcb")
    /// Key is the sensor location/name, value is temperature in Celsius
    pub temperatures: HashMap<String, f64>,
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

    /// Get sensor saturation value in DN
    ///
    /// Returns the pixel value at which the sensor saturates. This is used
    /// for guide star filtering to reject saturated sources.
    fn saturation_value(&self) -> f64;

    /// Get camera name/identifier
    ///
    /// Returns a human-readable name for this camera instance
    fn name(&self) -> &str;
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
    fn test_timestamp_creation() {
        let ts = Timestamp::new(100, 500_000_000);
        assert_eq!(ts.seconds, 100);
        assert_eq!(ts.nanos, 500_000_000);
    }

    #[test]
    fn test_timestamp_from_duration() {
        // Test exact second
        let duration = Duration::from_secs(5);
        let ts = Timestamp::from_duration(duration);
        assert_eq!(ts.seconds, 5);
        assert_eq!(ts.nanos, 0);

        // Test with nanoseconds
        let duration = Duration::new(10, 123_456_789);
        let ts = Timestamp::from_duration(duration);
        assert_eq!(ts.seconds, 10);
        assert_eq!(ts.nanos, 123_456_789);

        // Test large duration
        let duration = Duration::from_millis(1500);
        let ts = Timestamp::from_duration(duration);
        assert_eq!(ts.seconds, 1);
        assert_eq!(ts.nanos, 500_000_000);
    }

    #[test]
    fn test_timestamp_to_duration() {
        let ts = Timestamp::new(42, 123_456_789);
        let duration = ts.to_duration();
        assert_eq!(duration.as_secs(), 42);
        assert_eq!(duration.subsec_nanos(), 123_456_789);
    }

    #[test]
    fn test_timestamp_roundtrip() {
        let original = Duration::new(100, 999_999_999);
        let ts = Timestamp::from_duration(original);
        let recovered = ts.to_duration();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_timestamp_display() {
        let ts = Timestamp::new(42, 123_456_789);
        assert_eq!(ts.to_string(), "42.123456789");

        let ts = Timestamp::new(0, 1);
        assert_eq!(ts.to_string(), "0.000000001");

        let ts = Timestamp::new(100, 0);
        assert_eq!(ts.to_string(), "100.000000000");
    }

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

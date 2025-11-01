//! Camera abstraction layer for monocle testing infrastructure
//!
//! Provides a unified interface for camera operations that can be backed by
//! either the simulator (for testing) or actual hardware (for production).

pub mod mock;
pub mod ring_buffer;

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
    /// ADC bit depth (8, 12, 14, 16 bits typical)
    pub bit_depth: u8,
}

impl CameraConfig {
    /// Get saturation value in DN (Digital Numbers) based on bit depth
    ///
    /// Returns a simple estimate of sensor saturation based on ADC bit depth.
    /// This returns (2^bit_depth - 1), which is the maximum ADC value.
    ///
    /// # Note
    /// This is a simplified placeholder calculation. For more accurate saturation
    /// values, implementations should account for max well depth and conversion gain
    /// (DN per electron), as real sensors may saturate below the ADC maximum due to
    /// well capacity limits. See `simulator::hardware::sensor::SensorConfig::saturating_reading()`
    /// for a physics-based implementation.
    ///
    /// # Returns
    /// Maximum digital number value for the given bit depth
    pub fn get_saturation(&self) -> f64 {
        (2_u32.pow(self.bit_depth as u32) - 1) as f64
    }
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
    ///
    /// # Performance Notes
    /// This is not a high-performance interface and typically incurs camera startup
    /// overhead on each call. For continuous capture, use `stream()` instead.
    fn capture_frame(&mut self) -> CameraResult<(Array2<u16>, FrameMetadata)> {
        let mut result: Option<(Array2<u16>, FrameMetadata)> = None;

        self.stream(&mut |frame, metadata| {
            result = Some((frame.clone(), metadata.clone()));
            false
        })?;

        result.ok_or_else(|| CameraError::CaptureError("No frame captured".to_string()))
    }

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

    /// Get sensor saturation value in DN
    ///
    /// Returns the pixel value at which the sensor saturates. This is used
    /// for guide star filtering to reject saturated sources.
    fn saturation_value(&self) -> f64;

    /// Get camera name/identifier
    ///
    /// Returns a human-readable name for this camera instance
    fn name(&self) -> &str;

    /// Get ADC bit depth
    ///
    /// Returns the bit depth of the analog-to-digital converter (ADC)
    /// Typical values are 8, 12, 14, or 16 bits
    fn get_bit_depth(&self) -> u8;

    /// Set ADC bit depth
    ///
    /// # Arguments
    /// * `bit_depth` - New bit depth value (typically 8, 12, 14, or 16)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CameraError)` if bit depth is unsupported
    fn set_bit_depth(&mut self, bit_depth: u8) -> CameraResult<()>;

    /// Get camera serial number
    ///
    /// Returns a unique hardware identifier for this camera. This can be used
    /// for per-camera calibration data like hot pixel maps.
    ///
    /// # Returns
    /// A unique serial number string for this camera
    fn get_serial(&self) -> String;

    /// Get current gain setting
    ///
    /// # Returns
    /// Current gain value (camera-specific units)
    fn get_gain(&self) -> f64;

    /// Set camera gain
    ///
    /// # Arguments
    /// * `gain` - Gain value to set (camera-specific units)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CameraError)` if gain value is unsupported or out of range
    fn set_gain(&mut self, gain: f64) -> CameraResult<()>;

    /// Check if a given ROI size is valid for this camera
    ///
    /// Some cameras have hardware constraints on ROI dimensions (e.g., must be
    /// divisible by 4, minimum size requirements). This method allows validation
    /// before attempting to set an ROI.
    ///
    /// # Arguments
    /// * `width` - Desired ROI width in pixels
    /// * `height` - Desired ROI height in pixels
    ///
    /// # Returns
    /// * `Ok(())` if the ROI size is valid
    /// * `Err(CameraError)` with details if the ROI size is invalid
    fn check_roi_size(&self, width: usize, height: usize) -> CameraResult<()>;

    /// Stream frames continuously with a callback
    ///
    /// This method replaces the start/stop continuous capture pattern with a
    /// callback-based streaming API. The camera will continuously capture frames
    /// and invoke the callback with the frame data and metadata. The callback
    /// returns true to continue streaming or false to stop.
    ///
    /// # Arguments
    /// * `callback` - Function called for each frame. Returns true to continue, false to stop.
    ///
    /// # Returns
    /// * `Ok(())` when streaming completes normally (callback returned false)
    /// * `Err(CameraError)` if capture fails or timeout occurs
    ///
    /// # Performance Notes
    /// The callback is expected to return quickly. If you need to perform long-running
    /// operations on frame data, clone the data and process it on your own time/thread
    /// to avoid blocking the camera stream.
    ///
    /// # Notes
    /// Timeout is computed automatically based on the camera's current exposure time.
    fn stream(
        &mut self,
        callback: &mut dyn FnMut(&Array2<u16>, &FrameMetadata) -> bool,
    ) -> CameraResult<()>;
}

impl CameraInterface for Box<dyn CameraInterface> {
    fn set_roi(&mut self, roi: AABB) -> CameraResult<()> {
        (**self).set_roi(roi)
    }

    fn clear_roi(&mut self) -> CameraResult<()> {
        (**self).clear_roi()
    }

    fn set_exposure(&mut self, exposure: Duration) -> CameraResult<()> {
        (**self).set_exposure(exposure)
    }

    fn get_exposure(&self) -> Duration {
        (**self).get_exposure()
    }

    fn get_config(&self) -> &CameraConfig {
        (**self).get_config()
    }

    fn is_ready(&self) -> bool {
        (**self).is_ready()
    }

    fn get_roi(&self) -> Option<AABB> {
        (**self).get_roi()
    }

    fn saturation_value(&self) -> f64 {
        (**self).saturation_value()
    }

    fn name(&self) -> &str {
        (**self).name()
    }

    fn get_bit_depth(&self) -> u8 {
        (**self).get_bit_depth()
    }

    fn set_bit_depth(&mut self, bit_depth: u8) -> CameraResult<()> {
        (**self).set_bit_depth(bit_depth)
    }

    fn get_serial(&self) -> String {
        (**self).get_serial()
    }

    fn get_gain(&self) -> f64 {
        (**self).get_gain()
    }

    fn set_gain(&mut self, gain: f64) -> CameraResult<()> {
        (**self).set_gain(gain)
    }

    fn check_roi_size(&self, width: usize, height: usize) -> CameraResult<()> {
        (**self).check_roi_size(width, height)
    }

    fn stream(
        &mut self,
        callback: &mut dyn FnMut(&Array2<u16>, &FrameMetadata) -> bool,
    ) -> CameraResult<()> {
        (**self).stream(callback)
    }
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

    #[test]
    fn test_camera_config_saturation() {
        // Test 8-bit depth
        let config_8bit = CameraConfig {
            width: 640,
            height: 480,
            exposure: Duration::from_millis(100),
            bit_depth: 8,
        };
        assert_eq!(config_8bit.get_saturation(), 255.0);

        // Test 12-bit depth
        let config_12bit = CameraConfig {
            width: 640,
            height: 480,
            exposure: Duration::from_millis(100),
            bit_depth: 12,
        };
        assert_eq!(config_12bit.get_saturation(), 4095.0);

        // Test 14-bit depth
        let config_14bit = CameraConfig {
            width: 640,
            height: 480,
            exposure: Duration::from_millis(100),
            bit_depth: 14,
        };
        assert_eq!(config_14bit.get_saturation(), 16383.0);

        // Test 16-bit depth
        let config_16bit = CameraConfig {
            width: 640,
            height: 480,
            exposure: Duration::from_millis(100),
            bit_depth: 16,
        };
        assert_eq!(config_16bit.get_saturation(), 65535.0);
    }
}

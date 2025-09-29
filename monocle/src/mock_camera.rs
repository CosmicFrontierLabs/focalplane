//! Mock camera implementation for testing

use ndarray::Array2;
use shared::camera_interface::{
    CameraConfig, CameraError, CameraInterface, CameraResult, FrameMetadata, Timestamp,
};
use shared::image_proc::detection::aabb::AABB;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Mock camera for testing
pub struct MockCamera {
    frames: Vec<Array2<u16>>,
    frame_index: Arc<Mutex<usize>>,
    roi_set: Arc<AtomicBool>,
    current_roi: Arc<Mutex<Option<AABB>>>,
    config: CameraConfig,
    is_capturing: Arc<AtomicBool>,
    exposure: Duration,
    elapsed_time: Arc<Mutex<Duration>>,
}

impl MockCamera {
    /// Create a new mock camera with predefined frames
    pub fn new(frames: Vec<Array2<u16>>) -> Self {
        Self {
            frames,
            frame_index: Arc::new(Mutex::new(0)),
            roi_set: Arc::new(AtomicBool::new(false)),
            current_roi: Arc::new(Mutex::new(None)),
            config: CameraConfig {
                width: 100,
                height: 100,
                exposure: Duration::from_millis(10),
            },
            is_capturing: Arc::new(AtomicBool::new(false)),
            exposure: Duration::from_millis(10),
            elapsed_time: Arc::new(Mutex::new(Duration::ZERO)),
        }
    }

    /// Create a mock camera that returns the same frame repeatedly
    pub fn new_repeating(frame: Array2<u16>) -> Self {
        Self::new(vec![frame])
    }

    /// Check if ROI is currently set
    pub fn is_roi_set(&self) -> bool {
        self.roi_set.load(Ordering::SeqCst)
    }

    /// Get the current ROI if set
    pub fn get_current_roi(&self) -> Option<AABB> {
        *self.current_roi.lock().unwrap()
    }

    /// Reset frame index to beginning
    pub fn reset(&mut self) {
        *self.frame_index.lock().unwrap() = 0;
    }
}

impl CameraInterface for MockCamera {
    fn set_roi(&mut self, roi: AABB) -> CameraResult<()> {
        self.roi_set.store(true, Ordering::SeqCst);
        *self.current_roi.lock().unwrap() = Some(roi);
        Ok(())
    }

    fn clear_roi(&mut self) -> CameraResult<()> {
        self.roi_set.store(false, Ordering::SeqCst);
        *self.current_roi.lock().unwrap() = None;
        Ok(())
    }

    fn capture_frame(&mut self) -> CameraResult<(Array2<u16>, FrameMetadata)> {
        let mut index = self.frame_index.lock().unwrap();

        // For repeating mode (single frame), always return the same frame
        let frame_idx = if self.frames.len() == 1 {
            0
        } else {
            if *index >= self.frames.len() {
                return Err(CameraError::CaptureError("No more frames".to_string()));
            }
            let current = *index;
            *index += 1;
            current
        };

        let frame = self.frames[frame_idx].clone();

        // If ROI is set, extract the ROI from the frame
        let output_frame = if let Some(roi) = self.current_roi.lock().unwrap().as_ref() {
            // Ensure ROI is within bounds
            let (height, width) = frame.dim();
            let roi_min_row = roi.min_row.min(height - 1);
            let roi_max_row = roi.max_row.min(height - 1);
            let roi_min_col = roi.min_col.min(width - 1);
            let roi_max_col = roi.max_col.min(width - 1);

            // Extract ROI
            frame
                .slice(ndarray::s![
                    roi_min_row..=roi_max_row,
                    roi_min_col..=roi_max_col
                ])
                .to_owned()
        } else {
            frame
        };

        // Update elapsed time
        let mut elapsed = self.elapsed_time.lock().unwrap();
        *elapsed += self.exposure;

        let metadata = FrameMetadata {
            frame_number: (*self.frame_index.lock().unwrap()).saturating_sub(1) as u64,
            exposure: self.exposure,
            timestamp: Timestamp::from_duration(*elapsed),
            pointing: None,
            roi: *self.current_roi.lock().unwrap(),
            temperature_c: 20.0,
        };
        Ok((output_frame, metadata))
    }

    fn set_exposure(&mut self, exposure: Duration) -> CameraResult<()> {
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
        true
    }

    fn get_roi(&self) -> Option<AABB> {
        *self.current_roi.lock().unwrap()
    }

    fn start_continuous_capture(&mut self) -> CameraResult<()> {
        self.is_capturing.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn stop_continuous_capture(&mut self) -> CameraResult<()> {
        self.is_capturing.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn get_latest_frame(&mut self) -> Option<(Array2<u16>, FrameMetadata)> {
        if self.is_capturing.load(Ordering::SeqCst) {
            self.capture_frame().ok()
        } else {
            None
        }
    }

    fn is_capturing(&self) -> bool {
        self.is_capturing.load(Ordering::SeqCst)
    }
}

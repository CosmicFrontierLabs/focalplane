//! MONOCLE - Modular Orientation, Navigation & Optical Control Logic Engine
//!
//! Fine Guidance System state machine implementation based on the FGS ConOps.
//! Processes images through states: Idle -> Acquiring -> Calibrating -> Tracking

use ndarray::{Array2, ArrayView2};
use shared::camera_interface::{CameraInterface, Timestamp};
use shared::image_proc::detection::StarDetection;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub mod callback;
pub mod config;
pub mod filters;
pub mod mock_camera;
pub mod selection;
pub mod state;

use crate::callback::{CallbackId, FgsCallback, PositionEstimate, TrackingLostReason};
use shared::image_proc::detection::aabb::AABB;

// Re-export commonly used types for external use
pub use crate::callback::FgsCallbackEvent;
pub use crate::config::FgsConfig;
pub use crate::state::{FgsEvent, FgsState};

/// A selected guide star
#[derive(Debug, Clone)]
pub struct GuideStar {
    /// Unique identifier
    pub id: usize,
    /// Position in full frame
    pub x: f64,
    pub y: f64,
    /// Estimated flux
    pub flux: f64,
    /// Signal-to-noise ratio
    pub snr: f64,
    /// Region of interest for tracking (bounding box in pixel coordinates)
    pub roi: AABB,
    /// Estimated star diameter in pixels
    pub diameter: f64,
}

impl GuideStar {
    /// Compute the center position relative to the ROI coordinates
    pub fn roi_center(&self) -> (f64, f64) {
        let roi_center_x = self.x - self.roi.min_col as f64;
        let roi_center_y = self.y - self.roi.min_row as f64;
        (roi_center_x, roi_center_y)
    }
}

/// Guidance update produced by the system
#[derive(Debug, Clone)]
pub struct GuidanceUpdate {
    /// X position in frame coordinates
    pub x: f64,
    /// Y position in frame coordinates
    pub y: f64,
    /// Integrated flux of the tracked star
    pub flux: f64,
    /// Timestamp of update (from camera frame)
    pub timestamp: Timestamp,
}

/// Main Fine Guidance System state machine
pub struct FineGuidanceSystem<C: CameraInterface> {
    /// Camera interface for capturing frames
    camera: C,
    /// Current state
    state: FgsState,
    /// System configuration
    config: FgsConfig,
    /// Selected guide star (populated during calibration)
    guide_star: Option<GuideStar>,
    /// Accumulated frame sum (stored as f64 to avoid overflow)
    accumulated_frame: Option<Array2<f64>>,
    /// Number of frames accumulated
    frames_accumulated: usize,
    /// Detected stars from calibration (sorted by flux, brightest first)
    detected_stars: Vec<StarDetection>,
    /// Last guidance update
    last_update: Option<GuidanceUpdate>,
    /// Registered callbacks
    callbacks: Arc<Mutex<HashMap<CallbackId, FgsCallback>>>,
    /// Next callback ID
    next_callback_id: Arc<Mutex<CallbackId>>,
    /// Current track ID
    current_track_id: u32,
}

impl<C: CameraInterface> FineGuidanceSystem<C> {
    /// Create a new Fine Guidance System with a camera
    pub fn new(camera: C, config: FgsConfig) -> Self {
        Self {
            camera,
            state: FgsState::Idle,
            config,
            guide_star: None,
            accumulated_frame: None,
            frames_accumulated: 0,
            detected_stars: Vec::new(),
            last_update: None,
            callbacks: Arc::new(Mutex::new(HashMap::new())),
            next_callback_id: Arc::new(Mutex::new(0)),
            current_track_id: 0,
        }
    }

    /// Register a callback for FGS events
    pub fn register_callback<F>(&self, callback: F) -> CallbackId
    where
        F: Fn(&FgsCallbackEvent) + Send + Sync + 'static,
    {
        let mut callbacks = self.callbacks.lock().unwrap();
        let mut next_id = self.next_callback_id.lock().unwrap();

        let callback_id = *next_id;
        *next_id += 1;

        callbacks.insert(callback_id, Arc::new(callback));
        callback_id
    }

    /// Deregister a callback
    pub fn deregister_callback(&self, callback_id: CallbackId) -> bool {
        let mut callbacks = self.callbacks.lock().unwrap();
        callbacks.remove(&callback_id).is_some()
    }

    /// Get the number of registered callbacks
    pub fn callback_count(&self) -> usize {
        self.callbacks.lock().unwrap().len()
    }

    /// Emit an event to all registered callbacks
    fn emit_event(&self, event: &FgsCallbackEvent) {
        let callbacks = self.callbacks.lock().unwrap();
        for callback in callbacks.values() {
            callback(event);
        }
    }

    /// Handle transition from Idle to Acquiring
    fn handle_idle_start(&mut self) -> FgsState {
        log::info!("Starting FGS, entering Acquiring state");
        self.accumulated_frame = None;
        self.frames_accumulated = 0;
        self.guide_star = None;
        self.detected_stars.clear();
        FgsState::Acquiring {
            frames_collected: 0,
        }
    }

    /// Handle frame processing during Acquiring state
    fn handle_acquiring_frame(
        &mut self,
        frames_collected: usize,
        frame: ArrayView2<u16>,
        _timestamp: Timestamp,
    ) -> Result<FgsState, String> {
        let frames = frames_collected + 1;
        self.accumulate_frame(frame)?;

        if frames >= self.config.acquisition_frames {
            log::info!("Acquisition complete, entering Calibrating state");
            Ok(FgsState::Calibrating)
        } else {
            Ok(FgsState::Acquiring {
                frames_collected: frames,
            })
        }
    }

    /// Handle abort during Acquiring state
    fn handle_acquiring_abort(&mut self) -> FgsState {
        log::info!("Aborting acquisition, returning to Idle");
        self.accumulated_frame = None;
        self.frames_accumulated = 0;
        FgsState::Idle
    }

    /// Handle frame processing during Calibrating state
    fn handle_calibrating_frame(
        &mut self,
        frame: ArrayView2<u16>,
        timestamp: Timestamp,
    ) -> Result<FgsState, String> {
        self.detect_and_select_guides(frame)?;

        if let Some(guide_star) = &self.guide_star {
            log::info!("Calibration complete with guide star, entering Tracking");

            // Set camera ROI to track the guide star
            log::info!(
                "Setting camera ROI to {:?} for star at ({}, {})",
                guide_star.roi,
                guide_star.x,
                guide_star.y
            );
            if let Err(e) = self.camera.set_roi(guide_star.roi) {
                log::warn!("Failed to set camera ROI: {e}");
                // Continue anyway - camera may not support ROI
            }

            // Increment track ID for new tracking session
            self.current_track_id += 1;

            // Emit tracking started event
            self.emit_tracking_started_event(timestamp);

            Ok(FgsState::Tracking {
                frames_processed: 0,
            })
        } else {
            log::warn!("No suitable guide stars found, returning to Idle");
            Ok(FgsState::Idle)
        }
    }

    /// Handle frame processing during Tracking state
    fn handle_tracking_frame(
        &mut self,
        frames_processed: usize,
        frame: ArrayView2<u16>,
        timestamp: Timestamp,
    ) -> Result<FgsState, String> {
        let update_opt = self.track(frame, timestamp)?;

        // Only process if we got a valid update
        if let Some(update) = update_opt {
            if self.guide_star.is_some() {
                // Emit tracking update event
                self.emit_tracking_update_event(&update);

                self.last_update = Some(update.clone());
                Ok(FgsState::Tracking {
                    frames_processed: frames_processed + 1,
                })
            } else {
                log::warn!("Lost all guide stars, entering Reacquiring");

                // Clear camera ROI when losing tracking
                if let Err(e) = self.camera.clear_roi() {
                    log::warn!("Failed to clear camera ROI: {e}");
                }

                // Emit tracking lost event
                self.emit_tracking_lost_event(TrackingLostReason::SignalTooWeak, timestamp);

                Ok(FgsState::Reacquiring { attempts: 0 })
            }
        } else {
            // No update due to frame mismatch, stay in tracking
            Ok(FgsState::Tracking {
                frames_processed: frames_processed + 1,
            })
        }
    }

    /// Handle frame processing during Reacquiring state
    fn handle_reacquiring_frame(
        &mut self,
        attempts: usize,
        frame: ArrayView2<u16>,
        _timestamp: Timestamp,
    ) -> Result<FgsState, String> {
        let recovered = self.attempt_reacquisition(frame)?;

        if recovered {
            log::info!("Lock recovered, returning to Tracking");
            Ok(FgsState::Tracking {
                frames_processed: 0,
            })
        } else if attempts + 1 >= self.config.max_reacquisition_attempts {
            log::warn!("Reacquisition timeout, returning to Calibrating");
            Ok(FgsState::Calibrating)
        } else {
            Ok(FgsState::Reacquiring {
                attempts: attempts + 1,
            })
        }
    }

    /// Emit tracking started event
    fn emit_tracking_started_event(&self, timestamp: Timestamp) {
        if let Some(guide_star) = &self.guide_star {
            self.emit_event(&FgsCallbackEvent::TrackingStarted {
                track_id: self.current_track_id,
                initial_position: PositionEstimate {
                    x: guide_star.x,
                    y: guide_star.y,
                    timestamp,
                },
                num_guide_stars: 1,
            });
        }
    }

    /// Emit tracking update event
    fn emit_tracking_update_event(&self, update: &GuidanceUpdate) {
        // NOTE(meawoppl) - the timestamp in this field will need to be chained back to the sensor timestamp
        self.emit_event(&FgsCallbackEvent::TrackingUpdate {
            track_id: self.current_track_id,
            position: PositionEstimate {
                x: update.x,
                y: update.y,
                timestamp: update.timestamp,
            },
        });
    }

    /// Emit tracking lost event
    fn emit_tracking_lost_event(&self, reason: TrackingLostReason, timestamp: Timestamp) {
        if let Some(guide_star) = &self.guide_star {
            self.emit_event(&FgsCallbackEvent::TrackingLost {
                track_id: self.current_track_id,
                last_position: PositionEstimate {
                    x: guide_star.x,
                    y: guide_star.y,
                    timestamp,
                },
                reason,
            });
        }
    }

    /// Process an event and potentially transition states
    pub fn process_event(&mut self, event: FgsEvent<'_>) -> Result<Option<GuidanceUpdate>, String> {
        use FgsState::*;

        let new_state = match (&self.state, event) {
            // From Idle
            (Idle, FgsEvent::StartFgs) => self.handle_idle_start(),

            // From Acquiring
            (Acquiring { frames_collected }, FgsEvent::ProcessFrame(frame, timestamp)) => {
                self.handle_acquiring_frame(*frames_collected, frame, timestamp)?
            }
            (Acquiring { .. }, FgsEvent::Abort) => self.handle_acquiring_abort(),

            // From Calibrating
            (Calibrating, FgsEvent::ProcessFrame(frame, timestamp)) => {
                self.handle_calibrating_frame(frame, timestamp)?
            }

            // From Tracking
            (Tracking { frames_processed }, FgsEvent::ProcessFrame(frame, timestamp)) => {
                self.handle_tracking_frame(*frames_processed, frame, timestamp)?
            }
            (Tracking { .. }, FgsEvent::StopFgs) => {
                log::info!("Stopping FGS, returning to Idle");
                // Clear camera ROI when stopping
                if let Err(e) = self.camera.clear_roi() {
                    log::warn!("Failed to clear camera ROI: {e}");
                }
                Idle
            }

            // From Reacquiring
            (Reacquiring { attempts }, FgsEvent::ProcessFrame(frame, timestamp)) => {
                self.handle_reacquiring_frame(*attempts, frame, timestamp)?
            }
            (Reacquiring { .. }, FgsEvent::Abort) => {
                log::info!("Aborting reacquisition, returning to Idle");
                // Clear camera ROI when aborting
                if let Err(e) = self.camera.clear_roi() {
                    log::warn!("Failed to clear camera ROI: {e}");
                }
                Idle
            }

            // Invalid transitions
            _ => {
                log::warn!("Invalid state transition");
                self.state.clone()
            }
        };

        self.state = new_state;
        Ok(self.last_update.clone())
    }

    /// Process a single image frame
    pub fn process_frame(
        &mut self,
        frame: ArrayView2<u16>,
        timestamp: Timestamp,
    ) -> Result<Option<GuidanceUpdate>, String> {
        self.process_event(FgsEvent::ProcessFrame(frame, timestamp))
    }

    /// Capture and process the next frame from the camera
    pub fn process_next_frame(&mut self) -> Result<Option<GuidanceUpdate>, String> {
        let (frame, metadata) = self
            .camera
            .capture_frame()
            .map_err(|e| format!("Camera capture failed: {e}"))?;
        self.process_frame(frame.view(), metadata.timestamp)
    }

    /// Get the current state
    pub fn state(&self) -> &FgsState {
        &self.state
    }

    /// Get information about the currently tracked guide star
    pub fn guide_star(&self) -> Option<&GuideStar> {
        self.guide_star.as_ref()
    }

    /// Get the averaged accumulated frame
    pub fn get_averaged_frame(&self) -> Option<Array2<f64>> {
        self.accumulated_frame.as_ref().map(|frame| {
            if self.frames_accumulated > 0 {
                frame / self.frames_accumulated as f64
            } else {
                frame.clone()
            }
        })
    }

    /// Get the detected stars (sorted by flux, brightest first)
    pub fn get_detected_stars(&self) -> &[StarDetection] {
        &self.detected_stars
    }

    /// Accumulate frames during acquisition
    fn accumulate_frame(&mut self, frame: ArrayView2<u16>) -> Result<(), String> {
        match &mut self.accumulated_frame {
            Some(sum) => {
                // Add this frame to the existing sum
                if sum.shape() != frame.shape() {
                    return Err("Frame dimensions mismatch".to_string());
                }
                for ((i, j), value) in frame.indexed_iter() {
                    sum[[i, j]] += *value as f64;
                }
            }
            None => {
                // Initialize with first frame
                let mut sum = Array2::<f64>::zeros(frame.dim());
                for ((i, j), value) in frame.indexed_iter() {
                    sum[[i, j]] = *value as f64;
                }
                self.accumulated_frame = Some(sum);
            }
        }
        self.frames_accumulated += 1;
        Ok(())
    }

    /// Detect stars and select guide stars from averaged frame
    fn detect_and_select_guides(&mut self, _frame: ArrayView2<u16>) -> Result<(), String> {
        let averaged_frame = self
            .get_averaged_frame()
            .ok_or("No accumulated frame available for calibration")?;

        let (guide_star, detected_stars) =
            selection::detect_and_select_guides(averaged_frame.view(), &self.config)?;

        self.guide_star = guide_star;
        self.detected_stars = detected_stars;

        Ok(())
    }

    /// Create a circular mask centered at given position with specified radius
    fn create_circular_mask(
        shape: (usize, usize),
        center_x: f64,
        center_y: f64,
        radius: f64,
    ) -> ndarray::Array2<bool> {
        let (height, width) = shape;
        ndarray::Array2::from_shape_fn((height, width), |(row, col)| {
            let dx = col as f64 - center_x;
            let dy = row as f64 - center_y;
            let distance = (dx * dx + dy * dy).sqrt();
            distance <= radius
        })
    }

    /// Track guide star and compute guidance update
    fn track(
        &mut self,
        frame: ArrayView2<u16>,
        timestamp: Timestamp,
    ) -> Result<Option<GuidanceUpdate>, String> {
        use shared::image_proc::centroid::compute_centroid_from_mask;

        let guide_star = self
            .guide_star
            .as_ref()
            .ok_or("No guide star available for tracking")?;

        // Check if we're receiving an ROI frame that matches expected size
        let (frame_height, frame_width) = frame.dim();
        let roi_bounds = &guide_star.roi;
        let expected_roi_height = roi_bounds.max_row - roi_bounds.min_row + 1;
        let expected_roi_width = roi_bounds.max_col - roi_bounds.min_col + 1;

        // If frame size doesn't match expected ROI size, emit event and return None
        if frame_height != expected_roi_height || frame_width != expected_roi_width {
            log::warn!(
                "Expected ROI frame {expected_roi_height}x{expected_roi_width}, got {frame_height}x{frame_width} - skipping frame"
            );

            // Emit frame size mismatch event
            self.emit_event(&FgsCallbackEvent::FrameSizeMismatch {
                expected_width: expected_roi_width,
                expected_height: expected_roi_height,
                actual_width: frame_width,
                actual_height: frame_height,
            });

            // Return None to indicate no guidance update
            return Ok(None);
        }

        // Frame is the ROI we expected
        let roi = frame;
        let roi_min_row = roi_bounds.min_row;
        let roi_min_col = roi_bounds.min_col;
        let roi_max_row = roi_bounds.max_row;
        let roi_max_col = roi_bounds.max_col;

        // Convert ROI to f64 for centroid calculation
        let roi_f64 = roi.mapv(|v| v as f64);

        // Create mask for centroid calculation based on configurable radius
        let radius = self.config.fwhm * self.config.centroid_radius_multiplier;

        // Create circular mask centered on expected position within ROI
        let roi_height = roi_max_row - roi_min_row + 1;
        let roi_width = roi_max_col - roi_min_col + 1;
        let (roi_center_x, roi_center_y) = guide_star.roi_center();

        let mask =
            Self::create_circular_mask((roi_height, roi_width), roi_center_x, roi_center_y, radius);

        // Compute centroid within the masked region
        let centroid_result = compute_centroid_from_mask(&roi_f64.view(), &mask.view());

        // Convert centroid position back to full frame coordinates
        let new_x = roi_min_col as f64 + centroid_result.x;
        let new_y = roi_min_row as f64 + centroid_result.y;

        Ok(Some(GuidanceUpdate {
            x: new_x,
            y: new_y,
            flux: centroid_result.flux,
            timestamp,
        }))
    }

    /// Attempt to reacquire lost guide stars
    fn attempt_reacquisition(&mut self, _frame: ArrayView2<u16>) -> Result<bool, String> {
        // TODO: Implement reacquisition logic
        // 1. Search in expanded ROIs
        // 2. Try to match with known guide stars
        // 3. Return true if enough stars recovered
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_camera::MockCamera;
    use ndarray::Array2;
    use std::time::Duration;

    fn test_timestamp() -> Timestamp {
        Timestamp::from_duration(Duration::from_millis(100))
    }

    fn test_config() -> FgsConfig {
        FgsConfig {
            acquisition_frames: 3,
            filters: crate::config::GuideStarFilters {
                detection_threshold_sigma: 5.0,
                snr_min: 5.0,
                diameter_range: (2.0, 20.0),
                aspect_ratio_max: 2.5,
                saturation_value: 4000.0,
                saturation_search_radius: 3.0,
                minimum_edge_distance: 10.0,
            },
            max_guide_stars: 3,
            roi_size: 32,
            max_reacquisition_attempts: 5,
            centroid_radius_multiplier: 5.0,
            fwhm: 3.0,
        }
    }

    #[test]
    fn test_state_transitions() {
        let camera = MockCamera::new_repeating(Array2::<u16>::zeros((100, 100)));
        let mut fgs = FineGuidanceSystem::new(camera, test_config());

        // Should start in Idle
        assert_eq!(fgs.state(), &FgsState::Idle);

        // Start FGS
        let _ = fgs.process_event(FgsEvent::StartFgs);
        assert!(matches!(fgs.state(), FgsState::Acquiring { .. }));
    }

    #[test]
    fn test_process_frame() {
        let camera = MockCamera::new_repeating(Array2::<u16>::zeros((100, 100)));
        let mut fgs = FineGuidanceSystem::new(camera, test_config());
        let dummy_frame = Array2::<u16>::zeros((100, 100));

        // Should do nothing in Idle state
        let result = fgs.process_frame(dummy_frame.view(), test_timestamp());
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_frame_accumulation() {
        let camera = MockCamera::new_repeating(Array2::<u16>::zeros((10, 10)));
        let mut fgs = FineGuidanceSystem::new(camera, test_config());

        // Start FGS
        let _ = fgs.process_event(FgsEvent::StartFgs);

        // Create test frames with different values
        let frame1 = Array2::<u16>::from_elem((10, 10), 100);
        let frame2 = Array2::<u16>::from_elem((10, 10), 200);
        let frame3 = Array2::<u16>::from_elem((10, 10), 300);

        // Process first frame
        let _ = fgs.process_frame(frame1.view(), test_timestamp());
        assert_eq!(fgs.frames_accumulated, 1);
        assert!(matches!(
            fgs.state(),
            FgsState::Acquiring {
                frames_collected: 1
            }
        ));

        // Process second frame
        let _ = fgs.process_frame(frame2.view(), test_timestamp());
        assert_eq!(fgs.frames_accumulated, 2);
        assert!(matches!(
            fgs.state(),
            FgsState::Acquiring {
                frames_collected: 2
            }
        ));

        // Process third frame - should transition to Calibrating
        let _ = fgs.process_frame(frame3.view(), test_timestamp());
        assert_eq!(fgs.frames_accumulated, 3);
        assert!(matches!(fgs.state(), FgsState::Calibrating));

        // Check averaged frame
        let averaged = fgs
            .get_averaged_frame()
            .expect("Should have accumulated frame");
        let expected_avg = (100.0 + 200.0 + 300.0) / 3.0;
        assert!((averaged[[0, 0]] - expected_avg).abs() < 1e-10);
    }

    #[test]
    fn test_frame_accumulation_abort() {
        let camera = MockCamera::new_repeating(Array2::<u16>::zeros((10, 10)));
        let mut fgs = FineGuidanceSystem::new(camera, test_config());

        // Start FGS and accumulate some frames
        let _ = fgs.process_event(FgsEvent::StartFgs);
        let frame = Array2::<u16>::from_elem((10, 10), 100);
        let _ = fgs.process_frame(frame.view(), test_timestamp());
        assert_eq!(fgs.frames_accumulated, 1);

        // Abort should clear accumulation
        let _ = fgs.process_event(FgsEvent::Abort);
        assert_eq!(fgs.frames_accumulated, 0);
        assert!(fgs.accumulated_frame.is_none());
        assert_eq!(fgs.state(), &FgsState::Idle);
    }

    #[test]
    fn test_callback_registration() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let camera = MockCamera::new_repeating(Array2::<u16>::zeros((100, 100)));
        let fgs = FineGuidanceSystem::new(camera, test_config());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        // Register callback
        let callback_id = fgs.register_callback(move |_event| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Test event emission
        fgs.emit_event(&FgsCallbackEvent::TrackingStarted {
            track_id: 1,
            initial_position: PositionEstimate {
                x: 100.0,
                y: 200.0,
                timestamp: test_timestamp(),
            },
            num_guide_stars: 1,
        });

        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Deregister and test again
        assert!(fgs.deregister_callback(callback_id));

        fgs.emit_event(&FgsCallbackEvent::TrackingUpdate {
            track_id: 1,
            position: PositionEstimate {
                x: 101.0,
                y: 201.0,
                timestamp: test_timestamp(),
            },
        });

        // Counter should not have increased
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_multiple_callbacks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let camera = MockCamera::new_repeating(Array2::<u16>::zeros((100, 100)));
        let fgs = FineGuidanceSystem::new(camera, test_config());

        let counter1 = Arc::new(AtomicUsize::new(0));
        let counter2 = Arc::new(AtomicUsize::new(0));

        let c1_clone = counter1.clone();
        let c2_clone = counter2.clone();

        let _id1 = fgs.register_callback(move |_| {
            c1_clone.fetch_add(1, Ordering::SeqCst);
        });

        let _id2 = fgs.register_callback(move |_| {
            c2_clone.fetch_add(10, Ordering::SeqCst);
        });

        fgs.emit_event(&FgsCallbackEvent::TrackingLost {
            track_id: 1,
            last_position: PositionEstimate {
                x: 100.0,
                y: 200.0,
                timestamp: test_timestamp(),
            },
            reason: TrackingLostReason::SignalTooWeak,
        });

        assert_eq!(counter1.load(Ordering::SeqCst), 1);
        assert_eq!(counter2.load(Ordering::SeqCst), 10);
    }
}

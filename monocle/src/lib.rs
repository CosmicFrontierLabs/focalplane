//! MONOCLE - Modular Orientation, Navigation & Optical Control Logic Engine
//!
//! Fine Guidance System state machine implementation based on the FGS ConOps.
//! Processes images through states: Idle -> Acquiring -> Calibrating -> Tracking

use ndarray::{Array2, ArrayView2};
use shared::camera_interface::{CameraInterface, Timestamp};
use shared::image_proc::detection::StarDetection;
use shared::image_proc::source_snr::calculate_snr_at_position;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub mod callback;
pub mod config;
pub mod filters;
pub mod selection;
pub mod state;

use crate::callback::{CallbackId, FgsCallback, PositionEstimate, TrackingLostReason};
use shared::image_proc::detection::aabb::AABB;

// Re-export commonly used types for external use
pub use crate::callback::FgsCallbackEvent;
pub use crate::config::FgsConfig;
pub use crate::state::{FgsEvent, FgsState};

/// Camera settings update requested by FGS
///
/// The FGS decides what camera settings need to change but does not apply them directly.
/// The caller is responsible for applying these updates to the camera for subsequent frames.
#[derive(Debug, Clone, PartialEq)]
pub enum CameraSettingsUpdate {
    /// Set camera ROI to specified bounds
    SetROI(AABB),
    /// Clear camera ROI and return to full-frame readout
    ClearROI,
}

/// Result type for state handlers that may produce guidance updates and camera settings
type StateTransitionResult = (
    FgsState,
    (Option<GuidanceUpdate>, Vec<CameraSettingsUpdate>),
);

/// Apply camera settings updates to a camera
///
/// Helper function to apply a list of camera settings updates returned by the FGS.
/// Logs warnings if any setting fails to apply but continues with remaining settings.
///
/// # Arguments
/// * `camera` - The camera to apply settings to
/// * `updates` - List of settings updates to apply
pub fn apply_camera_settings<C: CameraInterface>(
    camera: &mut C,
    updates: Vec<CameraSettingsUpdate>,
) {
    for setting in updates {
        match setting {
            CameraSettingsUpdate::SetROI(roi) => {
                if let Err(e) = camera.set_roi(roi) {
                    log::warn!("Failed to set camera ROI: {e}");
                }
            }
            CameraSettingsUpdate::ClearROI => {
                if let Err(e) = camera.clear_roi() {
                    log::warn!("Failed to clear camera ROI: {e}");
                }
            }
        }
    }
}

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
pub struct FineGuidanceSystem {
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
    /// Registered callbacks
    callbacks: Arc<Mutex<HashMap<CallbackId, FgsCallback>>>,
    /// Next callback ID
    next_callback_id: Arc<Mutex<CallbackId>>,
    /// Current track ID
    current_track_id: u32,
    /// Total frames processed
    frame_counter: usize,
}

impl FineGuidanceSystem {
    /// Create a new Fine Guidance System
    pub fn new(config: FgsConfig) -> Self {
        Self {
            state: FgsState::Idle,
            config,
            guide_star: None,
            accumulated_frame: None,
            frames_accumulated: 0,
            detected_stars: Vec::new(),
            callbacks: Arc::new(Mutex::new(HashMap::new())),
            next_callback_id: Arc::new(Mutex::new(0)),
            current_track_id: 0,
            frame_counter: 0,
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
    fn handle_idle_start(&mut self) -> (FgsState, Vec<CameraSettingsUpdate>) {
        log::info!("Starting FGS, entering Acquiring state");
        self.accumulated_frame = None;
        self.frames_accumulated = 0;
        self.guide_star = None;
        self.detected_stars.clear();
        (
            FgsState::Acquiring {
                frames_collected: 0,
            },
            vec![],
        )
    }

    /// Handle frame processing during Acquiring state
    fn handle_acquiring_frame(
        &mut self,
        frames_collected: usize,
        frame: ArrayView2<u16>,
        _timestamp: Timestamp,
    ) -> Result<(FgsState, Vec<CameraSettingsUpdate>), String> {
        let frames = frames_collected + 1;
        let (height, width) = frame.dim();
        self.accumulate_frame(frame)?;

        log::info!(
            "Acquired calibration frame {}/{} (dimensions: {}x{})",
            frames,
            self.config.acquisition_frames,
            width,
            height
        );

        if frames >= self.config.acquisition_frames {
            log::info!("Acquisition complete, entering Calibrating state");
            Ok((FgsState::Calibrating, vec![]))
        } else {
            Ok((
                FgsState::Acquiring {
                    frames_collected: frames,
                },
                vec![],
            ))
        }
    }

    /// Handle abort during Acquiring state
    fn handle_acquiring_abort(&mut self) -> (FgsState, Vec<CameraSettingsUpdate>) {
        log::info!("Aborting acquisition, returning to Idle");
        self.accumulated_frame = None;
        self.frames_accumulated = 0;
        (FgsState::Idle, vec![])
    }

    /// Handle frame processing during Calibrating state
    fn handle_calibrating_frame(
        &mut self,
        frame: ArrayView2<u16>,
        timestamp: Timestamp,
    ) -> Result<(FgsState, Vec<CameraSettingsUpdate>), String> {
        self.detect_and_select_guides(frame)?;

        if let Some(guide_star) = &self.guide_star {
            log::info!("Calibration complete with guide star, entering Tracking");

            // Request camera ROI to be set for tracking the guide star
            log::info!(
                "Requesting camera ROI: x1={}, x2={}, y1={}, y2={} for star at ({:.2}, {:.2})",
                guide_star.roi.min_col,
                guide_star.roi.max_col,
                guide_star.roi.min_row,
                guide_star.roi.max_row,
                guide_star.x,
                guide_star.y
            );

            // Increment track ID for new tracking session
            self.current_track_id += 1;

            // Emit tracking started event
            self.emit_tracking_started_event(timestamp);

            Ok((
                FgsState::Tracking {
                    frames_processed: 0,
                },
                vec![CameraSettingsUpdate::SetROI(guide_star.roi)],
            ))
        } else {
            log::warn!("No suitable guide stars found, returning to Idle");
            Ok((FgsState::Idle, vec![]))
        }
    }

    /// Handle frame processing during Tracking state
    ///
    /// Returns (new_state, (guidance_update_if_fresh, camera_settings))
    fn handle_tracking_frame(
        &mut self,
        frames_processed: usize,
        frame: ArrayView2<u16>,
        timestamp: Timestamp,
    ) -> Result<StateTransitionResult, String> {
        let update_result = self.track(frame, timestamp);

        // Check for SNR dropout - triggers transition to Reacquiring
        if let Err(ref e) = update_result {
            if e.starts_with("SNR_DROPOUT:") {
                log::warn!("SNR dropped below threshold, entering Reacquiring: {e}");

                // Emit tracking lost event
                self.emit_tracking_lost_event(TrackingLostReason::SignalTooWeak, timestamp);

                // Clear guide star so reacquisition can find a new one
                self.guide_star = None;

                return Ok((
                    FgsState::Reacquiring { attempts: 0 },
                    (None, vec![CameraSettingsUpdate::ClearROI]),
                ));
            }
        }

        let update_opt = update_result?;

        // Only process if we got a valid update
        if let Some(update) = update_opt {
            if self.guide_star.is_some() {
                // Emit tracking update event
                self.emit_tracking_update_event(&update);

                Ok((
                    FgsState::Tracking {
                        frames_processed: frames_processed + 1,
                    },
                    (Some(update), vec![]),
                ))
            } else {
                log::warn!("Lost all guide stars, entering Reacquiring");

                // Emit tracking lost event
                self.emit_tracking_lost_event(TrackingLostReason::SignalTooWeak, timestamp);

                Ok((
                    FgsState::Reacquiring { attempts: 0 },
                    (None, vec![CameraSettingsUpdate::ClearROI]),
                ))
            }
        } else {
            // No update due to frame mismatch, stay in tracking
            Ok((
                FgsState::Tracking {
                    frames_processed: frames_processed + 1,
                },
                (None, vec![]),
            ))
        }
    }

    /// Handle frame processing during Reacquiring state
    fn handle_reacquiring_frame(
        &mut self,
        attempts: usize,
        frame: ArrayView2<u16>,
        _timestamp: Timestamp,
    ) -> Result<(FgsState, Vec<CameraSettingsUpdate>), String> {
        let recovered = self.attempt_reacquisition(frame)?;

        if recovered {
            log::info!("Lock recovered, returning to Tracking");
            Ok((
                FgsState::Tracking {
                    frames_processed: 0,
                },
                vec![],
            ))
        } else if attempts + 1 >= self.config.max_reacquisition_attempts {
            log::warn!("Reacquisition timeout, returning to Calibrating");
            Ok((FgsState::Calibrating, vec![]))
        } else {
            Ok((
                FgsState::Reacquiring {
                    attempts: attempts + 1,
                },
                vec![],
            ))
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
    ///
    /// Returns a tuple containing:
    /// - Optional guidance update (only fresh updates from Tracking state, never cached)
    /// - Vector of camera settings updates that the caller must apply
    pub fn process_event(
        &mut self,
        event: FgsEvent<'_>,
    ) -> Result<(Option<GuidanceUpdate>, Vec<CameraSettingsUpdate>), String> {
        use FgsState::*;

        // Capture frame data if this is a ProcessFrame event for FrameProcessed callback
        let frame_data_for_callback = if let FgsEvent::ProcessFrame(frame, timestamp) = &event {
            Some((Arc::new(frame.to_owned()), *timestamp))
        } else {
            None
        };

        let (new_state, guidance_update, camera_updates) = match (&self.state, event) {
            // From Idle
            (Idle, FgsEvent::StartFgs) => {
                let (state, settings) = self.handle_idle_start();
                (state, None, settings)
            }

            // From Acquiring
            (Acquiring { frames_collected }, FgsEvent::ProcessFrame(frame, timestamp)) => {
                let (state, settings) =
                    self.handle_acquiring_frame(*frames_collected, frame, timestamp)?;
                (state, None, settings)
            }
            (Acquiring { .. }, FgsEvent::Abort) => {
                let (state, settings) = self.handle_acquiring_abort();
                (state, None, settings)
            }

            // From Calibrating
            (Calibrating, FgsEvent::ProcessFrame(frame, timestamp)) => {
                let (state, settings) = self.handle_calibrating_frame(frame, timestamp)?;
                (state, None, settings)
            }

            // From Tracking - only state that produces guidance updates
            (Tracking { frames_processed }, FgsEvent::ProcessFrame(frame, timestamp)) => {
                let (state, (update, settings)) =
                    self.handle_tracking_frame(*frames_processed, frame, timestamp)?;
                (state, update, settings)
            }
            (Tracking { .. }, FgsEvent::StopFgs) => {
                log::info!("Stopping FGS, returning to Idle");
                (Idle, None, vec![CameraSettingsUpdate::ClearROI])
            }

            // From Reacquiring
            (Reacquiring { attempts }, FgsEvent::ProcessFrame(frame, timestamp)) => {
                let (state, settings) =
                    self.handle_reacquiring_frame(*attempts, frame, timestamp)?;
                (state, None, settings)
            }
            (Reacquiring { .. }, FgsEvent::Abort) => {
                log::info!("Aborting reacquisition, returning to Idle");
                (Idle, None, vec![CameraSettingsUpdate::ClearROI])
            }

            // Invalid transitions
            _ => {
                log::warn!("Invalid state transition");
                (self.state.clone(), None, vec![])
            }
        };

        self.state = new_state;

        // Emit FrameProcessed callback if this was a frame processing event
        if let Some((frame_data, timestamp)) = frame_data_for_callback {
            self.frame_counter += 1;

            let (track_id, position) = if let Some(ref update) = guidance_update {
                (
                    Some(self.current_track_id),
                    Some(callback::PositionEstimate {
                        x: update.x,
                        y: update.y,
                        timestamp,
                    }),
                )
            } else {
                (None, None)
            };

            self.emit_event(&callback::FgsCallbackEvent::FrameProcessed {
                frame_number: self.frame_counter,
                timestamp,
                frame_data,
                track_id,
                position,
            });
        }

        Ok((guidance_update, camera_updates))
    }

    /// Process a single image frame
    ///
    /// Returns a tuple containing:
    /// - Optional guidance update (only when processing frames in Tracking state)
    /// - Vector of camera settings updates that the caller must apply
    pub fn process_frame(
        &mut self,
        frame: ArrayView2<u16>,
        timestamp: Timestamp,
    ) -> Result<(Option<GuidanceUpdate>, Vec<CameraSettingsUpdate>), String> {
        self.process_event(FgsEvent::ProcessFrame(frame, timestamp))
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

        // Compute SNR to check if star is still trackable
        let aperture = guide_star.diameter / 2.0;
        let snr = calculate_snr_at_position(
            centroid_result.x,
            centroid_result.y,
            &roi_f64.view(),
            aperture,
            aperture * 2.0,
            aperture * 3.0,
        )
        .map_err(|e| format!("Failed to calculate SNR during tracking: {e}"))?;

        // Check if SNR dropped below threshold - return specific error for state machine handling
        if snr < self.config.snr_dropout_threshold {
            return Err(format!(
                "SNR_DROPOUT: {:.2} < {:.2}",
                snr, self.config.snr_dropout_threshold
            ));
        }

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
        // See TODO.md: Monocle (FGS/Tracking) - Implement reacquisition logic
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                bad_pixel_map: shared::bad_pixel_map::BadPixelMap::empty(),
                minimum_bad_pixel_distance: 5.0,
            },
            roi_size: 32,
            max_reacquisition_attempts: 5,
            centroid_radius_multiplier: 5.0,
            fwhm: 3.0,
            snr_dropout_threshold: 3.0,
        }
    }

    #[test]
    fn test_state_transitions() {
        let mut fgs = FineGuidanceSystem::new(test_config());

        // Should start in Idle
        assert_eq!(fgs.state(), &FgsState::Idle);

        // Start FGS
        let _ = fgs.process_event(FgsEvent::StartFgs);
        assert!(matches!(fgs.state(), FgsState::Acquiring { .. }));
    }

    #[test]
    fn test_process_frame() {
        let mut fgs = FineGuidanceSystem::new(test_config());
        let dummy_frame = Array2::<u16>::zeros((100, 100));

        // Should do nothing in Idle state
        let result = fgs.process_frame(dummy_frame.view(), test_timestamp());
        assert!(result.is_ok());
        let (update, _) = result.unwrap();
        assert!(update.is_none());
    }

    #[test]
    fn test_frame_accumulation() {
        let mut fgs = FineGuidanceSystem::new(test_config());

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
        let mut fgs = FineGuidanceSystem::new(test_config());

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

        let fgs = FineGuidanceSystem::new(test_config());
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

        let fgs = FineGuidanceSystem::new(test_config());

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

//! Runner for executing FGS with camera and motion over time
//!
//! Provides functionality to run a Fine Guidance System with a camera interface
//! and pointing motion for a specified duration, collecting results.

use monocle::{FgsCallbackEvent, FgsEvent, FgsState, FineGuidanceSystem};
use shared::camera_interface::CameraInterface;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Results from a runner execution
#[derive(Debug, Clone)]
pub struct RunnerResults {
    /// Total frames processed
    pub frames_processed: usize,
    /// Number of frames where tracking was active
    pub frames_tracking: usize,
    /// Number of frames where tracking was lost
    pub frames_lost: usize,
    /// Final FGS state
    pub final_state: FgsState,
    /// Any errors encountered during execution
    pub errors: Vec<String>,
    /// All events emitted during the run
    pub events: Vec<FgsCallbackEvent>,
}

/// Run FGS for specified duration
///
/// # Arguments
/// * `fgs` - Fine Guidance System instance that owns its camera
/// * `duration` - Total duration to run
/// * `frame_interval` - Time between frames
///
/// # Returns
/// * `RunnerResults` containing execution statistics
pub fn run_fgs<C: CameraInterface>(
    fgs: &mut FineGuidanceSystem<C>,
    duration: Duration,
    frame_interval: Duration,
) -> Result<RunnerResults, Box<dyn std::error::Error>> {
    let mut results = RunnerResults {
        frames_processed: 0,
        frames_tracking: 0,
        frames_lost: 0,
        final_state: fgs.state().clone(),
        errors: Vec::new(),
        events: Vec::new(),
    };

    // Set up event collection
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();

    let callback_id = fgs.register_callback(move |event| {
        events_clone.lock().unwrap().push(event.clone());
    });

    // Calculate number of frames
    let num_frames = (duration.as_millis() / frame_interval.as_millis()) as usize;

    // Start FGS
    fgs.process_event(FgsEvent::StartFgs)?;

    // Run for specified duration
    for frame_num in 0..num_frames {
        // FGS now owns the camera and captures internally
        match fgs.process_next_frame() {
            Ok(_) => {
                results.frames_processed += 1;

                // Track state
                match fgs.state() {
                    FgsState::Tracking { .. } => results.frames_tracking += 1,
                    FgsState::Idle => results.frames_lost += 1,
                    _ => {}
                }
            }
            Err(e) => {
                results.errors.push(format!("Frame {frame_num} error: {e}"));
                // Don't break on errors, continue processing
            }
        }
    }

    results.final_state = fgs.state().clone();

    // Collect all events
    results.events = Arc::try_unwrap(events)
        .map(|mutex| mutex.into_inner().unwrap())
        .unwrap_or_else(|arc| arc.lock().unwrap().clone());

    // Deregister the callback to avoid leaking resources
    fgs.deregister_callback(callback_id);

    Ok(results)
}

/// Extended runner with callback support
///
/// Allows registering a callback that gets called after each frame is processed.
///
/// # Arguments
/// * `fgs` - Fine Guidance System instance that owns its camera
/// * `duration` - Total duration to run
/// * `frame_interval` - Time between frames
/// * `callback` - Function called after each frame with (frame_num, state, time)
pub fn run_fgs_with_callback<C, F>(
    fgs: &mut FineGuidanceSystem<C>,
    duration: Duration,
    frame_interval: Duration,
    mut callback: F,
) -> Result<RunnerResults, Box<dyn std::error::Error>>
where
    C: CameraInterface,
    F: FnMut(usize, &FgsState, Duration),
{
    let mut results = RunnerResults {
        frames_processed: 0,
        frames_tracking: 0,
        frames_lost: 0,
        final_state: fgs.state().clone(),
        errors: Vec::new(),
        events: Vec::new(),
    };

    // Set up event collection
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();

    let callback_id = fgs.register_callback(move |event| {
        events_clone.lock().unwrap().push(event.clone());
    });

    let num_frames = (duration.as_millis() / frame_interval.as_millis()) as usize;

    fgs.process_event(FgsEvent::StartFgs)?;

    for frame_num in 0..num_frames {
        let current_time =
            Duration::from_millis(frame_num as u64 * frame_interval.as_millis() as u64);

        // FGS now owns the camera and captures internally
        match fgs.process_next_frame() {
            Ok(_) => {
                results.frames_processed += 1;

                let state = fgs.state();
                match state {
                    FgsState::Tracking { .. } => results.frames_tracking += 1,
                    FgsState::Idle => results.frames_lost += 1,
                    _ => {}
                }

                // Call callback with current state
                callback(frame_num, state, current_time);
            }
            Err(e) => {
                results.errors.push(format!("Frame {frame_num}: {e}"));
            }
        }
    }

    results.final_state = fgs.state().clone();

    // Collect all events
    results.events = Arc::try_unwrap(events)
        .map(|mutex| mutex.into_inner().unwrap())
        .unwrap_or_else(|arc| arc.lock().unwrap().clone());

    // Deregister the callback to avoid leaking resources
    fgs.deregister_callback(callback_id);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion_profiles::StaticPointing;
    use crate::SimulatorCamera;
    use monocle::config::FgsConfig;
    use simulator::hardware::{SatelliteConfig, TelescopeConfig};
    use simulator::units::{Length, LengthExt, Temperature, TemperatureExt};
    use starfield::catalogs::binary_catalog::{BinaryCatalog, MinimalStar};

    fn create_test_catalog() -> BinaryCatalog {
        // Create a simple test catalog with a few bright stars close together
        // Using much smaller offsets to ensure they appear in a 256x256 frame
        BinaryCatalog::from_stars(
            vec![
                MinimalStar::new(1, 0.0, 0.0, 5.0),  // Center star
                MinimalStar::new(2, 0.01, 0.0, 6.0), // Very close, 0.01 degree offset
                MinimalStar::new(3, 0.0, 0.01, 7.0), // Very close, 0.01 degree offset
            ],
            "Test catalog",
        )
    }

    #[test]
    fn test_runner_basic() {
        // Create test camera
        // Use more reasonable telescope parameters for testing
        // Longer focal length gives smaller field of view, keeping stars closer in pixels
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.3),
            Length::from_meters(3.0), // f/10 telescope
            0.9,
        );
        let sensor = simulator::hardware::sensor::models::IMX455
            .clone()
            .with_dimensions(256, 256);
        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(0.0));
        let catalog = create_test_catalog();
        let motion = Box::new(StaticPointing::new(0.0, 0.0));
        let camera = SimulatorCamera::new(satellite, catalog, motion);

        // Create FGS
        let config = FgsConfig {
            acquisition_frames: 1,
            min_guide_star_snr: 3.0,
            max_guide_stars: 1,
            roi_size: 32,
            ..Default::default()
        };
        let mut fgs = FineGuidanceSystem::new(camera, config);

        // Run for 1 second at 10 Hz
        // Camera already has StaticPointing motion built in from create_test_catalog
        let results =
            run_fgs(&mut fgs, Duration::from_secs(1), Duration::from_millis(100)).unwrap();

        // Print debug info if test fails
        if results.frames_processed != 10 {
            eprintln!("Errors: {:?}", results.errors);
            eprintln!("Final state: {:?}", results.final_state);
            eprintln!("Events: {} events recorded", results.events.len());
            eprintln!("Events detail: {:?}", results.events);
            eprintln!("Frames processed: {}", results.frames_processed);
            eprintln!("Frames tracking: {}", results.frames_tracking);
        }

        // Should have processed 10 frames
        assert_eq!(results.frames_processed, 10);
        // Should be tracking after initial acquisition and calibration
        assert!(results.frames_tracking > 0);
    }

    #[test]
    fn test_runner_with_callback() {
        // Create test setup
        // Use more reasonable telescope parameters for testing
        // Longer focal length gives smaller field of view, keeping stars closer in pixels
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.3),
            Length::from_meters(3.0), // f/10 telescope
            0.9,
        );
        let sensor = simulator::hardware::sensor::models::IMX455
            .clone()
            .with_dimensions(256, 256);
        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(0.0));
        let catalog = create_test_catalog();
        let motion = Box::new(StaticPointing::new(0.0, 0.0));
        let camera = SimulatorCamera::new(satellite, catalog, motion);

        let config = FgsConfig {
            acquisition_frames: 1,
            ..Default::default()
        };
        let mut fgs = FineGuidanceSystem::new(camera, config);

        // Track states seen
        let mut states_seen = Vec::new();

        let results = run_fgs_with_callback(
            &mut fgs,
            Duration::from_millis(500),
            Duration::from_millis(100),
            |frame_num, state, _time| {
                states_seen.push((frame_num, state.clone()));
            },
        )
        .unwrap();

        // Should have 5 frames
        assert_eq!(results.frames_processed, 5);
        assert_eq!(states_seen.len(), 5);

        // Print states for debugging if test fails
        if states_seen.is_empty() {
            panic!("No states were seen!");
        }

        // Check that we process frames correctly
        // With acquisition_frames: 1, we should see:
        // Frame 0: Acquiring (frames_collected: 0)
        // Frame 1: Calibrating
        // Frame 2+: Tracking or Idle
        println!("States seen:");
        for (frame, state) in &states_seen {
            println!("  Frame {}: {:?}", frame, state);
        }

        // At least check we have some state progression
        assert!(!states_seen.is_empty(), "Should have seen some states");
    }

    #[test]
    fn test_callback_cleanup() {
        // Test that callbacks are properly deregistered after runner completes
        // Use more reasonable telescope parameters for testing
        // Longer focal length gives smaller field of view, keeping stars closer in pixels
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.3),
            Length::from_meters(3.0), // f/10 telescope
            0.9,
        );
        let sensor = simulator::hardware::sensor::models::IMX455
            .clone()
            .with_dimensions(256, 256);
        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(0.0));
        let catalog = create_test_catalog();
        let motion = Box::new(StaticPointing::new(0.0, 0.0));
        let camera = SimulatorCamera::new(satellite, catalog, motion);

        let config = FgsConfig {
            acquisition_frames: 1,
            ..Default::default()
        };
        let mut fgs = FineGuidanceSystem::new(camera, config);

        // Get initial callback count (should be 0)
        let initial_count = fgs.callback_count();
        assert_eq!(initial_count, 0, "Should start with no callbacks");

        // Run - this registers and deregisters a callback
        let results = run_fgs(
            &mut fgs,
            Duration::from_millis(300),
            Duration::from_millis(100),
        )
        .unwrap();

        // Verify callbacks were cleaned up
        let final_count = fgs.callback_count();
        assert_eq!(
            final_count, initial_count,
            "Callback count should return to initial value after run"
        );

        // Verify processing happened
        println!("Events collected: {}", results.events.len());
        println!("Frames processed: {}", results.frames_processed);
        assert!(results.frames_processed > 0, "Should have processed frames");

        // Run with callback function - this also registers and deregisters a callback
        let mut callback_invoked = false;
        let results2 = run_fgs_with_callback(
            &mut fgs,
            Duration::from_millis(200),
            Duration::from_millis(100),
            |_, _, _| {
                callback_invoked = true;
            },
        )
        .unwrap();

        // Verify callbacks were cleaned up again
        let final_count2 = fgs.callback_count();
        assert_eq!(
            final_count2, initial_count,
            "Callback count should return to initial value after callback run"
        );

        assert!(callback_invoked, "User callback should have been invoked");
        assert!(
            results2.frames_processed > 0,
            "Should have processed frames in callback run"
        );
    }
}

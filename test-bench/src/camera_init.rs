//! Shared camera initialization for test-bench binaries.
//!
//! Provides unified camera initialization logic that can be used by multiple binaries
//! (cam_serve, cam_track, etc.) with conditional compilation based on feature flags.

use anyhow::Context;
use clap::{Parser, ValueEnum};
use shared::camera_interface::{CameraInterface, SensorBitDepth};
use shared::image_size::PixelShape;

#[derive(Debug, Clone, ValueEnum)]
pub enum CameraType {
    Mock,
    #[cfg(feature = "playerone")]
    Poa,
    #[cfg(feature = "nsv455")]
    Nsv,
}

/// Common camera initialization arguments
///
/// Supports different camera types via feature flags:
/// - Mock camera: Always available
/// - PlayerOne cameras: Requires "playerone" feature
/// - NSV455 cameras: Requires "nsv455" feature
///
/// IMPORTANT: playerone and nsv455 features are mutually exclusive due to USB conflicts.
#[derive(Parser, Debug, Clone)]
pub struct CameraArgs {
    /// Type of camera to use
    #[arg(short = 't', long, value_enum)]
    pub camera_type: CameraType,

    // Mock camera options
    /// Width for mock camera
    #[arg(long, default_value = "1024")]
    pub width: usize,

    /// Height for mock camera
    #[arg(long, default_value = "1024")]
    pub height: usize,

    // PlayerOne camera options
    #[cfg(feature = "playerone")]
    /// PlayerOne camera ID
    #[arg(short = 'i', long, default_value = "0")]
    pub camera_id: i32,

    // NSV455 camera options
    #[cfg(feature = "nsv455")]
    /// V4L2 device path for NSV455
    #[arg(short = 'd', long, default_value = "/dev/video0")]
    pub device_path: String,
}

/// Initialize camera based on arguments
///
/// Returns a boxed CameraInterface that can be used by the caller.
pub fn initialize_camera(args: &CameraArgs) -> anyhow::Result<Box<dyn CameraInterface>> {
    match args.camera_type {
        CameraType::Mock => {
            tracing::info!("Initializing mock camera ({}x{})", args.width, args.height);
            let camera = create_mock_camera(args.width, args.height)?;
            Ok(Box::new(camera))
        }
        #[cfg(feature = "playerone")]
        CameraType::Poa => {
            tracing::info!("Initializing PlayerOne camera with ID {}", args.camera_id);
            use hardware::poa::camera::PlayerOneCamera;
            let camera = PlayerOneCamera::new(args.camera_id)
                .map_err(|e| anyhow::anyhow!("Failed to initialize POA camera: {e}"))?;
            Ok(Box::new(camera))
        }
        #[cfg(feature = "nsv455")]
        CameraType::Nsv => {
            tracing::info!("Initializing NSV455 camera at {}", args.device_path);
            use hardware::nsv455::camera::nsv455_camera::NSV455Camera;
            let camera = NSV455Camera::from_device(args.device_path.clone())
                .map_err(|e| anyhow::anyhow!("Failed to initialize NSV455 camera: {e}"))?;
            Ok(Box::new(camera))
        }
    }
}

fn create_mock_camera(
    width: usize,
    height: usize,
) -> anyhow::Result<shared::camera_interface::mock::MockCameraInterface> {
    use crate::display_patterns::apriltag;
    use shared::camera_interface::mock::MockCameraInterface;

    tracing::info!("Generating AprilTag calibration pattern...");
    tracing::info!("  Target size: {}x{}", width, height);

    let apriltag_frame = apriltag::generate_as_array(width, height)
        .context("Failed to generate AprilTag pattern")?;
    let inverted_frame = apriltag_frame.mapv(|v| 65535 - v);

    let mut camera = MockCameraInterface::new_repeating(
        PixelShape::with_width_height(width, height),
        SensorBitDepth::Bits16,
        inverted_frame,
    );
    camera
        .set_exposure(std::time::Duration::from_millis(100))
        .unwrap();

    Ok(camera)
}

//! Shared camera initialization for test-bench binaries.
//!
//! Provides unified camera initialization logic that can be used by multiple binaries
//! (cam_serve, cam_track, etc.) with conditional compilation based on feature flags.

use clap::{Args, Parser, ValueEnum};
use shared::camera_interface::{CameraInterface, SensorBitDepth};
use shared::image_size::PixelShape;
use std::time::Duration;

/// Shared exposure time argument for clap-based binaries.
///
/// Use `#[command(flatten)]` to include this in your Args struct.
/// Provides a standardized way to specify exposure time in milliseconds
/// with a convenient method to get the value as `Duration`.
#[derive(Args, Debug, Clone, Copy)]
pub struct ExposureArgs {
    #[arg(
        short = 'e',
        long,
        default_value = "10",
        help = "Exposure time in milliseconds",
        long_help = "Initial exposure time for the camera in milliseconds. Can be adjusted \
            at runtime via the web UI. Longer exposures collect more photons but reduce \
            frame rate. Typical range: 1-1000ms depending on application."
    )]
    pub exposure_ms: u64,
}

impl ExposureArgs {
    /// Returns the exposure time as a `Duration`.
    pub fn as_duration(&self) -> Duration {
        Duration::from_millis(self.exposure_ms)
    }
}

/// Optional exposure time argument with sub-millisecond precision.
///
/// Use `#[command(flatten)]` to include this in your Args struct.
/// Unlike `ExposureArgs`, this is optional (uses camera default if not specified)
/// and accepts floating-point values for sub-millisecond precision.
#[derive(Args, Debug, Clone, Copy)]
pub struct OptionalExposureArgs {
    #[arg(
        short = 'e',
        long,
        value_name = "MS",
        help = "Override camera exposure time in milliseconds",
        long_help = "Set a specific exposure time for the camera. Supports decimal values \
            for sub-millisecond precision (e.g., '0.5' for 500µs). If not specified, \
            uses the camera's default exposure setting. Longer exposures reveal more \
            thermal noise (dark current) while shorter exposures primarily measure read noise."
    )]
    pub exposure_ms: Option<f64>,
}

impl OptionalExposureArgs {
    /// Returns the exposure time as a `Duration`, if specified.
    pub fn as_duration(&self) -> Option<Duration> {
        self.exposure_ms
            .map(|ms| Duration::from_micros((ms * 1000.0) as u64))
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub enum CameraType {
    Mock,
    #[cfg(feature = "playerone")]
    Poa,
    #[cfg(feature = "nsv455")]
    Nsv,
}

/// Common camera initialization arguments.
///
/// Supports different camera types via feature flags:
/// - Mock camera: Always available (for testing)
/// - PlayerOne cameras: Requires "playerone" feature
/// - NSV455 cameras: Requires "nsv455" feature
///
/// IMPORTANT: playerone and nsv455 features are mutually exclusive due to USB conflicts.
#[derive(Parser, Debug, Clone)]
pub struct CameraArgs {
    #[arg(
        short = 't',
        long,
        value_enum,
        help = "Type of camera to use",
        long_help = "Camera type to initialize. Available types depend on compiled features:\n  \
            - mock: Simulated camera for testing (always available)\n  \
            - poa: PlayerOne astronomy camera (requires 'playerone' feature)\n  \
            - nsv: NSV455 V4L2 camera (requires 'nsv455' feature)\n\n\
            Note: playerone and nsv455 features are mutually exclusive."
    )]
    pub camera_type: CameraType,

    #[arg(
        long,
        default_value = "1024",
        help = "Width for mock camera in pixels",
        long_help = "Sensor width in pixels for the mock camera. Only used when --camera-type=mock. \
            Determines the simulated sensor resolution."
    )]
    pub width: usize,

    #[arg(
        long,
        default_value = "1024",
        help = "Height for mock camera in pixels",
        long_help = "Sensor height in pixels for the mock camera. Only used when --camera-type=mock. \
            Determines the simulated sensor resolution."
    )]
    pub height: usize,

    #[cfg(feature = "playerone")]
    #[arg(
        short = 'i',
        long,
        default_value = "0",
        help = "PlayerOne camera ID (0 = first camera)",
        long_help = "Index of the PlayerOne camera to use (0-based). Use playerone_info to \
            list connected cameras and their IDs. Only used when --camera-type=poa."
    )]
    pub camera_id: i32,

    #[cfg(feature = "nsv455")]
    #[arg(
        short = 'd',
        long,
        default_value = "/dev/video0",
        help = "V4L2 device path for NSV455 camera",
        long_help = "Linux V4L2 device path for the NSV455 camera. Typically /dev/video0 or \
            /dev/video1 depending on connection order. Use 'v4l2-ctl --list-devices' to \
            find available devices. Only used when --camera-type=nsv."
    )]
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
    use ndarray::Array2;
    use shared::camera_interface::mock::MockCameraInterface;

    tracing::info!("Creating mock camera with checkerboard pattern...");
    tracing::info!("  Target size: {}x{}", width, height);

    // Generate a simple checkerboard pattern for testing
    let checker_size = 64;
    let mut frame = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let checker_x = x / checker_size;
            let checker_y = y / checker_size;
            let is_white = (checker_x + checker_y) % 2 == 0;
            frame[[y, x]] = if is_white { 60000u16 } else { 5000u16 };
        }
    }

    let mut camera = MockCameraInterface::new_repeating(
        PixelShape::with_width_height(width, height),
        SensorBitDepth::Bits16,
        frame,
    );
    camera
        .set_exposure(std::time::Duration::from_millis(100))
        .unwrap();

    Ok(camera)
}

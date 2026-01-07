use anyhow::Result;
use image::{ImageBuffer, Rgb};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use crate::display_patterns as patterns;
use patterns::motion_profile::MotionPoint;

/// Unified pattern configuration for all calibration patterns.
/// Used by both CLI (calibrate) and web server (calibrate_serve).
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PatternConfig {
    // Basic patterns (available in both CLI and web)
    April,
    Check {
        checker_size: u32,
    },
    Usaf,
    Static {
        pixel_size: u32,
    },
    Pixel,
    CirclingPixel {
        orbit_count: u32,
        orbit_radius_percent: u32,
    },
    Uniform {
        level: u8,
    },
    WigglingGaussian {
        fwhm: f64,
        wiggle_radius: f64,
        intensity: f64,
    },
    PixelGrid {
        spacing: u32,
    },
    SiemensStar {
        spokes: u32,
    },

    // Advanced patterns (CLI only - require external resources)
    // These are skipped in serde because they contain non-serializable state
    #[serde(skip)]
    MotionProfile {
        base_image: image::RgbImage,
        motion_profile: Vec<MotionPoint>,
        motion_scale: f64,
    },
    #[serde(skip)]
    GyroWalk {
        base_image: image::RgbImage,
        gyro_state: Arc<Mutex<patterns::gyro_walk::GyroWalkState>>,
        frame_rate_hz: f64,
    },
    #[serde(skip)]
    OpticalCalibration {
        runner: Arc<Mutex<patterns::optical_calibration::CalibrationRunner>>,
        pattern_size: shared::image_size::PixelShape,
    },
    #[serde(skip)]
    RemoteControlled {
        state: Arc<Mutex<patterns::remote_controlled::RemotePatternState>>,
        pattern_size: shared::image_size::PixelShape,
    },
}

impl std::fmt::Debug for PatternConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::April => write!(f, "April"),
            Self::Check { checker_size } => f
                .debug_struct("Check")
                .field("checker_size", checker_size)
                .finish(),
            Self::Usaf => write!(f, "Usaf"),
            Self::Static { pixel_size } => f
                .debug_struct("Static")
                .field("pixel_size", pixel_size)
                .finish(),
            Self::Pixel => write!(f, "Pixel"),
            Self::CirclingPixel {
                orbit_count,
                orbit_radius_percent,
            } => f
                .debug_struct("CirclingPixel")
                .field("orbit_count", orbit_count)
                .field("orbit_radius_percent", orbit_radius_percent)
                .finish(),
            Self::Uniform { level } => f.debug_struct("Uniform").field("level", level).finish(),
            Self::WigglingGaussian {
                fwhm,
                wiggle_radius,
                intensity,
            } => f
                .debug_struct("WigglingGaussian")
                .field("fwhm", fwhm)
                .field("wiggle_radius", wiggle_radius)
                .field("intensity", intensity)
                .finish(),
            Self::PixelGrid { spacing } => f
                .debug_struct("PixelGrid")
                .field("spacing", spacing)
                .finish(),
            Self::SiemensStar { spokes } => f
                .debug_struct("SiemensStar")
                .field("spokes", spokes)
                .finish(),
            Self::MotionProfile { motion_scale, .. } => f
                .debug_struct("MotionProfile")
                .field("motion_scale", motion_scale)
                .finish_non_exhaustive(),
            Self::GyroWalk { frame_rate_hz, .. } => f
                .debug_struct("GyroWalk")
                .field("frame_rate_hz", frame_rate_hz)
                .finish_non_exhaustive(),
            Self::OpticalCalibration { pattern_size, .. } => f
                .debug_struct("OpticalCalibration")
                .field("pattern_size", pattern_size)
                .finish_non_exhaustive(),
            Self::RemoteControlled { pattern_size, .. } => f
                .debug_struct("RemoteControlled")
                .field("pattern_size", pattern_size)
                .finish_non_exhaustive(),
        }
    }
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self::April
    }
}

impl PatternConfig {
    /// Returns true if this pattern requires per-frame regeneration.
    pub fn is_animated(&self) -> bool {
        matches!(
            self,
            Self::Static { .. }
                | Self::CirclingPixel { .. }
                | Self::WigglingGaussian { .. }
                | Self::MotionProfile { .. }
                | Self::GyroWalk { .. }
                | Self::OpticalCalibration { .. }
                | Self::RemoteControlled { .. }
        )
    }

    /// Returns the display name for this pattern.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::April => "AprilTag Array",
            Self::Check { .. } => "Checkerboard",
            Self::Usaf => "USAF-1951 Target",
            Self::Static { .. } => "Digital Static",
            Self::Pixel => "Center Pixel",
            Self::CirclingPixel { .. } => "Circling Pixel",
            Self::Uniform { .. } => "Uniform Screen",
            Self::WigglingGaussian { .. } => "Wiggling Gaussian",
            Self::PixelGrid { .. } => "Pixel Grid",
            Self::SiemensStar { .. } => "Siemens Star",
            Self::MotionProfile { .. } => "Motion Profile",
            Self::GyroWalk { .. } => "Gyro Walk",
            Self::OpticalCalibration { .. } => "Optical Calibration",
            Self::RemoteControlled { .. } => "Remote Controlled",
        }
    }

    /// Generate the pattern as an image.
    pub fn generate(&self, width: u32, height: u32) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        match self {
            Self::April => patterns::apriltag::generate(width, height),
            Self::Check { checker_size } => Ok(patterns::checkerboard::generate(
                width,
                height,
                *checker_size,
            )),
            Self::Usaf => patterns::usaf::generate(width, height),
            Self::Static { pixel_size } => {
                Ok(patterns::static_noise::generate(width, height, *pixel_size))
            }
            Self::Pixel => Ok(patterns::pixel::generate(width, height)),
            Self::CirclingPixel {
                orbit_count,
                orbit_radius_percent,
            } => Ok(patterns::circling_pixel::generate(
                width,
                height,
                *orbit_count,
                *orbit_radius_percent,
            )),
            Self::Uniform { level } => Ok(patterns::uniform::generate(width, height, *level)),
            Self::WigglingGaussian {
                fwhm,
                wiggle_radius,
                intensity,
            } => Ok(patterns::wiggling_gaussian::generate(
                width,
                height,
                *fwhm,
                *wiggle_radius,
                *intensity,
            )),
            Self::PixelGrid { spacing } => {
                Ok(patterns::pixel_grid::generate(width, height, *spacing))
            }
            Self::SiemensStar { spokes } => {
                Ok(patterns::siemens_star::generate(width, height, *spokes))
            }
            // Advanced patterns start with black - animation fills them in
            Self::MotionProfile { .. }
            | Self::GyroWalk { .. }
            | Self::OpticalCalibration { .. }
            | Self::RemoteControlled { .. } => {
                Ok(ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0])))
            }
        }
    }

    /// Generate the pattern directly into a buffer (for animated patterns).
    pub fn generate_into_buffer(&self, buffer: &mut [u8], width: u32, height: u32) {
        use std::time::SystemTime;

        match self {
            Self::Static { pixel_size } => {
                patterns::static_noise::generate_into_buffer(buffer, width, height, *pixel_size);
            }
            Self::CirclingPixel {
                orbit_count,
                orbit_radius_percent,
            } => {
                patterns::circling_pixel::generate_into_buffer(
                    buffer,
                    width,
                    height,
                    *orbit_count,
                    *orbit_radius_percent,
                );
            }
            Self::WigglingGaussian {
                fwhm,
                wiggle_radius,
                intensity,
            } => {
                patterns::wiggling_gaussian::generate_into_buffer(
                    buffer,
                    width,
                    height,
                    *fwhm,
                    *wiggle_radius,
                    *intensity,
                );
            }
            Self::MotionProfile {
                base_image,
                motion_profile,
                motion_scale,
            } => {
                let elapsed = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap();
                patterns::motion_profile::generate_into_buffer(
                    buffer,
                    width,
                    height,
                    base_image,
                    motion_profile,
                    elapsed,
                    *motion_scale,
                );
            }
            Self::GyroWalk {
                base_image,
                gyro_state,
                frame_rate_hz,
            } => {
                let elapsed = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap();
                let frame_interval = std::time::Duration::from_secs_f64(1.0 / frame_rate_hz);
                patterns::gyro_walk::generate_into_buffer(
                    buffer,
                    width,
                    height,
                    base_image,
                    gyro_state,
                    elapsed,
                    frame_interval,
                );
            }
            Self::OpticalCalibration {
                runner,
                pattern_size,
            } => {
                patterns::optical_calibration::generate_into_buffer(buffer, *pattern_size, runner);
            }
            Self::RemoteControlled {
                state,
                pattern_size,
            } => {
                patterns::remote_controlled::generate_into_buffer(buffer, *pattern_size, state);
            }
            // Non-animated patterns don't use buffer generation
            _ => {}
        }
    }

    /// Apply color inversion to an image.
    pub fn apply_invert(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>) {
        for pixel in img.pixels_mut() {
            pixel[0] = 255 - pixel[0];
            pixel[1] = 255 - pixel[1];
            pixel[2] = 255 - pixel[2];
        }
    }

    /// Apply color inversion to a buffer.
    pub fn apply_invert_buffer(buffer: &mut [u8]) {
        for byte in buffer.iter_mut() {
            *byte = 255 - *byte;
        }
    }
}

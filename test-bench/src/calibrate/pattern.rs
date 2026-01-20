use anyhow::Result;
use image::{ImageBuffer, Rgb};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::display_patterns as patterns;
use crate::display_patterns::{CircularMotion, MotionTrajectory, Position2D};

/// Unified pattern configuration for all calibration patterns.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PatternConfig {
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
            Self::RemoteControlled { pattern_size, .. } => f
                .debug_struct("RemoteControlled")
                .field("pattern_size", pattern_size)
                .finish_non_exhaustive(),
        }
    }
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self::Uniform { level: 0 }
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
            // Animated patterns start with black - animation fills them in
            Self::RemoteControlled { .. } => {
                Ok(ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0])))
            }
        }
    }

    /// Generate the pattern directly into a buffer (for animated patterns).
    pub fn generate_into_buffer(&self, buffer: &mut [u8], width: u32, height: u32) {
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

impl MotionTrajectory for PatternConfig {
    fn position_at(
        &self,
        elapsed: Duration,
        display_width: u32,
        display_height: u32,
    ) -> Position2D {
        match self {
            Self::WigglingGaussian { wiggle_radius, .. } => {
                let motion = CircularMotion::new(*wiggle_radius, 10.0);
                motion.position_at(elapsed)
            }
            Self::CirclingPixel {
                orbit_radius_percent,
                ..
            } => {
                // Convert % of FOV to pixels
                let fov_size = display_width.min(display_height) as f64;
                let radius_px = fov_size * (*orbit_radius_percent as f64 / 200.0);
                let motion = CircularMotion::new(radius_px, 60.0);
                motion.position_at(elapsed)
            }
            // Static patterns have no motion
            _ => Position2D { x: 0.0, y: 0.0 },
        }
    }
}

//! Motion trajectory trait for patterns with moving elements.
//!
//! Patterns that move (like WigglingGaussian or CirclingPixel) implement this trait
//! to expose their position for gyro simulation and other motion-aware systems.

use shared::image_size::PixelShape;
use std::time::Duration;

/// 2D position in pixel coordinates (relative to display center).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position2D {
    pub x: f64,
    pub y: f64,
}

/// Trait for patterns that have predictable motion trajectories.
///
/// Implement this for any pattern that moves in a predictable way,
/// allowing gyro simulation to emit synchronized angle data.
pub trait MotionTrajectory {
    /// Get the position at a given elapsed time from pattern start.
    ///
    /// Returns position in pixels relative to display center.
    ///
    /// # Arguments
    /// * `elapsed` - Time since the motion started (typically since pattern activation)
    /// * `display_size` - Display dimensions in pixels (for computing radius from % values)
    fn position_at(&self, elapsed: Duration, display_size: PixelShape) -> Position2D;
}

/// Circular motion parameters (shared between WigglingGaussian, CirclingPixel, etc.)
#[derive(Debug, Clone, Copy)]
pub struct CircularMotion {
    /// Radius of the circular path in pixels
    pub radius_px: f64,
    /// Period of one complete rotation in seconds
    pub period_s: f64,
    /// Phase offset in radians (0 = start at +X direction)
    pub phase_offset: f64,
}

impl CircularMotion {
    /// Create a new circular motion with default phase.
    pub fn new(radius_px: f64, period_s: f64) -> Self {
        Self {
            radius_px,
            period_s,
            phase_offset: 0.0,
        }
    }

    /// Compute position at a given elapsed time.
    pub fn position_at(&self, elapsed: Duration) -> Position2D {
        let t = elapsed.as_secs_f64();
        let angle = (t / self.period_s) * std::f64::consts::TAU + self.phase_offset;
        Position2D {
            x: self.radius_px * angle.cos(),
            y: self.radius_px * angle.sin(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_circular_motion_position() {
        let motion = CircularMotion::new(10.0, 10.0);

        // At t=0, should be at (radius, 0)
        let pos = motion.position_at(Duration::from_secs(0));
        assert_abs_diff_eq!(pos.x, 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pos.y, 0.0, epsilon = 1e-10);

        // At t=2.5 (1/4 period), should be at (0, radius)
        let pos = motion.position_at(Duration::from_secs_f64(2.5));
        assert_abs_diff_eq!(pos.x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pos.y, 10.0, epsilon = 1e-10);
    }
}

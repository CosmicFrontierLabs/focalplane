//! Motion models for simulating various types of object movement
//!
//! This module provides traits and implementations for different types of
//! motion models used in astronomical simulations. These models produce
//! quaternion orientations as a function of time.

use std::f64::consts::PI;

use nalgebra::Vector3;

use super::quaternion::Quaternion;

/// A trait defining a motion model that produces quaternion orientation as a function of time
pub trait MotionModel {
    /// Calculate the quaternion orientation at the given time in seconds
    fn orientation_at(&self, time_seconds: f64) -> Quaternion;
}

/// A motion model that simulates an object spinning around the X axis at a constant rate
pub struct XAxisSpinner {
    /// Angular velocity in radians per second
    angular_velocity: f64,
}

impl XAxisSpinner {
    /// Create a new XAxisSpinner with the given rotation rate in RPM (revolutions per minute)
    pub fn new(rpm: f64) -> Self {
        // Convert RPM to radians per second: rpm * (2π rad/rev) * (1 min / 60 sec)
        let angular_velocity = rpm * 2.0 * PI / 60.0;
        Self { angular_velocity }
    }
}

impl MotionModel for XAxisSpinner {
    fn orientation_at(&self, time_seconds: f64) -> Quaternion {
        // Calculate the angle at the given time
        let angle = self.angular_velocity * time_seconds;

        // Create a rotation quaternion around the X axis
        let axis = Vector3::new(1.0, 0.0, 0.0);
        Quaternion::from_axis_angle(&axis, angle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    #[test]
    fn test_x_axis_spinner_creation() {
        let spinner = XAxisSpinner::new(1.0); // 1 RPM
        let expected_angular_velocity = 2.0 * PI / 60.0; // 2π/60 rad/s
        assert_relative_eq!(
            spinner.angular_velocity,
            expected_angular_velocity,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_x_axis_spinner_orientation() {
        let spinner = XAxisSpinner::new(1.0); // 1 RPM

        // Test initial orientation (t=0)
        let initial = spinner.orientation_at(0.0);
        assert_eq!(initial, Quaternion::identity());

        // Test after quarter rotation (15 seconds at 1 RPM)
        let quarter_turn = spinner.orientation_at(15.0);
        let expected_angle = PI / 2.0; // 90 degrees
        let expected = Quaternion::from_axis_angle(&Vector3::new(1.0, 0.0, 0.0), expected_angle);

        assert_relative_eq!(quarter_turn.w, expected.w, epsilon = 1e-10);
        assert_relative_eq!(quarter_turn.x, expected.x, epsilon = 1e-10);
        assert_relative_eq!(quarter_turn.y, expected.y, epsilon = 1e-10);
        assert_relative_eq!(quarter_turn.z, expected.z, epsilon = 1e-10);

        // Test vector rotation - should rotate (0,1,0) to (0,0,1) after quarter turn
        let v = Vector3::new(0.0, 1.0, 0.0);
        let rotated = quarter_turn.rotate_vector(&v);

        assert_relative_eq!(rotated[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_full_rotation() {
        let spinner = XAxisSpinner::new(1.0); // 1 RPM

        // Test after full rotation (60 seconds at 1 RPM)
        let full_turn = spinner.orientation_at(60.0);

        // Should be approximately the identity quaternion again
        // Note: After a full rotation, we can get either (1,0,0,0) or (-1,0,0,0)
        // since both represent the same rotation but with different signs
        let w_abs = full_turn.w.abs();
        assert_relative_eq!(w_abs, 1.0, epsilon = 1e-10);
        assert_relative_eq!(full_turn.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(full_turn.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(full_turn.z, 0.0, epsilon = 1e-10);
    }
}

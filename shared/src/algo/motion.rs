//! Motion models for simulating various types of object movement
//!
//! This module provides traits and implementations for different types of
//! motion models used in astronomical simulations. These models produce
//! quaternion orientations as a function of time.

use std::f64::consts::PI;

use nalgebra::Vector3;

use crate::units::{Angle, AngleExt};
use meter_math::Quaternion;

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

/// A motion model that simulates high-frequency small amplitude wobble in X-Y plane
/// This can be used to model telescope jitter, tracking errors, or atmospheric effects
pub struct XYWobble {
    /// Amplitude of wobble in radians
    amplitude_rad: f64,
    /// Frequency of wobble in Hertz
    frequency_hz: f64,
    /// Phase offset for X axis wobble in radians
    x_phase_offset: f64,
    /// Phase offset for Y axis wobble in radians
    y_phase_offset: f64,
}

impl XYWobble {
    /// Create a new XYWobble with the given amplitude and frequency in Hz
    pub fn new(amplitude: Angle, frequency_hz: f64) -> Self {
        let amplitude_rad = amplitude.as_radians();

        // Default phase offset puts X and Y 90 degrees out of phase for elliptical motion
        Self {
            amplitude_rad,
            frequency_hz,
            x_phase_offset: 0.0,
            y_phase_offset: PI / 2.0,
        }
    }

    /// Create a new XYWobble with the given amplitude in arcseconds and frequency in Hz (backward compatibility)
    pub fn new_arcsec(amplitude_arcsec: f64, frequency_hz: f64) -> Self {
        let amplitude = Angle::from_arcseconds(amplitude_arcsec);
        Self::new(amplitude, frequency_hz)
    }

    /// Create a new XYWobble with custom phase offsets
    ///
    /// This allows for more complex wobble patterns:
    /// - When x_phase_offset == y_phase_offset: linear wobble along a line
    /// - When x_phase_offset - y_phase_offset == π/2: circular/elliptical wobble
    /// - Other values create more complex Lissajous patterns
    pub fn with_phase_offsets(
        amplitude: Angle,
        frequency_hz: f64,
        x_phase_offset: f64,
        y_phase_offset: f64,
    ) -> Self {
        let amplitude_rad = amplitude.as_radians();

        Self {
            amplitude_rad,
            frequency_hz,
            x_phase_offset,
            y_phase_offset,
        }
    }

    /// Create a new XYWobble with custom phase offsets using arcsecond amplitude (backward compatibility)
    pub fn with_phase_offsets_arcsec(
        amplitude_arcsec: f64,
        frequency_hz: f64,
        x_phase_offset: f64,
        y_phase_offset: f64,
    ) -> Self {
        let amplitude = Angle::from_arcseconds(amplitude_arcsec);
        Self::with_phase_offsets(amplitude, frequency_hz, x_phase_offset, y_phase_offset)
    }
}

impl MotionModel for XYWobble {
    fn orientation_at(&self, time_seconds: f64) -> Quaternion {
        // Calculate the angular position at the given time for each axis
        let omega = 2.0 * PI * self.frequency_hz;
        let x_angle = self.amplitude_rad * (omega * time_seconds + self.x_phase_offset).sin();
        let y_angle = self.amplitude_rad * (omega * time_seconds + self.y_phase_offset).sin();

        // Create rotation quaternions for each axis
        // For small angles, we can approximate by simple composition
        let x_rotation = Quaternion::from_axis_angle(&Vector3::new(1.0, 0.0, 0.0), y_angle);
        let y_rotation = Quaternion::from_axis_angle(&Vector3::new(0.0, 1.0, 0.0), -x_angle);

        // Combine rotations (apply y rotation first, then x rotation)
        x_rotation * y_rotation
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

    #[test]
    fn test_xy_wobble_creation() {
        // Test with 100 milliarcseconds and 3Hz
        let amplitude_mas = 100.0; // 100 milliarcsec
        let frequency_hz = 3.0;
        let amplitude = Angle::from_milliarcseconds(amplitude_mas);
        let wobble = XYWobble::new(amplitude, frequency_hz);

        // Check internal values
        let expected_amplitude_rad = amplitude.as_radians();
        assert_relative_eq!(
            wobble.amplitude_rad,
            expected_amplitude_rad,
            epsilon = 1e-15
        );
        assert_relative_eq!(wobble.frequency_hz, frequency_hz, epsilon = 1e-15);
        assert_relative_eq!(wobble.x_phase_offset, 0.0, epsilon = 1e-15);
        assert_relative_eq!(wobble.y_phase_offset, PI / 2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_xy_wobble_with_phase_offsets() {
        let amplitude_arcsec = 0.1; // 100 milliarcsec
        let frequency_hz = 3.0;
        let x_phase = 0.1;
        let y_phase = 0.2;

        let amplitude = Angle::from_arcseconds(amplitude_arcsec);
        let wobble = XYWobble::with_phase_offsets(amplitude, frequency_hz, x_phase, y_phase);

        assert_relative_eq!(wobble.x_phase_offset, x_phase, epsilon = 1e-15);
        assert_relative_eq!(wobble.y_phase_offset, y_phase, epsilon = 1e-15);
    }

    #[test]
    fn test_xy_wobble_orientation() {
        let amplitude_arcsec = 0.1; // 100 milliarcsec
        let frequency_hz = 3.0;
        let amplitude = Angle::from_arcseconds(amplitude_arcsec);
        let wobble = XYWobble::new(amplitude, frequency_hz);

        // Test initial orientation (t=0)
        let initial = wobble.orientation_at(0.0);

        // At t=0, sin(phase) of X is 0, and sin(π/2) of Y is 1
        // So we should see a small rotation around X axis of amplitude*sin(π/2)
        let expected_amplitude_rad = amplitude.as_radians();
        let expected_initial =
            Quaternion::from_axis_angle(&Vector3::new(1.0, 0.0, 0.0), expected_amplitude_rad);

        assert_relative_eq!(initial.w, expected_initial.w, epsilon = 1e-10);
        assert_relative_eq!(initial.x, expected_initial.x, epsilon = 1e-10);
        assert_relative_eq!(initial.y, expected_initial.y, epsilon = 1e-10);
        assert_relative_eq!(initial.z, expected_initial.z, epsilon = 1e-10);

        // Test at t = 1/(4*frequency) - quarter period - max X rotation, zero Y rotation
        let quarter_period = 0.25 / frequency_hz;
        let quarter_wobble = wobble.orientation_at(quarter_period);

        // Apply to a test vector pointing in Z direction
        let v = Vector3::new(0.0, 0.0, 1.0);
        let rotated = quarter_wobble.rotate_vector(&v);

        // Should rotate slightly around Y (shifting Z toward -X)
        assert!(rotated[0] < 0.0); // X component should be negative
        assert_relative_eq!(rotated[1], 0.0, epsilon = 1e-10); // Y should be unchanged
        assert!(rotated[2] < 1.0); // Z should be slightly less than 1
    }

    #[test]
    fn test_xy_wobble_circular_path() {
        let amplitude_arcsec = 1.0; // Use larger amplitude to make test more reliable
        let frequency_hz = 1.0;
        let amplitude = Angle::from_arcseconds(amplitude_arcsec);
        let wobble = XYWobble::new(amplitude, frequency_hz);

        // Test vector pointing in Z direction
        let v = Vector3::new(0.0, 0.0, 1.0);

        // Sample multiple points in time over one period
        let period = 1.0 / frequency_hz;
        let num_samples = 8;

        let mut positions = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f64 * period / num_samples as f64;
            let orientation = wobble.orientation_at(t);
            let rotated = orientation.rotate_vector(&v);
            positions.push((rotated[0], rotated[1]));
        }

        // Verify that rotated positions trace approximately circular path in X-Y plane
        // For circular motion, consecutive points should have similar distances from origin
        let distances: Vec<f64> = positions
            .iter()
            .map(|(x, y)| (x * x + y * y).sqrt())
            .collect();

        // All distances should be approximately equal for circular motion
        let avg_distance: f64 = distances.iter().sum::<f64>() / distances.len() as f64;

        for &dist in &distances {
            // Allow some tolerance for numerical approximation
            assert_relative_eq!(dist, avg_distance, epsilon = 1e-5);
        }
    }
}

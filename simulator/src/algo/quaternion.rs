//! Quaternion implementation for 3D rotations
//!
//! This module provides a quaternion implementation for representing and
//! applying 3D rotations, particularly useful for modeling star blur kernels
//! with angular shifts and rotations.

use nalgebra::{Matrix3, Vector3};
use std::ops::{Add, Mul};

/// A quaternion representing a rotation in 3D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    /// Real/scalar component (w)
    pub w: f64,
    /// First complex component (i)
    pub x: f64,
    /// Second complex component (j)
    pub y: f64,
    /// Third complex component (k)
    pub z: f64,
}

impl Quaternion {
    /// Create a new quaternion
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Create an identity quaternion (no rotation)
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create a quaternion from axis-angle representation
    pub fn from_axis_angle(axis: &Vector3<f64>, angle: f64) -> Self {
        let half_angle = angle / 2.0;
        let sin_half_angle = half_angle.sin();

        Self {
            w: half_angle.cos(),
            x: axis[0] * sin_half_angle,
            y: axis[1] * sin_half_angle,
            z: axis[2] * sin_half_angle,
        }
    }

    /// Create a quaternion from Euler angles (ZYX convention)
    pub fn from_euler_angles(roll: f64, pitch: f64, yaw: f64) -> Self {
        // Convert to radians
        let roll_rad = roll.to_radians();
        let pitch_rad = pitch.to_radians();
        let yaw_rad = yaw.to_radians();

        // Calculate half angles
        let cy = (yaw_rad * 0.5).cos();
        let sy = (yaw_rad * 0.5).sin();
        let cp = (pitch_rad * 0.5).cos();
        let sp = (pitch_rad * 0.5).sin();
        let cr = (roll_rad * 0.5).cos();
        let sr = (roll_rad * 0.5).sin();

        Self {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
    }

    /// Calculate the norm (magnitude) of the quaternion
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize the quaternion to unit length
    pub fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm.abs() < 1e-10 {
            Self::identity()
        } else {
            Self {
                w: self.w / norm,
                x: self.x / norm,
                y: self.y / norm,
                z: self.z / norm,
            }
        }
    }

    /// Calculate the conjugate of the quaternion
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Calculate the inverse of the quaternion
    pub fn inverse(&self) -> Self {
        let norm_squared = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
        if norm_squared < 1e-10 {
            Self::identity()
        } else {
            let inv_norm_squared = 1.0 / norm_squared;
            Self {
                w: self.w * inv_norm_squared,
                x: -self.x * inv_norm_squared,
                y: -self.y * inv_norm_squared,
                z: -self.z * inv_norm_squared,
            }
        }
    }

    /// Rotate a 3D vector using this quaternion
    pub fn rotate_vector(&self, v: &Vector3<f64>) -> Vector3<f64> {
        // Convert vector to pure quaternion (w=0)
        let v_quat = Quaternion::new(0.0, v[0], v[1], v[2]);

        // Perform rotation: q * v * q^(-1)
        let rotated = *self * v_quat * self.conjugate();

        // Extract vector part
        Vector3::new(rotated.x, rotated.y, rotated.z)
    }

    /// Convert quaternion to 3x3 rotation matrix
    pub fn to_rotation_matrix(&self) -> Matrix3<f64> {
        let q = self.normalize();

        // Extract components for readability
        let w = q.w;
        let x = q.x;
        let y = q.y;
        let z = q.z;

        // Calculate matrix elements
        let xx = x * x;
        let xy = x * y;
        let xz = x * z;
        let xw = x * w;

        let yy = y * y;
        let yz = y * z;
        let yw = y * w;

        let zz = z * z;
        let zw = z * w;

        // Construct the rotation matrix
        Matrix3::new(
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - zw),
            2.0 * (xz + yw),
            2.0 * (xy + zw),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - xw),
            2.0 * (xz - yw),
            2.0 * (yz + xw),
            1.0 - 2.0 * (xx + yy),
        )
    }

    /// Convert quaternion to Euler angles (roll, pitch, yaw) in radians
    /// Returns (roll, pitch, yaw) where:
    /// - roll: rotation around x-axis
    /// - pitch: rotation around y-axis  
    /// - yaw: rotation around z-axis
    pub fn euler_angles(&self) -> (f64, f64, f64) {
        let q = self.normalize();

        let w = q.w;
        let x = q.x;
        let y = q.y;
        let z = q.z;

        // Calculate Euler angles from quaternion
        // Roll (x-axis rotation)
        let sin_roll = 2.0 * (w * x + y * z);
        let cos_roll = 1.0 - 2.0 * (x * x + y * y);
        let roll = sin_roll.atan2(cos_roll);

        // Pitch (y-axis rotation)
        let sin_pitch = 2.0 * (w * y - z * x);
        let pitch = if sin_pitch.abs() >= 1.0 {
            std::f64::consts::PI / 2.0 * sin_pitch.signum() // Use 90 degrees if out of range
        } else {
            sin_pitch.asin()
        };

        // Yaw (z-axis rotation)
        let sin_yaw = 2.0 * (w * z + x * y);
        let cos_yaw = 1.0 - 2.0 * (y * y + z * z);
        let yaw = sin_yaw.atan2(cos_yaw);

        (roll, pitch, yaw)
    }
}

// Quaternion multiplication
impl Mul for Quaternion {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

// Quaternion addition
impl Add for Quaternion {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quaternion_identity() {
        let q = Quaternion::identity();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
    }

    #[test]
    fn test_quaternion_norm() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let expected_norm = (1.0f64 + 4.0 + 9.0 + 16.0).sqrt();
        assert_relative_eq!(q.norm(), expected_norm);
    }

    #[test]
    fn test_quaternion_normalization() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q_normalized = q.normalize();

        // The norm of a normalized quaternion should be 1.0
        assert_relative_eq!(q_normalized.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_conjugate() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q_conj = q.conjugate();

        assert_eq!(q_conj.w, 1.0);
        assert_eq!(q_conj.x, -2.0);
        assert_eq!(q_conj.y, -3.0);
        assert_eq!(q_conj.z, -4.0);
    }

    #[test]
    fn test_quaternion_inverse() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0).normalize();
        let q_inv = q.inverse();

        // Check that q * q_inv = identity
        let identity = q * q_inv;
        assert_relative_eq!(identity.w, 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_multiplication() {
        // Test with well-known quaternion multiplication
        let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);

        let result = q1 * q2;

        // Expected values verified against standard quaternion multiplication
        assert_eq!(result.w, 1.0 * 5.0 - 2.0 * 6.0 - 3.0 * 7.0 - 4.0 * 8.0);
        assert_eq!(result.x, 1.0 * 6.0 + 2.0 * 5.0 + 3.0 * 8.0 - 4.0 * 7.0);
        assert_eq!(result.y, 1.0 * 7.0 - 2.0 * 8.0 + 3.0 * 5.0 + 4.0 * 6.0);
        assert_eq!(result.z, 1.0 * 8.0 + 2.0 * 7.0 - 3.0 * 6.0 + 4.0 * 5.0);
    }

    #[test]
    fn test_axis_angle_conversion() {
        // Test rotation around x-axis by 90 degrees
        let axis = Vector3::new(1.0, 0.0, 0.0);
        let angle = std::f64::consts::FRAC_PI_2; // 90 degrees

        let q = Quaternion::from_axis_angle(&axis, angle);

        // Expected values for 90-degree rotation around x-axis
        assert_relative_eq!(q.w, 0.7071067811865476, epsilon = 1e-10); // cos(π/4)
        assert_relative_eq!(q.x, 0.7071067811865475, epsilon = 1e-10); // sin(π/4)
        assert_relative_eq!(q.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(q.z, 0.0, epsilon = 1e-10);

        // Test rotation of a vector
        let v = Vector3::new(0.0, 1.0, 0.0);
        let rotated = q.rotate_vector(&v);

        // After 90 degree rotation around x, (0,1,0) should become approximately (0,0,1)
        assert_relative_eq!(rotated[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_rotation_matrix() {
        // Test conversion to rotation matrix for a 90-degree rotation around z-axis
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let angle = std::f64::consts::FRAC_PI_2; // 90 degrees

        let q = Quaternion::from_axis_angle(&axis, angle);
        let matrix = q.to_rotation_matrix();

        // Expected rotation matrix for 90-degree rotation around z-axis
        let expected = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        // Check matrix elements
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(matrix[(i, j)], expected[(i, j)], epsilon = 1e-10);
            }
        }

        // Test with a vector
        let v = Vector3::new(1.0, 0.0, 0.0);
        let rotated_matrix = matrix * v;
        let rotated_quat = q.rotate_vector(&v);

        // Both methods should give the same result (0,1,0)
        assert_relative_eq!(rotated_matrix[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated_matrix[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated_matrix[2], 0.0, epsilon = 1e-10);

        assert_relative_eq!(rotated_quat[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated_quat[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated_quat[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euler_angles_identity() {
        let q = Quaternion::identity();
        let (roll, pitch, yaw) = q.euler_angles();

        assert_relative_eq!(roll, 0.0, epsilon = 1e-10);
        assert_relative_eq!(pitch, 0.0, epsilon = 1e-10);
        assert_relative_eq!(yaw, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euler_angles_round_trip() {
        let test_cases = vec![
            (30.0, 45.0, 60.0),
            (90.0, 0.0, 0.0),
            (0.0, 0.0, 90.0),
            (-30.0, -45.0, -60.0),
            (45.0, 30.0, 15.0),
            (120.0, 60.0, 30.0),
        ];

        for (roll_deg, pitch_deg, yaw_deg) in test_cases {
            let q = Quaternion::from_euler_angles(roll_deg, pitch_deg, yaw_deg);
            let (roll_rad, pitch_rad, yaw_rad) = q.euler_angles();

            let roll_recovered = roll_rad.to_degrees();
            let pitch_recovered = pitch_rad.to_degrees();
            let yaw_recovered = yaw_rad.to_degrees();

            assert_relative_eq!(roll_recovered, roll_deg, epsilon = 1e-8);
            assert_relative_eq!(pitch_recovered, pitch_deg, epsilon = 1e-8);
            assert_relative_eq!(yaw_recovered, yaw_deg, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_euler_angles_special_cases() {
        let q1 = Quaternion::from_euler_angles(180.0, 0.0, 0.0);
        let (roll1, _pitch1, _yaw1) = q1.euler_angles();
        let roll1_deg = roll1.to_degrees();
        assert!(roll1_deg.abs() - 180.0 < 1e-8 || roll1_deg.abs() < 1e-8);

        let q2 = Quaternion::from_euler_angles(0.0, 0.0, 180.0);
        let (_roll2, _pitch2, yaw2) = q2.euler_angles();
        let yaw2_deg = yaw2.to_degrees();
        assert!(yaw2_deg.abs() - 180.0 < 1e-8 || yaw2_deg.abs() < 1e-8);
    }

    #[test]
    fn test_euler_angles_gimbal_lock() {
        let q = Quaternion::from_euler_angles(0.0, 90.0, 0.0);
        let (_, pitch, _) = q.euler_angles();

        assert_relative_eq!(pitch, std::f64::consts::FRAC_PI_2, epsilon = 1e-10);

        let q2 = Quaternion::from_euler_angles(0.0, -90.0, 0.0);
        let (_, pitch2, _) = q2.euler_angles();

        assert_relative_eq!(pitch2, -std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
    }

    #[test]
    fn test_euler_angles_axis_rotations() {
        let q_x =
            Quaternion::from_axis_angle(&Vector3::new(1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_4);
        let (roll, pitch, yaw) = q_x.euler_angles();
        assert_relative_eq!(roll, std::f64::consts::FRAC_PI_4, epsilon = 1e-10);
        assert_relative_eq!(pitch, 0.0, epsilon = 1e-10);
        assert_relative_eq!(yaw, 0.0, epsilon = 1e-10);

        let q_y =
            Quaternion::from_axis_angle(&Vector3::new(0.0, 1.0, 0.0), std::f64::consts::FRAC_PI_4);
        let (roll, pitch, yaw) = q_y.euler_angles();
        assert_relative_eq!(roll, 0.0, epsilon = 1e-10);
        assert_relative_eq!(pitch, std::f64::consts::FRAC_PI_4, epsilon = 1e-10);
        assert_relative_eq!(yaw, 0.0, epsilon = 1e-10);

        let q_z =
            Quaternion::from_axis_angle(&Vector3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_4);
        let (roll, pitch, yaw) = q_z.euler_angles();
        assert_relative_eq!(roll, 0.0, epsilon = 1e-10);
        assert_relative_eq!(pitch, 0.0, epsilon = 1e-10);
        assert_relative_eq!(yaw, std::f64::consts::FRAC_PI_4, epsilon = 1e-10);
    }

    #[test]
    fn test_euler_angles_edge_cases() {
        let test_cases = vec![
            (0.0, 0.0, 0.0),
            (360.0, 0.0, 0.0),
            (0.0, 360.0, 0.0),
            (0.0, 0.0, 360.0),
            (359.9, 0.0, 0.0),
            (0.1, 0.0, 0.0),
        ];

        for (roll_deg, pitch_deg, yaw_deg) in test_cases {
            let q = Quaternion::from_euler_angles(roll_deg, pitch_deg, yaw_deg);
            let (roll_rad, pitch_rad, yaw_rad) = q.euler_angles();

            let roll_recovered = roll_rad.to_degrees();
            let pitch_recovered = pitch_rad.to_degrees();
            let yaw_recovered = yaw_rad.to_degrees();

            assert!(roll_recovered.is_finite());
            assert!(pitch_recovered.is_finite());
            assert!(yaw_recovered.is_finite());

            assert!(roll_recovered >= -180.0 && roll_recovered <= 180.0);
            assert!(pitch_recovered >= -90.0 && pitch_recovered <= 90.0);
            assert!(yaw_recovered >= -180.0 && yaw_recovered <= 180.0);
        }
    }

    #[test]
    fn test_euler_angles_consistency_with_from_euler() {
        let original_angles = vec![
            (15.0, 30.0, 45.0),
            (0.0, 45.0, 0.0),
            (90.0, 0.0, 90.0),
            (-45.0, -30.0, -15.0),
            (120.0, 60.0, 30.0),
        ];

        for (roll_deg, pitch_deg, yaw_deg) in original_angles {
            let q1 = Quaternion::from_euler_angles(roll_deg, pitch_deg, yaw_deg);
            let (roll_rad, pitch_rad, yaw_rad) = q1.euler_angles();
            let q2 = Quaternion::from_euler_angles(
                roll_rad.to_degrees(),
                pitch_rad.to_degrees(),
                yaw_rad.to_degrees(),
            );

            assert_relative_eq!(q1.w, q2.w, epsilon = 1e-10);
            assert_relative_eq!(q1.x, q2.x, epsilon = 1e-10);
            assert_relative_eq!(q1.y, q2.y, epsilon = 1e-10);
            assert_relative_eq!(q1.z, q2.z, epsilon = 1e-10);
        }
    }
}

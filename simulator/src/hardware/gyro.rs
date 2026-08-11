//! Gyroscope model with angle random walk for attitude determination simulation.
//!
//! This module provides a 3-axis gyroscope model that simulates the accumulation
//! of angle error over time due to angle random walk (ARW). ARW is caused by white
//! noise in the gyro rate output, which when integrated becomes a random walk in
//! the estimated angle.
//!
//! # Angle Random Walk Physics
//!
//! Gyroscopes measure angular velocity (rate), which must be integrated over time
//! to obtain orientation angles. White noise in the rate measurement causes the
//! integrated angle to drift randomly over time. This drift is characterized by
//! the Angle Random Walk (ARW) parameter, typically specified in degrees/√hour or
//! radians/√second.
//!
//! The angle error grows as the square root of time:
//! σ_angle = ARW × √(Δt)
//!
//! # Usage Example
//!
//! Use pre-defined models via `models::exail_astrix_ns()` or create custom gyros
//! with specific ARW parameters using `GyroModel::new()`.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::time::Duration;

/// 3-axis gyroscope model with angle random walk.
///
/// Models the accumulation of orientation error over time due to white noise
/// in the gyro rate measurements. Each axis independently accumulates error
/// based on the angle random walk parameter.
#[derive(Debug, Clone)]
pub struct GyroModel {
    /// Angle random walk coefficient in radians/√second
    /// Typical values:
    /// - Tactical grade: 0.01-0.1 deg/√hr = 2.9e-6 to 2.9e-5 rad/√s
    /// - MEMS grade: 0.1-1.0 deg/√hr = 2.9e-5 to 2.9e-4 rad/√s
    arw_rad_per_root_sec: f64,

    /// Accumulated angle error on X axis (radians)
    angle_error_x: f64,

    /// Accumulated angle error on Y axis (radians)
    angle_error_y: f64,

    /// Accumulated angle error on Z axis (radians)
    angle_error_z: f64,

    /// Random number generator
    rng: rand::rngs::StdRng,
}

impl GyroModel {
    /// Create a new gyroscope model with specified angle random walk.
    ///
    /// # Arguments
    ///
    /// * `arw_rad_per_root_sec` - Angle random walk in radians/√second
    pub fn new(arw_rad_per_root_sec: f64) -> Self {
        Self::with_seed(arw_rad_per_root_sec, 0)
    }

    /// Create a new gyroscope model with a specific random seed.
    ///
    /// Useful for reproducible simulations.
    pub fn with_seed(arw_rad_per_root_sec: f64, seed: u64) -> Self {
        Self {
            arw_rad_per_root_sec,
            angle_error_x: 0.0,
            angle_error_y: 0.0,
            angle_error_z: 0.0,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Step the gyro forward in time, accumulating random walk error.
    ///
    /// The error added to each axis is drawn from a normal distribution with
    /// standard deviation σ = ARW × √(Δt), where Δt is the time step in seconds.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time step duration
    pub fn step(&mut self, dt: Duration) {
        let dt_sec = dt.as_secs_f64();
        let sigma = self.arw_rad_per_root_sec * dt_sec.sqrt();

        let normal = Normal::new(0.0, sigma).expect("Failed to create normal distribution");

        self.angle_error_x += normal.sample(&mut self.rng);
        self.angle_error_y += normal.sample(&mut self.rng);
        self.angle_error_z += normal.sample(&mut self.rng);
    }

    /// Get the current accumulated angle errors for all three axes.
    ///
    /// Returns (x_error, y_error, z_error) in radians.
    pub fn angle_errors(&self) -> (f64, f64, f64) {
        (self.angle_error_x, self.angle_error_y, self.angle_error_z)
    }

    /// Get the angle error on the X axis in radians.
    pub fn angle_error_x(&self) -> f64 {
        self.angle_error_x
    }

    /// Get the angle error on the Y axis in radians.
    pub fn angle_error_y(&self) -> f64 {
        self.angle_error_y
    }

    /// Get the angle error on the Z axis in radians.
    pub fn angle_error_z(&self) -> f64 {
        self.angle_error_z
    }

    /// Reset all accumulated angle errors to zero.
    pub fn reset(&mut self) {
        self.angle_error_x = 0.0;
        self.angle_error_y = 0.0;
        self.angle_error_z = 0.0;
    }

    /// Get the angle random walk parameter in radians/√second.
    pub fn arw(&self) -> f64 {
        self.arw_rad_per_root_sec
    }
}

/// Pre-defined gyroscope models with realistic specifications.
pub mod models {
    use super::GyroModel;

    /// Exail Astrix NS - High-precision fiber-optic gyroscope for space applications.
    ///
    /// Specifications:
    /// - ARW: 0.005°/√h (8.73e-7 rad/√s)
    /// - Bias stability: 0.02°/h
    /// - Technology: Fiber-Optic Gyroscope (FOG)
    /// - Mass: 1.4 kg
    /// - Power: 7 W
    /// - Dimensions: 100×100×100 mm
    ///
    /// This is a space-qualified gyroscope with exceptional performance,
    /// suitable for LEO and GEO missions requiring high-accuracy attitude determination.
    pub fn exail_astrix_ns() -> GyroModel {
        // ARW: 0.005 deg/√hr
        // Convert to rad/√s: 0.005 * (π/180) / 60
        let arw_deg_per_root_hr: f64 = 0.005;
        let arw_rad_per_root_sec = arw_deg_per_root_hr.to_radians() / 60.0;
        GyroModel::new(arw_rad_per_root_sec)
    }

    /// Exail Astrix NS High Performance variant.
    ///
    /// Specifications:
    /// - ARW: 0.0025°/√h (4.36e-7 rad/√s)
    /// - Enhanced performance variant of the Astrix NS
    pub fn exail_astrix_ns_high_performance() -> GyroModel {
        // ARW: 0.0025 deg/√hr
        let arw_deg_per_root_hr: f64 = 0.0025;
        let arw_rad_per_root_sec = arw_deg_per_root_hr.to_radians() / 60.0;
        GyroModel::new(arw_rad_per_root_sec)
    }

    /// Typical MEMS gyroscope (consumer grade).
    ///
    /// Specifications:
    /// - ARW: 0.5°/√h (typical consumer MEMS)
    pub fn typical_mems() -> GyroModel {
        let arw_deg_per_root_hr: f64 = 0.5;
        let arw_rad_per_root_sec = arw_deg_per_root_hr.to_radians() / 60.0;
        GyroModel::new(arw_rad_per_root_sec)
    }

    /// Tactical grade MEMS gyroscope.
    ///
    /// Specifications:
    /// - ARW: 0.05°/√h (tactical grade MEMS)
    pub fn tactical_grade() -> GyroModel {
        let arw_deg_per_root_hr: f64 = 0.05;
        let arw_rad_per_root_sec = arw_deg_per_root_hr.to_radians() / 60.0;
        GyroModel::new(arw_rad_per_root_sec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gyro_accumulates_error() {
        let mut gyro = GyroModel::with_seed(1e-4, 42);

        let initial_errors = gyro.angle_errors();
        assert_eq!(initial_errors, (0.0, 0.0, 0.0));

        gyro.step(Duration::from_secs(1));

        let after_errors = gyro.angle_errors();
        assert_ne!(after_errors.0, 0.0);
        assert_ne!(after_errors.1, 0.0);
        assert_ne!(after_errors.2, 0.0);
    }

    #[test]
    fn test_gyro_reset() {
        let mut gyro = GyroModel::with_seed(1e-4, 42);

        gyro.step(Duration::from_secs(1));
        assert_ne!(gyro.angle_errors(), (0.0, 0.0, 0.0));

        gyro.reset();
        assert_eq!(gyro.angle_errors(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_multiple_steps_accumulate() {
        let arw = 1e-4;
        let mut gyro = GyroModel::with_seed(arw, 42);

        for _ in 0..100 {
            gyro.step(Duration::from_secs(1));
        }

        let (x, y, z) = gyro.angle_errors();
        assert_ne!(x, 0.0);
        assert_ne!(y, 0.0);
        assert_ne!(z, 0.0);
    }
}

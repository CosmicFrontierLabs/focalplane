//! Optical alignment calibration for camera/display/optics system.
//!
//! Stores the affine transformation between display coordinates and sensor coordinates,
//! as determined by closed-loop calibration.

use serde::{Deserialize, Serialize};

/// Optical alignment calibration data.
///
/// Represents the affine transformation from display pixel coordinates to sensor pixel coordinates:
/// ```text
/// sensor_x = a * display_x + b * display_y + tx
/// sensor_y = c * display_x + d * display_y + ty
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpticalAlignment {
    /// Affine transform coefficient: x contribution to sensor_x
    pub a: f64,
    /// Affine transform coefficient: y contribution to sensor_x
    pub b: f64,
    /// Affine transform coefficient: x contribution to sensor_y
    pub c: f64,
    /// Affine transform coefficient: y contribution to sensor_y
    pub d: f64,
    /// Translation offset for sensor_x
    pub tx: f64,
    /// Translation offset for sensor_y
    pub ty: f64,
    /// Timestamp when calibration was performed (Unix epoch seconds)
    pub timestamp: u64,
    /// Number of calibration points used
    pub num_points: usize,
    /// RMS residual error in pixels (if computed)
    pub rms_error: Option<f64>,
}

impl OpticalAlignment {
    /// Create a new optical alignment from affine transform parameters
    pub fn new(a: f64, b: f64, c: f64, d: f64, tx: f64, ty: f64, num_points: usize) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            a,
            b,
            c,
            d,
            tx,
            ty,
            timestamp,
            num_points,
            rms_error: None,
        }
    }

    /// Apply transform: convert display coordinates to sensor coordinates
    pub fn display_to_sensor(&self, display_x: f64, display_y: f64) -> (f64, f64) {
        let sensor_x = self.a * display_x + self.b * display_y + self.tx;
        let sensor_y = self.c * display_x + self.d * display_y + self.ty;
        (sensor_x, sensor_y)
    }

    /// Get scale factors (magnitude of column vectors)
    pub fn scale(&self) -> (f64, f64) {
        let scale_x = (self.a * self.a + self.c * self.c).sqrt();
        let scale_y = (self.b * self.b + self.d * self.d).sqrt();
        (scale_x, scale_y)
    }

    /// Get rotation angle in radians
    pub fn rotation(&self) -> f64 {
        self.c.atan2(self.a)
    }

    /// Get rotation angle in degrees
    pub fn rotation_degrees(&self) -> f64 {
        self.rotation().to_degrees()
    }

    /// Save to JSON file
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load from JSON file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

impl Default for OpticalAlignment {
    /// Default identity transform (no scaling, rotation, or translation)
    fn default() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx: 0.0,
            ty: 0.0,
            timestamp: 0,
            num_points: 0,
            rms_error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let align = OpticalAlignment::default();
        let (sx, sy) = align.display_to_sensor(100.0, 200.0);
        assert!((sx - 100.0).abs() < 1e-10);
        assert!((sy - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale() {
        let align = OpticalAlignment::new(2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 100);
        let (scale_x, scale_y) = align.scale();
        assert!((scale_x - 2.0).abs() < 1e-10);
        assert!((scale_y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation() {
        // 90 degree rotation: a=0, b=-1, c=1, d=0
        let align = OpticalAlignment::new(0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 100);
        let rot_deg = align.rotation_degrees();
        assert!((rot_deg - 90.0).abs() < 1e-10);
    }
}

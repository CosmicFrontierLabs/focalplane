//! FSM Transform - Persistent calibration transform with conversion methods
//!
//! Provides a serializable transform that maps between FSM commands (µrad) and
//! sensor positions (pixels), supporting both forward and inverse conversions.

use nalgebra::{Matrix2, Vector2};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use thiserror::Error;

use super::{invert_matrix, FsmAxisCalibration, FsmSingularMatrixError};

/// Error during transform operations
#[derive(Error, Debug)]
pub enum FsmTransformError {
    /// IO error during save/load
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Transform matrix is singular (cannot compute inverse)
    #[error("singular matrix: {0}")]
    SingularMatrix(#[from] FsmSingularMatrixError),
}

/// Persistent FSM calibration transform
///
/// Maps between FSM commands in microradians and sensor centroid positions in pixels.
/// Supports JSON serialization for persistence across sessions.
///
/// The matrix is validated to be invertible on construction/load, so all conversion
/// methods are infallible at runtime.
///
/// # Coordinate System
///
/// The transform models the relationship:
/// ```text
/// sensor_delta = fsm_to_sensor * fsm_delta
/// ```
///
/// Where:
/// - `fsm_delta` is FSM motion in µrad
/// - `sensor_delta` is centroid motion in pixels
/// - `intercept_pixels` is the reference centroid position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsmTransform {
    /// Transform matrix from FSM µrad to sensor pixels [pixels/µrad]
    /// Stored as row-major [m00, m01, m10, m11]
    fsm_to_sensor: [f64; 4],

    /// Centroid position at reference FSM position (pixels)
    intercept_pixels: [f64; 2],

    /// Calibration timestamp (ISO 8601)
    #[serde(default)]
    calibration_timestamp: Option<String>,

    /// Optional description/notes
    #[serde(default)]
    description: Option<String>,
}

impl FsmTransform {
    /// Create a new transform from calibration results
    ///
    /// # Errors
    /// Returns error if the calibration matrix is singular (cannot be inverted)
    pub fn from_calibration(calibration: &FsmAxisCalibration) -> Result<Self, FsmTransformError> {
        let fsm_to_sensor = [
            calibration.fsm_to_sensor[(0, 0)],
            calibration.fsm_to_sensor[(0, 1)],
            calibration.fsm_to_sensor[(1, 0)],
            calibration.fsm_to_sensor[(1, 1)],
        ];

        // Validate matrix is invertible
        let matrix = Matrix2::new(
            fsm_to_sensor[0],
            fsm_to_sensor[1],
            fsm_to_sensor[2],
            fsm_to_sensor[3],
        );
        let _ = invert_matrix(&matrix)?;

        Ok(Self {
            fsm_to_sensor,
            intercept_pixels: [
                calibration.intercept_pixels.x,
                calibration.intercept_pixels.y,
            ],
            calibration_timestamp: Some(chrono::Utc::now().to_rfc3339()),
            description: None,
        })
    }

    /// Create a transform with an identity matrix (useful for testing)
    pub fn identity(intercept_pixels: Vector2<f64>) -> Self {
        Self {
            fsm_to_sensor: [1.0, 0.0, 0.0, 1.0],
            intercept_pixels: [intercept_pixels.x, intercept_pixels.y],
            calibration_timestamp: None,
            description: Some("Identity transform".to_string()),
        }
    }

    /// Set an optional description for this calibration
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Load transform from a JSON file
    ///
    /// # Errors
    /// Returns error if file cannot be read, JSON is invalid, or matrix is singular
    pub fn load(path: impl AsRef<Path>) -> Result<Self, FsmTransformError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let transform: Self = serde_json::from_reader(reader)?;

        // Validate matrix is invertible on load
        let matrix = transform.fsm_to_sensor_matrix();
        let _ = invert_matrix(&matrix)?;

        Ok(transform)
    }

    /// Save transform to a JSON file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), FsmTransformError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Get the intercept (reference centroid position)
    pub fn intercept_pixels(&self) -> Vector2<f64> {
        Vector2::new(self.intercept_pixels[0], self.intercept_pixels[1])
    }

    /// Get the calibration timestamp, if available
    pub fn calibration_timestamp(&self) -> Option<&str> {
        self.calibration_timestamp.as_deref()
    }

    /// Get the description, if available
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    // ============== Internal Matrix Access ==============

    /// Get the FSM-to-sensor transform matrix (internal use)
    fn fsm_to_sensor_matrix(&self) -> Matrix2<f64> {
        Matrix2::new(
            self.fsm_to_sensor[0],
            self.fsm_to_sensor[1],
            self.fsm_to_sensor[2],
            self.fsm_to_sensor[3],
        )
    }

    /// Compute the sensor-to-FSM inverse matrix on demand
    ///
    /// This is infallible because we validate invertibility on construction/load.
    fn sensor_to_fsm_matrix(&self) -> Matrix2<f64> {
        let matrix = self.fsm_to_sensor_matrix();
        // Safe to unwrap: we validated on construction/load
        invert_matrix(&matrix).expect("matrix was validated on construction")
    }

    // ============== Conversion Functions ==============

    /// Convert FSM angle (µrad) to sensor pixel position
    ///
    /// # Arguments
    /// * `angle_x` - FSM angle on axis 1 in µrad (relative to reference)
    /// * `angle_y` - FSM angle on axis 2 in µrad (relative to reference)
    ///
    /// # Returns
    /// `(pixel_x, pixel_y)` - Expected centroid position in pixels
    pub fn angle_to_pix(&self, angle_x: f64, angle_y: f64) -> (f64, f64) {
        let delta = self.fsm_to_sensor_matrix() * Vector2::new(angle_x, angle_y);
        let intercept = self.intercept_pixels();
        (intercept.x + delta.x, intercept.y + delta.y)
    }

    /// Convert sensor pixel position to FSM angle (µrad)
    ///
    /// # Arguments
    /// * `pixel_x` - Centroid X position in pixels
    /// * `pixel_y` - Centroid Y position in pixels
    ///
    /// # Returns
    /// `(angle_x, angle_y)` - FSM angles in µrad (relative to reference)
    pub fn pix_to_angle(&self, pixel_x: f64, pixel_y: f64) -> (f64, f64) {
        let intercept = self.intercept_pixels();
        let delta = Vector2::new(pixel_x - intercept.x, pixel_y - intercept.y);
        let angle = self.sensor_to_fsm_matrix() * delta;
        (angle.x, angle.y)
    }

    /// Convert FSM angle delta (µrad) to sensor pixel delta
    ///
    /// # Arguments
    /// * `delta_angle_x` - FSM motion on axis 1 in µrad
    /// * `delta_angle_y` - FSM motion on axis 2 in µrad
    ///
    /// # Returns
    /// `(delta_pix_x, delta_pix_y)` - Expected centroid motion in pixels
    pub fn angle_delta_to_pix_delta(&self, delta_angle_x: f64, delta_angle_y: f64) -> (f64, f64) {
        let delta = self.fsm_to_sensor_matrix() * Vector2::new(delta_angle_x, delta_angle_y);
        (delta.x, delta.y)
    }

    /// Convert sensor pixel delta to FSM angle delta (µrad)
    ///
    /// # Arguments
    /// * `delta_pix_x` - Centroid motion in X pixels
    /// * `delta_pix_y` - Centroid motion in Y pixels
    ///
    /// # Returns
    /// `(delta_angle_x, delta_angle_y)` - Required FSM motion in µrad
    pub fn pix_delta_to_angle_delta(&self, delta_pix_x: f64, delta_pix_y: f64) -> (f64, f64) {
        let delta = self.sensor_to_fsm_matrix() * Vector2::new(delta_pix_x, delta_pix_y);
        (delta.x, delta.y)
    }

    /// Compute FSM correction to move centroid from current to target position
    ///
    /// # Arguments
    /// * `current_x`, `current_y` - Current centroid position in pixels
    /// * `target_x`, `target_y` - Desired centroid position in pixels
    ///
    /// # Returns
    /// `(correction_x, correction_y)` - FSM motion required in µrad
    pub fn compute_correction(
        &self,
        current_x: f64,
        current_y: f64,
        target_x: f64,
        target_y: f64,
    ) -> (f64, f64) {
        let error_x = target_x - current_x;
        let error_y = target_y - current_y;
        self.pix_delta_to_angle_delta(error_x, error_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_transform() -> FsmTransform {
        // Create a simple scaled transform: 0.1 pixels/µrad, axes aligned
        FsmTransform {
            fsm_to_sensor: [0.1, 0.0, 0.0, 0.1],
            intercept_pixels: [512.0, 512.0],
            calibration_timestamp: None,
            description: Some("Test transform".to_string()),
        }
    }

    #[test]
    fn test_identity_transform() {
        let transform = FsmTransform::identity(Vector2::new(512.0, 512.0));

        // 100 µrad delta should give 100 pixel delta (identity matrix)
        let (dx, dy) = transform.angle_delta_to_pix_delta(100.0, 50.0);
        assert_relative_eq!(dx, 100.0, epsilon = 1e-10);
        assert_relative_eq!(dy, 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_forward_conversion() {
        let transform = create_test_transform();

        // 100 µrad delta on axis 1 -> 10 pixel delta
        let (dx, dy) = transform.angle_delta_to_pix_delta(100.0, 0.0);
        assert_relative_eq!(dx, 10.0, epsilon = 1e-10);
        assert_relative_eq!(dy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_conversion() {
        let transform = create_test_transform();

        // 10 pixels delta -> 100 µrad delta
        let (dx, dy) = transform.pix_delta_to_angle_delta(10.0, 0.0);
        assert_relative_eq!(dx, 100.0, epsilon = 1e-10);
        assert_relative_eq!(dy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let transform = create_test_transform();

        let (original_x, original_y) = (150.0, 75.0);
        let (pix_x, pix_y) = transform.angle_delta_to_pix_delta(original_x, original_y);
        let (recovered_x, recovered_y) = transform.pix_delta_to_angle_delta(pix_x, pix_y);

        assert_relative_eq!(recovered_x, original_x, epsilon = 1e-10);
        assert_relative_eq!(recovered_y, original_y, epsilon = 1e-10);
    }

    #[test]
    fn test_absolute_position_conversion() {
        let transform = create_test_transform();

        // At zero angle, should be at intercept
        let (px, py) = transform.angle_to_pix(0.0, 0.0);
        assert_relative_eq!(px, 512.0, epsilon = 1e-10);
        assert_relative_eq!(py, 512.0, epsilon = 1e-10);

        // At 100 µrad on both axes, should be at intercept + 10 pixels
        let (px, py) = transform.angle_to_pix(100.0, 100.0);
        assert_relative_eq!(px, 522.0, epsilon = 1e-10);
        assert_relative_eq!(py, 522.0, epsilon = 1e-10);

        // And back
        let (ax, ay) = transform.pix_to_angle(522.0, 522.0);
        assert_relative_eq!(ax, 100.0, epsilon = 1e-10);
        assert_relative_eq!(ay, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_delta_conversions() {
        let transform = create_test_transform();

        // 100 µrad delta should give 10 pixel delta
        let (dx, dy) = transform.angle_delta_to_pix_delta(100.0, 50.0);
        assert_relative_eq!(dx, 10.0, epsilon = 1e-10);
        assert_relative_eq!(dy, 5.0, epsilon = 1e-10);

        // And back
        let (ax, ay) = transform.pix_delta_to_angle_delta(dx, dy);
        assert_relative_eq!(ax, 100.0, epsilon = 1e-10);
        assert_relative_eq!(ay, 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_correction() {
        let transform = create_test_transform();

        // Need to move from (510, 510) to (520, 515)
        // Error is (10, 5) pixels, which needs (100, 50) µrad
        let (cx, cy) = transform.compute_correction(510.0, 510.0, 520.0, 515.0);

        assert_relative_eq!(cx, 100.0, epsilon = 1e-10);
        assert_relative_eq!(cy, 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let transform = create_test_transform();

        // Use tempfile for test
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_fsm_transform.json");

        // Save
        transform.save(&temp_path).unwrap();

        // Load
        let loaded = FsmTransform::load(&temp_path).unwrap();

        // Verify matrix matches
        assert_relative_eq!(loaded.fsm_to_sensor[0], transform.fsm_to_sensor[0]);
        assert_relative_eq!(loaded.fsm_to_sensor[1], transform.fsm_to_sensor[1]);
        assert_relative_eq!(loaded.fsm_to_sensor[2], transform.fsm_to_sensor[2]);
        assert_relative_eq!(loaded.fsm_to_sensor[3], transform.fsm_to_sensor[3]);

        assert_relative_eq!(loaded.intercept_pixels[0], transform.intercept_pixels[0]);
        assert_relative_eq!(loaded.intercept_pixels[1], transform.intercept_pixels[1]);

        // Verify conversions work after load
        let (dx, dy) = loaded.angle_delta_to_pix_delta(100.0, 0.0);
        assert_relative_eq!(dx, 10.0, epsilon = 1e-10);
        assert_relative_eq!(dy, 0.0, epsilon = 1e-10);

        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_rotated_axes() {
        // Create a transform with 45° rotated axes
        // At 45°, cos = sin ≈ 0.707
        let scale = 0.1; // pixels/µrad
        let cos45 = std::f64::consts::FRAC_1_SQRT_2;
        let sin45 = cos45;

        let transform = FsmTransform {
            fsm_to_sensor: [scale * cos45, -scale * sin45, scale * sin45, scale * cos45],
            intercept_pixels: [512.0, 512.0],
            calibration_timestamp: None,
            description: None,
        };

        // Moving axis 1 by 100 µrad should move diagonally
        let (dx, dy) = transform.angle_delta_to_pix_delta(100.0, 0.0);
        assert_relative_eq!(dx, 10.0 * cos45, epsilon = 1e-10);
        assert_relative_eq!(dy, 10.0 * sin45, epsilon = 1e-10);
    }

    #[test]
    fn test_singular_matrix_rejected_on_load() {
        // Create a file with a singular matrix
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_singular_transform.json");

        let json = r#"{
            "fsm_to_sensor": [1.0, 2.0, 2.0, 4.0],
            "intercept_pixels": [512.0, 512.0]
        }"#;

        std::fs::write(&temp_path, json).unwrap();

        // Load should fail with singular matrix error
        let result = FsmTransform::load(&temp_path);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FsmTransformError::SingularMatrix(_)
        ));

        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }
}

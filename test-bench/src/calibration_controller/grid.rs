//! Grid position generation for calibration patterns.
//!
//! Re-exports the shared grid generation function for use within calibration_controller.

pub use test_bench_shared::generate_centered_grid;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_generate_centered_grid_3x3() {
        let positions = generate_centered_grid(3, 100.0, 1000, 1000);

        assert_eq!(positions.len(), 9);

        // Center point should be at display center
        let center = positions[4]; // Middle of 3x3
        assert_abs_diff_eq!(center.0, 500.0, epsilon = 1e-10);
        assert_abs_diff_eq!(center.1, 500.0, epsilon = 1e-10);

        // Top-left should be offset by -100 in both directions
        let top_left = positions[0];
        assert_abs_diff_eq!(top_left.0, 400.0, epsilon = 1e-10);
        assert_abs_diff_eq!(top_left.1, 400.0, epsilon = 1e-10);

        // Bottom-right should be offset by +100 in both directions
        let bottom_right = positions[8];
        assert_abs_diff_eq!(bottom_right.0, 600.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bottom_right.1, 600.0, epsilon = 1e-10);
    }

    #[test]
    fn test_generate_centered_grid_1x1() {
        let positions = generate_centered_grid(1, 100.0, 2560, 2560);

        assert_eq!(positions.len(), 1);
        assert_abs_diff_eq!(positions[0].0, 1280.0, epsilon = 1e-10);
        assert_abs_diff_eq!(positions[0].1, 1280.0, epsilon = 1e-10);
    }

    #[test]
    fn test_generate_centered_grid_asymmetric_display() {
        let positions = generate_centered_grid(2, 50.0, 1920, 1080);

        assert_eq!(positions.len(), 4);

        // Center of 2x2 grid should still be at display center
        let avg_x: f64 = positions.iter().map(|(x, _)| x).sum::<f64>() / 4.0;
        let avg_y: f64 = positions.iter().map(|(_, y)| y).sum::<f64>() / 4.0;
        assert_abs_diff_eq!(avg_x, 960.0, epsilon = 1e-10);
        assert_abs_diff_eq!(avg_y, 540.0, epsilon = 1e-10);
    }
}

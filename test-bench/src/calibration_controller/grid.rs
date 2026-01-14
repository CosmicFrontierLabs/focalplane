//! Grid position generation for calibration patterns.

/// Generate a centered grid of spot positions.
///
/// Creates a grid of `grid_size × grid_size` positions centered on the display,
/// with each position separated by `grid_spacing` pixels.
///
/// # Arguments
/// * `grid_size` - Number of points per row/column (e.g., 5 for 5×5 grid)
/// * `grid_spacing` - Distance in pixels between adjacent grid points
/// * `display_width` - Display width in pixels
/// * `display_height` - Display height in pixels
///
/// # Returns
/// Vector of (x, y) positions in display coordinates, ordered row by row.
pub fn generate_centered_grid(
    grid_size: usize,
    grid_spacing: f64,
    display_width: u32,
    display_height: u32,
) -> Vec<(f64, f64)> {
    let center_x = display_width as f64 / 2.0;
    let center_y = display_height as f64 / 2.0;
    let half_extent = (grid_size - 1) as f64 / 2.0;

    let mut positions = Vec::with_capacity(grid_size * grid_size);
    for row in 0..grid_size {
        for col in 0..grid_size {
            let offset_x = (col as f64 - half_extent) * grid_spacing;
            let offset_y = (row as f64 - half_extent) * grid_spacing;
            positions.push((center_x + offset_x, center_y + offset_y));
        }
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_centered_grid_3x3() {
        let positions = generate_centered_grid(3, 100.0, 1000, 1000);

        assert_eq!(positions.len(), 9);

        // Center point should be at display center
        let center = positions[4]; // Middle of 3x3
        assert!((center.0 - 500.0).abs() < 1e-10);
        assert!((center.1 - 500.0).abs() < 1e-10);

        // Top-left should be offset by -100 in both directions
        let top_left = positions[0];
        assert!((top_left.0 - 400.0).abs() < 1e-10);
        assert!((top_left.1 - 400.0).abs() < 1e-10);

        // Bottom-right should be offset by +100 in both directions
        let bottom_right = positions[8];
        assert!((bottom_right.0 - 600.0).abs() < 1e-10);
        assert!((bottom_right.1 - 600.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_centered_grid_1x1() {
        let positions = generate_centered_grid(1, 100.0, 2560, 2560);

        assert_eq!(positions.len(), 1);
        assert!((positions[0].0 - 1280.0).abs() < 1e-10);
        assert!((positions[0].1 - 1280.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_centered_grid_asymmetric_display() {
        let positions = generate_centered_grid(2, 50.0, 1920, 1080);

        assert_eq!(positions.len(), 4);

        // Center of 2x2 grid should still be at display center
        let avg_x: f64 = positions.iter().map(|(x, _)| x).sum::<f64>() / 4.0;
        let avg_y: f64 = positions.iter().map(|(_, y)| y).sum::<f64>() / 4.0;
        assert!((avg_x - 960.0).abs() < 1e-10);
        assert!((avg_y - 540.0).abs() < 1e-10);
    }
}

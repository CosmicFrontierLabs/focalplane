//! Pattern commands for remote control of calibration displays.
//!
//! Used by desktop calibration controller to command what pattern
//! the OLED display should show via ZMQ.

use serde::{Deserialize, Serialize};

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

/// Commands sent from desktop to control OLED pattern display.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PatternCommand {
    /// Display a single Gaussian spot at the specified position.
    Spot {
        /// X position in display pixels (0 = left edge)
        x: f64,
        /// Y position in display pixels (0 = top edge)
        y: f64,
        /// Full-width at half-maximum in pixels
        fwhm: f64,
        /// Peak intensity (0.0 to 1.0, where 1.0 = white)
        intensity: f64,
    },

    /// Display multiple spots simultaneously.
    SpotGrid {
        /// List of (x, y) positions in display pixels
        positions: Vec<(f64, f64)>,
        /// Full-width at half-maximum in pixels (same for all spots)
        fwhm: f64,
        /// Peak intensity (0.0 to 1.0)
        intensity: f64,
    },

    /// Display uniform gray level across entire screen.
    Uniform {
        /// Gray level (0 = black, 255 = white)
        level: u8,
    },

    /// Clear display to black.
    Clear,
}

impl Default for PatternCommand {
    fn default() -> Self {
        Self::Clear
    }
}

impl PatternCommand {
    /// Create a spot command at the center of the display.
    pub fn centered_spot(width: u32, height: u32, fwhm: f64, intensity: f64) -> Self {
        Self::Spot {
            x: width as f64 / 2.0,
            y: height as f64 / 2.0,
            fwhm,
            intensity,
        }
    }

    /// Create a grid of spots centered on the display.
    pub fn centered_grid(
        width: u32,
        height: u32,
        grid_size: usize,
        spacing: f64,
        fwhm: f64,
        intensity: f64,
    ) -> Self {
        let positions = generate_centered_grid(grid_size, spacing, width, height);
        Self::SpotGrid {
            positions,
            fwhm,
            intensity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pattern_command_serialization() {
        let cmd = PatternCommand::Spot {
            x: 100.5,
            y: 200.5,
            fwhm: 5.0,
            intensity: 0.8,
        };

        let json = serde_json::to_string(&cmd).unwrap();
        let parsed: PatternCommand = serde_json::from_str(&json).unwrap();

        match parsed {
            PatternCommand::Spot {
                x,
                y,
                fwhm,
                intensity,
            } => {
                assert_relative_eq!(x, 100.5, epsilon = 1e-10);
                assert_relative_eq!(y, 200.5, epsilon = 1e-10);
                assert_relative_eq!(fwhm, 5.0, epsilon = 1e-10);
                assert_relative_eq!(intensity, 0.8, epsilon = 1e-10);
            }
            _ => panic!("Expected Spot variant"),
        }
    }

    #[test]
    fn test_spot_grid_serialization() {
        let cmd = PatternCommand::SpotGrid {
            positions: vec![(10.0, 20.0), (30.0, 40.0)],
            fwhm: 3.0,
            intensity: 1.0,
        };

        let json = serde_json::to_string(&cmd).unwrap();
        let parsed: PatternCommand = serde_json::from_str(&json).unwrap();

        match parsed {
            PatternCommand::SpotGrid {
                positions,
                fwhm,
                intensity,
            } => {
                assert_eq!(positions.len(), 2);
                assert_relative_eq!(positions[0].0, 10.0, epsilon = 1e-10);
                assert_relative_eq!(fwhm, 3.0, epsilon = 1e-10);
                assert_relative_eq!(intensity, 1.0, epsilon = 1e-10);
            }
            _ => panic!("Expected SpotGrid variant"),
        }
    }

    #[test]
    fn test_uniform_serialization() {
        let cmd = PatternCommand::Uniform { level: 128 };
        let json = serde_json::to_string(&cmd).unwrap();
        let parsed: PatternCommand = serde_json::from_str(&json).unwrap();

        match parsed {
            PatternCommand::Uniform { level } => {
                assert_eq!(level, 128);
            }
            _ => panic!("Expected Uniform variant"),
        }
    }

    #[test]
    fn test_clear_serialization() {
        let cmd = PatternCommand::Clear;
        let json = serde_json::to_string(&cmd).unwrap();
        let parsed: PatternCommand = serde_json::from_str(&json).unwrap();

        assert!(matches!(parsed, PatternCommand::Clear));
    }

    #[test]
    fn test_centered_spot() {
        let cmd = PatternCommand::centered_spot(1000, 800, 5.0, 0.9);
        match cmd {
            PatternCommand::Spot {
                x,
                y,
                fwhm,
                intensity,
            } => {
                assert_relative_eq!(x, 500.0, epsilon = 1e-10);
                assert_relative_eq!(y, 400.0, epsilon = 1e-10);
                assert_relative_eq!(fwhm, 5.0, epsilon = 1e-10);
                assert_relative_eq!(intensity, 0.9, epsilon = 1e-10);
            }
            _ => panic!("Expected Spot variant"),
        }
    }

    #[test]
    fn test_centered_grid() {
        let cmd = PatternCommand::centered_grid(1000, 1000, 3, 100.0, 5.0, 1.0);
        match cmd {
            PatternCommand::SpotGrid {
                positions,
                fwhm,
                intensity,
            } => {
                assert_eq!(positions.len(), 9);
                // Center should be at (500, 500)
                // Grid should span from (400, 400) to (600, 600)
                assert_relative_eq!(positions[0].0, 400.0, epsilon = 1e-10); // top-left x
                assert_relative_eq!(positions[0].1, 400.0, epsilon = 1e-10); // top-left y
                assert_relative_eq!(positions[4].0, 500.0, epsilon = 1e-10); // center x
                assert_relative_eq!(positions[4].1, 500.0, epsilon = 1e-10); // center y
                assert_relative_eq!(positions[8].0, 600.0, epsilon = 1e-10); // bottom-right x
                assert_relative_eq!(positions[8].1, 600.0, epsilon = 1e-10); // bottom-right y
                assert_relative_eq!(fwhm, 5.0, epsilon = 1e-10);
                assert_relative_eq!(intensity, 1.0, epsilon = 1e-10);
            }
            _ => panic!("Expected SpotGrid variant"),
        }
    }
}

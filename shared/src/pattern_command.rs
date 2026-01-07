//! Pattern commands for remote control of calibration displays.
//!
//! Used by desktop calibration controller to command what pattern
//! the OLED display should show via ZMQ.

use serde::{Deserialize, Serialize};

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
        let center_x = width as f64 / 2.0;
        let center_y = height as f64 / 2.0;
        let half_extent = (grid_size - 1) as f64 / 2.0;

        let mut positions = Vec::with_capacity(grid_size * grid_size);
        for row in 0..grid_size {
            for col in 0..grid_size {
                let offset_x = (col as f64 - half_extent) * spacing;
                let offset_y = (row as f64 - half_extent) * spacing;
                positions.push((center_x + offset_x, center_y + offset_y));
            }
        }

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
                assert!((x - 100.5).abs() < 1e-10);
                assert!((y - 200.5).abs() < 1e-10);
                assert!((fwhm - 5.0).abs() < 1e-10);
                assert!((intensity - 0.8).abs() < 1e-10);
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
                assert!((positions[0].0 - 10.0).abs() < 1e-10);
                assert!((fwhm - 3.0).abs() < 1e-10);
                assert!((intensity - 1.0).abs() < 1e-10);
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
                assert!((x - 500.0).abs() < 1e-10);
                assert!((y - 400.0).abs() < 1e-10);
                assert!((fwhm - 5.0).abs() < 1e-10);
                assert!((intensity - 0.9).abs() < 1e-10);
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
                assert!((positions[0].0 - 400.0).abs() < 1e-10); // top-left x
                assert!((positions[0].1 - 400.0).abs() < 1e-10); // top-left y
                assert!((positions[4].0 - 500.0).abs() < 1e-10); // center x
                assert!((positions[4].1 - 500.0).abs() < 1e-10); // center y
                assert!((positions[8].0 - 600.0).abs() < 1e-10); // bottom-right x
                assert!((positions[8].1 - 600.0).abs() < 1e-10); // bottom-right y
                assert!((fwhm - 5.0).abs() < 1e-10);
                assert!((intensity - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected SpotGrid variant"),
        }
    }
}

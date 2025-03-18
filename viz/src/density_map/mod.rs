//! Density map visualization
//!
//! This module provides ASCII-art visualizations for 2D density maps,
//! useful for displaying star distributions in RA/Dec coordinates
//! or other spatial visualizations.

use crate::Result;

/// A trait for objects that can be positioned on a 2D grid
pub trait PositionData {
    /// Get x coordinate (0.0 to 1.0 normalized)
    fn x(&self) -> f64;

    /// Get y coordinate (0.0 to 1.0 normalized)
    fn y(&self) -> f64;
}

/// Point implementation for simple x, y coordinates
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    /// Create a new point
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl PositionData for Point {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }
}

/// Configuration for density map visualization
#[derive(Debug, Clone)]
pub struct DensityMapConfig<'a> {
    /// Title for the density map
    pub title: Option<&'a str>,
    /// Label for the x-axis
    pub x_label: Option<&'a str>,
    /// Label for the top of the y-axis
    pub y_top_label: Option<&'a str>,
    /// Label for the bottom of the y-axis
    pub y_bottom_label: Option<&'a str>,
    /// Characters to use for density levels (from lowest to highest density)
    pub density_chars: &'a str,
    /// Width of the density map in characters
    pub width: usize,
    /// Height of the density map in characters
    pub height: usize,
}

impl Default for DensityMapConfig<'_> {
    fn default() -> Self {
        Self {
            title: None,
            x_label: None,
            y_top_label: None,
            y_bottom_label: None,
            density_chars: " .:#",
            width: 80,
            height: 24,
        }
    }
}

/// Create a density map visualization (ASCII art) for a set of positions
///
/// # Arguments
/// * `points` - Slice of items implementing PositionData
/// * `config` - Configuration for the density map visualization
///
/// # Returns
/// * String containing the ASCII art density map
pub fn create_density_map<T: PositionData>(
    points: &[T],
    config: &DensityMapConfig,
) -> Result<String> {
    // Create a 2D grid to count points
    let mut grid = vec![vec![0u32; config.width]; config.height];
    let mut max_count = 0;

    // Count points in each cell
    for point in points {
        // Map x (0-1) to grid coordinates
        let x = (point.x() * config.width as f64) as usize;
        let x = x.min(config.width - 1); // Clamp to valid range

        // Map y (0-1) to grid coordinates (invert y to match typical coordinate systems)
        let y = ((1.0 - point.y()) * config.height as f64) as usize;
        let y = y.min(config.height - 1); // Clamp to valid range

        // Count points in each grid cell
        grid[y][x] += 1;
        max_count = max_count.max(grid[y][x]);
    }

    // Now render the density map
    let mut output = String::new();

    // Add title if provided
    if let Some(title) = config.title {
        output.push_str(&format!("{}\n", title));
        output.push_str(&format!("{}\n", "=".repeat(title.len())));
    }

    // Add top y-axis label if provided
    if let Some(y_top) = config.y_top_label {
        output.push_str(&format!(
            "{}{}\n",
            y_top,
            " ".repeat(config.width.saturating_sub(y_top.len()))
        ));
    }

    // Choose characters for density representation based on max count
    let char_count = config.density_chars.chars().count();
    if char_count == 0 {
        return Err(crate::VizError::HistogramError(
            "Empty character set for density map".to_string(),
        ));
    }

    // Draw the grid with borders
    output.push_str(&format!("  {}\n", "-".repeat(config.width + 2)));
    for row in &grid {
        output.push_str("  |");
        for &count in row {
            // Map count to character index
            let char_idx = if max_count > 0 {
                ((count as f64 / max_count as f64) * (char_count - 1) as f64).round() as usize
            } else {
                0
            };
            let c = config.density_chars.chars().nth(char_idx).unwrap_or(' ');
            output.push(c);
        }
        output.push_str("|\n");
    }
    output.push_str(&format!("  {}\n", "-".repeat(config.width + 2)));

    // Add bottom y-axis label if provided
    if let Some(y_bottom) = config.y_bottom_label {
        output.push_str(&format!(
            "{}{}\n",
            y_bottom,
            " ".repeat(config.width.saturating_sub(y_bottom.len()))
        ));
    }

    // Add x-axis label if provided
    if let Some(x_label) = config.x_label {
        output.push_str(&format!("  {}\n", x_label));
    }

    // Add legend
    output.push_str(&format!(
        "  Legend: '{}' = no points, '{}' = highest density ({} points)\n",
        config.density_chars.chars().next().unwrap_or(' '),
        config.density_chars.chars().last().unwrap_or('#'),
        max_count
    ));

    Ok(output)
}

/// Create a celestial density map specifically for RA/Dec star coordinates
///
/// # Arguments
/// * `ra_dec_points` - Slice of (RA, Dec) tuples in degrees
/// * `width` - Width of the density map in characters
/// * `height` - Height of the density map in characters
/// * `chars` - String of characters to use for density levels (from lowest to highest density)
///
/// # Returns
/// * String containing the ASCII art celestial density map
pub fn create_celestial_density_map(
    ra_dec_points: &[(f64, f64)],
    width: usize,
    height: usize,
    chars: &str,
) -> Result<String> {
    // Convert RA/Dec points to normalized Points
    let normalized_points: Vec<Point> = ra_dec_points
        .iter()
        .map(|&(ra, dec)| {
            // Map RA (0-360) to x (0-1)
            let x = (ra % 360.0) / 360.0;

            // Map Dec (-90 to +90) to y (0-1) with north pole at top
            let y = (dec + 90.0) / 180.0;

            Point::new(x, y)
        })
        .collect();

    // Create a celestial-specific configuration
    let config = DensityMapConfig {
        title: Some("Star Density Map (RA vs Dec)"),
        x_label: Some("RA increases left to right (0° to 360°)"),
        y_top_label: Some("North Pole"),
        y_bottom_label: Some("South Pole"),
        density_chars: chars,
        width,
        height,
    };

    create_density_map(&normalized_points, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_density_map() {
        let points: Vec<Point> = vec![];
        let config = DensityMapConfig {
            width: 10,
            height: 5,
            density_chars: " .:#",
            ..Default::default()
        };

        let result = create_density_map(&points, &config);
        assert!(result.is_ok());

        let map = result.unwrap();
        assert!(map.contains("|          |")); // Empty lines
        assert!(map.contains("Legend: ' ' = no points, '#' = highest density"));
    }

    #[test]
    fn test_single_point() {
        let points = vec![Point::new(0.5, 0.5)];
        let config = DensityMapConfig {
            width: 10,
            height: 5,
            density_chars: " .:#",
            ..Default::default()
        };

        let result = create_density_map(&points, &config);
        assert!(result.is_ok());

        let map = result.unwrap();
        assert!(map.contains("#")); // Should have a '#' character for the max density
    }

    #[test]
    fn test_multiple_points() {
        let points = vec![
            Point::new(0.2, 0.2),
            Point::new(0.2, 0.2), // Duplicate to test density
            Point::new(0.8, 0.8),
        ];

        let config = DensityMapConfig {
            width: 10,
            height: 5,
            density_chars: " .:#",
            ..Default::default()
        };

        let result = create_density_map(&points, &config);
        assert!(result.is_ok());

        let map = result.unwrap();
        assert!(map.contains("#")); // Should have a '#' character for the max density
    }

    #[test]
    fn test_celestial_mapping() {
        let stars = vec![
            (0.0, 90.0),    // North pole
            (180.0, -90.0), // South pole
            (90.0, 0.0),    // Equator
        ];

        let result = create_celestial_density_map(&stars, 10, 5, " .:#");
        assert!(result.is_ok());

        let map = result.unwrap();
        assert!(map.contains("North Pole"));
        assert!(map.contains("South Pole"));
    }
}

//! ASCII density map visualization for astronomical data analysis and debugging.
//!
//! This module provides sophisticated 2D density visualization tools specifically
//! designed for astronomical applications, including star field analysis, detector
//! response mapping, and spatial distribution studies. Uses high-quality ASCII
//! rendering with configurable character sets and coordinate system mapping.
//!
//! # Visualization Philosophy
//!
//! ## Data Exploration and Analysis
//! Density maps serve multiple critical functions in astronomical data analysis:
//! - **Distribution analysis**: Visualize spatial clustering and patterns
//! - **Quality assessment**: Identify data artifacts and systematic errors
//! - **Algorithm debugging**: Verify coordinate transformations and projections
//! - **Quick inspection**: Rapid visual feedback during development
//!
//! ## ASCII Art Advantages
//! - **Terminal compatibility**: Works in any text-based environment
//! - **Logging integration**: Can be embedded in log files and reports
//! - **Lightweight**: Minimal dependencies and fast rendering
//! - **Version control**: Text-based output diffs cleanly in git
//!
//! # Coordinate System Support
//!
//! ## Normalized Coordinates
//! The core visualization engine uses normalized \[0,1\] coordinate space:
//! - **X-axis**: 0.0 (left) to 1.0 (right)
//! - **Y-axis**: 0.0 (bottom) to 1.0 (top)
//! - **Inversion handling**: Automatic Y-axis flipping for screen coordinates
//! - **Clamping**: Out-of-bounds coordinates safely mapped to edges
//!
//! ## Astronomical Coordinate Mapping
//! Specialized support for celestial coordinate systems:
//! - **Right Ascension**: 0°-360° → left to right mapping
//! - **Declination**: -90° to +90° → south pole to north pole
//! - **Equatorial grid**: Standard astronomical coordinate conventions
//! - **Projection handling**: Deals with coordinate singularities gracefully
//!
//! # Density Mapping Algorithm
//!
//! ## Spatial Binning
//! Points are accumulated into a 2D histogram grid:
//! 1. **Grid allocation**: Create width × height integer count array
//! 2. **Point mapping**: Transform coordinates to grid indices
//! 3. **Accumulation**: Increment count for each grid cell containing points
//! 4. **Statistics**: Track maximum count for normalization
//!
//! ## Character Mapping
//! Count values are mapped to ASCII characters for visualization:
//! - **Linear scaling**: Counts normalized to character range [0, n-1]
//! - **Round-to-nearest**: Smooth gradation between density levels
//! - **Configurable sets**: User-defined character progressions
//! - **Boundary handling**: Graceful handling of zero and maximum densities
//!
//! # Usage Examples
//!
//! ## Basic Point Distribution
//! ```rust
//! use viz::density_map::{Point, DensityMapConfig, create_density_map};
//!
//! // Create sample data points
//! let points = vec![
//!     Point::new(0.1, 0.2),  // Lower left
//!     Point::new(0.9, 0.8),  // Upper right
//!     Point::new(0.5, 0.5),  // Center
//!     Point::new(0.5, 0.5),  // Center (duplicate for higher density)
//! ];
//!
//! // Configure visualization
//! let config = DensityMapConfig {
//!     title: Some("Point Distribution"),
//!     x_label: Some("X Coordinate"),
//!     y_top_label: Some("Y=1.0"),
//!     y_bottom_label: Some("Y=0.0"),
//!     density_chars: " .o*#",
//!     width: 40,
//!     height: 20,
//!     ..Default::default()
//! };
//!
//! // Generate ASCII density map
//! let map = create_density_map(&points, &config).unwrap();
//! println!("{}", map);
//! ```
//!
//! ## Star Field Visualization
//! ```rust
//! use viz::density_map::create_celestial_density_map;
//!
//! // Star catalog data (RA, Dec in degrees)
//! let stars = vec![
//!     (0.0, 90.0),      // North pole star
//!     (90.0, 0.0),      // Equatorial star
//!     (180.0, -30.0),   // Southern sky star
//!     (270.0, 45.0),    // Northern sky star
//!     (45.0, -60.0),    // Southern constellation
//! ];
//!
//! // Create celestial density map
//! let sky_map = create_celestial_density_map(
//!     &stars,
//!     80,    // 80 character width
//!     24,    // 24 line height
//!     " .'*#" // Character progression
//! ).unwrap();
//!
//! println!("{}", sky_map);
//! ```
//!
//! ## Custom Position Data
//! ```rust
//! use viz::density_map::{PositionData, DensityMapConfig, create_density_map};
//!
//! // Custom data structure
//! struct Star {
//!     ra: f64,   // Right ascension (degrees)
//!     dec: f64,  // Declination (degrees)
//!     mag: f64,  // Magnitude
//! }
//!
//! impl PositionData for Star {
//!     fn x(&self) -> f64 {
//!         self.ra / 360.0  // Normalize RA to [0,1]
//!     }
//!     
//!     fn y(&self) -> f64 {
//!         (self.dec + 90.0) / 180.0  // Normalize Dec to [0,1]
//!     }
//! }
//!
//! let catalog = vec![
//!     Star { ra: 45.0, dec: 30.0, mag: 5.5 },
//!     Star { ra: 120.0, dec: -15.0, mag: 7.2 },
//!     Star { ra: 300.0, dec: 60.0, mag: 4.1 },
//! ];
//!
//! let config = DensityMapConfig::default();
//! let map = create_density_map(&catalog, &config).unwrap();
//! ```
//!
//! ## Algorithm Debugging
//! ```rust
//! use viz::density_map::{Point, DensityMapConfig, create_density_map};
//!
//! // Debug coordinate transformation results
//! fn debug_projection_results(projected_stars: &[(f64, f64)]) {
//!     let points: Vec<Point> = projected_stars.iter()
//!         .map(|&(x, y)| Point::new(x, y))
//!         .collect();
//!     
//!     let config = DensityMapConfig {
//!         title: Some("Projected Star Positions"),
//!         x_label: Some("Pixel X (normalized)"),
//!         y_top_label: Some("Pixel Y=0"),
//!         y_bottom_label: Some("Pixel Y=max"),
//!         density_chars: " .o#",
//!         width: 60,
//!         height: 30,
//!         ..Default::default()
//!     };
//!     
//!     let debug_map = create_density_map(&points, &config).unwrap();
//!     println!("\nProjection Debug Map:\n{}", debug_map);
//! }
//! ```
//!
//! # Character Set Design
//!
//! ## Density Progression
//! Character sets should provide clear visual progression from low to high density:
//! - **Basic**: `" .:#"` - Simple 4-level progression
//! - **Extended**: `" .'*oO#"` - 7-level with smooth gradation
//! - **High resolution**: `" .':^\"*oO#@"` - 11-level for detailed maps
//! - **Custom**: Application-specific symbols for domain visualization
//!
//! ## Visual Design Principles
//! - **Perceptual uniformity**: Each character step represents similar density increase
//! - **Terminal compatibility**: Characters display consistently across terminals
//! - **Readability**: Clear distinction between adjacent density levels
//! - **Aesthetic**: Pleasing visual progression from sparse to dense
//!
//! # Performance Characteristics
//!
//! ## Computational Complexity
//! - **Time**: O(n + w×h) where n=points, w=width, h=height
//! - **Space**: O(w×h) for the density grid
//! - **Rendering**: O(w×h) for ASCII generation
//! - **Memory**: Minimal overhead beyond grid storage
//!
//! ## Scalability Considerations
//! - **Large datasets**: Efficient for millions of points
//! - **High resolution**: Grid size limited by memory and terminal display
//! - **Real-time**: Fast enough for interactive debugging
//! - **Batch processing**: Suitable for automated analysis pipelines
//!
//! # Integration with Analysis Pipeline
//!
//! ## Development Workflow
//! Density maps integrate seamlessly into astronomical analysis:
//! - **Data validation**: Quick visual check of input coordinates
//! - **Algorithm verification**: Verify coordinate transformations
//! - **Parameter tuning**: Visualize effects of different settings
//! - **Result presentation**: Include in reports and documentation
//!
//! ## Automated Testing
//! ASCII maps enable automated visual regression testing:
//! - **Reference images**: Store expected map outputs as text
//! - **Diff comparison**: Git can diff ASCII maps directly
//! - **CI integration**: Include map generation in test suites
//! - **Documentation**: Embed maps in code comments and README files

use crate::Result;

/// Universal trait for 2D spatial positioning in normalized coordinate space.
///
/// Provides a standardized interface for extracting spatial coordinates from
/// arbitrary data structures. Enables density visualization of diverse datasets
/// including astronomical catalogs, detector measurements, and simulation results.
///
/// # Coordinate Convention
/// - **X-axis**: 0.0 (left) to 1.0 (right) in visualization space
/// - **Y-axis**: 0.0 (bottom) to 1.0 (top) in mathematical convention
/// - **Normalization**: Implementation must map data coordinates to \[0,1\] range
/// - **Clamping**: Values outside \[0,1\] will be clamped to grid boundaries
///
/// # Implementation Examples
/// ```rust
/// use viz::density_map::PositionData;
///
/// // Simple 2D point
/// struct Point2D { x: f64, y: f64 }
/// impl PositionData for Point2D {
///     fn x(&self) -> f64 { self.x }
///     fn y(&self) -> f64 { self.y }
/// }
///
/// // Astronomical star with coordinate transformation
/// struct Star { ra_deg: f64, dec_deg: f64 }
/// impl PositionData for Star {
///     fn x(&self) -> f64 { self.ra_deg / 360.0 }
///     fn y(&self) -> f64 { (self.dec_deg + 90.0) / 180.0 }
/// }
///
/// // Detector pixel with normalization
/// struct Pixel { col: usize, row: usize, width: usize, height: usize }
/// impl PositionData for Pixel {
///     fn x(&self) -> f64 { self.col as f64 / self.width as f64 }
///     fn y(&self) -> f64 { self.row as f64 / self.height as f64 }
/// }
/// ```
pub trait PositionData {
    /// Extract normalized X-coordinate for density mapping.
    ///
    /// # Returns
    /// X-position in normalized coordinates [0.0, 1.0] where:
    /// - 0.0 corresponds to the left edge of the visualization
    /// - 1.0 corresponds to the right edge of the visualization
    /// - Values outside \[0,1\] will be clamped to grid boundaries
    fn x(&self) -> f64;

    /// Extract normalized Y-coordinate for density mapping.
    ///
    /// # Returns
    /// Y-position in normalized coordinates [0.0, 1.0] where:
    /// - 0.0 corresponds to the bottom edge of the visualization
    /// - 1.0 corresponds to the top edge of the visualization
    /// - Values outside \[0,1\] will be clamped to grid boundaries
    fn y(&self) -> f64;
}

/// Simple 2D point implementation with normalized coordinates.
///
/// Provides a basic implementation of the PositionData trait for simple
/// (x, y) coordinate pairs. Useful for testing, debugging, and cases where
/// coordinate data is already in normalized form.
///
/// # Coordinate Range
/// While the struct accepts any f64 values, the density mapping expects
/// coordinates in the [0.0, 1.0] range. Values outside this range will
/// be clamped to the grid boundaries during visualization.
///
/// # Examples
/// ```rust
/// use viz::density_map::Point;
///
/// // Create points at various positions
/// let origin = Point::new(0.0, 0.0);       // Bottom-left corner
/// let center = Point::new(0.5, 0.5);       // Center of grid
/// let top_right = Point::new(1.0, 1.0);    // Top-right corner
///
/// // Points outside [0,1] range (will be clamped)
/// let outside = Point::new(-0.1, 1.2);     // Will map to (0.0, 1.0)
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    /// Create new point with specified coordinates.
    ///
    /// # Arguments
    /// * `x` - X-coordinate (typically in [0.0, 1.0] range)
    /// * `y` - Y-coordinate (typically in [0.0, 1.0] range)
    ///
    /// # Returns
    /// New Point instance with the specified coordinates
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

/// Comprehensive configuration for ASCII density map visualization.
///
/// Controls all aspects of density map generation including grid dimensions,
/// character sets, labeling, and coordinate system interpretation. Provides
/// flexible customization for different visualization needs while maintaining
/// sensible defaults for common use cases.
///
/// # Configuration Categories
/// - **Grid geometry**: Width and height in character cells
/// - **Character mapping**: Symbols used for different density levels
/// - **Labeling**: Title and axis labels for context
/// - **Coordinate system**: Interpretation of spatial coordinates
///
/// # Examples
/// ```rust
/// use viz::density_map::DensityMapConfig;
///
/// // Minimal configuration with defaults
/// let simple = DensityMapConfig::default();
///
/// // Fully customized configuration
/// let custom = DensityMapConfig {
///     title: Some("Star Distribution Analysis"),
///     x_label: Some("Right Ascension (hours)"),
///     y_top_label: Some("+90° Dec"),
///     y_bottom_label: Some("-90° Dec"),
///     density_chars: " .:*#@",
///     width: 120,
///     height: 40,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct DensityMapConfig<'a> {
    /// Optional title displayed above the density map.
    ///
    /// When provided, appears at the top of the visualization followed by
    /// an underline of equal signs. Useful for identifying the dataset
    /// or analysis context.
    pub title: Option<&'a str>,

    /// Optional label for the bottom X-axis.
    ///
    /// Describes the meaning of the horizontal coordinate, typically
    /// including units and value range. Appears below the density grid.
    pub x_label: Option<&'a str>,

    /// Optional label for the top edge of the Y-axis.
    ///
    /// Indicates the meaning of Y=1.0 in the coordinate system.
    /// For astronomical coordinates, often indicates "North Pole" or "+90°".
    pub y_top_label: Option<&'a str>,

    /// Optional label for the bottom edge of the Y-axis.
    ///
    /// Indicates the meaning of Y=0.0 in the coordinate system.
    /// For astronomical coordinates, often indicates "South Pole" or "-90°".
    pub y_bottom_label: Option<&'a str>,

    /// Character progression for density visualization.
    ///
    /// String of ASCII characters arranged from lowest density (first character)
    /// to highest density (last character). The choice of characters affects
    /// visual clarity and aesthetic appeal. Must contain at least one character.
    ///
    /// **Common progressions**:
    /// - Basic: `" .:#"` (4 levels)
    /// - Smooth: `" .'*oO#"` (7 levels)
    /// - High-res: `" .':*oO#@"` (9 levels)
    pub density_chars: &'a str,

    /// Width of the density grid in character columns.
    ///
    /// Determines the horizontal resolution of the density map. Higher values
    /// provide finer spatial resolution but require wider terminal displays.
    /// Typical range: 40-120 characters depending on application.
    pub width: usize,

    /// Height of the density grid in character rows.
    ///
    /// Determines the vertical resolution of the density map. Higher values
    /// provide finer spatial resolution but require taller terminal displays.
    /// Typical range: 20-50 rows depending on application.
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

/// Generate ASCII density map from spatial point data with full customization.
///
/// Creates a high-quality ASCII visualization of 2D point distributions using
/// configurable character sets and labeling. The algorithm bins points into
/// a regular grid, computes density statistics, and renders using progressive
/// character mapping for intuitive density representation.
///
/// # Algorithm Overview
/// 1. **Grid initialization**: Create width × height counting grid
/// 2. **Point binning**: Map each point to grid cell and increment counter
/// 3. **Statistics**: Find maximum density for normalization
/// 4. **Character mapping**: Map density counts to character progression
/// 5. **Rendering**: Generate formatted ASCII output with labels and legend
///
/// # Coordinate Transformation
/// - **X mapping**: point.x() → [0, width-1] grid column
/// - **Y mapping**: (1.0 - point.y()) → [0, height-1] grid row (inverted)
/// - **Clamping**: Out-of-bounds coordinates mapped to grid edges
/// - **Precision**: Sub-pixel coordinates rounded to nearest grid cell
///
/// # Character Density Mapping
/// Density values are linearly mapped to character indices:
/// ```text
/// char_index = round((count / max_count) × (num_chars - 1))
/// ```
/// This provides smooth gradation from sparse (first char) to dense (last char).
///
/// # Arguments
/// * `points` - Collection of objects implementing PositionData trait
/// * `config` - Comprehensive visualization configuration
///
/// # Returns
/// * `Ok(String)` - Complete ASCII density map with formatting
/// * `Err(VizError)` - Configuration or rendering error
///
/// # Examples
/// ```rust
/// use viz::density_map::{Point, DensityMapConfig, create_density_map};
///
/// // Create clustered point data
/// let points = vec![
///     Point::new(0.2, 0.3),  // Cluster 1
///     Point::new(0.25, 0.35),
///     Point::new(0.18, 0.28),
///     Point::new(0.8, 0.7),  // Cluster 2
///     Point::new(0.82, 0.72),
///     Point::new(0.78, 0.68),
/// ];
///
/// // Configure for detailed visualization
/// let config = DensityMapConfig {
///     title: Some("Point Clustering Analysis"),
///     x_label: Some("X Coordinate (normalized)"),
///     y_top_label: Some("Y=1.0"),
///     y_bottom_label: Some("Y=0.0"),
///     density_chars: " .o*#@",  // 6-level progression
///     width: 50,
///     height: 25,
/// };
///
/// let map = create_density_map(&points, &config)?;
/// println!("{}", map);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance Notes
/// - **Time complexity**: O(n + w*h) where n=points, w=width, h=height
/// - **Memory usage**: O(w*h) for density grid plus output string
/// - **Scalability**: Efficient for datasets up to millions of points
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
        output.push_str(&format!("{title}\n"));
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
        output.push_str(&format!("  {x_label}\n"));
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

/// Generate specialized celestial sphere density map for astronomical coordinates.
///
/// Creates an ASCII visualization specifically designed for Right Ascension and
/// Declination coordinates, with proper astronomical coordinate conventions,
/// appropriate labeling, and coordinate system mapping. Automatically handles
/// the spherical coordinate transformation to rectangular grid representation.
///
/// # Astronomical Coordinate Mapping
/// - **Right Ascension**: 0°-360° → left to right across full map width
/// - **Declination**: -90° (South Pole) → +90° (North Pole) bottom to top
/// - **Coordinate wrapping**: RA values automatically wrapped to [0,360) range
/// - **Polar handling**: Declination extremes map to grid edges
///
/// # Specialized Features
/// - **Astronomical labels**: Pre-configured with celestial coordinate labels
/// - **Pole indication**: Clear marking of North and South celestial poles
/// - **RA progression**: Indicates Right Ascension increases left to right
/// - **Standard title**: "Star Density Map (RA vs Dec)" for clarity
///
/// # Coordinate System Conventions
/// Follows standard astronomical conventions:
/// - **Equatorial coordinates**: J2000.0 coordinate system assumed
/// - **Angular units**: Input coordinates expected in decimal degrees
/// - **Hemisphere symmetry**: Equal treatment of northern/southern sky
/// - **Meridian alignment**: 0° RA at left edge, 360° RA wraps to left
///
/// # Arguments
/// * `ra_dec_points` - Star positions as (RA, Dec) tuples in degrees
/// * `width` - Horizontal resolution in characters (typically 60-120)
/// * `height` - Vertical resolution in characters (typically 20-40)
/// * `chars` - Character progression for density levels (e.g., " .o*#")
///
/// # Returns
/// * `Ok(String)` - Complete celestial density map with astronomical labels
/// * `Err(VizError)` - Configuration or rendering error
///
/// # Examples
/// ```rust
/// use viz::density_map::create_celestial_density_map;
///
/// // Sample star catalog data
/// let bright_stars = vec![
///     (0.0, 90.0),        // Polaris (North Star)
///     (95.99, 7.41),      // Betelgeuse (Orion)
///     (201.3, -11.2),     // Spica (Virgo)
///     (310.36, 45.28),    // Vega (Lyra)
///     (279.23, 38.78),    // Deneb (Cygnus)
/// ];
///
/// // Generate all-sky map
/// let sky_map = create_celestial_density_map(
///     &bright_stars,
///     80,         // 80-character width
///     24,         // 24-line height
///     " .'*#@"    // 6-level density characters
/// )?;
///
/// println!("{}", sky_map);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
/// - **Catalog validation**: Verify star catalog coordinate coverage
/// - **Survey planning**: Visualize observing field distributions
/// - **Sky coverage**: Assess completeness of astronomical surveys
/// - **Coordinate debugging**: Verify RA/Dec transformation accuracy
/// - **Constellation mapping**: Identify stellar groupings and patterns
///
/// # Limitations
/// - **Projection distortion**: Rectangular mapping distorts high-latitude regions
/// - **Coordinate singularities**: Poles appear as horizontal lines
/// - **Resolution limits**: Character grid limits fine-scale structure
/// - **Density quantization**: Limited number of density levels
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

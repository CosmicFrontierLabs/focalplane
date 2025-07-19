//! Comprehensive visualization toolkit for astronomical data analysis and quality assessment.
//!
//! This crate provides a complete suite of ASCII-based visualization tools specifically
//! designed for astronomical simulations, telescope observations, and photometric analysis.
//! Optimized for terminal-based workflows, automated analysis pipelines, and situations
//! where traditional graphical displays are unavailable or impractical.
//!
//! # Visualization Philosophy
//!
//! ## ASCII-First Design
//! All visualization tools prioritize text-based output for maximum compatibility:
//! - **Terminal native**: Works in any text-based environment
//! - **SSH compatible**: Full functionality over remote connections
//! - **Scriptable**: Easy integration into automated analysis workflows
//! - **Version control friendly**: Text-based outputs diff cleanly in git
//! - **Logging integration**: Visualizations can be embedded in log files
//!
//! ## Astronomical Focus
//! Visualization tools are specifically optimized for astronomical data characteristics:
//! - **Wide dynamic ranges**: Logarithmic scaling for stellar magnitude distributions
//! - **Statistical analysis**: Comprehensive distribution characterization
//! - **Quality assessment**: Tools for identifying systematic errors and outliers
//! - **Survey analysis**: Specialized support for completeness and bias studies
//!
//! # Core Modules
//!
//! ## Histogram Analysis (`histogram`)
//! Advanced statistical histograms with astronomical applications:
//! - **Magnitude histograms**: Unit-width bins aligned with astronomical conventions
//! - **Statistical analysis**: Mean, median, skewness, kurtosis with interpretation
//! - **Multiple scaling**: Linear and logarithmic bar scaling options
//! - **Quality metrics**: Sample size and confidence assessment
//!
//! ## Density Mapping (`density_map`)
//! 2D spatial distribution visualization for point data:
//! - **Star field visualization**: Plot stellar positions and clustering
//! - **Coordinate system support**: Celestial coordinates and detector pixels
//! - **ASCII density maps**: Character-based 2D histograms
//! - **Projection support**: Gnomonic and other astronomical projections
//!
//! # Usage Patterns
//!
//! ## Interactive Data Exploration
//! Quick analysis and quality assessment during development:
//! ```rust
//! use viz::histogram::create_magnitude_histogram;
//!
//! // Quick magnitude distribution check
//! let magnitudes = vec![12.1, 13.5, 14.2, 15.8, 16.1, 17.3];
//! let hist = create_magnitude_histogram(&magnitudes,
//!     Some("V-band Magnitudes".to_string()), false)?;
//! hist.print()?;
//!
//! // Statistical summary
//! println!("{}", hist.statistics_summary());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Automated Quality Assessment
//! Batch processing and pipeline integration:
//! ```rust,ignore
//! // NOTE: This doctest is ignored due to undefined functions
//! use viz::histogram::histogram;
//! use viz::density_map::create_celestial_density_map;
//!
//! // Photometric error analysis
//! let errors = load_photometric_errors()?;
//! let error_analysis = histogram(&errors, 20, 0.0..0.5,
//!     Some("Error Distribution".to_string()), true)?;
//!
//! // Spatial distribution check
//! let star_positions = load_star_catalog()?;
//! let sky_map = create_celestial_density_map(&star_positions, 80, 24, " .*#")?;
//!
//! // Save to analysis log
//! write_analysis_report(&error_analysis, &sky_map)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # fn load_photometric_errors() -> Result<Vec<f64>, Box<dyn std::error::Error>> { Ok(vec![]) }
//! # fn load_star_catalog() -> Result<Vec<(f64, f64)>, Box<dyn std::error::Error>> { Ok(vec![]) }
//! # fn write_analysis_report(a: &str, b: &str) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
//! ```
//!
//! ## Survey Completeness Studies
//! Specialized analysis for observational surveys:
//! ```rust,ignore
//! // NOTE: This doctest is ignored due to undefined functions
//! use viz::histogram::create_magnitude_histogram;
//!
//! // Load survey detection data
//! let detected_mags = load_survey_detections()?;
//!
//! // Create magnitude histogram with log scaling to emphasize faint end
//! let completeness = create_magnitude_histogram(&detected_mags,
//!     Some("Survey Completeness vs Magnitude".to_string()), true)?;
//!
//! // Check for completeness limit indicators
//! if let Some(skew) = completeness.skewness() {
//!     if skew < -0.5 {
//!         println!("Warning: Negative skewness suggests incomplete faint-end detection");
//!     }
//! }
//!
//! // Generate comprehensive report
//! println!("{}", completeness.statistics_summary());
//! completeness.print()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # fn load_survey_detections() -> Result<Vec<f64>, Box<dyn std::error::Error>> { Ok(vec![]) }
//! ```
//!
//! # Integration with Analysis Pipeline
//!
//! ## CI/CD Integration
//! Automated quality checks in continuous integration:
//! - **Regression testing**: Compare distributions against reference data
//! - **Performance monitoring**: Track analysis timing and memory usage
//! - **Quality gates**: Automatic detection of data quality issues
//! - **Report generation**: Standardized analysis outputs for review
//!
//! ## Development Workflow
//! Interactive analysis during algorithm development:
//! - **Real-time feedback**: Immediate visualization of algorithm outputs
//! - **Parameter tuning**: Visual assessment of parameter effects
//! - **Debug assistance**: Spatial and statistical debugging tools
//! - **Documentation**: Embed visualizations in code documentation
//!
//! # Performance Characteristics
//!
//! ## Computational Efficiency
//! - **Linear scaling**: O(n) performance for most operations
//! - **Memory efficient**: Minimal overhead beyond data storage
//! - **Lazy evaluation**: Statistics computed only when requested
//! - **Streaming friendly**: Can process data in chunks
//!
//! ## Output Quality
//! - **High resolution**: Fine-grained binning for detailed analysis
//! - **Configurable precision**: Adjustable output detail levels
//! - **Robust statistics**: Numerical stability for edge cases
//! - **Clear formatting**: Human-readable output with proper scaling
//!
//! # Error Handling
//!
//! The crate uses a comprehensive error handling system with specific error types
//! for different failure modes. All functions return `Result` types with detailed
//! error messages for debugging and user feedback.
//!
//! ## Common Error Scenarios
//! - **Invalid data**: Empty datasets, NaN values, out-of-range inputs
//! - **Configuration errors**: Invalid bin counts, malformed parameters
//! - **Formatting failures**: I/O errors during output generation
//! - **Resource limits**: Memory exhaustion with very large datasets
//!
//! # Future Enhancements
//!
//! ## Planned Features
//! - **3D visualization**: Volumetric data representation using ASCII
//! - **Time series**: Specialized plots for temporal astronomical data
//! - **Correlation analysis**: Cross-correlation visualization tools
//! - **Export formats**: CSV, JSON output for external analysis tools
//!
//! ## Performance Improvements
//! - **Parallel processing**: Multi-threaded histogram generation
//! - **Streaming algorithms**: Constant-memory algorithms for large datasets
//! - **Caching**: Intelligent caching for repeated analysis operations
//! - **Vectorization**: SIMD optimizations for statistical calculations

use std::fmt;
use thiserror::Error;

/// Comprehensive error types for visualization operations.
///
/// Provides detailed error reporting for all visualization failures including
/// data validation errors, formatting issues, and configuration problems.
/// Each error type includes specific context to help with debugging and
/// user feedback.
#[derive(Debug, Error)]
pub enum VizError {
    /// Histogram creation or analysis error.
    ///
    /// Includes issues with bin configuration, data validation,
    /// and statistical computation failures.
    #[error("Histogram error: {0}")]
    HistogramError(String),

    /// Text formatting or I/O error.
    ///
    /// Includes failures during ASCII output generation,
    /// string formatting, and file I/O operations.
    #[error("Formatting error: {0}")]
    FmtError(#[from] fmt::Error),
}

/// Standard Result type for all visualization operations.
///
/// Provides consistent error handling across the entire visualization
/// toolkit with detailed error context for debugging and user feedback.
pub type Result<T> = std::result::Result<T, VizError>;

pub mod density_map;
pub mod histogram;

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};
    use simulator::image_proc::overlay::overlay_to_image;

    #[test]
    fn test_text_rendering() {
        // Integration test for SVG text rendering capabilities
        // Tests various font families and text attributes to ensure
        // compatibility across different rendering environments
        let width = 800;
        let height = 600;
        let text_color = "#000000"; // Black text
        let bg_color = "#ffffff"; // White background
        let highlight_color = "#eeeeee"; // Light gray for text backgrounds

        // Create basic SVG with various text rendering approaches
        let svg_data = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
                <!-- Background rectangle -->
                <rect x="0" y="0" width="{width}" height="{height}" fill="{bg_color}" />
                
                <!-- Standard text elements with different fonts -->
                <rect x="20" y="20" width="760" height="30" fill="{highlight_color}" />
                <text x="30" y="40" font-family="sans-serif" font-size="20" fill="{text_color}">Text Test 1: sans-serif font</text>
                
                <rect x="20" y="60" width="760" height="30" fill="{highlight_color}" />
                <text x="30" y="80" font-family="serif" font-size="20" fill="{text_color}">Text Test 2: serif font</text>
                
                <rect x="20" y="100" width="760" height="30" fill="{highlight_color}" />
                <text x="30" y="120" font-family="monospace" font-size="20" fill="{text_color}">Text Test 3: monospace font</text>
                
                <!-- Text with explicit fallback fonts -->
                <rect x="20" y="140" width="760" height="30" fill="{highlight_color}" />
                <text x="30" y="160" font-family="Arial, Helvetica, sans-serif" font-size="20" fill="{text_color}">
                    Text Test 4: Multiple font fallbacks
                </text>
                
                <!-- Text with path-based approach (should always work) -->
                <rect x="20" y="180" width="760" height="30" fill="{highlight_color}" />
                <path d="M30 200 L50 200 L50 220 L30 220 Z" fill="{text_color}" />
                <path d="M60 200 L80 200 L80 220 L60 220 Z" fill="{text_color}" />
                <text x="100" y="215" font-family="sans-serif" font-size="20" fill="{text_color}">
                    Text Test 5: With path objects nearby
                </text>
                
                <!-- Styled text with different attributes -->
                <rect x="20" y="240" width="760" height="30" fill="{highlight_color}" />
                <text x="30" y="260" font-family="sans-serif" font-size="20" font-weight="bold" fill="{text_color}">
                    Text Test 6: Bold text
                </text>
                
                <rect x="20" y="280" width="760" height="30" fill="{highlight_color}" />
                <text x="30" y="300" font-family="sans-serif" font-size="20" font-style="italic" fill="{text_color}">
                    Text Test 7: Italic text
                </text>
                
                <!-- Text with explicit attributes -->
                <rect x="20" y="320" width="760" height="30" fill="{highlight_color}" />
                <text x="30" y="340" font-family="sans-serif" font-size="20" font-weight="bold" 
                      text-rendering="geometricPrecision" fill="{text_color}">
                    Text Test 8: With text-rendering attribute
                </text>
                
                <!-- SVG doesn't have direct ttf embedding, but we can try system fonts -->
                <rect x="20" y="360" width="760" height="30" fill="{highlight_color}" />
                <text x="30" y="380" font-family="Liberation Sans, Ubuntu, DejaVu Sans" font-size="20" fill="{text_color}">
                    Text Test 9: Common Linux system fonts
                </text>
                
                <!-- Simple shapes for verification -->
                <rect x="30" y="420" width="200" height="50" stroke="black" stroke-width="2" fill="blue" />
                <circle cx="400" cy="445" r="25" fill="red" />
                <line x1="500" y1="420" x2="700" y2="470" stroke="green" stroke-width="5" />
            </svg>"##
        );

        // Create a blank white image
        let mut image = RgbImage::new(width, height);
        for pixel in image.pixels_mut() {
            *pixel = Rgb([255, 255, 255]);
        }
        let base_image = DynamicImage::ImageRgb8(image);

        // Create output directory
        let output_dir = test_helpers::get_output_dir();

        // Process the SVG
        let result_image = overlay_to_image(&base_image, &svg_data);

        // Save the result
        let output_path = output_dir.join("text_rendering_test.png");
        result_image
            .save(&output_path)
            .expect("Failed to save text test image");

        // Verify the image size is correct
        assert_eq!(result_image.width(), { width });
        assert_eq!(result_image.height(), { height });

        // This test mainly checks if the overlay function works without errors
        // Visual verification needs to be done manually
    }
}

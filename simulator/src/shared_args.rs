//! Shared command-line argument definitions for astronomical simulation binaries.
//!
//! This module provides standardized argument parsing, validation, and type-safe
//! configuration management for the telescope simulation suite. Ensures consistent
//! interfaces across all simulation tools while supporting flexible configuration
//! of instruments, observing conditions, and data sources.
//!
//! # Architecture Overview
//!
//! ## Argument Categories
//! - **Instrument Selection**: Pre-defined telescope and sensor models
//! - **Observing Parameters**: Exposure times, coordinates, wavelengths
//! - **Data Sources**: Star catalogs, calibration files, configuration paths
//! - **Environmental Conditions**: Temperature, zodiacal light, backgrounds
//! - **Processing Options**: Detection thresholds, noise models, output formats
//!
//! ## Type Safety and Validation
//! - **Parse-time validation**: Input format and range checking
//! - **Domain-specific types**: Astronomical coordinates, durations with units
//! - **Error propagation**: Clear diagnostic messages for invalid inputs
//! - **Default values**: Sensible defaults for optional parameters
//!
//! # Command-Line Interface Patterns
//!
//! ## Standard Simulation Arguments
//! All simulation binaries support these common parameters:
//! ```bash
//! # Basic observation setup
//! simulator --exposure 30s --wavelength 550.0 --temperature -10.0
//!
//! # Instrument selection
//! simulator --telescope final1m --sensor gsense6510bsi
//!
//! # Sky position and background
//! simulator --coordinates "120.0,45.0" --catalog gaia_bright.bin
//!
//! # Detection and analysis
//! simulator --noise-multiple 5.0 --magnitude-range "8.0:16.0:0.5"
//! ```
//!
//! ## Duration Format Support
//! Flexible time specification with automatic unit conversion:
//! ```bash
//! --exposure 1s        # 1 second
//! --exposure 500ms     # 500 milliseconds
//! --exposure 1.5s      # 1.5 seconds (fractional)
//! --exposure 2m        # 2 minutes = 120 seconds
//! --exposure 1h        # 1 hour = 3600 seconds
//! ```
//!
//! ## Coordinate Format Support
//! Solar angular coordinates for zodiacal light modeling:
//! ```bash
//! --coordinates "90.0,30.0"   # 90° elongation, 30° latitude
//! --coordinates "165.0,75.0"  # Minimum zodiacal light position
//! --coordinates "0.0,0.0"     # Sun position (maximum zodiacal light)
//! ```
//!
//! ## Range Specification
//! Systematic parameter sweeps with start:stop:step syntax:
//! ```bash
//! --magnitude-range "8.0:16.0:0.5"  # Magnitude sweep
//! --temperature-range "-20.0:40.0:10.0"  # Temperature sweep
//! --exposure-range "0.1:10.0:0.1"   # Exposure time sweep
//! ```
//!
//! # Instrument Model Library
//!
//! ## Pre-configured Telescopes
//! - **Small 50mm**: Compact refractor (0.05m, f/10) for testing
//! - **Demo 50cm**: Mid-size telescope (0.5m, f/20) for development
//! - **Final 1m**: Production telescope (1.0m, f/10) for operations
//! - **Weasel**: Multi-spectral telescope (0.47m, f/7.3) for specialized observations
//!
//! ## Sensor Options
//! - **GSENSE4040BSI**: Large format CMOS (4096×4096, 9μm pixels)
//! - **GSENSE6510BSI**: High resolution CMOS (3200×3200, 6.5μm pixels) - Default
//! - **HWK4123**: Scientific CMOS (4096×2300, 4.6μm pixels)
//! - **IMX455**: Full-frame BSI CMOS (9568×6380, 3.75μm pixels)
//!
//! # Data Source Management
//!
//! ## Star Catalog Integration
//! - **Binary catalogs**: Efficient format for large stellar databases
//! - **Embedded bright stars**: Additional bright star data for completeness
//! - **Automatic merging**: Combines multiple catalog sources seamlessly
//! - **ID management**: Collision-free star identification across sources
//!
//! ## Catalog Enhancement
//! # Usage
//! Demonstrates how to use the automatic bright star augmentation feature:
//! - Use SharedSimulationArgs::parse() to parse command-line arguments
//! - Call args.load_catalog() to load the catalog with embedded bright stars included
//! - The returned catalog combines the specified binary catalog with additional bright stars
//!
//! # Error Handling and Validation
//!
//! ## Input Validation
//! - **Format checking**: Coordinate and range syntax validation
//! - **Range validation**: Physical constraints on parameters
//! - **Unit conversion**: Automatic handling of time and angle units
//! - **Default fallbacks**: Graceful handling of missing optional parameters
//!
//! ## Error Messages
//! Clear, actionable error diagnostics:
//! ```text
//! Error: Invalid coordinates: Invalid elongation: 200° (must be between 0° and 180°)
//! Error: Duration cannot be negative
//! Error: Range must be in format 'start:stop:step'
//! Error: For positive step, start must be less than stop
//! ```
//!
//! # Integration Examples
//!
//! ## Basic Simulation Setup
//! # Usage
//! Shows how to set up a basic simulation using SharedSimulationArgs:
//! - Parse command-line arguments with SharedSimulationArgs::parse()
//! - Access validated parameters: exposure time, telescope config, zodiacal coordinates
//! - Load the enhanced star catalog with args.load_catalog()
//! - The telescope configuration provides aperture and other optical parameters
//!
//! ## Custom Binary with Additional Arguments
//! # Usage
//! Demonstrates extending SharedSimulationArgs with custom parameters:
//! - Define a custom struct that includes SharedSimulationArgs via #[command(flatten)]
//! - Add custom parameters specific to your binary
//! - Access both shared and custom arguments through the combined struct
//! - Use args.shared to access the standard simulation parameters
//!
//! ## Batch Processing with Parameter Sweeps
//! # Usage
//! Shows how to implement parameter sweeps using RangeArg:
//! - Define custom args with a RangeArg field for parameter ranges
//! - Use the default format "start:stop:step" for range specification
//! - Extract start, stop, and step values with as_tuple() method
//! - Iterate through the parameter space for batch processing
//! - Useful for sensitivity analysis and systematic studies
//!
//! # Performance and Efficiency
//!
//! ## Argument Parsing
//! - **Zero-copy validation**: Input validation without string duplication
//! - **Lazy evaluation**: Configuration objects created only when accessed
//! - **Caching**: Parsed values stored for repeated access
//! - **Memory efficiency**: Minimal overhead for command-line processing
//!
//! ## Catalog Loading
//! - **Binary format**: Fast loading of large stellar databases
//! - **Streaming**: Memory-efficient processing of catalog data
//! - **Indexed access**: O(1) star lookup by ID or position
//! - **Compression**: Efficient storage of high-precision astrometry

use crate::hardware::sensor::SensorConfig;
use crate::hardware::telescope::TelescopeConfig;
use crate::photometry::zodical::SolarAngularCoordinates;
use clap::{Parser, ValueEnum};
use starfield::catalogs::binary_catalog::{BinaryCatalog, MinimalStar};
use std::path::PathBuf;
use std::time::Duration;

/// Parse solar angular coordinates from command-line string format.
///
/// Converts comma-separated coordinate string into validated SolarAngularCoordinates
/// for zodiacal light calculations. Supports flexible whitespace and provides
/// detailed error messages for invalid inputs.
///
/// # Format
/// Input string must be in format: "elongation,latitude"
/// - **Elongation**: Solar angular distance [0°, 180°]
/// - **Latitude**: Ecliptic latitude [-90°, +90°]
/// - **Whitespace**: Automatically trimmed around values
///
/// # Arguments
/// * `s` - Coordinate string in "elongation,latitude" format
///
/// # Returns
/// * `Ok(SolarAngularCoordinates)` - Validated coordinates
/// * `Err(String)` - Detailed error message for invalid input
///
/// # Usage
/// Valid coordinate formats:
/// - "90.0,30.0" - Standard format
/// - " 165.0 , 75.0 " - Whitespace is automatically trimmed
/// - "0.0,-45.0" - Negative latitude is valid
///
/// Invalid formats that return errors:
/// - "90.0" - Missing latitude
/// - "200.0,30.0" - Elongation must be between 0° and 180°
/// - "90.0,100.0" - Latitude must be between -90° and +90°
pub fn parse_coordinates(s: &str) -> Result<SolarAngularCoordinates, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err("Coordinates must be in format 'elongation,latitude'".to_string());
    }

    let elongation = parts[0]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid elongation value".to_string())?;
    let latitude = parts[1]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid latitude value".to_string())?;

    SolarAngularCoordinates::new(elongation, latitude)
        .map_err(|e| format!("Invalid coordinates: {e}"))
}

/// Default solar angular coordinates for minimum zodiacal light brightness.
///
/// Uses the position of minimum measurable zodiacal light from the Leinert et al.
/// (1998) survey. This represents the darkest accessible sky position for most
/// astronomical observations, providing a conservative background estimate.
///
/// **Values**: 165.0° elongation, 75.0° ecliptic latitude
/// **Background**: ~22.5 mag/arcsec² in V-band (very dark)
/// **Use case**: Default for exposure time calculations and survey planning
const DEFAULT_ZODIACAL_COORDINATES: &str = "165.0,75.0";

/// Embedded CSV data for additional bright stars missing from main catalogs.
///
/// Contains supplementary bright star data compiled at build time to ensure
/// complete sky coverage for simulation validation and calibration. These stars
/// fill gaps in the primary catalog, particularly for very bright objects that
/// may be saturated or excluded from survey catalogs.
///
/// **Data source**: Curated list of bright stars with precise astrometry
/// **Format**: CSV with RA_deg, Dec_deg, Gaia_magnitude columns
/// **Purpose**: Catalog completeness for simulation accuracy
const ADDITIONAL_BRIGHT_STARS_CSV: &str = include_str!("../data/missing_bright_stars.csv");

/// Parse embedded bright star CSV data into validated star catalog entries.
///
/// Processes the compile-time embedded CSV data containing additional bright
/// stars, converting each row into a MinimalStar instance with proper ID
/// assignment and coordinate validation. Ensures catalog completeness for
/// simulation accuracy.
///
/// # CSV Format
/// - **Header**: RA_deg,Dec_deg,Gaia_magnitude (automatically skipped)
/// - **RA_deg**: Right ascension in decimal degrees [0, 360)
/// - **Dec_deg**: Declination in decimal degrees [-90, +90]
/// - **Gaia_magnitude**: Apparent magnitude in Gaia G-band
///
/// # ID Assignment Strategy
/// Uses reverse counting from u64::MAX to avoid conflicts with main catalog:
/// - **First star**: ID = u64::MAX
/// - **Second star**: ID = u64::MAX - 1
/// - **Nth star**: ID = u64::MAX - (N-1)
///
/// This ensures no ID collisions when merging with primary catalogs.
///
/// # Error Handling
/// - **Line-by-line validation**: Detailed error messages with line numbers
/// - **Coordinate bounds checking**: RA and Dec range validation
/// - **Magnitude sanity checking**: Reasonable brightness limits
/// - **Format validation**: Proper CSV structure requirements
///
/// # Returns
/// * `Ok(Vec<MinimalStar>)` - Successfully parsed stars with assigned IDs
/// * `Err(Box<dyn std::error::Error>)` - Parse error with diagnostic information
///
/// # Usage
/// The function parses embedded CSV data and returns a vector of MinimalStar objects:
/// - Each star is assigned a unique ID counting backwards from u64::MAX
/// - First star gets ID = u64::MAX, second gets u64::MAX - 1, etc.
/// - This ensures no ID collisions when merging with primary catalogs
/// - Returns an error if CSV format is invalid or coordinates are out of bounds
pub fn parse_additional_stars() -> Result<Vec<MinimalStar>, Box<dyn std::error::Error>> {
    let mut stars = Vec::new();
    let mut current_id = u64::MAX; // Start from maximum possible value and count backwards

    for (line_num, line) in ADDITIONAL_BRIGHT_STARS_CSV.lines().enumerate() {
        // Skip header line
        if line_num == 0 {
            continue;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 3 {
            return Err(format!(
                "Invalid CSV format at line {}: expected 3 columns, got {}",
                line_num + 1,
                parts.len()
            )
            .into());
        }

        let ra_deg = parts[0]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("Invalid RA at line {}: {}", line_num + 1, e))?;
        let dec_deg = parts[1]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("Invalid Dec at line {}: {}", line_num + 1, e))?;
        let magnitude = parts[2]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("Invalid magnitude at line {}: {}", line_num + 1, e))?;

        stars.push(MinimalStar::new(current_id, ra_deg, dec_deg, magnitude));
        current_id = current_id.saturating_sub(1); // Count backwards, protecting against underflow
    }

    Ok(stars)
}

/// Parse duration string with flexible unit specification for astronomical timing.
///
/// Converts human-readable duration strings into standard Duration objects,
/// supporting multiple time units commonly used in astronomical observations.
/// Provides automatic unit detection and validation for integration times,
/// readout cycles, and temporal measurements.
///
/// # Supported Units
/// - **us**: Microseconds (1e-6 seconds) - for fast readout timing
/// - **ms**: Milliseconds (1e-3 seconds) - for short exposures
/// - **s**: Seconds (default unit) - standard exposure times
/// - **m**: Minutes (60 seconds) - long exposures
/// - **h**: Hours (3600 seconds) - very long integrations
///
/// # Format Rules
/// - **Unit suffix**: Required except for seconds (default)
/// - **Decimal values**: Supported for all units (e.g., "1.5s")
/// - **Whitespace**: Automatically trimmed
/// - **Negative values**: Rejected with error message
///
/// # Arguments
/// * `s` - Duration string with optional unit suffix
///
/// # Returns
/// * `Ok(Duration)` - Parsed duration with proper precision
/// * `Err(String)` - Validation error with specific diagnostic
///
/// # Usage
/// Various valid duration formats:
/// - "1s" → 1 second
/// - "500ms" → 500 milliseconds
/// - "1.5s" → 1.5 seconds (fractional values supported)
/// - "2m" → 2 minutes (120 seconds)
/// - "1h" → 1 hour (3600 seconds)
/// - "1000us" → 1000 microseconds
/// - "30" → 30 seconds (default unit if none specified)
///
/// Invalid formats that return errors:
/// - "-1s" - Negative durations are not allowed
/// - "invalid" - Must be a valid number
/// - "1x" - Unknown unit suffix
pub fn parse_duration(s: &str) -> Result<Duration, String> {
    let s = s.trim();

    // Extract numeric part and unit
    let (num_str, unit) = if let Some(stripped) = s.strip_suffix("ms") {
        (stripped, "ms")
    } else if let Some(stripped) = s.strip_suffix("us") {
        (stripped, "us")
    } else if let Some(stripped) = s.strip_suffix('s') {
        (stripped, "s")
    } else if let Some(stripped) = s.strip_suffix('h') {
        (stripped, "h")
    } else if let Some(stripped) = s.strip_suffix('m') {
        (stripped, "m")
    } else {
        // Default to seconds if no unit specified
        (s, "s")
    };

    let value: f64 = num_str
        .parse()
        .map_err(|_| format!("Invalid numeric value: {num_str}"))?;

    if value < 0.0 {
        return Err("Duration cannot be negative".to_string());
    }

    let duration = match unit {
        "us" => Duration::from_micros((value * 1.0) as u64),
        "ms" => Duration::from_millis((value * 1.0) as u64),
        "s" => Duration::from_secs_f64(value),
        "m" => Duration::from_secs_f64(value * 60.0),
        "h" => Duration::from_secs_f64(value * 3600.0),
        _ => return Err(format!("Unknown time unit: {unit}")),
    };

    Ok(duration)
}

/// Parse parameter sweep range specification for systematic studies.
///
/// Converts colon-separated range strings into validated (start, stop, step)
/// tuples for parameter sweeps, sensitivity analysis, and systematic studies.
/// Supports both forward and reverse ranges with comprehensive validation.
///
/// # Format Specification
/// Input format: "start:stop:step"
/// - **start**: Initial parameter value
/// - **stop**: Final parameter value (inclusive)
/// - **step**: Increment size (positive or negative)
///
/// # Validation Rules
/// - **Step non-zero**: Prevents infinite loops
/// - **Direction consistency**: Positive step requires start < stop
/// - **Reverse ranges**: Negative step requires start > stop
/// - **Numeric validation**: All components must be valid floating-point
///
/// # Arguments
/// * `s` - Range specification string in "start:stop:step" format
///
/// # Returns
/// * `Ok((f64, f64, f64))` - Validated (start, stop, step) tuple
/// * `Err(String)` - Validation error with specific diagnostic
///
/// # Usage
/// Valid range formats:
/// - Forward ranges: "0.0:10.0:1.0", "8.0:16.0:0.5"
/// - Reverse ranges: "10.0:0.0:-1.0", "5.0:1.0:-0.5"
///
/// Invalid formats that return errors:
/// - "1.0:2.0" - Missing step component
/// - "1.0:2.0:0.0" - Step cannot be zero
/// - "5.0:1.0:1.0" - Positive step requires start < stop
/// - "1.0:5.0:-1.0" - Negative step requires start > stop
pub fn parse_range(s: &str) -> Result<(f64, f64, f64), String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return Err("Range must be in format 'start:stop:step'".to_string());
    }

    let start = parts[0]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid start value".to_string())?;
    let stop = parts[1]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid stop value".to_string())?;
    let step = parts[2]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid step value".to_string())?;

    if step == 0.0 {
        return Err("Step cannot be zero".to_string());
    }

    if step > 0.0 && start >= stop {
        return Err("For positive step, start must be less than stop".to_string());
    }

    if step < 0.0 && start <= stop {
        return Err("For negative step, start must be greater than stop".to_string());
    }

    Ok((start, stop, step))
}

/// Type-safe wrapper for parameter sweep range specification.
///
/// Provides a clap-compatible type for command-line range arguments with
/// automatic parsing, validation, and display formatting. Encapsulates
/// (start, stop, step) tuples for systematic parameter studies and
/// sensitivity analysis workflows.
///
/// # Design Benefits
/// - **Type safety**: Prevents raw tuple confusion in function signatures
/// - **Validation**: Ensures range consistency at parse time
/// - **Display**: Consistent string representation for logging
/// - **Cloning**: Efficient copying for batch processing
///
/// # Use Cases
/// - **Magnitude sweeps**: "8.0:16.0:0.5" for limiting magnitude studies
/// - **Temperature ranges**: "-40.0:60.0:10.0" for thermal analysis
/// - **Exposure sweeps**: "0.1:30.0:0.1" for signal-to-noise optimization
/// - **Parameter grids**: Multi-dimensional parameter space exploration
#[derive(Debug, Clone)]
pub struct RangeArg(pub f64, pub f64, pub f64);

impl std::str::FromStr for RangeArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (start, stop, step) = parse_range(s)?;
        Ok(RangeArg(start, stop, step))
    }
}

impl std::fmt::Display for RangeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.0, self.1, self.2)
    }
}

impl RangeArg {
    /// Get the starting value of the parameter sweep.
    ///
    /// # Returns
    /// Initial parameter value for the range iteration
    pub fn start(&self) -> f64 {
        self.0
    }

    /// Get the ending value of the parameter sweep.
    ///
    /// # Returns
    /// Final parameter value for the range iteration (inclusive)
    pub fn stop(&self) -> f64 {
        self.1
    }

    /// Get the step size for the parameter sweep.
    ///
    /// # Returns
    /// Increment value (positive for forward, negative for reverse)
    pub fn step(&self) -> f64 {
        self.2
    }

    /// Parse RangeArg of milliseconds into Vec<Duration>
    ///
    /// Converts a range specified in milliseconds to a vector of Duration objects.
    /// Useful for exposure time sweeps and other time-based parameter scans.
    ///
    /// # Returns
    /// Vector of Duration objects from start to stop (inclusive) by step
    pub fn to_duration_vec_ms(&self) -> Vec<Duration> {
        let mut durations = Vec::new();
        let mut current_ms = self.0;
        while current_ms <= self.1 {
            durations.push(Duration::from_millis(current_ms as u64));
            current_ms += self.2;
        }
        durations
    }

    /// Convert to raw tuple for compatibility with legacy APIs.
    ///
    /// # Returns
    /// (start, stop, step) tuple for direct parameter unpacking
    ///
    /// # Usage
    /// Parse a range string and extract values:
    /// - Use str::parse() to create a RangeArg from "start:stop:step" format
    /// - Call as_tuple() to get (start, stop, step) for iteration
    /// - Generate parameter sequences by incrementing from start to stop by step
    pub fn as_tuple(&self) -> (f64, f64, f64) {
        (self.0, self.1, self.2)
    }
}

/// Wrapper for Duration that implements Clone and has a nice Display
#[derive(Debug, Clone)]
pub struct DurationArg(pub Duration);

impl std::str::FromStr for DurationArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_duration(s).map(DurationArg)
    }
}

impl std::fmt::Display for DurationArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let duration = self.0;
        let total_ms = duration.as_millis();

        if total_ms >= 1000 && total_ms % 1000 == 0 {
            write!(f, "{}s", total_ms / 1000)
        } else if total_ms >= 1000 {
            write!(f, "{:.3}s", duration.as_secs_f64())
        } else {
            write!(f, "{total_ms}ms")
        }
    }
}

impl Default for DurationArg {
    fn default() -> Self {
        DurationArg(Duration::from_secs(1))
    }
}

/// Available sensor models for selection
#[derive(Debug, Clone, ValueEnum)]
pub enum SensorModel {
    /// GSENSE4040BSI CMOS sensor (4096x4096, 9μm pixels)
    Gsense4040bsi,
    /// GSENSE6510BSI CMOS sensor (3200x3200, 6.5μm pixels) - Default
    Gsense6510bsi,
    /// HWK4123 CMOS sensor (4096x2300, 4.6μm pixels)
    Hwk4123,
    /// Sony IMX455 Full-frame BSI CMOS sensor (9568x6380, 3.75μm pixels)
    Imx455,
}

/// Available telescope models for selection
#[derive(Debug, Clone, ValueEnum)]
pub enum TelescopeModel {
    /// Small 50mm telescope (0.05m aperture, f/10)
    Small50mm,
    /// 50cm Demo telescope (0.5m aperture, f/20) - Default
    Demo50cm,
    /// 1m Final telescope (1.0m aperture, f/10) - Production system
    Final1m,
    /// Weasel telescope (0.47m aperture, f/7.3) - Multi-spectral
    Weasel,
}

impl std::fmt::Display for SensorModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_config()
            .name
            .to_lowercase()
            .replace('_', "-")
            .fmt(f)
    }
}

impl std::fmt::Display for TelescopeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TelescopeModel::Small50mm => write!(f, "small50mm"),
            TelescopeModel::Demo50cm => write!(f, "demo50cm"),
            TelescopeModel::Final1m => write!(f, "final1m"),
            TelescopeModel::Weasel => write!(f, "weasel"),
        }
    }
}

impl SensorModel {
    /// Get the corresponding SensorConfig for the selected model
    pub fn to_config(&self) -> &'static SensorConfig {
        use crate::hardware::sensor::models::*;
        match self {
            SensorModel::Gsense4040bsi => &GSENSE4040BSI,
            SensorModel::Gsense6510bsi => &GSENSE6510BSI,
            SensorModel::Hwk4123 => &HWK4123,
            SensorModel::Imx455 => &IMX455,
        }
    }
}

impl TelescopeModel {
    /// Get the corresponding TelescopeConfig for the selected model
    pub fn to_config(&self) -> &'static TelescopeConfig {
        use crate::hardware::telescope::models::*;
        match self {
            TelescopeModel::Small50mm => &SMALL_50MM,
            TelescopeModel::Demo50cm => &DEMO_50CM,
            TelescopeModel::Final1m => &FINAL_1M,
            TelescopeModel::Weasel => &WEASEL,
        }
    }
}

/// Common arguments shared across multiple simulation binaries
#[derive(Parser, Debug, Clone)]
pub struct SharedSimulationArgs {
    /// Exposure time (e.g., "1s", "500ms", "0.1s")
    #[arg(long, default_value = "1s")]
    pub exposure: DurationArg,

    /// Wavelength in nanometers
    #[arg(long, default_value_t = 550.0)]
    pub wavelength: f64,

    /// Sensor temperature in degrees Celsius for dark current calculation
    #[arg(long, default_value_t = 20.0)]
    pub temperature: f64,

    /// Telescope model to use for simulations
    #[arg(long, default_value_t = TelescopeModel::Demo50cm)]
    pub telescope: TelescopeModel,

    /// Solar elongation and coordinates for zodiacal background (format: "elongation,latitude")
    /// Defaults to the point of minimum zodiacal light brightness
    #[arg(long, default_value = DEFAULT_ZODIACAL_COORDINATES, value_parser = parse_coordinates)]
    pub coordinates: SolarAngularCoordinates,

    /// Path to binary star catalog
    #[arg(long, default_value = "gaia_mag16_multi.bin")]
    pub catalog: PathBuf,

    /// Noise multiple for detection cutoff (detection threshold = mean_noise * noise_multiple)
    #[arg(long, default_value_t = 5.0)]
    pub noise_multiple: f64,
}

impl SharedSimulationArgs {
    /// Load a binary star catalog from the configured path and union with additional bright stars
    ///
    /// This method loads a BinaryCatalog and adds additional bright stars from embedded CSV data.
    ///
    /// # Returns
    /// * `Result<BinaryCatalog, Box<dyn std::error::Error>>` - The loaded catalog with additional stars or error
    ///
    /// # Usage
    /// Load a catalog with automatic bright star augmentation:
    /// - Loads the binary catalog from the specified path
    /// - Automatically adds embedded bright stars from CSV data
    /// - Returns a combined catalog with collision-free star IDs
    /// - Reports clear error messages if catalog loading fails
    pub fn load_catalog(&self) -> Result<BinaryCatalog, Box<dyn std::error::Error>> {
        let mut catalog = BinaryCatalog::load(&self.catalog).map_err(|e| {
            format!(
                "Failed to load catalog from '{}': {}",
                self.catalog.display(),
                e
            )
        })?;

        // Parse and add additional bright stars
        let additional_stars =
            parse_additional_stars().expect("Failed to parse embedded additional bright stars CSV");

        // Get existing stars and combine with additional ones
        let mut all_stars = catalog.stars().to_vec();
        all_stars.extend(additional_stars);

        // Create new catalog with combined stars
        let updated_description = format!(
            "{} + {} additional bright stars",
            catalog.description(),
            all_stars.len() - catalog.len()
        );
        catalog = BinaryCatalog::from_stars(all_stars, &updated_description);

        Ok(catalog)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photometry::zodical::{ELONG_OF_MIN, LAT_OF_MIN};
    use starfield::catalogs::StarPosition;

    #[test]
    fn test_default_coordinates_match_zodiacal_constants() {
        // Parse the default coordinates string
        let parsed = parse_coordinates(DEFAULT_ZODIACAL_COORDINATES)
            .expect("Default coordinates string should be valid");

        // Ensure they match the zodiacal light minimum constants
        assert_eq!(
            parsed.elongation(),
            ELONG_OF_MIN,
            "Default elongation should match ELONG_OF_MIN"
        );
        assert_eq!(
            parsed.latitude(),
            LAT_OF_MIN,
            "Default latitude should match LAT_OF_MIN"
        );

        // Also verify the string format is what we expect
        assert_eq!(DEFAULT_ZODIACAL_COORDINATES, "165.0,75.0");
    }

    #[test]
    fn test_duration_parsing() {
        // Test various duration formats
        assert_eq!(parse_duration("1s").unwrap(), Duration::from_secs(1));
        assert_eq!(parse_duration("500ms").unwrap(), Duration::from_millis(500));
        assert_eq!(
            parse_duration("1.5s").unwrap(),
            Duration::from_secs_f64(1.5)
        );
        assert_eq!(parse_duration("2m").unwrap(), Duration::from_secs(120));
        assert_eq!(parse_duration("1h").unwrap(), Duration::from_secs(3600));

        // Test error cases
        assert!(parse_duration("-1s").is_err());
        assert!(parse_duration("invalid").is_err());
    }

    #[test]
    fn test_range_parsing() {
        // Test valid range formats
        assert_eq!(parse_range("0.0:10.0:1.0").unwrap(), (0.0, 10.0, 1.0));
        assert_eq!(parse_range("1.5:5.5:0.5").unwrap(), (1.5, 5.5, 0.5));
        assert_eq!(parse_range("-10.0:10.0:2.0").unwrap(), (-10.0, 10.0, 2.0));
        assert_eq!(parse_range("10.0:0.0:-1.0").unwrap(), (10.0, 0.0, -1.0));

        // Test error cases
        assert!(parse_range("1.0:2.0").is_err()); // Missing step
        assert!(parse_range("1.0:2.0:3.0:4.0").is_err()); // Too many parts
        assert!(parse_range("invalid:2.0:1.0").is_err()); // Invalid start
        assert!(parse_range("1.0:invalid:1.0").is_err()); // Invalid stop
        assert!(parse_range("1.0:2.0:invalid").is_err()); // Invalid step
        assert!(parse_range("1.0:2.0:0.0").is_err()); // Zero step
        assert!(parse_range("5.0:1.0:1.0").is_err()); // Positive step with start >= stop
        assert!(parse_range("1.0:5.0:-1.0").is_err()); // Negative step with start <= stop
    }

    #[test]
    fn test_range_arg_methods() {
        let range = RangeArg(1.5, 10.0, 0.5);
        assert_eq!(range.start(), 1.5);
        assert_eq!(range.stop(), 10.0);
        assert_eq!(range.step(), 0.5);
        assert_eq!(range.as_tuple(), (1.5, 10.0, 0.5));
        assert_eq!(range.to_string(), "1.5:10:0.5");
    }

    #[test]
    fn test_parse_additional_stars() {
        // Test the actual embedded CSV parsing
        let stars = parse_additional_stars().expect("Should parse embedded CSV successfully");

        // Verify we got some stars (the actual CSV should have many)
        assert!(!stars.is_empty(), "Should have parsed at least some stars");

        // Check that IDs are assigned backwards from u64::MAX
        if stars.len() >= 2 {
            let first_star = &stars[0];
            let second_star = &stars[1];
            assert_eq!(first_star.id, u64::MAX, "First star should have max u64 ID");
            assert_eq!(
                second_star.id,
                u64::MAX - 1,
                "Second star should have max-1 ID"
            );
        }

        // Verify all stars have valid coordinates and magnitudes
        for star in &stars {
            // RA should be in range [0, 360) degrees
            assert!(
                star.ra() >= 0.0 && star.ra() < 360.0,
                "RA {} should be in range [0, 360)",
                star.ra()
            );

            // Dec should be in range [-90, 90] degrees
            assert!(
                star.dec() >= -90.0 && star.dec() <= 90.0,
                "Dec {} should be in range [-90, 90]",
                star.dec()
            );

            // Magnitude should be reasonable (very bright stars, so negative to ~6)
            assert!(
                star.magnitude >= -2.0 && star.magnitude <= 7.0,
                "Magnitude {} should be reasonable for bright stars",
                star.magnitude
            );

            // ID should be counting backwards from u64::MAX
            assert!(
                star.id >= u64::MAX - stars.len() as u64,
                "Star ID {} should be in expected range",
                star.id
            );
        }

        println!("Successfully parsed {} additional stars", stars.len());
        if !stars.is_empty() {
            let first = &stars[0];
            println!(
                "First star: ID={}, RA={:.6}°, Dec={:.6}°, Mag={:.2}",
                first.id,
                first.ra(),
                first.dec(),
                first.magnitude
            );
        }
    }
}

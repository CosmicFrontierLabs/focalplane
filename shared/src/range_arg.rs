//! Type-safe range argument for parameter sweeps.
//!
//! Provides a clap-compatible type for command-line range arguments with
//! automatic parsing, validation, and display formatting.

use std::fmt;
use std::str::FromStr;

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
/// # Examples
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

impl FromStr for RangeArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (start, stop, step) = parse_range(s)?;
        Ok(RangeArg(start, stop, step))
    }
}

impl fmt::Display for RangeArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

    /// Generate a vector of all values in the range.
    ///
    /// # Returns
    /// `Ok(Vec<f64>)` containing all values from start to stop (inclusive) by step
    /// `Err(String)` if the range parameters are invalid
    ///
    /// # Errors
    /// - Returns error if step is zero or negative when stop > start
    /// - Returns error if step is positive when stop < start
    pub fn to_vec(&self) -> Result<Vec<f64>, String> {
        let (start, stop, step) = (self.0, self.1, self.2);

        // Validate range parameters
        if step == 0.0 {
            return Err("Step size cannot be zero".to_string());
        }

        if stop > start && step <= 0.0 {
            return Err(format!(
                "Invalid range: stop ({stop}) > start ({start}) but step ({step}) is not positive"
            ));
        }

        if stop < start && step >= 0.0 {
            return Err(format!(
                "Invalid range: stop ({stop}) < start ({start}) but step ({step}) is not negative"
            ));
        }

        let mut values = Vec::new();
        let mut current = start;

        // Generate values based on step direction
        if step > 0.0 {
            while current <= stop {
                values.push(current);
                current += step;
            }
        } else {
            while current >= stop {
                values.push(current);
                current += step;
            }
        }

        Ok(values)
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_range_arg_to_vec_ascending() {
        let range = RangeArg(0.0, 2.0, 0.5);
        let values = range.to_vec().unwrap();
        assert_eq!(values, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_range_arg_to_vec_descending() {
        let range = RangeArg(10.0, 8.0, -0.5);
        let values = range.to_vec().unwrap();
        assert_eq!(values, vec![10.0, 9.5, 9.0, 8.5, 8.0]);
    }

    #[test]
    fn test_range_arg_to_vec_inexact_end() {
        // Test range that doesn't hit stop exactly
        let range = RangeArg(0.0, 2.1, 0.5);
        let values = range.to_vec().unwrap();
        assert_eq!(values, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_range_arg_to_vec_single_value() {
        // Test single value range (start == stop)
        let range = RangeArg(5.0, 5.0, 1.0);
        let values = range.to_vec().unwrap();
        assert_eq!(values, vec![5.0]);
    }

    #[test]
    fn test_range_arg_to_vec_negative_values() {
        let range = RangeArg(-2.0, 0.0, 0.5);
        let values = range.to_vec().unwrap();
        assert_eq!(values, vec![-2.0, -1.5, -1.0, -0.5, 0.0]);
    }

    #[test]
    fn test_range_arg_to_vec_error_zero_step() {
        let range = RangeArg(0.0, 10.0, 0.0);
        let result = range.to_vec();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Step size cannot be zero"));
    }

    #[test]
    fn test_range_arg_to_vec_error_positive_step_descending() {
        // Test positive step with stop < start
        let range = RangeArg(10.0, 5.0, 1.0);
        let result = range.to_vec();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("stop (5) < start (10) but step (1) is not negative"));
    }

    #[test]
    fn test_range_arg_to_vec_error_negative_step_ascending() {
        // Test negative step with stop > start
        let range = RangeArg(5.0, 10.0, -1.0);
        let result = range.to_vec();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("stop (10) > start (5) but step (-1) is not positive"));
    }

    #[test]
    fn test_range_arg_to_vec_error_small_negative_step_ascending() {
        // Test edge case: very small negative step with stop > start
        let range = RangeArg(0.0, 1.0, -0.001);
        let result = range.to_vec();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("stop (1) > start (0) but step (-0.001) is not positive"));
    }

    #[test]
    fn test_range_arg_to_vec_fractional_values() {
        // Test with fractional values
        let range = RangeArg(0.1, 0.5, 0.1);
        let values = range.to_vec().unwrap();
        assert_eq!(values.len(), 5);
        assert!((values[0] - 0.1).abs() < 1e-10);
        assert!((values[1] - 0.2).abs() < 1e-10);
        assert!((values[2] - 0.3).abs() < 1e-10);
        assert!((values[3] - 0.4).abs() < 1e-10);
        assert!((values[4] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_range_arg_to_vec_large_step() {
        // Test with large step size
        let range = RangeArg(0.0, 100.0, 25.0);
        let values = range.to_vec().unwrap();
        assert_eq!(values, vec![0.0, 25.0, 50.0, 75.0, 100.0]);
    }
}

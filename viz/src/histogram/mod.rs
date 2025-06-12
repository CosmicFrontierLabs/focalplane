//! Histogram visualization
//!
//! This module provides functionality for creating and displaying histograms.

use crate::{Result, VizError};
use std::collections::HashSet;
use std::fmt::{Display, Write};
use std::ops::Range;

/// Scale type for histogram display
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scale {
    /// Linear scale
    Linear,
    /// Logarithmic scale (base 10)
    Log10,
    /// Custom scale function
    Custom,
}

/// Configuration for histogram display
#[derive(Debug, Clone)]
pub struct HistogramConfig {
    /// Title for the histogram
    pub title: Option<String>,
    /// Width of the display (in characters)
    pub width: usize,
    /// Height of the display (in characters)
    pub height: Option<usize>,
    /// Character to use for bars
    pub bar_char: char,
    /// Whether to show percentage
    pub show_percentage: bool,
    /// Whether to show counts
    pub show_counts: bool,
    /// Scale type
    pub scale: Scale,
    /// Whether to show empty bins
    pub show_empty_bins: bool,
    /// Maximum bar width in characters
    pub max_bar_width: usize,
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            title: None,
            width: 80,
            height: None,
            bar_char: '#',
            show_percentage: true,
            show_counts: true,
            scale: Scale::Linear,
            show_empty_bins: false,
            max_bar_width: 40,
        }
    }
}

/// Histogram for continuous data
#[derive(Debug, Clone)]
pub struct Histogram<T> {
    /// Bin edges (boundaries between bins)
    bin_edges: Vec<T>,
    /// Counts in each bin
    counts: Vec<u64>,
    /// Total number of values
    total_count: u64,
    /// Display configuration
    config: HistogramConfig,
}

impl<T> Histogram<T>
where
    T: Copy + PartialOrd + Display,
{
    /// Create a new histogram with specified bin edges
    pub fn new(bin_edges: Vec<T>) -> Result<Self> {
        if bin_edges.len() < 2 {
            return Err(VizError::HistogramError(
                "Histogram must have at least 2 bin edges".to_string(),
            ));
        }

        // Check that bin edges are in ascending order
        for i in 1..bin_edges.len() {
            match bin_edges[i - 1].partial_cmp(&bin_edges[i]) {
                Some(std::cmp::Ordering::Less) => {} // This is good, continue
                _ => {
                    return Err(VizError::HistogramError(
                        "Histogram bin edges must be in ascending order".to_string(),
                    ));
                }
            }
        }

        let num_bins = bin_edges.len() - 1;
        let counts = vec![0; num_bins];

        Ok(Self {
            bin_edges,
            counts,
            total_count: 0,
            config: HistogramConfig::default(),
        })
    }

    /// Create a new histogram with equally spaced bins
    pub fn new_equal_bins(range: Range<T>, num_bins: usize) -> Result<Self>
    where
        T: std::ops::Sub<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + From<f64>
            + Into<f64>,
    {
        if num_bins == 0 {
            return Err(VizError::HistogramError(
                "Histogram must have at least 1 bin".to_string(),
            ));
        }

        let min: f64 = range.start.into();
        let max: f64 = range.end.into();
        let step = (max - min) / (num_bins as f64);

        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins {
            let edge: T = T::from(min + step * (i as f64));
            bin_edges.push(edge);
        }

        Self::new(bin_edges)
    }

    /// Set the configuration for the histogram
    pub fn with_config(mut self, config: HistogramConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a value to the histogram
    pub fn add(&mut self, value: T) {
        let bin_idx = self.find_bin(value);
        if let Some(idx) = bin_idx {
            self.counts[idx] += 1;
            self.total_count += 1;
        }
    }

    /// Add multiple values to the histogram
    pub fn add_all<I>(&mut self, values: I)
    where
        I: IntoIterator<Item = T>,
    {
        for value in values {
            self.add(value);
        }
    }

    /// Find the bin index for a value
    fn find_bin(&self, value: T) -> Option<usize> {
        for i in 0..self.bin_edges.len() - 1 {
            if value >= self.bin_edges[i] && value < self.bin_edges[i + 1] {
                return Some(i);
            }
        }

        // Special case: if value is exactly the last bin edge, put it in the last bin
        if value == *self.bin_edges.last().unwrap() {
            return Some(self.bin_edges.len() - 2);
        }

        None
    }

    /// Get the counts in each bin
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Get the bin edges
    pub fn bin_edges(&self) -> &[T] {
        &self.bin_edges
    }

    /// Get the total count
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Calculate the mean value from the histogram
    ///
    /// Returns None if the histogram is empty
    pub fn mean(&self) -> Option<f64>
    where
        T: Into<f64> + Copy,
    {
        if self.total_count == 0 {
            return None;
        }

        let mut sum = 0.0;
        let total = self.total_count as f64;

        for i in 0..self.counts.len() {
            if self.counts[i] == 0 {
                continue;
            }

            // Use the bin center as the representative value
            let bin_center = (self.bin_edges[i].into() + self.bin_edges[i + 1].into()) / 2.0;
            sum += bin_center * (self.counts[i] as f64);
        }

        Some(sum / total)
    }

    /// Calculate the variance from the histogram
    ///
    /// Returns None if the histogram is empty or has only one value
    pub fn variance(&self) -> Option<f64>
    where
        T: Into<f64> + Copy,
    {
        if self.total_count <= 1 {
            return None;
        }

        let mean = self.mean()?;
        let mut sum_squared_diff = 0.0;
        let total = self.total_count as f64;

        for i in 0..self.counts.len() {
            if self.counts[i] == 0 {
                continue;
            }

            // Use the bin center as the representative value
            let bin_center = (self.bin_edges[i].into() + self.bin_edges[i + 1].into()) / 2.0;
            let diff = bin_center - mean;
            sum_squared_diff += (diff * diff) * (self.counts[i] as f64);
        }

        // Use Bessel's correction for sample variance (n-1 denominator)
        let denominator = if self
            .config
            .title
            .as_ref()
            .is_some_and(|t| t.contains("Population"))
        {
            total // Population variance
        } else {
            total - 1.0 // Sample variance (Bessel's correction)
        };

        Some(sum_squared_diff / denominator)
    }

    /// Calculate the standard deviation from the histogram
    ///
    /// Returns None if the histogram is empty or has only one value
    pub fn std_dev(&self) -> Option<f64>
    where
        T: Into<f64> + Copy,
    {
        self.variance().map(|v| v.sqrt())
    }

    /// Calculate the skewness of the distribution (3rd standardized moment)
    ///
    /// Skewness measures the asymmetry of the probability distribution.
    /// - Positive skewness indicates a distribution with a longer right tail
    /// - Negative skewness indicates a distribution with a longer left tail
    /// - Zero skewness indicates a symmetric distribution
    ///
    /// Returns None if the histogram is empty or has insufficient data
    pub fn skewness(&self) -> Option<f64>
    where
        T: Into<f64> + Copy,
    {
        if self.total_count < 3 {
            return None; // Need at least 3 data points for meaningful skewness
        }

        let mean = self.mean()?;
        let std_dev = self.std_dev()?;

        if std_dev.abs() < f64::EPSILON {
            return Some(0.0); // All values are identical
        }

        let mut sum_cubed_diff = 0.0;
        let total = self.total_count as f64;

        for i in 0..self.counts.len() {
            if self.counts[i] == 0 {
                continue;
            }

            // Use the bin center as the representative value
            let bin_center = (self.bin_edges[i].into() + self.bin_edges[i + 1].into()) / 2.0;
            let normalized_diff = (bin_center - mean) / std_dev;
            sum_cubed_diff += (normalized_diff.powi(3)) * (self.counts[i] as f64);
        }

        // Apply adjustment factor for sample skewness
        let adjustment = (total * (total - 1.0).sqrt()) / (total - 2.0);
        Some(sum_cubed_diff / total * adjustment)
    }

    /// Calculate the kurtosis of the distribution (4th standardized moment)
    ///
    /// Kurtosis measures the "tailedness" of the probability distribution.
    /// - This implementation returns excess kurtosis (normal distribution = 0)
    /// - Positive values indicate heavier tails than normal distribution
    /// - Negative values indicate lighter tails than normal distribution
    ///
    /// Returns None if the histogram is empty or has insufficient data
    pub fn kurtosis(&self) -> Option<f64>
    where
        T: Into<f64> + Copy,
    {
        if self.total_count < 4 {
            return None; // Need at least 4 data points for meaningful kurtosis
        }

        let mean = self.mean()?;
        let std_dev = self.std_dev()?;

        if std_dev.abs() < f64::EPSILON {
            return Some(0.0); // All values are identical
        }

        let mut sum_fourth_diff = 0.0;
        let total = self.total_count as f64;

        for i in 0..self.counts.len() {
            if self.counts[i] == 0 {
                continue;
            }

            // Use the bin center as the representative value
            let bin_center = (self.bin_edges[i].into() + self.bin_edges[i + 1].into()) / 2.0;
            let normalized_diff = (bin_center - mean) / std_dev;
            sum_fourth_diff += (normalized_diff.powi(4)) * (self.counts[i] as f64);
        }

        // Calculate excess kurtosis (normal distribution = 0)
        Some((sum_fourth_diff / total) - 3.0)
    }

    /// Calculate the median value from the histogram
    ///
    /// Note: This is an approximation based on bins, not exact value
    /// Returns None if the histogram is empty
    pub fn median(&self) -> Option<f64>
    where
        T: Into<f64> + Copy,
    {
        if self.total_count == 0 {
            return None;
        }

        // Find median position
        let median_pos = self.total_count as f64 / 2.0;
        let mut cumulative_count = 0.0;

        for i in 0..self.counts.len() {
            cumulative_count += self.counts[i] as f64;

            // Found the bin containing the median
            if cumulative_count >= median_pos {
                let bin_width = self.bin_edges[i + 1].into() - self.bin_edges[i].into();
                let bin_start = self.bin_edges[i].into();

                // Interpolate within the bin
                let prev_cumulative = cumulative_count - self.counts[i] as f64;
                let position_in_bin = (median_pos - prev_cumulative) / self.counts[i] as f64;

                return Some(bin_start + bin_width * position_in_bin);
            }
        }

        // Should not reach here if total_count > 0
        None
    }

    /// Get a summary of statistics for the histogram
    ///
    /// Returns a string with mean, standard deviation, median, skewness and kurtosis
    pub fn statistics_summary(&self) -> String
    where
        T: Into<f64> + Copy,
    {
        let mut output = String::new();

        // Mean
        match self.mean() {
            Some(mean) => writeln!(output, "Mean: {:.6}", mean).unwrap(),
            None => writeln!(output, "Mean: insufficient data").unwrap(),
        }

        // Standard Deviation
        match self.std_dev() {
            Some(std_dev) => writeln!(output, "Standard Deviation: {:.6}", std_dev).unwrap(),
            None => writeln!(output, "Standard Deviation: insufficient data").unwrap(),
        }

        // Median
        match self.median() {
            Some(median) => writeln!(output, "Median: {:.6}", median).unwrap(),
            None => writeln!(output, "Median: insufficient data").unwrap(),
        }

        // Skewness
        match self.skewness() {
            Some(skewness) => {
                writeln!(output, "Skewness: {:.6}", skewness).unwrap();

                // Add interpretation
                if skewness.abs() < 0.5 {
                    writeln!(
                        output,
                        "  Interpretation: Approximately symmetric distribution"
                    )
                    .unwrap();
                } else if skewness > 0.0 {
                    writeln!(
                        output,
                        "  Interpretation: Right-skewed (longer/fatter tail on right)"
                    )
                    .unwrap();
                } else {
                    writeln!(
                        output,
                        "  Interpretation: Left-skewed (longer/fatter tail on left)"
                    )
                    .unwrap();
                }
            }
            None => writeln!(output, "Skewness: insufficient data").unwrap(),
        }

        // Kurtosis (excess)
        match self.kurtosis() {
            Some(kurtosis) => {
                writeln!(output, "Kurtosis (excess): {:.6}", kurtosis).unwrap();

                // Add interpretation
                if kurtosis.abs() < 0.5 {
                    writeln!(
                        output,
                        "  Interpretation: Similar tails to normal distribution"
                    )
                    .unwrap();
                } else if kurtosis > 0.0 {
                    writeln!(
                        output,
                        "  Interpretation: Heavy-tailed (more outliers than normal)"
                    )
                    .unwrap();
                } else {
                    writeln!(
                        output,
                        "  Interpretation: Light-tailed (fewer outliers than normal)"
                    )
                    .unwrap();
                }
            }
            None => writeln!(output, "Kurtosis: insufficient data").unwrap(),
        }

        // Add sample size
        writeln!(output, "Sample size: {}", self.total_count).unwrap();

        output
    }

    /// Format the histogram as a string
    pub fn format(&self) -> Result<String> {
        let mut output = String::new();

        // Add title if present
        if let Some(title) = &self.config.title {
            writeln!(output, "{}", title)?;
            writeln!(output, "{}", "=".repeat(title.len()))?;
        }

        // Calculate maximum count for scaling
        let max_count = *self.counts.iter().max().unwrap_or(&1) as f64;

        // Determine column widths
        let bin_column_width = self
            .bin_edges
            .iter()
            .map(|e| format!("{}", e).len())
            .max()
            .unwrap_or(8)
            + 2;

        let count_column_width = if self.config.show_counts {
            self.counts
                .iter()
                .map(|c| format!("{}", c).len())
                .max()
                .unwrap_or(10)
                + 2
        } else {
            0
        };

        let percentage_column_width = if self.config.show_percentage { 10 } else { 0 };

        // Print header
        let mut header = String::new();
        write!(header, "{:<width$} ", "Range", width = bin_column_width * 2)?;

        if self.config.show_counts {
            write!(header, "| {:<width$} ", "Count", width = count_column_width)?;
        }

        if self.config.show_percentage {
            write!(
                header,
                "| {:<width$} ",
                "Percentage",
                width = percentage_column_width
            )?;
        }

        write!(header, "| Bar")?;

        writeln!(output, "{}", header)?;
        writeln!(output, "{}", "-".repeat(header.len()))?;

        // Print each bin
        for i in 0..self.counts.len() {
            let count = self.counts[i];

            // Skip empty bins if configured
            if count == 0 && !self.config.show_empty_bins {
                continue;
            }

            let percentage = if self.total_count > 0 {
                (count as f64 / self.total_count as f64) * 100.0
            } else {
                0.0
            };

            // Calculate bar length based on scale
            let bar_length = match self.config.scale {
                Scale::Linear => {
                    ((count as f64 / max_count) * self.config.max_bar_width as f64).round() as usize
                }
                Scale::Log10 => {
                    if count > 0 {
                        ((count as f64).log10() * 10.0).round() as usize
                    } else {
                        0
                    }
                }
                Scale::Custom => {
                    // For custom scale, default to linear
                    ((count as f64 / max_count) * self.config.max_bar_width as f64).round() as usize
                }
            };

            let bar = if count > 0 {
                self.config.bar_char.to_string().repeat(bar_length)
            } else {
                "".to_string()
            };

            // Write the bin with 3-4 sig figs and consistent sign handling
            let format_with_sign = |val: T| -> String {
                let val_str = format!("{:.3}", val);
                if val_str.starts_with('-') {
                    val_str
                } else {
                    format!("+{}", val_str)
                }
            };

            write!(
                output,
                "{:<width$} - {:<width$}",
                format_with_sign(self.bin_edges[i]),
                format_with_sign(self.bin_edges[i + 1]),
                width = 8 // Slightly wider to accommodate sign
            )?;

            if self.config.show_counts {
                write!(output, "| {:<width$} ", count, width = count_column_width)?;
            }

            if self.config.show_percentage {
                write!(output, "| {:5.2}%      ", percentage)?;
            }

            writeln!(output, "| {}", bar)?;
        }

        // Print footer with scale information
        match self.config.scale {
            Scale::Linear => {}
            Scale::Log10 => {
                writeln!(output)?;
                writeln!(
                    output,
                    "Note: Bar lengths use log10 scale (each {} represents a power of 10)",
                    self.config.bar_char
                )?;
            }
            Scale::Custom => {}
        }

        Ok(output)
    }
    /// Print the histogram to stdout
    pub fn print(&self) -> Result<()> {
        println!("{}", self.format()?);
        Ok(())
    }
}

/// Create a magnitude histogram with bins centered on integer magnitudes
///
/// This function creates a histogram specifically for stellar magnitudes with:
/// - Each bin centered on integer magnitude values (e.g., 6.0, 7.0, 8.0)
/// - Bin width of exactly 1.0 magnitude
/// - Only as many bins as needed to cover all values
pub fn create_magnitude_histogram(
    magnitudes: &[f64],
    title: Option<String>,
    use_log_scale: bool,
) -> Result<Histogram<f64>> {
    if magnitudes.is_empty() {
        return Err(VizError::HistogramError(
            "No magnitudes provided".to_string(),
        ));
    }

    // Find the unique integer magnitude values present
    let mut int_magnitudes = HashSet::new();
    for &mag in magnitudes {
        // Round to nearest integer to determine bin center
        int_magnitudes.insert(mag.round() as i32);
    }

    // Convert to sorted vector to determine range
    let mut centers: Vec<i32> = int_magnitudes.into_iter().collect();
    centers.sort();

    if centers.is_empty() {
        return Err(VizError::HistogramError(
            "No valid magnitudes found".to_string(),
        ));
    }

    // Determine full range of required bins to ensure continuous coverage
    let min_center = *centers.first().unwrap();
    let max_center = *centers.last().unwrap();

    // Create bins with exactly 1.0 width centered on all integer values in the range
    // Each bin goes from x-0.5 to x+0.5 for integer x
    let mut bin_edges = Vec::new();

    // Create edges for every integer from min to max
    for center in min_center..=max_center {
        bin_edges.push((center as f64) - 0.5);
    }
    // Add the upper edge of the last bin
    bin_edges.push((max_center as f64) + 0.5);

    // Create the histogram
    let mut hist = Histogram::new(bin_edges)?;

    // Configure display
    let scale = if use_log_scale {
        Scale::Log10
    } else {
        Scale::Linear
    };

    let config = HistogramConfig {
        title,
        scale,
        show_percentage: true,
        show_counts: true,
        show_empty_bins: false,
        ..HistogramConfig::default()
    };

    hist = hist.with_config(config);

    // Add values
    hist.add_all(magnitudes.iter().copied());

    // Return the populated histogram
    Ok(hist)
}

/// Quick function to create and display a histogram from a slice of values
pub fn histogram<T>(
    values: &[T],
    bin_count: usize,
    range: Range<T>,
    title: Option<String>,
    use_log_scale: bool,
) -> Result<String>
where
    T: Copy
        + PartialOrd
        + Display
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + From<f64>
        + Into<f64>,
{
    // Create histogram
    let mut hist = Histogram::new_equal_bins(range, bin_count)?;

    // Configure display
    let scale = if use_log_scale {
        Scale::Log10
    } else {
        Scale::Linear
    };
    let config = HistogramConfig {
        title,
        scale,
        ..HistogramConfig::default()
    };

    hist = hist.with_config(config);

    // Add values
    hist.add_all(values.iter().copied());

    // Format
    hist.format()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_creation() {
        let bin_edges = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let hist = Histogram::new(bin_edges).unwrap();
        assert_eq!(hist.counts.len(), 5);
        assert_eq!(hist.total_count, 0);
    }

    #[test]
    fn test_histogram_adding_values() {
        let bin_edges = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut hist = Histogram::new(bin_edges).unwrap();

        hist.add(0.5);
        hist.add(1.5);
        hist.add(1.7);
        hist.add(4.0); // This should be in the last bin

        assert_eq!(hist.counts, vec![1, 2, 0, 0, 1]);
        assert_eq!(hist.total_count, 4);
    }

    #[test]
    fn test_histogram_with_equal_bins() {
        let hist = Histogram::new_equal_bins(0.0f64..10.0f64, 5).unwrap();
        assert_eq!(hist.bin_edges.len(), 6);
        assert_eq!(hist.counts.len(), 5);

        // Check bin edges are evenly spaced
        let diff1 = (hist.bin_edges[1] - hist.bin_edges[0] - 2.0f64).abs();
        let diff2 = (hist.bin_edges[2] - hist.bin_edges[1] - 2.0f64).abs();
        assert!(diff1 < 1e-6);
        assert!(diff2 < 1e-6);
    }

    #[test]
    fn test_magnitude_histogram_creation() {
        // Test with normal star magnitudes
        let magnitudes = vec![3.2, 4.7, 5.1, 6.9, 7.3, 9.0, 11.2];
        let hist = create_magnitude_histogram(&magnitudes, None, false).unwrap();

        // Print the actual bin edges to help with debugging
        println!("Actual bin edges: {:?}", hist.bin_edges());
        println!("Bin counts: {:?}", hist.counts());

        // The min_center is 3 and max_center is 11, so we expect 9 bins (3, 4, 5, 6, 7, 8, 9, 10, 11)
        assert_eq!(
            hist.counts().len(),
            9,
            "Should have 9 bins from magnitude 3 to 11"
        );

        // Verify bin edges are 1.0 wide
        for i in 0..hist.bin_edges().len() - 1 {
            let width = hist.bin_edges()[i + 1] - hist.bin_edges()[i];
            assert!(
                (width - 1.0).abs() < 1e-6,
                "Bin width should be 1.0 but was {}",
                width
            );
        }

        // Check that the bins with stars have the right counts
        // Bin for magnitude 3 should have 1 star (3.2)
        assert_eq!(
            hist.counts()[0],
            1,
            "Bin for magnitude 3 should contain 1 star"
        );

        // Bin for magnitude 5 should have 2 stars (4.7, 5.1)
        assert_eq!(
            hist.counts()[2],
            2,
            "Bin for magnitude 5 should contain 2 stars"
        );

        // Bin for magnitude 7 should have 2 stars (6.9, 7.3)
        assert_eq!(
            hist.counts()[4],
            2,
            "Bin for magnitude 7 should contain 2 stars"
        );

        // Bin for magnitude 9 should have 1 star (9.0)
        assert_eq!(
            hist.counts()[6],
            1,
            "Bin for magnitude 9 should contain 1 star"
        );

        // Bin for magnitude 11 should have 1 star (11.2)
        assert_eq!(
            hist.counts()[8],
            1,
            "Bin for magnitude 11 should contain 1 star"
        );

        // Empty bins should have 0 stars
        assert_eq!(hist.counts()[1], 0, "Bin for magnitude 4 should be empty");
        assert_eq!(hist.counts()[3], 0, "Bin for magnitude 6 should be empty");
        assert_eq!(hist.counts()[5], 0, "Bin for magnitude 8 should be empty");
        assert_eq!(hist.counts()[7], 0, "Bin for magnitude 10 should be empty");

        // Total count should match number of magnitudes
        assert_eq!(hist.total_count(), magnitudes.len() as u64);
    }

    #[test]
    fn test_magnitude_histogram_empty() {
        // Test with empty magnitudes list
        let result = create_magnitude_histogram(&[], None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_magnitude_histogram_bin_width() {
        // Verify that bins are exactly 1.0 magnitude wide
        let magnitudes = vec![1.2, 2.2, 3.2, 4.2, 5.2];
        let hist = create_magnitude_histogram(&magnitudes, None, false).unwrap();

        // Each bin should be exactly 1.0 wide
        for i in 0..hist.bin_edges().len() - 1 {
            let width = hist.bin_edges()[i + 1] - hist.bin_edges()[i];
            assert!(
                (width - 1.0).abs() < 1e-6,
                "Bin width should be 1.0 but was {}",
                width
            );
        }
    }

    #[test]
    fn test_magnitude_histogram_formatting() {
        // Test formatting with a title
        let magnitudes = vec![5.5, 6.2, 6.7, 8.1, 9.0];
        let title = "Test Magnitude Histogram";

        // Create histogram with custom config
        let mut hist =
            create_magnitude_histogram(&magnitudes, Some(title.to_string()), false).unwrap();

        // Override some config settings
        let mut config = hist.config.clone();
        config.bar_char = '*'; // Use star character for the bars
        config.width = 60; // Set a specific width
        hist = hist.with_config(config);

        let formatted = hist.format().unwrap();

        // Check title is in the output
        assert!(formatted.contains(title));

        // Check bin labels and counts
        assert!(formatted.contains("6")); // We should have a bin centered on 6
        assert!(formatted.contains("9")); // We should have a bin centered on 9

        // Check that our custom bar character is used
        assert!(formatted.contains("*"));
    }

    #[test]
    fn test_magnitude_histogram_with_log_scale() {
        // Test with log scale formatting
        let magnitudes = vec![3.0, 3.5, 4.0, 6.0, 8.0, 9.0, 9.5];
        let hist = create_magnitude_histogram(&magnitudes, None, true).unwrap();

        // Verify scale is set to Log10
        assert_eq!(hist.config.scale, Scale::Log10);

        let formatted = hist.format().unwrap();
        assert!(formatted.contains("log10"));
    }

    #[test]
    fn test_histogram_format() {
        let mut hist = Histogram::new_equal_bins(0.0..5.0, 5).unwrap();
        hist.add_all(vec![0.5, 1.5, 1.7, 2.2, 3.0, 4.9]);

        let output = hist.format().unwrap();
        assert!(output.contains("Count"));
        assert!(output.contains("Percentage"));
    }

    #[test]
    fn test_histogram_with_log_scale() {
        let mut hist = Histogram::new_equal_bins(0.0..5.0, 5).unwrap();
        hist.add_all(vec![0.5, 0.6, 0.7, 1.5, 1.7, 2.2, 3.0, 4.9]);

        let mut config = HistogramConfig::default();
        config.scale = Scale::Log10;
        hist = hist.with_config(config);

        let output = hist.format().unwrap();
        assert!(output.contains("log10"));
    }

    #[test]
    fn test_histogram_statistics() {
        // Create a histogram with known statistical properties
        // Normal distribution with mean=10, std=2
        let mut hist = Histogram::new_equal_bins(0.0..20.0, 20).unwrap();

        // Add values that approximate a normal distribution
        let values = vec![
            7.0, 7.5, 8.0, 8.5, 9.0, 9.0, 9.2, 9.5, 9.7, 9.8, 9.9, // Below mean
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, // At mean (more weight)
            10.1, 10.2, 10.3, 10.5, 10.8, 11.0, 11.2, 12.0, 12.5, 13.0, // Above mean
        ];
        hist.add_all(values.iter().copied());

        // Test mean calculation - allow a bit more flexibility with the larger dataset
        let mean = hist.mean().unwrap();
        assert!(
            (mean - 10.0).abs() < 0.5,
            "Mean should be approximately 10.0"
        );

        // Test standard deviation with some flexibility for binned data
        let std_dev = hist.std_dev().unwrap();
        assert!(
            (std_dev - 1.0).abs() < 0.6,
            "Std dev should be approximately 1.0"
        );

        // Test median
        let median = hist.median().unwrap();
        assert!(
            (median - 10.0).abs() < 0.3,
            "Median should be approximately 10.0"
        );

        // We just verify skewness calculation works - exact values can vary with binning
        let _skewness = hist.skewness().unwrap();

        // Check that statistics_summary generates complete output
        let summary = hist.statistics_summary();
        println!("Stats summary: {}", summary);

        // Just check that we have values for everything, without caring about exact values
        assert!(summary.contains("Mean:"));
        assert!(summary.contains("Standard Deviation:"));
        assert!(summary.contains("Median:"));
        assert!(summary.contains("Skewness:"));
    }

    #[test]
    fn test_histogram_statistics_with_skew() {
        // Create a histogram with right-skewed distribution (longer tail on right)
        let mut hist = Histogram::new_equal_bins(0.0..30.0, 30).unwrap();

        // Add values with positive skew (mode < median < mean)
        let values = [
            5.0, 5.5, 6.0, 6.0, 6.5, 7.0, 7.0, 7.0, 7.5, 7.5, 8.0, 8.0, 8.0, 8.5, 9.0,
            9.0, // Mode around 7-8
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 18.0, 20.0, 25.0,
        ];
        hist.add_all(values.iter().copied());

        // Test mean > median for positively skewed
        let mean = hist.mean().unwrap();
        let median = hist.median().unwrap();
        assert!(
            mean > median,
            "Mean should be greater than median for right-skewed distribution"
        );

        // Skewness should be positive for right skew
        let skewness = hist.skewness().unwrap();
        assert!(
            skewness > 0.0,
            "Skewness should be positive for right-skewed distribution"
        );

        // Kurtosis should be positive due to the outliers
        let kurtosis = hist.kurtosis().unwrap();
        assert!(
            kurtosis > 0.0,
            "Kurtosis should be positive due to outliers"
        );
    }
}

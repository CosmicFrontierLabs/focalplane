//! Histogram visualization
//!
//! This module provides functionality for creating and displaying histograms.

use crate::{Result, VizError};
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
}

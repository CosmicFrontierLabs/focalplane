//! Tracking visualization plots for monocle FGS testing
//!
//! Provides plotting utilities to visualize tracking performance,
//! showing frame arrival times and estimated vs actual positions.

use plotters::prelude::*;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::time::Duration;

/// Data point for tracking visualization
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingDataPoint {
    /// Time since tracking started
    pub time: Duration,
    /// Actual position (from ground truth)
    pub actual_x: f64,
    pub actual_y: f64,
    /// Estimated position (from tracking system)
    pub estimated_x: f64,
    pub estimated_y: f64,
    /// Whether this is a frame arrival event
    pub is_frame_arrival: bool,
    /// Whether tracking lock is established at this point
    pub has_lock: bool,
}

/// Tracking plot configuration
pub struct TrackingPlotConfig {
    /// Output file name (without path)
    pub output_filename: String,
    /// Plot width in pixels
    pub width: u32,
    /// Plot height in pixels
    pub height: u32,
    /// Title for the plot
    pub title: String,
    /// Maximum time range to display (seconds)
    pub max_time_seconds: f64,
}

impl Default for TrackingPlotConfig {
    fn default() -> Self {
        Self {
            output_filename: "tracking_plot.png".to_string(),
            width: 1200,
            height: 800,
            title: "FGS Tracking Performance".to_string(),
            max_time_seconds: 30.0,
        }
    }
}

/// Tracking data collector for visualization
pub struct TrackingPlotter {
    /// Configuration
    config: TrackingPlotConfig,
    /// Collected data points
    data_points: VecDeque<TrackingDataPoint>,
    /// Time when lock was first established
    lock_established_time: Option<Duration>,
}

impl Default for TrackingPlotter {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackingPlotter {
    /// Create new tracking plotter with default config
    pub fn new() -> Self {
        Self::with_config(TrackingPlotConfig::default())
    }

    /// Create new tracking plotter with custom config
    pub fn with_config(config: TrackingPlotConfig) -> Self {
        Self {
            config,
            data_points: VecDeque::new(),
            lock_established_time: None,
        }
    }

    /// Add a tracking data point
    pub fn add_point(&mut self, point: TrackingDataPoint) {
        // Track when lock is first established
        if point.has_lock && self.lock_established_time.is_none() {
            self.lock_established_time = Some(point.time);
        }

        // Keep only recent data within time window
        let cutoff_time = point
            .time
            .saturating_sub(Duration::from_secs_f64(self.config.max_time_seconds));
        while let Some(oldest) = self.data_points.front() {
            if oldest.time < cutoff_time {
                self.data_points.pop_front();
            } else {
                break;
            }
        }

        self.data_points.push_back(point);
    }

    /// Generate the tracking plot
    pub fn generate_plot(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure plots directory exists
        fs::create_dir_all("plots")?;

        // Build full path in plots directory
        let output_path = format!("plots/{}", self.config.output_filename);

        // Create drawing backend using BitMapBackend for PNG output
        let root = BitMapBackend::new(&output_path, (self.config.width, self.config.height))
            .into_drawing_area();

        root.fill(&WHITE)?;

        // Split into two subplots for X and Y tracking
        let areas = root.split_evenly((2, 1));
        let upper = &areas[0];
        let lower = &areas[1];

        // Draw X-axis tracking plot
        self.draw_axis_plot(upper, "X Position Tracking", true)?;

        // Draw Y-axis tracking plot
        self.draw_axis_plot(lower, "Y Position Tracking", false)?;

        root.present()?;
        println!(
            "Tracking plot saved to: plots/{}",
            self.config.output_filename
        );

        Ok(())
    }

    /// Generate residuals plot in a 2x2 layout
    pub fn generate_residuals_plot(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure plots directory exists
        fs::create_dir_all("plots")?;

        // Build output path for residuals plot
        let residuals_filename = self
            .config
            .output_filename
            .replace(".png", "_residuals.png");
        let output_path = format!("plots/{residuals_filename}");

        // Create drawing backend
        let root = BitMapBackend::new(&output_path, (self.config.width, self.config.height))
            .into_drawing_area();

        root.fill(&WHITE)?;

        // Split into 2x2 grid
        let areas = root.split_evenly((2, 2));

        // Calculate residuals (only for points with lock)
        let residuals: Vec<(Duration, f64, f64)> = self
            .data_points
            .iter()
            .filter(|p| p.has_lock)
            .map(|p| {
                (
                    p.time,
                    p.estimated_x - p.actual_x,
                    p.estimated_y - p.actual_y,
                )
            })
            .collect();

        if !residuals.is_empty() {
            // Top left: X residuals time series
            self.draw_residual_timeseries(&areas[0], &residuals, true, "X Residuals")?;

            // Top right: X residuals histogram
            self.draw_residual_histogram(&areas[1], &residuals, true, "X Residual Distribution")?;

            // Bottom left: Y residuals time series
            self.draw_residual_timeseries(&areas[2], &residuals, false, "Y Residuals")?;

            // Bottom right: Y residuals histogram
            self.draw_residual_histogram(&areas[3], &residuals, false, "Y Residual Distribution")?;
        }

        root.present()?;
        println!("Residuals plot saved to: plots/{residuals_filename}");

        Ok(())
    }

    /// Draw residual time series plot
    fn draw_residual_timeseries(
        &self,
        area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        residuals: &[(Duration, f64, f64)],
        is_x_axis: bool,
        title: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if residuals.is_empty() {
            return Ok(());
        }

        // Get time bounds
        let time_min = 0.0;
        let time_max = residuals
            .last()
            .map(|(t, _, _)| t.as_secs_f64())
            .unwrap_or(1.0)
            * 1.05;

        // Get residual bounds
        let mut res_min = f64::INFINITY;
        let mut res_max = f64::NEG_INFINITY;

        for (_, x_res, y_res) in residuals {
            let res = if is_x_axis { *x_res } else { *y_res };
            res_min = res_min.min(res);
            res_max = res_max.max(res);
        }

        // Add margin and ensure symmetric around zero
        let max_abs = res_min.abs().max(res_max.abs()) * 1.1;
        res_min = -max_abs;
        res_max = max_abs;

        // Create chart
        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 25))
            .margin(5)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(time_min..time_max, res_min..res_max)?;

        chart
            .configure_mesh()
            .x_desc("Time (seconds)")
            .y_desc("Residual (pixels)")
            .x_label_formatter(&|x| format!("{x:.1}"))
            .y_label_formatter(&|y| format!("{y:.3}"))
            .draw()?;

        // Draw zero line
        chart.draw_series(LineSeries::new(
            vec![(time_min, 0.0), (time_max, 0.0)],
            &BLACK.mix(0.3),
        ))?;

        // Draw residuals
        let data: Vec<(f64, f64)> = residuals
            .iter()
            .map(|(t, x_res, y_res)| {
                let res = if is_x_axis { *x_res } else { *y_res };
                (t.as_secs_f64(), res)
            })
            .collect();

        chart.draw_series(LineSeries::new(data.clone(), &BLUE))?;

        // Also draw points for clarity
        chart.draw_series(
            data.iter()
                .map(|&(x, y)| Circle::new((x, y), 2, BLUE.filled())),
        )?;

        Ok(())
    }

    /// Draw residual histogram
    fn draw_residual_histogram(
        &self,
        area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        residuals: &[(Duration, f64, f64)],
        is_x_axis: bool,
        title: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if residuals.is_empty() {
            return Ok(());
        }

        // Extract residual values
        let res_values: Vec<f64> = residuals
            .iter()
            .map(|(_, x_res, y_res)| if is_x_axis { *x_res } else { *y_res })
            .collect();

        // Calculate statistics
        let mean = res_values.iter().sum::<f64>() / res_values.len() as f64;
        let variance =
            res_values.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / res_values.len() as f64;
        let std_dev = variance.sqrt();

        // Determine histogram bounds
        let min_val = res_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = res_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;
        let hist_min = min_val - range * 0.1;
        let hist_max = max_val + range * 0.1;

        // Create histogram bins
        let num_bins = 30;
        let bin_width = (hist_max - hist_min) / num_bins as f64;
        let mut bins = vec![0u32; num_bins];

        for &res in &res_values {
            let bin_idx = ((res - hist_min) / bin_width) as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            bins[bin_idx] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&1) as f64;

        // Create chart
        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 25))
            .margin(5)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(hist_min..hist_max, 0.0..max_count * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Residual (pixels)")
            .y_desc("Count")
            .x_label_formatter(&|x| format!("{x:.3}"))
            .y_label_formatter(&|y| format!("{y:.0}"))
            .draw()?;

        // Draw histogram bars
        chart.draw_series(bins.iter().enumerate().map(|(i, &count)| {
            let x0 = hist_min + i as f64 * bin_width;
            let x1 = x0 + bin_width;
            Rectangle::new([(x0, 0.0), (x1, count as f64)], BLUE.mix(0.5).filled())
        }))?;

        // Draw mean line
        chart
            .draw_series(LineSeries::new(
                vec![(mean, 0.0), (mean, max_count * 1.1)],
                &RED,
            ))?
            .label(format!("μ={mean:.3}"))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));

        // Draw ±σ lines
        chart.draw_series(LineSeries::new(
            vec![(mean - std_dev, 0.0), (mean - std_dev, max_count * 1.1)],
            RED.mix(0.5).stroke_width(1),
        ))?;

        chart
            .draw_series(LineSeries::new(
                vec![(mean + std_dev, 0.0), (mean + std_dev, max_count * 1.1)],
                RED.mix(0.5).stroke_width(1),
            ))?
            .label(format!("σ={std_dev:.3}"))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED.mix(0.5)));

        // Draw legend
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        Ok(())
    }

    /// Draw a single axis tracking plot
    fn draw_axis_plot(
        &self,
        area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        title: &str,
        is_x_axis: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get data bounds
        let (time_min, time_max, pos_min, pos_max) = self.get_data_bounds(is_x_axis);

        // Create chart
        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(time_min..time_max, pos_min..pos_max)?;

        // Configure mesh
        chart
            .configure_mesh()
            .x_desc("Time (seconds)")
            .y_desc("Position (pixels)")
            .x_label_formatter(&|x| format!("{x:.1}"))
            .y_label_formatter(&|y| format!("{y:.3}"))
            .draw()?;

        // Draw shaded regions for different states
        if !self.data_points.is_empty() {
            let mut last_time = 0.0;
            let mut last_state = if self.data_points[0].has_lock {
                "tracking"
            } else {
                "acquiring"
            };

            for point in &self.data_points {
                let time_sec = point.time.as_secs_f64();
                let current_state = if point.has_lock {
                    "tracking"
                } else {
                    "acquiring"
                };

                // When state changes, draw region for previous state
                let is_last = std::ptr::eq(point, self.data_points.back().unwrap());
                if current_state != last_state || is_last {
                    let color = match last_state {
                        "acquiring" => RED.mix(0.1), // Red for acquiring/starting to lock
                        "tracking" => BLUE.mix(0.1), // Blue for tracking
                        _ => WHITE.into(),
                    };

                    chart.draw_series(std::iter::once(Rectangle::new(
                        [(last_time, pos_min), (time_sec, pos_max)],
                        color.filled(),
                    )))?;

                    // If transitioning to locked, draw green band
                    if last_state == "acquiring" && current_state == "tracking" {
                        chart.draw_series(std::iter::once(Rectangle::new(
                            [(time_sec - 0.05, pos_min), (time_sec + 0.05, pos_max)],
                            GREEN.mix(0.2).filled(),
                        )))?;
                    }

                    last_time = time_sec;
                    last_state = current_state;
                }
            }

            // Draw final region
            if let Some(last_point) = self.data_points.back() {
                let final_time = last_point.time.as_secs_f64();
                let color = if last_point.has_lock {
                    BLUE.mix(0.1)
                } else {
                    RED.mix(0.1)
                };

                chart.draw_series(std::iter::once(Rectangle::new(
                    [(last_time, pos_min), (final_time, pos_max)],
                    color.filled(),
                )))?;
            }
        }

        // Prepare data for plotting
        let mut actual_data = Vec::new();
        let mut estimated_data = Vec::new();

        for point in &self.data_points {
            let time_sec = point.time.as_secs_f64();

            if is_x_axis {
                actual_data.push((time_sec, point.actual_x));
                if point.has_lock {
                    estimated_data.push((time_sec, point.estimated_x));
                }
            } else {
                actual_data.push((time_sec, point.actual_y));
                if point.has_lock {
                    estimated_data.push((time_sec, point.estimated_y));
                }
            }
        }

        // Draw actual position as blue points
        if !actual_data.is_empty() {
            chart
                .draw_series(
                    actual_data
                        .iter()
                        .map(|&(x, y)| Circle::new((x, y), 3, BLUE.filled())),
                )?
                .label("Actual Position")
                .legend(|(x, y)| Circle::new((x + 5, y), 3, BLUE.filled()));
        }

        // Draw estimated position as red points (only after lock)
        if !estimated_data.is_empty() {
            chart
                .draw_series(
                    estimated_data
                        .iter()
                        .map(|&(x, y)| Circle::new((x, y), 3, RED.filled())),
                )?
                .label("Estimated Position")
                .legend(|(x, y)| Circle::new((x + 5, y), 3, RED.filled()));

            // No error lines - just show the points
        }

        // Draw legend
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        Ok(())
    }

    /// Get data bounds for plotting
    fn get_data_bounds(&self, is_x_axis: bool) -> (f64, f64, f64, f64) {
        if self.data_points.is_empty() {
            return (0.0, self.config.max_time_seconds, 0.0, 100.0);
        }

        let time_min = 0.0;
        let time_max = self
            .data_points
            .back()
            .map(|p| p.time.as_secs_f64())
            .unwrap_or(self.config.max_time_seconds)
            .max(1.0);

        let (mut pos_min, mut pos_max) = (f64::INFINITY, f64::NEG_INFINITY);

        for point in &self.data_points {
            let (actual, estimated) = if is_x_axis {
                (point.actual_x, point.estimated_x)
            } else {
                (point.actual_y, point.estimated_y)
            };

            pos_min = pos_min.min(actual).min(estimated);
            pos_max = pos_max.max(actual).max(estimated);
        }

        // Add some margin
        let margin = (pos_max - pos_min) * 0.1;
        (
            time_min,
            time_max * 1.05,
            pos_min - margin,
            pos_max + margin,
        )
    }

    /// Clear all collected data
    pub fn clear(&mut self) {
        self.data_points.clear();
        self.lock_established_time = None;
    }

    /// Get number of collected data points
    pub fn len(&self) -> usize {
        self.data_points.len()
    }

    /// Check if plotter has any data
    pub fn is_empty(&self) -> bool {
        self.data_points.is_empty()
    }
}

/// Helper function to create a simple tracking plot
pub fn plot_tracking_data(
    data_points: Vec<TrackingDataPoint>,
    output_filename: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut plotter = TrackingPlotter::with_config(TrackingPlotConfig {
        output_filename: output_filename
            .as_ref()
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("tracking.png"))
            .to_string_lossy()
            .to_string(),
        ..Default::default()
    });

    for point in data_points {
        plotter.add_point(point);
    }

    plotter.generate_plot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracking_plotter_creation() {
        let plotter = TrackingPlotter::new();
        assert!(plotter.is_empty());
        assert_eq!(plotter.len(), 0);
    }

    #[test]
    fn test_add_tracking_points() {
        let mut plotter = TrackingPlotter::new();

        let point1 = TrackingDataPoint {
            time: Duration::from_secs(1),
            actual_x: 100.0,
            actual_y: 100.0,
            estimated_x: 100.0,
            estimated_y: 100.0,
            is_frame_arrival: true,
            has_lock: false,
        };

        let point2 = TrackingDataPoint {
            time: Duration::from_secs(2),
            actual_x: 101.0,
            actual_y: 100.5,
            estimated_x: 100.8,
            estimated_y: 100.3,
            is_frame_arrival: true,
            has_lock: true,
        };

        plotter.add_point(point1);
        assert_eq!(plotter.len(), 1);

        plotter.add_point(point2);
        assert_eq!(plotter.len(), 2);
        assert!(!plotter.is_empty());
    }

    #[test]
    fn test_data_windowing() {
        let mut plotter = TrackingPlotter::with_config(TrackingPlotConfig {
            max_time_seconds: 5.0,
            ..Default::default()
        });

        // Add points spanning more than the window
        for i in 0..10 {
            plotter.add_point(TrackingDataPoint {
                time: Duration::from_secs(i),
                actual_x: i as f64,
                actual_y: i as f64,
                estimated_x: i as f64,
                estimated_y: i as f64,
                is_frame_arrival: true,
                has_lock: i >= 2,
            });
        }

        // Should only keep points within the 5-second window
        assert!(plotter.len() <= 6); // May be 5 or 6 depending on exact timing
    }
}

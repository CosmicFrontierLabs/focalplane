//! Dark frame analysis for sensor characterization.
//!
//! Provides tools for analyzing dark frames (zero-light exposures) to characterize
//! sensor noise properties including read noise, hot pixels, and statistical outliers.
//!
//! Dark frame analysis captures multiple frames with no light and computes temporal
//! statistics per pixel to identify:
//! - **Hot pixels**: Pixels with abnormally high dark current
//! - **Dead pixels**: Pixels with no temporal variance (stuck/broken)
//! - **Read noise**: The fundamental sensor readout noise (temporal variation)
//! - **Variance outliers**: Pixels with unusual noise characteristics

use image::{Rgb, RgbImage};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Analysis results for a single pixel showing statistical deviation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PixelAnomaly {
    /// Dead pixel: continuously reads zero, no temporal variation
    Dead,
    /// Hot pixel: elevated mean with normal variance (thermal dark current)
    Hot {
        mean: f64,
        std_dev: f64,
        sigma_from_population: f64,
    },
    /// Stuck pixel: frozen at constant high value, no temporal variation
    Stuck {
        mean: f64,
        std_dev: f64,
        sigma_from_population: f64,
    },
}

/// Helper struct for internal anomaly detection methods.
#[derive(Debug, Clone)]
struct PixelInfo {
    pub x: usize,
    pub y: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub sigma_from_population: f64,
}

/// Complete dark frame analysis report with statistical characterization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkFrameReport {
    pub num_frames: usize,
    pub sensor_width: usize,
    pub sensor_height: usize,

    pub global_mean: f64,
    pub global_std_of_means: f64,

    pub median_read_noise: f64,
    pub mean_read_noise: f64,

    /// List of anomalous pixels with their (x, y) coordinates and classification
    pub anomalies: Vec<((usize, usize), PixelAnomaly)>,

    pub temperature_readings: HashMap<String, Vec<f64>>,
}

impl DarkFrameReport {
    /// Generate a human-readable markdown report of sensor characterization.
    ///
    /// Creates a formatted report with executive summary, sensor specifications,
    /// noise characteristics, anomaly statistics, and recommendations.
    pub fn generate_human_report(&self) -> String {
        let mut report = String::new();

        let total_pixels = self.sensor_width * self.sensor_height;
        let num_anomalies = self.anomalies.len();
        let anomaly_percent = (num_anomalies as f64 / total_pixels as f64) * 100.0;

        let mut hot_count = 0;
        let mut stuck_count = 0;
        let mut dead_count = 0;

        for (_, anomaly) in &self.anomalies {
            match anomaly {
                PixelAnomaly::Hot { .. } => hot_count += 1,
                PixelAnomaly::Stuck { .. } => stuck_count += 1,
                PixelAnomaly::Dead => dead_count += 1,
            }
        }

        report.push_str("# Dark Frame Analysis Report\n\n");

        report.push_str("## Executive Summary\n\n");
        report.push_str(&format!(
            "Analyzed {} dark frames from a {}×{} pixel sensor ({} total pixels).\n\n",
            self.num_frames, self.sensor_width, self.sensor_height, total_pixels
        ));
        report.push_str(&format!(
            "{num_anomalies} anomalous pixels detected ({anomaly_percent:.3}% of total)\n\n"
        ));

        report.push_str("---\n\n");
        report.push_str("## Sensor Specifications\n\n");
        report.push_str(&format!(
            "- **Resolution**: {}×{} pixels\n",
            self.sensor_width, self.sensor_height
        ));
        report.push_str(&format!("- **Total Pixels**: {total_pixels}\n"));
        report.push_str(&format!("- **Frames Analyzed**: {}\n\n", self.num_frames));

        report.push_str("---\n\n");
        report.push_str("## Noise Characteristics\n\n");
        report.push_str(&format!(
            "- **Global Mean (Bias Level)**: {:.2} ADU\n",
            self.global_mean
        ));
        report.push_str(&format!(
            "- **Std Dev of Means**: {:.2} ADU\n",
            self.global_std_of_means
        ));
        report.push_str(&format!(
            "- **Median Read Noise**: {:.3} ADU ({:.3} e⁻ at gain=1)\n",
            self.median_read_noise, self.median_read_noise
        ));
        report.push_str(&format!(
            "- **Mean Read Noise**: {:.3} ADU ({:.3} e⁻ at gain=1)\n\n",
            self.mean_read_noise, self.mean_read_noise
        ));

        report.push_str("---\n\n");
        report.push_str("## Pixel Anomalies\n\n");
        report.push_str(&format!(
            "**Total Anomalies**: {num_anomalies} ({anomaly_percent:.3}%)\n\n"
        ));

        report.push_str("### Breakdown by Type\n\n");
        report.push_str("| Anomaly Type | Count | Percentage | Description |\n");
        report.push_str("|--------------|-------|------------|-------------|\n");
        report.push_str(&format!(
            "| Dead Pixels | {} | {:.4}% | Zero signal, no variation |\n",
            dead_count,
            (dead_count as f64 / total_pixels as f64) * 100.0
        ));
        report.push_str(&format!(
            "| Stuck Pixels | {} | {:.4}% | Constant high value, zero variance |\n",
            stuck_count,
            (stuck_count as f64 / total_pixels as f64) * 100.0
        ));
        report.push_str(&format!(
            "| Hot Pixels | {} | {:.4}% | Elevated dark current |\n\n",
            hot_count,
            (hot_count as f64 / total_pixels as f64) * 100.0
        ));

        if !self.temperature_readings.is_empty() {
            report.push_str("---\n\n");
            report.push_str("## Temperature Readings\n\n");

            for (sensor, readings) in &self.temperature_readings {
                if !readings.is_empty() {
                    let mean = readings.iter().sum::<f64>() / readings.len() as f64;
                    let min = readings.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max = readings.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let range = max - min;

                    report.push_str(&format!("### {sensor}\n\n"));
                    report.push_str(&format!("- **Mean**: {mean:.1}°C\n"));
                    report.push_str(&format!("- **Range**: [{min:.1}°C, {max:.1}°C]\n"));
                    report.push_str(&format!("- **Variation**: {range:.2}°C\n\n"));
                }
            }
        }

        report.push_str("---\n\n");
        report.push_str(&format!(
            "*Report generated from {} dark frame exposures*\n",
            self.num_frames
        ));

        report
    }
}

/// Dark frame analysis engine that stores frames and computes statistics on demand.
///
/// Stores all dark frames in memory and computes per-pixel temporal statistics
/// when requested.
pub struct DarkFrameAnalysis {
    width: usize,
    height: usize,
    frames: Vec<Array2<u16>>,
    temperature_readings: HashMap<String, Vec<f64>>,
}

impl DarkFrameAnalysis {
    /// Create a new analysis for a sensor of the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            frames: Vec::new(),
            temperature_readings: HashMap::new(),
        }
    }

    /// Add a dark frame to the analysis.
    ///
    /// Stores the frame in memory for later statistical computation.
    pub fn add_frame(&mut self, frame: &Array2<u16>) {
        assert_eq!(
            frame.shape(),
            &[self.height, self.width],
            "Frame dimensions must match analysis dimensions"
        );

        self.frames.push(frame.clone());
    }

    /// Add temperature readings from frame metadata.
    pub fn add_temperature_readings(&mut self, temps: HashMap<String, f64>) {
        for (key, value) in temps {
            self.temperature_readings
                .entry(key)
                .or_default()
                .push(value);
        }
    }

    /// Finalize the analysis (no-op for compatibility).
    ///
    /// This method exists for API compatibility but does nothing since
    /// statistics are computed on demand.
    pub fn finalize(&mut self) {
        if self.frames.len() < 2 {
            panic!("Need at least 2 frames for variance calculation");
        }
    }

    /// Compute and return the per-pixel mean array.
    pub fn mean(&self) -> Array2<f64> {
        let mut mean = Array2::zeros((self.height, self.width));
        let n = self.frames.len() as f64;

        for frame in &self.frames {
            for ((y, x), &value) in frame.indexed_iter() {
                mean[[y, x]] += value as f64;
            }
        }

        mean.mapv_inplace(|v| v / n);
        mean
    }

    /// Compute and return the per-pixel variance array.
    pub fn variance(&self) -> Array2<f64> {
        assert!(
            self.frames.len() >= 2,
            "Need at least 2 frames for variance calculation"
        );

        let mean = self.mean();
        let mut variance = Array2::zeros((self.height, self.width));
        let n = (self.frames.len() - 1) as f64;

        for frame in &self.frames {
            for ((y, x), &value) in frame.indexed_iter() {
                let diff = value as f64 - mean[[y, x]];
                variance[[y, x]] += diff * diff;
            }
        }

        variance.mapv_inplace(|v| v / n);
        variance
    }

    /// Compute and return the per-pixel standard deviation array.
    pub fn std_dev(&self) -> Array2<f64> {
        self.variance().mapv(f64::sqrt)
    }

    /// Calculate global mean of all pixel means.
    pub fn global_mean(&self) -> f64 {
        self.mean().mean().unwrap()
    }

    /// Calculate global standard deviation of pixel means.
    pub fn global_std_of_means(&self) -> f64 {
        let mean_array = self.mean();
        let mean = mean_array.mean().unwrap();
        let variance = mean_array.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / (mean_array.len() - 1) as f64;
        variance.sqrt()
    }

    /// Estimate read noise as the median of per-pixel standard deviations.
    ///
    /// For short dark exposures where dark current is negligible, the temporal
    /// standard deviation per pixel is dominated by read noise. Using the median
    /// provides a robust estimate resistant to outliers.
    pub fn median_read_noise(&self) -> f64 {
        let std_dev = self.std_dev();
        let mut values: Vec<f64> = std_dev.iter().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values[values.len() / 2]
    }

    /// Estimate read noise as the mean of per-pixel standard deviations.
    pub fn mean_read_noise(&self) -> f64 {
        self.std_dev().mean().unwrap()
    }

    /// Identify hot pixels as those with mean significantly above population.
    ///
    /// Hot pixels have abnormally high dark current, appearing as consistently
    /// bright pixels in dark frames. Detected as pixels with mean > global_mean + threshold*sigma.
    fn hot_pixels(&self, threshold_sigma: f64) -> Vec<PixelInfo> {
        let mean_array = self.mean();
        let global_mean = mean_array.mean().unwrap();
        let global_std = self.global_std_of_means();
        let threshold = global_mean + threshold_sigma * global_std;
        let std_dev = self.std_dev();

        let mut hot_pixels = Vec::new();

        for ((y, x), &mean_val) in mean_array.indexed_iter() {
            if mean_val > threshold {
                hot_pixels.push(PixelInfo {
                    x,
                    y,
                    mean: mean_val,
                    std_dev: std_dev[[y, x]],
                    sigma_from_population: (mean_val - global_mean) / global_std,
                });
            }
        }

        hot_pixels.sort_by(|a, b| {
            b.sigma_from_population
                .partial_cmp(&a.sigma_from_population)
                .unwrap()
        });
        hot_pixels
    }

    /// Identify dead pixels as those with zero or near-zero mean AND variance.
    ///
    /// Dead pixels continuously read zero (or near-zero), showing no variation
    /// across frames and no signal. Distinct from stuck hot pixels which have
    /// high mean but low variance.
    pub fn dead_pixels(&self, variance_threshold: f64) -> Vec<(usize, usize)> {
        let mean_array = self.mean();
        let variance = self.variance();
        let mean_threshold = 10.0;
        let mut dead = Vec::new();

        for ((y, x), &var) in variance.indexed_iter() {
            let mean_val = mean_array[[y, x]];
            if var < variance_threshold && mean_val < mean_threshold {
                dead.push((x, y));
            }
        }

        dead
    }

    /// Identify stuck pixels as those with low variance but elevated mean.
    ///
    /// Stuck/hot pixels are frozen at a constant high value, showing no temporal
    /// variation. These differ from regular hot pixels which may still have normal
    /// read noise variance.
    fn stuck_pixels(&self, variance_threshold: f64, mean_threshold: f64) -> Vec<PixelInfo> {
        let mean_array = self.mean();
        let variance = self.variance();
        let std_dev = self.std_dev();
        let global_mean = mean_array.mean().unwrap();
        let global_std = self.global_std_of_means();

        let mut stuck = Vec::new();

        for ((y, x), &var) in variance.indexed_iter() {
            let mean_val = mean_array[[y, x]];
            if var < variance_threshold && mean_val > mean_threshold {
                stuck.push(PixelInfo {
                    x,
                    y,
                    mean: mean_val,
                    std_dev: std_dev[[y, x]],
                    sigma_from_population: (mean_val - global_mean) / global_std,
                });
            }
        }

        stuck.sort_by(|a, b| {
            b.sigma_from_population
                .partial_cmp(&a.sigma_from_population)
                .unwrap()
        });
        stuck
    }

    /// Generate a comprehensive analysis report.
    pub fn generate_report(
        &self,
        hot_pixel_threshold: f64,
        dead_pixel_variance_threshold: f64,
    ) -> DarkFrameReport {
        let stuck_mean_threshold =
            self.global_mean() + hot_pixel_threshold * self.global_std_of_means();

        let mut anomalies = Vec::new();

        // Add dead pixels
        for (x, y) in self.dead_pixels(dead_pixel_variance_threshold) {
            anomalies.push(((x, y), PixelAnomaly::Dead));
        }

        // Add stuck pixels
        for pixel in self.stuck_pixels(dead_pixel_variance_threshold, stuck_mean_threshold) {
            anomalies.push((
                (pixel.x, pixel.y),
                PixelAnomaly::Stuck {
                    mean: pixel.mean,
                    std_dev: pixel.std_dev,
                    sigma_from_population: pixel.sigma_from_population,
                },
            ));
        }

        // Add hot pixels
        for pixel in self.hot_pixels(hot_pixel_threshold) {
            anomalies.push((
                (pixel.x, pixel.y),
                PixelAnomaly::Hot {
                    mean: pixel.mean,
                    std_dev: pixel.std_dev,
                    sigma_from_population: pixel.sigma_from_population,
                },
            ));
        }

        DarkFrameReport {
            num_frames: self.frames.len(),
            sensor_width: self.width,
            sensor_height: self.height,
            global_mean: self.global_mean(),
            global_std_of_means: self.global_std_of_means(),
            median_read_noise: self.median_read_noise(),
            mean_read_noise: self.mean_read_noise(),
            anomalies,
            temperature_readings: self.temperature_readings.clone(),
        }
    }

    /// Get number of frames analyzed.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Generate visualization with anomalous pixels marked with colored X's.
    ///
    /// Creates an RGB image with the mean dark frame as grayscale background
    /// and colored X marks (5 pixels long, 1 pixel wide) indicating pixel anomalies.
    ///
    /// Colors:
    /// - Red: Stuck pixels (high mean, zero variance)
    /// - Blue: Dead pixels (zero mean, zero variance)
    /// - Orange: Hot pixels (elevated mean with normal variance)
    pub fn visualize_anomalies(&self, report: &DarkFrameReport) -> RgbImage {
        let (height, width) = (self.height, self.width);
        let mut img = RgbImage::new(width as u32, height as u32);
        let mean_array = self.mean();

        let max_val = mean_array.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_val = mean_array.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = (max_val - min_val).max(1.0);

        for y in 0..height {
            for x in 0..width {
                let normalized = ((mean_array[[y, x]] - min_val) / range * 255.0) as u8;
                img.put_pixel(
                    x as u32,
                    y as u32,
                    Rgb([normalized, normalized, normalized]),
                );
            }
        }

        let draw_x = |img: &mut RgbImage, x: usize, y: usize, color: Rgb<u8>| {
            let arm_len = 2i32;
            for offset in -arm_len..=arm_len {
                let x1 = (x as i32 + offset).max(0).min(width as i32 - 1) as u32;
                let y1 = (y as i32 + offset).max(0).min(height as i32 - 1) as u32;
                img.put_pixel(x1, y1, color);

                let x2 = (x as i32 + offset).max(0).min(width as i32 - 1) as u32;
                let y2 = (y as i32 - offset).max(0).min(height as i32 - 1) as u32;
                img.put_pixel(x2, y2, color);
            }
        };

        for ((x, y), anomaly) in &report.anomalies {
            let color = match anomaly {
                PixelAnomaly::Dead => Rgb([0, 0, 255]),         // Blue
                PixelAnomaly::Stuck { .. } => Rgb([255, 0, 0]), // Red
                PixelAnomaly::Hot { .. } => Rgb([255, 128, 0]), // Orange
            };
            draw_x(&mut img, *x, *y, color);
        }

        img
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dark_frame_basic() {
        let mut analysis = DarkFrameAnalysis::new(4, 4);

        let frame1 = Array2::from_elem((4, 4), 100u16);
        let frame2 = Array2::from_elem((4, 4), 102u16);
        let frame3 = Array2::from_elem((4, 4), 98u16);

        analysis.add_frame(&frame1);
        analysis.add_frame(&frame2);
        analysis.add_frame(&frame3);

        analysis.finalize();

        assert_eq!(analysis.num_frames(), 3);
        assert!((analysis.global_mean() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_hot_pixel_detection() {
        let mut analysis = DarkFrameAnalysis::new(4, 4);

        for _ in 0..10 {
            let mut frame = Array2::from_elem((4, 4), 100u16);
            frame[[1, 1]] = 200;
            analysis.add_frame(&frame);
        }

        analysis.finalize();
        let hot_pixels = analysis.hot_pixels(3.0);

        assert!(!hot_pixels.is_empty());
        assert_eq!(hot_pixels[0].x, 1);
        assert_eq!(hot_pixels[0].y, 1);
    }

    #[test]
    fn test_variance_calculation() {
        let mut analysis = DarkFrameAnalysis::new(2, 2);

        let frame1 = Array2::from_shape_vec((2, 2), vec![100, 101, 102, 103]).unwrap();
        let frame2 = Array2::from_shape_vec((2, 2), vec![102, 103, 104, 105]).unwrap();
        let frame3 = Array2::from_shape_vec((2, 2), vec![98, 99, 100, 101]).unwrap();

        analysis.add_frame(&frame1);
        analysis.add_frame(&frame2);
        analysis.add_frame(&frame3);

        analysis.finalize();

        let variance = analysis.variance();
        assert!(variance[[0, 0]] > 0.0);
    }
}

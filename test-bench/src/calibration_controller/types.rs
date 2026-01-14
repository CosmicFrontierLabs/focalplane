//! Data structures for the calibration controller.

use std::collections::VecDeque;

use shared::optical_alignment::OpticalAlignment;

/// A single measurement sample (sensor position + diameter).
#[derive(Debug, Clone, Copy)]
pub struct Measurement {
    pub sensor_x: f64,
    pub sensor_y: f64,
    pub diameter: f64,
}

/// Rolling buffer of measurements for a grid position.
#[derive(Debug, Clone)]
pub struct MeasurementBuffer {
    pub display_x: f64,
    pub display_y: f64,
    pub row: usize,
    pub col: usize,
    samples: VecDeque<Measurement>,
    max_samples: usize,
}

impl MeasurementBuffer {
    pub fn new(display_x: f64, display_y: f64, row: usize, col: usize, max_samples: usize) -> Self {
        Self {
            display_x,
            display_y,
            row,
            col,
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    pub fn push(&mut self, m: Measurement) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(m);
    }

    pub fn average(&self) -> Option<(f64, f64, f64)> {
        if self.samples.is_empty() {
            return None;
        }
        let n = self.samples.len() as f64;
        let (sx, sy, sd) = self
            .samples
            .iter()
            .fold((0.0, 0.0, 0.0), |(ax, ay, ad), m| {
                (ax + m.sensor_x, ay + m.sensor_y, ad + m.diameter)
            });
        Some((sx / n, sy / n, sd / n))
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Calibration measurement at a grid position with focus quality.
#[derive(Debug, Clone)]
pub struct CalibrationPoint {
    /// Grid row index
    pub row: usize,
    /// Grid column index
    pub col: usize,
    /// Display X coordinate
    pub display_x: f64,
    /// Display Y coordinate
    pub display_y: f64,
    /// Sensor X coordinate (averaged)
    pub sensor_x: f64,
    /// Sensor Y coordinate (averaged)
    pub sensor_y: f64,
    /// Average spot diameter (focus quality indicator)
    pub avg_diameter: f64,
}

/// Application state for the TUI.
pub struct App {
    pub points: Vec<CalibrationPoint>,
    pub alignment: Option<OpticalAlignment>,
    pub sensor_width: u32,
    pub sensor_height: u32,
    pub display_width: u32,
    pub display_height: u32,
    pub expected_diameter: f64,
    pub current_position: Option<(usize, usize)>,
    pub current_display_xy: Option<(f64, f64)>,
    pub cycle_count: u32,
    pub tracking_count: usize,
    /// Rolling log buffer for display
    pub logs: VecDeque<String>,
    max_logs: usize,
}

impl App {
    pub fn new(
        sensor_width: u32,
        sensor_height: u32,
        display_width: u32,
        display_height: u32,
        expected_diameter: f64,
    ) -> Self {
        Self {
            points: Vec::new(),
            alignment: None,
            sensor_width,
            sensor_height,
            display_width,
            display_height,
            expected_diameter,
            current_position: None,
            current_display_xy: None,
            cycle_count: 0,
            tracking_count: 0,
            logs: VecDeque::with_capacity(100),
            max_logs: 100,
        }
    }

    pub fn log(&mut self, msg: String) {
        if self.logs.len() >= self.max_logs {
            self.logs.pop_front();
        }
        self.logs.push_back(msg);
    }

    pub fn update(
        &mut self,
        points: Vec<CalibrationPoint>,
        alignment: Option<OpticalAlignment>,
        current_position: Option<(usize, usize)>,
        current_display_xy: Option<(f64, f64)>,
        cycle_count: u32,
        tracking_count: usize,
    ) {
        self.points = points;
        self.alignment = alignment;
        self.current_position = current_position;
        self.current_display_xy = current_display_xy;
        self.cycle_count = cycle_count;
        self.tracking_count = tracking_count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measurement_buffer_average() {
        let mut buffer = MeasurementBuffer::new(100.0, 200.0, 0, 0, 10);

        // Empty buffer returns None
        assert!(buffer.average().is_none());

        // Single measurement
        buffer.push(Measurement {
            sensor_x: 10.0,
            sensor_y: 20.0,
            diameter: 5.0,
        });
        let (x, y, d) = buffer.average().unwrap();
        assert!((x - 10.0).abs() < 1e-10);
        assert!((y - 20.0).abs() < 1e-10);
        assert!((d - 5.0).abs() < 1e-10);

        // Multiple measurements
        buffer.push(Measurement {
            sensor_x: 20.0,
            sensor_y: 40.0,
            diameter: 7.0,
        });
        let (x, y, d) = buffer.average().unwrap();
        assert!((x - 15.0).abs() < 1e-10);
        assert!((y - 30.0).abs() < 1e-10);
        assert!((d - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_measurement_buffer_rolling() {
        let mut buffer = MeasurementBuffer::new(0.0, 0.0, 0, 0, 3);

        buffer.push(Measurement {
            sensor_x: 1.0,
            sensor_y: 0.0,
            diameter: 0.0,
        });
        buffer.push(Measurement {
            sensor_x: 2.0,
            sensor_y: 0.0,
            diameter: 0.0,
        });
        buffer.push(Measurement {
            sensor_x: 3.0,
            sensor_y: 0.0,
            diameter: 0.0,
        });
        assert_eq!(buffer.len(), 3);

        // Adding 4th should drop first
        buffer.push(Measurement {
            sensor_x: 4.0,
            sensor_y: 0.0,
            diameter: 0.0,
        });
        assert_eq!(buffer.len(), 3);
        let (x, _, _) = buffer.average().unwrap();
        assert!((x - 3.0).abs() < 1e-10); // (2+3+4)/3 = 3
    }
}

//! Multiple sensor array configuration for mosaic detector systems.
//!
//! This module provides structures for managing arrays of multiple sensors
//! positioned in a focal plane, as commonly used in large astronomical cameras
//! and survey instruments.

use crate::hardware::sensor::SensorConfig;
use crate::units::{Length, LengthExt};

/// Position of a sensor in the array focal plane.
///
/// The x/y coordinates represent the center of the sensor in the array
/// coordinate system.
#[derive(Debug, Clone)]
pub struct SensorPosition {
    /// X offset of sensor center from array origin in millimeters
    pub x_mm: f64,
    /// Y offset of sensor center from array origin in millimeters  
    pub y_mm: f64,
}

/// A single sensor with its position in the array.
#[derive(Debug, Clone)]
pub struct PositionedSensor {
    /// The sensor configuration
    pub sensor: SensorConfig,
    /// Position in the array coordinate system
    pub position: SensorPosition,
}

/// Array of multiple sensors positioned in a focal plane.
///
/// Represents a mosaic detector system where multiple sensors are arranged
/// to cover a larger field of view. Each sensor has its own position specified
/// in millimeters from the array origin.
///
/// # Coordinate System
/// - Origin (0, 0) is at the array center
/// - X increases to the right
/// - Y increases upward
/// - Positions are in millimeters
#[derive(Debug, Clone)]
pub struct SensorArray {
    /// Collection of sensors with their positions
    pub sensors: Vec<PositionedSensor>,
}

impl SensorArray {
    /// Create a new sensor array from positioned sensors.
    pub fn new(sensors: Vec<PositionedSensor>) -> Self {
        Self { sensors }
    }

    /// Create a sensor array with a single sensor at the origin.
    pub fn single(sensor: SensorConfig) -> Self {
        Self {
            sensors: vec![PositionedSensor {
                sensor,
                position: SensorPosition {
                    x_mm: 0.0,
                    y_mm: 0.0,
                },
            }],
        }
    }

    /// Get the AABB of a specific sensor in array space (millimeters).
    ///
    /// Returns the axis-aligned bounding box of the sensor at the given index,
    /// with coordinates in the array coordinate system (millimeters).
    pub fn sensor_aabb_mm(&self, index: usize) -> Option<(f64, f64, f64, f64)> {
        self.sensors.get(index).map(|ps| {
            let (width_um, height_um) = ps.sensor.dimensions_um();
            let width_mm = Length::from_micrometers(width_um).as_millimeters();
            let height_mm = Length::from_micrometers(height_um).as_millimeters();

            let min_x = ps.position.x_mm - width_mm / 2.0;
            let max_x = ps.position.x_mm + width_mm / 2.0;
            let min_y = ps.position.y_mm - height_mm / 2.0;
            let max_y = ps.position.y_mm + height_mm / 2.0;

            (min_x, min_y, max_x, max_y)
        })
    }

    /// Get AABBs for all sensors in array space (millimeters).
    ///
    /// Returns a vector of axis-aligned bounding boxes, one for each sensor,
    /// with coordinates in the array coordinate system (millimeters).
    pub fn all_sensor_aabbs_mm(&self) -> Vec<(f64, f64, f64, f64)> {
        self.sensors
            .iter()
            .enumerate()
            .filter_map(|(i, _)| self.sensor_aabb_mm(i))
            .collect()
    }

    /// Get the combined AABB encompassing all sensors (millimeters).
    ///
    /// Returns the minimal axis-aligned bounding box that contains all sensors
    /// in the array, with coordinates in millimeters.
    pub fn total_aabb_mm(&self) -> Option<(f64, f64, f64, f64)> {
        if self.sensors.is_empty() {
            return None;
        }

        let aabbs = self.all_sensor_aabbs_mm();
        if aabbs.is_empty() {
            return None;
        }

        let min_x = aabbs.iter().map(|a| a.0).fold(f64::INFINITY, f64::min);
        let min_y = aabbs.iter().map(|a| a.1).fold(f64::INFINITY, f64::min);
        let max_x = aabbs.iter().map(|a| a.2).fold(f64::NEG_INFINITY, f64::max);
        let max_y = aabbs.iter().map(|a| a.3).fold(f64::NEG_INFINITY, f64::max);

        Some((min_x, min_y, max_x, max_y))
    }

    /// Get the total number of pixels across all sensors.
    pub fn total_pixel_count(&self) -> usize {
        self.sensors
            .iter()
            .map(|ps| ps.sensor.width_px * ps.sensor.height_px)
            .sum()
    }

    /// Get the number of sensors in the array.
    pub fn sensor_count(&self) -> usize {
        self.sensors.len()
    }

    /// Convert a point from array space (mm) to a specific sensor's pixel space.
    ///
    /// Returns Some((pixel_x, pixel_y, sensor_index)) if the point falls on a sensor,
    /// or None if the point is not on any sensor.
    pub fn mm_to_pixel(&self, x_mm: f64, y_mm: f64) -> Option<(f64, f64, usize)> {
        for (index, ps) in self.sensors.iter().enumerate() {
            let (width_um, height_um) = ps.sensor.dimensions_um();
            let width_mm = Length::from_micrometers(width_um).as_millimeters();
            let height_mm = Length::from_micrometers(height_um).as_millimeters();

            // Check if point is within this sensor's bounds
            let rel_x = x_mm - ps.position.x_mm;
            let rel_y = y_mm - ps.position.y_mm;

            if rel_x.abs() <= width_mm / 2.0 && rel_y.abs() <= height_mm / 2.0 {
                // Convert to pixel coordinates (0,0 at top-left of sensor)
                let pixel_x = (rel_x + width_mm / 2.0) / ps.sensor.pixel_size.as_millimeters();
                let pixel_y = (height_mm / 2.0 - rel_y) / ps.sensor.pixel_size.as_millimeters();

                return Some((pixel_x, pixel_y, index));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::sensor::models::GSENSE4040BSI;

    #[test]
    fn test_single_sensor_array() {
        let sensor = GSENSE4040BSI.clone();
        let array = SensorArray::single(sensor.clone());

        assert_eq!(array.sensor_count(), 1);
        assert_eq!(
            array.total_pixel_count(),
            sensor.width_px * sensor.height_px
        );

        // Check AABB for single sensor at origin
        let aabb = array.sensor_aabb_mm(0).unwrap();
        let (width_um, height_um) = sensor.dimensions_um();
        let half_width_mm = width_um / 2000.0;
        let half_height_mm = height_um / 2000.0;

        assert!((aabb.0 + half_width_mm).abs() < 1e-6);
        assert!((aabb.1 + half_height_mm).abs() < 1e-6);
        assert!((aabb.2 - half_width_mm).abs() < 1e-6);
        assert!((aabb.3 - half_height_mm).abs() < 1e-6);
    }

    #[test]
    fn test_two_sensor_array() {
        let sensor = GSENSE4040BSI.clone();
        let array = SensorArray::new(vec![
            PositionedSensor {
                sensor: sensor.clone(),
                position: SensorPosition {
                    x_mm: -20.0,
                    y_mm: 0.0,
                },
            },
            PositionedSensor {
                sensor: sensor.clone(),
                position: SensorPosition {
                    x_mm: 20.0,
                    y_mm: 0.0,
                },
            },
        ]);

        assert_eq!(array.sensor_count(), 2);
        assert_eq!(
            array.total_pixel_count(),
            2 * sensor.width_px * sensor.height_px
        );

        let aabbs = array.all_sensor_aabbs_mm();
        assert_eq!(aabbs.len(), 2);

        // Check total AABB spans both sensors
        let total = array.total_aabb_mm().unwrap();
        assert!(total.0 < -19.0); // Left edge
        assert!(total.2 > 19.0); // Right edge
    }

    #[test]
    fn test_mm_to_pixel_conversion() {
        let sensor = GSENSE4040BSI.clone();
        let array = SensorArray::single(sensor.clone());

        // Point at origin should map to center of sensor
        let result = array.mm_to_pixel(0.0, 0.0);
        assert!(result.is_some());

        let (px, py, idx) = result.unwrap();
        assert_eq!(idx, 0);
        assert!((px - sensor.width_px as f64 / 2.0).abs() < 1.0);
        assert!((py - sensor.height_px as f64 / 2.0).abs() < 1.0);

        // Point far outside should return None
        let result = array.mm_to_pixel(1000.0, 1000.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_quad_sensor_array() {
        let sensor = GSENSE4040BSI.clone();
        let offset = 20.0; // mm

        let array = SensorArray::new(vec![
            PositionedSensor {
                sensor: sensor.clone(),
                position: SensorPosition {
                    x_mm: -offset,
                    y_mm: offset,
                },
            },
            PositionedSensor {
                sensor: sensor.clone(),
                position: SensorPosition {
                    x_mm: offset,
                    y_mm: offset,
                },
            },
            PositionedSensor {
                sensor: sensor.clone(),
                position: SensorPosition {
                    x_mm: -offset,
                    y_mm: -offset,
                },
            },
            PositionedSensor {
                sensor: sensor.clone(),
                position: SensorPosition {
                    x_mm: offset,
                    y_mm: -offset,
                },
            },
        ]);

        assert_eq!(array.sensor_count(), 4);

        // Check that each quadrant maps to correct sensor
        assert_eq!(array.mm_to_pixel(-offset, offset).unwrap().2, 0); // Top-left
        assert_eq!(array.mm_to_pixel(offset, offset).unwrap().2, 1); // Top-right
        assert_eq!(array.mm_to_pixel(-offset, -offset).unwrap().2, 2); // Bottom-left
        assert_eq!(array.mm_to_pixel(offset, -offset).unwrap().2, 3); // Bottom-right
    }
}

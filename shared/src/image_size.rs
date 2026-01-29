//! Image dimensions and size utilities

use serde::{Deserialize, Serialize};
use std::fmt;

/// Image dimensions structure
///
/// Represents the width and height of an image sensor or frame.
/// Uses usize for direct compatibility with ndarray indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PixelShape {
    /// Image width in pixels
    pub width: usize,
    /// Image height in pixels
    pub height: usize,
}

impl PixelShape {
    /// Create a new PixelShape
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Create a new PixelShape (alias for new)
    pub fn with_width_height(width: usize, height: usize) -> Self {
        Self::new(width, height)
    }

    /// Get total number of pixels
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Convert to tuple (width, height)
    pub fn to_tuple(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Create from tuple (width, height)
    pub fn from_tuple(dimensions: (usize, usize)) -> Self {
        Self {
            width: dimensions.0,
            height: dimensions.1,
        }
    }

    /// Get center point as (x, y) float coordinates
    pub fn center(&self) -> (f64, f64) {
        (self.width as f64 / 2.0, self.height as f64 / 2.0)
    }

    /// Check if a point is within bounds
    pub fn contains(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    /// Check if a float point is within bounds
    pub fn contains_f64(&self, x: f64, y: f64) -> bool {
        x >= 0.0 && y >= 0.0 && x < self.width as f64 && y < self.height as f64
    }

    /// Convert to u32 tuple for image/display APIs
    pub fn to_u32_tuple(&self) -> (u32, u32) {
        (self.width as u32, self.height as u32)
    }
}

impl From<(usize, usize)> for PixelShape {
    fn from(dimensions: (usize, usize)) -> Self {
        Self::from_tuple(dimensions)
    }
}

impl From<PixelShape> for (usize, usize) {
    fn from(size: PixelShape) -> Self {
        size.to_tuple()
    }
}

impl From<(u32, u32)> for PixelShape {
    fn from(dimensions: (u32, u32)) -> Self {
        Self::new(dimensions.0 as usize, dimensions.1 as usize)
    }
}

impl From<PixelShape> for (u32, u32) {
    fn from(size: PixelShape) -> Self {
        size.to_u32_tuple()
    }
}

impl Default for PixelShape {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl fmt::Display for PixelShape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let size = PixelShape::new(1920, 1080);
        assert_eq!(size.width, 1920);
        assert_eq!(size.height, 1080);
    }

    #[test]
    fn test_with_width_height() {
        let shape = PixelShape::with_width_height(640, 480);
        assert_eq!(shape.width, 640);
        assert_eq!(shape.height, 480);
    }

    #[test]
    fn test_pixel_count() {
        let size = PixelShape::new(1920, 1080);
        assert_eq!(size.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_pixel_count_zero() {
        let shape = PixelShape::new(0, 100);
        assert_eq!(shape.pixel_count(), 0);
    }

    #[test]
    fn test_to_tuple() {
        let size = PixelShape::new(800, 600);
        assert_eq!(size.to_tuple(), (800, 600));
    }

    #[test]
    fn test_from_tuple() {
        let size = PixelShape::from_tuple((1024, 768));
        assert_eq!(size.width, 1024);
        assert_eq!(size.height, 768);
    }

    #[test]
    fn test_from_tuple_roundtrip() {
        let original = PixelShape::new(512, 384);
        let tuple = original.to_tuple();
        let recovered = PixelShape::from_tuple(tuple);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_center() {
        let size = PixelShape::new(100, 200);
        assert_eq!(size.center(), (50.0, 100.0));
    }

    #[test]
    fn test_contains() {
        let size = PixelShape::new(100, 100);
        assert!(size.contains(0, 0));
        assert!(size.contains(99, 99));
        assert!(!size.contains(100, 0));
        assert!(!size.contains(0, 100));
    }

    #[test]
    fn test_contains_f64() {
        let size = PixelShape::new(100, 100);
        assert!(size.contains_f64(0.0, 0.0));
        assert!(size.contains_f64(99.9, 99.9));
        assert!(!size.contains_f64(100.0, 0.0));
        assert!(!size.contains_f64(-0.1, 0.0));
    }

    #[test]
    fn test_from_impl_usize() {
        let size: PixelShape = (320usize, 240usize).into();
        assert_eq!(size.width, 320);
        assert_eq!(size.height, 240);
    }

    #[test]
    fn test_into_tuple_impl() {
        let size = PixelShape::new(160, 120);
        let tuple: (usize, usize) = size.into();
        assert_eq!(tuple, (160, 120));
    }

    #[test]
    fn test_from_u32_tuple() {
        let size: PixelShape = (640u32, 480u32).into();
        assert_eq!(size.width, 640);
        assert_eq!(size.height, 480);
    }

    #[test]
    fn test_to_u32_tuple() {
        let size = PixelShape::new(1920, 1080);
        assert_eq!(size.to_u32_tuple(), (1920u32, 1080u32));
    }

    #[test]
    fn test_display() {
        let size = PixelShape::new(2560, 2560);
        assert_eq!(format!("{}", size), "2560x2560");
    }

    #[test]
    fn test_debug() {
        let shape = PixelShape::new(64, 64);
        let debug_str = format!("{:?}", shape);
        assert!(debug_str.contains("64"));
        assert!(debug_str.contains("PixelShape"));
    }

    #[test]
    fn test_clone_and_copy() {
        let shape = PixelShape::new(100, 100);
        let cloned = shape.clone();
        let copied = shape; // Copy trait
        assert_eq!(shape, cloned);
        assert_eq!(shape, copied);
    }

    #[test]
    fn test_equality() {
        let a = PixelShape::new(50, 60);
        let b = PixelShape::new(50, 60);
        let c = PixelShape::new(60, 50);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PixelShape::new(100, 200));
        set.insert(PixelShape::new(100, 200)); // duplicate
        set.insert(PixelShape::new(200, 100)); // different
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_serde_roundtrip() {
        let original = PixelShape::new(1920, 1080);
        let json = serde_json::to_string(&original).unwrap();
        let recovered: PixelShape = serde_json::from_str(&json).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_with_width_height_alias() {
        let size1 = PixelShape::new(100, 200);
        let size2 = PixelShape::with_width_height(100, 200);
        assert_eq!(size1, size2);
    }
}

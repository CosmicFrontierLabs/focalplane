//! Image dimensions and size utilities

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Image dimensions structure
///
/// Represents the width and height of an image sensor or frame.
/// Provides convenience methods for creating arrays and calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PixelShape {
    /// Image width in pixels
    pub width: usize,
    /// Image height in pixels
    pub height: usize,
}

impl PixelShape {
    /// Create a new PixelShape
    pub fn with_width_height(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Create an empty array with this size
    ///
    /// Returns an ndarray Array2 of zeros with shape (height, width).
    /// Note the row-major ordering convention: rows (height) come first.
    pub fn empty_array<T>(&self) -> Array2<T>
    where
        T: ndarray::NdFloat + Default,
    {
        Array2::default((self.height, self.width))
    }

    /// Create an empty u16 array with this size
    ///
    /// Specialized version for u16 which is the most common camera data type.
    pub fn empty_array_u16(&self) -> Array2<u16> {
        Array2::zeros((self.height, self.width))
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

impl fmt::Display for PixelShape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_width_height() {
        let shape = PixelShape::with_width_height(640, 480);
        assert_eq!(shape.width, 640);
        assert_eq!(shape.height, 480);
    }

    #[test]
    fn test_empty_array_f64() {
        let shape = PixelShape::with_width_height(100, 50);
        let arr: Array2<f64> = shape.empty_array();
        assert_eq!(arr.shape(), &[50, 100]); // (height, width) row-major
        assert!(arr.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_empty_array_f32() {
        let shape = PixelShape::with_width_height(32, 64);
        let arr: Array2<f32> = shape.empty_array();
        assert_eq!(arr.shape(), &[64, 32]); // (height, width)
        assert!(arr.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_empty_array_u16() {
        let shape = PixelShape::with_width_height(256, 128);
        let arr = shape.empty_array_u16();
        assert_eq!(arr.shape(), &[128, 256]); // (height, width)
        assert!(arr.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_pixel_count() {
        let shape = PixelShape::with_width_height(1920, 1080);
        assert_eq!(shape.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_pixel_count_zero() {
        let shape = PixelShape::with_width_height(0, 100);
        assert_eq!(shape.pixel_count(), 0);
    }

    #[test]
    fn test_to_tuple() {
        let shape = PixelShape::with_width_height(800, 600);
        assert_eq!(shape.to_tuple(), (800, 600));
    }

    #[test]
    fn test_from_tuple() {
        let shape = PixelShape::from_tuple((1024, 768));
        assert_eq!(shape.width, 1024);
        assert_eq!(shape.height, 768);
    }

    #[test]
    fn test_from_tuple_roundtrip() {
        let original = PixelShape::with_width_height(512, 384);
        let tuple = original.to_tuple();
        let recovered = PixelShape::from_tuple(tuple);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_from_impl() {
        let shape: PixelShape = (320, 240).into();
        assert_eq!(shape.width, 320);
        assert_eq!(shape.height, 240);
    }

    #[test]
    fn test_into_tuple_impl() {
        let shape = PixelShape::with_width_height(160, 120);
        let tuple: (usize, usize) = shape.into();
        assert_eq!(tuple, (160, 120));
    }

    #[test]
    fn test_display() {
        let shape = PixelShape::with_width_height(1280, 720);
        assert_eq!(format!("{}", shape), "1280x720");
    }

    #[test]
    fn test_debug() {
        let shape = PixelShape::with_width_height(64, 64);
        let debug_str = format!("{:?}", shape);
        assert!(debug_str.contains("64"));
        assert!(debug_str.contains("PixelShape"));
    }

    #[test]
    fn test_clone_and_copy() {
        let shape = PixelShape::with_width_height(100, 100);
        let cloned = shape.clone();
        let copied = shape; // Copy trait
        assert_eq!(shape, cloned);
        assert_eq!(shape, copied);
    }

    #[test]
    fn test_equality() {
        let a = PixelShape::with_width_height(50, 60);
        let b = PixelShape::with_width_height(50, 60);
        let c = PixelShape::with_width_height(60, 50);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PixelShape::with_width_height(100, 200));
        set.insert(PixelShape::with_width_height(100, 200)); // duplicate
        set.insert(PixelShape::with_width_height(200, 100)); // different
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_serde_roundtrip() {
        let original = PixelShape::with_width_height(1920, 1080);
        let json = serde_json::to_string(&original).unwrap();
        let recovered: PixelShape = serde_json::from_str(&json).unwrap();
        assert_eq!(original, recovered);
    }
}

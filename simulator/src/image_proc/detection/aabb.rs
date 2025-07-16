//! Axis-Aligned Bounding Box implementation for astronomical object detection.
//!
//! This module provides efficient data structures and operations for working with
//! axis-aligned bounding boxes (AABBs) in 2D image space. Essential for star detection,
//! galaxy identification, and other astronomical object detection pipelines.
//!
//! # Key Features
//!
//! - **Efficient overlap detection**: Fast algorithms for merging overlapping detections
//! - **Padding support**: Configurable padding for grouping nearby objects
//! - **Geometric operations**: Area, containment, center calculations
//! - **Batch processing**: Union and merge operations for multiple bounding boxes
//! - **Conversion utilities**: Seamless conversion between AABB and tuple formats
//!
//! # Common Applications
//!
//! - **Star detection**: Bounding boxes around detected stellar objects
//! - **Galaxy detection**: Region identification for extended astronomical sources
//! - **Artifact removal**: Merging detections of the same object across multiple thresholds
//! - **Region of interest**: Defining sub-regions for detailed analysis
//! - **Quality filtering**: Size and shape filtering based on bounding box properties
//!
//! # Usage
//!
//! Create bounding boxes for astronomical object detection, merge overlapping
//! detections, and calculate geometric properties for spatial analysis.

/// Axis-Aligned Bounding Box for 2D image regions.
///
/// Represents a rectangular region in image coordinates (row, column) using
/// inclusive bounds. Commonly used in astronomical object detection to define
/// regions containing stars, galaxies, or other celestial objects.
///
/// # Coordinate System
/// - **Rows (y-axis)**: Increase downward from top of image
/// - **Columns (x-axis)**: Increase rightward from left of image
/// - **Bounds**: Both min and max coordinates are inclusive
///
/// # Memory Layout
/// Optimized for fast overlap tests and minimal memory usage (32 bytes on 64-bit systems).
///
/// # Usage
/// Represents rectangular regions in image coordinates for star detection
/// and object tracking with inclusive bounds and optimized memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AABB {
    /// Minimum row (y) coordinate (inclusive)
    pub min_row: usize,
    /// Minimum column (x) coordinate (inclusive)
    pub min_col: usize,
    /// Maximum row (y) coordinate (inclusive)
    pub max_row: usize,
    /// Maximum column (x) coordinate (inclusive)
    pub max_col: usize,
}

impl AABB {
    /// Create a new empty AABB with invalid bounds.
    ///
    /// The resulting AABB has min coordinates set to `usize::MAX` and max
    /// coordinates set to 0, making it invalid until points are added via
    /// `expand_to_include()`.
    ///
    /// # Usage
    /// Used as starting point for incrementally building bounding boxes
    /// by adding points with expand_to_include().
    pub fn new() -> Self {
        Self {
            min_row: usize::MAX,
            min_col: usize::MAX,
            max_row: 0,
            max_col: 0,
        }
    }

    /// Create an AABB from explicit coordinate bounds.
    ///
    /// # Arguments
    /// * `min_row` - Top edge (minimum y-coordinate, inclusive)
    /// * `min_col` - Left edge (minimum x-coordinate, inclusive)
    /// * `max_row` - Bottom edge (maximum y-coordinate, inclusive)
    /// * `max_col` - Right edge (maximum x-coordinate, inclusive)
    ///
    /// # Usage
    /// Creates AABB with explicit coordinate bounds for astronomical
    /// object detection regions.
    pub fn from_coords(min_row: usize, min_col: usize, max_row: usize, max_col: usize) -> Self {
        Self {
            min_row,
            min_col,
            max_row,
            max_col,
        }
    }

    /// Create an AABB from a coordinate tuple.
    ///
    /// # Arguments
    /// * `coords` - Tuple of (min_row, min_col, max_row, max_col)
    ///
    /// # Usage
    /// Convenient constructor from coordinate tuple format
    /// for interface with external libraries.
    pub fn from_tuple(coords: (usize, usize, usize, usize)) -> Self {
        Self {
            min_row: coords.0,
            min_col: coords.1,
            max_row: coords.2,
            max_col: coords.3,
        }
    }

    /// Convert AABB to a coordinate tuple.
    ///
    /// # Returns
    /// Tuple of (min_row, min_col, max_row, max_col)
    ///
    /// Useful for interfacing with external libraries or serialization.
    pub fn to_tuple(&self) -> (usize, usize, usize, usize) {
        (self.min_row, self.min_col, self.max_row, self.max_col)
    }

    /// Check if this AABB overlaps with another AABB.
    ///
    /// Returns true if the bounding boxes share any pixels (including edge contact).
    /// Uses efficient separating axis theorem for fast overlap detection.
    ///
    /// # Arguments
    /// * `other` - The other bounding box to test against
    ///
    /// # Returns
    /// `true` if any part of the AABBs overlap, `false` otherwise
    ///
    /// # Usage
    /// Fast overlap detection for star detection merging
    /// and duplicate removal algorithms.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.min_row <= other.max_row
            && self.max_row >= other.min_row
            && self.min_col <= other.max_col
            && self.max_col >= other.min_col
    }

    /// Check if this AABB overlaps with another when expanded by padding.
    ///
    /// Temporarily expands this AABB by the specified padding amount in all
    /// directions, then checks for overlap with the other AABB. Useful for
    /// grouping nearby detections that should be considered the same object.
    ///
    /// # Arguments
    /// * `other` - The other bounding box to test against
    /// * `padding` - Number of pixels to expand this AABB in all directions
    ///
    /// # Returns
    /// `true` if the padded AABB overlaps with the other AABB
    ///
    /// # Usage
    /// Groups nearby detections that should be considered the same object
    /// by expanding bounds temporarily for overlap testing.
    pub fn overlaps_with_padding(&self, other: &Self, padding: usize) -> bool {
        // Apply padding for overlap check
        let min_row = self.min_row.saturating_sub(padding);
        let min_col = self.min_col.saturating_sub(padding);
        let max_row = self.max_row + padding;
        let max_col = self.max_col + padding;

        // Check if the padded box overlaps with the other
        min_row <= other.max_row
            && max_row >= other.min_row
            && min_col <= other.max_col
            && max_col >= other.min_col
    }

    /// Merge this AABB with another, creating the smallest AABB containing both.
    ///
    /// Creates a new AABB with bounds that encompass both input AABBs.
    /// The result is the smallest axis-aligned rectangle that contains
    /// all pixels from both original boxes.
    ///
    /// # Arguments
    /// * `other` - The other AABB to merge with
    ///
    /// # Returns
    /// New AABB containing both input AABBs
    ///
    /// # Usage
    /// Combines overlapping detections into single bounding box
    /// for duplicate removal and object consolidation.
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            min_row: self.min_row.min(other.min_row),
            min_col: self.min_col.min(other.min_col),
            max_row: self.max_row.max(other.max_row),
            max_col: self.max_col.max(other.max_col),
        }
    }

    /// Expand this AABB to include the specified point.
    ///
    /// Modifies the AABB bounds to ensure the given point is contained within
    /// the bounding box. If the point is already contained, no change occurs.
    ///
    /// # Arguments
    /// * `row` - Y-coordinate of the point to include
    /// * `col` - X-coordinate of the point to include
    ///
    /// # Usage
    /// Incrementally builds bounding box by adding detection points,
    /// useful for growing regions during object detection.
    pub fn expand_to_include(&mut self, row: usize, col: usize) {
        self.min_row = self.min_row.min(row);
        self.min_col = self.min_col.min(col);
        self.max_row = self.max_row.max(row);
        self.max_col = self.max_col.max(col);
    }

    /// Calculate the width of the AABB in pixels.
    ///
    /// # Returns
    /// Width as number of pixels (max_col - min_col + 1)
    ///
    /// # Usage
    /// Calculates horizontal extent of detection region
    /// for size filtering and analysis.
    pub fn width(&self) -> usize {
        self.max_col - self.min_col + 1
    }

    /// Calculate the height of the AABB in pixels.
    ///
    /// # Returns
    /// Height as number of pixels (max_row - min_row + 1)
    ///
    /// # Usage
    /// Calculates vertical extent of detection region
    /// for size filtering and analysis.
    pub fn height(&self) -> usize {
        self.max_row - self.min_row + 1
    }

    /// Calculate the area of the AABB in square pixels.
    ///
    /// # Returns
    /// Area as number of pixels (width × height)
    ///
    /// Useful for filtering detections by size or estimating object extent.
    ///
    /// # Usage
    /// Computes total pixel area for object size filtering
    /// and detection quality assessment.
    pub fn area(&self) -> usize {
        self.width() * self.height()
    }

    /// Check if the AABB has valid bounds.
    ///
    /// Returns false for empty AABBs created with `new()` or AABBs where
    /// min coordinates are greater than max coordinates.
    ///
    /// # Returns
    /// `true` if the AABB represents a valid region, `false` otherwise
    ///
    /// # Usage
    /// Validates bounding box before geometric calculations
    /// and prevents errors from malformed regions.
    pub fn is_valid(&self) -> bool {
        self.min_row <= self.max_row && self.min_col <= self.max_col
    }

    /// Create a new AABB expanded by padding in all directions.
    ///
    /// Returns a new AABB that is larger than the current one by the specified
    /// padding amount. Uses saturating subtraction to prevent underflow.
    ///
    /// # Arguments
    /// * `padding` - Number of pixels to expand in all directions
    ///
    /// # Returns
    /// New padded AABB
    ///
    /// # Usage
    /// Creates expanded region for grouping nearby objects
    /// or adding safety margins around detections.
    pub fn with_padding(&self, padding: usize) -> Self {
        Self {
            min_row: self.min_row.saturating_sub(padding),
            min_col: self.min_col.saturating_sub(padding),
            max_row: self.max_row + padding,
            max_col: self.max_col + padding,
        }
    }

    /// Check if this AABB contains the specified point.
    ///
    /// Tests whether the point (row, col) lies within the AABB bounds,
    /// including on the boundary (inclusive bounds).
    ///
    /// # Arguments
    /// * `row` - Y-coordinate to test
    /// * `col` - X-coordinate to test
    ///
    /// # Returns
    /// `true` if the point is within the AABB bounds
    ///
    /// # Usage
    /// Tests pixel membership for region-based analysis
    /// and spatial filtering operations.
    pub fn contains_point(&self, row: usize, col: usize) -> bool {
        row >= self.min_row && row <= self.max_row && col >= self.min_col && col <= self.max_col
    }

    /// Check if this AABB completely contains another AABB.
    ///
    /// Returns true if all points of the other AABB lie within this AABB's bounds.
    /// The contained AABB may touch the boundary of the containing AABB.
    ///
    /// # Arguments
    /// * `other` - The AABB to test for containment
    ///
    /// # Returns
    /// `true` if this AABB completely contains the other AABB
    ///
    /// # Usage
    /// Determines hierarchical relationships between detection regions
    /// for nested object analysis.
    pub fn contains(&self, other: &Self) -> bool {
        self.min_row <= other.min_row
            && self.max_row >= other.max_row
            && self.min_col <= other.min_col
            && self.max_col >= other.max_col
    }

    /// Calculate the center point of the AABB.
    ///
    /// Returns the center as floating-point coordinates to handle AABBs
    /// with even dimensions accurately.
    ///
    /// # Returns
    /// Tuple of (x_center, y_center) in image coordinates
    ///
    /// # Usage
    /// Calculates geometric center for object positioning
    /// and centroid-based analysis.
    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_col as f64 + self.max_col as f64) / 2.0,
            (self.min_row as f64 + self.max_row as f64) / 2.0,
        )
    }
}

impl Default for AABB {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the union (bounding box) of multiple AABBs.
///
/// Creates the smallest AABB that contains all input AABBs. Useful for
/// finding the overall extent of multiple detections or creating summary
/// regions for astronomical object catalogs.
///
/// # Arguments
/// * `boxes` - Slice of AABBs to compute union for
///
/// # Returns
/// * `Some(AABB)` - Union bounding box if input is non-empty
/// * `None` - If input slice is empty
///
/// # Usage
/// Finds overall field extent for multiple astronomical detections
/// or creates summary regions for object catalogs.
pub fn union_aabbs(boxes: &[AABB]) -> Option<AABB> {
    if boxes.is_empty() {
        return None;
    }

    let mut result = boxes[0];
    for bbox in boxes.iter().skip(1) {
        result = result.merge(bbox);
    }

    Some(result)
}

/// Convert coordinate tuples to AABB objects.
///
/// Batch conversion utility for interfacing with external detection algorithms
/// or deserializing bounding box data.
///
/// # Arguments
/// * `bboxes` - Slice of coordinate tuples (min_row, min_col, max_row, max_col)
///
/// # Returns
/// Vector of AABB objects
///
/// # Usage
/// Batch conversion utility for interfacing with external detection
/// algorithms or deserializing bounding box data.
pub fn tuples_to_aabbs(bboxes: &[(usize, usize, usize, usize)]) -> Vec<AABB> {
    bboxes.iter().map(|&bbox| AABB::from_tuple(bbox)).collect()
}

/// Convert AABB objects to coordinate tuples.
///
/// Batch conversion utility for serialization or interfacing with external
/// libraries that expect tuple-based bounding box formats.
///
/// # Arguments
/// * `boxes` - Slice of AABB objects
///
/// # Returns
/// Vector of coordinate tuples (min_row, min_col, max_row, max_col)
///
/// # Usage
/// Batch conversion utility for serialization or interfacing with
/// external libraries expecting tuple-based formats.
pub fn aabbs_to_tuples(boxes: &[AABB]) -> Vec<(usize, usize, usize, usize)> {
    boxes.iter().map(|bbox| bbox.to_tuple()).collect()
}

/// Merge overlapping AABBs to eliminate duplicate detections.
///
/// Combines AABBs that overlap (with optional padding) into larger boxes that
/// encompass all the original overlapping regions. Essential for cleaning up
/// detection results where the same astronomical object may be detected multiple
/// times at different thresholds or scales.
///
/// # Algorithm
/// Uses a greedy approach: for each unprocessed AABB, finds all overlapping AABBs
/// and merges them recursively until no more overlaps are found. This ensures
/// transitively overlapping boxes are properly combined.
///
/// # Arguments
/// * `boxes` - Slice of AABBs to merge (input detections)
/// * `padding` - Optional padding pixels for overlap testing (helps group nearby objects)
///
/// # Returns
/// Vector of merged AABBs with no overlaps between them
///
/// # Performance
/// Time complexity: O(n²) in worst case, O(n) for well-separated objects
///
/// # Usage
/// Eliminates duplicate detections by combining overlapping regions.
/// Essential for cleaning detection results where the same object
/// may be detected multiple times at different thresholds.
pub fn merge_overlapping_aabbs(boxes: &[AABB], padding: Option<usize>) -> Vec<AABB> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let padding = padding.unwrap_or(0);

    // Make a copy of the input boxes
    let boxes_copy = boxes.to_vec();

    // Track which boxes have been merged
    let mut merged = vec![false; boxes_copy.len()];
    let mut result = Vec::new();

    for i in 0..boxes_copy.len() {
        // Skip if this box was already merged
        if merged[i] {
            continue;
        }

        // Start with the current box
        let mut current_box = boxes_copy[i];
        merged[i] = true;

        // Flag to track if any merge happened in this iteration
        let mut merge_happened = true;

        // Keep merging boxes until no more overlaps are found
        while merge_happened {
            merge_happened = false;

            for j in 0..boxes_copy.len() {
                // Skip if box already merged or is the current box
                if merged[j] || i == j {
                    continue;
                }

                // Check for overlap with padding
                if current_box.overlaps_with_padding(&boxes_copy[j], padding) {
                    // Merge the boxes
                    current_box = current_box.merge(&boxes_copy[j]);
                    merged[j] = true;
                    merge_happened = true;
                }
            }
        }

        // Add the merged box to the result
        result.push(current_box);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_creation() {
        // Create an empty AABB
        let aabb = AABB::new();
        assert_eq!(aabb.min_row, usize::MAX);
        assert_eq!(aabb.min_col, usize::MAX);
        assert_eq!(aabb.max_row, 0);
        assert_eq!(aabb.max_col, 0);

        // Create from coordinates
        let aabb = AABB::from_coords(10, 20, 30, 40);
        assert_eq!(aabb.min_row, 10);
        assert_eq!(aabb.min_col, 20);
        assert_eq!(aabb.max_row, 30);
        assert_eq!(aabb.max_col, 40);

        // Create from tuple
        let aabb = AABB::from_tuple((10, 20, 30, 40));
        assert_eq!(aabb.min_row, 10);
        assert_eq!(aabb.min_col, 20);
        assert_eq!(aabb.max_row, 30);
        assert_eq!(aabb.max_col, 40);

        // Convert to tuple
        assert_eq!(aabb.to_tuple(), (10, 20, 30, 40));

        // Default implementation
        let default_aabb: AABB = Default::default();
        assert_eq!(default_aabb.min_row, usize::MAX);
    }

    #[test]
    fn test_aabb_expand() {
        let mut aabb = AABB::new();

        // Expand to include a point
        aabb.expand_to_include(10, 20);
        assert_eq!(aabb.min_row, 10);
        assert_eq!(aabb.min_col, 20);
        assert_eq!(aabb.max_row, 10);
        assert_eq!(aabb.max_col, 20);

        // Expand to include another point
        aabb.expand_to_include(5, 30);
        assert_eq!(aabb.min_row, 5);
        assert_eq!(aabb.min_col, 20);
        assert_eq!(aabb.max_row, 10);
        assert_eq!(aabb.max_col, 30);
    }

    #[test]
    fn test_aabb_overlap() {
        // Two overlapping boxes
        let aabb1 = AABB::from_coords(10, 10, 20, 20);
        let aabb2 = AABB::from_coords(15, 15, 25, 25);
        assert!(aabb1.overlaps(&aabb2));
        assert!(aabb2.overlaps(&aabb1));

        // Non-overlapping boxes
        let aabb3 = AABB::from_coords(30, 30, 40, 40);
        assert!(!aabb1.overlaps(&aabb3));
        assert!(!aabb3.overlaps(&aabb1));

        // Almost overlapping boxes with padding
        let aabb4 = AABB::from_coords(22, 22, 30, 30);
        assert!(!aabb1.overlaps(&aabb4));
        assert!(aabb1.overlaps_with_padding(&aabb4, 2)); // With 2 pixel padding
    }

    #[test]
    fn test_aabb_merge() {
        let aabb1 = AABB::from_coords(10, 10, 20, 20);
        let aabb2 = AABB::from_coords(15, 15, 25, 25);

        // Merge two boxes
        let merged = aabb1.merge(&aabb2);
        assert_eq!(merged.min_row, 10);
        assert_eq!(merged.min_col, 10);
        assert_eq!(merged.max_row, 25);
        assert_eq!(merged.max_col, 25);
    }

    #[test]
    fn test_aabb_dimensions() {
        let aabb = AABB::from_coords(10, 20, 29, 49);

        // Test dimensions
        assert_eq!(aabb.width(), 30); // 49 - 20 + 1
        assert_eq!(aabb.height(), 20); // 29 - 10 + 1
        assert_eq!(aabb.area(), 600); // 30 * 20
    }

    #[test]
    fn test_aabb_validity() {
        // Valid AABB
        let valid = AABB::from_coords(10, 20, 30, 40);
        assert!(valid.is_valid());

        // Invalid AABB (empty)
        let invalid = AABB::new();
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_aabb_with_padding() {
        let aabb = AABB::from_coords(10, 20, 30, 40);
        let padded = aabb.with_padding(5);

        assert_eq!(padded.min_row, 5);
        assert_eq!(padded.min_col, 15);
        assert_eq!(padded.max_row, 35);
        assert_eq!(padded.max_col, 45);
    }

    #[test]
    fn test_aabb_contains() {
        let aabb = AABB::from_coords(10, 20, 30, 40);

        // Test point containment
        assert!(aabb.contains_point(10, 20)); // Min corner
        assert!(aabb.contains_point(30, 40)); // Max corner
        assert!(aabb.contains_point(20, 30)); // Middle
        assert!(!aabb.contains_point(5, 5)); // Outside

        // Test AABB containment
        let inner = AABB::from_coords(15, 25, 25, 35);
        assert!(aabb.contains(&inner));

        let outer = AABB::from_coords(5, 15, 35, 45);
        assert!(!aabb.contains(&outer));
        assert!(outer.contains(&aabb));

        let overlap = AABB::from_coords(15, 15, 35, 35);
        assert!(!aabb.contains(&overlap));
        assert!(!overlap.contains(&aabb));
    }

    #[test]
    fn test_aabb_center() {
        let aabb = AABB::from_coords(10, 20, 30, 40);
        let center = aabb.center();

        assert_eq!(center.0, 30.0); // X center = (20 + 40) / 2
        assert_eq!(center.1, 20.0); // Y center = (10 + 30) / 2
    }

    #[test]
    fn test_union_aabbs() {
        let boxes = vec![
            AABB::from_coords(10, 10, 20, 20),
            AABB::from_coords(15, 15, 25, 25),
            AABB::from_coords(30, 30, 40, 40),
        ];

        let union = union_aabbs(&boxes).unwrap();
        assert_eq!(union.min_row, 10);
        assert_eq!(union.min_col, 10);
        assert_eq!(union.max_row, 40);
        assert_eq!(union.max_col, 40);

        // Test empty case
        let empty: Vec<AABB> = vec![];
        assert!(union_aabbs(&empty).is_none());
    }

    #[test]
    fn test_merge_overlapping_aabbs() {
        // Create some test boxes
        let boxes = vec![
            AABB::from_coords(10, 10, 20, 20),
            AABB::from_coords(15, 15, 25, 25),
            AABB::from_coords(50, 50, 60, 60),
        ];

        // Merge overlapping boxes with no padding
        let merged = merge_overlapping_aabbs(&boxes, None);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].to_tuple(), (10, 10, 25, 25)); // First two boxes merged
        assert_eq!(merged[1].to_tuple(), (50, 50, 60, 60)); // Third box unchanged

        // Merge with padding
        let boxes = vec![
            AABB::from_coords(10, 10, 20, 20),
            AABB::from_coords(25, 25, 35, 35), // Not overlapping, but close enough with padding
            AABB::from_coords(50, 50, 60, 60),
        ];

        let merged = merge_overlapping_aabbs(&boxes, Some(5));
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].to_tuple(), (10, 10, 35, 35)); // First two boxes merged
        assert_eq!(merged[1].to_tuple(), (50, 50, 60, 60)); // Third box unchanged
    }

    #[test]
    fn test_tuple_conversions() {
        // Create test tuple boxes
        let tuples = vec![(10, 10, 20, 20), (30, 30, 40, 40)];

        // Convert to AABBs
        let aabbs = tuples_to_aabbs(&tuples);
        assert_eq!(aabbs.len(), 2);
        assert_eq!(aabbs[0].min_row, 10);
        assert_eq!(aabbs[0].max_col, 20);
        assert_eq!(aabbs[1].min_col, 30);
        assert_eq!(aabbs[1].max_row, 40);

        // Convert back to tuples
        let tuples_back = aabbs_to_tuples(&aabbs);
        assert_eq!(tuples_back, tuples);

        // Round-trip conversion
        assert_eq!(aabbs_to_tuples(&tuples_to_aabbs(&tuples)), tuples);
    }
}

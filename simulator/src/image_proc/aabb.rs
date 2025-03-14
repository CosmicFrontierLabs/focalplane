//! Axis-Aligned Bounding Box implementation
//!
//! This module provides a data structure and operations for working with
//! axis-aligned bounding boxes in 2D space, commonly used for object detection
//! and image processing tasks.

/// Axis-Aligned Bounding Box (AABB) representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AABB {
    /// Minimum row (y) coordinate
    pub min_row: usize,
    /// Minimum column (x) coordinate
    pub min_col: usize,
    /// Maximum row (y) coordinate
    pub max_row: usize,
    /// Maximum column (x) coordinate
    pub max_col: usize,
}

impl AABB {
    /// Create a new empty AABB
    pub fn new() -> Self {
        Self {
            min_row: usize::MAX,
            min_col: usize::MAX,
            max_row: 0,
            max_col: 0,
        }
    }

    /// Create an AABB from coordinates
    pub fn from_coords(min_row: usize, min_col: usize, max_row: usize, max_col: usize) -> Self {
        Self {
            min_row,
            min_col,
            max_row,
            max_col,
        }
    }

    /// Create an AABB from a tuple of coordinates (min_row, min_col, max_row, max_col)
    pub fn from_tuple(coords: (usize, usize, usize, usize)) -> Self {
        Self {
            min_row: coords.0,
            min_col: coords.1,
            max_row: coords.2,
            max_col: coords.3,
        }
    }

    /// Convert AABB to a tuple (min_row, min_col, max_row, max_col)
    pub fn to_tuple(&self) -> (usize, usize, usize, usize) {
        (self.min_row, self.min_col, self.max_row, self.max_col)
    }

    /// Check if this AABB overlaps with another
    pub fn overlaps(&self, other: &Self) -> bool {
        self.min_row <= other.max_row
            && self.max_row >= other.min_row
            && self.min_col <= other.max_col
            && self.max_col >= other.min_col
    }

    /// Check if this AABB overlaps with another considering padding
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

    /// Merge this AABB with another, creating a new AABB that contains both
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            min_row: self.min_row.min(other.min_row),
            min_col: self.min_col.min(other.min_col),
            max_row: self.max_row.max(other.max_row),
            max_col: self.max_col.max(other.max_col),
        }
    }

    /// Expand this AABB to include the given point
    pub fn expand_to_include(&mut self, row: usize, col: usize) {
        self.min_row = self.min_row.min(row);
        self.min_col = self.min_col.min(col);
        self.max_row = self.max_row.max(row);
        self.max_col = self.max_col.max(col);
    }

    /// Calculate the width of the AABB
    pub fn width(&self) -> usize {
        self.max_col - self.min_col + 1
    }

    /// Calculate the height of the AABB
    pub fn height(&self) -> usize {
        self.max_row - self.min_row + 1
    }

    /// Calculate the area of the AABB
    pub fn area(&self) -> usize {
        self.width() * self.height()
    }

    /// Check if the AABB is valid (not empty)
    pub fn is_valid(&self) -> bool {
        self.min_row <= self.max_row && self.min_col <= self.max_col
    }

    /// Create a new AABB that is padded by the given amount in all directions
    pub fn with_padding(&self, padding: usize) -> Self {
        Self {
            min_row: self.min_row.saturating_sub(padding),
            min_col: self.min_col.saturating_sub(padding),
            max_row: self.max_row + padding,
            max_col: self.max_col + padding,
        }
    }

    /// Check if this AABB contains the given point
    pub fn contains_point(&self, row: usize, col: usize) -> bool {
        row >= self.min_row && row <= self.max_row && col >= self.min_col && col <= self.max_col
    }

    /// Check if this AABB completely contains another AABB
    pub fn contains(&self, other: &Self) -> bool {
        self.min_row <= other.min_row
            && self.max_row >= other.max_row
            && self.min_col <= other.min_col
            && self.max_col >= other.max_col
    }

    /// Calculate the center point of the AABB
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

/// Find the union of multiple AABBs
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

/// Convert a slice of tuples (min_row, min_col, max_row, max_col) to AABB objects
pub fn tuples_to_aabbs(bboxes: &[(usize, usize, usize, usize)]) -> Vec<AABB> {
    bboxes.iter().map(|&bbox| AABB::from_tuple(bbox)).collect()
}

/// Convert a slice of AABB objects to tuples (min_row, min_col, max_row, max_col)
pub fn aabbs_to_tuples(boxes: &[AABB]) -> Vec<(usize, usize, usize, usize)> {
    boxes.iter().map(|bbox| bbox.to_tuple()).collect()
}

/// Merge overlapping AABBs
///
/// This function combines AABBs that overlap into larger boxes that encompass all
/// the original overlapping regions.
///
/// # Arguments
/// * `boxes` - Slice of AABBs to merge
/// * `padding` - Optional padding to add around each box when checking for overlap
///
/// # Returns
/// * Vector of merged AABBs
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

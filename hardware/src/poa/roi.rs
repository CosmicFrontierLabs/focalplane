use playerone_sdk::ROI;
use shared::image_proc::detection::aabb::AABB;

/// Compare two ROI structs for equality
pub fn roi_eq(a: &ROI, b: &ROI) -> bool {
    a.start_x == b.start_x && a.start_y == b.start_y && a.width == b.width && a.height == b.height
}

/// Convert AABB to PlayerOne SDK ROI format
pub fn aabb_to_roi(aabb: &AABB) -> ROI {
    let width = (aabb.max_col - aabb.min_col + 1) as u32;
    let height = (aabb.max_row - aabb.min_row + 1) as u32;
    ROI {
        start_x: aabb.min_col as u32,
        start_y: aabb.min_row as u32,
        width,
        height,
    }
}

/// Convert PlayerOne SDK ROI to AABB format
pub fn roi_to_aabb(roi: &ROI) -> AABB {
    AABB {
        min_col: roi.start_x as usize,
        min_row: roi.start_y as usize,
        max_col: (roi.start_x + roi.width - 1) as usize,
        max_row: (roi.start_y + roi.height - 1) as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roi_eq() {
        let roi1 = ROI {
            start_x: 10,
            start_y: 20,
            width: 100,
            height: 200,
        };
        let roi2 = ROI {
            start_x: 10,
            start_y: 20,
            width: 100,
            height: 200,
        };
        let roi3 = ROI {
            start_x: 10,
            start_y: 20,
            width: 100,
            height: 201,
        };

        assert!(roi_eq(&roi1, &roi2));
        assert!(!roi_eq(&roi1, &roi3));
    }

    #[test]
    fn test_aabb_to_roi() {
        let aabb = AABB {
            min_col: 100,
            min_row: 200,
            max_col: 299,
            max_row: 399,
        };

        let roi = aabb_to_roi(&aabb);
        let expected = ROI {
            start_x: 100,
            start_y: 200,
            width: 200,
            height: 200,
        };

        assert!(roi_eq(&roi, &expected));
    }

    #[test]
    fn test_roi_to_aabb() {
        let roi = ROI {
            start_x: 50,
            start_y: 75,
            width: 640,
            height: 480,
        };

        let aabb = roi_to_aabb(&roi);

        assert_eq!(aabb.min_col, 50);
        assert_eq!(aabb.min_row, 75);
        assert_eq!(aabb.max_col, 689);
        assert_eq!(aabb.max_row, 554);
    }

    #[test]
    fn test_roundtrip_aabb_roi_aabb() {
        let original = AABB {
            min_col: 10,
            min_row: 20,
            max_col: 109,
            max_row: 79,
        };

        let roi = aabb_to_roi(&original);
        let recovered = roi_to_aabb(&roi);

        assert_eq!(original, recovered);
    }

    #[test]
    fn test_roundtrip_roi_aabb_roi() {
        let original = ROI {
            start_x: 32,
            start_y: 64,
            width: 128,
            height: 256,
        };

        let aabb = roi_to_aabb(&original);
        let recovered = aabb_to_roi(&aabb);

        assert!(roi_eq(&original, &recovered));
    }
}

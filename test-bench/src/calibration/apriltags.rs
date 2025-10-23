use anyhow::Result;
use apriltag::{Detection, Detector, Family};
use image::GrayImage;
use std::collections::HashMap;

pub fn filter_apriltags_by_size(
    detections: Vec<Detection>,
    min_size_pixels: f64,
) -> Vec<Detection> {
    let original_count = detections.len();
    let filtered: Vec<Detection> = detections
        .into_iter()
        .filter(|det| {
            let corners = det.corners();
            let width = ((corners[1][0] - corners[0][0]).powi(2)
                + (corners[1][1] - corners[0][1]).powi(2))
            .sqrt();
            let height = ((corners[3][0] - corners[0][0]).powi(2)
                + (corners[3][1] - corners[0][1]).powi(2))
            .sqrt();
            let size = width.min(height);
            size >= min_size_pixels
        })
        .collect();

    let filtered_count = filtered.len();
    let removed_count = original_count - filtered_count;

    if removed_count > 0 {
        tracing::debug!(
            "AprilTag filtering: {original_count} detected, {removed_count} removed (< {min_size_pixels:.0}px), {filtered_count} kept"
        );
    } else {
        tracing::debug!(
            "AprilTag filtering: {original_count} detected, {filtered_count} kept (all >= {min_size_pixels:.0}px)"
        );
    }

    filtered
}

pub fn detect_apriltags(img: &GrayImage) -> Result<Vec<Detection>> {
    let mut detector = Detector::builder()
        .add_family_bits(Family::tag_16h5(), 2)
        .build()?;

    let (width, height) = (img.width() as usize, img.height() as usize);
    let stride = width;

    let mut apriltag_img =
        unsafe { apriltag::Image::new_uinit_with_stride(width, height, stride)? };

    let img_data = img.as_raw();
    unsafe {
        std::ptr::copy_nonoverlapping(
            img_data.as_ptr(),
            apriltag_img.as_mut().as_mut_ptr(),
            width * height,
        );
    }

    let detections = detector.detect(&apriltag_img);

    Ok(detections)
}

pub fn build_tag_grid_map(detections: &[Detection]) -> HashMap<usize, (f64, f64)> {
    let mut map = HashMap::new();
    for det in detections {
        let center = det.center();
        map.insert(det.id(), (center[0], center[1]));
    }
    map
}

use anyhow::Result;
use apriltag::Detection;
use ndarray::Array2;
use shared::image_proc::centroid::compute_centroid_from_mask;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PixelMeasurement {
    pub position: (f64, f64),
    pub diameter: f64,
    pub flux: f64,
    pub aspect_ratio: f64,
    pub angle: f64,
}

pub fn measure_psf_pixels(
    frame: &Array2<u16>,
    detections: &[Detection],
) -> Result<Vec<PixelMeasurement>> {
    let mut measurements = Vec::new();

    let detection_map = build_tag_grid_map(detections);

    for row in 0..4 {
        for col in 0..4 {
            let top_left_id = row * 5 + col;
            let top_right_id = row * 5 + col + 1;
            let bottom_left_id = (row + 1) * 5 + col;
            let bottom_right_id = (row + 1) * 5 + col + 1;

            if let (Some(&tl), Some(&tr), Some(&bl), Some(&br)) = (
                detection_map.get(&top_left_id),
                detection_map.get(&top_right_id),
                detection_map.get(&bottom_left_id),
                detection_map.get(&bottom_right_id),
            ) {
                let white_block_x = (tl.0 + tr.0 + bl.0 + br.0) / 4.0;
                let white_block_y = (tl.1 + tr.1 + bl.1 + br.1) / 4.0;

                let tag_spacing = ((tr.0 - tl.0).powi(2) + (tr.1 - tl.1).powi(2)).sqrt();
                let search_radius = tag_spacing * 0.3;

                if let Some(pixel_psf) =
                    analyze_psf_at_position(frame, (white_block_x, white_block_y), search_radius)
                {
                    measurements.push(pixel_psf);
                }
            }
        }
    }

    Ok(measurements)
}

fn build_tag_grid_map(detections: &[Detection]) -> HashMap<usize, (f64, f64)> {
    let mut map = HashMap::new();
    for det in detections {
        let center = det.center();
        map.insert(det.id(), (center[0], center[1]));
    }
    map
}

fn analyze_psf_at_position(
    frame: &Array2<u16>,
    pos: (f64, f64),
    search_radius: f64,
) -> Option<PixelMeasurement> {
    let (height, width) = frame.dim();
    let (cx, cy) = (pos.0 as usize, pos.1 as usize);
    let radius = search_radius as usize;

    if cx < radius || cy < radius || cx >= width - radius || cy >= height - radius {
        return None;
    }

    let x_start = cx.saturating_sub(radius);
    let y_start = cy.saturating_sub(radius);
    let x_end = (cx + radius).min(width);
    let y_end = (cy + radius).min(height);

    let roi_width = x_end - x_start;
    let roi_height = y_end - y_start;

    let mut roi = Array2::zeros((roi_height, roi_width));
    for y in 0..roi_height {
        for x in 0..roi_width {
            roi[[y, x]] = frame[[y_start + y, x_start + x]] as f64;
        }
    }

    let center_row = roi_height as f64 / 2.0;
    let center_col = roi_width as f64 / 2.0;
    let mask_radius = (roi_width.min(roi_height) as f64 * 0.2) / 2.0;

    let mut mask = Array2::from_elem((roi_height, roi_width), false);
    for row in 0..roi_height {
        for col in 0..roi_width {
            let dy = row as f64 - center_row;
            let dx = col as f64 - center_col;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist <= mask_radius {
                mask[[row, col]] = true;
            }
        }
    }

    let result = compute_centroid_from_mask(&roi.view(), &mask.view());

    if result.flux > 0.0 {
        let abs_x = x_start as f64 + result.x;
        let abs_y = y_start as f64 + result.y;
        let angle = 0.5 * result.m_xy.atan2(result.m_xx - result.m_yy);

        Some(PixelMeasurement {
            position: (abs_x, abs_y),
            diameter: result.diameter,
            flux: result.flux,
            aspect_ratio: result.aspect_ratio,
            angle,
        })
    } else {
        None
    }
}

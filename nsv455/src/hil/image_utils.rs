use base64::{engine::general_purpose::STANDARD, Engine as _};
use log::debug;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ImageData {
    pub data: String,      // Base64 encoded raw u16 image data
    pub shape: Vec<usize>, // [height, width]
    pub dtype: String,     // Always "uint16"
}

#[allow(dead_code)]
pub fn raw_image_to_base64_json(frame_data: &[u8], width: u32, height: u32) -> ImageData {
    let encoded = STANDARD.encode(frame_data);

    ImageData {
        data: encoded,
        shape: vec![height as usize, width as usize],
        dtype: "uint16".to_string(),
    }
}

pub fn compute_histogram(pixels: &[u8]) -> Vec<u32> {
    let mut histogram = vec![0u32; 256];
    for &pixel in pixels {
        histogram[pixel as usize] += 1;
    }
    histogram
}

pub fn process_raw_to_pixels(frame_data: &[u8], width: u32, height: u32) -> Vec<u8> {
    // Only the 8096x6324 resolution has 96 pixel padding
    // All other resolutions have different stride patterns
    let stride = if width == 8096 && height == 6324 {
        // Max resolution: 96 pixel padding (192 bytes)
        (width as usize + 96) * 2
    } else {
        // For all other resolutions, calculate stride from actual data size
        frame_data.len() / height as usize
    };

    let stride_pixels = stride / 2;
    let padding_pixels = stride_pixels - width as usize;
    let mut pixels_8bit = Vec::with_capacity((width * height) as usize);

    debug!(
        "Frame {width}x{height}: stride={stride} bytes ({width}px + {padding_pixels}px padding)"
    );

    for y in 0..height {
        let row_start = (y as usize) * stride;
        for x in 0..width {
            let pixel_offset = row_start + (x as usize) * 2;
            if pixel_offset + 1 < frame_data.len() {
                let value =
                    u16::from_le_bytes([frame_data[pixel_offset], frame_data[pixel_offset + 1]]);
                pixels_8bit.push((value >> 8) as u8);
            }
        }
    }
    pixels_8bit
}

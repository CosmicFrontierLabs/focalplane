use crate::display_assets::Assets;
use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb};
use ndarray::{concatenate, Array2, Axis};

const TAG_SIZE: u32 = 8;
const GRID_SIZE: usize = 9;

fn load_apriltag_as_array(tag_id: usize) -> Result<Array2<u8>> {
    let tag_path = format!("apriltags/tag16h5/{tag_id:02}.png");
    let asset_data = Assets::get(&tag_path)
        .with_context(|| format!("Failed to load embedded AprilTag: {tag_path}"))?;
    let tag_img = image::load_from_memory(&asset_data.data)
        .with_context(|| format!("Failed to decode AprilTag image: {tag_path}"))?
        .to_luma8();

    let mut array = Array2::zeros((TAG_SIZE as usize, TAG_SIZE as usize));
    for y in 0..TAG_SIZE as usize {
        for x in 0..TAG_SIZE as usize {
            array[[y, x]] = tag_img.get_pixel(x as u32, y as u32)[0];
        }
    }
    Ok(array)
}

fn create_vertical_bars() -> Array2<u8> {
    let pattern = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ];

    let mut array = Array2::zeros((TAG_SIZE as usize, TAG_SIZE as usize));
    for y in 0..TAG_SIZE as usize {
        for x in 0..TAG_SIZE as usize {
            array[[y, x]] = if pattern[y][x] == 1 { 0 } else { 255 };
        }
    }
    array
}

fn create_horizontal_bars() -> Array2<u8> {
    create_vertical_bars().t().to_owned()
}

fn create_white_block() -> Array2<u8> {
    Array2::from_elem((TAG_SIZE as usize, TAG_SIZE as usize), 255u8)
}

fn scale_array(array: &Array2<u8>, scale: usize) -> Array2<u8> {
    let (h, w) = array.dim();
    let mut scaled = Array2::zeros((h * scale, w * scale));

    for y in 0..h {
        for x in 0..w {
            let value = array[[y, x]];
            for dy in 0..scale {
                for dx in 0..scale {
                    scaled[[y * scale + dy, x * scale + dx]] = value;
                }
            }
        }
    }
    scaled
}

pub fn generate(width: u32, height: u32) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let target_size = width.min(height);
    let target_array_size = (target_size as f32 * 0.9) as u32;
    let cell_size = target_array_size / GRID_SIZE as u32;
    let scale_factor = (cell_size / TAG_SIZE) as usize;
    let actual_cell_size = TAG_SIZE as usize * scale_factor;
    let total_array_size = actual_cell_size * GRID_SIZE;

    println!("  Tag size: {TAG_SIZE}x{TAG_SIZE}");
    println!("  Scale factor: {scale_factor}");
    println!("  Cell size: {actual_cell_size}x{actual_cell_size}");
    println!("  Grid: {GRID_SIZE}x{GRID_SIZE}");
    println!("  Total array size: {total_array_size}x{total_array_size}");

    let tags: Vec<Array2<u8>> = (0..25).map(load_apriltag_as_array).collect::<Result<_>>()?;
    let v_bars = create_vertical_bars();
    let h_bars = create_horizontal_bars();
    let white = create_white_block();

    let mut rows = Vec::new();

    for grid_row in 0..GRID_SIZE {
        let is_tag_row = grid_row % 2 == 0;

        let mut row_blocks = Vec::new();

        for grid_col in 0..GRID_SIZE {
            let is_tag_col = grid_col % 2 == 0;

            let block = if is_tag_row && is_tag_col {
                let tag_row = grid_row / 2;
                let tag_col = grid_col / 2;
                let tag_id = tag_row * 5 + tag_col;
                tags[tag_id].view()
            } else if is_tag_row && !is_tag_col {
                h_bars.view()
            } else if !is_tag_row && is_tag_col {
                v_bars.view()
            } else {
                white.view()
            };

            row_blocks.push(block);
        }

        let row_views: Vec<_> = row_blocks.iter().map(|b| b.view()).collect();
        let row = concatenate(Axis(1), &row_views).context("Failed to concatenate row")?;
        rows.push(row);
    }

    let row_views: Vec<_> = rows.iter().map(|r| r.view()).collect();
    let full_pattern = concatenate(Axis(0), &row_views).context("Failed to concatenate rows")?;

    let mut scaled_pattern = scale_array(&full_pattern, scale_factor);

    for grid_row in 0..GRID_SIZE {
        for grid_col in 0..GRID_SIZE {
            let is_white_block = (grid_row % 2 == 1) && (grid_col % 2 == 1);
            if is_white_block {
                let center_y = (grid_row * actual_cell_size) + (actual_cell_size / 2);
                let center_x = (grid_col * actual_cell_size) + (actual_cell_size / 2);
                scaled_pattern[[center_y, center_x]] = 0;
            }
        }
    }

    let mut img = ImageBuffer::from_pixel(width, height, Rgb([255, 255, 255]));

    let offset_x = (width as usize - total_array_size) / 2;
    let offset_y = (height as usize - total_array_size) / 2;

    for y in 0..total_array_size {
        for x in 0..total_array_size {
            let value = scaled_pattern[[y, x]];
            img.put_pixel(
                (offset_x + x) as u32,
                (offset_y + y) as u32,
                Rgb([value, value, value]),
            );
        }
    }

    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apriltag_pattern_generation() {
        let result = generate(1024, 1024);
        if let Err(e) = &result {
            panic!("AprilTag pattern generation failed: {e}");
        }

        let img = result.unwrap();
        assert_eq!(img.width(), 1024);
        assert_eq!(img.height(), 1024);
    }
}

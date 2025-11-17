use anyhow::{Context, Result};
use image::{imageops::FilterType, RgbImage};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct MotionPoint {
    pub time_s: f64,
    pub x_pixels: f64,
    pub y_pixels: f64,
}

pub fn load_motion_profile(path: &PathBuf) -> Result<Vec<MotionPoint>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open motion profile CSV: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut points = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.context("Failed to read line from CSV")?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line
            .split(&[',', '\t', ' '][..])
            .filter(|s| !s.is_empty())
            .collect();

        if parts.len() < 3 {
            anyhow::bail!(
                "Line {} has {} fields, expected 3 (t, x, y): {}",
                line_num + 1,
                parts.len(),
                line
            );
        }

        let time_s: f64 = parts[0]
            .parse()
            .with_context(|| format!("Failed to parse time on line {}", line_num + 1))?;
        let x_pixels: f64 = parts[1]
            .parse()
            .with_context(|| format!("Failed to parse x on line {}", line_num + 1))?;
        let y_pixels: f64 = parts[2]
            .parse()
            .with_context(|| format!("Failed to parse y on line {}", line_num + 1))?;

        points.push(MotionPoint {
            time_s,
            x_pixels,
            y_pixels,
        });
    }

    if points.is_empty() {
        anyhow::bail!("Motion profile CSV is empty or has no valid data points");
    }

    println!("Loaded {} motion profile points", points.len());
    println!(
        "  Time range: {:.3}s to {:.3}s",
        points[0].time_s,
        points.last().unwrap().time_s
    );

    Ok(points)
}

pub fn load_and_downsample_image(
    path: &PathBuf,
    target_width: u32,
    target_height: u32,
) -> Result<RgbImage> {
    let img = image::open(path)
        .with_context(|| format!("Failed to load image: {}", path.display()))?
        .into_rgb8();

    let original_width = img.width();
    let original_height = img.height();

    if original_width == target_width && original_height == target_height {
        println!("  Image already at target size");
        return Ok(img);
    }

    println!(
        "  Downsampling from {original_width}x{original_height} to {target_width}x{target_height}"
    );

    let downsampled =
        image::imageops::resize(&img, target_width, target_height, FilterType::Lanczos3);

    Ok(downsampled)
}

fn bilinear_sample_wraparound(base_img: &RgbImage, x: f64, y: f64) -> [u8; 3] {
    let img_width = base_img.width() as i32;
    let img_height = base_img.height() as i32;

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let wraparound = |coord: i32, size: i32| -> u32 {
        let mut wrapped = coord % size;
        if wrapped < 0 {
            wrapped += size;
        }
        wrapped as u32
    };

    let x0_safe = wraparound(x0, img_width);
    let x1_safe = wraparound(x1, img_width);
    let y0_safe = wraparound(y0, img_height);
    let y1_safe = wraparound(y1, img_height);

    let p00 = base_img.get_pixel(x0_safe, y0_safe);
    let p10 = base_img.get_pixel(x1_safe, y0_safe);
    let p01 = base_img.get_pixel(x0_safe, y1_safe);
    let p11 = base_img.get_pixel(x1_safe, y1_safe);

    let mut result = [0u8; 3];
    for c in 0..3 {
        let v00 = p00[c] as f64;
        let v10 = p10[c] as f64;
        let v01 = p01[c] as f64;
        let v11 = p11[c] as f64;

        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;
        let v = v0 * (1.0 - fy) + v1 * fy;

        result[c] = v.round().clamp(0.0, 255.0) as u8;
    }
    result
}

pub fn generate_into_buffer(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    base_img: &RgbImage,
    motion_profile: &[MotionPoint],
    elapsed: Duration,
    motion_scale: f64,
) {
    let elapsed_s = elapsed.as_secs_f64();
    let total_duration = motion_profile.last().unwrap().time_s;
    let current_time = elapsed_s % total_duration;

    let mut idx = 0;
    for (i, point) in motion_profile.iter().enumerate() {
        if point.time_s > current_time {
            idx = i.saturating_sub(1);
            break;
        }
    }

    let (x_pixels, y_pixels) = if idx + 1 < motion_profile.len() {
        let p0 = &motion_profile[idx];
        let p1 = &motion_profile[idx + 1];
        let t = (current_time - p0.time_s) / (p1.time_s - p0.time_s);
        let x = p0.x_pixels + t * (p1.x_pixels - p0.x_pixels);
        let y = p0.y_pixels + t * (p1.y_pixels - p0.y_pixels);
        (x * motion_scale, y * motion_scale)
    } else {
        (
            motion_profile[idx].x_pixels * motion_scale,
            motion_profile[idx].y_pixels * motion_scale,
        )
    };

    buffer.fill(0);

    let img_width = base_img.width();
    let img_height = base_img.height();
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    // Parallel rendering with rayon - process rows in parallel
    buffer
        .par_chunks_mut((width * 3) as usize)
        .enumerate()
        .for_each(|(dst_y, row)| {
            for dst_x in 0..width {
                // Map destination pixel back to source image coordinates
                let src_x = dst_x as f64 - center_x - x_pixels + img_width as f64 / 2.0;
                let src_y = dst_y as f64 - center_y - y_pixels + img_height as f64 / 2.0;

                // Sample with wraparound boundaries
                let pixel = bilinear_sample_wraparound(base_img, src_x, src_y);
                let idx = (dst_x * 3) as usize;
                row[idx] = pixel[0];
                row[idx + 1] = pixel[1];
                row[idx + 2] = pixel[2];
            }
        });
}

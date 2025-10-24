use image::{ImageBuffer, Rgb};
use std::time::SystemTime;

pub fn generate_into_buffer(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    sigma: f64,
    wiggle_radius_pixels: f64,
) {
    buffer.fill(0);

    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    let elapsed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    let rotation_period = 10.0;
    let angle = (elapsed % rotation_period) / rotation_period * 2.0 * std::f64::consts::PI;

    let gaussian_x = center_x + wiggle_radius_pixels * angle.cos();
    let gaussian_y = center_y + wiggle_radius_pixels * angle.sin();

    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - gaussian_x;
            let dy = y as f64 - gaussian_y;
            let dist_sq = dx * dx + dy * dy;

            let intensity = (-(dist_sq / (2.0 * sigma * sigma))).exp();
            let pixel_value = (intensity * 255.0).clamp(0.0, 255.0) as u8;

            let offset = ((y * width + x) * 3) as usize;
            buffer[offset] = pixel_value;
            buffer[offset + 1] = pixel_value;
            buffer[offset + 2] = pixel_value;
        }
    }
}

pub fn generate(
    width: u32,
    height: u32,
    sigma: f64,
    wiggle_radius_pixels: f64,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut buffer = vec![0u8; (width * height * 3) as usize];
    generate_into_buffer(&mut buffer, width, height, sigma, wiggle_radius_pixels);
    ImageBuffer::from_raw(width, height, buffer).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wiggling_gaussian_pattern_generation() {
        let img = generate(640, 480, 20.0, 50.0);
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);
    }
}

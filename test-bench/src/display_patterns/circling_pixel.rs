use image::{ImageBuffer, Rgb};
use shared::image_size::PixelShape;
use std::time::SystemTime;

pub fn generate_into_buffer(
    buffer: &mut [u8],
    size: PixelShape,
    orbit_count: u32,
    radius_percent: u32,
) {
    let (width, height) = size.to_u32_tuple();
    buffer.fill(0);

    let center_x = width / 2;
    let center_y = height / 2;

    let elapsed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    let rotation_period = 60.0;
    let base_angle = (elapsed % rotation_period) / rotation_period * 2.0 * std::f64::consts::PI;

    let fov_size = width.min(height) as f64;
    let radius = fov_size * (radius_percent as f64 / 200.0);

    let put_white_pixel = |buffer: &mut [u8], x: u32, y: u32, w: u32, h: u32| {
        if x < w && y < h {
            let offset = ((y * w + x) * 3) as usize;
            buffer[offset] = 255;
            buffer[offset + 1] = 255;
            buffer[offset + 2] = 255;
        }
    };

    put_white_pixel(buffer, center_x, center_y, width, height);

    for i in 0..orbit_count {
        let angle_offset = (i as f64 / orbit_count as f64) * 2.0 * std::f64::consts::PI;
        let angle = base_angle + angle_offset;

        let moving_x = (center_x as f64 + radius * angle.cos()).round() as u32;
        let moving_y = (center_y as f64 + radius * angle.sin()).round() as u32;

        put_white_pixel(buffer, moving_x, moving_y, width, height);
    }
}

pub fn generate(
    size: PixelShape,
    orbit_count: u32,
    radius_percent: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = size.to_u32_tuple();
    let mut buffer = vec![0u8; (width * height * 3) as usize];
    generate_into_buffer(&mut buffer, size, orbit_count, radius_percent);
    ImageBuffer::from_raw(width, height, buffer).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circling_pixel_pattern_generation() {
        let img = generate(PixelShape::new(640, 480), 1, 50);
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);

        let center_pixel = img.get_pixel(320, 240);
        assert_eq!(*center_pixel, Rgb([255, 255, 255]));
    }
}

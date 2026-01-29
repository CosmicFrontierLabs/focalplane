use image::{ImageBuffer, Rgb};
use shared::image_size::PixelShape;
use std::time::SystemTime;

use super::shared::{compute_normalization_factor, render_gaussian_spot, BlendMode};

pub fn generate_into_buffer(
    buffer: &mut [u8],
    size: PixelShape,
    fwhm_pixels: f64,
    wiggle_radius_pixels: f64,
    max_intensity: f64,
) {
    buffer.fill(0);

    let (center_x, center_y) = size.center();

    let elapsed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    let rotation_period = 10.0;
    let angle = (elapsed % rotation_period) / rotation_period * 2.0 * std::f64::consts::PI;

    let gaussian_x = center_x + wiggle_radius_pixels * angle.cos();
    let gaussian_y = center_y + wiggle_radius_pixels * angle.sin();

    let normalization_factor = compute_normalization_factor(fwhm_pixels, max_intensity);

    render_gaussian_spot(
        buffer,
        size,
        gaussian_x,
        gaussian_y,
        fwhm_pixels,
        normalization_factor,
        BlendMode::Overwrite,
    );
}

pub fn generate(
    size: PixelShape,
    fwhm_pixels: f64,
    wiggle_radius_pixels: f64,
    max_intensity: f64,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = size.to_u32_tuple();
    let mut buffer = vec![0u8; (width * height * 3) as usize];
    generate_into_buffer(
        &mut buffer,
        size,
        fwhm_pixels,
        wiggle_radius_pixels,
        max_intensity,
    );
    ImageBuffer::from_raw(width, height, buffer).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wiggling_gaussian_pattern_generation() {
        let img = generate(PixelShape::new(640, 480), 47.0, 50.0, 255.0);
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);
    }
}

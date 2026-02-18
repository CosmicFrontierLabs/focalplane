use image::{ImageBuffer, Rgb};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use shared::image_size::PixelShape;
use std::time::SystemTime;

/// Generate a random pixel pattern into an existing buffer.
///
/// Picks a single pixel at a uniform random position, holding it for
/// `dwell_secs` seconds before jumping to a new random position. The
/// position is derived deterministically from the current dwell period
/// so it stays stable between frames within the same period.
pub fn generate_into_buffer(buffer: &mut [u8], size: PixelShape, intensity: u8, dwell_secs: f64) {
    let (width, height) = size.to_u32_tuple();
    buffer.fill(0);

    let elapsed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // Which dwell period are we in? Use this as a seed so position
    // is stable within each period but changes between periods.
    let period_index = (elapsed / dwell_secs) as u64;
    let mut rng = StdRng::seed_from_u64(period_index);

    let x = rng.random_range(0..width);
    let y = rng.random_range(0..height);

    let offset = ((y * width + x) * 3) as usize;
    buffer[offset] = intensity;
    buffer[offset + 1] = intensity;
    buffer[offset + 2] = intensity;
}

pub fn generate(size: PixelShape, intensity: u8, dwell_secs: f64) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = size.to_u32_tuple();
    let mut buffer = vec![0u8; (width * height * 3) as usize];
    generate_into_buffer(&mut buffer, size, intensity, dwell_secs);
    ImageBuffer::from_raw(width, height, buffer).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_pixel_has_exactly_one_lit_pixel() {
        let img = generate(PixelShape::new(100, 100), 255, 30.0);
        let lit: usize = img
            .pixels()
            .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
            .count();
        assert_eq!(lit, 1);
    }

    #[test]
    fn test_random_pixel_intensity() {
        let img = generate(PixelShape::new(100, 100), 128, 30.0);
        let bright = img.pixels().find(|p| p[0] > 0).unwrap();
        assert_eq!(*bright, Rgb([128, 128, 128]));
    }
}

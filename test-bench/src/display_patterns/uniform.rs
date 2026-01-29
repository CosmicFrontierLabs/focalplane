use image::{ImageBuffer, Rgb};
use shared::image_size::PixelShape;

pub fn generate(size: PixelShape, level: u8) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = size.to_u32_tuple();
    ImageBuffer::from_pixel(width, height, Rgb([level, level, level]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_pattern_generation() {
        let img = generate(PixelShape::new(100, 100), 128);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);

        let pixel = img.get_pixel(50, 50);
        assert_eq!(*pixel, Rgb([128, 128, 128]));

        let corner_pixel = img.get_pixel(0, 0);
        assert_eq!(*corner_pixel, Rgb([128, 128, 128]));
    }

    #[test]
    fn test_uniform_black() {
        let img = generate(PixelShape::new(100, 100), 0);
        let pixel = img.get_pixel(50, 50);
        assert_eq!(*pixel, Rgb([0, 0, 0]));
    }

    #[test]
    fn test_uniform_white() {
        let img = generate(PixelShape::new(100, 100), 255);
        let pixel = img.get_pixel(50, 50);
        assert_eq!(*pixel, Rgb([255, 255, 255]));
    }
}

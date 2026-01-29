use image::{ImageBuffer, Rgb};
use shared::image_size::PixelShape;

pub fn generate(size: PixelShape, checker_size: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = size.to_u32_tuple();
    let mut img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let checker_x = x / checker_size;
            let checker_y = y / checker_size;
            let is_white = (checker_x + checker_y) % 2 == 0;

            let color = if is_white {
                Rgb([255, 255, 255])
            } else {
                Rgb([0, 0, 0])
            };

            img.put_pixel(x, y, color);
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkerboard_pattern_generation() {
        let img = generate(PixelShape::new(800, 600), 50);
        assert_eq!(img.width(), 800);
        assert_eq!(img.height(), 600);

        let corner_pixel = img.get_pixel(0, 0);
        assert_eq!(*corner_pixel, Rgb([255, 255, 255]));
    }
}

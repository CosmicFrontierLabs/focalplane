use image::{ImageBuffer, Rgb};

pub fn generate(width: u32, height: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));
    let center_x = width / 2;
    let center_y = height / 2;
    img.put_pixel(center_x, center_y, Rgb([255, 255, 255]));
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_pattern_generation() {
        let img = generate(100, 100);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);

        let center_pixel = img.get_pixel(50, 50);
        assert_eq!(*center_pixel, Rgb([255, 255, 255]));

        let other_pixel = img.get_pixel(0, 0);
        assert_eq!(*other_pixel, Rgb([0, 0, 0]));
    }
}

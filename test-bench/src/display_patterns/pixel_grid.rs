use image::{ImageBuffer, Rgb};

pub fn generate(width: u32, height: u32, spacing: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));

    if spacing == 0 {
        return img;
    }

    for y in (0..height).step_by(spacing as usize) {
        for x in (0..width).step_by(spacing as usize) {
            img.put_pixel(x, y, Rgb([255, 255, 255]));
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_grid_pattern_generation() {
        let img = generate(640, 480, 10);
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);

        assert_eq!(*img.get_pixel(0, 0), Rgb([255, 255, 255]));
        assert_eq!(*img.get_pixel(10, 0), Rgb([255, 255, 255]));
        assert_eq!(*img.get_pixel(0, 10), Rgb([255, 255, 255]));
        assert_eq!(*img.get_pixel(5, 5), Rgb([0, 0, 0]));
    }
}

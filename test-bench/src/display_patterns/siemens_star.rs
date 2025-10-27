use image::{ImageBuffer, Rgb};

pub fn generate(width: u32, height: u32, num_spokes: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));

    if num_spokes == 0 {
        return img;
    }

    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - center_x;
            let dy = y as f64 - center_y;

            let angle = dy.atan2(dx);
            let normalized_angle = (angle + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);

            let spoke_index = (normalized_angle * num_spokes as f64).floor() as u32;

            if spoke_index % 2 == 0 {
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_siemens_star_pattern_generation() {
        let img = generate(640, 480, 16);
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);

        assert_eq!(*img.get_pixel(320, 240), Rgb([255, 255, 255]));
    }

    #[test]
    fn test_siemens_star_zero_spokes() {
        let img = generate(100, 100, 0);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);
        assert_eq!(*img.get_pixel(50, 50), Rgb([0, 0, 0]));
    }
}

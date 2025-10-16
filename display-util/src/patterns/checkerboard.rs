use image::{ImageBuffer, Rgb};

pub fn generate(width: u32, height: u32, checker_size: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
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

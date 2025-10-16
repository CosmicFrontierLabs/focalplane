use image::{ImageBuffer, Rgb};

pub fn generate(width: u32, height: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));
    let center_x = width / 2;
    let center_y = height / 2;
    img.put_pixel(center_x, center_y, Rgb([255, 255, 255]));
    img
}

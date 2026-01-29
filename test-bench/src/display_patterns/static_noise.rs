use image::{ImageBuffer, Rgb};
use rand::Rng;
use shared::image_size::PixelShape;

pub fn generate_into_buffer(buffer: &mut [u8], size: PixelShape, block_size: u32) {
    let (width, height) = size.to_u32_tuple();
    let mut rng = rand::rng();
    let blocks_x = width.div_ceil(block_size);
    let blocks_y = height.div_ceil(block_size);

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let value: u8 = rng.random();

            let start_x = block_x * block_size;
            let start_y = block_y * block_size;
            let end_x = (start_x + block_size).min(width);
            let end_y = (start_y + block_size).min(height);

            for y in start_y..end_y {
                let row_offset = (y * width * 3) as usize;
                for x in start_x..end_x {
                    let pixel_offset = row_offset + (x * 3) as usize;
                    buffer[pixel_offset] = value;
                    buffer[pixel_offset + 1] = value;
                    buffer[pixel_offset + 2] = value;
                }
            }
        }
    }
}

pub fn generate(size: PixelShape, block_size: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = size.to_u32_tuple();
    let mut buffer = vec![0u8; (width * height * 3) as usize];
    generate_into_buffer(&mut buffer, size, block_size);
    ImageBuffer::from_raw(width, height, buffer).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_noise_pattern_generation() {
        let img = generate(PixelShape::new(640, 480), 10);
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);
    }
}

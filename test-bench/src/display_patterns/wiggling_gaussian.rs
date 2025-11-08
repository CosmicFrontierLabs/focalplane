use image::{ImageBuffer, Rgb};
use shared::image_proc::airy::PixelScaledAiryDisk;
use shared::units::{LengthExt, Wavelength};
use std::time::SystemTime;

pub fn generate_into_buffer(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    sigma: f64,
    wiggle_radius_pixels: f64,
    max_intensity: f64,
) {
    buffer.fill(0);

    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    let elapsed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    let rotation_period = 10.0;
    let angle = (elapsed % rotation_period) / rotation_period * 2.0 * std::f64::consts::PI;

    let gaussian_x = center_x + wiggle_radius_pixels * angle.cos();
    let gaussian_y = center_y + wiggle_radius_pixels * angle.sin();

    let reference_wavelength = Wavelength::from_nanometers(550.0);
    let psf = PixelScaledAiryDisk::with_fwhm(sigma * 2.355, reference_wavelength);

    let cutoff_radius = psf.first_zero();
    let x_min = (gaussian_x - cutoff_radius).max(0.0) as u32;
    let x_max = (gaussian_x + cutoff_radius).min(width as f64) as u32;
    let y_min = (gaussian_y - cutoff_radius).max(0.0) as u32;
    let y_max = (gaussian_y + cutoff_radius).min(height as f64) as u32;

    for y in y_min..y_max {
        for x in x_min..x_max {
            let pixel_x = x as f64 - gaussian_x;
            let pixel_y = y as f64 - gaussian_y;

            let pixel_flux = psf.pixel_flux_simpson(pixel_x, pixel_y, max_intensity);
            let pixel_value = pixel_flux.clamp(0.0, 255.0) as u8;

            let offset = ((y * width + x) * 3) as usize;
            buffer[offset] = pixel_value;
            buffer[offset + 1] = pixel_value;
            buffer[offset + 2] = pixel_value;
        }
    }
}

pub fn generate(
    width: u32,
    height: u32,
    sigma: f64,
    wiggle_radius_pixels: f64,
    max_intensity: f64,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut buffer = vec![0u8; (width * height * 3) as usize];
    generate_into_buffer(
        &mut buffer,
        width,
        height,
        sigma,
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
        let img = generate(640, 480, 20.0, 50.0, 255.0);
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);
    }
}

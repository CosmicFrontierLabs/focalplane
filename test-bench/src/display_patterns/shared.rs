//! Shared utilities for display pattern generation.

use shared::image_proc::airy::PixelScaledAiryDisk;
use shared::image_size::PixelShape;
use shared::units::{LengthExt, Wavelength};

/// Reference wavelength for PSF calculations (550nm green light).
pub const REFERENCE_WAVELENGTH_NM: f64 = 550.0;

/// Compute normalization factor for a given FWHM to achieve target max intensity.
///
/// This renders a test PSF and finds the max pixel value, then returns
/// the scaling factor needed to hit the target intensity.
pub fn compute_normalization_factor(fwhm_pixels: f64, target_max_intensity: f64) -> f64 {
    let reference_wavelength = Wavelength::from_nanometers(REFERENCE_WAVELENGTH_NM);
    let psf = PixelScaledAiryDisk::with_fwhm(fwhm_pixels, reference_wavelength);

    let test_size = 64;
    let center = test_size as f64 / 2.0;

    let mut max_pixel_value: f64 = 0.0;
    for y in 0..test_size {
        for x in 0..test_size {
            let pixel_x = x as f64 - center;
            let pixel_y = y as f64 - center;
            let pixel_flux = psf.pixel_flux_simpson(pixel_x, pixel_y, 1.0);
            max_pixel_value = max_pixel_value.max(pixel_flux);
        }
    }

    if max_pixel_value > 0.0 {
        target_max_intensity / max_pixel_value
    } else {
        1.0
    }
}

/// Blending mode for rendering spots.
#[derive(Debug, Clone, Copy, Default)]
pub enum BlendMode {
    /// Overwrite existing pixel values
    #[default]
    Overwrite,
    /// Add to existing values with saturation
    Additive,
}

/// Render a gaussian-like spot (Airy disk PSF) into an RGB buffer.
///
/// # Arguments
/// * `buffer` - RGB24 buffer (size.width * size.height * 3 bytes)
/// * `size` - Buffer dimensions in pixels
/// * `spot_x` - Spot center X coordinate
/// * `spot_y` - Spot center Y coordinate
/// * `fwhm_pixels` - Full width at half maximum in pixels
/// * `normalization_factor` - Intensity scaling factor (from `compute_normalization_factor`)
/// * `blend_mode` - How to combine with existing pixels
pub fn render_gaussian_spot(
    buffer: &mut [u8],
    size: PixelShape,
    spot_x: f64,
    spot_y: f64,
    fwhm_pixels: f64,
    normalization_factor: f64,
    blend_mode: BlendMode,
) {
    let (width, height) = size.to_u32_tuple();
    let reference_wavelength = Wavelength::from_nanometers(REFERENCE_WAVELENGTH_NM);
    let psf = PixelScaledAiryDisk::with_fwhm(fwhm_pixels, reference_wavelength);

    let cutoff_radius = psf.first_zero();
    let x_min = (spot_x - cutoff_radius).max(0.0) as u32;
    let x_max = (spot_x + cutoff_radius).min(width as f64) as u32;
    let y_min = (spot_y - cutoff_radius).max(0.0) as u32;
    let y_max = (spot_y + cutoff_radius).min(height as f64) as u32;

    for y in y_min..y_max {
        for x in x_min..x_max {
            let pixel_x = x as f64 - spot_x;
            let pixel_y = y as f64 - spot_y;

            let pixel_flux = psf.pixel_flux_simpson(pixel_x, pixel_y, normalization_factor);
            let pixel_value = pixel_flux.clamp(0.0, 255.0) as u8;

            let offset = ((y * width + x) * 3) as usize;
            match blend_mode {
                BlendMode::Overwrite => {
                    buffer[offset] = pixel_value;
                    buffer[offset + 1] = pixel_value;
                    buffer[offset + 2] = pixel_value;
                }
                BlendMode::Additive => {
                    buffer[offset] = buffer[offset].saturating_add(pixel_value);
                    buffer[offset + 1] = buffer[offset + 1].saturating_add(pixel_value);
                    buffer[offset + 2] = buffer[offset + 2].saturating_add(pixel_value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_factor_reasonable() {
        let factor = compute_normalization_factor(5.0, 255.0);
        // Should be positive and reasonable
        assert!(factor > 0.0);
        assert!(factor < 10000.0);
    }

    #[test]
    fn test_render_spot_overwrite() {
        let mut buffer = vec![128u8; 100 * 100 * 3];
        let norm = compute_normalization_factor(5.0, 255.0);
        render_gaussian_spot(
            &mut buffer,
            PixelShape::new(100, 100),
            50.0,
            50.0,
            5.0,
            norm,
            BlendMode::Overwrite,
        );

        // Center pixel should be bright
        let center_offset = (50 * 100 + 50) * 3;
        assert!(buffer[center_offset] > 200);
    }

    #[test]
    fn test_render_spot_additive() {
        let mut buffer = vec![100u8; 100 * 100 * 3];
        let norm = compute_normalization_factor(5.0, 100.0);
        render_gaussian_spot(
            &mut buffer,
            PixelShape::new(100, 100),
            50.0,
            50.0,
            5.0,
            norm,
            BlendMode::Additive,
        );

        // Center pixel should be brighter than base (100 + ~100 = ~200)
        let center_offset = (50 * 100 + 50) * 3;
        assert!(buffer[center_offset] > 150);
    }
}

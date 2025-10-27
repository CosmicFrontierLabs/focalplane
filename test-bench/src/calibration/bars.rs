use anyhow::Result;
use apriltag::Detection;
use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::HashMap;

/// Compute FFT-based contrast from 1D intensity profile.
///
/// This function analyzes the frequency content of an intensity profile
/// (typically extracted perpendicular to bar patterns) using FFT. The contrast
/// is measured as the amplitude of the fundamental frequency normalized by the DC component.
///
/// # Arguments
/// * `intensities` - 1D intensity profile perpendicular to bars
/// * `bar_spacing` - Expected spacing between bars in pixels
///
/// # Returns
/// * `Result<(contrast, spatial_frequency)>` - FFT-based contrast and spatial frequency in cycles/pixel
///
fn measure_bar_contrast_fft(intensities: &[f64], bar_spacing: f64) -> Result<(f64, f64)> {
    if intensities.is_empty() {
        anyhow::bail!("Cannot measure contrast: empty intensity profile");
    }

    let n = intensities.len();

    // Convert to complex numbers for FFT
    let mut buffer: Vec<Complex<f64>> = intensities
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();

    // Perform FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    // DC component (average intensity)
    let dc = buffer[0].norm() / n as f64;

    if dc < 0.001 {
        anyhow::bail!("Cannot measure contrast: DC component too low ({dc:.6})");
    }

    // Expected fundamental frequency bin
    // bar_spacing is the period of the pattern (one complete cycle: black + white)
    let profile_length = n as f64;
    let cycles_in_profile = profile_length / bar_spacing;
    let fundamental_bin = cycles_in_profile.round() as usize;

    if fundamental_bin == 0 || fundamental_bin >= n / 2 {
        anyhow::bail!("Cannot measure contrast: invalid fundamental bin {fundamental_bin} (n={n})");
    }

    // Get amplitude at fundamental frequency (normalized by DC)
    let fundamental_amplitude = buffer[fundamental_bin].norm() / n as f64;

    // Contrast from FFT amplitude
    // For perfect square wave with amplitude A, fundamental FFT amplitude = (4/π) * A / 2
    // DC component = mean intensity
    // Michelson contrast = (I_max - I_min) / (I_max + I_min) = A / mean
    // So: contrast = (π/2) * (fundamental_amplitude / dc)
    let contrast = (std::f64::consts::PI / 2.0) * (fundamental_amplitude / dc);

    let spatial_frequency = 1.0 / (2.0 * bar_spacing);

    Ok((contrast.min(1.0), spatial_frequency))
}

#[derive(Debug, Clone)]
pub struct BarMeasurement {
    pub position: (f64, f64),
    pub contrast: f64,
    pub spatial_frequency_cycles_per_pixel: f64,
    pub orientation: BarOrientation,
}

#[derive(Debug, Clone, Copy)]
pub enum BarOrientation {
    Horizontal,
    Vertical,
}

pub fn measure_bar_sharpness(
    frame: &Array2<u16>,
    detections: &[Detection],
    bit_depth: u8,
) -> Result<Vec<BarMeasurement>> {
    let mut measurements = Vec::new();

    let detection_map = build_tag_grid_map(detections);

    for row in 0..5 {
        for col in 0..4 {
            let left_id = row * 5 + col;
            let right_id = row * 5 + col + 1;

            if let (Some(&left), Some(&right)) =
                (detection_map.get(&left_id), detection_map.get(&right_id))
            {
                let bar_x = (left.0 + right.0) / 2.0;
                let bar_y = (left.1 + right.1) / 2.0;
                let bar_spacing = ((right.0 - left.0).powi(2) + (right.1 - left.1).powi(2)).sqrt();

                let intensities = extract_bar_intensity_profile(
                    frame,
                    (bar_x, bar_y),
                    bar_spacing,
                    BarOrientation::Horizontal,
                    bit_depth,
                );

                if let Ok((contrast, spatial_freq)) =
                    measure_bar_contrast_fft(&intensities, bar_spacing)
                {
                    measurements.push(BarMeasurement {
                        position: (bar_x, bar_y),
                        contrast,
                        spatial_frequency_cycles_per_pixel: spatial_freq,
                        orientation: BarOrientation::Horizontal,
                    });
                }
            }
        }
    }

    for row in 0..4 {
        for col in 0..5 {
            let top_id = row * 5 + col;
            let bottom_id = (row + 1) * 5 + col;

            if let (Some(&top), Some(&bottom)) =
                (detection_map.get(&top_id), detection_map.get(&bottom_id))
            {
                let bar_x = (top.0 + bottom.0) / 2.0;
                let bar_y = (top.1 + bottom.1) / 2.0;
                let bar_spacing = ((bottom.0 - top.0).powi(2) + (bottom.1 - top.1).powi(2)).sqrt();

                let intensities = extract_bar_intensity_profile(
                    frame,
                    (bar_x, bar_y),
                    bar_spacing,
                    BarOrientation::Vertical,
                    bit_depth,
                );

                if let Ok((contrast, spatial_freq)) =
                    measure_bar_contrast_fft(&intensities, bar_spacing)
                {
                    measurements.push(BarMeasurement {
                        position: (bar_x, bar_y),
                        contrast,
                        spatial_frequency_cycles_per_pixel: spatial_freq,
                        orientation: BarOrientation::Vertical,
                    });
                }
            }
        }
    }

    Ok(measurements)
}

/// Extract intensity profile from u16 frame perpendicular to bars, normalized by bit depth.
///
/// # Arguments
/// * `frame` - u16 frame data
/// * `pos` - Center position for sampling
/// * `bar_spacing` - Bar spacing in pixels
/// * `orientation` - Bar orientation (horizontal or vertical)
/// * `bit_depth` - ADC bit depth (8, 10, 12, 14, 16)
///
/// # Returns
/// * `Vec<f64>` - Normalized intensity values [0.0, 1.0] perpendicular to bars
///
pub fn extract_bar_intensity_profile(
    frame: &Array2<u16>,
    pos: (f64, f64),
    bar_spacing: f64,
    orientation: BarOrientation,
    bit_depth: u8,
) -> Vec<f64> {
    let (height, width) = frame.dim();
    let (cx, cy) = (pos.0 as i32, pos.1 as i32);
    let sample_radius = (bar_spacing * 0.4) as i32;

    if cx < sample_radius
        || cy < sample_radius
        || cx >= (width as i32 - sample_radius)
        || cy >= (height as i32 - sample_radius)
    {
        return Vec::new();
    }

    let max_value = ((1u32 << bit_depth) - 1) as f64;
    let mut intensities = Vec::new();

    match orientation {
        BarOrientation::Horizontal => {
            for dy in -sample_radius..=sample_radius {
                let y = (cy + dy) as usize;
                if y < height {
                    let pixel_value = frame[[y, cx as usize]] as f64;
                    intensities.push(pixel_value / max_value);
                }
            }
        }
        BarOrientation::Vertical => {
            for dx in -sample_radius..=sample_radius {
                let x = (cx + dx) as usize;
                if x < width {
                    let pixel_value = frame[[cy as usize, x]] as f64;
                    intensities.push(pixel_value / max_value);
                }
            }
        }
    }

    intensities
}

fn build_tag_grid_map(detections: &[Detection]) -> HashMap<usize, (f64, f64)> {
    let mut map = HashMap::new();
    for det in detections {
        let center = det.center();
        map.insert(det.id(), (center[0], center[1]));
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_contrast_on_synthetic_bars() {
        let bar_spacing = 32.0;
        let profile_len: usize = 256;

        let mut perfect_profile = Vec::new();
        for i in 0..profile_len {
            let val = if (i % 32) < 16 { 255.0 } else { 0.0 };
            perfect_profile.push(val);
        }

        let (contrast, freq) = measure_bar_contrast_fft(&perfect_profile, bar_spacing)
            .expect("Should measure contrast");

        println!("Perfect bars: contrast={:.6}, freq={:.6}", contrast, freq);
        assert!(contrast > 0.9, "Perfect bars should have high contrast");

        let mut blurred_profile = Vec::new();
        for i in 0..profile_len {
            let mut sum = 0.0;
            let mut count = 0;
            let start = i.saturating_sub(2);
            let end = (i + 2).min(profile_len - 1);
            for j in start..=end {
                sum += perfect_profile[j];
                count += 1;
            }
            blurred_profile.push(sum / count as f64);
        }

        let (blurred_contrast, _) = measure_bar_contrast_fft(&blurred_profile, bar_spacing)
            .expect("Should measure blurred contrast");

        println!("Blurred bars: contrast={:.6}", blurred_contrast);
        assert!(
            blurred_contrast < contrast,
            "Blurred bars should have lower contrast: {} vs {}",
            blurred_contrast,
            contrast
        );
    }
}

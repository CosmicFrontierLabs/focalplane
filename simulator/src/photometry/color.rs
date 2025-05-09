//! Color perception utilities for stellar spectra
//!
//! This module provides functions for mapping astronomical spectra
//! to human-perceived colors, as well as utilities for stellar
//! classification.

use super::human::HumanVision;
use super::spectrum::Spectrum;
use std::fmt;
use std::time::Duration;

/// The spectral classification system for stars, known as the Morgan-Keenan system.
///
/// This system classifies stars based on their spectral characteristics, primarily temperature.
/// The main classes from hottest to coolest are: O, B, A, F, G, K, M.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralClass {
    /// O-type stars: Very hot and bright, with temperatures above 30,000K
    /// These stars appear blue-white and have strong ionized helium lines
    O,

    /// B-type stars: Hot and blue-white stars, temperatures 10,000-30,000K
    B,

    /// A-type stars: White stars with strong hydrogen lines, temperatures 7,500-10,000K
    A,

    /// F-type stars: Yellow-white stars, temperatures 6,000-7,500K
    F,

    /// G-type stars: Yellow stars like our Sun, temperatures 5,200-6,000K
    G,

    /// K-type stars: Orange stars, temperatures 3,700-5,200K
    K,

    /// M-type stars: Red stars, temperatures below 3,700K
    M,
}

impl fmt::Display for SpectralClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display as a single character
        write!(
            f,
            "{}",
            match self {
                SpectralClass::O => 'O',
                SpectralClass::B => 'B',
                SpectralClass::A => 'A',
                SpectralClass::F => 'F',
                SpectralClass::G => 'G',
                SpectralClass::K => 'K',
                SpectralClass::M => 'M',
            }
        )
    }
}

/// Generate a sequence of stellar temperatures on a logarithmic scale
///
/// This ensures more even spacing across the range of stellar temperatures.
///
/// # Arguments
///
/// * `min_temp` - Minimum temperature in Kelvin
/// * `max_temp` - Maximum temperature in Kelvin
/// * `n_temps` - Number of temperatures to generate
///
/// # Returns
///
/// Vector of temperatures in Kelvin
pub fn generate_temperature_sequence(min_temp: f64, max_temp: f64, n_temps: usize) -> Vec<f64> {
    if n_temps <= 1 {
        return vec![min_temp];
    }

    let log_min = min_temp.ln();
    let log_max = max_temp.ln();
    let step = (log_max - log_min) / ((n_temps - 1) as f64);

    (0..n_temps)
        .map(|i| (log_min + i as f64 * step).exp())
        .collect()
}

/// Determine the spectral class (O, B, A, F, G, K, M) from temperature
///
/// # Arguments
///
/// * `temperature` - Stellar temperature in Kelvin
///
/// # Returns
///
/// Star's spectral class as a SpectralClass enum
pub fn temperature_to_spectral_class(temperature: f64) -> SpectralClass {
    if temperature >= 30000.0 {
        SpectralClass::O
    } else if temperature >= 10000.0 {
        SpectralClass::B
    } else if temperature >= 7500.0 {
        SpectralClass::A
    } else if temperature >= 6000.0 {
        SpectralClass::F
    } else if temperature >= 5200.0 {
        SpectralClass::G
    } else if temperature >= 3700.0 {
        SpectralClass::K
    } else {
        SpectralClass::M
    }
}

/// Calculate RGB values from spectrum using human vision models
///
/// This uses the human vision quantum efficiency models to simulate
/// how the human eye would perceive the spectrum.
///
/// # Arguments
///
/// * `spectrum` - The stellar spectrum to analyze
///
/// # Returns
///
/// Tuple of (r, g, b) values, each normalized to 0.0-1.0 range
pub fn spectrum_to_rgb_values(spectrum: &impl Spectrum) -> (f64, f64, f64) {
    // Set up duration and aperture for photo-electron calculations
    let duration = Duration::from_secs(1);
    let aperture_cm2 = 1.0;

    // Get quantum efficiency curves for red, green, and blue photoreceptors
    let red_qe = HumanVision::red();
    let green_qe = HumanVision::green_blue(); // Using green-blue as primary green
    let blue_qe = HumanVision::blue();

    // Calculate photo-electrons for each channel
    let red = spectrum.photo_electrons(&red_qe, aperture_cm2, &duration);
    let green = spectrum.photo_electrons(&green_qe, aperture_cm2, &duration);
    let blue = spectrum.photo_electrons(&blue_qe, aperture_cm2, &duration);

    // Normalize to the highest value to avoid clipping
    let max_value = red.max(green).max(blue);
    let scale = if max_value > 0.0 {
        1.0 / max_value
    } else {
        0.0
    };

    (red * scale, green * scale, blue * scale)
}

/// Calculate simple color temperature index
///
/// Returns a value where negative numbers indicate cool stars (redder)
/// and positive numbers indicate hot stars (bluer)
///
/// # Arguments
///
/// * `spectrum` - The stellar spectrum to analyze
///
/// # Returns
///
/// Color temperature index (negative for red, positive for blue)
pub fn color_temperature_index(spectrum: &impl Spectrum) -> f64 {
    // Set up duration and aperture for photo-electron calculations
    let duration = Duration::from_secs(1);
    let aperture_cm2 = 1.0;

    // Get quantum efficiency curves for red and blue
    let red_qe = HumanVision::red();
    let blue_qe = HumanVision::blue();

    // Calculate photo-electrons for red and blue
    let red = spectrum.photo_electrons(&red_qe, aperture_cm2, &duration);
    let blue = spectrum.photo_electrons(&blue_qe, aperture_cm2, &duration);

    // Color index as log ratio of blue to red
    if red > 0.0 && blue > 0.0 {
        (blue / red).ln()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photometry::stellar::BlackbodyStellarSpectrum;
    use approx::assert_relative_eq;

    #[test]
    fn test_temperature_sequence() {
        // Test with 5 temps from 1000K to 10000K
        let temps = generate_temperature_sequence(1000.0, 10000.0, 5);

        // Should get 5 temperatures
        assert_eq!(temps.len(), 5);

        // First and last should be min and max
        assert_relative_eq!(temps[0], 1000.0, epsilon = 0.1);
        assert_relative_eq!(temps[4], 10000.0, epsilon = 0.1);

        // Should be increasing
        for i in 1..temps.len() {
            assert!(temps[i] > temps[i - 1]);
        }

        // Test with single temperature
        let single_temp = generate_temperature_sequence(5000.0, 10000.0, 1);
        assert_eq!(single_temp.len(), 1);
        assert_eq!(single_temp[0], 5000.0);
    }

    #[test]
    fn test_spectral_class() {
        // Test each spectral class boundary
        assert_eq!(temperature_to_spectral_class(50000.0), SpectralClass::O);
        assert_eq!(temperature_to_spectral_class(30000.0), SpectralClass::O);
        assert_eq!(temperature_to_spectral_class(20000.0), SpectralClass::B);
        assert_eq!(temperature_to_spectral_class(9000.0), SpectralClass::A);
        assert_eq!(temperature_to_spectral_class(6500.0), SpectralClass::F);
        assert_eq!(temperature_to_spectral_class(5800.0), SpectralClass::G); // Sun
        assert_eq!(temperature_to_spectral_class(4000.0), SpectralClass::K);
        assert_eq!(temperature_to_spectral_class(3000.0), SpectralClass::M);
    }

    #[test]
    fn test_spectral_class_display() {
        // Test that display trait works correctly
        assert_eq!(format!("{}", SpectralClass::O), "O");
        assert_eq!(format!("{}", SpectralClass::B), "B");
        assert_eq!(format!("{}", SpectralClass::A), "A");
        assert_eq!(format!("{}", SpectralClass::F), "F");
        assert_eq!(format!("{}", SpectralClass::G), "G");
        assert_eq!(format!("{}", SpectralClass::K), "K");
        assert_eq!(format!("{}", SpectralClass::M), "M");
    }

    #[test]
    fn test_spectrum_to_rgb() {
        // Create spectra for different temperature stars
        let hot_star = BlackbodyStellarSpectrum::new(30000.0, 1.0); // Blue O star
        let sun_like = BlackbodyStellarSpectrum::new(5800.0, 1.0); // Yellow G star
        let cool_star = BlackbodyStellarSpectrum::new(3000.0, 1.0); // Red M star

        // Get RGB values
        let (hot_r, hot_g, hot_b) = spectrum_to_rgb_values(&hot_star);
        let (sun_r, sun_g, sun_b) = spectrum_to_rgb_values(&sun_like);
        let (cool_r, cool_g, cool_b) = spectrum_to_rgb_values(&cool_star);

        // Hot stars should have more blue
        assert!(hot_b > hot_r);

        // Sun-like stars should have balanced RGB
        assert!(sun_g > 0.5);

        // Cool stars should have more red
        assert!(cool_r > cool_b);

        // All values should be between 0 and 1
        assert!(hot_r >= 0.0 && hot_r <= 1.0);
        assert!(hot_g >= 0.0 && hot_g <= 1.0);
        assert!(hot_b >= 0.0 && hot_b <= 1.0);

        assert!(sun_r >= 0.0 && sun_r <= 1.0);
        assert!(sun_g >= 0.0 && sun_g <= 1.0);
        assert!(sun_b >= 0.0 && sun_b <= 1.0);

        assert!(cool_r >= 0.0 && cool_r <= 1.0);
        assert!(cool_g >= 0.0 && cool_g <= 1.0);
        assert!(cool_b >= 0.0 && cool_b <= 1.0);
    }

    #[test]
    fn test_color_temperature_index() {
        // Create spectra for different temperature stars
        let hot_star = BlackbodyStellarSpectrum::new(30000.0, 1.0); // Blue O star
        let sun_like = BlackbodyStellarSpectrum::new(5800.0, 1.0); // Yellow G star
        let cool_star = BlackbodyStellarSpectrum::new(3000.0, 1.0); // Red M star

        // Get color indices
        let hot_index = color_temperature_index(&hot_star);
        let sun_index = color_temperature_index(&sun_like);
        let cool_index = color_temperature_index(&cool_star);

        // Hot stars should have positive index
        assert!(hot_index > 0.0);

        // Cool stars should have negative index
        assert!(cool_index < 0.0);

        // Index should decrease with temperature
        assert!(hot_index > sun_index);
        assert!(sun_index > cool_index);
    }
}

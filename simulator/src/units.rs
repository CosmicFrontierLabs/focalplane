//! Type-safe physical units for astronomical simulations
//!
//! This module provides strongly-typed units using the `uom` crate to prevent
//! unit confusion errors at compile time. Starting with temperature units
//! and expanding to other physical quantities.

use uom::si::f64::*;
use uom::si::length::{centimeter, meter, micrometer, millimeter, nanometer};
use uom::si::thermodynamic_temperature::degree_celsius;

/// Type alias for temperature with convenient methods
pub type Temperature = ThermodynamicTemperature;

/// Type alias for length measurements with convenient methods
pub type Length = uom::si::f64::Length;

/// Extension trait for temperature conversions
pub trait TemperatureExt {
    /// Create temperature from degrees Celsius
    fn from_celsius(celsius: f64) -> Self;

    /// Get temperature in degrees Celsius
    fn as_celsius(&self) -> f64;

    /// Create temperature from Kelvin
    fn from_kelvin(kelvin: f64) -> Self;

    /// Get temperature in Kelvin
    fn as_kelvin(&self) -> f64;
}

/// Extension trait for length conversions commonly used in optics and sensors
pub trait LengthExt {
    /// Create length from nanometers (wavelengths)
    fn from_nanometers(nm: f64) -> Self;

    /// Get length in nanometers
    fn as_nanometers(&self) -> f64;

    /// Create length from micrometers (pixel sizes)
    fn from_micrometers(um: f64) -> Self;

    /// Get length in micrometers
    fn as_micrometers(&self) -> f64;

    /// Create length from millimeters
    fn from_millimeters(mm: f64) -> Self;

    /// Get length in millimeters
    fn as_millimeters(&self) -> f64;

    /// Create length from centimeters
    fn from_centimeters(cm: f64) -> Self;

    /// Get length in centimeters
    fn as_centimeters(&self) -> f64;

    /// Create length from meters
    fn from_meters(m: f64) -> Self;

    /// Get length in meters
    fn as_meters(&self) -> f64;
}

impl TemperatureExt for Temperature {
    fn from_celsius(celsius: f64) -> Self {
        Temperature::new::<degree_celsius>(celsius)
    }

    fn as_celsius(&self) -> f64 {
        self.get::<degree_celsius>()
    }

    fn from_kelvin(kelvin: f64) -> Self {
        Temperature::new::<uom::si::thermodynamic_temperature::kelvin>(kelvin)
    }

    fn as_kelvin(&self) -> f64 {
        self.get::<uom::si::thermodynamic_temperature::kelvin>()
    }
}

impl LengthExt for Length {
    fn from_nanometers(nm: f64) -> Self {
        Length::new::<nanometer>(nm)
    }

    fn as_nanometers(&self) -> f64 {
        self.get::<nanometer>()
    }

    fn from_micrometers(um: f64) -> Self {
        Length::new::<micrometer>(um)
    }

    fn as_micrometers(&self) -> f64 {
        self.get::<micrometer>()
    }

    fn from_millimeters(mm: f64) -> Self {
        Length::new::<millimeter>(mm)
    }

    fn as_millimeters(&self) -> f64 {
        self.get::<millimeter>()
    }

    fn from_centimeters(cm: f64) -> Self {
        Length::new::<centimeter>(cm)
    }

    fn as_centimeters(&self) -> f64 {
        self.get::<centimeter>()
    }

    fn from_meters(m: f64) -> Self {
        Length::new::<meter>(m)
    }

    fn as_meters(&self) -> f64 {
        self.get::<meter>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_temperature_conversions() {
        // Test Celsius to Kelvin
        let temp_c = Temperature::from_celsius(0.0);
        assert_relative_eq!(temp_c.as_kelvin(), 273.15, epsilon = 0.01);

        let temp_c = Temperature::from_celsius(-40.0);
        assert_relative_eq!(temp_c.as_kelvin(), 233.15, epsilon = 0.01);

        let temp_c = Temperature::from_celsius(20.0);
        assert_relative_eq!(temp_c.as_kelvin(), 293.15, epsilon = 0.01);

        // Test Kelvin to Celsius
        let temp_k = Temperature::from_kelvin(273.15);
        assert_relative_eq!(temp_k.as_celsius(), 0.0, epsilon = 0.01);

        let temp_k = Temperature::from_kelvin(373.15);
        assert_relative_eq!(temp_k.as_celsius(), 100.0, epsilon = 0.01);
    }

    #[test]
    fn test_temperature_math() {
        let temp1 = Temperature::from_celsius(20.0);
        let temp2 = Temperature::from_celsius(30.0);

        // Can average temperatures by converting to kelvin values
        let avg_kelvin = (temp1.as_kelvin() + temp2.as_kelvin()) / 2.0;
        let avg_temp = Temperature::from_kelvin(avg_kelvin);
        assert_relative_eq!(avg_temp.as_celsius(), 25.0, epsilon = 0.01);

        // Temperature differences
        let diff_kelvin = temp2.as_kelvin() - temp1.as_kelvin();
        assert_relative_eq!(diff_kelvin, 10.0, epsilon = 0.01);
    }

    #[test]
    fn test_length_conversions() {
        // Test nanometer conversions (wavelengths)
        let wavelength = Length::from_nanometers(550.0);
        assert_relative_eq!(wavelength.as_nanometers(), 550.0, epsilon = 0.001);
        assert_relative_eq!(wavelength.as_micrometers(), 0.55, epsilon = 0.001);
        assert_relative_eq!(wavelength.as_millimeters(), 0.00055, epsilon = 0.000001);
        assert_relative_eq!(wavelength.as_meters(), 0.00000055, epsilon = 1e-9);

        // Test micrometer conversions (pixel sizes)
        let pixel = Length::from_micrometers(4.6);
        assert_relative_eq!(pixel.as_micrometers(), 4.6, epsilon = 0.001);
        assert_relative_eq!(pixel.as_nanometers(), 4600.0, epsilon = 0.1);
        assert_relative_eq!(pixel.as_millimeters(), 0.0046, epsilon = 0.00001);
        assert_relative_eq!(pixel.as_meters(), 0.0000046, epsilon = 1e-8);

        // Test millimeter conversions
        let aperture = Length::from_millimeters(100.0);
        assert_relative_eq!(aperture.as_millimeters(), 100.0, epsilon = 0.001);
        assert_relative_eq!(aperture.as_centimeters(), 10.0, epsilon = 0.001);
        assert_relative_eq!(aperture.as_meters(), 0.1, epsilon = 0.001);
        assert_relative_eq!(aperture.as_micrometers(), 100000.0, epsilon = 0.1);

        // Test centimeter conversions
        let sensor_size = Length::from_centimeters(2.36);
        assert_relative_eq!(sensor_size.as_centimeters(), 2.36, epsilon = 0.001);
        assert_relative_eq!(sensor_size.as_millimeters(), 23.6, epsilon = 0.01);
        assert_relative_eq!(sensor_size.as_micrometers(), 23600.0, epsilon = 0.1);

        // Test meter conversions
        let focal_length = Length::from_meters(0.5);
        assert_relative_eq!(focal_length.as_meters(), 0.5, epsilon = 0.001);
        assert_relative_eq!(focal_length.as_millimeters(), 500.0, epsilon = 0.01);
        assert_relative_eq!(focal_length.as_centimeters(), 50.0, epsilon = 0.01);
    }

    #[test]
    fn test_length_math() {
        let pixel1 = Length::from_micrometers(4.6);
        let pixel2 = Length::from_micrometers(9.0);

        // Addition
        let sum = pixel1 + pixel2;
        assert_relative_eq!(sum.as_micrometers(), 13.6, epsilon = 0.001);

        // Subtraction
        let diff = pixel2 - pixel1;
        assert_relative_eq!(diff.as_micrometers(), 4.4, epsilon = 0.001);

        // Multiplication by scalar
        let doubled = pixel1 * 2.0;
        assert_relative_eq!(doubled.as_micrometers(), 9.2, epsilon = 0.001);

        // Division by scalar
        let halved = pixel2 / 2.0;
        assert_relative_eq!(halved.as_micrometers(), 4.5, epsilon = 0.001);

        // Division of lengths gives dimensionless ratio
        let ratio = pixel2.as_micrometers() / pixel1.as_micrometers();
        assert_relative_eq!(ratio, 9.0 / 4.6, epsilon = 0.001);
    }

    #[test]
    fn test_typical_sensor_values() {
        // Typical pixel sizes
        let small_pixel = Length::from_micrometers(2.4);
        let large_pixel = Length::from_micrometers(24.0);
        assert!(small_pixel < large_pixel);

        // Typical wavelengths
        let uv = Length::from_nanometers(350.0);
        let visible = Length::from_nanometers(550.0);
        let infrared = Length::from_nanometers(900.0);
        assert!(uv < visible);
        assert!(visible < infrared);

        // Convert wavelength to micrometers for comparison with pixel size
        assert!(visible.as_micrometers() < small_pixel.as_micrometers());
    }
}

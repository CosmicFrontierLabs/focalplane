//! Type-safe physical units for astronomical simulations
//!
//! This module provides strongly-typed units using the `uom` crate to prevent
//! unit confusion errors at compile time. Starting with temperature units
//! and expanding to other physical quantities.

use uom::si::angle::{degree, radian};
use uom::si::f64::*;
use uom::si::length::{centimeter, meter, micrometer, millimeter, nanometer};
use uom::si::thermodynamic_temperature::degree_celsius;

/// Type alias for temperature with convenient methods
pub type Temperature = ThermodynamicTemperature;

/// Type alias for length measurements with convenient methods
pub type Length = uom::si::f64::Length;

/// Type alias for wavelength measurements (specialized Length for optical wavelengths)
///
/// While physically identical to Length, this type alias improves code clarity
/// when dealing with electromagnetic wavelengths in nanometers.
pub type Wavelength = Length;

/// Type alias for angular measurements with convenient methods
pub type Angle = uom::si::f64::Angle;

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

/// Extension trait for angular conversions commonly used in astronomy
pub trait AngleExt {
    /// Create angle from degrees (human-friendly input)
    fn from_degrees(degrees: f64) -> Self;

    /// Get angle in degrees
    fn as_degrees(&self) -> f64;

    /// Create angle from radians (mathematical operations)
    fn from_radians(radians: f64) -> Self;

    /// Get angle in radians
    fn as_radians(&self) -> f64;

    /// Create angle from arcseconds (telescope pointing accuracy)
    fn from_arcseconds(arcseconds: f64) -> Self;

    /// Get angle in arcseconds
    fn as_arcseconds(&self) -> f64;

    /// Create angle from milliarcseconds (high precision astrometry)
    fn from_milliarcseconds(mas: f64) -> Self;

    /// Get angle in milliarcseconds
    fn as_milliarcseconds(&self) -> f64;
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

impl AngleExt for Angle {
    fn from_degrees(degrees: f64) -> Self {
        Angle::new::<degree>(degrees)
    }

    fn as_degrees(&self) -> f64 {
        self.get::<degree>()
    }

    fn from_radians(radians: f64) -> Self {
        Angle::new::<radian>(radians)
    }

    fn as_radians(&self) -> f64 {
        self.get::<radian>()
    }

    fn from_arcseconds(arcseconds: f64) -> Self {
        // Convert arcseconds to degrees: 1 degree = 3600 arcseconds
        let degrees = arcseconds / 3600.0;
        Angle::new::<degree>(degrees)
    }

    fn as_arcseconds(&self) -> f64 {
        self.as_degrees() * 3600.0
    }

    fn from_milliarcseconds(mas: f64) -> Self {
        // Convert milliarcseconds to degrees: 1 degree = 3,600,000 milliarcseconds
        let degrees = mas / 3_600_000.0;
        Angle::new::<degree>(degrees)
    }

    fn as_milliarcseconds(&self) -> f64 {
        self.as_degrees() * 3_600_000.0
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

    #[test]
    fn test_angular_conversions() {
        use std::f64::consts::PI;

        // Test basic degree/radian conversions
        let angle_90_deg = Angle::from_degrees(90.0);
        assert_relative_eq!(angle_90_deg.as_radians(), PI / 2.0, epsilon = 1e-10);
        assert_relative_eq!(angle_90_deg.as_degrees(), 90.0, epsilon = 1e-10);

        let angle_pi_rad = Angle::from_radians(PI);
        assert_relative_eq!(angle_pi_rad.as_degrees(), 180.0, epsilon = 1e-10);
        assert_relative_eq!(angle_pi_rad.as_radians(), PI, epsilon = 1e-10);

        // Test arcsecond conversions
        let angle_1_arcsec = Angle::from_arcseconds(1.0);
        assert_relative_eq!(angle_1_arcsec.as_arcseconds(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(angle_1_arcsec.as_degrees(), 1.0 / 3600.0, epsilon = 1e-12);
        assert_relative_eq!(
            angle_1_arcsec.as_radians(),
            (1.0 / 3600.0) * PI / 180.0,
            epsilon = 1e-12
        );

        // Test milliarcsecond conversions
        let angle_1000_mas = Angle::from_milliarcseconds(1000.0);
        assert_relative_eq!(angle_1000_mas.as_milliarcseconds(), 1000.0, epsilon = 1e-10);
        assert_relative_eq!(angle_1000_mas.as_arcseconds(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(angle_1000_mas.as_degrees(), 1.0 / 3600.0, epsilon = 1e-12);

        // Test small angle conversions (common in astronomy)
        let angle_100_mas = Angle::from_milliarcseconds(100.0);
        assert_relative_eq!(angle_100_mas.as_arcseconds(), 0.1, epsilon = 1e-10);
        assert_relative_eq!(angle_100_mas.as_degrees(), 0.1 / 3600.0, epsilon = 1e-12);
    }

    #[test]
    fn test_angular_math() {
        // Test angle arithmetic
        let angle1 = Angle::from_degrees(30.0);
        let angle2 = Angle::from_degrees(60.0);

        // Addition
        let sum = angle1 + angle2;
        assert_relative_eq!(sum.as_degrees(), 90.0, epsilon = 1e-10);

        // Subtraction
        let diff = angle2 - angle1;
        assert_relative_eq!(diff.as_degrees(), 30.0, epsilon = 1e-10);

        // Multiplication by scalar
        let doubled = angle1 * 2.0;
        assert_relative_eq!(doubled.as_degrees(), 60.0, epsilon = 1e-10);

        // Division by scalar
        let halved = angle2 / 2.0;
        assert_relative_eq!(halved.as_degrees(), 30.0, epsilon = 1e-10);

        // Division of angles gives dimensionless ratio
        let ratio = angle2.as_degrees() / angle1.as_degrees();
        assert_relative_eq!(ratio, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_typical_astronomical_angles() {
        use std::f64::consts::PI;

        // Full circle
        let full_circle = Angle::from_degrees(360.0);
        assert_relative_eq!(full_circle.as_radians(), 2.0 * PI, epsilon = 1e-10);

        // Typical telescope pointing accuracy (sub-arcsecond)
        let pointing_accuracy = Angle::from_milliarcseconds(100.0);
        assert_relative_eq!(pointing_accuracy.as_arcseconds(), 0.1, epsilon = 1e-10);
        assert!(pointing_accuracy.as_degrees() < 0.001);

        // Airy disk size (order of arcseconds for visible light)
        let airy_disk = Angle::from_arcseconds(2.0);
        assert_relative_eq!(airy_disk.as_milliarcseconds(), 2000.0, epsilon = 1e-10);
        assert_relative_eq!(airy_disk.as_degrees(), 2.0 / 3600.0, epsilon = 1e-12);

        // Field of view (degrees)
        let fov = Angle::from_degrees(1.0);
        assert_relative_eq!(fov.as_arcseconds(), 3600.0, epsilon = 1e-10);
        assert_relative_eq!(fov.as_milliarcseconds(), 3_600_000.0, epsilon = 1e-5);

        // Test ordering
        assert!(pointing_accuracy < airy_disk);
        assert!(airy_disk < fov);
    }

    #[test]
    fn test_angular_precision_limits() {
        // Test very small angles (sub-milliarcsecond precision)
        let micro_arcsec = Angle::from_milliarcseconds(0.001);
        assert_relative_eq!(micro_arcsec.as_milliarcseconds(), 0.001, epsilon = 1e-12);

        // Test very large angles
        let large_angle = Angle::from_degrees(720.0); // Two full rotations
        assert_relative_eq!(large_angle.as_degrees(), 720.0, epsilon = 1e-10);

        // Test conversion chain consistency
        let original_mas = 1234.567;
        let angle = Angle::from_milliarcseconds(original_mas);
        let recovered_mas = angle.as_milliarcseconds();
        assert_relative_eq!(original_mas, recovered_mas, epsilon = 1e-10);
    }
}

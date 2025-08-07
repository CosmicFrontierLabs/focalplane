//! Type-safe physical units for astronomical simulations
//!
//! This module provides strongly-typed units using the `uom` crate to prevent
//! unit confusion errors at compile time. Starting with temperature units
//! and expanding to other physical quantities.

use uom::si::f64::*;
use uom::si::thermodynamic_temperature::degree_celsius;

/// Type alias for temperature with convenient methods
pub type Temperature = ThermodynamicTemperature;

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
}

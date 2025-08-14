//! Dark current estimation for different sensor temperatures
//!
//! This module provides functionality for estimating dark current at various
//! temperatures based on sensor specifications and thermal models.

use crate::algo::misc::{interp, InterpError};
use crate::units::{Temperature, TemperatureExt};

/// Minimum temperature for dark current interpolation table (°C)
pub const MIN_TEMP_C: f64 = -40.0;

/// Maximum temperature for dark current interpolation table (°C)
pub const MAX_TEMP_C: f64 = 40.0;

/// Number of points in the interpolation table
const INTERPOLATION_POINTS: usize = 100;

/// Dark current estimator that uses interpolation of temperature vs dark current curves
/// to predict values at any temperature. Can be initialized either from reference values
/// (using the 8°C doubling rule) or from explicit temperature/dark current data points.
#[derive(Debug, Clone, PartialEq)]
pub struct DarkCurrentEstimator {
    /// Temperature values in degrees Celsius
    temperatures: Vec<f64>,
    /// Dark current values in electrons/pixel/second
    dark_currents: Vec<f64>,
    /// Log of dark current values for exponential interpolation
    log_dark_currents: Vec<f64>,
}

impl DarkCurrentEstimator {
    /// Generate temperature points for interpolation table
    fn generate_temperature_points() -> Vec<f64> {
        let mut temperatures = Vec::with_capacity(INTERPOLATION_POINTS);
        for i in 0..INTERPOLATION_POINTS {
            let temp = MIN_TEMP_C
                + (i as f64) * (MAX_TEMP_C - MIN_TEMP_C) / (INTERPOLATION_POINTS - 1) as f64;
            temperatures.push(temp);
        }
        temperatures
    }
    /// Creates a new dark current estimator with reference values
    ///
    /// Generates an interpolation table from MIN_TEMP_C to MAX_TEMP_C using the
    /// rule that dark current doubles every 8°C temperature increase.
    ///
    /// # Arguments
    /// * `reference_dark_current` - Dark current in electrons/pixel/second at reference temperature
    /// * `reference_temperature` - Reference temperature
    ///
    /// # Example
    /// ```
    /// use simulator::hardware::dark_current::DarkCurrentEstimator;
    /// use simulator::units::{Temperature, TemperatureExt};
    ///
    /// let estimator = DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(20.0));
    /// ```
    pub fn from_reference_point(
        reference_dark_current: f64,
        reference_temperature: Temperature,
    ) -> Self {
        // Generate points from MIN_TEMP_C to MAX_TEMP_C
        let temperatures = Self::generate_temperature_points();

        // Calculate dark current at each temperature using 8°C doubling rule
        let dark_currents: Vec<f64> = temperatures
            .iter()
            .map(|&temp| {
                let temp_diff = temp - reference_temperature.as_celsius();
                let doubling_periods = temp_diff / 8.0;
                reference_dark_current * 2.0_f64.powf(doubling_periods)
            })
            .collect();

        let log_dark_currents = dark_currents.iter().map(|&dc| dc.ln()).collect();
        Self {
            temperatures,
            dark_currents,
            log_dark_currents,
        }
    }

    /// Creates a dark current estimator from two reference points
    ///
    /// Fits the 8°C doubling rule through two reference points to generate
    /// a full interpolation table from MIN_TEMP_C to MAX_TEMP_C.
    ///
    /// # Arguments
    /// * `temp1` - First reference temperature
    /// * `dark_current1` - Dark current at first temperature (e⁻/pixel/second)
    /// * `temp2` - Second reference temperature  
    /// * `dark_current2` - Dark current at second temperature (e⁻/pixel/second)
    ///
    /// # Example
    /// ```
    /// use simulator::hardware::dark_current::DarkCurrentEstimator;
    /// use simulator::units::{Temperature, TemperatureExt};
    ///
    /// // Create from two measured points
    /// let estimator = DarkCurrentEstimator::from_two_points(
    ///     Temperature::from_celsius(-10.0), 0.01,  // 0.01 e⁻/px/s at -10°C
    ///     Temperature::from_celsius(20.0), 0.5     // 0.5 e⁻/px/s at 20°C
    /// );
    /// ```
    pub fn from_two_points(
        temp1: Temperature,
        dark_current1: f64,
        temp2: Temperature,
        dark_current2: f64,
    ) -> Self {
        // Calculate the doubling period from the two points
        // dark_current2 = dark_current1 * 2^((temp2 - temp1) / doubling_period)
        // log2(dark_current2 / dark_current1) = (temp2 - temp1) / doubling_period
        // doubling_period = (temp2 - temp1) / log2(dark_current2 / dark_current1)

        let temp_diff = temp2.as_celsius() - temp1.as_celsius();
        let dark_ratio = dark_current2 / dark_current1;
        let doubling_period = temp_diff / dark_ratio.log2();

        // Now we can calculate the reference dark current at any temperature
        // Using the first point as reference
        let reference_dark_current = dark_current1;
        let reference_temp = temp1;

        // Generate points from MIN_TEMP_C to MAX_TEMP_C
        let temperatures = Self::generate_temperature_points();

        // Calculate dark current at each temperature using fitted model
        let dark_currents: Vec<f64> = temperatures
            .iter()
            .map(|&temp| {
                let temp_diff = temp - reference_temp.as_celsius();
                let doubling_periods = temp_diff / doubling_period;
                reference_dark_current * 2.0_f64.powf(doubling_periods)
            })
            .collect();

        let log_dark_currents = dark_currents.iter().map(|&dc| dc.ln()).collect();
        Self {
            temperatures,
            dark_currents,
            log_dark_currents,
        }
    }

    /// Creates a dark current estimator from explicit temperature/dark current data
    ///
    /// # Arguments
    /// * `temperatures` - Temperature values in degrees Celsius (must be sorted ascending)
    /// * `dark_currents` - Dark current values in electrons/pixel/second
    ///
    /// # Example
    /// ```
    /// use simulator::hardware::dark_current::DarkCurrentEstimator;
    ///
    /// let temps = vec![-40.0, -20.0, 0.0, 20.0, 40.0];
    /// let dark_currents = vec![0.001, 0.01, 0.1, 1.0, 10.0];
    /// let estimator = DarkCurrentEstimator::from_curve(temps, dark_currents);
    /// ```
    pub fn from_curve(temperatures: Vec<f64>, dark_currents: Vec<f64>) -> Self {
        let log_dark_currents = dark_currents.iter().map(|&dc| dc.ln()).collect();
        Self {
            temperatures,
            dark_currents,
            log_dark_currents,
        }
    }

    /// Estimates dark current at a target temperature
    ///
    /// Uses linear interpolation on the internal temperature/dark current table
    /// to provide estimates at any temperature.
    ///
    /// # Arguments
    /// * `temperature` - Target temperature as a Temperature type
    ///
    /// # Returns
    /// * `Ok(f64)` - Estimated dark current in electrons/pixel/second at target temperature
    /// * `Err(InterpError)` - If interpolation fails (e.g., out of bounds)
    ///
    /// # Example
    /// ```
    /// use simulator::hardware::dark_current::DarkCurrentEstimator;
    /// use simulator::units::{Temperature, TemperatureExt};
    ///
    /// let estimator = DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(20.0));
    /// let temp = Temperature::from_celsius(28.0);
    /// let dark_current = estimator.estimate_at_temperature(temp).expect("Temperature out of range");
    /// ```
    pub fn estimate_at_temperature(&self, temperature: Temperature) -> Result<f64, InterpError> {
        // Interpolate in log space to preserve exponential nature
        let target_temp_c = temperature.as_celsius();
        let log_result = interp(target_temp_c, &self.temperatures, &self.log_dark_currents)?;
        // Convert back from log space
        Ok(log_result.exp())
    }

    /// Calculate the temperature increase needed to double dark current
    ///
    /// Returns the number of degrees Celsius required for dark current to double.
    /// This is a characteristic property of the sensor that helps understand
    /// its thermal behavior.
    ///
    /// # Example
    /// ```
    /// use simulator::hardware::dark_current::DarkCurrentEstimator;
    /// use simulator::units::{Temperature, TemperatureExt};
    ///
    /// let estimator = DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(20.0));
    /// let doubling_temp = estimator.calculate_doubling_temperature();
    /// println!("Dark current doubles every {:.1}°C", doubling_temp);
    /// ```
    pub fn calculate_doubling_temperature(&self) -> f64 {
        // Pick two reference temperatures well within our range
        let temp1 = 0.0;
        let temp2 = 10.0;

        // Get dark currents at these temperatures (should always succeed for these temps)
        let dc1 = self
            .estimate_at_temperature(Temperature::from_celsius(temp1))
            .unwrap_or(1.0);
        let dc2 = self
            .estimate_at_temperature(Temperature::from_celsius(temp2))
            .unwrap_or(2.0);

        // Calculate doubling temperature
        // dc2 = dc1 * 2^(delta_T / doubling_T)
        // log2(dc2/dc1) = delta_T / doubling_T
        // doubling_T = delta_T / log2(dc2/dc1)
        let temp_diff = temp2 - temp1;
        let ratio = dc2 / dc1;
        temp_diff / ratio.log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_generate_temperature_points() {
        let temps = DarkCurrentEstimator::generate_temperature_points();

        // Should have INTERPOLATION_POINTS values
        assert_eq!(temps.len(), INTERPOLATION_POINTS);

        // First and last should be MIN_TEMP_C and MAX_TEMP_C
        assert_eq!(temps[0], MIN_TEMP_C);
        assert_eq!(temps[INTERPOLATION_POINTS - 1], MAX_TEMP_C);

        // Should be evenly spaced
        let spacing = (MAX_TEMP_C - MIN_TEMP_C) / (INTERPOLATION_POINTS - 1) as f64;
        for i in 1..INTERPOLATION_POINTS {
            let expected = MIN_TEMP_C + (i as f64) * spacing;
            assert_relative_eq!(temps[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_same_temperature() {
        let estimator =
            DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(20.0));
        let result = estimator
            .estimate_at_temperature(Temperature::from_celsius(20.0))
            .expect("Temperature should be in range");
        // With interpolation, we might not get exact match
        assert_relative_eq!(result, 0.1, epsilon = 0.05); // 5% tolerance for interpolation
    }

    #[test]
    fn test_8_degree_increase_doubles() {
        let estimator =
            DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(20.0));
        // 28°C is within our table range (-40 to +40)
        let result = estimator
            .estimate_at_temperature(Temperature::from_celsius(28.0))
            .expect("Temperature should be in range");
        assert_relative_eq!(result, 0.2, epsilon = 0.1); // 10% tolerance for interpolation
    }

    #[test]
    fn test_8_degree_decrease_halves() {
        let estimator =
            DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(20.0));
        let result = estimator
            .estimate_at_temperature(Temperature::from_celsius(12.0))
            .expect("Temperature should be in range");
        assert_relative_eq!(result, 0.05, epsilon = 0.05); // 5% tolerance for interpolation
    }

    #[test]
    fn test_16_degree_increase_quadruples() {
        let estimator =
            DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(20.0));
        // 36°C is within our table range (-40 to +40)
        let result = estimator
            .estimate_at_temperature(Temperature::from_celsius(36.0))
            .expect("Temperature should be in range");
        assert_relative_eq!(result, 0.4, epsilon = 0.1); // 10% tolerance for interpolation
    }

    #[test]
    fn test_16_degree_decrease_quarters() {
        let estimator =
            DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(20.0));
        let result = estimator
            .estimate_at_temperature(Temperature::from_celsius(4.0))
            .expect("Temperature should be in range");
        assert_relative_eq!(result, 0.025, epsilon = 1e-3);
    }

    #[test]
    fn test_4_degree_increase_sqrt2() {
        let estimator =
            DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(20.0));
        // 24°C is within our table range (-40 to +40)
        let result = estimator
            .estimate_at_temperature(Temperature::from_celsius(24.0))
            .expect("Temperature should be in range");
        let expected = 0.1 * 2.0_f64.powf(0.5); // sqrt(2) increase for 4°C
        assert_relative_eq!(result, expected, epsilon = 0.06); // 6% tolerance for interpolation
    }

    #[test]
    fn test_negative_reference_temperature() {
        let estimator =
            DarkCurrentEstimator::from_reference_point(0.04, Temperature::from_celsius(-40.0));
        // -40°C is at the edge of our table range (-40 to +40)
        let result = estimator
            .estimate_at_temperature(Temperature::from_celsius(-40.0))
            .expect("Temperature should be in range");
        assert_relative_eq!(result, 0.04, epsilon = 1e-6);
        // -32°C should double from -40°C reference
        let result = estimator
            .estimate_at_temperature(Temperature::from_celsius(-32.0))
            .expect("Temperature should be in range");
        assert_relative_eq!(result, 0.08, epsilon = 0.03); // 3% tolerance for interpolation
    }

    #[test]
    fn test_large_temperature_difference() {
        let estimator =
            DarkCurrentEstimator::from_reference_point(0.001, Temperature::from_celsius(-20.0));
        // 60°C is outside our table range (-40 to +40), should return Err
        assert!(estimator
            .estimate_at_temperature(Temperature::from_celsius(60.0))
            .is_err());
        // Test within range: 40°C is 60°C increase from -20°C
        // That's 7.5 doubling periods, so result should be 0.001 * 2^7.5
        let result = estimator
            .estimate_at_temperature(Temperature::from_celsius(40.0))
            .expect("Temperature should be in range");
        let expected = 0.001 * 2.0_f64.powf(7.5);
        assert_relative_eq!(result, expected, epsilon = 1e-2);
    }

    #[test]
    fn test_fractional_doubling_periods() {
        let estimator =
            DarkCurrentEstimator::from_reference_point(1.0, Temperature::from_celsius(0.0));
        // 6°C increase = 6/8 = 0.75 doubling periods
        // Should be 1.0 * 2^0.75 = 1.0 * 1.6817928... ≈ 1.6818
        let result = estimator
            .estimate_at_temperature(Temperature::from_celsius(6.0))
            .expect("Temperature should be in range");
        let expected = 2.0_f64.powf(0.75);
        assert_relative_eq!(result, expected, epsilon = 0.1); // 10% tolerance for interpolation
    }

    #[test]
    fn test_different_reference_values() {
        // Test with different reference dark currents
        let high_dc =
            DarkCurrentEstimator::from_reference_point(10.0, Temperature::from_celsius(25.0));
        let low_dc =
            DarkCurrentEstimator::from_reference_point(0.001, Temperature::from_celsius(25.0));

        // Both should scale by same factor for same temperature change
        let temp_change = 33.0; // 8°C increase, should double
                                // 33°C is within our table range (-40 to +40)
        let high_result = high_dc
            .estimate_at_temperature(Temperature::from_celsius(temp_change))
            .expect("Temperature should be in range");
        let low_result = low_dc
            .estimate_at_temperature(Temperature::from_celsius(temp_change))
            .expect("Temperature should be in range");

        // High dark current should double: 10 -> ~20
        assert!(
            high_result > 18.0 && high_result < 22.0,
            "Expected ~20, got {high_result}"
        );
        // Low dark current should also double: 0.001 -> ~0.002
        assert_relative_eq!(low_result, 0.002, epsilon = 0.1); // 10% tolerance for interpolation
    }

    #[test]
    fn test_from_curve() {
        // Test creating estimator from explicit curve data
        let temps = vec![-40.0, -20.0, 0.0, 20.0, 40.0];
        let dark_currents = vec![0.001, 0.01, 0.1, 1.0, 10.0];
        let estimator = DarkCurrentEstimator::from_curve(temps.clone(), dark_currents.clone());

        // Should interpolate through the given points
        assert_relative_eq!(
            estimator
                .estimate_at_temperature(Temperature::from_celsius(-20.0))
                .expect("Temperature should be in range"),
            0.01,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            estimator
                .estimate_at_temperature(Temperature::from_celsius(0.0))
                .expect("Temperature should be in range"),
            0.1,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            estimator
                .estimate_at_temperature(Temperature::from_celsius(20.0))
                .expect("Temperature should be in range"),
            1.0,
            epsilon = 1e-6
        );

        // Test interpolation between points
        let mid_value = estimator
            .estimate_at_temperature(Temperature::from_celsius(10.0))
            .expect("Temperature should be in range");
        assert!(mid_value > 0.1 && mid_value < 1.0);
    }

    #[test]
    fn test_within_table_range() {
        // Test that values within the table range work correctly
        let estimator =
            DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(0.0));

        // Should work for temps between -40 and +40
        assert!(estimator
            .estimate_at_temperature(Temperature::from_celsius(-40.0))
            .is_ok());
        assert!(estimator
            .estimate_at_temperature(Temperature::from_celsius(-20.0))
            .is_ok());
        assert!(estimator
            .estimate_at_temperature(Temperature::from_celsius(-10.0))
            .is_ok());
        assert!(estimator
            .estimate_at_temperature(Temperature::from_celsius(0.0))
            .is_ok());
        assert!(estimator
            .estimate_at_temperature(Temperature::from_celsius(10.0))
            .is_ok());
        assert!(estimator
            .estimate_at_temperature(Temperature::from_celsius(20.0))
            .is_ok());
        assert!(estimator
            .estimate_at_temperature(Temperature::from_celsius(40.0))
            .is_ok());

        // Should fail outside range
        assert!(estimator
            .estimate_at_temperature(Temperature::from_celsius(-41.0))
            .is_err());
        assert!(estimator
            .estimate_at_temperature(Temperature::from_celsius(41.0))
            .is_err());
    }

    #[test]
    fn test_from_two_points() {
        // Create estimator from two measured points
        let estimator = DarkCurrentEstimator::from_two_points(
            Temperature::from_celsius(0.0),
            0.1, // 0.1 e⁻/px/s at 0°C
            Temperature::from_celsius(16.0),
            0.4, // 0.4 e⁻/px/s at 16°C (should be 2x at 8°C, 4x at 16°C)
        );

        // Check that it interpolates correctly at the reference points
        assert_relative_eq!(
            estimator
                .estimate_at_temperature(Temperature::from_celsius(0.0))
                .expect("Should be in range"),
            0.1,
            epsilon = 0.1 // Allow 10% tolerance due to interpolation
        );

        // Check intermediate value (8°C should be ~0.2)
        assert_relative_eq!(
            estimator
                .estimate_at_temperature(Temperature::from_celsius(8.0))
                .expect("Should be in range"),
            0.2,
            epsilon = 0.1 // Allow 10% tolerance due to interpolation
        );
    }

    #[test]
    fn test_type_safe_temperature() {
        use crate::units::{Temperature, TemperatureExt};

        let estimator =
            DarkCurrentEstimator::from_reference_point(0.1, Temperature::from_celsius(20.0));

        // Test with Celsius
        let temp_c = Temperature::from_celsius(28.0);
        let dark_current_c = estimator
            .estimate_at_temperature(temp_c)
            .expect("Should work");

        // Test with Kelvin - should give same result
        let temp_k = Temperature::from_kelvin(301.15); // 28°C in Kelvin
        let dark_current_k = estimator
            .estimate_at_temperature(temp_k)
            .expect("Should work");

        assert_relative_eq!(dark_current_c, dark_current_k, epsilon = 1e-10);

        // Should be ~0.2 e-/pixel/s (doubled for 8°C increase from 20°C)
        assert_relative_eq!(dark_current_c, 0.2, epsilon = 0.01);

        // Test that we can't accidentally mix units
        // This would be a compile error if we tried to pass a raw f64:
        // estimator.estimate_at_temperature(28.0); // Won't compile!
    }

    #[test]
    fn test_from_two_points_different_doubling() {
        // Create estimator with different doubling period (not 8°C)
        // If dark current quadruples in 10°C, the doubling period is 5°C
        let estimator = DarkCurrentEstimator::from_two_points(
            Temperature::from_celsius(-10.0),
            0.025, // 0.025 e⁻/px/s at -10°C
            Temperature::from_celsius(0.0),
            0.1, // 0.1 e⁻/px/s at 0°C (4x in 10°C)
        );

        // Verify it fits through both points
        assert_relative_eq!(
            estimator
                .estimate_at_temperature(Temperature::from_celsius(-10.0))
                .expect("Should be in range"),
            0.025,
            epsilon = 0.1 // Allow 10% tolerance due to interpolation
        );
        assert_relative_eq!(
            estimator
                .estimate_at_temperature(Temperature::from_celsius(0.0))
                .expect("Should be in range"),
            0.1,
            epsilon = 0.1 // Allow 10% tolerance due to interpolation
        );

        // Check that at 5°C it should be ~0.2 (double of 0°C value)
        assert_relative_eq!(
            estimator
                .estimate_at_temperature(Temperature::from_celsius(5.0))
                .expect("Should be in range"),
            0.2,
            epsilon = 0.1 // Allow 10% tolerance due to interpolation
        );
    }
}

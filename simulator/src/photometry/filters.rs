//! Standard astronomical filters (U/B/V) quantum efficiency models
//!
//! This module provides implementations of standard astronomical filters
//! based on the Johnson-Cousins UBV photometric system.

use super::QuantumEfficiency;

/// Hardcoded U filter transmission data from the Zodical Light Curve - SpectrumPassbands.csv
/// Format: (wavelength_nm, transmission)
const U_FILTER_DATA: [(f64, f64); 21] = [
    (250.0, 0.0),
    (260.0, 0.0),
    (270.0, 0.0),
    (280.0, 0.0),
    (290.0, 0.05),
    (300.0, 0.2),
    (310.0, 0.4),
    (320.0, 0.6),
    (330.0, 0.8),
    (340.0, 0.95),
    (350.0, 1.0),
    (360.0, 0.95),
    (370.0, 0.8),
    (380.0, 0.6),
    (390.0, 0.4),
    (400.0, 0.2),
    (410.0, 0.1),
    (420.0, 0.05),
    (430.0, 0.0),
    (440.0, 0.0),
    (450.0, 0.0),
];

/// Hardcoded B filter transmission data from the Zodical Light Curve - SpectrumPassbands.csv
/// Format: (wavelength_nm, transmission)
const B_FILTER_DATA: [(f64, f64); 23] = [
    (350.0, 0.0),
    (360.0, 0.05),
    (370.0, 0.1),
    (380.0, 0.2),
    (390.0, 0.35),
    (400.0, 0.55),
    (410.0, 0.7),
    (420.0, 0.85),
    (430.0, 0.95),
    (440.0, 1.0),
    (450.0, 1.0),
    (460.0, 0.95),
    (470.0, 0.85),
    (480.0, 0.7),
    (490.0, 0.55),
    (500.0, 0.4),
    (510.0, 0.3),
    (520.0, 0.2),
    (530.0, 0.1),
    (540.0, 0.05),
    (550.0, 0.0),
    (560.0, 0.0),
    (570.0, 0.0),
];

/// Hardcoded V filter transmission data from the Zodical Light Curve - SpectrumPassbands.csv
/// Format: (wavelength_nm, transmission)
const V_FILTER_DATA: [(f64, f64); 27] = [
    (450.0, 0.0),
    (460.0, 0.0),
    (470.0, 0.05),
    (480.0, 0.15),
    (490.0, 0.3),
    (500.0, 0.5),
    (510.0, 0.7),
    (520.0, 0.85),
    (530.0, 0.95),
    (540.0, 1.0),
    (550.0, 1.0),
    (560.0, 0.95),
    (570.0, 0.85),
    (580.0, 0.7),
    (590.0, 0.6),
    (600.0, 0.45),
    (610.0, 0.35),
    (620.0, 0.25),
    (630.0, 0.15),
    (640.0, 0.1),
    (650.0, 0.05),
    (660.0, 0.03),
    (670.0, 0.01),
    (680.0, 0.0),
    (690.0, 0.0),
    (700.0, 0.0),
    (710.0, 0.0),
];

/// Create a QuantumEfficiency object from a slice of (wavelength, transmission) tuples
///
/// # Arguments
///
/// * `data` - A slice of tuples containing (wavelength_nm, transmission)
///
/// # Returns
///
/// A Result containing a new QuantumEfficiency object
fn create_qe_from_data(
    data: &[(f64, f64)],
) -> Result<QuantumEfficiency, super::quantum_efficiency::QuantumEfficiencyError> {
    let wavelengths: Vec<f64> = data.iter().map(|(w, _)| *w).collect();
    let efficiencies: Vec<f64> = data.iter().map(|(_, e)| *e).collect();

    QuantumEfficiency::from_table(wavelengths, efficiencies)
}

/// Get a QuantumEfficiency object for the U filter
///
/// # Returns
///
/// A Result containing a new QuantumEfficiency object for the U filter
pub fn u_filter() -> Result<QuantumEfficiency, super::quantum_efficiency::QuantumEfficiencyError> {
    create_qe_from_data(&U_FILTER_DATA)
}

/// Get a QuantumEfficiency object for the B filter
///
/// # Returns
///
/// A Result containing a new QuantumEfficiency object for the B filter
pub fn b_filter() -> Result<QuantumEfficiency, super::quantum_efficiency::QuantumEfficiencyError> {
    create_qe_from_data(&B_FILTER_DATA)
}

/// Get a QuantumEfficiency object for the V filter
///
/// # Returns
///
/// A Result containing a new QuantumEfficiency object for the V filter
pub fn v_filter() -> Result<QuantumEfficiency, super::quantum_efficiency::QuantumEfficiencyError> {
    create_qe_from_data(&V_FILTER_DATA)
}

/// Get all three standard UBV filters at once
///
/// # Returns
///
/// A tuple containing Result<QuantumEfficiency> for each filter (U, B, V)
pub fn ubv_filters() -> (
    Result<QuantumEfficiency, super::quantum_efficiency::QuantumEfficiencyError>,
    Result<QuantumEfficiency, super::quantum_efficiency::QuantumEfficiencyError>,
    Result<QuantumEfficiency, super::quantum_efficiency::QuantumEfficiencyError>,
) {
    (u_filter(), b_filter(), v_filter())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_filter_creation() {
        // Test that all filters can be created successfully
        let u_result = u_filter();
        let b_result = b_filter();
        let v_result = v_filter();

        assert!(u_result.is_ok(), "U filter creation failed");
        assert!(b_result.is_ok(), "B filter creation failed");
        assert!(v_result.is_ok(), "V filter creation failed");
    }

    #[test]
    fn test_u_filter_wavelength_range() {
        let u = u_filter().unwrap();
        let band = u.band();

        // Check that the wavelength range matches our data
        assert_eq!(band.lower_nm, 250.0);
        assert_eq!(band.upper_nm, 450.0);

        // Check transmission at key wavelengths
        assert_eq!(u.at(250.0), 0.0);
        assert_eq!(u.at(350.0), 1.0);
        assert_eq!(u.at(450.0), 0.0);

        // Check interpolated value
        assert_relative_eq!(u.at(335.0), 0.9, epsilon = 0.1);
    }

    #[test]
    fn test_b_filter_wavelength_range() {
        let b = b_filter().unwrap();
        let band = b.band();

        // Check that the wavelength range matches our data
        assert_eq!(band.lower_nm, 350.0);
        assert_eq!(band.upper_nm, 570.0);

        // Check transmission at key wavelengths
        assert_eq!(b.at(350.0), 0.0);
        assert_eq!(b.at(440.0), 1.0);
        assert_eq!(b.at(450.0), 1.0);
        assert_eq!(b.at(570.0), 0.0);

        // Check interpolated value
        assert_relative_eq!(b.at(500.0), 0.4, epsilon = 0.01);
    }

    #[test]
    fn test_v_filter_wavelength_range() {
        let v = v_filter().unwrap();
        let band = v.band();

        // Check that the wavelength range matches our data
        assert_eq!(band.lower_nm, 450.0);
        assert_eq!(band.upper_nm, 710.0);

        // Check transmission at key wavelengths
        assert_eq!(v.at(450.0), 0.0);
        assert_eq!(v.at(540.0), 1.0);
        assert_eq!(v.at(550.0), 1.0);
        assert_eq!(v.at(710.0), 0.0);

        // Check interpolated value
        assert_relative_eq!(v.at(600.0), 0.45, epsilon = 0.01);
    }
}

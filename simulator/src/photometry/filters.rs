//! Johnson-Cousins UBV photometric system implementation for astronomical observations.
//!
//! This module provides high-fidelity implementations of the standard UBV broadband
//! filters used in stellar photometry since the 1950s. These filters form the
//! foundation of astronomical magnitude systems and enable precise stellar
//! classification, distance measurements, and astrophysical analysis.
//!
//! # Historical Context
//!
//! The UBV system was established by Harold Johnson and William Morgan at
//! Yerkes Observatory, later refined by Alan Cousins. It remains the primary
//! photometric standard for:
//!
//! - **Stellar classification**: Temperature and luminosity determination
//! - **Distance measurements**: Using color-magnitude diagrams
//! - **Extinction corrections**: Interstellar reddening analysis  
//! - **Synthetic photometry**: Calibrating space telescope observations
//!
//! # Filter Characteristics
//!
//! ## U Filter (Ultraviolet, ~365nm)
//! - **Effective wavelength**: 365nm
//! - **Bandwidth**: ~60nm (300-400nm)
//! - **Primary use**: Hot star detection, UV excess measurement
//! - **Atmospheric limitation**: Severely affected by Earth's atmosphere
//!
//! ## B Filter (Blue, ~445nm)  
//! - **Effective wavelength**: 445nm
//! - **Bandwidth**: ~90nm (400-500nm)
//! - **Primary use**: Blue magnitude, B-V color index
//! - **Standard reference**: Fundamental for stellar photometry
//!
//! ## V Filter (Visual, ~551nm)
//! - **Effective wavelength**: 551nm  
//! - **Bandwidth**: ~85nm (500-600nm)
//! - **Primary use**: Visual magnitude (closest to human eye response)
//! - **Calibration**: Tied to historical photographic magnitudes
//!
//! # Data Source
//!
//! Filter transmission curves are based on the authoritative data from the
//! "Zodiacal Light Curve - SpectrumPassbands.csv" dataset, providing accurate
//! representations of the standard Johnson-Cousins system.
//!
//! # Examples
//!
//! ## Basic Filter Usage
//! Create individual UBV filters and check transmission values
//! at specific wavelengths for photometric analysis.
//!
//! ## Color Index Calculation
//! Calculate B-V color indices from stellar spectra using standard
//! photometric filters for stellar classification.
//!
//! ## Complete UBV Photometry
//! Perform comprehensive UBV photometry for stellar population
//! analysis and temperature classification.

use super::{quantum_efficiency::QuantumEfficiencyError, QuantumEfficiency};

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

/// Create QuantumEfficiency object from filter transmission data.
///
/// Converts raw filter transmission tables into interpolatable QuantumEfficiency
/// objects suitable for photometric calculations. Handles validation and error
/// checking for malformed filter data.
///
/// # Arguments
/// * `data` - Slice of (wavelength_nm, transmission) tuples defining filter curve
///
/// # Returns
/// Result containing QuantumEfficiency object, or error if data is invalid
///
/// # Errors
/// Returns QuantumEfficiencyError if:
/// - Wavelength data is not monotonically increasing
/// - Transmission values are outside [0.0, 1.0] range
/// - Data contains NaN or infinite values
fn create_qe_from_data(data: &[(f64, f64)]) -> Result<QuantumEfficiency, QuantumEfficiencyError> {
    let wavelengths: Vec<f64> = data.iter().map(|(w, _)| *w).collect();
    let efficiencies: Vec<f64> = data.iter().map(|(_, e)| *e).collect();

    QuantumEfficiency::from_table(wavelengths, efficiencies)
}

/// Create Johnson U-band photometric filter (ultraviolet, ~365nm).
///
/// Returns the standard Johnson U filter with transmission curve optimized for
/// ultraviolet stellar photometry. Essential for detecting hot stars, measuring
/// UV excess in active galaxies, and studying stellar atmospheres.
///
/// # Filter Properties
/// - **Effective wavelength**: 365nm
/// - **FWHM bandwidth**: ~60nm  
/// - **Wavelength range**: 290-430nm
/// - **Peak transmission**: 1.0 at 350nm
/// - **Atmospheric cutoff**: <300nm (space-only)
///
/// # Returns
/// U-band QuantumEfficiency
///
/// # Usage
/// Standard Johnson U-band filter for UV stellar photometry
/// and hot star detection.
pub fn u_filter() -> QuantumEfficiency {
    create_qe_from_data(&U_FILTER_DATA).expect("U filter data is hardcoded and should never fail")
}

/// Create Johnson B-band photometric filter (blue, ~445nm).
///
/// Returns the standard Johnson B filter, the fundamental blue reference for
/// stellar photometry. Central to the B-V color index system and essential
/// for stellar classification and distance measurements.
///
/// # Filter Properties  
/// - **Effective wavelength**: 445nm
/// - **FWHM bandwidth**: ~90nm
/// - **Wavelength range**: 360-540nm  
/// - **Peak transmission**: 1.0 at 440-450nm
/// - **Standard reference**: Primary blue magnitude system
///
/// # Returns
/// B-band QuantumEfficiency
///
/// # Usage
/// Standard Johnson B-band filter for blue stellar photometry
/// and color index calculations.
pub fn b_filter() -> QuantumEfficiency {
    create_qe_from_data(&B_FILTER_DATA).expect("B filter data is hardcoded and should never fail")
}

/// Create Johnson V-band photometric filter (visual, ~551nm).
///
/// Returns the standard Johnson V filter, designed to match human eye sensitivity
/// and serve as the fundamental visual magnitude reference. Forms the basis of
/// stellar magnitude systems and photometric calibration.
///
/// # Filter Properties
/// - **Effective wavelength**: 551nm  
/// - **FWHM bandwidth**: ~85nm
/// - **Wavelength range**: 470-680nm
/// - **Peak transmission**: 1.0 at 540-550nm
/// - **Human eye match**: Closely approximates photopic vision
///
/// # Returns
/// V-band QuantumEfficiency
///
/// # Usage
/// Standard Johnson V-band filter matching human eye sensitivity
/// for visual magnitude reference and stellar classification.
pub fn v_filter() -> QuantumEfficiency {
    create_qe_from_data(&V_FILTER_DATA).expect("V filter data is hardcoded and should never fail")
}

/// Create complete Johnson UBV photometric filter set for multi-band observations.
///
/// Returns all three standard photometric filters simultaneously, enabling efficient
/// setup for comprehensive stellar photometry, color index calculations, and
/// spectral energy distribution analysis.
///
/// # Filter Set Applications
/// - **Color-color diagrams**: U-B vs B-V analysis for stellar classification
/// - **Extinction measurements**: Multi-band reddening determination  
/// - **Distance modulus**: Color-magnitude diagram fitting
/// - **Population synthesis**: Stellar population age and metallicity
/// - **Synthetic photometry**: Space telescope calibration and validation
///
/// # Returns
/// Tuple containing (U, B, V) QuantumEfficiency objects
///
/// # Usage
/// Complete UBV filter set for multi-band stellar photometry
/// and comprehensive color index analysis.
pub fn ubv_filters() -> (QuantumEfficiency, QuantumEfficiency, QuantumEfficiency) {
    (u_filter(), b_filter(), v_filter())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_filter_creation() {
        // Test that all filters can be created successfully
        let _u = u_filter();
        let _b = b_filter();
        let _v = v_filter();
        // If we get here, all filters were created successfully
    }

    #[test]
    fn test_u_filter_wavelength_range() {
        let u = u_filter();
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
        let b = b_filter();
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
        let v = v_filter();
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

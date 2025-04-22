//! Sensor configuration for simulating detector characteristics

#![allow(clippy::approx_constant)]

use once_cell::sync::Lazy;

use crate::photometry::quantum_efficiency::QuantumEfficiency;

/// Configuration for a sensor detector
#[derive(Debug, Clone)]
pub struct SensorConfig {
    /// Quantum efficiency as a function of wavelength (nm)
    pub quantum_efficiency: QuantumEfficiency,

    /// Width of sensor in pixels
    pub width_px: u32,

    /// Height of sensor in pixels
    pub height_px: u32,

    /// Pixel size in microns
    pub pixel_size_um: f64,

    /// Read noise in electrons per pixel
    pub read_noise_e: f64,

    /// Dark current in electrons per pixel per second
    pub dark_current_e_p_s: f64,

    /// Name/model of the sensor
    pub name: String,

    /// Bit depth of the sensor
    pub bit_depth: u8,

    /// DN (Digital Numbers) per electron (typically at the highest gain setting)
    pub dn_per_electron: f64,

    /// Max well depth in electrons
    pub max_well_depth_e: f64,
}

impl SensorConfig {
    /// Create a new sensor configuration
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        quantum_efficiency: QuantumEfficiency,
        width_px: u32,
        height_px: u32,
        pixel_size_um: f64,
        read_noise_e: f64,
        dark_current_e_p_s: f64,
        bit_depth: u8,
        dn_per_electron: f64,
        max_well_depth_e: f64,
    ) -> Self {
        Self {
            name: name.into(),
            quantum_efficiency,
            width_px,
            height_px,
            pixel_size_um,
            read_noise_e,
            dark_current_e_p_s,
            bit_depth,
            dn_per_electron,
            max_well_depth_e,
        }
    }

    /// Get quantum efficiency at specified wavelength (nm)
    pub fn qe_at_wavelength(&self, wavelength_nm: u32) -> f64 {
        // Convert u32 to f64 for our QuantumEfficiency type
        self.quantum_efficiency.at(wavelength_nm as f64)
    }

    /// Get sensor dimensions in microns
    pub fn dimensions_um(&self) -> (f64, f64) {
        (
            self.width_px as f64 * self.pixel_size_um,
            self.height_px as f64 * self.pixel_size_um,
        )
    }

    // TODO(meawoppl) - inline this as an attribute
    /// Estimate DN (Digital Numbers) per electron based on sensor characteristics
    ///
    /// This is an approximation that assumes the ADC bit depth is sufficient to
    /// capture the read noise with at least 2 bits of precision. In practice,
    /// the relationship between electrons and DN depends on gain settings and
    /// other factors specific to the camera implementation.
    pub fn dn_per_electron(&self) -> f64 {
        self.dn_per_electron
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_qe_interpolation() {
        // Create QE curve with five points, first and last must be 0 for QuantumEfficiency
        let wavelengths = vec![300.0, 400.0, 500.0, 600.0, 700.0];
        let efficiencies = vec![0.0, 0.4, 0.6, 0.5, 0.0];
        let qe = QuantumEfficiency::from_table(wavelengths, efficiencies).unwrap();

        let sensor = SensorConfig::new("Test", qe, 1024, 1024, 5.5, 2.0, 0.01, 8, 3.0, 1e20);

        // Exact matches (use approximate comparison for float values)
        assert_relative_eq!(sensor.qe_at_wavelength(400), 0.4, epsilon = 1e-5);
        assert_relative_eq!(sensor.qe_at_wavelength(500), 0.6, epsilon = 1e-5);
        assert_relative_eq!(sensor.qe_at_wavelength(600), 0.5, epsilon = 1e-5);

        // Interpolated values
        assert_relative_eq!(sensor.qe_at_wavelength(450), 0.5, epsilon = 1e-5);
        assert_relative_eq!(sensor.qe_at_wavelength(550), 0.55, epsilon = 1e-5);

        // Outside range (should be 0)
        assert_eq!(sensor.qe_at_wavelength(300), 0.0);
        assert_eq!(sensor.qe_at_wavelength(700), 0.0);
    }

    #[test]
    fn test_sensor_dimensions() {
        let qe = create_flat_qe(0.5);
        let sensor = SensorConfig::new("Test", qe, 1024, 768, 5.5, 2.0, 0.01, 8, 3.0, 1e20);
        let (width_um, height_um) = sensor.dimensions_um();

        assert_eq!(width_um, 1024.0 * 5.5);
        assert_eq!(height_um, 768.0 * 5.5);
    }
}

/// Create a simple flat QE function with constant efficiency across all wavelengths
pub fn create_flat_qe(efficiency: f64) -> QuantumEfficiency {
    // Convert to f32 for our QuantumEfficiency type
    // Create a simple QE curve with 0 at edges and constant value in visible range
    let wavelengths = vec![300.0, 400.0, 700.0, 800.0];
    let efficiencies = vec![0.0, efficiency, efficiency, 0.0];

    QuantumEfficiency::from_table(wavelengths, efficiencies)
        .expect("Failed to create flat QE curve")
}

/// Standard sensor models
pub mod models {
    use super::*;

    /// GSENSE4040BSI CMOS sensor with detailed QE curve from manufacturer data
    pub static GSENSE4040BSI: Lazy<SensorConfig> = Lazy::new(|| {
        // QE data from QE-gsense4040bsi.csv (trunced to 3 decimal places)
        // We'll use a subset of the data to keep the vector size reasonable
        // while still capturing the important features of the curve
        let wavelengths = vec![
            150.0, // Guess for 0?
            200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 230.0, 235.0, 240.0, 245.0, 250.0, 255.0,
            260.0, 265.0, 270.0, 275.0, 280.0, 285.0, 290.0, 295.0, 300.0, 305.0, 310.0, 315.0,
            320.0, 325.0, 330.0, 335.0, 340.0, 345.0, 350.0, 355.0, 360.0, 365.0, 370.0, 375.0,
            380.0, 385.0, 390.0, 395.0, 400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0,
            440.0, 445.0, 450.0, 455.0, 460.0, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0,
            500.0, 505.0, 510.0, 515.0, 520.0, 525.0, 530.0, 535.0, 540.0, 545.0, 550.0, 555.0,
            560.0, 565.0, 570.0, 575.0, 580.0, 585.0, 590.0, 595.0, 600.0, 605.0, 610.0, 615.0,
            620.0, 625.0, 630.0, 635.0, 640.0, 645.0, 650.0, 655.0, 660.0, 665.0, 670.0, 675.0,
            680.0, 685.0, 690.0, 695.0, 700.0, 705.0, 710.0, 715.0, 720.0, 725.0, 730.0, 735.0,
            740.0, 745.0, 750.0, 755.0, 760.0, 765.0, 770.0, 775.0, 780.0, 785.0, 790.0, 795.0,
            800.0, 805.0, 810.0, 815.0, 820.0, 825.0, 830.0, 835.0, 840.0, 845.0, 850.0, 855.0,
            860.0, 865.0, 870.0, 875.0, 880.0, 885.0, 890.0, 895.0, 900.0, 905.0, 910.0, 915.0,
            920.0, 925.0, 930.0, 935.0, 940.0, 945.0, 950.0, 955.0, 960.0, 965.0, 970.0, 975.0,
            980.0, 985.0, 990.0, 995.0, 1000.0, 1005.0, 1010.0, 1015.0, 1020.0, 1025.0, 1030.0,
            1035.0, 1040.0, 1045.0, 1050.0, 1055.0, 1060.0, 1065.0, 1070.0, 1075.0, 1080.0, 1085.0,
            1090.0, 1095.0, 1100.0, 1110.0,
        ];

        let efficiencies = vec![
            0.0, 0.394, 0.394, 0.402, 0.411, 0.403, 0.418, 0.451, 0.494, 0.518, 0.483, 0.479,
            0.486, 0.472, 0.444, 0.423, 0.405, 0.416, 0.447, 0.474, 0.500, 0.525, 0.539, 0.556,
            0.549, 0.560, 0.570, 0.556, 0.562, 0.572, 0.571, 0.565, 0.566, 0.552, 0.564, 0.580,
            0.609, 0.611, 0.643, 0.678, 0.688, 0.667, 0.685, 0.704, 0.742, 0.724, 0.774, 0.738,
            0.775, 0.778, 0.824, 0.819, 0.831, 0.843, 0.823, 0.850, 0.852, 0.846, 0.865, 0.886,
            0.871, 0.876, 0.868, 0.876, 0.861, 0.878, 0.869, 0.891, 0.871, 0.904, 0.897, 0.908,
            0.885, 0.901, 0.856, 0.842, 0.862, 0.893, 0.880, 0.893, 0.873, 0.876, 0.887, 0.888,
            0.884, 0.864, 0.869, 0.852, 0.842, 0.864, 0.838, 0.834, 0.832, 0.840, 0.836, 0.812,
            0.829, 0.792, 0.818, 0.814, 0.782, 0.775, 0.757, 0.753, 0.731, 0.730, 0.708, 0.683,
            0.677, 0.680, 0.674, 0.648, 0.650, 0.624, 0.636, 0.595, 0.594, 0.591, 0.566, 0.540,
            0.530, 0.532, 0.530, 0.509, 0.490, 0.488, 0.481, 0.477, 0.498, 0.433, 0.437, 0.455,
            0.383, 0.439, 0.354, 0.390, 0.338, 0.339, 0.337, 0.322, 0.318, 0.300, 0.312, 0.266,
            0.318, 0.252, 0.253, 0.229, 0.231, 0.225, 0.198, 0.213, 0.165, 0.204, 0.153, 0.174,
            0.142, 0.146, 0.140, 0.122, 0.124, 0.108, 0.103, 0.098, 0.089, 0.083, 0.074, 0.068,
            0.060, 0.054, 0.048, 0.042, 0.039, 0.034, 0.031, 0.028, 0.024, 0.022, 0.020, 0.017,
            0.015, 0.014, 0.0,
        ];

        let qe = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create GSENSE4040BSI QE curve");

        SensorConfig::new(
            "GSENSE4040BSI",
            qe,
            4096,
            4096,
            9.0,
            2.3,
            0.04,
            12,
            0.35,
            39_200.0,
        )
    });

    /// GSENSE6510BSI CMOS sensor with detailed QE curve from manufacturer chart found here
    /// https://www.gpixel.com/en/details_155.html
    pub static GSENSE6510BSI: Lazy<SensorConfig> = Lazy::new(|| {
        let wavelengths = vec![
            150.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
            310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0, 410.0, 420.0,
            430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0, 510.0, 520.0, 530.0, 540.0,
            550.0, 560.0, 570.0, 580.0, 590.0, 600.0, 610.0, 620.0, 630.0, 640.0, 650.0, 660.0,
            670.0, 680.0, 690.0, 700.0, 710.0, 720.0, 730.0, 740.0, 750.0, 760.0, 770.0, 780.0,
            790.0, 800.0, 810.0, 820.0, 830.0, 840.0, 850.0, 860.0, 870.0, 880.0, 890.0, 900.0,
            910.0, 920.0, 930.0, 940.0, 950.0, 960.0, 970.0, 980.0, 990.0, 1000.0, 1010.0, 1020.0,
            1030.0, 1040.0, 1050.0, 1060.0, 1070.0, 1080.0, 1090.0, 1100.0,
        ];

        let efficiencies = vec![
            0.0, 0.22, 0.25, 0.28, 0.31, 0.35, 0.38, 0.4, 0.38, 0.33, 0.35, 0.4, 0.36, 0.31, 0.35,
            0.39, 0.41, 0.39, 0.38, 0.44, 0.53, 0.6, 0.67, 0.73, 0.78, 0.82, 0.86, 0.88, 0.9, 0.92,
            0.93, 0.94, 0.95, 0.95, 0.96, 0.96, 0.96, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9,
            0.89, 0.88, 0.86, 0.84, 0.82, 0.8, 0.78, 0.75, 0.73, 0.7, 0.68, 0.65, 0.63, 0.6, 0.58,
            0.55, 0.53, 0.5, 0.48, 0.45, 0.43, 0.4, 0.38, 0.35, 0.33, 0.3, 0.28, 0.25, 0.23, 0.21,
            0.19, 0.17, 0.15, 0.13, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02,
            0.01, 0.01, 0.01, 0.0,
        ];

        let qe = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create GSENSE6510BSI QE curve");

        // NB: Dark current is spec'ed at -10°C
        SensorConfig::new(
            "GSENSE6510BSI",
            qe,
            3200,
            3200,
            6.5,
            0.7,
            0.2,
            12,
            0.35,
            21_000.0,
        )
    });

    /// HWK4123 CMOS sensor with detailed QE curve
    pub static HWK4123: Lazy<SensorConfig> = Lazy::new(|| {
        // Detailed QE curve data
        let wavelengths = vec![
            200.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0,
            360.0, 370.0, 380.0, 390.0, 400.0, 410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0,
            480.0, 490.0, 500.0, 510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0,
            600.0, 610.0, 620.0, 630.0, 640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0, 710.0,
            720.0, 730.0, 740.0, 750.0, 760.0, 770.0, 780.0, 790.0, 800.0, 810.0, 820.0, 830.0,
            840.0, 850.0, 860.0, 870.0, 880.0, 890.0, 900.0, 910.0, 920.0, 930.0, 940.0, 950.0,
            960.0, 970.0, 980.0, 990.0, 1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0,
        ];

        let efficiencies = vec![
            0.0, 0.01, 0.04, 0.08, 0.12, 0.14, 0.18, 0.22, 0.28, 0.33, 0.39, 0.45, 0.5, 0.55, 0.6,
            0.65, 0.7, 0.74, 0.78, 0.82, 0.85, 0.88, 0.9, 0.9, 0.9, 0.89, 0.88, 0.86, 0.85, 0.83,
            0.81, 0.8, 0.78, 0.76, 0.74, 0.72, 0.7, 0.68, 0.66, 0.65, 0.63, 0.62, 0.61, 0.6, 0.59,
            0.58, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46,
            0.45, 0.45, 0.45, 0.43, 0.4, 0.37, 0.34, 0.31, 0.28, 0.25, 0.23, 0.2, 0.18, 0.15, 0.13,
            0.11, 0.09, 0.07, 0.05, 0.04, 0.02, 0.01, 0.0,
        ];

        let qe = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create HWK4123 QE curve");

        // 7.42 DN/e- at 32x gain
        // 0.242 DN/e- at 1x gain (using 1x gain here)
        SensorConfig::new("HWK4123", qe, 4096, 2300, 4.6, 0.25, 0.1, 12, 7.42, 7_500.0)
    });

    /// Sony IMX455 Full-frame BSI CMOS sensor
    /// Data from: "Characterization of Sony IMX455 sensor for astronomical applications"
    /// https://arxiv.org/pdf/2207.13052
    pub static IMX455: Lazy<SensorConfig> = Lazy::new(|| {
        // QE curve from manufacturer data
        // Note: We already have zero at the endpoints as required by QuantumEfficiency
        let wavelengths = vec![
            300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 520.0,
            540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0, 720.0, 740.0, 760.0,
            780.0, 800.0, 820.0, 840.0, 860.0, 880.0, 900.0, 920.0, 940.0, 960.0, 980.0, 1000.0,
        ];

        let efficiencies = vec![
            0.0, 0.05, 0.05, 0.12, 0.22, 0.35, 0.52, 0.68, 0.82, 0.90, 0.94, 0.94, 0.92, 0.86,
            0.80, 0.72, 0.64, 0.56, 0.48, 0.42, 0.36, 0.30, 0.25, 0.22, 0.18, 0.16, 0.14, 0.12,
            0.10, 0.09, 0.08, 0.06, 0.05, 0.04, 0.03, 0.0,
        ];

        let qe = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create IMX455 QE curve");

        // Max well depth is from here:
        // https://player-one-astronomy.com/product/zeus-455m-pro-imx455-usb3-0-mono-cooled-camera/

        SensorConfig::new(
            "IMX455", qe, 9568, 6380, 3.75,  // Pixel pitch in microns
            2.67,  // Read noise in electrons (from arxiv paper)
            0.002, // Dark current in e-/px/s at -20°C (from arxiv paper)
            16, 0.343, 71_600.0,
        )
    });
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn test_predefined_sensors() {
        // Check GSENSE4040BSI properties
        assert_eq!(models::GSENSE4040BSI.name, "GSENSE4040BSI");
        assert_eq!(models::GSENSE4040BSI.width_px, 4096);
        assert_eq!(models::GSENSE4040BSI.height_px, 4096);
        assert_eq!(models::GSENSE4040BSI.pixel_size_um, 9.0);
        assert_eq!(models::GSENSE4040BSI.read_noise_e, 2.3);
        assert_eq!(models::GSENSE4040BSI.dark_current_e_p_s, 0.04);
        // QE should be close to 0.9 at 550nm for this sensor
        assert!(models::GSENSE4040BSI.qe_at_wavelength(550) > 0.85);

        // Check GSENSE6510BSI properties
        assert_eq!(models::GSENSE6510BSI.name, "GSENSE6510BSI");
        assert_eq!(models::GSENSE6510BSI.width_px, 3200);
        assert_eq!(models::GSENSE6510BSI.height_px, 3200);
        assert_eq!(models::GSENSE6510BSI.pixel_size_um, 6.5);
        assert_eq!(models::GSENSE6510BSI.read_noise_e, 0.7);
        assert_eq!(models::GSENSE6510BSI.dark_current_e_p_s, 0.2);
        // QE should peak around 520-550nm for this sensor
        assert!(models::GSENSE6510BSI.qe_at_wavelength(550) > 0.95);

        // Check HWK4123 properties
        assert_eq!(models::HWK4123.name, "HWK4123");
        assert_eq!(models::HWK4123.width_px, 4096);
        assert_eq!(models::HWK4123.height_px, 2300);
        assert_eq!(models::HWK4123.pixel_size_um, 4.6);
        assert_eq!(models::HWK4123.read_noise_e, 0.25);
        assert_eq!(models::HWK4123.dark_current_e_p_s, 0.1);
        // QE should be close to 0.8 at 550nm for this sensor
        assert!(models::HWK4123.qe_at_wavelength(550) > 0.75);

        // Check IMX455 properties
        assert_eq!(models::IMX455.name, "IMX455");
        assert_eq!(models::IMX455.width_px, 9568);
        assert_eq!(models::IMX455.height_px, 6380);
        assert_eq!(models::IMX455.pixel_size_um, 3.75);
        assert_eq!(models::IMX455.read_noise_e, 2.67);
        assert_eq!(models::IMX455.dark_current_e_p_s, 0.002);

        // Get actual QE value to print for debugging
        let qe_400 = models::IMX455.qe_at_wavelength(400);
        let qe_500 = models::IMX455.qe_at_wavelength(500);
        let qe_700 = models::IMX455.qe_at_wavelength(700);

        // Just do basic checks to make sure IMX455 QE curve is reasonable
        assert!(qe_400 >= 0.0);
        assert!(qe_500 >= 0.0);
        assert!(qe_700 >= 0.0);

        // The peak QE should be around 500nm
        assert!(qe_500 > qe_400);
        assert!(qe_500 > qe_700);
    }
}

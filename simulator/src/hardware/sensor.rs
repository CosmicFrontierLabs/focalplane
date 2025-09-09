//! Sensor configuration and models for astronomical detector simulation.
//!
//! This module provides comprehensive models for CCD and CMOS image sensors
//! used in astronomical applications. It includes realistic quantum efficiency
//! curves, noise characteristics, and thermal properties for accurate
//! photometric and detection simulations.
//!
//! # Key Features
//!
//! - **Spectral Response**: Detailed quantum efficiency curves from manufacturer data
//! - **Noise Modeling**: Read noise, dark current, and temperature dependencies
//! - **Geometry**: Pixel size, array dimensions, and field-of-view calculations
//! - **Electronics**: ADC characteristics, gain settings, and well depth limits
//! - **Performance**: Frame rate and dynamic range specifications
//!
//! # Sensor Models
//!
//! The module includes several real-world sensor configurations:
//! - **GSENSE4040BSI**: High-performance 4K scientific CMOS (9μm pixels)
//! - **GSENSE6510BSI**: Optimized for visible light (6.5μm pixels, low noise)
//! - **HWK4123**: Wide-format sensor with excellent NIR response
//! - **IMX455**: Full-frame Sony sensor popular in astronomy
//!
//! # Physics Models
//!
//! ## Quantum Efficiency
//! Wavelength-dependent conversion efficiency from photons to electrons,
//! incorporating:
//! - Anti-reflection coatings
//! - Silicon absorption characteristics  
//! - Microlens efficiency
//! - Color filter effects (for color sensors)
//!
//! ## Noise Sources
//! - **Read Noise**: Electronic noise from amplifiers and ADC
//! - **Dark Current**: Thermal generation with temperature dependence
//! - **Shot Noise**: Poisson statistics from photon arrival
//!

#![allow(clippy::approx_constant)]

use once_cell::sync::Lazy;
use std::fmt;

use crate::hardware::dark_current::DarkCurrentEstimator;
use crate::hardware::read_noise::ReadNoiseEstimator;
use crate::photometry::quantum_efficiency::QuantumEfficiency;
use crate::units::{Area, AreaExt, Length, LengthExt, Temperature, TemperatureExt};

/// Sensor dimensions in pixels and physical size
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorGeometry {
    /// Width in pixels
    width: usize,
    /// Height in pixels  
    height: usize,
    /// Physical pixel size (square pixels assumed)
    pixel_size: Length,
}

impl SensorGeometry {
    /// Create new sensor dimensions
    pub fn of_width_height(width: usize, height: usize, pixel_size: Length) -> Self {
        Self {
            width,
            height,
            pixel_size,
        }
    }

    /// Get width and height in pixels as tuple
    pub fn get_pixel_width_height(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Get width and height in physical units
    pub fn get_width_height(&self) -> (Length, Length) {
        let width_length =
            Length::from_micrometers(self.width as f64 * self.pixel_size.as_micrometers());
        let height_length =
            Length::from_micrometers(self.height as f64 * self.pixel_size.as_micrometers());
        (width_length, height_length)
    }

    /// Get pixel size
    pub fn pixel_size(&self) -> Length {
        self.pixel_size
    }

    /// Get total number of pixels
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Get aspect ratio (width/height)
    pub fn aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }

    /// Get total area of the sensor
    pub fn total_area(&self) -> Area {
        let (width, height) = self.get_width_height();
        Area::from_square_meters(width.as_meters() * height.as_meters())
    }
}

impl fmt::Display for SensorGeometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}×{} pixels ({:.1}μm pitch)",
            self.width,
            self.height,
            self.pixel_size.as_micrometers()
        )
    }
}

/// Complete sensor configuration for astronomical detector simulation.
///
/// Represents a digital image sensor (CCD or CMOS) with all physical and
/// electronic characteristics needed for realistic astronomical observations.
/// Combines spectral response, noise properties, geometry, and performance
/// parameters into a unified model.
///
/// # Key Capabilities
///
/// - **Spectral modeling**: Wavelength-dependent quantum efficiency
/// - **Noise simulation**: Read noise, dark current, and shot noise
/// - **Geometric calculations**: Pixel scale, sensor dimensions, field of view
/// - **Dynamic range**: Well depth, bit depth, and gain characteristics
/// - **Thermal modeling**: Temperature-dependent dark current
///
#[derive(Debug, Clone)]
pub struct SensorConfig {
    /// Wavelength-dependent quantum efficiency curve from manufacturer data
    pub quantum_efficiency: QuantumEfficiency,

    /// Sensor dimensions (width, height, and pixel size)
    pub dimensions: SensorGeometry,

    /// Read noise estimator for temperature and exposure-dependent noise
    pub read_noise_estimator: ReadNoiseEstimator,

    /// Temperature-dependent dark current model
    pub dark_current_estimator: DarkCurrentEstimator,

    /// Sensor model name or identifier
    pub name: String,

    /// ADC bit depth (8, 12, 14, 16 bits typical)
    pub bit_depth: u8,

    /// Conversion gain in DN per electron (camera-dependent)
    pub dn_per_electron: f64,

    /// Full well capacity in electrons (saturation limit)
    pub max_well_depth_e: f64,

    /// Maximum sustainable frame rate in Hz
    pub max_frame_rate_fps: f64,
}

impl SensorConfig {
    /// Create a new sensor configuration
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        quantum_efficiency: QuantumEfficiency,
        geometry: SensorGeometry,
        read_noise_estimator: ReadNoiseEstimator,
        dark_current_estimator: DarkCurrentEstimator,
        bit_depth: u8,
        dn_per_electron: f64,
        max_well_depth_e: f64,
        max_frame_rate_fps: f64,
    ) -> Self {
        Self {
            name: name.into(),
            quantum_efficiency,
            dimensions: geometry,
            read_noise_estimator,
            dark_current_estimator,
            bit_depth,
            dn_per_electron,
            max_well_depth_e,
            max_frame_rate_fps,
        }
    }

    /// Create a duplicate sensor configuration with new dimensions
    pub fn with_dimensions(&self, width_px: usize, height_px: usize) -> Self {
        let mut clone = self.clone();
        clone.dimensions =
            SensorGeometry::of_width_height(width_px, height_px, self.dimensions.pixel_size());
        clone
    }

    /// Get quantum efficiency at specified wavelength (nm)
    pub fn qe_at_wavelength(&self, wavelength_nm: u32) -> f64 {
        // Convert u32 to Wavelength for our QuantumEfficiency type
        use crate::units::{Length, LengthExt};
        self.quantum_efficiency
            .at(Length::from_nanometers(wavelength_nm as f64))
    }

    /// Get dark current at a specific temperature in electrons/pixel/second
    pub fn dark_current_at_temperature(&self, temperature: Temperature) -> f64 {
        self.dark_current_estimator
            .estimate_at_temperature(temperature)
            .expect("Temperature out of interpolation range")
    }

    /// Get the pixel size
    pub fn pixel_size(&self) -> Length {
        self.dimensions.pixel_size()
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

        let geometry = SensorGeometry::of_width_height(1024, 1024, Length::from_micrometers(5.5));
        let sensor = SensorConfig::new(
            "Test",
            qe,
            geometry,
            ReadNoiseEstimator::constant(2.0),
            DarkCurrentEstimator::from_reference_point(0.01, Temperature::from_celsius(20.0)),
            8,
            3.0,
            1e20,
            30.0,
        );

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
        let geometry = SensorGeometry::of_width_height(1024, 768, Length::from_micrometers(5.5));
        let sensor = SensorConfig::new(
            "Test",
            qe,
            geometry,
            ReadNoiseEstimator::constant(2.0),
            DarkCurrentEstimator::from_reference_point(0.01, Temperature::from_celsius(20.0)),
            8,
            3.0,
            1e20,
            30.0,
        );
        let (width, height) = sensor.dimensions.get_width_height();

        assert!((width.as_micrometers() - 1024.0 * 5.5).abs() < 1e-10);
        assert!((height.as_micrometers() - 768.0 * 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_sensor_geometry_creation() {
        let geometry = SensorGeometry::of_width_height(1920, 1080, Length::from_micrometers(3.45));

        // Test basic accessors
        assert_eq!(geometry.get_pixel_width_height(), (1920, 1080));
        assert_eq!(geometry.pixel_size(), Length::from_micrometers(3.45));
    }

    #[test]
    fn test_sensor_geometry_physical_dimensions() {
        let pixel_size = Length::from_micrometers(5.5);
        let geometry = SensorGeometry::of_width_height(2048, 1536, pixel_size);

        let (width, height) = geometry.get_width_height();

        // Expected physical dimensions
        let expected_width = Length::from_micrometers(2048.0 * 5.5);
        let expected_height = Length::from_micrometers(1536.0 * 5.5);

        assert_relative_eq!(
            width.as_micrometers(),
            expected_width.as_micrometers(),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            height.as_micrometers(),
            expected_height.as_micrometers(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_sensor_geometry_pixel_count() {
        let geometry = SensorGeometry::of_width_height(4096, 3072, Length::from_micrometers(9.0));
        assert_eq!(geometry.pixel_count(), 4096 * 3072);

        // Test small sensor
        let small = SensorGeometry::of_width_height(640, 480, Length::from_micrometers(5.0));
        assert_eq!(small.pixel_count(), 640 * 480);
    }

    #[test]
    fn test_sensor_geometry_aspect_ratio() {
        // 16:9 aspect ratio
        let widescreen = SensorGeometry::of_width_height(1920, 1080, Length::from_micrometers(3.0));
        assert_relative_eq!(widescreen.aspect_ratio(), 16.0 / 9.0, epsilon = 1e-10);

        // 4:3 aspect ratio
        let standard = SensorGeometry::of_width_height(1024, 768, Length::from_micrometers(5.0));
        assert_relative_eq!(standard.aspect_ratio(), 4.0 / 3.0, epsilon = 1e-10);

        // Square sensor
        let square = SensorGeometry::of_width_height(2048, 2048, Length::from_micrometers(6.5));
        assert_relative_eq!(square.aspect_ratio(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sensor_geometry_total_area() {
        let geometry = SensorGeometry::of_width_height(1000, 1000, Length::from_micrometers(10.0));

        // Total area should be (1000 * 10μm) * (1000 * 10μm) = 10mm * 10mm = 100mm² = 1cm²
        let area = geometry.total_area();
        assert_relative_eq!(area.as_square_centimeters(), 1.0, epsilon = 1e-10);

        // Test another case
        let geometry2 = SensorGeometry::of_width_height(2000, 1500, Length::from_micrometers(5.0));
        // Total area = (2000 * 5μm) * (1500 * 5μm) = 10mm * 7.5mm = 75mm² = 0.75cm²
        let area2 = geometry2.total_area();
        assert_relative_eq!(area2.as_square_centimeters(), 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_sensor_geometry_display() {
        let geometry = SensorGeometry::of_width_height(4096, 2160, Length::from_micrometers(3.76));
        let display_str = format!("{geometry}");
        assert_eq!(display_str, "4096×2160 pixels (3.8μm pitch)");

        // Test with different precision
        let geometry2 = SensorGeometry::of_width_height(1920, 1080, Length::from_micrometers(5.5));
        let display_str2 = format!("{geometry2}");
        assert_eq!(display_str2, "1920×1080 pixels (5.5μm pitch)");
    }

    #[test]
    fn test_sensor_geometry_equality() {
        let geom1 = SensorGeometry::of_width_height(1024, 768, Length::from_micrometers(5.5));
        let geom2 = SensorGeometry::of_width_height(1024, 768, Length::from_micrometers(5.5));
        let geom3 = SensorGeometry::of_width_height(1024, 768, Length::from_micrometers(6.0));
        let geom4 = SensorGeometry::of_width_height(2048, 768, Length::from_micrometers(5.5));

        assert_eq!(geom1, geom2);
        assert_ne!(geom1, geom3); // Different pixel size
        assert_ne!(geom1, geom4); // Different width
    }

    #[test]
    fn test_sensor_geometry_copy_clone() {
        let original = SensorGeometry::of_width_height(3840, 2160, Length::from_micrometers(2.9));

        // Test Copy
        let copied = original;
        assert_eq!(copied.get_pixel_width_height(), (3840, 2160));
        assert_eq!(original.get_pixel_width_height(), (3840, 2160)); // Original still valid due to Copy

        // Test Clone
        let cloned = original;
        assert_eq!(cloned.get_pixel_width_height(), (3840, 2160));
        assert_eq!(cloned.pixel_size(), Length::from_micrometers(2.9));
    }

    #[test]
    fn test_with_dimensions() {
        use std::time::Duration;
        let qe = create_flat_qe(0.5);
        let geometry = SensorGeometry::of_width_height(1024, 768, Length::from_micrometers(5.5));
        let original = SensorConfig::new(
            "Test",
            qe,
            geometry,
            ReadNoiseEstimator::constant(2.0),
            DarkCurrentEstimator::from_reference_point(0.01, Temperature::from_celsius(20.0)),
            8,
            3.0,
            1e20,
            30.0,
        );

        // Create resized sensor
        let resized = original.with_dimensions(2048, 1536);

        // Check new dimensions
        assert_eq!(resized.dimensions.get_pixel_width_height(), (2048, 1536));

        // Check dimensions in microns
        let (width, height) = resized.dimensions.get_width_height();
        assert!((width.as_micrometers() - 2048.0 * 5.5).abs() < 1e-10);
        assert!((height.as_micrometers() - 1536.0 * 5.5).abs() < 1e-10);

        // Verify other properties remain the same
        assert_eq!(resized.name, original.name);
        assert_eq!(resized.pixel_size(), original.pixel_size());
        // Read noise estimator should be cloned properly
        assert_eq!(
            resized
                .read_noise_estimator
                .estimate(20.0, Duration::from_secs_f64(0.2))
                .unwrap(),
            original
                .read_noise_estimator
                .estimate(20.0, Duration::from_secs_f64(0.2))
                .unwrap()
        );
        assert_eq!(
            resized.dark_current_estimator,
            original.dark_current_estimator
        );
        assert_eq!(resized.bit_depth, original.bit_depth);
        assert_eq!(resized.dn_per_electron, original.dn_per_electron);
        assert_eq!(resized.max_well_depth_e, original.max_well_depth_e);
        assert_eq!(resized.max_frame_rate_fps, original.max_frame_rate_fps);

        // Verify QE stays the same
        assert_relative_eq!(resized.qe_at_wavelength(500), 0.5, epsilon = 1e-5);
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

        let geometry = SensorGeometry::of_width_height(4096, 4096, Length::from_micrometers(9.0));
        SensorConfig::new(
            "GSENSE4040BSI",
            qe,
            geometry,
            ReadNoiseEstimator::constant(2.3),
            DarkCurrentEstimator::from_reference_point(0.04, Temperature::from_celsius(-40.0)), // 0.04 e-/px/s at -40°C
            12,
            0.35,
            39_200.0,
            24.0,
        )
    });

    /// GSENSE6510BSI CMOS sensor with detailed QE curve from manufacturer chart found here
    /// <https://www.gpixel.com/en/details_155.html>
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

        let geometry = SensorGeometry::of_width_height(3200, 3200, Length::from_micrometers(6.5));
        SensorConfig::new(
            "GSENSE6510BSI",
            qe,
            geometry,
            ReadNoiseEstimator::constant(0.7),
            DarkCurrentEstimator::from_reference_point(0.2, Temperature::from_celsius(-10.0)), // 0.2 e-/px/s at -10°C
            12,
            0.35,
            21_000.0,
            88.0,
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
        let geometry = SensorGeometry::of_width_height(4096, 2300, Length::from_micrometers(4.6));
        SensorConfig::new(
            "HWK4123",
            qe,
            geometry,
            ReadNoiseEstimator::hwk4123(),
            DarkCurrentEstimator::from_two_points(
                Temperature::from_celsius(-20.0),
                0.0198,
                Temperature::from_celsius(0.0),
                0.1,
            ), // 0.0198 e-/px/s at -20°C, 0.1 e-/px/s at 0°C
            12,
            7.42,
            7_500.0,
            120.0,
        )
    });

    /// Sony IMX455 Full-frame BSI CMOS sensor
    /// Data from: "Characterization of Sony IMX455 sensor for astronomical applications"
    /// Sources:
    /// <https://arxiv.org/pdf/2207.13052>
    /// <https://www.qhyccd.com/scientific-camera-qhy600pro-imx455/>
    ///
    /// NOTE: We are using parameter here consistent with the gain setting of "60" in "High gain mode"
    /// which is just past the knee where the read noise is very low, and well depth is still broad enough to
    /// support a wide range of source brightnesses. See this document for more details:
    /// <https://docs.google.com/spreadsheets/d/16WdFvMo3rj3Z9252pq32agsLV-wm7YNacvqfkEgSOAI/edit?gid=1094380256#gid=1094380256>
    /// QE data from 300-400nm is from Ajay/Tohovavohu, past that is the QXY measurements
    pub static IMX455: Lazy<SensorConfig> = Lazy::new(|| {
        // QE curve from manufacturer data
        // Note: We already have zero at the endpoints as required by QuantumEfficiency
        let wavelengths = vec![
            300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 401.689, 415.2006, 423.6453, 431.2456,
            445.6017, 460.8023, 472.2027, 489.5144, 500.4926, 522.8712, 581.14, 610.6967, 644.4757,
            672.7657, 682.4771, 691.7664, 697.6777, 718.3673, 733.1457, 741.1682, 754.2576,
            774.1027, 799.0148, 809.993, 826.038, 837.4384, 857.2836, 872.9064, 886.8403, 902.4631,
            923.9972, 941.7312, 959.0429, 982.6882, 1000.8445,
        ];

        let efficiencies = vec![
            0.0, 0.05, 0.05, 0.12, 0.22, 0.35, 0.7186, 0.8003, 0.8428, 0.8711, 0.9057, 0.9198,
            0.9088, 0.8868, 0.9057, 0.8774, 0.7704, 0.684, 0.6053, 0.5393, 0.5173, 0.4921, 0.4654,
            0.423, 0.4025, 0.3852, 0.3601, 0.327, 0.2846, 0.2799, 0.2516, 0.2437, 0.1965, 0.1934,
            0.1509, 0.1557, 0.1148, 0.1148, 0.0723, 0.0692, 0.0,
        ];

        let qe = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create IMX455 QE curve");

        // Max well depth is from here:
        // https://player-one-astronomy.com/product/zeus-455m-pro-imx455-usb3-0-mono-cooled-camera/

        let geometry = SensorGeometry::of_width_height(9568, 6380, Length::from_micrometers(3.76));
        SensorConfig::new(
            "IMX455",
            qe,
            geometry,
            ReadNoiseEstimator::constant(1.58),
            DarkCurrentEstimator::from_curve(
                vec![-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
                vec![
                    0.0022, 0.0032, 0.0046, 0.0068, 0.0105, 0.0357, 0.0675, 0.1231, 0.2208,
                ],
            ), // Measured dark current data for IMX455
            16,
            1.0 / 0.4,
            26_000.0,
            21.33,
        )
    });

    /// Collection of all available sensor models
    pub static ALL_SENSORS: Lazy<Vec<SensorConfig>> = Lazy::new(|| {
        vec![
            GSENSE4040BSI.clone(),
            GSENSE6510BSI.clone(),
            HWK4123.clone(),
            IMX455.clone(),
        ]
    });
}

#[cfg(test)]
mod model_tests {
    use approx::assert_relative_eq;
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_predefined_sensors() {
        // Check GSENSE4040BSI properties
        assert_eq!(models::GSENSE4040BSI.name, "GSENSE4040BSI");
        assert_eq!(
            models::GSENSE4040BSI.dimensions.get_pixel_width_height(),
            (4096, 4096)
        );
        assert_eq!(
            models::GSENSE4040BSI.pixel_size(),
            Length::from_micrometers(9.0)
        );
        // Check read noise at room temperature with 1s exposure
        assert_eq!(
            models::GSENSE4040BSI
                .read_noise_estimator
                .estimate(20.0, Duration::from_secs_f64(0.2))
                .unwrap(),
            2.3
        );
        assert_relative_eq!(
            models::GSENSE4040BSI.dark_current_at_temperature(Temperature::from_celsius(-40.0)),
            0.04,
            epsilon = 1e-6
        );
        assert_eq!(models::GSENSE4040BSI.max_frame_rate_fps, 24.0);
        // QE should be close to 0.9 at 550nm for this sensor
        assert!(models::GSENSE4040BSI.qe_at_wavelength(550) > 0.85);

        // Check GSENSE6510BSI properties
        assert_eq!(models::GSENSE6510BSI.name, "GSENSE6510BSI");
        assert_eq!(
            models::GSENSE6510BSI.dimensions.get_pixel_width_height(),
            (3200, 3200)
        );
        assert_eq!(
            models::GSENSE6510BSI.pixel_size(),
            Length::from_micrometers(6.5)
        );
        // Check read noise at room temperature with 1s exposure
        assert_eq!(
            models::GSENSE6510BSI
                .read_noise_estimator
                .estimate(20.0, Duration::from_secs_f64(0.2))
                .unwrap(),
            0.7
        );
        assert_relative_eq!(
            models::GSENSE6510BSI.dark_current_at_temperature(Temperature::from_celsius(-10.0)),
            0.2,
            epsilon = 1e-6
        );
        assert_eq!(models::GSENSE6510BSI.max_frame_rate_fps, 88.0);
        // QE should peak around 520-550nm for this sensor
        assert!(models::GSENSE6510BSI.qe_at_wavelength(550) > 0.95);

        // Check HWK4123 properties
        assert_eq!(models::HWK4123.name, "HWK4123");
        assert_eq!(
            models::HWK4123.dimensions.get_pixel_width_height(),
            (4096, 2300)
        );
        assert_eq!(models::HWK4123.pixel_size(), Length::from_micrometers(4.6));
        // Check read noise at room temperature with 1s exposure
        // For HWK4123, this should be ~0.301 based on the calibration data at 20°C, 5Hz
        let hwk_read_noise = models::HWK4123
            .read_noise_estimator
            .estimate(20.0, Duration::from_secs_f64(1.0 / 5.0))
            .unwrap();
        assert!((hwk_read_noise - 0.301).abs() < 0.01);
        assert_relative_eq!(
            models::HWK4123.dark_current_at_temperature(Temperature::from_celsius(0.0)),
            0.1,
            epsilon = 1e-6
        );
        // HWK4123 now uses different doubling rate from two reference points
        // Just verify it's in reasonable range
        let hwk_dc_20 =
            models::HWK4123.dark_current_at_temperature(Temperature::from_celsius(20.0));
        assert!(
            hwk_dc_20 > 0.4 && hwk_dc_20 < 0.6,
            "HWK4123 dark current at 20°C: {hwk_dc_20}"
        );
        assert_eq!(models::HWK4123.max_frame_rate_fps, 120.0);
        // QE should be close to 0.8 at 550nm for this sensor
        assert!(models::HWK4123.qe_at_wavelength(550) > 0.75);

        // Check IMX455 properties
        assert_eq!(models::IMX455.name, "IMX455");
        assert_eq!(
            models::IMX455.dimensions.get_pixel_width_height(),
            (9568, 6380)
        );
        assert_eq!(models::IMX455.pixel_size(), Length::from_micrometers(3.76));
        // Check read noise at room temperature with 0.2s exposure
        assert_eq!(
            models::IMX455
                .read_noise_estimator
                .estimate(20.0, Duration::from_secs_f64(0.2))
                .unwrap(),
            1.58
        );
        assert_relative_eq!(
            models::IMX455.dark_current_at_temperature(Temperature::from_celsius(-20.0)),
            0.0022,
            epsilon = 1e-6
        );
        assert_eq!(models::IMX455.max_frame_rate_fps, 21.33);

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

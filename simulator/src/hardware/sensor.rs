//! Sensor configuration for simulating detector characteristics

use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Quantum Efficiency function represented as a mapping of wavelength (nm) to efficiency (0.0-1.0)
pub type QEFunction = HashMap<u32, f64>;

/// Configuration for a sensor detector
#[derive(Debug, Clone)]
pub struct SensorConfig {
    /// Quantum efficiency as a function of wavelength (nm)
    pub quantum_efficiency: QEFunction,
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
}

impl SensorConfig {
    /// Create a new sensor configuration
    pub fn new(
        name: impl Into<String>,
        quantum_efficiency: QEFunction,
        width_px: u32,
        height_px: u32,
        pixel_size_um: f64,
        read_noise_e: f64,
        dark_current_e_p_s: f64,
    ) -> Self {
        Self {
            name: name.into(),
            quantum_efficiency,
            width_px,
            height_px,
            pixel_size_um,
            read_noise_e,
            dark_current_e_p_s,
        }
    }

    /// Get quantum efficiency at specified wavelength (nm)
    /// Uses linear interpolation between nearest wavelength points
    pub fn qe_at_wavelength(&self, wavelength_nm: u32) -> f64 {
        // Handle exact matches
        if let Some(qe) = self.quantum_efficiency.get(&wavelength_nm) {
            return *qe;
        }

        // Find nearest points for interpolation
        let wavelengths: Vec<u32> = self.quantum_efficiency.keys().copied().collect();
        if wavelengths.is_empty() {
            return 0.0;
        }

        // Find closest wavelength points below and above target
        let below = wavelengths.iter().filter(|&&w| w < wavelength_nm).max();
        let above = wavelengths.iter().filter(|&&w| w > wavelength_nm).min();

        match (below, above) {
            (Some(&w_below), Some(&w_above)) => {
                // Interpolate between nearest points
                let qe_below = self.quantum_efficiency[&w_below];
                let qe_above = self.quantum_efficiency[&w_above];
                let range = w_above as f64 - w_below as f64;
                let factor = (wavelength_nm as f64 - w_below as f64) / range;
                qe_below + factor * (qe_above - qe_below)
            }
            (Some(&w_below), None) => {
                // Extrapolate from closest point (below)
                self.quantum_efficiency[&w_below]
            }
            (None, Some(&w_above)) => {
                // Extrapolate from closest point (above)
                self.quantum_efficiency[&w_above]
            }
            (None, None) => 0.0, // Empty QE function
        }
    }

    /// Get sensor dimensions in microns
    pub fn dimensions_um(&self) -> (f64, f64) {
        (
            self.width_px as f64 * self.pixel_size_um,
            self.height_px as f64 * self.pixel_size_um,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qe_interpolation() {
        let mut qe = HashMap::new();
        qe.insert(400, 0.4);
        qe.insert(500, 0.6);
        qe.insert(600, 0.5);

        let sensor = SensorConfig::new("Test", qe, 1024, 1024, 5.5, 2.0, 0.01);

        // Exact matches
        assert_eq!(sensor.qe_at_wavelength(400), 0.4);
        assert_eq!(sensor.qe_at_wavelength(500), 0.6);
        assert_eq!(sensor.qe_at_wavelength(600), 0.5);

        // Interpolated values
        assert_eq!(sensor.qe_at_wavelength(450), 0.5);
        assert_eq!(sensor.qe_at_wavelength(550), 0.55);

        // Extrapolated (edge cases)
        assert_eq!(sensor.qe_at_wavelength(300), 0.4);
        assert_eq!(sensor.qe_at_wavelength(700), 0.5);
    }

    #[test]
    fn test_sensor_dimensions() {
        let mut qe = HashMap::new();
        qe.insert(500, 0.5);

        let sensor = SensorConfig::new("Test", qe, 1024, 768, 5.5, 2.0, 0.01);
        let (width_um, height_um) = sensor.dimensions_um();

        assert_eq!(width_um, 1024.0 * 5.5);
        assert_eq!(height_um, 768.0 * 5.5);
    }
}

/// Create a simple flat QE function with constant efficiency across all wavelengths
pub fn create_flat_qe(efficiency: f64) -> QEFunction {
    let mut qe = HashMap::new();
    // Add typical visible light wavelength range (400-700nm)
    qe.insert(400, efficiency);
    qe.insert(700, efficiency);
    qe
}

/// Standard sensor models
pub mod models {
    use super::*;

    /// GSENSE4040BSI CMOS sensor
    pub static GSENSE4040BSI: Lazy<SensorConfig> = Lazy::new(|| {
        SensorConfig::new(
            "GSENSE4040BSI",
            create_flat_qe(0.9),
            4096,
            4096,
            9.0,
            2.3,
            0.04,
        )
    });

    /// HWK4123 CMOS sensor
    pub static HWK4123: Lazy<SensorConfig> = Lazy::new(|| {
        SensorConfig::new("HWK4123", create_flat_qe(0.91), 4096, 2300, 4.6, 0.25, 0.1)
    });

    /// Sony IMX455 Full-frame BSI CMOS sensor
    /// Data from: "Characterization of Sony IMX455 sensor for astronomical applications"
    /// https://arxiv.org/pdf/2207.13052
    pub static IMX455: Lazy<SensorConfig> = Lazy::new(|| {
        // QE curve from manufacturer data
        let mut qe = HashMap::new();
        qe.insert(300, 0.05);
        qe.insert(320, 0.05);
        qe.insert(340, 0.12);
        qe.insert(360, 0.22);
        qe.insert(380, 0.35);
        qe.insert(400, 0.52);
        qe.insert(420, 0.68);
        qe.insert(440, 0.82);
        qe.insert(460, 0.90);
        qe.insert(480, 0.94);
        qe.insert(500, 0.94);
        qe.insert(520, 0.92);
        qe.insert(540, 0.86);
        qe.insert(560, 0.80);
        qe.insert(580, 0.72);
        qe.insert(600, 0.64);
        qe.insert(620, 0.56);
        qe.insert(640, 0.48);
        qe.insert(660, 0.42);
        qe.insert(680, 0.36);
        qe.insert(700, 0.30);
        qe.insert(720, 0.25);
        qe.insert(740, 0.22);
        qe.insert(760, 0.18);
        qe.insert(780, 0.16);
        qe.insert(800, 0.14);
        qe.insert(820, 0.12);
        qe.insert(840, 0.10);
        qe.insert(860, 0.09);
        qe.insert(880, 0.08);
        qe.insert(900, 0.06);
        qe.insert(920, 0.05);
        qe.insert(940, 0.04);
        qe.insert(960, 0.03);
        qe.insert(980, 0.02);
        qe.insert(1000, 0.02);

        SensorConfig::new(
            "IMX455", qe, 9568, 6380, 3.75,  // Pixel pitch in microns
            2.67,  // Read noise in electrons (from arxiv paper)
            0.002, // Dark current in e-/px/s at -20°C (from arxiv paper)
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
        assert_eq!(models::GSENSE4040BSI.qe_at_wavelength(550), 0.9);

        // Check HWK4123 properties
        assert_eq!(models::HWK4123.name, "HWK4123");
        assert_eq!(models::HWK4123.width_px, 4096);
        assert_eq!(models::HWK4123.height_px, 2300);
        assert_eq!(models::HWK4123.pixel_size_um, 4.6);
        assert_eq!(models::HWK4123.read_noise_e, 0.25);
        assert_eq!(models::HWK4123.dark_current_e_p_s, 0.1);
        assert_eq!(models::HWK4123.qe_at_wavelength(550), 0.91);

        // Check IMX455 properties
        assert_eq!(models::IMX455.name, "IMX455");
        assert_eq!(models::IMX455.width_px, 9568);
        assert_eq!(models::IMX455.height_px, 6380);
        assert_eq!(models::IMX455.pixel_size_um, 3.75);
        assert_eq!(models::IMX455.read_noise_e, 2.67);
        assert_eq!(models::IMX455.dark_current_e_p_s, 0.002);
        assert_eq!(models::IMX455.qe_at_wavelength(400), 0.52);
        assert_eq!(models::IMX455.qe_at_wavelength(500), 0.94);
        assert_eq!(models::IMX455.qe_at_wavelength(700), 0.30);
    }
}

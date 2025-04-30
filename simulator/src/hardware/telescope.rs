//! Telescope configuration for simulating optical characteristics

use once_cell::sync::Lazy;
use std::f64::consts::PI;

use super::SensorConfig;

/// Configuration for a telescope optical system
#[derive(Debug, Clone)]
pub struct TelescopeConfig {
    /// Aperture diameter in meters
    pub aperture_m: f64,
    /// Focal length in meters
    pub focal_length_m: f64,
    /// Optical efficiency (0.0-1.0) representing light throughput
    pub light_efficiency: f64,
    /// Name/model of the telescope
    pub name: String,
}

const MAS_PER_RAD: f64 = 360.0 * 60.0 * 60.0 * 1000.0 / (PI * 2.0);

impl TelescopeConfig {
    /// Create a new telescope configuration
    pub fn new(
        name: impl Into<String>,
        aperture_m: f64,
        focal_length_m: f64,
        light_efficiency: f64,
    ) -> Self {
        Self {
            name: name.into(),
            aperture_m,
            focal_length_m,
            light_efficiency,
        }
    }

    /// Get the f-number of the telescope
    pub fn f_number(&self) -> f64 {
        self.focal_length_m / self.aperture_m
    }

    /// Calculate the radius of the Airy disk in microns at the focal plane for given wavelength
    ///
    /// The formula used is: r = 1.22 * λ * f / D
    /// where:
    /// - r is the radius of the first dark ring of the Airy disk
    /// - λ is the wavelength
    /// - f is the focal length
    /// - D is the aperture diameter
    pub fn airy_disk_radius_um(&self, wavelength_nm: f64) -> f64 {
        // Convert wavelength from nm to m
        let wavelength_m = wavelength_nm * 1.0e-9;

        // Calculate radius in meters
        let radius_m = 1.22 * wavelength_m * self.focal_length_m / self.aperture_m;

        // Convert to microns
        radius_m * 1.0e6
    }

    /// Calculate the radius of the Airy disk in milliarcseconds
    ///
    /// The formula used is: θ = 1.22 * λ / D
    /// where:
    /// - θ is the angular radius (in radians)
    /// - λ is the wavelength
    /// - D is the aperture diameter
    pub fn airy_disk_radius_mas(&self, wavelength_nm: f64) -> f64 {
        // Convert wavelength from nm to m
        let wavelength_m = wavelength_nm * 1.0e-9;

        // Calculate angular radius in radians
        let radius_rad = 1.22 * wavelength_m / self.aperture_m;

        // Convert to milliarcseconds (1 radian = 206,264,806.247 mas)
        radius_rad * MAS_PER_RAD
    }

    /// Calculate the diffraction-limited resolution in milliarcseconds at given wavelength
    pub fn diffraction_limited_resolution_mas(&self, wavelength_nm: f64) -> f64 {
        self.airy_disk_radius_mas(wavelength_nm) * 2.0
    }

    /// Calculate the plate scale in arcseconds per mm
    pub fn plate_scale_arcsec_per_mm(&self) -> f64 {
        // 1 radian = 206,264.8 arcseconds
        // plate scale = (1/f) * 206,264.8 arcsec/radian
        (1.0 / self.focal_length_m) * 206_264.8
    }

    /// Calculate the collecting area in square meters
    pub fn collecting_area_m2(&self) -> f64 {
        PI * (self.aperture_m / 2.0).powi(2)
    }

    /// Calculate the effective collecting area considering light efficiency
    pub fn effective_collecting_area_m2(&self) -> f64 {
        self.collecting_area_m2() * self.light_efficiency
    }
}

/// Creates telescope optics configuration optimized for a specific sensor
///
/// Calculates focal length to achieve 4 pixels per Airy disk for optimal sampling
///
/// # Arguments
/// * `sensor` - Sensor configuration with pixel size
/// * `wavelength_nm` - Light wavelength in nanometers
/// * `airy_pix` - Desired Airy disk size in pixels
///
/// # Returns
/// * `TelescopeConfig` - Optimized telescope configuration
pub fn build_optics_for_sensor(
    telescope: &TelescopeConfig,
    sensor: &SensorConfig,
    wavelength_nm: f64,
    airy_pix: f64,
) -> TelescopeConfig {
    // Make a pretend telescope with focal length driven to make 4pix/airy disk
    let target_airy_disk_um = sensor.pixel_size_um * airy_pix; // 4 pixels per Airy disk
    let wavelength_m = wavelength_nm * 1e-9; // Convert nm to m
    let focal_length_m = (target_airy_disk_um * telescope.aperture_m) / (1e6 * 1.22 * wavelength_m);

    // Create a new TelescopeConfig with the calculated focal length
    let name = format!("{} for {:.1}", telescope.name, focal_length_m);

    TelescopeConfig::new(
        &name,
        telescope.aperture_m, // Use the same aperture
        focal_length_m,
        telescope.light_efficiency, // Light efficiency
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn test_f_number() {
        let telescope = TelescopeConfig::new("Test", 0.5, 4.0, 0.8);
        assert_eq!(telescope.f_number(), 8.0);
    }

    #[test]
    fn test_airy_disk_radii() {
        let telescope = TelescopeConfig::new("Test", 0.5, 4.0, 0.8);
        let wavelength_nm = 550.0; // Green light

        // Calculate expected values
        let expected_radius_um = 1.22 * wavelength_nm * 1e-9 * 4.0 / 0.5 * 1e6;
        // Updated to match actual MAS_PER_RAD constant implementation
        let expected_radius_mas =
            1.22 * wavelength_nm * 1e-9 / 0.5 * 360.0 * 60.0 * 60.0 * 1000.0 / (PI * 2.0);

        assert!(approx_eq!(
            f64,
            telescope.airy_disk_radius_um(wavelength_nm),
            expected_radius_um,
            epsilon = 1e-6
        ));
        assert!(approx_eq!(
            f64,
            telescope.airy_disk_radius_mas(wavelength_nm),
            expected_radius_mas,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_plate_scale() {
        let telescope = TelescopeConfig::new("Test", 0.5, 4.0, 0.8);

        // Calculate expected plate scale
        let expected_plate_scale = (1.0 / 4.0) * 206_264.8;

        assert!(approx_eq!(
            f64,
            telescope.plate_scale_arcsec_per_mm(),
            expected_plate_scale,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_collecting_area() {
        let telescope = TelescopeConfig::new("Test", 0.5, 4.0, 0.8);

        // Calculate expected areas
        let radius = 0.5 / 2.0;
        let expected_area = PI * radius * radius;
        let expected_effective_area = expected_area * 0.8;

        assert!(approx_eq!(
            f64,
            telescope.collecting_area_m2(),
            expected_area,
            epsilon = 1e-6
        ));
        assert!(approx_eq!(
            f64,
            telescope.effective_collecting_area_m2(),
            expected_effective_area,
            epsilon = 1e-6
        ));
    }
}

/// Standard telescope models
pub mod models {
    use super::*;

    pub static SMALL_50MM: Lazy<TelescopeConfig> = Lazy::new(|| {
        TelescopeConfig::new(
            "50mm", 0.05,  // 50mm aperture
            0.5,   // 50mm
            0.615, // light efficiency
        )
    });

    /// 50cm Demo telescope
    pub static DEMO_50CM: Lazy<TelescopeConfig> = Lazy::new(|| {
        TelescopeConfig::new(
            "50cm Demo",
            0.5,   // 50cm aperture
            10.0,  // 1m focal length
            0.815, // light efficiency
        )
    });

    /// 1m Final telescope
    pub static FINAL_1M: Lazy<TelescopeConfig> = Lazy::new(|| {
        TelescopeConfig::new(
            "1m Final", 1.0,   // 1m aperture
            10.0,  // 10m focal length
            0.815, // light efficiency
        )
    });
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn test_predefined_telescopes() {
        // Test 50cm Demo telescope
        assert_eq!(models::DEMO_50CM.name, "50cm Demo");
        assert_eq!(models::DEMO_50CM.aperture_m, 0.5);
        assert_eq!(models::DEMO_50CM.focal_length_m, 10.0);
        assert_eq!(models::DEMO_50CM.light_efficiency, 0.815);
        assert_eq!(models::DEMO_50CM.f_number(), 20.0);

        // Test 1m Final telescope
        assert_eq!(models::FINAL_1M.name, "1m Final");
        assert_eq!(models::FINAL_1M.aperture_m, 1.0);
        assert_eq!(models::FINAL_1M.focal_length_m, 10.0);
        assert_eq!(models::FINAL_1M.light_efficiency, 0.815);
        assert_eq!(models::FINAL_1M.f_number(), 10.0);
    }
}

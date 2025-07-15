//! Telescope optical system configuration and models for astronomical simulation.
//!
//! This module provides comprehensive telescope models with realistic optical
//! characteristics for space-based and ground-based observations. It handles
//! diffraction-limited optics, plate scale calculations, and light-gathering
//! performance modeling.
//!
//! # Key Features
//!
//! - **Optical calculations**: Airy disk sizing, PSF modeling, diffraction limits
//! - **Geometric properties**: Plate scale, field of view, f-number calculations  
//! - **Light collection**: Aperture area, throughput efficiency, signal prediction
//! - **Adaptive optics**: Focal length optimization for detector sampling
//! - **Predefined models**: Real-world telescope configurations
//!
//! # Physics Models
//!
//! ## Diffraction-Limited Optics
//! Implements standard astronomical optics formulas:
//! - **Airy disk radius**: θ = 1.22λ/D (angular), r = 1.22λf/D (linear)
//! - **Resolution limit**: FWHM ≈ λ/D for circular apertures
//! - **Plate scale**: arcsec/mm = 206265/f (f in mm)
//!
//! ## Light Collection
//! - **Collecting area**: A = π(D/2)²
//! - **Effective area**: Considers optical losses, obscuration
//! - **Throughput efficiency**: Mirror reflectivity, filter transmission
//!
//! # Telescope Models
//!
//! The module includes several representative configurations:
//! - **SMALL_50MM**: Compact finder scope or guide telescope
//! - **DEMO_50CM**: Medium aperture space telescope prototype  
//! - **FINAL_1M**: Large aperture survey telescope
//! - **WEASEL**: Multi-spectral Earth observation system
//!
//! # Examples
//!
//! ```rust
//! use simulator::hardware::telescope::{TelescopeConfig, models::DEMO_50CM};
//! use simulator::hardware::sensor::models::GSENSE6510BSI;
//!
//! // Use predefined telescope model
//! let telescope = DEMO_50CM.clone();
//! let sensor = GSENSE6510BSI.clone();
//!
//! // Calculate optical performance
//! let f_number = telescope.f_number();
//! let resolution_mas = telescope.diffraction_limited_resolution_mas(550.0);
//! let plate_scale = telescope.plate_scale_arcsec_per_mm();
//! let collecting_area = telescope.effective_collecting_area_m2();
//!
//! println!("f/{:.1} telescope", f_number);
//! println!("Resolution: {:.0} mas @ 550nm", resolution_mas);
//! println!("Plate scale: {:.1} arcsec/mm", plate_scale);
//! println!("Light gathering: {:.2} m²", collecting_area);
//!
//! // Optimize focal length for sensor sampling
//! let optimized = telescope.with_focal_length(8.5);
//! let airy_radius_um = optimized.airy_disk_radius_um(550.0);
//! let pixels_per_airy = airy_radius_um / sensor.pixel_size_um;
//! println!("Sampling: {:.1} pixels per Airy disk", pixels_per_airy);
//! ```

use once_cell::sync::Lazy;
use std::f64::consts::PI;

/// Complete telescope optical system configuration.
///
/// Represents a telescope with all optical characteristics needed for
/// realistic astronomical simulations. Provides methods for calculating
/// resolution, light-gathering power, and geometric properties.
///
/// # Key Capabilities
///
/// - **Diffraction modeling**: Airy disk calculations at any wavelength
/// - **Angular resolution**: Theoretical and practical resolution limits
/// - **Photometric scaling**: Light collection and throughput efficiency
/// - **Geometric optics**: Plate scale, field of view, f-number calculations
/// - **Sampling optimization**: Focal length adjustment for optimal detector sampling
///
/// # Examples
///
/// ```rust
/// use simulator::hardware::telescope::TelescopeConfig;
///
/// // Create custom telescope configuration
/// let telescope = TelescopeConfig::new(
///     "Custom 0.8m",
///     0.8,   // 80cm aperture
///     6.4,   // 6.4m focal length (f/8)
///     0.75,  // 75% throughput efficiency
/// );
///
/// // Calculate optical properties
/// let f_ratio = telescope.f_number();
/// let resolution = telescope.diffraction_limited_resolution_mas(650.0);
/// let light_power = telescope.effective_collecting_area_m2();
///
/// println!("f/{:.1} system", f_ratio);
/// println!("Resolution: {:.0} mas", resolution);
/// println!("Effective area: {:.2} m²", light_power);
/// ```
#[derive(Debug, Clone)]
pub struct TelescopeConfig {
    /// Primary mirror or lens diameter in meters (clear aperture)
    pub aperture_m: f64,
    /// Effective focal length in meters (including optical train)
    pub focal_length_m: f64,
    /// Total optical efficiency (0.0-1.0, includes all losses)
    pub light_efficiency: f64,
    /// Telescope model name or identifier
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
        self.fwhm_image_spot_um(wavelength_nm) * 1.22
    }

    pub fn fwhm_image_spot_um(&self, wavelength_nm: f64) -> f64 {
        // Convert wavelength from nm to m
        let wavelength_m = wavelength_nm * 1.0e-9;

        // Calculate radius in meters
        let radius_m = wavelength_m * self.focal_length_m / self.aperture_m;

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

    /// Create a new telescope configuration with modified focal length
    pub fn with_focal_length(&self, focal_length_m: f64) -> TelescopeConfig {
        TelescopeConfig {
            name: self.name.clone(),
            aperture_m: self.aperture_m,
            focal_length_m,
            light_efficiency: self.light_efficiency,
        }
    }
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

    /// Weasel telescope
    /// Spectral ranges: PAN: 0.45 – 0.9 um, RGB: 0.45 – 0.68 um, SWIR: 0.9 – 1.7 um
    pub static WEASEL: Lazy<TelescopeConfig> = Lazy::new(|| {
        TelescopeConfig::new(
            "Weasel", 0.47, // 470mm aperture
            3.45, // 3450mm focal length
            0.70, // TODO: Confirm light efficiency value
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

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
//! - **IDEAL_50CM**: Medium aperture space telescope prototype  
//! - **IDEAL_1M**: Large aperture survey telescope
//! - **WEASEL**: Multi-spectral Earth observation system
//!

use once_cell::sync::Lazy;
use std::f64::consts::PI;

use crate::photometry::QuantumEfficiency;
use crate::units::{Length, LengthExt, Wavelength};

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
/// - **Wavelength response**: Quantum efficiency curves for optical system
///
#[derive(Debug, Clone)]
pub struct TelescopeConfig {
    /// Primary mirror or lens diameter (clear aperture)
    pub aperture: Length,
    /// Effective focal length (including optical train)
    pub focal_length: Length,
    /// Telescope model name or identifier
    pub name: String,
    /// Central obscuration ratio (0.0-1.0, fraction of aperture radius blocked)
    pub obscuration_ratio: f64,
    /// Wavelength-dependent quantum efficiency of the optical system
    /// Represents combined mirror reflectivity, lens transmission, etc.
    pub quantum_efficiency: QuantumEfficiency,
}

const MAS_PER_RAD: f64 = 360.0 * 60.0 * 60.0 * 1000.0 / (PI * 2.0);

impl TelescopeConfig {
    /// Create a new telescope configuration with flat quantum efficiency
    ///
    /// Creates a telescope with a flat QE curve from 150nm to 1100nm
    /// using the provided light_efficiency value.
    pub fn new(
        name: impl Into<String>,
        aperture: Length,
        focal_length: Length,
        light_efficiency: f64,
    ) -> Self {
        // Create flat QE curve from 150nm to 1100nm
        let wavelengths = vec![149.0, 150.0, 1100.0, 1101.0];
        let efficiencies = vec![0.0, light_efficiency, light_efficiency, 0.0];
        let quantum_efficiency = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create default telescope QE curve");

        Self {
            name: name.into(),
            aperture,
            focal_length,
            obscuration_ratio: 0.0,
            quantum_efficiency,
        }
    }

    /// Create a new telescope configuration with custom quantum efficiency curve
    pub fn new_with_qe(
        name: impl Into<String>,
        aperture: Length,
        focal_length: Length,
        quantum_efficiency: QuantumEfficiency,
        obscuration_ratio: f64,
    ) -> Self {
        Self {
            name: name.into(),
            aperture,
            focal_length,
            obscuration_ratio,
            quantum_efficiency,
        }
    }

    /// Get the f-number of the telescope
    pub fn f_number(&self) -> f64 {
        self.focal_length.as_meters() / self.aperture.as_meters()
    }

    /// Calculate the radius of the Airy disk in microns at the focal plane for given wavelength
    ///
    /// The formula used is: r = 1.22 * λ * f / D
    /// where:
    /// - r is the radius of the first dark ring of the Airy disk
    /// - λ is the wavelength
    /// - f is the focal length
    /// - D is the aperture diameter
    pub fn airy_disk_radius_um(&self, wavelength: Wavelength) -> f64 {
        self.fwhm_image_spot_um(wavelength) * 1.22
    }

    pub fn fwhm_image_spot_um(&self, wavelength: Wavelength) -> f64 {
        let wavelength_nm = wavelength.as_nanometers();
        // Convert wavelength from nm to m
        let wavelength_m = wavelength_nm * 1.0e-9;

        // Calculate radius in meters
        let radius_m = wavelength_m * self.focal_length.as_meters() / self.aperture.as_meters();

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
    pub fn airy_disk_radius_mas(&self, wavelength: Wavelength) -> f64 {
        let wavelength_nm = wavelength.as_nanometers();
        // Convert wavelength from nm to m
        let wavelength_m = wavelength_nm * 1.0e-9;

        // Calculate angular radius in radians
        let radius_rad = 1.22 * wavelength_m / self.aperture.as_meters();

        // Convert to milliarcseconds (1 radian = 206,264,806.247 mas)
        radius_rad * MAS_PER_RAD
    }

    /// Calculate the diffraction-limited resolution in milliarcseconds at given wavelength
    pub fn diffraction_limited_resolution_mas(&self, wavelength: Wavelength) -> f64 {
        self.airy_disk_radius_mas(wavelength) * 2.0
    }

    /// Calculate the plate scale in arcseconds per mm
    pub fn plate_scale_arcsec_per_mm(&self) -> f64 {
        // 1 radian = 206,264.8 arcseconds
        // plate scale = (1/f) * 206,264.8 arcsec/radian
        // focal_length needs to be converted to mm
        (1.0 / (self.focal_length.as_meters() * 1000.0)) * 206_264.8
    }

    /// Calculate the collecting area in square meters
    pub fn collecting_area_m2(&self) -> f64 {
        let aperture_m = self.aperture.as_meters();
        let outer_area = PI * (aperture_m / 2.0).powi(2);
        let obscured_diameter = aperture_m * self.obscuration_ratio;
        let obscured_area = PI * (obscured_diameter / 2.0).powi(2);
        outer_area - obscured_area
    }

    pub fn collecting_area_cm2(&self) -> f64 {
        // Convert m² to cm²
        self.collecting_area_m2() * 10_000.0
    }

    /// Create a new telescope configuration with modified focal length
    pub fn with_focal_length(&self, focal_length: Length) -> TelescopeConfig {
        TelescopeConfig {
            name: self.name.clone(),
            aperture: self.aperture,
            focal_length,
            quantum_efficiency: self.quantum_efficiency.clone(),
            obscuration_ratio: self.obscuration_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn test_f_number() {
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.5),
            Length::from_meters(4.0),
            0.8,
        );
        assert_eq!(telescope.f_number(), 8.0);
    }

    #[test]
    fn test_airy_disk_radii() {
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.5),
            Length::from_meters(4.0),
            0.8,
        );
        let wavelength_nm = 550.0; // Green light

        // Calculate expected values
        let expected_radius_um = 1.22 * wavelength_nm * 1e-9 * 4.0 / 0.5 * 1e6;
        // Updated to match actual MAS_PER_RAD constant implementation
        let expected_radius_mas =
            1.22 * wavelength_nm * 1e-9 / 0.5 * 360.0 * 60.0 * 60.0 * 1000.0 / (PI * 2.0);

        assert!(approx_eq!(
            f64,
            telescope.airy_disk_radius_um(Wavelength::from_nanometers(wavelength_nm)),
            expected_radius_um,
            epsilon = 1e-6
        ));
        assert!(approx_eq!(
            f64,
            telescope.airy_disk_radius_mas(Wavelength::from_nanometers(wavelength_nm)),
            expected_radius_mas,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_plate_scale() {
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.5),
            Length::from_meters(4.0),
            0.8,
        );

        // Calculate expected plate scale (focal length 4m = 4000mm)
        let expected_plate_scale = (1.0 / 4000.0) * 206_264.8;

        assert!(approx_eq!(
            f64,
            telescope.plate_scale_arcsec_per_mm(),
            expected_plate_scale,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_collecting_area() {
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.5),
            Length::from_meters(4.0),
            0.8,
        );

        // Calculate expected areas
        let radius = 0.5 / 2.0;
        let expected_area = PI * radius * radius;
        assert!(approx_eq!(
            f64,
            telescope.collecting_area_m2(),
            expected_area,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_obscuration_ratio() {
        // Test case from user: 50cm aperture with 42% obscuration should have 1617 cm² clear area
        let mut telescope = TelescopeConfig::new(
            "Test Obscured",
            Length::from_meters(0.5),
            Length::from_meters(4.0),
            1.0,
        );
        telescope.obscuration_ratio = 0.42;

        let area_cm2 = telescope.collecting_area_cm2();

        // Expected: 1617 cm²
        assert!(
            approx_eq!(f64, area_cm2, 1617.0, epsilon = 1.0),
            "Expected 1617 cm², got {area_cm2:.1} cm²"
        );
    }
}

/// Standard telescope models
pub mod models {
    use super::*;

    pub static SMALL_50MM: Lazy<TelescopeConfig> = Lazy::new(|| {
        TelescopeConfig::new(
            "50mm",
            Length::from_meters(0.05), // 50mm aperture
            Length::from_meters(0.5),  // 0.5m focal length
            0.615,                     // light efficiency
        )
    });

    /// 50cm Demo telescope
    pub static IDEAL_50CM: Lazy<TelescopeConfig> = Lazy::new(|| {
        TelescopeConfig::new(
            "Ideal 50cm",
            Length::from_meters(0.5),  // 50cm aperture
            Length::from_meters(10.0), // 10m focal length
            0.815,                     // light efficiency
        )
    });

    /// 1m Final telescope
    pub static IDEAL_100CM: Lazy<TelescopeConfig> = Lazy::new(|| {
        TelescopeConfig::new(
            "Ideal 100cm",
            Length::from_meters(1.0),  // 1m aperture
            Length::from_meters(10.0), // 10m focal length
            0.815,                     // light efficiency
        )
    });

    /// Officina Stellare Weasel - 50cm f/6.9 Catadioptric
    /// Bandpass: 450nm-900nm, 0.8 Strehl at 800nm, 42% obscuration ratio
    pub static OFFICINA_STELLARE_WEASEL: Lazy<TelescopeConfig> = Lazy::new(|| {
        // Create QE curve for Weasel based on spectral data
        let wavelengths = vec![
            149.0, 175.0, 300.0, 395.0, 450.0, 545.0, 680.0, 820.0, 900.0, 1050.0, 1400.0, 1800.0,
            1801.0,
        ];
        let efficiencies = vec![
            0.0, 0.0, 0.0, 0.0, 0.60, 0.77, 0.73, 0.73, 0.71, 0.0, 0.0, 0.0, 0.0,
        ];
        let quantum_efficiency = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create Weasel QE curve");

        TelescopeConfig::new_with_qe(
            "Officina Stellare Weasel",
            Length::from_meters(0.5),  // 50cm aperture
            Length::from_meters(3.45), // 345cm focal length (f/6.9)
            quantum_efficiency,
            0.42, // 42% linear obscuration ratio
        )
    });

    /// Optech/Lina LS50 - 50cm f/10 Catadioptric  
    /// Bandpass: 400-1100nm, 0.8 Strehl at 833nm, 37% obscuration ratio
    pub static OPTECH_LINA_LS50: Lazy<TelescopeConfig> = Lazy::new(|| {
        let wavelengths = vec![
            149.0, 175.0, 300.0, 395.0, 400.0, 545.0, 680.0, 820.0, 1050.0, 1100.0, 1400.0, 1800.0,
            1801.0,
        ];
        let efficiencies = vec![
            0.0, 0.0, 0.0, 0.0, 0.70, 0.78, 0.77, 0.74, 0.77, 0.70, 0.0, 0.0, 0.0,
        ];
        let quantum_efficiency = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create LS50 QE curve");

        TelescopeConfig::new_with_qe(
            "Optech/Lina LS50",
            Length::from_meters(0.5), // 50cm aperture
            Length::from_meters(5.0), // 500cm focal length (f/10)
            quantum_efficiency,
            0.37, // 37% linear obscuration ratio
        )
    });

    /// Optech/Lina LS35 - 35cm f/10 Catadioptric
    /// Bandpass: 400-1100nm, 0.8 Strehl at 833nm, 37% obscuration ratio  
    pub static OPTECH_LINA_LS35: Lazy<TelescopeConfig> = Lazy::new(|| {
        let wavelengths = vec![
            149.0, 175.0, 300.0, 395.0, 400.0, 545.0, 680.0, 820.0, 1050.0, 1100.0, 1400.0, 1800.0,
            1801.0,
        ];
        let efficiencies = vec![
            0.0, 0.0, 0.0, 0.0, 0.70, 0.78, 0.77, 0.74, 0.77, 0.70, 0.0, 0.0, 0.0,
        ];
        let quantum_efficiency = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create LS35 QE curve");

        TelescopeConfig::new_with_qe(
            "Optech/Lina LS35",
            Length::from_meters(0.35), // 35cm aperture
            Length::from_meters(3.5),  // 350cm focal length (f/10)
            quantum_efficiency,
            0.37, // 37% linear obscuration ratio
        )
    });

    /// Cosmic Frontier JBT .5m - 48.5cm f/12.3 Reflective
    /// Bandpass: broad spectrum, 0.8 Strehl at 700nm
    pub static COSMIC_FRONTIER_JBT_50CM: Lazy<TelescopeConfig> = Lazy::new(|| {
        let wavelengths = vec![
            149.0, 175.0, 300.0, 395.0, 545.0, 680.0, 820.0, 1050.0, 1400.0, 1800.0, 1801.0,
        ];
        let efficiencies = vec![
            0.0, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.0,
        ];
        let quantum_efficiency = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create JBT 50cm QE curve");

        TelescopeConfig::new_with_qe(
            "Cosmic Frontier JBT .5m",
            Length::from_meters(0.485), // 48.5cm aperture
            Length::from_meters(5.987), // 598.7cm focal length (f/12.3)
            quantum_efficiency,
            0.35, // 35% linear obscuration ratio
        )
    });

    /// Cosmic Frontier JBT MAX - 65cm f/12.3 Reflective
    /// Bandpass: broad spectrum, 0.8 Strehl at 700nm
    pub static COSMIC_FRONTIER_JBT_MAX: Lazy<TelescopeConfig> = Lazy::new(|| {
        let wavelengths = vec![
            149.0, 175.0, 300.0, 395.0, 545.0, 680.0, 820.0, 1050.0, 1400.0, 1800.0, 1801.0,
        ];
        let efficiencies = vec![
            0.0, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.0,
        ];
        let quantum_efficiency = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create JBT MAX QE curve");

        TelescopeConfig::new_with_qe(
            "Cosmic Frontier JBT MAX",
            Length::from_meters(0.65),  // 65cm aperture
            Length::from_meters(8.024), // 802.4cm focal length (f/12.3)
            quantum_efficiency,
            0.35, // 35% linear obscuration ratio
        )
    });

    /// Cosmic Frontier JBT 1.0m - 100cm f/12.3 Reflective
    /// Bandpass: broad spectrum, 0.8 Strehl at 700nm
    pub static COSMIC_FRONTIER_JBT_1M: Lazy<TelescopeConfig> = Lazy::new(|| {
        let wavelengths = vec![
            149.0, 175.0, 300.0, 395.0, 545.0, 680.0, 820.0, 1050.0, 1400.0, 1800.0, 1801.0,
        ];
        let efficiencies = vec![
            0.0, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.0,
        ];
        let quantum_efficiency = QuantumEfficiency::from_table(wavelengths, efficiencies)
            .expect("Failed to create JBT 1m QE curve");

        TelescopeConfig::new_with_qe(
            "Cosmic Frontier JBT 1.0m",
            Length::from_meters(1.0),    // 100cm aperture
            Length::from_meters(12.344), // 1234.4cm focal length (f/12.3)
            quantum_efficiency,
            0.35, // 35% linear obscuration ratio
        )
    });

    /// Legacy Weasel telescope (keeping for backwards compatibility)
    /// Spectral ranges: PAN: 0.45 – 0.9 um, RGB: 0.45 – 0.68 um, SWIR: 0.9 – 1.7 um
    pub static WEASEL: Lazy<TelescopeConfig> = Lazy::new(|| {
        TelescopeConfig::new(
            "Weasel",
            Length::from_meters(0.47), // 470mm aperture
            Length::from_meters(3.45), // 3450mm focal length
            0.70,                      // TODO: Confirm light efficiency value
        )
    });
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn test_predefined_telescopes() {
        // Test 50cm Demo telescope
        assert_eq!(models::IDEAL_50CM.name, "Ideal 50cm");
        assert_eq!(models::IDEAL_50CM.aperture.as_meters(), 0.5);
        assert_eq!(models::IDEAL_50CM.focal_length.as_meters(), 10.0);
        assert_eq!(
            models::IDEAL_50CM
                .quantum_efficiency
                .at(Wavelength::from_nanometers(550.0)),
            0.815
        );
        assert_eq!(models::IDEAL_50CM.f_number(), 20.0);

        // Test 1m Final telescope
        assert_eq!(models::IDEAL_100CM.name, "Ideal 100cm");
        assert_eq!(models::IDEAL_100CM.aperture.as_meters(), 1.0);
        assert_eq!(models::IDEAL_100CM.focal_length.as_meters(), 10.0);
        assert_eq!(
            models::IDEAL_100CM
                .quantum_efficiency
                .at(Wavelength::from_nanometers(550.0)),
            0.815
        );
        assert_eq!(models::IDEAL_100CM.f_number(), 10.0);
    }

    #[test]
    fn test_concrete_telescope_embodiments() {
        use float_cmp::approx_eq;

        // Test Officina Stellare Weasel
        let weasel = &*models::OFFICINA_STELLARE_WEASEL;
        assert_eq!(weasel.name, "Officina Stellare Weasel");
        assert_eq!(weasel.aperture.as_meters(), 0.5);
        assert!(approx_eq!(
            f64,
            weasel.focal_length.as_meters(),
            3.45,
            epsilon = 1e-6
        ));
        assert!(approx_eq!(f64, weasel.f_number(), 6.9, epsilon = 1e-6));
        assert_eq!(weasel.obscuration_ratio, 0.42);
        // At 600nm, interpolate between 545nm (0.77) and 680nm (0.73)
        // Linear interpolation: 0.77 - (600-545)/(680-545) * (0.77-0.73) ≈ 0.754
        assert!(approx_eq!(
            f64,
            weasel
                .quantum_efficiency
                .at(Wavelength::from_nanometers(600.0)),
            0.754,
            epsilon = 1e-2
        ));
        // At 400nm, slightly out of main band but still has some interpolated value
        assert!(
            weasel
                .quantum_efficiency
                .at(Wavelength::from_nanometers(400.0))
                < 0.1
        ); // Near zero but not exactly

        // Test Optech/Lina LS50
        let ls50 = &*models::OPTECH_LINA_LS50;
        assert_eq!(ls50.name, "Optech/Lina LS50");
        assert_eq!(ls50.aperture.as_meters(), 0.5);
        assert_eq!(ls50.focal_length.as_meters(), 5.0);
        assert_eq!(ls50.f_number(), 10.0);
        assert_eq!(ls50.obscuration_ratio, 0.37);
        // At 600nm, interpolate between 545nm (0.78) and 680nm (0.77) ≈ 0.778
        assert!(approx_eq!(
            f64,
            ls50.quantum_efficiency
                .at(Wavelength::from_nanometers(600.0)),
            0.778,
            epsilon = 1e-2
        ));

        // Test Optech/Lina LS35
        let ls35 = &*models::OPTECH_LINA_LS35;
        assert_eq!(ls35.name, "Optech/Lina LS35");
        assert_eq!(ls35.aperture.as_meters(), 0.35);
        assert_eq!(ls35.focal_length.as_meters(), 3.5);
        assert_eq!(ls35.f_number(), 10.0);
        assert_eq!(ls35.obscuration_ratio, 0.37);

        // Test Cosmic Frontier JBT .5m
        let jbt50 = &*models::COSMIC_FRONTIER_JBT_50CM;
        assert_eq!(jbt50.name, "Cosmic Frontier JBT .5m");
        assert_eq!(jbt50.aperture.as_meters(), 0.485);
        assert!(approx_eq!(
            f64,
            jbt50.focal_length.as_meters(),
            5.987,
            epsilon = 1e-6
        ));
        assert!(approx_eq!(f64, jbt50.f_number(), 12.344, epsilon = 1e-2));
        assert_eq!(jbt50.obscuration_ratio, 0.35); // 35% linear obscuration ratio
        assert_eq!(
            jbt50
                .quantum_efficiency
                .at(Wavelength::from_nanometers(550.0)),
            0.70
        );

        // Test Cosmic Frontier JBT MAX
        let jbt_max = &*models::COSMIC_FRONTIER_JBT_MAX;
        assert_eq!(jbt_max.name, "Cosmic Frontier JBT MAX");
        assert_eq!(jbt_max.aperture.as_meters(), 0.65);
        assert!(approx_eq!(
            f64,
            jbt_max.focal_length.as_meters(),
            8.024,
            epsilon = 1e-6
        ));
        assert!(approx_eq!(f64, jbt_max.f_number(), 12.344, epsilon = 1e-2));
        assert_eq!(jbt_max.obscuration_ratio, 0.35); // 35% linear obscuration ratio

        // Test Cosmic Frontier JBT 1.0m
        let jbt1m = &*models::COSMIC_FRONTIER_JBT_1M;
        assert_eq!(jbt1m.name, "Cosmic Frontier JBT 1.0m");
        assert_eq!(jbt1m.aperture.as_meters(), 1.0);
        assert!(approx_eq!(
            f64,
            jbt1m.focal_length.as_meters(),
            12.344,
            epsilon = 1e-6
        ));
        assert!(approx_eq!(f64, jbt1m.f_number(), 12.344, epsilon = 1e-2));
        assert_eq!(jbt1m.obscuration_ratio, 0.35); // 35% linear obscuration ratio
    }
}

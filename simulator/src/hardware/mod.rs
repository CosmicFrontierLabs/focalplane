//! Hardware module for telescope, sensor, and satellite configurations.
//!
//! This module provides comprehensive models for space telescope hardware,
//! including optical systems, sensor characteristics, and integrated satellite
//! configurations. It supports realistic simulation of astronomical observations
//! with proper modeling of:
//!
//! # Components
//!
//! - **Telescopes**: Optical parameters including aperture, focal length,
//!   light efficiency, and geometric properties
//! - **Sensors**: Detailed CCD/CMOS models with quantum efficiency curves,
//!   noise characteristics, pixel geometry, and temperature dependencies
//! - **Satellites**: Integrated telescope-sensor configurations with
//!   environmental parameters and observation modes
//! - **Dark Current**: Temperature-dependent noise modeling for thermal
//!   characterization of sensors
//!
//! # Physics Models
//!
//! The module incorporates realistic physics including:
//! - Airy disk PSF calculations for diffraction-limited optics
//! - Quantum efficiency spectral response curves
//! - Temperature-dependent dark current scaling
//! - Photon noise and read noise contributions
//! - Optical throughput and vignetting effects
//!
//! # Usage Examples
//!
//! ```rust
//! use simulator::hardware::{SatelliteConfig, TelescopeConfig, SensorConfig};
//! use simulator::hardware::telescope::models::IDEAL_50CM;
//! use simulator::hardware::sensor::models::GSENSE6510BSI;
//! use simulator::units::{LengthExt, Temperature, TemperatureExt};
//!
//! // Create a satellite configuration
//! let satellite = SatelliteConfig::new(
//!     IDEAL_50CM.clone(),
//!     GSENSE6510BSI.clone(),
//!     Temperature::from_celsius(-10.0),  // Temperature in °C
//! );
//!
//! // Calculate field of view
//! let fov_degrees = simulator::star_math::field_diameter(
//!     &satellite.telescope,
//!     &satellite.sensor
//! );
//! ```

pub mod dark_current;
pub mod read_noise;
pub mod satellite;
pub mod sensor;
pub mod sensor_array;
pub mod sensor_noise;
pub mod telescope;

pub use satellite::SatelliteConfig;
pub use sensor::SensorConfig;
pub use sensor_array::SensorArray;
pub use telescope::TelescopeConfig;

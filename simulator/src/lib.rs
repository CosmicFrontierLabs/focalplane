//! Comprehensive space telescope simulation framework.
//!
//! This crate provides a complete end-to-end simulation pipeline for space telescope
//! systems, from celestial scene generation through optical modeling to detector
//! readout and image analysis. Designed for high-fidelity modeling of astronomical
//! observations with realistic physics and sensor characteristics.
//!
//! # Architecture Overview
//!
//! The simulator is organized into several interconnected modules:
//!
//! ## Core Simulation Chain
//!
//! 1. **Scene Generation** (`scene`): Create realistic star fields from catalogs
//! 2. **Stellar Physics** (`star_math`, `photometry`): Convert celestial coordinates and magnitudes to physical flux
//! 3. **Optical System** (`hardware::telescope`): Model telescope optics, PSF, and throughput  
//! 4. **Sensor Simulation** (`hardware::sensor`): Realistic detector response with noise
//! 5. **Image Processing** (`image_proc`): Detection, analysis, and visualization
//! 6. **Tracking Algorithms** (`algo`): Star tracking and attitude determination
//!
//! ## Key Physics Models
//!
//! - **Stellar photometry**: Accurate magnitude-to-flux conversion with spectral models
//! - **Telescope optics**: Airy disk PSF, vignetting, and optical aberrations
//! - **Sensor physics**: Quantum efficiency, read noise, dark current, and shot noise
//! - **Astronomical backgrounds**: Zodiacal light and scattered light contributions
//!
//! # Quick Start Example
//!
//! ```rust,ignore
//! // NOTE: This doctest is ignored due to API changes in detection module
//! use simulator::{
//!     Scene,
//!     hardware::{SatelliteConfig, telescope::models::IDEAL_50CM, sensor::models::GSENSE6510BSI},
//!     image_proc::{detect_stars_unified, StarFinder},
//!     photometry::zodical::SolarAngularCoordinates
//! };
//! use starfield::{Equatorial, catalogs::tycho};
//! use std::time::Duration;
//!
//! // 1. Create satellite configuration
//! let satellite_config = SatelliteConfig::new(
//!     IDEAL_50CM.clone(),
//!     GSENSE6510BSI.clone(),
//!     -10.0,  // Sensor temperature
//!     550.0   // Orbit altitude
//! );
//!
//! // 2. Load star catalog
//! let catalog_stars = tycho::load_from_sample(1000).unwrap();
//!
//! // 3. Create scene with pointing and exposure
//! let zodiacal_coords = SolarAngularCoordinates::zodiacal_minimum();
//! let scene = Scene::from_catalog(
//!     satellite_config,
//!     catalog_stars,
//!     Equatorial::from_degrees(180.0, -30.0), // Pointing
//!     Duration::from_secs(30), // 30-second exposure
//!     zodiacal_coords,
//! );
//!
//! // 4. Render the scene
//! let result = scene.render();
//!
//! // 5. Detect stars  
//! let detections = detect_stars_unified(
//!     result.quantized_image.view(),
//!     StarFinder::Dao,  // DAO photometry algorithm
//!     2.5,              // PSF size
//!     10.0,             // Background RMS  
//!     5.0               // Detection threshold
//! ).expect("Star detection failed");
//!
//! println!("Detected {} stars in simulated image", detections.len());
//! ```
//!
//! # Module Organization
//!
//! ## Algorithms (`algo`)
//! - **ICP**: Iterative Closest Point for star pattern matching
//! - **Motion**: Attitude dynamics and kinematics
//! - **PSD**: Power spectral density analysis for tracking performance
//! - **Quaternions**: 3D rotation mathematics
//! - **Splines**: Cubic spline interpolation for smooth trajectories
//!
//! ## Hardware Modeling (`hardware`)
//! - **Telescope**: Optical system modeling with realistic PSF
//! - **Sensor**: Detector physics with temperature-dependent characteristics
//! - **Satellite**: Spacecraft configuration and pointing systems
//! - **Star Projection**: Coordinate transformations and flux calculations
//!
//! ## Image Processing (`image_proc`)
//! - **Detection**: Multi-algorithm star detection (DAO, IRAF, centroiding)
//! - **Noise**: Realistic sensor noise simulation
//! - **Convolution**: PSF application and image filtering
//! - **Rendering**: High-quality astronomical scene generation
//! - **I/O**: FITS and standard image format support
//!
//! ## Photometry (`photometry`)
//! - **Spectral Models**: Stellar spectra and color calculations
//! - **Quantum Efficiency**: Wavelength-dependent sensor response
//! - **Filters**: Photometric band definitions and throughput curves  
//! - **Backgrounds**: Zodiacal light and scattered light modeling
//!
//! # Performance Characteristics
//!
//! - **Real-time capable**: Optimized for tracking loop frequencies (1-100 Hz)
//! - **Memory efficient**: Streaming algorithms for large star catalogs
//! - **Parallel processing**: Multi-threaded image processing and detection
//! - **Accuracy validated**: Matches real telescope performance within measurement uncertainties
//!
//! # Use Cases
//!
//! - **Algorithm development**: Test star tracking and attitude determination
//! - **Performance prediction**: Estimate detection sensitivity and accuracy
//! - **Mission design**: Optimize telescope and sensor parameters
//! - **Validation**: Compare simulation results with on-orbit data
//! - **Education**: Understand space telescope physics and image processing

pub mod algo; // misc module lives here
pub mod hardware;
pub mod image_proc; // render module lives here
pub mod photometry;
pub mod scene;
pub mod shared_args;
pub mod sims;
pub mod star_math;

// Re-export from shared for compatibility
pub use shared::units;

// Re-exports for easier access
pub use hardware::sensor::SensorConfig;
pub use hardware::telescope::TelescopeConfig;
pub use photometry::quantum_efficiency::QuantumEfficiency;
pub use photometry::spectrum::{Spectrum, CGS};
pub use photometry::trapezoid::trap_integrate;
pub use scene::Scene;
pub use shared::algo::icp::{iterative_closest_point, ICPResult};
pub use shared::image_proc::histogram_stretch::stretch_histogram;
pub use star_math::{field_diameter, filter_stars_in_field, pixel_scale, star_data_to_fluxes};
pub use starfield::catalogs::StarPosition;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

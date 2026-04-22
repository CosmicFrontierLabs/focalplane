//! Astronomical scene modeling for realistic telescope simulations.
//!
//! This module provides high-level scene abstraction for combining stellar fields,
//! instrument configurations, and observing parameters into unified simulation
//! objects. Essential for mission planning, detector characterization, and
//! astronomical survey optimization.
//!
//! # Scene Architecture
//!
//! A `Scene` represents a complete observational setup:
//! - **Instrument**: Telescope + sensor configuration with optical properties
//! - **Targets**: Star catalog with positions, magnitudes, and colors
//! - **Pointing**: Celestial coordinates defining field center
//! - **Exposure**: Integration time and observing conditions
//! - **Environment**: Background models (zodiacal light, thermal, etc.)
//!
//! # Coordinate Systems and Projections
//!
//! ## Celestial to Pixel Mapping
//! The scene handles complex coordinate transformations:
//! - **Input**: Equatorial coordinates (RA, Dec) from star catalogs
//! - **Projection**: Gnomonic (tangent plane) projection for small fields
//! - **Distortion**: Optical distortion models (radial, field curvature)
//! - **Output**: Pixel coordinates with sub-pixel precision
//!
//! ## Field of View Calculations
//! Automatic field coverage and star selection:
//! - **FOV determination**: From focal length and sensor dimensions
//! - **Margin handling**: PSF padding for edge stars
//! - **Catalog filtering**: Only stars within extended field boundaries
//! - **Flux calculation**: Realistic photoelectron rates per star
//!
//! # Rendering Pipeline Integration
//!
//! ## Two-Stage Architecture
//! 1. **Scene Creation**: Project stars, calculate fluxes (one-time setup)
//! 2. **Image Rendering**: Generate detector images with noise (per exposure)
//!
//! ## Performance Optimization
//! - **Cached projections**: Star positions computed once per pointing
//! - **Reusable renderers**: Multiple exposures without re-projection
//! - **Memory efficiency**: Minimal duplication of catalog data
//! - **Parallel rendering**: Thread-safe operations for batch processing
//!
//! # Usage Patterns
//!
//! ## Single Exposure Simulation
//! Create scene from star catalog with satellite configuration, telescope,
//! sensor, and zodiacal coordinates for rendering realistic astronomical images.
//!
//! # Astronomical Accuracy
//!
//! ## Photometric Precision
//! - **Magnitude system**: Standard Johnson-Cousins UBV photometry
//! - **Color modeling**: B-V color indices with temperature conversion
//! - **Extinction**: Atmospheric extinction models (ground-based)
//! - **Calibration**: Cross-validated with space telescope measurements
//!
//! ## Astrometric Precision
//! - **Coordinate accuracy**: Sub-milliarcsecond projection precision
//! - **Proper motion**: Support for stellar motion over time
//! - **Parallax**: Distance-dependent position shifts
//! - **Aberration**: Light-travel time and velocity corrections
//!
//! ## Detector Realism
//! - **Noise models**: Read noise, dark current, shot noise, quantization
//! - **PSF modeling**: Airy disks with optical aberrations
//! - **Quantum efficiency**: Wavelength-dependent detector response
//! - **Saturation**: Realistic well depth and blooming effects
//!
//! # Integration with Simulation Ecosystem
//!
//! ## Hardware Modeling
//! - **Telescope**: Aperture, focal length, optical efficiency
//! - **Detector**: Pixel size, QE curves, noise characteristics
//! - **Environment**: Operating temperature, background radiation
//!
//! ## Catalog Integration
//! - **Gaia DR3**: Billion-star catalog with precise astrometry
//! - **Hipparcos**: Bright star catalog with photometry
//! - **Synthetic**: Galactic model populations for survey planning
//!
//! ## Output Formats
//! - **FITS images**: Standard astronomical image format
//! - **Photometry tables**: Extracted stellar measurements
//! - **Performance metrics**: Signal-to-noise, limiting magnitudes
//! - **Diagnostic data**: Background levels, noise contributions

use crate::hardware::satellite::FocalPlaneConfig;
use crate::image_proc::render::{
    project_stars_to_focal_plane, route_stars_to_sensors, Renderer, RenderingResult, StarInFrame,
};
use crate::photometry::zodiacal::SolarAngularCoordinates;
use shared::units::LengthExt;
use starfield::catalogs::StarData;
use starfield::Equatorial;
use std::time::Duration;

/// Complete astronomical observation scene for single or multi-sensor focal planes.
///
/// Handles the full pipeline from star catalog to rendered detector images.
/// A single sensor is treated as a 1-element array, so the same code path
/// handles both single-sensor and multi-sensor (e.g. SPENCER) configurations.
///
/// # Architecture
///
/// ```text
/// Catalog → focal plane mm projection → route to sensors → per-sensor render
/// ```
///
/// Each sensor gets its own `SatelliteConfig` (QE, noise, pixel scale) and
/// produces an independent `RenderingResult`.
///
/// # Usage
///
/// For a single sensor, wrap your `SatelliteConfig` via
/// `FocalPlaneConfig::from_satellite()`. For multi-sensor arrays, construct
/// a `FocalPlaneConfig` directly with a `SensorArray`.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Focal plane configuration (telescope + sensor array + temperature)
    pub focal_plane: FocalPlaneConfig,

    /// Stars projected and routed to each sensor's pixel coordinates.
    /// Indexed by sensor index in the array.
    pub per_sensor_stars: Vec<Vec<StarInFrame>>,

    /// Celestial pointing center for the observation.
    pub pointing_center: Equatorial,

    /// Solar angular coordinates for zodiacal light calculation.
    pub zodiacal_coordinates: SolarAngularCoordinates,
}

impl Scene {
    /// Create scene from a star catalog with automatic projection and routing.
    ///
    /// Projects all catalog stars onto the focal plane with one pass, then
    /// routes them to individual sensors. PSF padding is computed from the
    /// telescope's Airy disk at the first sensor's pixel scale.
    ///
    /// # Arguments
    /// * `focal_plane` - Focal plane configuration (single or multi-sensor)
    /// * `catalog_stars` - Star catalog covering the full array field
    /// * `pointing_center` - Telescope pointing direction
    /// * `zodiacal_coordinates` - Solar position for background calculation
    pub fn from_catalog(
        focal_plane: FocalPlaneConfig,
        catalog_stars: Vec<StarData>,
        pointing_center: Equatorial,
        zodiacal_coordinates: SolarAngularCoordinates,
    ) -> Self {
        let first_sat = focal_plane
            .satellite_for_sensor(0)
            .expect("Array must have at least one sensor");
        let airy_pix = first_sat.airy_disk_pixel_space();
        let pixel_size_mm = first_sat.sensor.pixel_size().as_millimeters();
        let padding_mm = airy_pix.first_zero() * 2.0 * pixel_size_mm;

        let star_refs: Vec<&StarData> = catalog_stars.iter().collect();
        let fp_stars =
            project_stars_to_focal_plane(&star_refs, &pointing_center, &focal_plane, padding_mm);
        let per_sensor_stars = route_stars_to_sensors(&fp_stars, &focal_plane, padding_mm);

        Self {
            focal_plane,
            per_sensor_stars,
            pointing_center,
            zodiacal_coordinates,
        }
    }

    /// Create scene from pre-computed stars with known pixel positions.
    ///
    /// Bypasses catalog projection. Each inner `Vec<StarInFrame>` corresponds
    /// to one sensor in the array. For a single-sensor scene, pass
    /// `vec![stars]`.
    pub fn from_stars(
        focal_plane: FocalPlaneConfig,
        per_sensor_stars: Vec<Vec<StarInFrame>>,
        pointing_center: Equatorial,
        zodiacal_coordinates: SolarAngularCoordinates,
    ) -> Self {
        Self {
            focal_plane,
            per_sensor_stars,
            pointing_center,
            zodiacal_coordinates,
        }
    }

    /// Render all sensors and return one `RenderingResult` per sensor.
    pub fn render(&self, exposure: &Duration) -> Vec<RenderingResult> {
        self.render_with_seed(exposure, None)
    }

    /// Render all sensors with an optional base RNG seed.
    ///
    /// Each sensor gets a unique seed derived from the base seed offset by
    /// sensor index, producing reproducible but independent noise.
    pub fn render_with_seed(
        &self,
        exposure: &Duration,
        base_seed: Option<u64>,
    ) -> Vec<RenderingResult> {
        let sensor_count = self.focal_plane.array.sensor_count();
        let mut results = Vec::with_capacity(sensor_count);

        for sensor_idx in 0..sensor_count {
            let sat = self
                .focal_plane
                .satellite_for_sensor(sensor_idx)
                .expect("sensor index in range");

            let renderer = Renderer::from_stars(&self.per_sensor_stars[sensor_idx], sat);

            let seed = base_seed.map(|s| s + sensor_idx as u64);
            let rendered = renderer.render_with_seed(exposure, &self.zodiacal_coordinates, seed);

            results.push(RenderingResult {
                quantized_image: rendered.quantized_image,
                star_image: rendered.star_image,
                zodiacal_image: rendered.zodiacal_image,
                sensor_noise_image: rendered.sensor_noise_image,
                rendered_stars: rendered.rendered_stars,
                sensor_config: self.focal_plane.array.sensors[sensor_idx].sensor.clone(),
            });
        }

        results
    }

    /// Get the number of sensors in the array.
    pub fn sensor_count(&self) -> usize {
        self.focal_plane.array.sensor_count()
    }
}

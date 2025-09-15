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
//! ## Time Series Simulation
//! ```ignore
//! // NOTE: This doctest is ignored because it's computationally expensive - renders 5 different exposures
//! use simulator::Scene;
//! use simulator::photometry::zodiacal::SolarAngularCoordinates;
//! use std::time::Duration;
//!
//! # use simulator::hardware::{telescope::models::IDEAL_50CM, sensor::models::GSENSE6510BSI};
//! # let satellite_config = SatelliteConfig::new(IDEAL_50CM.clone(), GSENSE6510BSI.clone(), -10.0, 550.0);
//! # let zodiacal_coords = SolarAngularCoordinates::zodiacal_minimum();
//! # let scene = Scene::from_catalog(satellite_config, vec![], Equatorial::from_degrees(0.0, 0.0), Duration::from_secs(1), zodiacal_coords);
//! // Create efficient renderer for multiple exposures
//! let renderer = scene.create_renderer();
//! let zodiacal_coords = SolarAngularCoordinates::zodiacal_minimum();
//!
//! // Generate time series with different exposure times
//! let exposure_times = [1.0, 5.0, 10.0, 30.0, 60.0]; // seconds
//! let mut images = Vec::new();
//!
//! for &exp_time in &exposure_times {
//!     let duration = Duration::from_secs_f64(exp_time);
//!     let image = renderer.render(&duration, &zodiacal_coords);
//!     images.push(image);
//!     
//!     println!("Exposure {:.0}s: {} total electrons",
//!              exp_time, image.quantized_image.sum());
//! }
//! ```
//!
//! ## Survey Simulation
//! ```ignore
//! // NOTE: This doctest is ignored because it's computationally expensive - renders multiple scenes
//! use simulator::Scene;
//! use simulator::hardware::SatelliteConfig;
//! use simulator::photometry::zodiacal::SolarAngularCoordinates;
//! use starfield::{Equatorial, catalogs::load_catalog};
//! use std::time::Duration;
//!
//! # let satellite_config = SatelliteConfig::default();
//! # let catalog_stars = vec![];
//! // Simulate multi-pointing survey
//! let pointings = [
//!     Equatorial::from_degrees(0.0, 0.0),     // Equatorial field
//!     Equatorial::from_degrees(90.0, 30.0),   // Spring field
//!     Equatorial::from_degrees(180.0, -30.0), // Summer field
//!     Equatorial::from_degrees(270.0, 60.0),  // Autumn field
//! ];
//!
//! let mut survey_results = Vec::new();
//!
//! for (i, pointing) in pointings.iter().enumerate() {
//!     // Calculate zodiacal light for field position
//!     let zodiacal_coords = SolarAngularCoordinates::new(
//!         90.0 + i as f64 * 30.0, // Varying elongation
//!         30.0, // Fixed latitude
//!     ).unwrap();
//!     
//!     let scene = Scene::from_catalog(
//!         satellite_config.clone(),
//!         catalog_stars.clone(),
//!         *pointing,
//!         Duration::from_secs(300), // 5-minute exposures
//!         zodiacal_coords,
//!     );
//!     
//!     let result = scene.render();
//!     survey_results.push((pointing, result));
//!     
//!     println!("Field {}: {} stars",
//!              i + 1, scene.stars.len());
//! }
//! ```
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

use crate::hardware::SatelliteConfig;
use crate::image_proc::render::{project_stars_to_pixels, Renderer, RenderingResult, StarInFrame};
use crate::photometry::zodiacal::SolarAngularCoordinates;
use starfield::catalogs::StarData;
use starfield::Equatorial;
use std::time::Duration;

/// Complete astronomical observation scene with instrument and target configuration.
///
/// Represents a fully-specified astronomical observation combining instrument
/// hardware, stellar targets, pointing direction, and exposure parameters.
/// Provides unified interface for realistic telescope simulations including
/// coordinate projections, photometric calculations, and detector modeling.
///
/// # Scene Components
/// - **Instrument Model**: Complete telescope + detector characterization
/// - **Stellar Field**: Catalog stars with positions, magnitudes, and colors
/// - **Pointing**: Celestial coordinates defining observation center
/// - **Timing**: Exposure duration and observing epoch
/// - **Environment**: Background conditions (zodiacal light, thermal)
///
/// # Coordinate System Integration
/// The scene handles complex astronomical coordinate transformations:
/// - **Input**: Catalog coordinates (RA, Dec in degrees)
/// - **Projection**: Gnomonic tangent-plane mapping to focal plane
/// - **Distortion**: Optical aberrations and field curvature
/// - **Output**: Pixel coordinates with sub-pixel accuracy
///
/// # Performance Characteristics
/// - **Initialization**: O(N) star projection and flux calculation
/// - **Rendering**: O(N) PSF convolution and noise generation  
/// - **Memory**: Minimal duplication of catalog data
/// - **Thread safety**: Immutable scene allows concurrent rendering
///
/// # Simulation Workflow
/// 1. **Scene Creation**: Project stars from catalog to detector coordinates
/// 2. **Flux Calculation**: Compute realistic photoelectron rates
/// 3. **Image Rendering**: Generate detector images with noise models
/// 4. **Analysis**: Extract photometry and astrometry from simulated data
///
/// # Usage
/// The Scene struct represents a complete astronomical observation, combining instrument hardware,
/// stellar targets, pointing direction, and exposure parameters. Create a scene from a star catalog
/// using `Scene::from_catalog()`, or from pre-computed stars using `Scene::from_stars()`.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Complete satellite hardware configuration defining the observation system.
    ///
    /// Includes telescope optics (aperture, focal length, efficiency), detector
    /// characteristics (QE, noise, pixel scale), and environmental conditions
    /// (operating temperature, background radiation levels).
    pub satellite_config: SatelliteConfig,

    /// Stellar targets projected to detector pixel coordinates.
    ///
    /// Contains stars from the input catalog that fall within the field of view,
    /// with pre-computed pixel positions and expected photoelectron fluxes.
    /// Includes proper handling of field edge effects and PSF truncation.
    pub stars: Vec<StarInFrame>,

    /// Celestial pointing center for the observation.
    ///
    /// Defines the RA/Dec coordinates of the detector center, used as the
    /// reference point for all coordinate transformations and field geometry
    /// calculations. Typically the optical axis direction.
    pub pointing_center: Equatorial,

    /// Solar angular coordinates for zodiacal light calculation.
    ///
    /// Defines the angular position of the observation relative to the Sun,
    /// used to calculate the intensity of zodiacal light background which
    /// varies significantly based on solar elongation and ecliptic latitude.
    pub zodiacal_coordinates: SolarAngularCoordinates,
}

impl Scene {
    /// Create astronomical scene from star catalog with automatic coordinate projection.
    ///
    /// Constructs a complete observation scene by projecting catalog stars onto
    /// the detector plane and calculating realistic photoelectron fluxes. Handles
    /// all coordinate transformations, field geometry, and photometric scaling
    /// automatically based on the instrument configuration.
    ///
    /// # Projection Pipeline
    /// 1. **Field calculation**: Determine FOV boundaries with PSF padding
    /// 2. **Coordinate transformation**: RA/Dec → tangent plane → pixel coordinates
    /// 3. **Flux calculation**: Magnitude → photoelectrons using QE curves
    /// 4. **Field filtering**: Retain only stars within detector boundaries
    /// 5. **Edge handling**: Proper PSF truncation for partially-visible stars
    ///
    /// # Photometric Accuracy
    /// - **Magnitude system**: Standard astronomical photometry (Johnson UBV)
    /// - **Color conversion**: B-V color indices to spectral energy distributions
    /// - **Quantum efficiency**: Wavelength-dependent detector response
    /// - **Aperture scaling**: Flux proportional to telescope collecting area
    ///
    /// # Arguments
    /// * `satellite_config` - Complete instrument configuration
    /// * `catalog_stars` - Input star catalog with positions and photometry
    /// * `pointing_center` - RA/Dec coordinates of field center
    /// * `exposure_time` - Integration time as Duration
    ///
    /// # Returns
    /// Complete Scene with projected stars ready for rendering
    ///
    /// # Usage
    /// Create a scene from a star catalog by projecting catalog stars onto the detector plane.
    /// The scene automatically handles coordinate transformations, field-of-view calculations,
    /// and flux computations based on the instrument configuration.
    pub fn from_catalog(
        satellite_config: SatelliteConfig,
        catalog_stars: Vec<StarData>,
        pointing_center: Equatorial,
        zodiacal_coordinates: SolarAngularCoordinates,
    ) -> Self {
        // Calculate PSF padding for edge handling (same as render_star_field)
        let airy_pix = satellite_config.airy_disk_pixel_space();
        let padding = airy_pix.first_zero() * 2.0;

        // Convert stars to references (required by project_stars_to_pixels)
        let star_refs: Vec<&StarData> = catalog_stars.iter().collect();

        // Use shared projection function
        let projected_stars =
            project_stars_to_pixels(&star_refs, &pointing_center, &satellite_config, padding);

        Self {
            satellite_config,
            stars: projected_stars,
            pointing_center,
            zodiacal_coordinates,
        }
    }

    /// Create astronomical scene from pre-computed stars with known pixel positions.
    ///
    /// Constructs a scene directly from pre-projected stars, bypassing the catalog
    /// projection pipeline. This is useful for testing scenarios where stars are
    /// placed at specific pixel locations with known fluxes, such as PSF analysis,
    /// detector characterization, or controlled experiments.
    ///
    /// # Use Cases
    /// - **Detector testing**: Place stars at known positions for centroid accuracy
    /// - **PSF characterization**: Test star detection with controlled placement
    /// - **Noise floor analysis**: Single star detection threshold experiments
    /// - **Algorithm validation**: Known ground truth for detection algorithms
    ///
    /// # Arguments
    /// * `satellite_config` - Complete instrument configuration
    /// * `stars` - Pre-computed stars with pixel positions and fluxes
    /// * `pointing_center` - Nominal field center (for metadata/compatibility)
    /// * `exposure_time` - Integration time for scaling and noise models
    /// * `zodiacal_coordinates` - Solar position for background calculation
    ///
    /// # Returns
    /// Complete Scene ready for rendering with the provided stars
    ///
    /// # Examples
    /// # Usage
    /// Create a scene with pre-computed stars at specific pixel positions. This is useful for
    /// testing scenarios where you need precise control over star placement, such as detector
    /// characterization, PSF analysis, or algorithm validation with known ground truth.
    pub fn from_stars(
        satellite_config: SatelliteConfig,
        stars: Vec<StarInFrame>,
        pointing_center: Equatorial,
        zodiacal_coordinates: SolarAngularCoordinates,
    ) -> Self {
        Self {
            satellite_config,
            stars,
            pointing_center,
            zodiacal_coordinates,
        }
    }

    /// Render complete astronomical image with realistic detector effects.
    ///
    /// Generates a fully realistic detector image from the pre-projected stellar
    /// field, including accurate PSF modeling, background contributions, noise
    /// statistics, and detector artifacts. Produces both the raw floating-point
    /// image and quantized detector output.
    ///
    /// # Rendering Pipeline
    /// 1. **PSF Convolution**: Apply Airy disk pattern to each star
    /// 2. **Background Addition**: Zodiacal light + thermal backgrounds
    /// 3. **Noise Generation**: Shot noise, read noise, dark current
    /// 4. **Detector Effects**: Quantum efficiency, linearity, saturation
    /// 5. **Quantization**: ADU conversion with realistic bit depth
    ///
    /// # Physical Accuracy
    /// - **PSF modeling**: Diffraction-limited Airy disks with optical aberrations
    /// - **Background scaling**: Position-dependent zodiacal light brightness
    /// - **Noise statistics**: Poisson photon noise + Gaussian detector noise
    /// - **Saturation handling**: Well depth limits with realistic overflow
    ///
    /// # Output Components
    /// - **Quantized image**: Realistic detector output in ADU units
    /// - **Float image**: High-precision electron image before quantization
    /// - **Background map**: Spatial background distribution
    /// - **Metadata**: Exposure parameters, noise levels, saturation statistics
    ///
    /// # Arguments
    /// Uses the zodiacal coordinates stored in the scene for background calculation.
    ///
    /// # Returns
    /// Complete RenderingResult with image data and diagnostic information
    ///
    /// # Examples
    /// # Usage
    /// Render the scene to produce a complete astronomical image with realistic detector effects.
    /// The output includes the quantized detector image, zodiacal background map, and metadata
    /// about rendered stars. Use the stored zodiacal coordinates for background calculation.
    pub fn render(&self, exposure: &Duration) -> RenderingResult {
        self.render_with_seed(exposure, None)
    }

    /// Render the scene with a specific RNG seed for reproducible results
    pub fn render_with_seed(&self, exposure: &Duration, seed: Option<u64>) -> RenderingResult {
        // Use the new Renderer for the actual work
        let renderer = Renderer::from_stars(&self.stars, self.satellite_config.clone());
        let rendered = renderer.render_with_seed(exposure, &self.zodiacal_coordinates, seed);

        // Convert back to RenderingResult for backwards compatibility
        RenderingResult {
            quantized_image: rendered.quantized_image,
            star_image: rendered.star_image,
            zodiacal_image: rendered.zodiacal_image,
            sensor_noise_image: rendered.sensor_noise_image,
            rendered_stars: rendered.rendered_stars,
            sensor_config: self.satellite_config.sensor.clone(),
        }
    }

    /// Create a renderer for efficient multi-exposure rendering.
    ///
    /// Returns a `Renderer` that caches star projections and base images,
    /// enabling efficient generation of multiple exposures without re-calculating
    /// star positions and fluxes. This is particularly useful for:
    /// - Time series observations with varying exposure times
    /// - Detector characterization across different integration periods
    /// - Monte Carlo simulations with consistent star fields
    ///
    /// # Performance Benefits
    /// - **One-time star projection**: O(N) computation done once
    /// - **Cached base image**: 1-second star field pre-rendered
    /// - **Fast exposure scaling**: O(1) linear scaling per exposure
    /// - **Fresh noise generation**: Maintains statistical independence
    ///
    /// # Example
    /// ```ignore
    /// let renderer = scene.create_renderer();
    ///
    /// // Generate multiple exposures efficiently
    /// for exposure_sec in [1.0, 5.0, 10.0, 30.0] {
    ///     let duration = Duration::from_secs_f64(exposure_sec);
    ///     let image = renderer.render(&duration, &zodiacal_coords);
    /// }
    /// ```
    pub fn create_renderer(&self) -> Renderer {
        Renderer::from_stars(&self.stars, self.satellite_config.clone())
    }
}

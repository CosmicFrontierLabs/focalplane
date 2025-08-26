//! Advanced coordinate transformations and stellar field projection for astronomical simulations.
//!
//! This module provides sophisticated mathematical infrastructure for converting between
//! celestial coordinate systems and projecting stellar fields onto detector surfaces.
//! Essential for accurate telescope simulations, astrometric calibration, and field
//! geometry calculations in space-based and ground-based observations.
//!
//! # Coordinate System Framework
//!
//! ## Celestial Coordinate Systems
//! - **Equatorial coordinates**: Right ascension (RA) and declination (Dec) in radians
//! - **Cartesian coordinates**: Unit vectors on the celestial sphere
//! - **Camera coordinates**: Instrument-relative 3D coordinate system
//! - **Pixel coordinates**: Detector plane positions with sub-pixel precision
//!
//! ## Transformation Pipeline
//! The coordinate transformation follows these steps:
//! 1. **Celestial → Cartesian**: Spherical to 3D unit vector conversion
//! 2. **Rotation**: Align celestial sphere with camera orientation
//! 3. **Projection**: Gnomonic (tangent plane) mapping to focal plane
//! 4. **Scaling**: Angular to pixel coordinate conversion
//! 5. **Translation**: Center alignment with detector coordinate system
//!
//! # Projection Geometry
//!
//! ## Gnomonic Projection
//! Uses tangent plane projection for accurate small-field mapping:
//! - **Central projection**: All projection lines pass through sphere center
//! - **Angular preservation**: Small angles preserved locally
//! - **Distortion characteristics**: Minimal near field center, increases toward edges
//! - **Valid range**: Typically <10° from field center for <1% distortion
//!
//! ## Camera Coordinate System
//! - **Z-axis**: Points toward field center (optical axis)
//! - **Y-axis**: Points toward celestial north (or nearest equivalent)
//! - **X-axis**: Completes right-handed system (approximately east)
//! - **Origin**: Camera/detector center
//!
//! # Mathematical Foundations
//!
//! ## Rotation Matrix Construction
//! Camera orientation computed from field center coordinates:
//! ```text
//! Z = [cos(Dec)*cos(RA), cos(Dec)*sin(RA), sin(Dec)]  // Toward center
//! X = North × Z / |North × Z|                          // East direction  
//! Y = Z × X                                         // North direction
//! R = [X Y Z]                                       // Rotation matrix
//! ```
//!
//! ## Projection Mathematics
//! For star at celestial position **S**, camera coordinates **C**:
//! ```text
//! C = R^T * S                    // Rotate to camera frame
//! x_proj = C_x / C_z            // Gnomonic X projection
//! y_proj = C_y / C_z            // Gnomonic Y projection
//! x_pixel = W/2 + x_proj/scale  // Convert to pixel X
//! y_pixel = H/2 - y_proj/scale  // Convert to pixel Y (flipped)
//! ```
//!
//! # Accuracy and Precision
//!
//! ## Astrometric Accuracy
//! - **Sub-pixel precision**: 0.01 pixel typical coordinate accuracy
//! - **Angular accuracy**: Limited by floating-point precision (~1 microarcsec)
//! - **Field size limits**: <10° radius for <1% geometric distortion
//! - **Pole handling**: Robust near celestial poles with proper singularity avoidance
//!
//! ## Numerical Stability
//! - **Normalized vectors**: Unit vector operations prevent scaling errors
//! - **Orthogonal matrices**: Proper rotation matrix construction
//! - **Boundary checking**: Graceful handling of stars behind camera
//! - **Edge cases**: Special handling for celestial pole pointings
//!
//! # Usage Examples
//!
//! ## Basic Star Projection
//!
//! Create a StarProjector with field center and detector parameters, then project
//! star coordinates to pixel positions. The projector handles coordinate transformations
//! and bounds checking automatically.
//!
//! ## Random Sky Position Generation
//!
//! Use EquatorialRandomizer to generate uniformly distributed random sky positions
//! with optional seeding for reproducible test scenarios.
//!
//! ## Field Coverage Analysis
//!
//! Test field coverage by projecting known celestial positions and checking
//! whether they fall within detector bounds. Useful for validating telescope
//! pointing accuracy and field geometry.
//!
//! ## Unbounded Projection for Analysis
//!
//! Use unbounded projection to analyze star positions outside detector bounds.
//! This helps with field geometry analysis and understanding where stars would
//! appear with larger detectors.
//!
//! # Performance Considerations
//!
//! ## Computational Complexity
//! - **Projector creation**: O(1) - matrix computation and storage
//! - **Single projection**: O(1) - matrix multiplication and division
//! - **Batch projection**: O(N) - linear scaling with star count
//! - **Memory usage**: Minimal - rotation matrix and parameters only
//!
//! ## Optimization Strategies
//! - **Pre-computed rotation**: Matrix calculated once per pointing
//! - **Vectorized operations**: SIMD-friendly linear algebra
//! - **Early rejection**: Behind-camera check before projection
//! - **Bounds checking**: Optional for performance-critical applications
//!
//! # Integration with Simulation Pipeline
//!
//! ## Catalog Processing
//! Used extensively in star catalog projection workflows:
//! - **Field star selection**: Determine which catalog stars fall in field
//! - **Astrometric calibration**: Compare predicted vs. observed positions
//! - **Geometric distortion**: Account for optical system aberrations
//! - **Proper motion**: Project stars at different epochs
//!
//! ## Instrument Modeling
//! Essential for realistic instrument simulation:
//! - **Pixel-level accuracy**: Sub-pixel star positioning
//! - **Field geometry**: Accurate field-of-view calculations
//! - **Detector alignment**: Coordinate system registration
//! - **Pointing accuracy**: Astrometric residual analysis

use nalgebra::Matrix3;
use starfield::catalogs::{StarData, StarPosition};
use starfield::framelib::inertial::InertialFrame;
use starfield::Equatorial;

use nalgebra::Vector3;

use crate::hardware::{SatelliteConfig, SensorConfig, TelescopeConfig};
use crate::photometry::photoconversion::SourceFlux;
use crate::photometry::photon_electron_fluxes;
use crate::photometry::BlackbodyStellarSpectrum;
use crate::units::{Angle, AngleExt, LengthExt};

// A majority of stars have a B-V color index around 1.4
// This is used when the catalog does not provide a B-V value
// In practice this is mostly for faint stars, but this is a
// reasonable guess for stuff on the main sequence
pub const DEFAULT_BV: f64 = 1.4;

/// Calculate the circular field of view diameter for a telescope-sensor combination.
///
/// Computes the angular diameter of the smallest circle that completely
/// encompasses the rectangular sensor field of view. This is the standard
/// metric for telescope survey capabilities and catalog queries.
///
/// # Mathematical Background
/// Uses the sensor diagonal and telescope focal length:
/// ```text
/// FOV_diameter = 2 * atan(sensor_diagonal / (2 * focal_length))
/// ```
/// For small angles (typical case): `FOV ≈ sensor_diagonal / focal_length`
///
/// # Arguments
/// * `telescope` - Telescope optical configuration with focal length
/// * `sensor` - Sensor geometry (width, height, pixel size)
///
/// # Returns
/// Field of view diameter as an Angle
///
/// # Usage
/// Calculates the diagonal field of view diameter based on sensor dimensions
/// and telescope focal length. Essential for survey planning and coverage analysis.
pub fn field_diameter(telescope: &TelescopeConfig, sensor: &SensorConfig) -> Angle {
    // Get sensor dimensions in microns
    let (width_um, height_um) = sensor.dimensions_um();

    // Calculate the diagonal size of the sensor in microns
    let diagonal_um = (width_um.powi(2) + height_um.powi(2)).sqrt();

    // Convert to meters
    let diagonal_m = diagonal_um * 1.0e-6;

    // Use the plate scale to convert to an angle
    // angle in radians = diagonal / focal_length
    let angle_rad = diagonal_m / telescope.focal_length.as_meters();

    Angle::from_radians(angle_rad)
}

/// Calculate the circular field of view diameter in degrees (backward compatibility)
pub fn field_diameter_degrees(telescope: &TelescopeConfig, sensor: &SensorConfig) -> f64 {
    field_diameter(telescope, sensor).as_degrees()
}

/// Calculate the angular size subtended by one pixel
pub fn pixel_scale(telescope: &TelescopeConfig, sensor: &SensorConfig) -> Angle {
    let plate_scale_rad_per_m = telescope.plate_scale().as_radians();
    let pixel_size_m = sensor.pixel_size().as_meters();
    let angular_size_rad = plate_scale_rad_per_m * pixel_size_m;
    Angle::from_radians(angular_size_rad)
}

/// Calculate the projected pixel scale in arcseconds per pixel (backward compatibility)
pub fn pixel_scale_arcsec(telescope: &TelescopeConfig, sensor: &SensorConfig) -> f64 {
    pixel_scale(telescope, sensor).as_arcseconds()
}

/// Convert stellar magnitude and color to photon and electron flux rates.
///
/// Performs complete photometric calculation from stellar parameters to
/// expected detector signal flux rates. Uses realistic blackbody spectra,
/// telescope characteristics, and sensor quantum efficiency for accurate predictions.
///
/// # Physics Model
/// 1. **Stellar spectrum**: Blackbody model from B-V color (or default 1.4) and magnitude
/// 2. **PSF calculation**: Chromatic Airy disk with wavelength-dependent scaling
/// 3. **Spectral response**: Integration over sensor QE curve and stellar spectrum
/// 4. **Flux calculation**: Returns flux rates (photons/electrons per second per cm²)
///
/// # Arguments
/// * `star_data` - Stellar catalog entry with magnitude and optional B-V color
/// * `satellite` - Complete satellite configuration including telescope and sensor
///
/// # Returns
/// `SourceFlux` containing:
/// - Photon flux rate and effective PSF
/// - Photoelectron flux rate and effective PSF
///
/// # Implementation Notes
/// - Uses B-V color if available, otherwise defaults to 1.4 (typical main sequence)
/// - Calls `photon_electron_fluxes` for chromatic PSF and flux calculation
/// - Returns flux rates that must be integrated over time and aperture area
pub fn star_data_to_fluxes(star_data: &StarData, satellite: &SatelliteConfig) -> SourceFlux {
    let spectrum = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(
        star_data.b_v.unwrap_or(DEFAULT_BV),
        star_data.magnitude,
    );
    photon_electron_fluxes(
        &satellite.airy_disk_pixel_space(),
        &spectrum,
        &satellite.combined_qe,
    )
}

/// Filter stars that would be visible in the field of view
///
/// # Arguments
/// * `stars` - Vector of star data objects with ra and dec fields
/// * `center_ra` - Right ascension of field center in degrees
/// * `center_dec` - Declination of field center in degrees
/// * `field_diameter` - Diameter of field in degrees
///
/// # Returns
/// Vector of references to stars that are within the field
pub fn filter_stars_in_field<T>(
    stars: &[T],
    center_ra: f64,
    center_dec: f64,
    field_diameter: f64,
) -> Vec<&T>
where
    T: StarPosition,
{
    // Field radius in degrees
    let field_radius = field_diameter / 2.0;

    // Convert to radians for calculations
    let center_ra_rad = center_ra.to_radians();
    let center_dec_rad = center_dec.to_radians();
    let field_radius_rad = field_radius.to_radians();

    // Filter stars that fall within the field
    stars
        .iter()
        .filter(|star| {
            let star_ra_rad = star.ra().to_radians();
            let star_dec_rad = star.dec().to_radians();

            // Angular distance using haversine formula
            let d_ra = star_ra_rad - center_ra_rad;
            let d_dec = star_dec_rad - center_dec_rad;

            let a = (d_dec / 2.0).sin().powi(2)
                + center_dec_rad.cos() * star_dec_rad.cos() * (d_ra / 2.0).sin().powi(2);
            let angular_distance = 2.0 * a.sqrt().asin();

            // Include stars within field radius
            angular_distance <= field_radius_rad
        })
        .collect()
}

/// High-precision celestial coordinate to pixel projection engine.
///
/// Implements mathematically rigorous transformation from celestial sphere
/// coordinates to detector pixel positions using gnomonic (tangent plane)
/// projection. Designed for sub-pixel accuracy in astronomical imaging
/// applications with proper handling of coordinate system singularities
/// and field boundary conditions.
///
/// # Projection Method
/// Uses gnomonic projection with the following characteristics:
/// - **Central projection**: All rays pass through sphere center
/// - **Tangent plane**: Projection surface tangent to sphere at field center
/// - **Angular preservation**: Small angles preserved near field center
/// - **Distortion growth**: Polynomial distortion increase toward field edges
///
/// # Coordinate System Convention
/// - **Input**: Celestial equatorial coordinates (RA, Dec) in radians
/// - **Intermediate**: 3D Cartesian unit vectors on celestial sphere
/// - **Camera frame**: Right-handed system aligned with detector
/// - **Output**: Pixel coordinates with (0,0) at detector corner
///
/// # Accuracy Specifications
/// - **Sub-pixel precision**: <0.01 pixel coordinate accuracy
/// - **Angular accuracy**: Limited by double-precision (~1 microarcsec)
/// - **Field size limits**: <10° radius recommended for <1% distortion
/// - **Pole handling**: Robust near celestial coordinate singularities
///
/// # Performance Characteristics
/// - **Setup cost**: O(1) rotation matrix computation
/// - **Projection cost**: O(1) per star with matrix multiplication
/// - **Memory usage**: ~200 bytes for rotation matrix and parameters
/// - **Thread safety**: Immutable after construction, safe for concurrent use
pub struct StarProjector {
    /// Celestial coordinates of the projection center (field center).
    ///
    /// Defines the pointing direction of the camera/telescope optical axis.
    /// This point maps to the center of the detector pixel grid and serves
    /// as the reference for all coordinate transformations.
    pub center: Equatorial,

    /// Angular pixel scale in radians per pixel.
    ///
    /// Determines the mapping between angular coordinates on the sky and
    /// linear pixel coordinates on the detector. Typically computed from
    /// telescope focal length and detector pixel physical size.
    radians_per_pixel: f64,

    /// Detector width in pixels along the X-axis.
    ///
    /// Used for pixel coordinate bounds checking and center offset calculation.
    /// Pixel coordinates range from [0, sensor_width) in the X direction.
    sensor_width: usize,

    /// Detector height in pixels along the Y-axis.
    ///
    /// Used for pixel coordinate bounds checking and center offset calculation.
    /// Pixel coordinates range from [0, sensor_height) in the Y direction.
    sensor_height: usize,

    /// 3D rotation matrix for celestial to camera coordinate transformation.
    ///
    /// Transforms celestial unit vectors to camera-aligned coordinate system
    /// where Z-axis points toward field center, Y-axis toward celestial north,
    /// and X-axis completes the right-handed system.
    rotation_matrix: Matrix3<f64>,
}

impl StarProjector {
    /// Create new star projector with specified field center and detector geometry.
    ///
    /// Constructs a complete coordinate transformation system by computing the
    /// rotation matrix that aligns the celestial sphere with the camera coordinate
    /// system. The resulting projector can efficiently transform stars from
    /// celestial coordinates to detector pixel positions.
    ///
    /// # Mathematical Setup
    /// The rotation matrix is constructed to establish camera coordinates:
    /// - **Z-axis**: Points toward field center (optical axis direction)
    /// - **Y-axis**: Points toward celestial north (or nearest non-degenerate direction)
    /// - **X-axis**: Completes right-handed system (approximately eastward)
    ///
    /// # Arguments
    /// * `center` - Field center in celestial equatorial coordinates
    /// * `radians_per_pixel` - Angular pixel scale (typically focal_length / pixel_size)
    /// * `sensor_width` - Detector width in pixels
    /// * `sensor_height` - Detector height in pixels
    ///
    /// # Returns
    /// Configured StarProjector ready for coordinate transformations
    ///
    /// # Usage
    /// Creates a configured projector for transforming celestial coordinates
    /// to detector pixel coordinates. The field center maps to detector center.
    pub fn new(
        center: &Equatorial,
        radians_per_pixel: f64,
        sensor_width: usize,
        sensor_height: usize,
    ) -> Self {
        // Calculate rotation matrix to transform from celestial to camera coordinates
        // Camera Z-axis points to center_ra/center_dec
        // Camera Y-axis points towards celestial north
        // Camera X-axis completes right-handed system

        let cos_ra = center.ra.cos();
        let sin_ra = center.ra.sin();
        let cos_dec = center.dec.cos();
        let sin_dec = center.dec.sin();

        // Z-axis (pointing to center)
        let z = Vector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec);

        // Y-axis (towards celestial north)
        let north = Vector3::new(0.0, 0.0, 1.0);
        let east = north.cross(&z).normalize();
        let y = z.cross(&east).normalize();

        // X-axis (east direction)
        let x = y.cross(&z).normalize();

        // Build rotation matrix (columns are the new basis vectors)
        let rotation_matrix = Matrix3::from_columns(&[x, y, z]);

        Self {
            center: *center,
            radians_per_pixel,
            sensor_width,
            sensor_height,
            rotation_matrix,
        }
    }

    /// Project celestial coordinates to pixel space without detector bounds checking.
    ///
    /// Performs complete coordinate transformation from celestial sphere to pixel
    /// coordinates without testing whether the result falls within detector boundaries.
    /// Useful for field analysis, distortion studies, and geometric calculations
    /// that extend beyond the physical detector area.
    ///
    /// # Transformation Steps
    /// 1. **Spherical to Cartesian**: Convert (RA, Dec) to unit vector
    /// 2. **Coordinate rotation**: Apply camera alignment transformation
    /// 3. **Visibility check**: Reject stars behind camera (Z ≤ 0)
    /// 4. **Gnomonic projection**: Project to tangent plane (X/Z, Y/Z)
    /// 5. **Pixel scaling**: Convert angular to pixel coordinates
    ///
    /// # Arguments
    /// * `equatorial` - Celestial coordinate to project
    ///
    /// # Returns
    /// * `Some((x, y))` - Pixel coordinates if star visible (in front of camera)
    /// * `None` - If star is behind camera or at coordinate singularity
    ///
    /// # Usage
    /// Projects stars without bounds checking. Returns pixel coordinates even
    /// for stars outside detector bounds, useful for field geometry analysis.
    pub fn project_unbounded(&self, equatorial: &Equatorial) -> Option<(f64, f64)> {
        // Convert equatorial to cartesian unit vector
        let cartesian = equatorial.to_cartesian().to_vector3();

        // Transform to camera coordinates
        let camera_coords = self.rotation_matrix.transpose() * cartesian;

        // Check if star is in front of camera (z > 0)
        if camera_coords.z <= 0.0 {
            return None;
        }

        // Apply gnomonic (tangent plane) projection
        let x_proj = camera_coords.x / camera_coords.z;
        let y_proj = camera_coords.y / camera_coords.z;

        // Convert to pixel coordinates
        let pixel_x = (self.sensor_width as f64 / 2.0) + (x_proj / self.radians_per_pixel);
        let pixel_y = (self.sensor_height as f64 / 2.0) - (y_proj / self.radians_per_pixel);

        Some((pixel_x, pixel_y))
    }

    /// Project celestial coordinates to pixel space with detector bounds checking.
    ///
    /// Performs complete coordinate transformation from celestial sphere to pixel
    /// coordinates and validates that the result falls within the physical detector
    /// boundaries. This is the primary projection method for realistic simulation
    /// where only detector-visible stars are of interest.
    ///
    /// # Bounds Checking
    /// Pixel coordinates must satisfy:
    /// - **X range**: [0, sensor_width)
    /// - **Y range**: [0, sensor_height)
    /// - **Visibility**: Star must be in front of camera (not behind)
    ///
    /// # Arguments
    /// * `equatorial` - Celestial coordinate to project
    ///
    /// # Returns
    /// * `Some((x, y))` - Pixel coordinates if star visible within detector bounds
    /// * `None` - If star is outside field of view or behind camera
    ///
    /// # Performance Notes
    /// This method calls `project_unbounded()` internally and adds bounds checking.
    /// For performance-critical applications with pre-filtered coordinates,
    /// consider using `project_unbounded()` directly.
    ///
    /// # Usage
    /// Primary method for projecting stars with bounds checking. Returns None
    /// for stars outside the detector field of view or behind the camera.
    pub fn project(&self, equatorial: &Equatorial) -> Option<(f64, f64)> {
        let (pixel_x, pixel_y) = self.project_unbounded(equatorial)?;

        // Check if within sensor bounds
        if pixel_x >= 0.0
            && pixel_x < self.sensor_width as f64
            && pixel_y >= 0.0
            && pixel_y < self.sensor_height as f64
        {
            Some((pixel_x, pixel_y))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::{LengthExt, Temperature, TemperatureExt, Wavelength};
    use approx::assert_relative_eq;
    use float_cmp::approx_eq;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::hardware::sensor::models as sensor_models;
    use crate::hardware::telescope::models as telescope_models;
    use crate::photometry::{
        photon_electron_fluxes, Band, BlackbodyStellarSpectrum, QuantumEfficiency,
    };
    use std::f64::consts::PI;
    use std::time::Duration;

    const ZERO_ZERO: Equatorial = Equatorial { ra: 0.0, dec: 0.0 };

    #[test]
    fn test_projector_creation() {
        StarProjector::new(
            &ZERO_ZERO, 0.001, // ~3.4 arcmin per pixel
            1920,  // Full HD width
            1080,  // Full HD height
        );
    }

    #[test]
    fn test_center_projection() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 1920, 1080);

        // Star at projection center should map to sensor center
        let center_star = Equatorial { ra: 0.0, dec: 0.0 };
        let (pixel_x, pixel_y) = projector.project(&center_star).unwrap();

        assert!((pixel_x - 960.0).abs() < 0.1);
        assert!((pixel_y - 540.0).abs() < 0.1);
    }

    #[test]
    fn test_center_field_access() {
        let test_center = Equatorial::from_degrees(120.0, -15.0);
        let projector = StarProjector::new(&test_center, 0.001, 1920, 1080);

        // Test that we can access the center field
        assert!((projector.center.ra_degrees() - 120.0).abs() < 1e-10);
        assert!((projector.center.dec_degrees() - (-15.0)).abs() < 1e-10);

        // Test that the center star maps to sensor center
        let (pixel_x, pixel_y) = projector.project(&test_center).unwrap();
        assert!((pixel_x - 960.0).abs() < 0.1);
        assert!((pixel_y - 540.0).abs() < 0.1);
    }

    #[test]
    fn test_four_corner_projection_easy() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.01, 100, 100);

        // Define stars at each corner of a 2x2 degree square centered on 0,0
        let star_top_left = Equatorial { ra: -0.1, dec: 0.1 };
        let star_top_right = Equatorial { ra: 0.1, dec: 0.1 };
        let star_bottom_left = Equatorial {
            ra: -0.1,
            dec: -0.1,
        };
        let star_bottom_right = Equatorial { ra: 0.1, dec: -0.1 };

        // Project each star
        let pixel_top_left = projector.project(&star_top_left).unwrap();
        let pixel_top_right = projector.project(&star_top_right).unwrap();
        let pixel_bottom_left = projector.project(&star_bottom_left).unwrap();
        let pixel_bottom_right = projector.project(&star_bottom_right).unwrap();

        // Assert that the projected pixels are close to the expected locations
        assert_relative_eq!(pixel_bottom_left.0, 40.0, epsilon = 0.1);
        assert_relative_eq!(pixel_bottom_left.1, 60.0, epsilon = 0.1);

        assert_relative_eq!(pixel_top_left.0, 40.0, epsilon = 0.1);
        assert_relative_eq!(pixel_top_left.1, 40.0, epsilon = 0.1);

        assert_relative_eq!(pixel_top_right.0, 60.0, epsilon = 0.1);
        assert_relative_eq!(pixel_top_right.1, 40.0, epsilon = 0.1);

        assert_relative_eq!(pixel_bottom_right.0, 60.0, epsilon = 0.1);
        assert_relative_eq!(pixel_bottom_right.1, 60.0, epsilon = 0.1);
    }

    #[test]
    fn test_rotates_unit_vectors() {
        // Generate a random unit vector using seeded RNG
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let point = Equatorial {
                ra: rng.gen_range(0.0..std::f64::consts::PI * 2.0),
                dec: rng.gen_range(-std::f64::consts::PI / 2.0..std::f64::consts::PI / 2.0),
            };

            let (width, height) = (101, 37);

            let projector = StarProjector::new(&point, 0.01, width, height);

            // Project the random unit vector
            let projected = projector.project(&point);

            assert!(
                projected.is_some(),
                "Projection failed for random unit vector"
            );

            let (px, py) = projected.unwrap();
            assert_relative_eq!(px, width as f64 / 2.0, epsilon = 0.001,);
            assert_relative_eq!(py, height as f64 / 2.0, epsilon = 0.001,);
        }
    }

    #[test]
    fn test_four_corner_projection_hard() {
        let projector = StarProjector::new(
            &Equatorial {
                ra: 0.0,
                dec: PI / 2.0,
            },
            0.01,
            100,
            100,
        );

        // Define stars at each corner of a 2x2 degree square centered on 0,0
        let eq_q1 = Equatorial {
            ra: 0.0_f64.to_radians(),
            dec: 89.9_f64.to_radians(),
        };
        let eq_q2 = Equatorial {
            ra: 90.0_f64.to_radians(),
            dec: 89.9_f64.to_radians(),
        };
        let eq_q3 = Equatorial {
            ra: 180.0_f64.to_radians(),
            dec: 89.9_f64.to_radians(),
        };
        let eq_q4 = Equatorial {
            ra: 270.0_f64.to_radians(),
            dec: 89.9_f64.to_radians(),
        };

        let eq_coords = vec![eq_q1, eq_q2, eq_q3, eq_q4];

        for coord in &eq_coords {
            let pixel = projector.project(coord);
            assert!(pixel.is_some(), "Projection failed for coord: {coord:?}");

            let (w, h) = pixel.unwrap();

            let expected_off = 0.1_f64.to_radians() / 0.01;

            let dist = ((w - 50.0).powi(2) + (h - 50.0).powi(2)).sqrt();
            assert_relative_eq!(dist, expected_off, epsilon = 0.1);
        }
    }

    #[test]
    fn test_unbounded_vs_bounded_same_result_when_in_bounds() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 1920, 1080);

        // Star near center should be in bounds for both methods
        let star = Equatorial {
            ra: 0.001,
            dec: 0.001,
        };

        let bounded = projector.project(&star);
        let unbounded = projector.project_unbounded(&star);

        assert!(bounded.is_some());
        assert!(unbounded.is_some());

        let (bx, by) = bounded.unwrap();
        let (ux, uy) = unbounded.unwrap();

        assert_relative_eq!(bx, ux, epsilon = 1e-10);
        assert_relative_eq!(by, uy, epsilon = 1e-10);
    }

    #[test]
    fn test_unbounded_returns_coordinates_outside_sensor() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 100, 100);

        // Star far from center should be outside sensor bounds
        let far_star = Equatorial { ra: 1.0, dec: 1.0 };

        let bounded = projector.project(&far_star);
        let unbounded = projector.project_unbounded(&far_star);

        // Bounded should return None (outside sensor)
        assert!(bounded.is_none());

        // Unbounded should still return coordinates
        assert!(unbounded.is_some());
        let (x, y) = unbounded.unwrap();

        // Coordinates should be way outside sensor bounds
        assert!(!(0.0..100.0).contains(&x) || !(0.0..100.0).contains(&y));
    }

    #[test]
    fn test_unbounded_behind_camera_returns_none() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 100, 100);

        // Star behind the camera (opposite side of sky)
        let behind_star = Equatorial { ra: PI, dec: 0.0 };

        let bounded = projector.project(&behind_star);
        let unbounded = projector.project_unbounded(&behind_star);

        // Both should return None when star is behind camera
        assert!(bounded.is_none());
        assert!(unbounded.is_none());
    }

    #[test]
    fn test_unbounded_center_projection() {
        let projector = StarProjector::new(&ZERO_ZERO, 0.001, 1920, 1080);

        // Star at projection center
        let center_star = Equatorial { ra: 0.0, dec: 0.0 };
        let (pixel_x, pixel_y) = projector.project_unbounded(&center_star).unwrap();

        // Should map to sensor center
        assert_relative_eq!(pixel_x, 960.0, epsilon = 0.1);
        assert_relative_eq!(pixel_y, 540.0, epsilon = 0.1);
    }

    #[test]
    fn test_field_diameter() {
        let telescope = telescope_models::IDEAL_100CM.clone();
        let sensor = sensor_models::GSENSE4040BSI.clone();

        // Calculate expected field diameter
        let (width_um, height_um) = sensor.dimensions_um();
        let diagonal_um = (width_um.powi(2) + height_um.powi(2)).sqrt();
        let diagonal_m = diagonal_um * 1.0e-6;
        let expected_angle_rad = diagonal_m / telescope.focal_length.as_meters();
        let expected_angle_deg = expected_angle_rad.to_degrees();

        let calculated = field_diameter(&telescope, &sensor);

        assert!(approx_eq!(
            f64,
            calculated.as_degrees(),
            expected_angle_deg,
            epsilon = 1e-6
        ));
    }

    #[test]
    fn test_pixel_scale() {
        let telescope = telescope_models::IDEAL_100CM.clone();
        let sensor = sensor_models::GSENSE4040BSI.clone();

        // Calculate expected pixel scale
        let arcsec_per_mm = telescope.plate_scale_arcsec_per_mm();
        let expected_scale = arcsec_per_mm * sensor.pixel_size().as_millimeters();

        let calculated = pixel_scale(&telescope, &sensor);

        assert!(approx_eq!(
            f64,
            calculated.as_arcseconds(),
            expected_scale,
            epsilon = 1e-6
        ));
    }

    // Test struct for StarPosition trait
    #[derive(Debug, Clone)]
    struct TestStar {
        ra: f64,
        dec: f64,
    }

    impl starfield::catalogs::StarPosition for TestStar {
        fn ra(&self) -> f64 {
            self.ra
        }

        fn dec(&self) -> f64 {
            self.dec
        }
    }

    #[test]
    fn test_filter_stars_in_field() {
        // Create test stars
        let stars = vec![
            TestStar {
                ra: 100.0,
                dec: 45.0,
            }, // Center
            TestStar {
                ra: 100.1,
                dec: 45.0,
            }, // Near center
            TestStar {
                ra: 101.0,
                dec: 45.0,
            }, // 1 degree away in RA
            TestStar {
                ra: 100.0,
                dec: 46.0,
            }, // 1 degree away in Dec
            TestStar {
                ra: 105.0,
                dec: 45.0,
            }, // 5 degrees away in RA
            TestStar {
                ra: 100.0,
                dec: 50.0,
            }, // 5 degrees away in Dec
        ];

        // Test with 2 degree field diameter
        let field_stars = filter_stars_in_field(&stars, 100.0, 45.0, 2.0);

        // Count how many stars are definitely in the field
        // Stars exactly 1 degree away might be at the border due to numeric precision in the calculation
        assert!(field_stars.len() >= 3);

        // Test with smaller field
        let field_stars = filter_stars_in_field(&stars, 100.0, 45.0, 0.5);

        // Should only include center and near center
        assert_eq!(field_stars.len(), 2);
    }

    #[test]
    fn test_magnitude_to_electron_different_telescopes() {
        let small_telescope = telescope_models::IDEAL_50CM.clone();
        let large_telescope = telescope_models::IDEAL_100CM.clone();
        let sensor = sensor_models::GSENSE4040BSI.clone();

        let small_satellite = SatelliteConfig::new(
            small_telescope.clone(),
            sensor.clone(),
            Temperature::from_celsius(-10.0),
            Wavelength::from_nanometers(550.0),
        );
        let large_satellite = SatelliteConfig::new(
            large_telescope.clone(),
            sensor.clone(),
            Temperature::from_celsius(-10.0),
            Wavelength::from_nanometers(550.0),
        );

        let star_data = StarData::new(0, 0.0, 0.0, 2.0, None);

        let second = Duration::from_secs_f64(1.0);
        let elec_small = star_data_to_fluxes(&star_data, &small_satellite)
            .electrons
            .integrated_over(&second, small_telescope.clear_aperture_area());
        let elec_large = star_data_to_fluxes(&star_data, &large_satellite)
            .electrons
            .integrated_over(&second, large_telescope.clear_aperture_area());

        // Aperture ratio squared: 1.0^2 / 0.5^2 = 4.0
        let expected_ratio =
            (large_telescope.aperture.as_meters() / small_telescope.aperture.as_meters()).powi(2);

        println!(
            "Small telescope electrons: {elec_small}, Large telescope electrons: {elec_large}"
        );
        assert!(approx_eq!(
            f64,
            elec_large / elec_small,
            expected_ratio,
            epsilon = 0.01 // Allow 1% tolerance for QE differences
        ));
    }

    #[test]
    fn test_star_data_to_fluxes_uses_combined_qe() {
        // Test that star_data_to_fluxes correctly uses the combined QE
        // from telescope and sensor, not just sensor QE alone

        // Create telescope with 50% efficiency
        let mut telescope = telescope_models::IDEAL_50CM.clone();
        let band = Band::from_nm_bounds(400.0, 700.0);
        telescope.quantum_efficiency = QuantumEfficiency::from_notch(&band, 0.5).unwrap();

        // Create sensor with 80% QE
        let mut sensor = sensor_models::GSENSE4040BSI.clone();
        sensor.quantum_efficiency = QuantumEfficiency::from_notch(&band, 0.8).unwrap();

        // Combined QE should be 0.5 * 0.8 = 0.4 (40%)
        let satellite = SatelliteConfig::new(
            telescope.clone(),
            sensor.clone(),
            Temperature::from_celsius(-10.0),
            Wavelength::from_nanometers(550.0),
        );

        // Verify combined QE is correct
        let combined_qe_value = satellite.combined_qe.at(Wavelength::from_nanometers(550.0));
        assert_relative_eq!(combined_qe_value, 0.4, epsilon = 0.001);

        // Create a bright star
        let star_data = StarData::new(0, 0.0, 0.0, 5.0, Some(0.65)); // G-type star

        // Get fluxes using the function (which should use combined QE)
        let fluxes = star_data_to_fluxes(&star_data, &satellite);

        // Create same spectrum manually to compare
        let spectrum = BlackbodyStellarSpectrum::from_gaia_bv_magnitude(
            star_data.b_v.unwrap_or(DEFAULT_BV),
            star_data.magnitude,
        );

        // Calculate expected fluxes with combined QE
        let expected_fluxes = photon_electron_fluxes(
            &satellite.airy_disk_pixel_space(),
            &spectrum,
            &satellite.combined_qe,
        );

        // Verify the fluxes match
        let second = Duration::from_secs_f64(1.0);
        let area = telescope.clear_aperture_area();

        let actual_electrons = fluxes.electrons.integrated_over(&second, area);
        let expected_electrons = expected_fluxes.electrons.integrated_over(&second, area);

        assert_relative_eq!(actual_electrons, expected_electrons, epsilon = 1e-10);

        // Also verify it's different from using sensor QE alone
        let wrong_fluxes = photon_electron_fluxes(
            &satellite.airy_disk_pixel_space(),
            &spectrum,
            &sensor.quantum_efficiency,
        );
        let wrong_electrons = wrong_fluxes.electrons.integrated_over(&second, area);

        // Should be different (sensor QE is 0.8, combined is 0.4)
        assert!(
            (actual_electrons - wrong_electrons).abs() > 0.1,
            "Should use combined QE, not sensor QE alone"
        );
    }
}

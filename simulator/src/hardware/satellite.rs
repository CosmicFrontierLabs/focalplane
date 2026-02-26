use super::{sensor::SensorConfig, sensor_array::SensorArray, telescope::TelescopeConfig};
use crate::photometry::QuantumEfficiency;
use shared::image_proc::airy::PixelScaledAiryDisk;
use shared::units::{Angle, AngleExt, Length, LengthExt, Temperature};

/// Complete satellite configuration combining telescope optics and sensor.
///
/// Represents an integrated space telescope system with all parameters needed
/// for realistic astronomical observation simulations. Combines optical and
/// detector characteristics with environmental conditions.
///
/// # Key Capabilities
///
/// - **Optical calculations**: Field of view, plate scale, Airy disk sizing
/// - **PSF modeling**: Diffraction-limited point spread functions
/// - **Sampling optimization**: Adjustable focal length for optimal pixel sampling
/// - **System integration**: Unified telescope-sensor parameter access
///
/// # Physics Models
///
/// The configuration enables realistic modeling of:
/// - Diffraction-limited optics with wavelength-dependent PSF
/// - Detector quantum efficiency and noise characteristics
/// - Temperature-dependent dark current
/// - Geometric field-of-view calculations
///
#[derive(Debug, Clone)]
pub struct SatelliteConfig {
    /// Telescope optical configuration (aperture, focal length, efficiency)
    pub telescope: TelescopeConfig,
    /// Sensor detector configuration (QE, noise, geometry)
    pub sensor: SensorConfig,
    /// Operating temperature (affects dark current)
    pub temperature: Temperature,
    /// Combined quantum efficiency of telescope optics and sensor
    pub combined_qe: QuantumEfficiency,
}

impl SatelliteConfig {
    /// Create a new satellite configuration.
    ///
    /// Combines telescope optics, sensor characteristics, and operating
    /// conditions into a unified configuration for space telescope simulation.
    ///
    /// # Arguments
    /// * `telescope` - Optical system configuration (aperture, focal length, etc.)
    /// * `sensor` - Detector configuration (QE, noise, pixel size, etc.)  
    /// * `temperature` - Operating temperature (affects thermal noise)
    ///
    /// # Returns
    /// New SatelliteConfig ready for astronomical simulations
    ///
    pub fn new(telescope: TelescopeConfig, sensor: SensorConfig, temperature: Temperature) -> Self {
        // Calculate combined QE from telescope and sensor
        let combined_qe =
            QuantumEfficiency::product(&telescope.quantum_efficiency, &sensor.quantum_efficiency)
                .expect("Failed to combine telescope and sensor QE curves");

        Self {
            telescope,
            sensor,
            temperature,
            combined_qe,
        }
    }

    /// Get the effective collecting area accounting for telescope efficiency.
    ///
    /// Get the plate scale (angular size per unit focal plane distance)
    pub fn plate_scale(&self) -> Angle {
        self.telescope.plate_scale()
    }

    /// Get the plate scale in arcseconds per millimeter at the focal plane.
    ///
    /// Fundamental optical property relating angular sky coordinates
    /// to linear focal plane dimensions.
    ///
    /// # Returns
    /// Plate scale in arcseconds per millimeter
    pub fn plate_scale_arcsec_per_mm(&self) -> f64 {
        self.telescope.plate_scale_arcsec_per_mm()
    }

    /// Get the plate scale per pixel (angular size subtended by one pixel)
    pub fn plate_scale_per_pixel(&self) -> Angle {
        let plate_scale_rad_per_m = self.telescope.plate_scale().as_radians();
        let pixel_size_m = self.sensor.pixel_size().as_meters();
        let angular_size_rad = plate_scale_rad_per_m * pixel_size_m;
        Angle::from_radians(angular_size_rad)
    }

    /// Get the plate scale in arcseconds per pixel.
    ///
    /// Critical parameter for astrometric accuracy and detection algorithms.
    /// Determines the angular resolution of the imaging system.
    ///
    /// # Returns
    /// Plate scale in arcseconds per pixel
    ///
    pub fn plate_scale_arcsec_per_pixel(&self) -> f64 {
        self.plate_scale_per_pixel().as_arcseconds()
    }

    /// Get the field of view for the sensor
    ///
    /// Returns the angular dimensions of the full sensor array
    pub fn field_of_view(&self) -> (Angle, Angle) {
        let angular_per_pixel = self.plate_scale_per_pixel();
        let (width, height) = self.sensor.dimensions.get_pixel_width_height();
        let width_angle = angular_per_pixel * (width as f64);
        let height_angle = angular_per_pixel * (height as f64);
        (width_angle, height_angle)
    }

    /// Calculate the field of view in steradians
    ///
    /// Returns the solid angle covered by the full sensor in steradians,
    /// using small angle approximation (valid for typical telescope FOVs).
    ///
    /// # Returns
    /// Field of view in steradians
    ///
    pub fn field_of_view_steradians(&self) -> f64 {
        let (width_angle, height_angle) = self.field_of_view();
        // Small angle approximation for solid angle
        width_angle.as_radians() * height_angle.as_radians()
    }

    /// Create a PixelScaledAiryDisk in pixel space for this satellite configuration
    ///
    /// # Arguments
    /// * `wavelength_nm` - Observing wavelength in nanometers
    ///
    /// # Returns
    /// A PixelScaledAiryDisk scaled to pixels for this telescope/sensor combination
    pub fn airy_disk_pixel_space(&self) -> PixelScaledAiryDisk {
        // Get Airy disk radius in microns from telescope
        let airy_radius_um = self.telescope.airy_disk_radius_um();

        // Convert to pixels using sensor pixel size
        let airy_radius_pixels = airy_radius_um / self.sensor.pixel_size().as_micrometers();

        // Create scaled Airy disk with pixel radius
        PixelScaledAiryDisk::with_first_zero(airy_radius_pixels, self.telescope.corrected_to)
    }

    /// Create a PixelScaledAiryDisk based on FWHM sampling for this satellite configuration
    ///
    /// This method creates an Airy disk scaled to the current FWHM sampling ratio of the
    /// telescope/sensor combination. This is consistent with the `with_fwhm_sampling()` method
    /// and provides the appropriate PSF for the current optical configuration.
    ///
    /// # Returns
    /// A PixelScaledAiryDisk with FWHM sized according to the current sampling ratio
    ///
    pub fn airy_disk_fwhm_sampled(&self) -> PixelScaledAiryDisk {
        // Get current FWHM sampling ratio (pixels per FWHM)
        let fwhm_pixels = self.fwhm_sampling_ratio();

        // Create scaled Airy disk with this FWHM size
        PixelScaledAiryDisk::with_fwhm(fwhm_pixels, self.telescope.corrected_to)
    }

    /// Adjust telescope focal length to achieve specific FWHM sampling in pixels
    ///
    /// This method creates a new SatelliteConfig with modified telescope focal length
    /// to achieve the desired number of pixels per FWHM of the point spread function.
    /// Useful for matching optical sampling across different sensors or optimizing
    /// for specific detection algorithms.
    ///
    /// # Arguments
    /// * `q` - Target sampling ratio (pixels per FWHM)
    ///
    /// # Returns
    /// A new SatelliteConfig with adjusted telescope focal length
    ///
    pub fn with_fwhm_sampling(&self, q: f64) -> SatelliteConfig {
        let current_q =
            self.telescope.fwhm_image_spot() / self.sensor.pixel_size().as_micrometers();

        let ratio = q / current_q;
        let new_focal_length = Length::from_meters(self.telescope.focal_length.as_meters() * ratio);

        SatelliteConfig::new(
            self.telescope.with_focal_length(new_focal_length),
            self.sensor.clone(),
            self.temperature,
        )
    }

    /// Get the current FWHM sampling ratio (pixels per FWHM)
    ///
    /// Returns the number of pixels per FWHM of the point spread function
    /// for the current telescope/sensor configuration at the observing wavelength.
    ///
    /// # Returns
    /// Number of pixels per FWHM of the PSF
    ///
    pub fn fwhm_sampling_ratio(&self) -> f64 {
        self.telescope.fwhm_image_spot() / self.sensor.pixel_size().as_micrometers()
    }

    /// Generate a descriptive string for this satellite configuration
    ///
    /// Returns a human-readable description including telescope specifications
    /// (aperture, focal length, f-number) and sensor name, useful for logging,
    /// reports, and user interfaces.
    ///
    /// # Returns
    /// A string describing the satellite configuration
    ///
    pub fn description(&self) -> String {
        format!(
            "{} ({:.2}m f/{:.1}) + {}",
            self.telescope.name,
            self.telescope.aperture.as_meters(),
            self.telescope.f_number(),
            self.sensor.name
        )
    }
}

/// Complete focal plane array configuration.
///
/// Combines telescope optics with a multi-sensor array and operating
/// conditions for simulating mosaic detector systems like SPENCER.
/// This is the array-level analog of [`SatelliteConfig`], which handles
/// a single sensor.
///
/// # Usage
///
/// Use `satellite_for_sensor` to get a per-sensor `SatelliteConfig` for
/// rendering, flux calculation, and noise generation.
#[derive(Debug, Clone)]
pub struct FocalPlaneConfig {
    /// Telescope optical configuration (shared across all sensors)
    pub telescope: TelescopeConfig,
    /// Multi-sensor array with positions
    pub array: SensorArray,
    /// Operating temperature (affects dark current for all sensors)
    pub temperature: Temperature,
}

impl FocalPlaneConfig {
    /// Create a new focal plane configuration.
    pub fn new(telescope: TelescopeConfig, array: SensorArray, temperature: Temperature) -> Self {
        Self {
            telescope,
            array,
            temperature,
        }
    }

    /// Create a FocalPlaneConfig from a single SatelliteConfig.
    ///
    /// Wraps the satellite's sensor in a single-element array at the origin.
    pub fn from_satellite(satellite: &SatelliteConfig) -> Self {
        Self {
            telescope: satellite.telescope.clone(),
            array: SensorArray::single(satellite.sensor.clone()),
            temperature: satellite.temperature,
        }
    }

    /// Build a SatelliteConfig for a specific sensor in the array.
    ///
    /// Returns `None` if the sensor index is out of bounds.
    pub fn satellite_for_sensor(&self, sensor_index: usize) -> Option<SatelliteConfig> {
        self.array.sensors.get(sensor_index).map(|ps| {
            SatelliteConfig::new(self.telescope.clone(), ps.sensor.clone(), self.temperature)
        })
    }

    /// Plate scale in radians per millimeter at the focal plane.
    ///
    /// This is telescope-only and independent of any sensor's pixel size.
    pub fn plate_scale_rad_per_mm(&self) -> f64 {
        self.telescope.plate_scale().as_radians() / 1000.0
    }

    /// Get the total AABB of the sensor array in mm.
    pub fn total_aabb_mm(&self) -> Option<(f64, f64, f64, f64)> {
        self.array.total_aabb_mm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::{AreaExt, TemperatureExt};
    use approx::assert_relative_eq;

    #[test]
    fn test_satellite_config_creation() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::GSENSE4040BSI.clone();
        let temp = Temperature::from_celsius(-10.0);

        let satellite = SatelliteConfig::new(telescope, sensor, temp);
        assert_eq!(satellite.temperature, temp);
        assert!(satellite.telescope.clear_aperture_area().as_square_meters() > 0.0);
        assert!(satellite.plate_scale_arcsec_per_pixel() > 0.0);
    }

    #[test]
    fn test_field_of_view_calculation() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::GSENSE6510BSI.clone();

        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(-10.0));

        let (width_arcmin, height_arcmin) = {
            let this = &satellite;
            let (width_angle, height_angle) = this.field_of_view();
            (
                width_angle.as_arcseconds() / 60.0,
                height_angle.as_arcseconds() / 60.0,
            )
        };
        assert!(width_arcmin > 0.0);
        assert!(height_arcmin > 0.0);
    }

    #[test]
    fn test_plate_scale_per_pixel() {
        // Create a test telescope with known focal length
        let aperture = Length::from_meters(0.1); // 100mm aperture
        let focal_length = Length::from_meters(1.0); // 1000mm focal length
        let telescope = TelescopeConfig::new("Test Telescope", aperture, focal_length, 0.8);

        // Create a test sensor with known pixel size
        let sensor = crate::hardware::sensor::models::GSENSE4040BSI.clone();
        let pixel_size_um = sensor.pixel_size().as_micrometers();
        assert_eq!(pixel_size_um, 9.0); // GSENSE4040BSI has 9μm pixels

        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(-10.0));

        // Calculate expected plate scale
        // plate_scale = arctan(pixel_size / focal_length) ≈ pixel_size / focal_length (small angle)
        let pixel_size_m = 9.0e-6; // 9 micrometers
        let focal_length_m = 1.0; // 1 meter
        let expected_rad_per_pixel = pixel_size_m / focal_length_m;

        // Convert to arcseconds for easier comparison
        let expected_arcsec_per_pixel = expected_rad_per_pixel * 206265.0; // rad to arcsec

        let actual_rad_per_pixel = satellite.plate_scale_per_pixel().as_radians();
        let actual_arcsec_per_pixel = satellite.plate_scale_arcsec_per_pixel();

        // Check radians per pixel (should be 9e-6 radians)
        assert_relative_eq!(
            actual_rad_per_pixel,
            expected_rad_per_pixel,
            epsilon = 1e-12
        );

        // Check arcseconds per pixel (should be ~1.86 arcsec/pixel)
        assert_relative_eq!(
            actual_arcsec_per_pixel,
            expected_arcsec_per_pixel,
            epsilon = 0.01
        );
        assert_relative_eq!(actual_arcsec_per_pixel, 1.855, epsilon = 0.01); // Precise value
    }

    #[test]
    fn test_field_of_view_vs_plate_scale_consistency() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.2),
            Length::from_meters(1.0),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::GSENSE4040BSI.clone();

        // Get sensor dimensions in pixels before moving sensor
        let (width_px, height_px) = sensor.dimensions.get_pixel_width_height();

        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(-10.0));

        // Calculate FOV from plate scale
        let rad_per_pixel = satellite.plate_scale_per_pixel().as_radians();
        let expected_width_rad = rad_per_pixel * width_px as f64;
        let expected_height_rad = rad_per_pixel * height_px as f64;

        // Get FOV from dedicated method
        let (actual_width, actual_height) = satellite.field_of_view();
        let actual_width_rad = actual_width.as_radians();
        let actual_height_rad = actual_height.as_radians();

        // They should match exactly
        assert_relative_eq!(actual_width_rad, expected_width_rad, epsilon = 1e-10);
        assert_relative_eq!(actual_height_rad, expected_height_rad, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_disk_pixel_space() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::HWK4123.clone();

        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(-10.0));

        let airy_disk = satellite.airy_disk_pixel_space();

        // Airy disk should have a reasonable first zero radius in pixels
        assert!(airy_disk.first_zero() > 0.0);
        assert!(airy_disk.first_zero() < 100.0); // Should be reasonable size
    }

    #[test]
    fn test_with_fwhm_sampling() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::GSENSE4040BSI.clone();

        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(-10.0));

        // Get current sampling ratio
        let original_sampling = satellite.fwhm_sampling_ratio();

        // Create version with 2.5 pixels per FWHM
        let target_sampling = 2.5;
        let resampled = satellite.with_fwhm_sampling(target_sampling);

        // Check that new sampling ratio matches target
        let new_sampling = resampled.fwhm_sampling_ratio();
        assert_relative_eq!(new_sampling, target_sampling, epsilon = 1e-10);

        // Check that other parameters remain unchanged
        assert_eq!(resampled.sensor.name, satellite.sensor.name);
        assert_eq!(resampled.temperature, satellite.temperature);
        assert_eq!(
            resampled.telescope.aperture.as_meters(),
            satellite.telescope.aperture.as_meters()
        );

        // Focal length should have changed proportionally
        let expected_ratio = target_sampling / original_sampling;
        let actual_ratio = resampled.telescope.focal_length.as_meters()
            / satellite.telescope.focal_length.as_meters();
        assert_relative_eq!(actual_ratio, expected_ratio, epsilon = 1e-10);
    }

    #[test]
    fn test_fwhm_sampling_ratio() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::HWK4123.clone();

        let satellite = SatelliteConfig::new(
            telescope.clone(),
            sensor.clone(),
            Temperature::from_celsius(-10.0),
        );

        let sampling = satellite.fwhm_sampling_ratio();

        // Should return a positive ratio
        assert!(sampling > 0.0);

        // Should match manual calculation
        let expected = telescope.fwhm_image_spot() / sensor.pixel_size().as_micrometers();
        assert_relative_eq!(sampling, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_airy_disk_fwhm_sampled() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::GSENSE4040BSI.clone();

        let satellite = SatelliteConfig::new(
            telescope.clone(),
            sensor.clone(),
            Temperature::from_celsius(-10.0),
        );

        // Get Airy disk scaled to current FWHM sampling
        let airy_disk = satellite.airy_disk_fwhm_sampled();

        // The FWHM should match the current sampling ratio
        let expected_fwhm = satellite.fwhm_sampling_ratio();
        assert_relative_eq!(airy_disk.fwhm(), expected_fwhm, epsilon = 1e-10);

        // Test consistency: if we create a new satellite with different sampling,
        // the airy disk FWHM should change accordingly
        let resampled_satellite = satellite.with_fwhm_sampling(3.0);
        let resampled_airy = resampled_satellite.airy_disk_fwhm_sampled();

        assert_relative_eq!(resampled_airy.fwhm(), 3.0, epsilon = 1e-10);

        // Verify that the FWHM is different from the original
        assert!((airy_disk.fwhm() - resampled_airy.fwhm()).abs() > 1e-6);
    }

    #[test]
    fn test_description() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::GSENSE4040BSI.clone();

        let satellite = SatelliteConfig::new(
            telescope.clone(),
            sensor.clone(),
            Temperature::from_celsius(-10.0),
        );

        let description = satellite.description();

        // Should include telescope name, aperture, f-number, and sensor name
        assert!(description.contains("Test Scope"));
        assert!(description.contains("0.50m"));
        assert!(description.contains("f/5.0")); // f-number = focal_length / aperture = 2.5 / 0.5 = 5.0
        assert!(description.contains("GSENSE4040BSI"));
        assert!(description.contains("+")); // Should have the '+' separator

        // Verify the exact format
        let expected = "Test Scope (0.50m f/5.0) + GSENSE4040BSI";
        assert_eq!(description, expected);
    }

    #[test]
    fn test_focal_plane_config_creation() {
        use crate::hardware::sensor_array::SPENCER_ARRAY_PLAN;

        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let fp = FocalPlaneConfig::new(
            telescope,
            SPENCER_ARRAY_PLAN.clone(),
            Temperature::from_celsius(-20.0),
        );

        assert_eq!(fp.array.sensor_count(), 4);
        assert!(fp.total_aabb_mm().is_some());
    }

    #[test]
    fn test_focal_plane_satellite_for_sensor() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::GSENSE4040BSI.clone();
        let fp = FocalPlaneConfig::new(
            telescope.clone(),
            SensorArray::single(sensor.clone()),
            Temperature::from_celsius(-10.0),
        );

        let sat = fp.satellite_for_sensor(0).unwrap();
        assert_eq!(sat.sensor.name, sensor.name);
        assert_eq!(sat.telescope.name, telescope.name);

        assert!(fp.satellite_for_sensor(1).is_none());
    }

    #[test]
    fn test_focal_plane_from_satellite_roundtrip() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::GSENSE4040BSI.clone();
        let sat = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(-10.0));

        let fp = FocalPlaneConfig::from_satellite(&sat);
        assert_eq!(fp.array.sensor_count(), 1);

        let roundtrip = fp.satellite_for_sensor(0).unwrap();
        assert_eq!(roundtrip.sensor.name, sat.sensor.name);
        assert_relative_eq!(
            roundtrip.plate_scale_per_pixel().as_radians(),
            sat.plate_scale_per_pixel().as_radians(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_focal_plane_plate_scale_rad_per_mm() {
        let telescope = TelescopeConfig::new(
            "Test Scope",
            Length::from_meters(0.1),
            Length::from_meters(1.0),
            0.8,
        );
        let sensor = crate::hardware::sensor::models::GSENSE4040BSI.clone();
        let fp = FocalPlaneConfig::new(
            telescope,
            SensorArray::single(sensor),
            Temperature::from_celsius(-10.0),
        );

        // plate_scale = 1/focal_length_m = 1.0 rad/m
        // rad_per_mm = 1.0 / 1000 = 0.001
        assert_relative_eq!(fp.plate_scale_rad_per_mm(), 0.001, epsilon = 1e-10);
    }
}

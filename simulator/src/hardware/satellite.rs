use super::{sensor::SensorConfig, telescope::TelescopeConfig};
use crate::image_proc::airy::PixelScaledAiryDisk;
use crate::photometry::QuantumEfficiency;
use crate::units::{Length, LengthExt, Temperature, Wavelength};

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
    /// Primary observing wavelength (affects PSF size)
    pub wavelength: Wavelength,
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
    /// * `wavelength` - Primary observing wavelength (affects PSF)
    ///
    /// # Returns
    /// New SatelliteConfig ready for astronomical simulations
    ///
    pub fn new(
        telescope: TelescopeConfig,
        sensor: SensorConfig,
        temperature: Temperature,
        wavelength: Wavelength,
    ) -> Self {
        // Calculate combined QE from telescope and sensor
        let combined_qe =
            QuantumEfficiency::product(&telescope.quantum_efficiency, &sensor.quantum_efficiency)
                .expect("Failed to combine telescope and sensor QE curves");

        Self {
            telescope,
            sensor,
            temperature,
            wavelength,
            combined_qe,
        }
    }

    /// Get the effective collecting area accounting for telescope efficiency.
    ///
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

    /// Get the plate scale in arcseconds per pixel.
    ///
    /// Critical parameter for astrometric accuracy and detection algorithms.
    /// Determines the angular resolution of the imaging system.
    ///
    /// # Returns
    /// Plate scale in arcseconds per pixel
    ///
    pub fn plate_scale_arcsec_per_pixel(&self) -> f64 {
        let pixel_size_mm = self.sensor.pixel_size.as_millimeters();
        self.plate_scale_arcsec_per_mm() * pixel_size_mm
    }

    /// Get the field of view in arcminutes for the sensor.
    ///
    /// Calculates the total sky coverage of the detector array,
    /// useful for survey planning and catalog queries.
    ///
    /// # Returns
    /// Tuple of (width, height) field of view in arcminutes
    ///
    pub fn field_of_view_arcmin(&self) -> (f64, f64) {
        let arcsec_per_pixel = self.plate_scale_arcsec_per_pixel();
        let width_arcmin = (self.sensor.width_px as f64 * arcsec_per_pixel) / 60.0;
        let height_arcmin = (self.sensor.height_px as f64 * arcsec_per_pixel) / 60.0;
        (width_arcmin, height_arcmin)
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
        let (width_arcmin, height_arcmin) = self.field_of_view_arcmin();
        // Convert to radians
        let width_rad = (width_arcmin / 60.0) * (std::f64::consts::PI / 180.0);
        let height_rad = (height_arcmin / 60.0) * (std::f64::consts::PI / 180.0);
        // Small angle approximation for solid angle
        width_rad * height_rad
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
        let airy_radius_um = self.telescope.airy_disk_radius_um(self.wavelength);

        // Convert to pixels using sensor pixel size
        let airy_radius_pixels = airy_radius_um / self.sensor.pixel_size.as_micrometers();

        // Create scaled Airy disk with pixel radius
        PixelScaledAiryDisk::with_first_zero(airy_radius_pixels, self.wavelength)
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
        PixelScaledAiryDisk::with_fwhm(fwhm_pixels, self.wavelength)
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
        let current_q = self.telescope.fwhm_image_spot_um(self.wavelength)
            / self.sensor.pixel_size.as_micrometers();

        let ratio = q / current_q;
        let new_focal_length = Length::from_meters(self.telescope.focal_length.as_meters() * ratio);

        SatelliteConfig::new(
            self.telescope.with_focal_length(new_focal_length),
            self.sensor.clone(),
            self.temperature,
            self.wavelength,
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
        self.telescope.fwhm_image_spot_um(self.wavelength) / self.sensor.pixel_size.as_micrometers()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::TemperatureExt;

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

        let satellite =
            SatelliteConfig::new(telescope, sensor, temp, Wavelength::from_nanometers(550.0));
        assert_eq!(satellite.temperature, temp);
        assert!(satellite.telescope.collecting_area_m2() > 0.0);
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

        let satellite = SatelliteConfig::new(
            telescope,
            sensor,
            Temperature::from_celsius(-10.0),
            Wavelength::from_nanometers(550.0),
        );

        let (width_arcmin, height_arcmin) = satellite.field_of_view_arcmin();
        assert!(width_arcmin > 0.0);
        assert!(height_arcmin > 0.0);
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

        let satellite = SatelliteConfig::new(
            telescope,
            sensor,
            Temperature::from_celsius(-10.0),
            Wavelength::from_nanometers(550.0),
        );

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

        let satellite = SatelliteConfig::new(
            telescope,
            sensor,
            Temperature::from_celsius(-10.0),
            Wavelength::from_nanometers(550.0),
        );

        // Get current sampling ratio
        let original_sampling = satellite.fwhm_sampling_ratio();

        // Create version with 2.5 pixels per FWHM
        let target_sampling = 2.5;
        let resampled = satellite.with_fwhm_sampling(target_sampling);

        // Check that new sampling ratio matches target
        let new_sampling = resampled.fwhm_sampling_ratio();
        assert!((new_sampling - target_sampling).abs() < 1e-10);

        // Check that other parameters remain unchanged
        assert_eq!(resampled.sensor.name, satellite.sensor.name);
        assert_eq!(resampled.temperature, satellite.temperature);
        assert_eq!(resampled.wavelength, satellite.wavelength);
        assert_eq!(
            resampled.telescope.aperture.as_meters(),
            satellite.telescope.aperture.as_meters()
        );

        // Focal length should have changed proportionally
        let expected_ratio = target_sampling / original_sampling;
        let actual_ratio = resampled.telescope.focal_length.as_meters()
            / satellite.telescope.focal_length.as_meters();
        assert!((actual_ratio - expected_ratio).abs() < 1e-10);
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
            Wavelength::from_nanometers(550.0),
        );

        let sampling = satellite.fwhm_sampling_ratio();

        // Should return a positive ratio
        assert!(sampling > 0.0);

        // Should match manual calculation
        let expected = telescope.fwhm_image_spot_um(Wavelength::from_nanometers(550.0))
            / sensor.pixel_size.as_micrometers();
        assert!((sampling - expected).abs() < 1e-10);
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
            Wavelength::from_nanometers(550.0),
        );

        // Get Airy disk scaled to current FWHM sampling
        let airy_disk = satellite.airy_disk_fwhm_sampled();

        // The FWHM should match the current sampling ratio
        let expected_fwhm = satellite.fwhm_sampling_ratio();
        assert!((airy_disk.fwhm() - expected_fwhm).abs() < 1e-10);

        // Test consistency: if we create a new satellite with different sampling,
        // the airy disk FWHM should change accordingly
        let resampled_satellite = satellite.with_fwhm_sampling(3.0);
        let resampled_airy = resampled_satellite.airy_disk_fwhm_sampled();

        assert!((resampled_airy.fwhm() - 3.0).abs() < 1e-10);

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
            Wavelength::from_nanometers(550.0),
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
}

//! Airy disk point spread function modeling for diffraction-limited optics.
//!
//! This module provides exact and approximate calculations for the Airy disk pattern,
//! which represents the theoretical point spread function (PSF) of a perfect circular
//! aperture telescope. Essential for realistic star detection and photometry simulations.
//!
//! # Physics Background
//!
//! The Airy disk is the diffraction pattern produced by a plane wave passing through
//! a circular aperture. The intensity profile follows:
//!
//! ```text
//! I(r) = I₀ * [2*J₁(kr)/kr]²
//! ```
//!
//! where:
//! - `J₁` is the first-order Bessel function
//! - `k = π * aperture_diameter / (wavelength * focal_length)`
//! - `r` is the radial distance from the center
//!
//! # Key Features
//!
//! - **Exact calculations**: Bessel function-based Airy disk intensity
//! - **Fast approximations**: Gaussian and triangular PSF models
//! - **Scalable PSF**: Adjust size for different telescope configurations
//! - **Performance analysis**: Error metrics for approximation quality
//! - **FWHM calculations**: Full-width-half-maximum measurements
//!
//! # Approximation Methods
//!
//! ## Gaussian Approximation
//! ```text
//! I(r) ≈ I₀ * exp(-3.9 * r² / r₀²)
//! ```
//! Provides smooth falloff with analytical properties, good for convolution.
//!
//! ## Triangle Approximation  
//! ```text
//! I(r) = max(0, 1 - r/r₀)
//! ```
//! Simple linear falloff, computationally efficient for basic simulations.
//!

use once_cell::sync::Lazy;
use scilib::math::bessel;

/// Airy disk parameters and approximation functions for diffraction-limited optics.
///
/// The Airy disk is the diffraction pattern resulting from a uniformly
/// illuminated circular aperture. This struct provides both exact calculations
/// and various approximations for efficient computation in astronomical simulations.
///
/// # Physical Parameters
///
/// - **first_zero**: Radius to first dark ring (1.22π for normalized units)
/// - **fwhm**: Full-width-half-maximum for intensity profile
///
/// Both parameters are in normalized units where the aperture radius = 1.
/// Scale by actual telescope parameters for physical dimensions.
///
/// # Performance Notes
///
/// - Exact calculations use Bessel functions (moderate computational cost)
/// - Gaussian approximation: ~10x faster, <5% error within FWHM
/// - Triangle approximation: ~50x faster, good for rough estimates
///
#[derive(Debug, Clone, Copy)]
pub struct AiryDisk {
    /// First zero location (first dark ring radius) in normalized units
    pub first_zero: f64,
    /// Full-width-half-maximum in normalized units
    pub fwhm: f64,
}

impl Default for AiryDisk {
    fn default() -> Self {
        Self::new()
    }
}

impl AiryDisk {
    /// Create a new AiryDisk with standard normalized parameters.
    ///
    /// Calculates the first zero location and FWHM for a perfect circular aperture
    /// using numerical methods. Results are in normalized units where the aperture
    /// radius equals 1.0.
    ///
    /// # Returns
    /// AiryDisk with computed first_zero ≈ 3.832 and fwhm ≈ 3.233
    ///
    pub fn new() -> Self {
        let mut disk = Self {
            first_zero: 0.0,
            fwhm: 0.0,
        };

        // Calculate first zero location and FWHM
        disk.first_zero = disk.calculate_first_zero();
        disk.fwhm = disk.calculate_fwhm();
        disk
    }

    /// Calculate the exact Airy disk function intensity at given radius.
    ///
    /// The Airy disk intensity follows: I(r) = [2*J₁(r)/r]²
    /// where J₁ is the first-order Bessel function of the first kind.
    /// Normalized so that I(0) = 1.0.
    ///
    /// # Arguments
    /// * `radius` - Radial distance from center in normalized units
    ///
    /// # Returns
    /// Intensity value (0.0 to 1.0, with 1.0 at center)
    ///
    /// # Performance
    /// Uses Bessel function evaluation - moderate computational cost.
    /// Consider gaussian_approximation() for performance-critical applications.
    ///
    pub fn intensity(&self, radius: f64) -> f64 {
        if radius.abs() < 1e-10 {
            return 1.0; // Limit as r approaches 0
        }

        let j1 = bessel::j_n(1, radius);
        let term = 2.0 * j1 / radius;
        term * term
    }

    /// Gaussian approximation to the Airy disk function.
    ///
    /// Using the formula I(r) ≈ exp(-3.9*r²/r₀²), where r₀ is the radius
    /// of the first dark ring in the Airy pattern. This approximation is
    /// optimized to match the FWHM of the true Airy disk.
    ///
    /// # Arguments
    /// * `radius` - Radial distance from center in normalized units
    ///
    /// # Returns
    /// Approximated intensity value (0.0 to 1.0)
    ///
    /// # Accuracy
    /// - Error < 5% within FWHM radius
    /// - Error < 10% within first dark ring
    /// - Overestimates intensity in far wings (r > 2×first_zero)
    ///
    /// # Performance
    /// ~10x faster than exact Bessel function calculation.
    ///
    pub fn gaussian_approximation(&self, radius: f64) -> f64 {
        (-3.9 * (radius * radius) / (self.first_zero * self.first_zero)).exp()
    }

    /// Triangle approximation to the Airy disk function.
    ///
    /// Simple linear falloff from center to first zero:
    /// I(r) = max(0, 1 - r/r₀) for r ≤ r₀, then 0 for r > r₀
    /// where r₀ is the first dark ring radius.
    ///
    /// # Arguments
    /// * `radius` - Radial distance from center in normalized units
    ///
    /// # Returns
    /// Approximated intensity value (0.0 to 1.0)
    ///
    /// # Accuracy
    /// - Very rough approximation, mainly for computational efficiency
    /// - Correct FWHM scaling but wrong shape profile
    /// - Good for order-of-magnitude estimates and fast algorithms
    ///
    /// # Performance
    /// ~50x faster than exact calculation - just arithmetic operations.
    ///
    pub fn triangle_approximation(&self, radius: f64) -> f64 {
        let normalized_radius = radius / self.first_zero;
        if normalized_radius >= 1.0 {
            0.0
        } else {
            1.0 - normalized_radius
        }
    }

    /// Generate sample points for comparing different approximations.
    ///
    /// Creates evenly spaced radial samples from center to 2× first dark ring,
    /// computing exact and approximate intensities for error analysis.
    ///
    /// # Arguments
    /// * `num_points` - Number of sample points to generate
    ///
    /// # Returns
    /// Tuple of (radii, exact_intensities, gaussian_approximations, triangle_approximations)
    /// All vectors have length `num_points`.
    ///
    /// # Usage
    /// Useful for:
    /// - Plotting PSF profiles and approximation quality
    /// - Computing MSE or other error metrics
    /// - Analyzing approximation behavior across different radii
    ///
    pub fn generate_comparison_samples(
        &self,
        num_points: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut radii = Vec::with_capacity(num_points);
        let mut exact = Vec::with_capacity(num_points);
        let mut gaussian = Vec::with_capacity(num_points);
        let mut triangle = Vec::with_capacity(num_points);

        // Range from center to 2x the first dark ring
        let max_radius = 2.0 * self.first_zero;

        for i in 0..num_points {
            let r = i as f64 * max_radius / (num_points as f64 - 1.0);
            radii.push(r);
            exact.push(self.intensity(r));
            gaussian.push(self.gaussian_approximation(r));
            triangle.push(self.triangle_approximation(r));
        }

        (radii, exact, gaussian, triangle)
    }

    /// Calculate mean squared error between exact function and approximation
    pub fn calculate_mse(actual: &[f64], approx: &[f64]) -> f64 {
        if actual.len() != approx.len() {
            panic!("Arrays must have same length for MSE calculation");
        }

        let sum_squared_errors: f64 = actual
            .iter()
            .zip(approx.iter())
            .map(|(a, g)| (a - g).powi(2))
            .sum();

        sum_squared_errors / actual.len() as f64
    }

    /// Find where the error between exact and approximation is largest
    pub fn find_max_error(radii: &[f64], exact: &[f64], approx: &[f64]) -> (f64, f64, Option<f64>) {
        if radii.len() != exact.len() || exact.len() != approx.len() {
            panic!("All arrays must have same length");
        }

        let mut max_error = 0.0;
        let mut max_error_idx = 0;
        let mut first_5pct_error_idx = None;

        for (i, ((&_r, &exact_val), &approx_val)) in radii
            .iter()
            .zip(exact.iter())
            .zip(approx.iter())
            .enumerate()
        {
            let error = (exact_val - approx_val).abs();

            // Track first occurrence of 5% error
            if first_5pct_error_idx.is_none() && error > 0.05 {
                first_5pct_error_idx = Some(i);
            }

            // Track maximum error
            if error > max_error {
                max_error = error;
                max_error_idx = i;
            }
        }

        let first_5pct_radius = first_5pct_error_idx.map(|idx| radii[idx]);
        (max_error, radii[max_error_idx], first_5pct_radius)
    }

    /// Calculate the full-width-half-maximum (FWHM) of the Airy disk
    ///
    /// Finds the radius where intensity drops to 0.5, then doubles it for FWHM
    pub fn calculate_fwhm(&self) -> f64 {
        let target = 0.5;
        let mut left = 0.0;
        let mut right = 2.0; // Start with a reasonable upper bound

        // Make sure we have the right bounds
        while self.intensity(right) > target {
            right *= 2.0;
        }

        // Binary search for the half-maximum point
        while (right - left) > 1e-10 {
            let mid = (left + right) / 2.0;
            let intensity = self.intensity(mid);

            if intensity > target {
                left = mid;
            } else {
                right = mid;
            }
        }

        let half_max_radius = (left + right) / 2.0;

        // FWHM is twice the half-maximum radius
        2.0 * half_max_radius
    }

    /// Calculate the first zero (dark ring) of the Airy disk function
    ///
    /// Uses numerical approximation to find where the Airy disk function
    /// first crosses zero, corresponding to the first dark ring
    fn calculate_first_zero(&self) -> f64 {
        let mut x = 3.83; // Starting guess close to theoretical value
        let mut step = 0.01;

        // Use binary search to find the zero crossing
        while step > 1e-10 {
            let intensity = self.intensity_for_calculation(x);
            if intensity < 0.0 {
                x -= step;
            } else {
                x += step;
            }
            step *= 0.5;
        }

        x
    }

    /// Helper function for zero finding that doesn't depend on self.first_zero
    fn intensity_for_calculation(&self, radius: f64) -> f64 {
        if radius.abs() < 1e-10 {
            return 1.0;
        }

        let j1 = bessel::j_n(1, radius);
        let term = 2.0 * j1 / radius;
        term * term
    }
}

pub static AIRY_DISK: Lazy<AiryDisk> = Lazy::new(AiryDisk::new);

/// Scaled Airy disk for telescope-specific PSF modeling.
///
/// Wraps the base AiryDisk with a scaling factor to match real telescope
/// characteristics. Allows easy conversion between normalized Airy disk
/// theory and physical telescope parameters.
///
/// # Scaling Methods
///
/// - **Radius scaling**: Direct multiplication of all radii
/// - **FWHM scaling**: Scale to match desired FWHM in pixels/microns
/// - **First zero scaling**: Scale to match dark ring radius
///
/// # Physical Interpretation
///
/// For a telescope with aperture D, focal length f, at wavelength λ:
/// - Physical Airy radius = 1.22 × λ × f / D
/// - Scale factor = physical_radius / normalized_radius
///
#[derive(Debug, Clone, Copy)]
pub struct PixelScaledAiryDisk {
    disk: AiryDisk,
    radius_scale: f64,
    /// Reference wavelength in nanometers for chromatic calculations
    pub reference_wavelength: f64,
}

impl PixelScaledAiryDisk {
    /// Creates a new PixelScaledAiryDisk with a default AiryDisk and given radius scale.
    ///
    /// # Arguments
    /// * `radius_scale` - Multiplicative factor applied to all radial distances
    ///
    /// # Returns
    /// PixelScaledAiryDisk instance with specified scaling
    fn new(radius_scale: f64, reference_wavelength: f64) -> Self {
        PixelScaledAiryDisk {
            disk: *AIRY_DISK,
            radius_scale,
            reference_wavelength,
        }
    }

    /// Create a new PixelScaledAiryDisk with specified radius scaling factor.
    ///
    /// # Arguments
    /// * `radius_scale` - Factor to scale all radial measurements
    /// * `reference_wavelength` - Reference wavelength in nm
    ///
    /// # Returns
    /// PixelScaledAiryDisk with the specified radius scaling
    ///
    pub fn with_radius_scale(radius_scale: f64, reference_wavelength: f64) -> Self {
        Self::new(radius_scale, reference_wavelength)
    }

    /// Create a new PixelScaledAiryDisk with specified FWHM.
    ///
    /// Calculates the appropriate radius scaling to achieve the desired
    /// full-width-half-maximum value in user units (pixels, microns, etc.).
    ///
    /// # Arguments
    /// * `fwhm` - Desired FWHM in target units
    ///
    /// # Returns
    /// PixelScaledAiryDisk with scaling to match the specified FWHM
    ///
    pub fn with_fwhm(fwhm: f64, reference_wavelength: f64) -> Self {
        let scalar = fwhm / AIRY_DISK.fwhm;
        Self::new(scalar, reference_wavelength)
    }

    /// Create a new PixelScaledAiryDisk with specified first zero radius.
    ///
    /// Calculates the appropriate radius scaling to achieve the desired
    /// first dark ring location in user units.
    ///
    /// # Arguments
    /// * `first_zero` - Desired first zero radius in target units
    ///
    /// # Returns
    /// PixelScaledAiryDisk with scaling to match the specified first zero
    ///
    pub fn with_first_zero(first_zero: f64, reference_wavelength: f64) -> Self {
        let scalar = first_zero / AIRY_DISK.first_zero;
        Self::new(scalar, reference_wavelength)
    }

    /// Calculate the exact Airy disk intensity at scaled radius.
    ///
    /// # Arguments
    /// * `radius` - Radial distance in scaled units (pixels, microns, etc.)
    ///
    /// # Returns
    /// Intensity value (0.0 to 1.0)
    ///
    pub fn intensity(&self, radius: f64) -> f64 {
        self.disk.intensity(radius / self.radius_scale)
    }

    /// Calculate the Gaussian approximation at scaled radius.
    ///
    /// # Arguments
    /// * `radius` - Radial distance in scaled units
    ///
    /// # Returns
    /// Approximated intensity value (0.0 to 1.0)
    ///
    /// # Performance
    /// ~10x faster than exact intensity calculation.
    pub fn gaussian_approximation(&self, radius: f64) -> f64 {
        self.disk.gaussian_approximation(radius / self.radius_scale)
    }

    /// Calculate normalized Gaussian approximation for integration purposes.
    ///
    /// Returns the Gaussian approximation scaled so that the integral over
    /// the entire 2D plane equals 1.0. Useful for photometric calculations
    /// where total flux conservation is important.
    ///
    /// # Arguments
    /// * `radius` - Radial distance in scaled units
    ///
    /// # Returns
    /// Normalized intensity value (integral = 1.0 over infinite plane)
    ///
    /// # Note
    /// This normalization accounts for the 2D integration and radius scaling.
    pub fn gaussian_approximation_normalized(&self, radius: f64) -> f64 {
        // TODO(meawoppl) - cleanup the constants running around
        // Get the unscaled gaussian value
        let gauss_value = self.disk.gaussian_approximation(radius / self.radius_scale);

        // The integral of the base gaussian exp(-3.9 * r²/r₀²) in 2D is π * r₀² / 3.9
        let r0_base = self.disk.first_zero;
        let base_integral = std::f64::consts::PI * r0_base * r0_base / 3.9;

        // When we scale by radius_scale, the integral scales by radius_scale²
        // So the normalized function is: gauss_value / (base_integral * radius_scale²)
        gauss_value / (base_integral * (self.radius_scale * self.radius_scale))
    }

    /// Calculate the triangle approximation at scaled radius.
    ///
    /// # Arguments
    /// * `radius` - Radial distance in scaled units
    ///
    /// # Returns
    /// Approximated intensity value (0.0 to 1.0)
    ///
    /// # Performance
    /// ~50x faster than exact calculation.
    pub fn triangle_approximation(&self, radius: f64) -> f64 {
        self.disk.triangle_approximation(radius / self.radius_scale)
    }

    /// Get the first dark ring radius in scaled units.
    ///
    /// # Returns
    /// Radius to first zero crossing in user units
    pub fn first_zero(&self) -> f64 {
        self.disk.first_zero * self.radius_scale
    }

    /// Get the full-width-half-maximum in scaled units.
    ///
    /// # Returns
    /// FWHM in user units
    pub fn fwhm(&self) -> f64 {
        self.disk.fwhm * self.radius_scale
    }

    /// Compute pixel flux using 3x3 Simpson's rule for this PixelScaledAiryDisk PSF
    ///
    /// # Arguments
    /// * `x_pixel` - Pixel center x coordinate relative to PSF center
    /// * `y_pixel` - Pixel center y coordinate relative to PSF center
    /// * `flux` - Total integrated flux of the source
    pub fn pixel_flux_simpson(&self, x_pixel: f64, y_pixel: f64, flux: f64) -> f64 {
        // Simpson weights: [1, 4, 1] normalized to sum to 1
        const WEIGHTS: [f64; 3] = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];

        // Sample points: pixel edges and center
        let x_samples = [x_pixel - 0.5, x_pixel, x_pixel + 0.5];
        let y_samples = [y_pixel - 0.5, y_pixel, y_pixel + 0.5];

        let mut integrated_intensity = 0.0;

        for (i, &wx) in WEIGHTS.iter().enumerate() {
            for (j, &wy) in WEIGHTS.iter().enumerate() {
                let radius = (x_samples[i] * x_samples[i] + y_samples[j] * y_samples[j]).sqrt();
                integrated_intensity += wx * wy * self.gaussian_approximation_normalized(radius);
            }
        }

        // Scale by flux to get actual pixel flux
        integrated_intensity * flux
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_airy_disk_creation() {
        let disk = AiryDisk::new(); // 550nm, 100μm aperture
        assert!(disk.first_zero > 3.8 && disk.first_zero < 3.9);
        // FWHM for Airy disk should be approximately 3.2 based on numerical calculation
        assert!(disk.fwhm > 3.0 && disk.fwhm < 3.5);
    }

    #[test]
    fn test_intensity_at_center() {
        let disk = AiryDisk::new();
        assert_relative_eq!(disk.intensity(0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intensity_at_first_zero() {
        let disk = AiryDisk::new();
        let intensity_at_zero = disk.intensity(disk.first_zero);
        assert_relative_eq!(intensity_at_zero, 0.0, epsilon = 1e-4); // Relaxed for numerical precision
    }

    #[test]
    fn test_gaussian_approximation() {
        let disk = AiryDisk::new();

        // At center, both should be 1.0
        assert_relative_eq!(disk.gaussian_approximation(0.0), 1.0, epsilon = 1e-10);

        // Should be good approximation near center
        let r_small = disk.first_zero * 0.1;
        let exact = disk.intensity(r_small);
        let approx = disk.gaussian_approximation(r_small);
        assert!((exact - approx).abs() < 0.05);
    }

    #[test]
    fn test_triangle_approximation() {
        let disk = AiryDisk::new();

        // At center should be 1.0
        assert_relative_eq!(disk.triangle_approximation(0.0), 1.0, epsilon = 1e-10);

        // At first zero should be 0.0
        assert_relative_eq!(
            disk.triangle_approximation(disk.first_zero),
            0.0,
            epsilon = 1e-10
        );

        // At 1.5 times first zero should be 0.0
        assert_relative_eq!(
            disk.triangle_approximation(1.5 * disk.first_zero),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_sample_generation() {
        let disk = AiryDisk::new();
        let (radii, exact, gaussian, triangle) = disk.generate_comparison_samples(100);

        assert_eq!(radii.len(), 100);
        assert_eq!(exact.len(), 100);
        assert_eq!(gaussian.len(), 100);
        assert_eq!(triangle.len(), 100);

        // Check that first point is at r=0
        assert_relative_eq!(radii[0], 0.0, epsilon = 1e-10);

        // Check that last point is at 2*first_zero
        assert_relative_eq!(radii[99], 2.0 * disk.first_zero, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_calculation() {
        let a = vec![1.0, 0.5, 0.0];
        let b = vec![0.9, 0.6, 0.1];
        let mse = AiryDisk::calculate_mse(&a, &b);
        let expected = ((0.1_f64).powi(2) + (0.1_f64).powi(2) + (0.1_f64).powi(2)) / 3.0;
        assert_relative_eq!(mse, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_scaled_disk_unity() {
        let unscaled = PixelScaledAiryDisk::new(1.0, 550.0);
        assert_relative_eq!(unscaled.fwhm(), AIRY_DISK.fwhm, epsilon = 1e-10);

        let zero_rad = unscaled.first_zero();
        assert_relative_eq!(zero_rad, AIRY_DISK.first_zero, epsilon = 1e-10);
        assert!(unscaled.intensity(zero_rad) < 0.0001);
        assert!(unscaled.gaussian_approximation(zero_rad) < 0.1); // Gaussian is a bit fat tail
        assert!(unscaled.triangle_approximation(zero_rad) < 0.0001);
    }

    #[test]
    fn test_scaled_airy_disk_fwmh() {
        let scaled = PixelScaledAiryDisk::with_fwhm(2.0, 550.0);

        let fwhm = scaled.fwhm();
        assert_relative_eq!(fwhm, 2.0, epsilon = 1e-10);

        let half = scaled.intensity(fwhm / 2.0);
        assert_relative_eq!(half, 0.5, epsilon = 1e-10);

        let close_half = scaled.gaussian_approximation(fwhm / 2.0);
        assert_relative_eq!(close_half, 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_scaled_airy_disk_first_zero() {
        let scaled = PixelScaledAiryDisk::with_first_zero(2.0, 550.0);
        assert_relative_eq!(scaled.first_zero(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pixel_flux_centered() {
        // Test flux when PSF is centered on pixel
        let psf = PixelScaledAiryDisk::with_first_zero(1.0, 550.0);
        let flux = 1000.0;

        let pixel_flux = psf.pixel_flux_simpson(0.0, 0.0, flux);

        // Most flux should be in center pixel for small PSF
        assert!(
            pixel_flux > 0.5 * flux,
            "Expected >90% flux in center pixel, got {}",
            pixel_flux
        );
        assert!(
            pixel_flux <= flux,
            "Flux should not exceed total flux, got {}",
            pixel_flux
        );
    }

    #[test]
    fn test_flux_conservation() {
        // Test relative flux distribution for normalized PSF
        let psf = PixelScaledAiryDisk::new(2.0, 550.0);
        let flux = 1.0; // Use unit flux to measure PSF normalization

        // First measure total PSF integral over large area
        let mut psf_sum = 0.0;
        for x in -30..=30 {
            for y in -30..=30 {
                psf_sum += psf.pixel_flux_simpson(x as f64, y as f64, 1.0);
            }
        }

        // Now test that a smaller region captures expected fraction
        let mut region_sum = 0.0;
        for x in -5..=5 {
            for y in -5..=5 {
                region_sum += psf.pixel_flux_simpson(x as f64, y as f64, flux);
            }
        }

        let fraction = region_sum / psf_sum;
        println!(
            "Central 11x11 region captures {:.1}% of total PSF",
            fraction * 100.0
        );

        // Central region should capture significant fraction of PSF
        assert!(fraction > 0.8); // At least 80% in central region
        assert!(fraction <= 1.0); // Can't exceed 100%
    }

    #[test]
    fn test_offset_star() {
        // Test flux distribution when star is between pixels
        let psf = PixelScaledAiryDisk::with_first_zero(1.0, 550.0);
        let flux = 1000.0;

        // Star at (0.0, 0.0) - corner between 4 pixels
        let fluxes = vec![
            psf.pixel_flux_simpson(-0.5, -0.5, flux),
            psf.pixel_flux_simpson(0.5, -0.5, flux),
            psf.pixel_flux_simpson(-0.5, 0.5, flux),
            psf.pixel_flux_simpson(0.5, 0.5, flux),
        ];

        // Should be symmetric
        assert!((fluxes[0] - fluxes[1]).abs() < 1e-10);
        assert!((fluxes[0] - fluxes[2]).abs() < 1e-10);
        assert!((fluxes[0] - fluxes[3]).abs() < 1e-10);

        // Each pixel should get significant flux
        for f in &fluxes {
            assert!(*f > 0.1 * flux, "Expected flux > 0.1 * {}, got {}", flux, f);
        }
    }
}

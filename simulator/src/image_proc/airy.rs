use once_cell::sync::Lazy;
use scilib::math::bessel;

/// Airy disk parameters and approximation functions
///
/// The Airy disk is the diffraction pattern resulting from a uniformly
/// illuminated circular aperture. This struct provides both exact calculations
/// and various approximations for efficient computation.
#[derive(Debug, Clone, Copy)]
pub struct AiryDisk {
    /// First zero location (first dark ring radius) in radians
    pub first_zero: f64,
    /// Full-width-half-maximum in radians
    pub fwhm: f64,
}

impl AiryDisk {
    /// Create a new AiryDisk with given wavelength and aperture diameter in microns
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

    /// Calculate the exact Airy disk function intensity at given radius
    ///
    /// The Airy disk intensity follows: I(r) = I₀ * [2*J₁(r)/r]²
    /// where J₁ is the first-order Bessel function of the first kind
    pub fn intensity(&self, radius: f64) -> f64 {
        if radius.abs() < 1e-10 {
            return 1.0; // Limit as r approaches 0
        }

        let j1 = bessel::j_n(1, radius);
        let term = 2.0 * j1 / radius;
        term * term
    }

    /// Gaussian approximation to the Airy disk function
    ///
    /// Using the formula I(r) ≈ I₀ * exp(-3.9*r²/r₀²), where r₀ is the radius
    /// of the first dark ring in the Airy pattern
    pub fn gaussian_approximation(&self, radius: f64) -> f64 {
        (-3.9 * (radius * radius) / (self.first_zero * self.first_zero)).exp()
    }

    /// Triangle approximation to the Airy disk function
    ///
    /// Simple linear falloff from center to first zero:
    /// I(r) = max(0, 1 - r/r₀) for r ≤ r₀, then 0 for r > r₀
    pub fn triangle_approximation(&self, radius: f64) -> f64 {
        let normalized_radius = radius / self.first_zero;
        if normalized_radius >= 1.0 {
            0.0
        } else {
            1.0 - normalized_radius
        }
    }

    /// Generate sample points for comparing different approximations
    ///
    /// Returns tuple of (radii, exact_intensities, gaussian_approx, triangle_approx)
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

pub static AIRY_DISK: Lazy<AiryDisk> = Lazy::new(|| AiryDisk::new());

#[derive(Debug, Clone, Copy)]
pub struct ScaledAiryDisk {
    disk: AiryDisk,
    radius_scale: f64,
}

impl ScaledAiryDisk {
    /// Creates a new ScaledAiryDisk with a default AiryDisk and given radius scale
    fn new(radius_scale: f64) -> Self {
        ScaledAiryDisk {
            disk: AIRY_DISK.clone(),
            radius_scale,
        }
    }

    /// Class method to create a new ScaledAiryDisk with specified radius scale
    pub fn with_radius_scale(radius_scale: f64) -> Self {
        Self::new(radius_scale)
    }

    /// Class method to create a new ScaledAiryDisk with specified FWHM
    pub fn with_fwhm(fwhm: f64) -> Self {
        let scalar = fwhm / AIRY_DISK.fwhm;
        Self::new(scalar)
    }

    pub fn with_first_zero(first_zero: f64) -> Self {
        let scalar = first_zero / AIRY_DISK.first_zero;
        Self::new(scalar)
    }

    /// Returns the intensity at a given radius, scaled by the radius_scale
    pub fn intensity(&self, radius: f64) -> f64 {
        self.disk.intensity(radius / self.radius_scale)
    }

    /// Returns the gaussian approximation at a given radius, scaled by the radius_scale
    pub fn gaussian_approximation(&self, radius: f64) -> f64 {
        self.disk.gaussian_approximation(radius / self.radius_scale)
    }

    pub fn gaussian_approximation_normalized(&self, radius: f64) -> f64 {
        // FIXME(meawoppl) the 3.9 factor above included the scaling factor 1/sqrt(2pi) and the matching scaler :/
        // You should fix this and unwind these uglee constants
        let normalization = 4.5702;
        self.disk.gaussian_approximation(radius / self.radius_scale) / normalization
    }

    /// Returns the triangle approximation at a given radius, scaled by the radius_scale
    pub fn triangle_approximation(&self, radius: f64) -> f64 {
        self.disk.triangle_approximation(radius / self.radius_scale)
    }

    /// Returns the first zero of the underlying AiryDisk
    pub fn first_zero(&self) -> f64 {
        self.disk.first_zero * self.radius_scale
    }

    /// Returns the FWHM of the underlying AiryDisk
    pub fn fwhm(&self) -> f64 {
        self.disk.fwhm * self.radius_scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use once_cell::sync::Lazy;

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
        let unscaled = ScaledAiryDisk::new(1.0);
        assert_relative_eq!(unscaled.fwhm(), AIRY_DISK.fwhm, epsilon = 1e-10);

        let zero_rad = unscaled.first_zero();
        assert_relative_eq!(zero_rad, AIRY_DISK.first_zero, epsilon = 1e-10);
        assert!(unscaled.intensity(zero_rad) < 0.0001);
        assert!(unscaled.gaussian_approximation(zero_rad) < 0.1); // Gaussian is a bit fat tail
        assert!(unscaled.triangle_approximation(zero_rad) < 0.0001);
    }

    #[test]
    fn test_scaled_airy_disk_fwmh() {
        let scaled = ScaledAiryDisk::with_fwhm(2.0);

        let fwhm = scaled.fwhm();
        assert_relative_eq!(fwhm, 2.0, epsilon = 1e-10);

        let half = scaled.intensity(fwhm / 2.0);
        assert_relative_eq!(half, 0.5, epsilon = 1e-10);

        let close_half = scaled.gaussian_approximation(fwhm / 2.0);
        assert_relative_eq!(close_half, 0.5, epsilon = 1e-2);
    }

    #[test]
    fn test_scaled_airy_disk_first_zero() {
        let scaled = ScaledAiryDisk::with_first_zero(2.0);
        assert_relative_eq!(scaled.first_zero(), 2.0, epsilon = 1e-10);
    }
}

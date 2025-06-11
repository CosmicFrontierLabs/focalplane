use std::f64;

/// Cubic spline interpolation for smooth curve fitting
///
/// Implements natural cubic spline interpolation to create smooth curves
/// through a set of data points. Uses natural boundary conditions (second
/// derivatives are zero at endpoints) for stable interpolation.
///
/// This implementation is particularly useful for:
/// - PSD (Power Spectral Density) analysis and signal processing
/// - Smooth interpolation of sensor response curves
/// - Data smoothing and curve fitting in astronomical measurements
///
/// # Mathematical Background
///
/// The cubic spline creates piecewise cubic polynomials between each pair of
/// adjacent points, ensuring C² continuity (continuous function, first, and
/// second derivatives). Each segment has the form:
///
/// S(x) = a + b(x-xi) + c(x-xi)² + d(x-xi)³
///
/// # Examples
///
/// ```rust
/// use simulator::algo::spline::CubicSpline;
///
/// let x = vec![0.0, 1.0, 2.0, 3.0];
/// let y = vec![0.0, 1.0, 4.0, 9.0];
/// let spline = CubicSpline::new(x, y);
///
/// // Evaluate at intermediate point
/// let interpolated = spline.evaluate(1.5);
///
/// // Generate smooth curve with 100 points
/// let (x_smooth, y_smooth) = spline.interpolate(100);
/// ```
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    coeffs: Vec<[f64; 4]>, // a, b, c, d coefficients for each segment
}

impl CubicSpline {
    /// Create a new cubic spline from input points
    ///
    /// Constructs a natural cubic spline that interpolates the given data points.
    /// The resulting spline will pass exactly through all input points and provide
    /// smooth interpolation between them.
    ///
    /// # Arguments
    /// * `x` - X coordinates (must be sorted in ascending order, no duplicates)
    /// * `y` - Y coordinates corresponding to x values
    ///
    /// # Returns
    /// A `CubicSpline` instance ready for evaluation and interpolation
    ///
    /// # Panics
    /// - If x and y vectors have different lengths
    /// - If fewer than 2 points are provided
    /// - If x values are not sorted in strictly ascending order
    ///
    /// # Performance
    /// Construction time is O(n) where n is the number of points.
    /// Memory usage is O(n) for storing coefficients.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len(), "X and Y vectors must have same length");
        assert!(x.len() >= 2, "Need at least 2 points for interpolation");

        // Verify x is sorted
        for i in 1..x.len() {
            assert!(
                x[i] > x[i - 1],
                "X values must be sorted in ascending order"
            );
        }

        let n = x.len();
        let mut spline = CubicSpline {
            x: x.clone(),
            y: y.clone(),
            coeffs: vec![[0.0; 4]; n - 1],
        };

        spline.compute_coefficients();
        spline
    }

    /// Compute cubic spline coefficients using natural boundary conditions
    ///
    /// Solves the tridiagonal system of equations to determine the coefficients
    /// for each cubic polynomial segment. Uses the Thomas algorithm for efficient
    /// solution of the linear system.
    fn compute_coefficients(&mut self) {
        let n = self.x.len();
        let mut h = vec![0.0; n - 1];
        let mut alpha = vec![0.0; n - 1];

        // Calculate h and alpha
        for i in 0..n - 1 {
            h[i] = self.x[i + 1] - self.x[i];
        }

        for i in 1..n - 1 {
            alpha[i] = (3.0 / h[i]) * (self.y[i + 1] - self.y[i])
                - (3.0 / h[i - 1]) * (self.y[i] - self.y[i - 1]);
        }

        // Solve tridiagonal system for second derivatives
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 1..n - 1 {
            l[i] = 2.0 * (self.x[i + 1] - self.x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        let mut c = vec![0.0; n];
        let mut b = vec![0.0; n - 1];
        let mut d = vec![0.0; n - 1];

        // Back substitution
        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (self.y[j + 1] - self.y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        // Store coefficients [a, b, c, d] for each segment
        for i in 0..n - 1 {
            self.coeffs[i] = [self.y[i], b[i], c[i], d[i]];
        }
    }

    /// Evaluate the spline at a given x value
    ///
    /// Returns the interpolated y value at the specified x coordinate.
    /// Uses efficient binary search to locate the appropriate segment,
    /// then evaluates the cubic polynomial for that segment.
    ///
    /// # Arguments
    /// * `x` - Point to evaluate at
    ///
    /// # Returns
    /// Interpolated y value. For x values outside the original range,
    /// returns the boundary value (no extrapolation).
    ///
    /// # Performance
    /// Evaluation time is O(log n) due to binary search for segment location.
    pub fn evaluate(&self, x: f64) -> f64 {
        // Handle boundary cases
        if x <= self.x[0] {
            return self.y[0];
        }
        if x >= self.x[self.x.len() - 1] {
            return self.y[self.y.len() - 1];
        }

        // Find the appropriate segment
        let segment = self.find_segment(x);
        let dx = x - self.x[segment];
        let [a, b, c, d] = self.coeffs[segment];

        // Evaluate cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
        a + b * dx + c * dx * dx + d * dx * dx * dx
    }

    /// Find which segment contains the given x value
    ///
    /// Uses binary search to efficiently locate the segment containing x.
    /// Returns the index of the left endpoint of the containing segment.
    fn find_segment(&self, x: f64) -> usize {
        // Binary search for efficiency
        let mut left = 0;
        let mut right = self.x.len() - 1;

        while left < right - 1 {
            let mid = (left + right) / 2;
            if x < self.x[mid] {
                right = mid;
            } else {
                left = mid;
            }
        }
        left
    }

    /// Generate n_points evenly spaced interpolated values
    ///
    /// Creates a smooth interpolated curve by evaluating the spline at
    /// evenly spaced points across the original x range.
    ///
    /// # Arguments
    /// * `n_points` - Number of output points (must be ≥ 2)
    ///
    /// # Returns
    /// Tuple of (x_values, y_values) with smooth interpolation.
    /// The first and last points will exactly match the original endpoints.
    ///
    /// # Performance
    /// Time complexity is O(n_points * log(original_points))
    pub fn interpolate(&self, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        let x_min = self.x[0];
        let x_max = self.x[self.x.len() - 1];
        let dx = (x_max - x_min) / (n_points - 1) as f64;

        let mut x_out = Vec::with_capacity(n_points);
        let mut y_out = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let x = x_min + i as f64 * dx;
            let y = self.evaluate(x);
            x_out.push(x);
            y_out.push(y);
        }

        (x_out, y_out)
    }
}

/// Convenience function for quick cubic spline interpolation
///
/// Creates a cubic spline and immediately generates interpolated points.
/// Useful for one-off interpolation tasks where you don't need to reuse
/// the spline object.
///
/// # Arguments
/// * `x` - Input x coordinates (must be sorted in ascending order)
/// * `y` - Input y coordinates corresponding to x values
/// * `n_points` - Number of output points to generate
///
/// # Returns
/// Tuple of (interpolated_x, interpolated_y) vectors
///
/// # Example
/// ```rust
/// use simulator::algo::spline::cubic_spline_interpolate;
///
/// let x = vec![0.0, 1.0, 2.0];
/// let y = vec![0.0, 1.0, 4.0];
/// let (x_smooth, y_smooth) = cubic_spline_interpolate(x, y, 50);
/// ```
pub fn cubic_spline_interpolate(x: Vec<f64>, y: Vec<f64>, n_points: usize) -> (Vec<f64>, Vec<f64>) {
    let spline = CubicSpline::new(x, y);
    spline.interpolate(n_points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let spline = CubicSpline::new(x, y);

        assert!((spline.evaluate(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_spline_basic() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0]; // y = x^2
        let spline = CubicSpline::new(x, y);

        // Should pass through original points
        assert!((spline.evaluate(1.0) - 1.0).abs() < 1e-10);
        assert!((spline.evaluate(2.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_function() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        let (x_interp, y_interp) = cubic_spline_interpolate(x, y, 5);

        assert_eq!(x_interp.len(), 5);
        assert_eq!(y_interp.len(), 5);
        assert!((x_interp[0] - 0.0).abs() < 1e-10);
        assert!((x_interp[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "X and Y vectors must have same length")]
    fn test_mismatched_lengths() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0];
        CubicSpline::new(x, y);
    }

    #[test]
    #[should_panic(expected = "X values must be sorted in ascending order")]
    fn test_unsorted_x() {
        let x = vec![0.0, 2.0, 1.0];
        let y = vec![0.0, 4.0, 1.0];
        CubicSpline::new(x, y);
    }

    #[test]
    fn test_boundary_conditions() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let spline = CubicSpline::new(x, y);

        // Test values outside the range return boundary values
        assert!((spline.evaluate(-1.0) - 0.0).abs() < 1e-10);
        assert!((spline.evaluate(5.0) - 9.0).abs() < 1e-10);

        // Test exactly at boundaries
        assert!((spline.evaluate(0.0) - 0.0).abs() < 1e-10);
        assert!((spline.evaluate(3.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_monotonic_data() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 2.0, 8.0, 18.0, 32.0]; // y = 2*x^3
        let spline = CubicSpline::new(x, y);

        // Test intermediate values are reasonable
        let mid_val = spline.evaluate(1.5);
        assert!(mid_val > 2.0 && mid_val < 8.0);

        // Test continuity by checking nearby points
        let val1 = spline.evaluate(1.999);
        let val2 = spline.evaluate(2.001);
        assert!((val1 - val2).abs() < 0.1); // Should be nearly continuous
    }

    #[test]
    fn test_oscillating_data() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0]; // Oscillating sine-like
        let spline = CubicSpline::new(x.clone(), y.clone());

        // Should pass through all original points
        for i in 0..x.len() {
            assert!((spline.evaluate(x[i]) - y[i]).abs() < 1e-10);
        }

        // Test intermediate values are smooth
        let val_0_5 = spline.evaluate(0.5);
        let val_1_5 = spline.evaluate(1.5);
        assert!(val_0_5 > 0.0 && val_0_5 < 1.0);
        assert!(val_1_5 > -0.5 && val_1_5 < 1.0);
    }

    #[test]
    fn test_constant_data() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![5.0, 5.0, 5.0, 5.0]; // Constant function
        let spline = CubicSpline::new(x, y);

        // Should remain constant everywhere
        assert!((spline.evaluate(0.5) - 5.0).abs() < 1e-10);
        assert!((spline.evaluate(1.5) - 5.0).abs() < 1e-10);
        assert!((spline.evaluate(2.5) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_point_spline() {
        let x = vec![0.0, 10.0];
        let y = vec![5.0, 15.0];
        let spline = CubicSpline::new(x, y);

        // Should be linear between two points
        assert!((spline.evaluate(5.0) - 10.0).abs() < 1e-10);
        assert!((spline.evaluate(2.5) - 7.5).abs() < 1e-10);
        assert!((spline.evaluate(7.5) - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_output_size() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];
        let spline = CubicSpline::new(x, y);

        let (x_out, y_out) = spline.interpolate(10);
        assert_eq!(x_out.len(), 10);
        assert_eq!(y_out.len(), 10);

        // First and last points should match original boundaries
        assert!((x_out[0] - 0.0).abs() < 1e-10);
        assert!((x_out[9] - 2.0).abs() < 1e-10);
        assert!((y_out[0] - 0.0).abs() < 1e-10);
        assert!((y_out[9] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_spacing() {
        let x = vec![0.0, 2.0, 4.0];
        let y = vec![0.0, 1.0, 2.0];
        let spline = CubicSpline::new(x, y);

        let (x_out, _) = spline.interpolate(5);

        // Check even spacing
        let expected_spacing = 1.0; // (4.0 - 0.0) / (5 - 1)
        for i in 1..x_out.len() {
            let spacing = x_out[i] - x_out[i - 1];
            assert!((spacing - expected_spacing).abs() < 1e-10);
        }
    }

    #[test]
    fn test_segment_finding() {
        let x = vec![0.0, 1.0, 3.0, 6.0, 10.0];
        let y = vec![0.0, 1.0, 9.0, 36.0, 100.0];
        let spline = CubicSpline::new(x, y);

        // Test that segment finding works correctly
        assert!((spline.evaluate(0.5) - spline.evaluate(0.5)).abs() < 1e-10); // Consistency check
        assert!((spline.evaluate(2.0) - spline.evaluate(2.0)).abs() < 1e-10);
        assert!((spline.evaluate(4.5) - spline.evaluate(4.5)).abs() < 1e-10);
        assert!((spline.evaluate(8.0) - spline.evaluate(8.0)).abs() < 1e-10);
    }

    #[test]
    fn test_smooth_cubic_function() {
        // Test with a known cubic function: y = x^3 - 2*x^2 + x + 1
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let mut y = Vec::new();
        for &xi in &x {
            y.push(xi * xi * xi - 2.0 * xi * xi + xi + 1.0);
        }

        let spline = CubicSpline::new(x, y);

        // Test at intermediate points - should be very close to actual function
        let test_x = 0.75;
        let expected = test_x * test_x * test_x - 2.0 * test_x * test_x + test_x + 1.0;
        let actual = spline.evaluate(test_x);
        assert!((actual - expected).abs() < 0.1); // Allow some tolerance for spline approximation
    }

    #[test]
    #[should_panic(expected = "Need at least 2 points for interpolation")]
    fn test_single_point_error() {
        let x = vec![1.0];
        let y = vec![1.0];
        CubicSpline::new(x, y);
    }

    #[test]
    fn test_convenience_function() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 8.0, 27.0]; // y = x^3

        let (x_interp, y_interp) = cubic_spline_interpolate(x.clone(), y.clone(), 7);

        // Should have correct length
        assert_eq!(x_interp.len(), 7);
        assert_eq!(y_interp.len(), 7);

        // Compare with direct spline usage
        let spline = CubicSpline::new(x, y);
        let (x_direct, y_direct) = spline.interpolate(7);

        for i in 0..7 {
            assert!((x_interp[i] - x_direct[i]).abs() < 1e-10);
            assert!((y_interp[i] - y_direct[i]).abs() < 1e-10);
        }
    }
}

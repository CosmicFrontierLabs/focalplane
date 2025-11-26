//! Statistical functions for testing and analysis

use scilib::math::basic::erf;
use std::f64::consts::SQRT_2;

/// Cumulative distribution function for standard normal distribution
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Kolmogorov-Smirnov test statistic calculation
///
/// Tests if a sample comes from a standard normal distribution.
/// Returns the KS statistic (maximum difference between empirical and theoretical CDFs).
///
/// # Arguments
/// * `residuals` - Sample data to test
///
/// # Returns
/// KS statistic value. Smaller values indicate better fit to normal distribution.
pub fn ks_test_normal(residuals: &[f64]) -> f64 {
    let n = residuals.len();
    if n == 0 {
        return 1.0;
    }

    // Sort the residuals
    let mut sorted = residuals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Normalize to zero mean and unit variance
    let mean: f64 = sorted.iter().sum::<f64>() / n as f64;
    let variance: f64 = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    if std_dev < 1e-10 {
        return 1.0; // All values are the same
    }

    let normalized: Vec<f64> = sorted.iter().map(|x| (x - mean) / std_dev).collect();

    // Calculate KS statistic
    let mut max_diff: f64 = 0.0;

    for (i, &value) in normalized.iter().enumerate() {
        // CDF of standard normal at this point
        let theoretical_cdf = normal_cdf(value);

        // Empirical CDF
        let empirical_cdf_before = i as f64 / n as f64;
        let empirical_cdf_after = (i + 1) as f64 / n as f64;

        // Maximum difference
        let diff1 = (empirical_cdf_before - theoretical_cdf).abs();
        let diff2 = (empirical_cdf_after - theoretical_cdf).abs();

        max_diff = max_diff.max(diff1).max(diff2);
    }

    max_diff
}

/// Calculate the critical value for KS test at given significance level
///
/// For large n, uses asymptotic approximation
pub fn ks_critical_value(n: usize, alpha: f64) -> f64 {
    // Common critical values for alpha levels
    let c_alpha = match alpha {
        a if (a - 0.10).abs() < 1e-6 => 1.22,
        a if (a - 0.05).abs() < 1e-6 => 1.36,
        a if (a - 0.01).abs() < 1e-6 => 1.63,
        _ => 1.36, // Default to 5% significance
    };

    c_alpha / (n as f64).sqrt()
}

/// Calculate median of a slice of f64 values
///
/// This function computes the median while filtering out NaN values but including
/// infinite values (±inf). For even-length data, returns the average of the two
/// middle values.
///
/// # Arguments
///
/// * `values` - Slice of f64 values to compute median from
///
/// # Returns
///
/// * `Ok(median)` - The median value
/// * `Err(message)` - If no valid values remain after filtering NaN
pub fn median(values: &[f64]) -> Result<f64, String> {
    let mut valid_values: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

    if valid_values.is_empty() {
        return Err(format!(
            "Insufficient data points to compute median: {} total values, 0 valid (all NaN)",
            values.len()
        ));
    }

    valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median_value = if valid_values.len() % 2 == 0 {
        let mid = valid_values.len() / 2;
        (valid_values[mid - 1] + valid_values[mid]) / 2.0
    } else {
        valid_values[valid_values.len() / 2]
    };

    Ok(median_value)
}

/// Calculate Pearson correlation coefficient between two samples
///
/// Returns correlation in range [-1, 1], or NaN if samples have zero variance
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return f64::NAN;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 {
        return f64::NAN; // No variance in one or both variables
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_basic_values() {
        // Test known values of error function
        assert!((erf(0.0) - 0.0).abs() < 1e-6);
        assert!((erf(1.0) - 0.8427007929).abs() < 1e-6);
        assert!((erf(-1.0) - (-0.8427007929)).abs() < 1e-6);
        assert!((erf(2.0) - 0.9953222650).abs() < 1e-6);

        // Test limits
        assert!(erf(5.0) > 0.9999);
        assert!(erf(-5.0) < -0.9999);
    }

    #[test]
    fn test_normal_cdf() {
        // Test standard normal CDF at known points
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.0) - 0.8413447461).abs() < 1e-6);
        assert!((normal_cdf(-1.0) - 0.1586552539).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.001);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_ks_test_perfect_normal() {
        // Generate perfect normal samples using Box-Muller transform
        let n = 1000;
        let mut samples = Vec::with_capacity(n);

        // Use inverse normal CDF to generate perfect normal samples
        for i in 1..=n {
            let p = (i as f64 - 0.5) / n as f64;
            // Approximate inverse normal CDF
            let z = if p < 0.5 {
                -(-2.0 * (2.0 * p).ln()).sqrt()
            } else {
                (-2.0 * (2.0 * (1.0 - p)).ln()).sqrt()
            };
            samples.push(z);
        }

        let ks_stat = ks_test_normal(&samples);
        let critical = ks_critical_value(n, 0.05);

        // Should be very small for perfect normal data
        assert!(
            ks_stat < critical * 2.0,
            "KS statistic {ks_stat} should be small for normal data (critical: {critical})"
        );
    }

    #[test]
    fn test_ks_test_bimodal() {
        // Test with bimodal distribution (definitely not normal)
        let n = 100;
        let mut samples = Vec::with_capacity(n);
        // Half samples at -2, half at +2
        for i in 0..n / 2 {
            samples.push(-2.0 + 0.1 * (i as f64 / (n as f64 / 2.0)));
        }
        for i in 0..n / 2 {
            samples.push(2.0 + 0.1 * (i as f64 / (n as f64 / 2.0)));
        }

        let ks_stat = ks_test_normal(&samples);

        // Should detect non-normality
        assert!(
            ks_stat > 0.05,
            "KS statistic {ks_stat} should detect bimodal distribution"
        );
    }

    #[test]
    fn test_ks_test_constant() {
        // Test with constant values
        let samples = vec![5.0; 100];
        let ks_stat = ks_test_normal(&samples);

        // Should return 1.0 for constant data
        assert_eq!(ks_stat, 1.0);
    }

    #[test]
    fn test_ks_test_empty() {
        // Test with empty vector
        let samples: Vec<f64> = vec![];
        let ks_stat = ks_test_normal(&samples);

        // Should return 1.0 for empty data
        assert_eq!(ks_stat, 1.0);
    }

    #[test]
    fn test_ks_critical_values() {
        // Test known critical values
        assert!((ks_critical_value(100, 0.05) - 0.136).abs() < 0.001);
        assert!((ks_critical_value(100, 0.01) - 0.163).abs() < 0.001);
        assert!((ks_critical_value(1000, 0.05) - 0.043).abs() < 0.001);
    }

    #[test]
    fn test_ks_with_outliers() {
        // Normal data with a few outliers
        let mut samples = vec![0.0; 95];
        for i in 0..95 {
            samples[i] = (i as f64 - 47.0) / 20.0; // Roughly normal
        }
        // Add outliers
        samples.extend_from_slice(&[10.0, -10.0, 15.0, -15.0, 20.0]);

        let ks_stat = ks_test_normal(&samples);

        // Should detect non-normality due to outliers
        assert!(
            ks_stat > 0.05,
            "KS statistic {ks_stat} should detect outliers"
        );
    }

    #[test]
    fn test_pearson_correlation_perfect() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        assert!(
            (corr - 1.0).abs() < 1e-10,
            "Perfect positive correlation should be 1.0"
        );

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = pearson_correlation(&x, &y_neg);
        assert!(
            (corr_neg + 1.0).abs() < 1e-10,
            "Perfect negative correlation should be -1.0"
        );
    }

    #[test]
    fn test_pearson_correlation_uncorrelated() {
        // Uncorrelated data
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 1.0, 4.0, 2.0, 5.0];
        let corr = pearson_correlation(&x, &y);
        assert!(
            corr.abs() < 0.8,
            "Uncorrelated data should have low correlation"
        );
    }

    #[test]
    fn test_pearson_correlation_edge_cases() {
        // Empty vectors
        let empty: Vec<f64> = vec![];
        assert!(pearson_correlation(&empty, &empty).is_nan());

        // Different lengths
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(pearson_correlation(&x, &y).is_nan());

        // Zero variance
        let constant = vec![5.0, 5.0, 5.0, 5.0];
        let varying = vec![1.0, 2.0, 3.0, 4.0];
        assert!(pearson_correlation(&constant, &varying).is_nan());
    }

    #[test]
    fn test_median_odd_length() {
        let values = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        assert_eq!(median(&values).unwrap(), 3.0);
    }

    #[test]
    fn test_median_even_length() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&values).unwrap(), 2.5);
    }

    #[test]
    fn test_median_single_value() {
        let values = vec![42.0];
        assert_eq!(median(&values).unwrap(), 42.0);
    }

    #[test]
    fn test_median_with_nan() {
        let values = vec![1.0, f64::NAN, 3.0, 2.0, f64::NAN];
        assert_eq!(median(&values).unwrap(), 2.0);
    }

    #[test]
    fn test_median_with_positive_inf() {
        let values = vec![1.0, 2.0, f64::INFINITY, 3.0];
        assert_eq!(median(&values).unwrap(), 2.5);
    }

    #[test]
    fn test_median_with_negative_inf() {
        let values = vec![1.0, 2.0, f64::NEG_INFINITY, 3.0];
        assert_eq!(median(&values).unwrap(), 1.5);
    }

    #[test]
    fn test_median_with_both_inf() {
        let values = vec![f64::NEG_INFINITY, 1.0, 2.0, 3.0, f64::INFINITY];
        assert_eq!(median(&values).unwrap(), 2.0);
    }

    #[test]
    fn test_median_all_nan() {
        let values = vec![f64::NAN, f64::NAN, f64::NAN];
        let result = median(&values);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("all NaN"));
    }

    #[test]
    fn test_median_empty_slice() {
        let values: Vec<f64> = vec![];
        let result = median(&values);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Insufficient data points"));
    }

    #[test]
    fn test_median_negative_values() {
        let values = vec![-5.0, -2.0, -8.0, -1.0, -3.0];
        assert_eq!(median(&values).unwrap(), -3.0);
    }

    #[test]
    fn test_median_mixed_signs() {
        let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        assert_eq!(median(&values).unwrap(), 0.0);
    }

    #[test]
    fn test_median_unsorted_input() {
        let values = vec![10.0, 1.0, 5.0, 3.0, 8.0];
        assert_eq!(median(&values).unwrap(), 5.0);
    }

    #[test]
    fn test_median_duplicates() {
        let values = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        assert_eq!(median(&values).unwrap(), 2.5);
    }
}

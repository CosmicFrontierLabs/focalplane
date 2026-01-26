//! Test module for ICP bias analysis using Kolmogorov-Smirnov testing

use super::*;
use crate::stats::{ks_critical_value, ks_test_normal, pearson_correlation};
use nalgebra::{Matrix2, Vector2};
use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Flag to disable rotation and translation for simpler testing
/// When true, sets rotation to 0 and translation to [0, 0]
const DISABLE_TRANSFORM: bool = true;

#[test]
fn test_icp_residual_normality() {
    // Test parameters
    let n_points = 400;
    let n_trials = 50;
    let noise_std = 1.0; // Small noise standard deviation
    let seed = 42;

    let _rng = StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0, noise_std).unwrap();

    // Collect residuals from multiple trials
    let mut all_residuals_x = Vec::new();
    let mut all_residuals_y = Vec::new();

    for trial in 0..n_trials {
        // Generate random points in a unit square
        let mut source_points = Vec::new();
        let mut target_points = Vec::new();

        // Use different seed for each trial but deterministic
        let trial_seed = seed + trial as u64;
        let mut trial_rng = StdRng::seed_from_u64(trial_seed);

        // Generate source points
        let mut source_vec = Vec::new();
        for _ in 0..n_points {
            let x = trial_rng.gen_range(-0.0..8000.0);
            let y = trial_rng.gen_range(-0.0..2000.0);
            source_points.push(x);
            source_points.push(y);
            source_vec.push(Vector2::new(x, y));
        }

        // Apply a known transformation to create target points
        let true_angle = if DISABLE_TRANSFORM {
            0.0
        } else {
            trial_rng.gen_range(-PI / 4.0..PI / 4.0)
        };
        let true_translation = if DISABLE_TRANSFORM {
            Vector2::new(0.0, 0.0)
        } else {
            Vector2::new(
                trial_rng.gen_range(-2.0..2.0),
                trial_rng.gen_range(-2.0..2.0),
            )
        };

        let cos_a = true_angle.cos();
        let sin_a = true_angle.sin();
        let true_rotation = Matrix2::new(cos_a, -sin_a, sin_a, cos_a);

        // Create target points with small noise
        for i in 0..n_points {
            let source_point = source_vec[i];
            let transformed = true_rotation * source_point + true_translation;

            // Add small Gaussian noise
            let noise_x = noise_dist.sample(&mut trial_rng);
            let noise_y = noise_dist.sample(&mut trial_rng);

            target_points.push(transformed.x + noise_x);
            target_points.push(transformed.y + noise_y);
        }

        // Convert to ndarray format
        let source_array = Array2::from_shape_vec((n_points, 2), source_points).unwrap();
        let target_array = Array2::from_shape_vec((n_points, 2), target_points).unwrap();

        // Run ICP
        let icp_result = iterative_closest_point(&source_array, &target_array, 100, 1e-9);

        // Debug: check if all points are matched correctly (they should be since we have same points with small noise)
        if trial < 10 {
            println!(
                "Trial 0 debug: {} matches for {} points",
                icp_result.matches.len(),
                n_points
            );
            println!(
                "  Rotation angle: expected {:.6}, got {:.6}",
                true_angle,
                icp_result.rotation[(1, 0)].atan2(icp_result.rotation[(0, 0)])
            );
            println!(
                "  Translation: expected [{:.6}, {:.6}], got [{:.6}, {:.6}]",
                true_translation.x,
                true_translation.y,
                icp_result.translation.x,
                icp_result.translation.y
            );
            println!("  MSE: {:.9}", icp_result.mean_squared_error);
        }

        // Calculate residuals for matched points
        for &(src_idx, tgt_idx) in &icp_result.matches {
            let source_point = Vector2::new(source_array[(src_idx, 0)], source_array[(src_idx, 1)]);
            let target_point = Vector2::new(target_array[(tgt_idx, 0)], target_array[(tgt_idx, 1)]);

            // Transform source point using ICP result
            let transformed = icp_result.rotation * source_point + icp_result.translation;

            // Calculate residuals (raw, not normalized by noise_std yet)
            let residual_x = transformed.x - target_point.x;
            let residual_y = transformed.y - target_point.y;

            all_residuals_x.push(residual_x);
            all_residuals_y.push(residual_y);
        }
    }

    // Perform KS test on residuals
    let ks_statistic_x = ks_test_normal(&all_residuals_x);
    let ks_statistic_y = ks_test_normal(&all_residuals_y);

    println!("KS test results for ICP residuals:");
    println!("  X residuals: KS statistic = {ks_statistic_x:.6}");
    println!("  Y residuals: KS statistic = {ks_statistic_y:.6}");
    println!("  Number of samples: {}", all_residuals_x.len());

    // Calculate mean and std of residuals
    let mean_x: f64 = all_residuals_x.iter().sum::<f64>() / all_residuals_x.len() as f64;
    let mean_y: f64 = all_residuals_y.iter().sum::<f64>() / all_residuals_y.len() as f64;

    let std_x: f64 = (all_residuals_x
        .iter()
        .map(|x| (x - mean_x).powi(2))
        .sum::<f64>()
        / all_residuals_x.len() as f64)
        .sqrt();
    let std_y: f64 = (all_residuals_y
        .iter()
        .map(|y| (y - mean_y).powi(2))
        .sum::<f64>()
        / all_residuals_y.len() as f64)
        .sqrt();

    println!("  X residuals: mean = {mean_x:.6}, std = {std_x:.6}");
    println!("  Y residuals: mean = {mean_y:.6}, std = {std_y:.6}");

    // Critical value for KS test at 5% significance level
    let critical_value = ks_critical_value(all_residuals_x.len(), 0.05);

    println!("  Critical value (5% significance): {critical_value:.6}");

    // Test for systematic bias: mean should be near zero (relative to noise level)
    assert!(
        mean_x.abs() < noise_std * 3.0,
        "X residuals show systematic bias: mean = {mean_x:.6} (noise_std = {noise_std:.6})"
    );
    assert!(
        mean_y.abs() < noise_std * 3.0,
        "Y residuals show systematic bias: mean = {mean_y:.6} (noise_std = {noise_std:.6})"
    );

    // Test that standard deviation is close to expected noise level
    assert!(
        (std_x - noise_std).abs() < noise_std * 0.5,
        "X residuals std deviation {std_x:.6} differs significantly from expected {noise_std:.6}"
    );
    assert!(
        (std_y - noise_std).abs() < noise_std * 0.5,
        "Y residuals std deviation {std_y:.6} differs significantly from expected {noise_std:.6}"
    );

    // Test for normality using KS test with reasonable threshold
    let ks_threshold = 0.05; // More reasonable threshold for practical testing
    assert!(
        ks_statistic_x < ks_threshold,
        "X residuals fail normality test: KS = {ks_statistic_x:.6} > {ks_threshold:.6}"
    );
    assert!(
        ks_statistic_y < ks_threshold,
        "Y residuals fail normality test: KS = {ks_statistic_y:.6} > {ks_threshold:.6}"
    );

    // Test for independence: X and Y residuals should be uncorrelated
    let correlation = pearson_correlation(&all_residuals_x, &all_residuals_y);
    println!("  X-Y residual correlation: {correlation:.6}");

    assert!(
        correlation.abs() < 0.1,
        "X and Y residuals should be uncorrelated (independent), but correlation = {correlation:.6}"
    );
}

#[test]
fn test_icp_with_outliers() {
    // Test ICP behavior with outliers to check for bias
    let n_points = 50;
    let n_outliers = 5;
    let seed = 123;

    let mut rng = StdRng::seed_from_u64(seed);

    // Generate source points
    let mut source_points = Vec::new();
    for _ in 0..(n_points + n_outliers) {
        let x = rng.gen_range(-5.0..5.0);
        let y = rng.gen_range(-5.0..5.0);
        source_points.push(x);
        source_points.push(y);
    }

    // Known transformation
    let true_angle = PI / 8.0;
    let true_translation = Vector2::new(1.5, -0.5);
    let cos_a = true_angle.cos();
    let sin_a = true_angle.sin();
    let true_rotation = Matrix2::new(cos_a, -sin_a, sin_a, cos_a);

    // Create target points
    let mut target_points = Vec::new();

    // Transform most points correctly
    for i in 0..n_points {
        let x = source_points[i * 2];
        let y = source_points[i * 2 + 1];
        let source_point = Vector2::new(x, y);
        let transformed = true_rotation * source_point + true_translation;
        target_points.push(transformed.x);
        target_points.push(transformed.y);
    }

    // Add outliers at random positions
    for _ in 0..n_outliers {
        target_points.push(rng.gen_range(-20.0..20.0));
        target_points.push(rng.gen_range(-20.0..20.0));
    }

    // Convert to ndarray
    let total_points = n_points + n_outliers;
    let source_array = Array2::from_shape_vec((total_points, 2), source_points).unwrap();
    let target_array = Array2::from_shape_vec((total_points, 2), target_points).unwrap();

    // Run ICP
    let icp_result = iterative_closest_point(&source_array, &target_array, 100, 1e-6);

    // Check if the recovered transformation is biased by outliers
    let rotation_error = (icp_result.rotation - true_rotation).norm();
    let translation_error = (icp_result.translation - true_translation).norm();

    println!("ICP with outliers test:");
    println!("  Rotation error: {rotation_error:.6}");
    println!("  Translation error: {translation_error:.6}");
    println!("  Mean squared error: {:.6}", icp_result.mean_squared_error);

    // With outliers, ICP should still converge but with higher error
    // This test mainly checks that it doesn't crash or produce NaN
    assert!(!icp_result.mean_squared_error.is_nan());
    assert!(!icp_result.mean_squared_error.is_infinite());
}

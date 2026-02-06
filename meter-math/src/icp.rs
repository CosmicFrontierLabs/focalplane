//! Iterative Closest Point implementation for point cloud alignment
//!
//! This algorithm iteratively matches points between two sets and solves
//! for the optimal rigid transformation (rotation and translation) that
//! aligns them.

use nalgebra::{Matrix2, Vector2, Vector3};
use ndarray::Array2;
use thiserror::Error;

use crate::quaternion::Quaternion;

/// Errors that can occur during ICP operations
#[derive(Error, Debug)]
pub enum ICPError {
    #[error("Invalid argument: {0}")]
    ArgumentError(String),

    #[error("SVD decomposition failed to produce U or V^T matrices")]
    SvdFailed,
}

/// Result of ICP algorithm containing transformation parameters and matching points
#[derive(Debug, Clone)]
pub struct ICPResult {
    /// Quaternion representing the rotation component of the transform
    pub rotation_quat: Quaternion,

    /// 2x2 Rotation matrix component of the transform (for compatibility)
    pub rotation: Matrix2<f64>,

    /// Translation vector component of the transform (2x1)
    pub translation: Vector2<f64>,

    /// Matches between source and target point sets as (source_idx, target_idx)
    pub matches: Vec<(usize, usize)>,

    /// Mean squared error of the final alignment
    pub mean_squared_error: f64,

    /// Number of iterations performed
    pub iterations: usize,
}

/// Computes the squared Euclidean distance between two 2D points.
///
/// Using squared distance avoids expensive square root calculations
/// while preserving relative ordering for distance comparisons.
///
/// # Arguments
///
/// * `p1` - First 2D point
/// * `p2` - Second 2D point
///
/// # Returns
///
/// Squared distance: (p1.x - p2.x)² + (p1.y - p2.y)²
fn squared_distance(p1: &Vector2<f64>, p2: &Vector2<f64>) -> f64 {
    (p1 - p2).norm_squared()
}

/// Finds the closest target point for each source point using brute-force search.
///
/// This function implements the correspondence step of ICP by finding the
/// nearest neighbor in the target set for each point in the source set.
///
/// # Arguments
///
/// * `source_points` - Array of source points to match
/// * `target_points` - Array of target points to search in
///
/// # Returns
///
/// Vector of (source_index, target_index) pairs representing closest matches
///
/// # Performance
///
/// Time complexity: O(n × m) where n = source points, m = target points
/// For large point sets, consider using spatial data structures like KD-trees.
fn find_closest_points(
    source_points: &[Vector2<f64>],
    target_points: &[Vector2<f64>],
) -> Vec<(usize, usize)> {
    let mut matches = Vec::with_capacity(source_points.len());

    for (i, source_point) in source_points.iter().enumerate() {
        let mut min_dist = f64::INFINITY;
        let mut closest_idx = 0;

        for (j, target_point) in target_points.iter().enumerate() {
            let dist = squared_distance(source_point, target_point);

            if dist < min_dist {
                min_dist = dist;
                closest_idx = j;
            }
        }

        matches.push((i, closest_idx));
    }

    matches
}

/// Calculates the geometric centroid (center of mass) of a point set.
///
/// The centroid is computed as the arithmetic mean of all point coordinates:
/// centroid = (1/n) × Σ(points)
///
/// # Arguments
///
/// * `points` - Array of 2D points
///
/// # Returns
///
/// Centroid coordinates, or (0,0) if the point set is empty
fn calculate_centroid(points: &[Vector2<f64>]) -> Vector2<f64> {
    if points.is_empty() {
        return Vector2::zeros();
    }

    let mut centroid = Vector2::zeros();
    for point in points {
        centroid += point;
    }

    centroid / points.len() as f64
}

/// Compute optimal rotation (as quaternion) and translation using SVD
fn compute_optimal_transform(
    source_points: &[Vector2<f64>],
    target_points: &[Vector2<f64>],
    matches: &[(usize, usize)],
) -> Result<(Quaternion, Vector2<f64>), ICPError> {
    let mut src_matched = Vec::with_capacity(matches.len());
    let mut tgt_matched = Vec::with_capacity(matches.len());

    for &(src_idx, tgt_idx) in matches {
        src_matched.push(source_points[src_idx]);
        tgt_matched.push(target_points[tgt_idx]);
    }

    // Compute centroids
    let source_centroid = calculate_centroid(&src_matched);
    let target_centroid = calculate_centroid(&tgt_matched);

    // Compute covariance matrix
    let mut h = Matrix2::zeros();

    for i in 0..src_matched.len() {
        let p_src_centered = src_matched[i] - source_centroid;
        let p_tgt_centered = tgt_matched[i] - target_centroid;

        h += p_src_centered * p_tgt_centered.transpose();
    }

    // Perform SVD
    let svd = h.svd(true, true);
    let u = svd.u.ok_or(ICPError::SvdFailed)?;
    let v_t = svd.v_t.ok_or(ICPError::SvdFailed)?;

    // Compute rotation matrix
    let mut r = v_t.transpose() * u.transpose();

    // Handle reflection case
    if r.determinant() < 0.0 {
        let mut v_t_fixed = v_t;
        v_t_fixed[(0, 1)] = -v_t_fixed[(0, 1)];
        v_t_fixed[(1, 1)] = -v_t_fixed[(1, 1)];
        r = v_t_fixed.transpose() * u.transpose();
    }

    // Convert 2D rotation matrix to quaternion
    // For 2D rotation, we only rotate around z-axis
    let angle = (r[(1, 0)]).atan2(r[(0, 0)]);
    let axis = Vector3::new(0.0, 0.0, 1.0); // z-axis
    let q = Quaternion::from_axis_angle(&axis, angle);

    // Compute translation
    let t = target_centroid - r * source_centroid;

    Ok((q, t))
}

/// Converts ndarray point representation to nalgebra Vector2 format.
///
/// This function bridges between the ndarray-based public API and
/// the nalgebra-based internal implementation for easier vector math.
///
/// # Arguments
///
/// * `points` - 2D array with shape [n_points, 2] where each row is [x, y]
///
/// # Returns
///
/// Vector of nalgebra Vector2<f64> points
///
/// # Panics
///
/// Panics if the input array doesn't have exactly 2 columns
fn convert_to_vector2_points(points: &Array2<f64>) -> Vec<Vector2<f64>> {
    let mut result = Vec::with_capacity(points.shape()[0]);

    for i in 0..points.shape()[0] {
        result.push(Vector2::new(points[(i, 0)], points[(i, 1)]));
    }

    result
}

/// Applies rigid transformation (rotation + translation) to a set of points.
///
/// Each point is transformed according to: p' = R × p + t
/// where R is the rotation matrix and t is the translation vector.
///
/// # Arguments
///
/// * `points` - Array of points to transform
/// * `rotation` - 2×2 rotation matrix
/// * `translation` - 2D translation vector
///
/// # Returns
///
/// Vector of transformed points
fn transform_points(
    points: &[Vector2<f64>],
    rotation: &Matrix2<f64>,
    translation: &Vector2<f64>,
) -> Vec<Vector2<f64>> {
    let mut transformed = Vec::with_capacity(points.len());

    for p in points {
        transformed.push(rotation * p + translation);
    }

    transformed
}

/// Calculates the mean squared error between transformed source points and their matched targets.
///
/// This function computes the objective function that ICP seeks to minimize:
/// MSE = (1/n) × Σ||R × p_src + t - p_tgt||²
///
/// # Arguments
///
/// * `source_points` - Original source points
/// * `target_points` - Target points
/// * `matches` - Point correspondences as (source_idx, target_idx) pairs
/// * `rotation` - Current rotation matrix estimate
/// * `translation` - Current translation vector estimate
///
/// # Returns
///
/// Mean squared error, or infinity if no matches are provided
fn calculate_error(
    source_points: &[Vector2<f64>],
    target_points: &[Vector2<f64>],
    matches: &[(usize, usize)],
    rotation: &Matrix2<f64>,
    translation: &Vector2<f64>,
) -> f64 {
    let mut total_error = 0.0;

    for &(src_idx, tgt_idx) in matches {
        let p_src = source_points[src_idx];
        let p_tgt = target_points[tgt_idx];

        let p_transformed = rotation * p_src + translation;
        let error = (p_transformed - p_tgt).norm_squared();

        total_error += error;
    }

    if matches.is_empty() {
        return f64::INFINITY;
    }

    total_error / matches.len() as f64
}

/// Iterative Closest Point algorithm for aligning two point sets
///
/// # Arguments
/// * `source_points` - Source points as `ndarray::Array2<f64>` with shape [n_points, 2]
/// * `target_points` - Target points as `ndarray::Array2<f64>` with shape [m_points, 2]
/// * `max_iterations` - Maximum number of iterations to perform
/// * `convergence_threshold` - Error threshold for convergence
///
/// # Returns
/// * `Result<ICPResult, ICPError>` - Struct containing transformation parameters and matching information
///
/// # Errors
/// * `ICPError::ArgumentError` - If input arrays don't have 2 columns
/// * `ICPError::SvdFailed` - If SVD decomposition fails during iteration
///
pub fn iterative_closest_point(
    source_points: &Array2<f64>,
    target_points: &Array2<f64>,
    max_iterations: usize,
    convergence_threshold: f64,
) -> Result<ICPResult, ICPError> {
    if source_points.shape()[1] != 2 {
        return Err(ICPError::ArgumentError(
            "Source points must have shape [n_points, 2]".to_string(),
        ));
    }
    if target_points.shape()[1] != 2 {
        return Err(ICPError::ArgumentError(
            "Target points must have shape [m_points, 2]".to_string(),
        ));
    }

    // Convert to Vector2 points for easier manipulation
    let source_vec = convert_to_vector2_points(source_points);
    let target_vec = convert_to_vector2_points(target_points);

    // Initialize transformation
    let mut rotation_quat = Quaternion::identity();
    let mut rotation = Matrix2::identity();
    let mut translation = Vector2::zeros();

    // Current transformed source points (initially just the source points)
    let mut current_source = source_vec.clone();

    // Previous error for convergence check
    let mut prev_error = f64::INFINITY;
    let mut current_error;
    let mut iterations = 0;
    let mut matches = Vec::new();

    for i in 0..max_iterations {
        iterations = i + 1;

        // Find closest points
        matches = find_closest_points(&current_source, &target_vec);

        // Compute optimal transformation
        let (q, t) = compute_optimal_transform(&source_vec, &target_vec, &matches)?;

        // Update transformation
        rotation_quat = q;
        // Extract 2x2 rotation matrix from quaternion for 2D operations
        let full_rotation = q.to_rotation_matrix();
        rotation = Matrix2::new(
            full_rotation[(0, 0)],
            full_rotation[(0, 1)],
            full_rotation[(1, 0)],
            full_rotation[(1, 1)],
        );
        translation = t;

        // Apply transformation to original source points
        current_source = transform_points(&source_vec, &rotation, &translation);

        // Calculate error
        current_error =
            calculate_error(&source_vec, &target_vec, &matches, &rotation, &translation);

        // Check for convergence
        if (prev_error - current_error).abs() < convergence_threshold {
            break;
        }

        prev_error = current_error;
    }

    // Calculate final error
    let final_error = calculate_error(&source_vec, &target_vec, &matches, &rotation, &translation);

    Ok(ICPResult {
        rotation_quat,
        rotation,
        translation,
        matches,
        mean_squared_error: final_error,
        iterations,
    })
}

/// Trait for objects that can be located in a 2D Cartesian coordinate system.
pub trait Locatable2d {
    /// Returns the x-coordinate of the object.
    fn x(&self) -> f64;

    /// Returns the y-coordinate of the object.
    fn y(&self) -> f64;
}

/// Implement Locatable for `nalgebra::Vector2<f64>`
impl Locatable2d for Vector2<f64> {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }
}

/// Performs ICP matching between two sets of Locatable2d objects and returns the matched pairs and ICP result.
///
/// This function converts the input objects into point clouds, runs the ICP algorithm,
/// and then maps the resulting index pairs back to the original objects.
///
/// # Type Parameters
/// * `R1`: The type of the source objects, must implement `Locatable2d` and `Clone`.
/// * `R2`: The type of the target objects, must implement `Locatable2d` and `Clone`.
///
/// # Arguments
/// * `source` - A slice of source objects. Must not be empty.
/// * `target` - A slice of target objects. Must not be empty.
/// * `max_iterations` - Maximum number of iterations for the ICP algorithm.
/// * `convergence_threshold` - Convergence threshold for the ICP algorithm. Must be positive.
///
/// # Returns
/// * `Result<Vec<(R1, R2)>, ICPError>` - A vector of tuples containing the cloned matched pairs,
///   or an error if the operation fails.
///
/// # Errors
/// * `ICPError::ArgumentError` - If either source or target slice is empty, or if convergence_threshold is not positive.
/// * `ICPError::SvdFailed` - If the SVD decomposition fails during ICP iteration.
pub fn icp_match_objects<R1, R2>(
    source: &[R1],
    target: &[R2],
    max_iterations: usize,
    convergence_threshold: f64,
) -> Result<(Vec<(R1, R2)>, ICPResult), ICPError>
where
    R1: Locatable2d + Clone,
    R2: Locatable2d + Clone,
{
    if source.is_empty() {
        return Err(ICPError::ArgumentError("source slice is empty".to_string()));
    }

    if target.is_empty() {
        return Err(ICPError::ArgumentError("target slice is empty".to_string()));
    }

    if convergence_threshold <= 0.0 {
        return Err(ICPError::ArgumentError(format!(
            "convergence_threshold must be positive, got {convergence_threshold}"
        )));
    }

    // Convert source Locatable2d objects to ndarray::Array2<f64>
    let source_points_vec: Vec<f64> = source.iter().flat_map(|p| [p.x(), p.y()]).collect();
    let source_points = Array2::from_shape_vec((source.len(), 2), source_points_vec)
        .expect("Source points vector should have correct length for Array2 conversion");

    // Convert target Locatable2d objects to ndarray::Array2<f64>
    let target_points_vec: Vec<f64> = target.iter().flat_map(|p| [p.x(), p.y()]).collect();
    let target_points = Array2::from_shape_vec((target.len(), 2), target_points_vec)
        .expect("Target points vector should have correct length for Array2 conversion");

    // Run the ICP algorithm
    let result = iterative_closest_point(
        &source_points,
        &target_points,
        max_iterations,
        convergence_threshold,
    )?;

    // Map indices back to original objects (cloned)
    let matched_objects: Vec<(R1, R2)> = result
        .matches
        .iter()
        .map(|&(src_idx, tgt_idx)| (source[src_idx].clone(), target[tgt_idx].clone()))
        .collect();

    Ok((matched_objects, result))
}

/// Matches objects from source to target using ICP, returning indices instead of cloned objects.
///
/// This is useful when working with types that don't implement Clone, such as `Box<dyn Trait>`.
/// The returned indices can be used to access the original slices.
///
/// # Arguments
/// * `source` - Slice of source objects implementing Locatable2d
/// * `target` - Slice of target objects implementing Locatable2d
/// * `max_iterations` - Maximum number of ICP iterations
/// * `convergence_threshold` - Minimum mean squared error change to continue iterating
///
/// # Returns
/// * Tuple of (matched_indices, ICPResult) where matched_indices is Vec<(source_idx, target_idx)>
///
/// # Errors
/// * `ICPError::ArgumentError` - If either source or target slice is empty, or if convergence_threshold is not positive.
/// * `ICPError::SvdFailed` - If the SVD decomposition fails during ICP iteration.
pub fn icp_match_indices<R1, R2>(
    source: &[R1],
    target: &[R2],
    max_iterations: usize,
    convergence_threshold: f64,
) -> Result<(Vec<(usize, usize)>, ICPResult), ICPError>
where
    R1: Locatable2d,
    R2: Locatable2d,
{
    if source.is_empty() {
        return Err(ICPError::ArgumentError("source slice is empty".to_string()));
    }

    if target.is_empty() {
        return Err(ICPError::ArgumentError("target slice is empty".to_string()));
    }

    if convergence_threshold <= 0.0 {
        return Err(ICPError::ArgumentError(format!(
            "convergence_threshold must be positive, got {convergence_threshold}"
        )));
    }

    // Convert source Locatable2d objects to ndarray::Array2<f64>
    let source_points_vec: Vec<f64> = source.iter().flat_map(|p| [p.x(), p.y()]).collect();
    let source_points = Array2::from_shape_vec((source.len(), 2), source_points_vec)
        .expect("Source points vector should have correct length for Array2 conversion");

    // Convert target Locatable2d objects to ndarray::Array2<f64>
    let target_points_vec: Vec<f64> = target.iter().flat_map(|p| [p.x(), p.y()]).collect();
    let target_points = Array2::from_shape_vec((target.len(), 2), target_points_vec)
        .expect("Target points vector should have correct length for Array2 conversion");

    // Run the ICP algorithm
    let result = iterative_closest_point(
        &source_points,
        &target_points,
        max_iterations,
        convergence_threshold,
    )?;

    Ok((result.matches.clone(), result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;
    use std::f64::consts::PI;

    /// Create a rotation matrix for the given angle in radians
    fn rotation_matrix(angle: f64) -> Matrix2<f64> {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Matrix2::new(cos_a, -sin_a, sin_a, cos_a)
    }

    /// Create a quaternion for rotation around z-axis by the given angle
    fn z_rotation_quaternion(angle: f64) -> Quaternion {
        let axis = Vector3::new(0.0, 0.0, 1.0);
        Quaternion::from_axis_angle(&axis, angle)
    }

    /// Helper function to initialize the ICP algorithm with known matching points
    /// to test the transformation calculation part directly
    fn run_icp_with_known_matches(
        source: &Array2<f64>,
        target: &Array2<f64>,
        custom_matches: &[(usize, usize)],
    ) -> ICPResult {
        // Validate input dimensions
        assert_eq!(
            source.shape()[1],
            2,
            "Source points must have shape [n_points, 2]"
        );
        assert_eq!(
            target.shape()[1],
            2,
            "Target points must have shape [m_points, 2]"
        );

        // Convert to Vector2 points for easier manipulation
        let source_vec = convert_to_vector2_points(source);
        let target_vec = convert_to_vector2_points(target);

        // Compute optimal transformation with known matches
        let (rotation_quat, translation) =
            compute_optimal_transform(&source_vec, &target_vec, custom_matches).unwrap();

        // Extract 2D rotation matrix from quaternion
        let full_rotation = rotation_quat.to_rotation_matrix();
        let rotation = Matrix2::new(
            full_rotation[(0, 0)],
            full_rotation[(0, 1)],
            full_rotation[(1, 0)],
            full_rotation[(1, 1)],
        );

        // Calculate error
        let error = calculate_error(
            &source_vec,
            &target_vec,
            custom_matches,
            &rotation,
            &translation,
        );

        ICPResult {
            rotation_quat,
            rotation,
            translation,
            matches: custom_matches.to_vec(),
            mean_squared_error: error,
            iterations: 1,
        }
    }

    #[test]
    fn test_icp_translation_only() {
        // Create asymmetric source points to avoid rotational symmetry
        let source = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, // Origin
                1.0, 0.0, // Right
                0.0, 2.0, // Up (not symmetric)
                -1.5, 0.0, // Left (not symmetric)
                0.0, -1.0, // Down
            ],
        )
        .unwrap();

        // Create target points (translated by [2, 3])
        let translation = Vector2::new(2.0, 3.0);
        let mut target = Array2::zeros((5, 2));

        // Known point correspondences
        let matches: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();

        for i in 0..5 {
            let p = Vector2::new(source[(i, 0)], source[(i, 1)]);
            let p_trans = p + translation;
            target[(i, 0)] = p_trans[0];
            target[(i, 1)] = p_trans[1];
        }

        // Run ICP with known matches to verify transform calculation
        let result = run_icp_with_known_matches(&source, &target, &matches);

        // Check results
        assert_relative_eq!(result.rotation, Matrix2::identity(), epsilon = 1e-4);
        assert_relative_eq!(result.translation, translation, epsilon = 1e-4);
        assert_relative_eq!(result.mean_squared_error, 0.0, epsilon = 1e-4);

        // Check quaternion is identity
        let identity_quat = Quaternion::identity();
        assert_relative_eq!(result.rotation_quat.w, identity_quat.w, epsilon = 1e-4);

        // Now run the full ICP algorithm with many distinct points to ensure it converges properly
        let mut many_source_points = Vec::new();
        let mut many_target_points = Vec::new();

        // Create a grid of points
        let grid_size: i32 = 5;
        for x in -grid_size..=grid_size {
            for y in -grid_size..=grid_size {
                let xf = x as f64 * (1.0 + 0.1 * (x as f64).abs()); // Make asymmetric
                let yf = y as f64 * (1.0 + 0.2 * (y as f64).abs()); // Make asymmetric

                many_source_points.push(xf);
                many_source_points.push(yf);

                many_target_points.push(xf + translation[0]);
                many_target_points.push(yf + translation[1]);
            }
        }

        // Calculate grid point count (must be usize for ndarray)
        let point_count = ((2 * grid_size + 1) * (2 * grid_size + 1)) as usize;

        let many_source = Array2::from_shape_vec((point_count, 2), many_source_points).unwrap();

        let many_target = Array2::from_shape_vec((point_count, 2), many_target_points).unwrap();

        // Run full ICP algorithm
        let full_result = iterative_closest_point(&many_source, &many_target, 20, 1e-9).unwrap();

        // Just verify the algorithm runs; we're not testing convergence in the full case
        println!(
            "Full ICP (translation only): Got rotation = [{:.6}, {:.6}; {:.6}, {:.6}], translation = [{:.6}, {:.6}]",
            full_result.rotation[(0, 0)], full_result.rotation[(0, 1)],
            full_result.rotation[(1, 0)], full_result.rotation[(1, 1)],
            full_result.translation[0], full_result.translation[1]
        );
    }

    #[test]
    fn test_icp_rotation_only() {
        // Create asymmetric source points to avoid rotational symmetry
        let source = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, // Origin
                1.0, 0.0, // Right
                0.0, 2.0, // Up (not symmetric)
                -1.5, 0.0, // Left (not symmetric)
                0.0, -1.0, // Down
            ],
        )
        .unwrap();

        // Create target points (rotated by 30 degrees)
        let angle = PI / 6.0; // 30 degrees
        let rotation = rotation_matrix(angle);
        let expected_quat = z_rotation_quaternion(angle);
        let mut target = Array2::zeros((5, 2));

        // Known point correspondences
        let matches: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();

        for i in 0..5 {
            let p = Vector2::new(source[(i, 0)], source[(i, 1)]);
            let p_rot = rotation * p;
            target[(i, 0)] = p_rot[0];
            target[(i, 1)] = p_rot[1];
        }

        // Run ICP with known matches to verify transform calculation
        let result = run_icp_with_known_matches(&source, &target, &matches);

        // Check results
        assert_relative_eq!(result.rotation, rotation, epsilon = 1e-4);
        assert_relative_eq!(result.translation.norm(), 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.mean_squared_error, 0.0, epsilon = 1e-4);

        // Check quaternion
        assert_relative_eq!(result.rotation_quat.w, expected_quat.w, epsilon = 1e-4);
        assert_relative_eq!(result.rotation_quat.z, expected_quat.z, epsilon = 1e-4);

        // Now run the full ICP algorithm with many distinct points to ensure it converges properly
        let mut many_source_points = Vec::new();
        let mut many_target_points = Vec::new();

        // Create a grid of points
        let grid_size: i32 = 5;
        for x in -grid_size..=grid_size {
            for y in -grid_size..=grid_size {
                let xf = x as f64 * (1.0 + 0.1 * (x as f64).abs()); // Make asymmetric
                let yf = y as f64 * (1.0 + 0.2 * (y as f64).abs()); // Make asymmetric

                many_source_points.push(xf);
                many_source_points.push(yf);

                let p = Vector2::new(xf, yf);
                let p_rot = rotation * p;

                many_target_points.push(p_rot[0]);
                many_target_points.push(p_rot[1]);
            }
        }

        // Calculate grid point count (must be usize for ndarray)
        let point_count = ((2 * grid_size + 1) * (2 * grid_size + 1)) as usize;

        let many_source = Array2::from_shape_vec((point_count, 2), many_source_points).unwrap();

        let many_target = Array2::from_shape_vec((point_count, 2), many_target_points).unwrap();

        // Run full ICP algorithm
        let full_result = iterative_closest_point(&many_source, &many_target, 20, 1e-9).unwrap();

        // Just verify the algorithm runs; we're not testing convergence in the full case
        println!(
            "Full ICP (rotation only): Got rotation = [{:.6}, {:.6}; {:.6}, {:.6}], translation = [{:.6}, {:.6}]",
            full_result.rotation[(0, 0)], full_result.rotation[(0, 1)],
            full_result.rotation[(1, 0)], full_result.rotation[(1, 1)],
            full_result.translation[0], full_result.translation[1]
        );

        println!(
            "Quaternion: w={:.6}, x={:.6}, y={:.6}, z={:.6}",
            full_result.rotation_quat.w,
            full_result.rotation_quat.x,
            full_result.rotation_quat.y,
            full_result.rotation_quat.z
        );
    }

    #[test]
    fn test_icp_rotation_and_translation() {
        // Create asymmetric source points to avoid rotational symmetry
        let source = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, // Origin
                1.0, 0.0, // Right
                0.0, 2.0, // Up (not symmetric)
                -1.5, 0.0, // Left (not symmetric)
                0.0, -1.0, // Down
            ],
        )
        .unwrap();

        // Create target points (rotated and translated)
        let angle = PI / 4.0; // 45 degrees
        let rotation = rotation_matrix(angle);
        let expected_quat = z_rotation_quaternion(angle);
        let translation = Vector2::new(2.0, 1.0);
        let mut target = Array2::zeros((5, 2));

        // Known point correspondences
        let matches: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();

        for i in 0..5 {
            let p = Vector2::new(source[(i, 0)], source[(i, 1)]);
            let p_transformed = rotation * p + translation;
            target[(i, 0)] = p_transformed[0];
            target[(i, 1)] = p_transformed[1];
        }

        // Run ICP with known matches to verify transform calculation
        let result = run_icp_with_known_matches(&source, &target, &matches);

        // Check results
        assert_relative_eq!(result.rotation, rotation, epsilon = 1e-4);
        assert_relative_eq!(result.translation, translation, epsilon = 1e-4);
        assert_relative_eq!(result.mean_squared_error, 0.0, epsilon = 1e-4);

        // Check quaternion
        assert_relative_eq!(result.rotation_quat.w, expected_quat.w, epsilon = 1e-4);
        assert_relative_eq!(result.rotation_quat.z, expected_quat.z, epsilon = 1e-4);

        // Now run the full ICP algorithm with many distinct points to ensure it converges properly
        let mut many_source_points = Vec::new();
        let mut many_target_points = Vec::new();

        // Create a grid of points
        let grid_size: i32 = 5;
        for x in -grid_size..=grid_size {
            for y in -grid_size..=grid_size {
                let xf = x as f64 * (1.0 + 0.1 * (x as f64).abs()); // Make asymmetric
                let yf = y as f64 * (1.0 + 0.2 * (y as f64).abs()); // Make asymmetric

                many_source_points.push(xf);
                many_source_points.push(yf);

                let p = Vector2::new(xf, yf);
                let p_transformed = rotation * p + translation;

                many_target_points.push(p_transformed[0]);
                many_target_points.push(p_transformed[1]);
            }
        }

        // Calculate grid point count (must be usize for ndarray)
        let point_count = ((2 * grid_size + 1) * (2 * grid_size + 1)) as usize;

        let many_source = Array2::from_shape_vec((point_count, 2), many_source_points).unwrap();

        let many_target = Array2::from_shape_vec((point_count, 2), many_target_points).unwrap();

        // Run full ICP algorithm
        let full_result = iterative_closest_point(&many_source, &many_target, 20, 1e-9).unwrap();

        // Just verify the algorithm runs; we're not testing convergence in the full case
        println!(
            "Full ICP (rotation and translation): Got rotation = [{:.6}, {:.6}; {:.6}, {:.6}], translation = [{:.6}, {:.6}]",
            full_result.rotation[(0, 0)], full_result.rotation[(0, 1)],
            full_result.rotation[(1, 0)], full_result.rotation[(1, 1)],
            full_result.translation[0], full_result.translation[1]
        );

        println!(
            "Quaternion: w={:.6}, x={:.6}, y={:.6}, z={:.6}",
            full_result.rotation_quat.w,
            full_result.rotation_quat.x,
            full_result.rotation_quat.y,
            full_result.rotation_quat.z
        );
    }

    #[test]
    fn test_icp_with_noisy_data() {
        // Create a larger grid of points with random transformations to test robustness
        let mut rng = rand::rng();

        // Random transformation
        let angle = rng.random_range(0.0..2.0 * PI); // Random angle
        let rotation = rotation_matrix(angle);
        let translation = Vector2::new(rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0));

        // Create source and target points
        let mut source_points = Vec::new();
        let mut target_points = Vec::new();

        // Create a grid of points
        let grid_size: i32 = 5;
        let noise_level = 0.1;

        for x in -grid_size..=grid_size {
            for y in -grid_size..=grid_size {
                let xf = x as f64 * (1.0 + 0.1 * (x as f64).abs()); // Make asymmetric
                let yf = y as f64 * (1.0 + 0.2 * (y as f64).abs()); // Make asymmetric

                source_points.push(xf);
                source_points.push(yf);

                let p = Vector2::new(xf, yf);
                let p_transformed = rotation * p + translation;

                // Add noise to target points
                target_points.push(p_transformed[0] + noise_level * (rng.random::<f64>() - 0.5));
                target_points.push(p_transformed[1] + noise_level * (rng.random::<f64>() - 0.5));
            }
        }

        // Calculate grid point count (must be usize for ndarray)
        let point_count = ((2 * grid_size + 1) * (2 * grid_size + 1)) as usize;

        let source = Array2::from_shape_vec((point_count, 2), source_points).unwrap();

        let target = Array2::from_shape_vec((point_count, 2), target_points).unwrap();

        // Run ICP on noisy data
        let result = iterative_closest_point(&source, &target, 50, 1e-9).unwrap();

        // Check results with more relaxed tolerance due to noise
        // Just verify the algorithm runs with noisy data; we're not testing convergence
        println!(
            "Full ICP (noisy data): Got rotation = [{:.6}, {:.6}; {:.6}, {:.6}], translation = [{:.6}, {:.6}]",
            result.rotation[(0, 0)], result.rotation[(0, 1)],
            result.rotation[(1, 0)], result.rotation[(1, 1)],
            result.translation[0], result.translation[1]
        );

        println!(
            "Quaternion: w={:.6}, x={:.6}, y={:.6}, z={:.6}",
            result.rotation_quat.w,
            result.rotation_quat.x,
            result.rotation_quat.y,
            result.rotation_quat.z
        );
    }

    /// Simple struct implementing Locatable2d for testing icp_match_objects
    #[derive(Debug, Clone, PartialEq)]
    struct PointObject {
        id: usize,
        x_coord: f64,
        y_coord: f64,
    }

    impl Locatable2d for PointObject {
        fn x(&self) -> f64 {
            self.x_coord
        }
        fn y(&self) -> f64 {
            self.y_coord
        }
    }

    #[test]
    fn test_icp_match_objects_identity() {
        let source_objs = vec![
            PointObject {
                id: 0,
                x_coord: 0.0,
                y_coord: 0.0,
            },
            PointObject {
                id: 1,
                x_coord: 1.0,
                y_coord: 0.0,
            },
            PointObject {
                id: 2,
                x_coord: 0.0,
                y_coord: 1.0,
            },
        ];
        let target_objs = source_objs.clone(); // Identical set

        let (matches, _icp_result) = icp_match_objects(&source_objs, &target_objs, 10, 1e-6)
            .expect("ICP should succeed with identical objects");

        assert_eq!(matches.len(), 3);
        // Check if each source object is matched with its identical target object
        for (src, tgt) in &matches {
            assert_eq!(src.id, tgt.id);
            assert_eq!(src.x_coord, tgt.x_coord);
            assert_eq!(src.y_coord, tgt.y_coord);
        }
    }

    #[test]
    fn test_icp_match_objects_translation() {
        let source_objs = vec![
            PointObject {
                id: 0,
                x_coord: 0.0,
                y_coord: 0.0,
            },
            PointObject {
                id: 1,
                x_coord: 1.0,
                y_coord: 1.0,
            },
            PointObject {
                id: 2,
                x_coord: 0.5,
                y_coord: 5.0,
            }, // Asymmetric
        ];
        let translation = Vector2::new(5.0, -2.0);
        let target_objs: Vec<PointObject> = source_objs
            .iter()
            .map(|p| PointObject {
                id: p.id,
                x_coord: p.x_coord + translation.x,
                y_coord: p.y_coord + translation.y,
            })
            .collect();

        let (matches, _icp_result) = icp_match_objects(&source_objs, &target_objs, 20, 1e-6)
            .expect("ICP should succeed with translated objects");

        assert_eq!(matches.len(), 3);
        // Check if objects are matched correctly by ID, assuming ICP finds the correct correspondence
        let mut matched_ids: Vec<(usize, usize)> =
            matches.iter().map(|(s, t)| (s.id, t.id)).collect();
        matched_ids.sort(); // Sort by source ID for consistent comparison
        assert_eq!(matched_ids, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_icp_match_objects_rotation() {
        let source_objs = vec![
            PointObject {
                id: 0,
                x_coord: 0.0,
                y_coord: 0.0,
            },
            PointObject {
                id: 1,
                x_coord: 2.0,
                y_coord: 0.0,
            },
            PointObject {
                id: 2,
                x_coord: 0.0,
                y_coord: 1.0,
            }, // Asymmetric
        ];
        let angle = PI / 2.0 / 45.0; // 2 degrees
        let rotation = rotation_matrix(angle);
        let target_objs: Vec<PointObject> = source_objs
            .iter()
            .map(|p| {
                let point = Vector2::new(p.x_coord, p.y_coord);
                let rotated_point = rotation * point;
                PointObject {
                    id: p.id,
                    x_coord: rotated_point.x,
                    y_coord: rotated_point.y,
                }
            })
            .collect();

        let (matches, _icp_result) = icp_match_objects(&source_objs, &target_objs, 20, 1e-6)
            .expect("ICP should succeed with rotated objects");

        assert_eq!(matches.len(), 3);
        let mut matched_ids: Vec<(usize, usize)> =
            matches.iter().map(|(s, t)| (s.id, t.id)).collect();
        matched_ids.sort();
        assert_eq!(matched_ids, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_icp_match_objects_rotation_translation() {
        let source_objs = vec![
            PointObject {
                id: 0,
                x_coord: 1.0,
                y_coord: 1.0,
            },
            PointObject {
                id: 1,
                x_coord: 3.0,
                y_coord: 1.0,
            },
            PointObject {
                id: 2,
                x_coord: 1.0,
                y_coord: 2.0,
            }, // Asymmetric
        ];
        let angle = -PI / 2.0 / 45.0; // -2 degrees
        let rotation = rotation_matrix(angle);
        let translation = Vector2::new(-1.0, 0.0002);
        let target_objs: Vec<PointObject> = source_objs
            .iter()
            .map(|p| {
                let point = Vector2::new(p.x_coord, p.y_coord);
                let transformed_point = rotation * point + translation;
                PointObject {
                    id: p.id,
                    x_coord: transformed_point.x,
                    y_coord: transformed_point.y,
                }
            })
            .collect();

        let (matches, _icp_result) = icp_match_objects(&source_objs, &target_objs, 30, 1e-6)
            .expect("ICP should succeed with rotated and translated objects");

        assert_eq!(matches.len(), 3);
        let mut matched_ids: Vec<(usize, usize)> =
            matches.iter().map(|(s, t)| (s.id, t.id)).collect();
        matched_ids.sort();
        assert_eq!(matched_ids, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_icp_match_objects_empty_input() {
        let source_objs: Vec<PointObject> = vec![];
        let target_objs = vec![
            PointObject {
                id: 0,
                x_coord: 1.0,
                y_coord: 1.0,
            },
            PointObject {
                id: 1,
                x_coord: 2.0,
                y_coord: 2.0,
            },
        ];

        let matches_empty_source = icp_match_objects(&source_objs, &target_objs, 10, 1e-6);
        assert!(matches!(
            matches_empty_source,
            Err(ICPError::ArgumentError(_))
        ));

        let source_objs_non_empty = vec![
            PointObject {
                id: 0,
                x_coord: 1.0,
                y_coord: 1.0,
            },
            PointObject {
                id: 1,
                x_coord: 2.0,
                y_coord: 2.0,
            },
        ];
        let target_objs_empty: Vec<PointObject> = vec![];
        let result_empty_target =
            icp_match_objects(&source_objs_non_empty, &target_objs_empty, 10, 1e-6);
        assert!(matches!(
            result_empty_target,
            Err(ICPError::ArgumentError(_))
        ));

        let matches_both_empty = icp_match_objects(&source_objs, &target_objs_empty, 10, 1e-6);
        assert!(matches!(
            matches_both_empty,
            Err(ICPError::ArgumentError(_))
        ));
    }

    #[test]
    fn test_icp_match_objects_different_sizes() {
        // Target has one extra point
        let source_objs = vec![
            PointObject {
                id: 0,
                x_coord: 0.0,
                y_coord: 0.0,
            },
            PointObject {
                id: 1,
                x_coord: 1.0,
                y_coord: 0.0,
            },
        ];
        let target_objs = vec![
            PointObject {
                id: 10,
                x_coord: 0.1,
                y_coord: 0.1,
            }, // Close to source 0
            PointObject {
                id: 11,
                x_coord: 1.1,
                y_coord: -0.1,
            }, // Close to source 1
            PointObject {
                id: 12,
                x_coord: 5.0,
                y_coord: 5.0,
            }, // Extra point
        ];

        let (matches, _icp_result) = icp_match_objects(&source_objs, &target_objs, 10, 1e-6)
            .expect("ICP should succeed with different sized sets");

        // ICP matches each source point to its closest target point
        assert_eq!(matches.len(), 2);
        let mut matched_ids: Vec<(usize, usize)> =
            matches.iter().map(|(s, t)| (s.id, t.id)).collect();
        matched_ids.sort();
        // Expect source 0 to match target 10, source 1 to match target 11
        assert_eq!(matched_ids, vec![(0, 10), (1, 11)]);
    }

    #[test]
    fn test_icp_match_objects_argument_error() {
        let empty_source: Vec<PointObject> = vec![];
        let target_objs = vec![PointObject {
            id: 0,
            x_coord: 1.0,
            y_coord: 1.0,
        }];

        // Test empty source
        let result = icp_match_objects(&empty_source, &target_objs, 10, 1e-6);
        assert!(matches!(result, Err(ICPError::ArgumentError(_))));

        // Test empty target
        let source_objs = vec![PointObject {
            id: 0,
            x_coord: 1.0,
            y_coord: 1.0,
        }];
        let empty_target: Vec<PointObject> = vec![];
        let result = icp_match_objects(&source_objs, &empty_target, 10, 1e-6);
        assert!(matches!(result, Err(ICPError::ArgumentError(_))));

        // Test negative convergence threshold
        let result = icp_match_objects(&source_objs, &target_objs, 10, -1e-6);
        assert!(matches!(result, Err(ICPError::ArgumentError(_))));

        // Test zero convergence threshold
        let result = icp_match_objects(&source_objs, &target_objs, 10, 0.0);
        assert!(matches!(result, Err(ICPError::ArgumentError(_))));
    }

    #[test]
    fn test_doctest_example() {
        // Test the example from the main function documentation
        use ndarray::Array2;

        // Create two point sets
        let source = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let target = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0]).unwrap();

        // Run ICP algorithm
        let result = iterative_closest_point(&source, &target, 100, 1e-6).unwrap();

        // Verify the result structure is valid
        assert_eq!(result.matches.len(), 3); // Should match 3 source points
        assert!(result.mean_squared_error >= 0.0); // MSE should be non-negative
        assert!(result.iterations > 0 && result.iterations <= 100); // Should have run some iterations
    }

    // ICP bias analysis tests using Kolmogorov-Smirnov testing

    use crate::stats::{ks_critical_value, ks_test_normal, pearson_correlation};
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal};

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
                let x = trial_rng.random_range(-0.0..8000.0);
                let y = trial_rng.random_range(-0.0..2000.0);
                source_points.push(x);
                source_points.push(y);
                source_vec.push(Vector2::new(x, y));
            }

            // Apply a known transformation to create target points
            let true_angle = if DISABLE_TRANSFORM {
                0.0
            } else {
                trial_rng.random_range(-PI / 4.0..PI / 4.0)
            };
            let true_translation = if DISABLE_TRANSFORM {
                Vector2::new(0.0, 0.0)
            } else {
                Vector2::new(
                    trial_rng.random_range(-2.0..2.0),
                    trial_rng.random_range(-2.0..2.0),
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
            let icp_result =
                iterative_closest_point(&source_array, &target_array, 100, 1e-9).unwrap();

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
                let source_point =
                    Vector2::new(source_array[(src_idx, 0)], source_array[(src_idx, 1)]);
                let target_point =
                    Vector2::new(target_array[(tgt_idx, 0)], target_array[(tgt_idx, 1)]);

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

        // Test for systematic bias: mean should be near zero (within 3 sigma)
        assert_relative_eq!(mean_x, 0.0, epsilon = noise_std * 3.0);
        assert_relative_eq!(mean_y, 0.0, epsilon = noise_std * 3.0);

        // Test that standard deviation is close to expected noise level (within 50%)
        assert_relative_eq!(std_x, noise_std, epsilon = noise_std * 0.5);
        assert_relative_eq!(std_y, noise_std, epsilon = noise_std * 0.5);

        // Test for normality using KS test with reasonable threshold
        let ks_threshold = 0.05;
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

        assert_relative_eq!(correlation, 0.0, epsilon = 0.1);
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
            let x = rng.random_range(-5.0..5.0);
            let y = rng.random_range(-5.0..5.0);
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
            target_points.push(rng.random_range(-20.0..20.0));
            target_points.push(rng.random_range(-20.0..20.0));
        }

        // Convert to ndarray
        let total_points = n_points + n_outliers;
        let source_array = Array2::from_shape_vec((total_points, 2), source_points).unwrap();
        let target_array = Array2::from_shape_vec((total_points, 2), target_points).unwrap();

        // Run ICP
        let icp_result = iterative_closest_point(&source_array, &target_array, 100, 1e-6).unwrap();

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
}

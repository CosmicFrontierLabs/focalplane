//! Parallel processing utilities for image and array operations
//!
//! This module provides functions for processing arrays in parallel
//! with deterministic seeding for reproducible results.

use ndarray::{Array2, Axis};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Process an Array2 in parallel chunks with deterministic seeding
///
/// This function processes a 2D array in parallel using row-wise chunks
/// for better cache locality and deterministic results. Each chunk gets
/// a unique RNG seeded from the base seed plus the chunk index.
///
/// # Arguments
/// * `array` - The 2D array to process
/// * `seed` - Base seed for random number generation
/// * `chunk_size` - Optional chunk size (number of rows per chunk). Defaults to 64 if None.
/// * `processor` - Closure that processes each chunk with its own RNG
///
/// # Type Parameters
/// * `F` - Closure type that takes a mutable array chunk and a random number generator
///
/// # Returns
/// The processed array
pub fn process_array_in_parallel_chunks<F>(
    mut array: Array2<f64>,
    seed: u64,
    chunk_size: Option<usize>,
    processor: F,
) -> Array2<f64>
where
    F: Fn(&mut ndarray::ArrayViewMut2<f64>, &mut StdRng) + Send + Sync,
{
    let chunk_size = chunk_size.unwrap_or(64);

    array
        .axis_chunks_iter_mut(Axis(0), chunk_size)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut chunk)| {
            // Each chunk gets its own RNG with a deterministic seed derived from the base seed
            let chunk_seed = seed.wrapping_add(chunk_idx as u64);
            let mut rng = StdRng::seed_from_u64(chunk_seed);

            // Apply the processor to this chunk with its RNG
            processor(&mut chunk, &mut rng);
        });

    array
}

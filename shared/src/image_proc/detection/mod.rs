//! Astronomical object detection algorithms and utilities.
//!
//! This module provides a comprehensive suite of star and object detection algorithms
//! for astronomical image analysis. Includes both sophisticated detection methods
//! (DAO, IRAF) and simple approaches (naive centroiding, thresholding) suitable for
//! different astronomical applications and performance requirements.
//!
//! # Module Organization
//!
//! - **detection**: Unified interface for multiple detection algorithms
//! - **config**: Pre-optimized configurations for space telescope observations
//! - **naive**: Simple centroiding-based detection for basic applications
//! - **thresholding**: Threshold-based detection and connected component analysis
//! - **aabb**: Axis-aligned bounding boxes for region management
//!
//! # Algorithm Comparison
//!
//! | Algorithm | Speed | Accuracy | Best Use Case |
//! |-----------|-------|----------|---------------|
//! | DAO       | Slow  | Highest  | Crowded fields, faint sources |
//! | IRAF      | Medium| High     | General-purpose detection |
//! | Naive     | Fast  | Medium   | Bright, isolated sources |
//! | Threshold | Fastest| Basic   | Quick analysis, preprocessing |
//!

pub mod aabb;
pub mod config;
pub mod naive;
pub mod thresholding;
pub mod unified;

pub use aabb::{aabbs_to_tuples, merge_overlapping_aabbs, tuples_to_aabbs, union_aabbs, AABB};
pub use naive::{calculate_star_centroid, detect_stars, get_centroids, StarDetection};
pub use thresholding::{apply_threshold, connected_components, get_bounding_boxes, otsu_threshold};
pub use unified::{detect_stars as detect_stars_unified, DetectionError, StarFinder};

pub mod aabb;
pub mod config;
pub mod detection;
pub mod naive;
pub mod thresholding;

pub use aabb::*;
pub use detection::{detect_stars as detect_stars_unified, StarFinder};
pub use naive::{
    calculate_star_centroid, detect_stars, do_detections, get_centroids, StarDetection,
};
pub use thresholding::*;

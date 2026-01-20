pub mod apriltag;
pub mod assets;
pub mod checkerboard;
pub mod circling_pixel;
pub mod motion;
pub mod pixel;
pub mod pixel_grid;
pub mod remote_controlled;
pub mod shared;
pub mod siemens_star;
pub mod static_noise;
pub mod uniform;
pub mod usaf;
pub mod wiggling_gaussian;

pub use motion::{CircularMotion, MotionTrajectory, Position2D};

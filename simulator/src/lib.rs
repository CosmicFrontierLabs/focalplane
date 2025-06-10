//! Space telescope optical and sensor simulation
//!
//! This crate provides functionality for simulating the optical system
//! and sensor hardware of a space telescope, as well as image processing
//! routines for analyzing the resulting images.

pub mod algo;
pub mod hardware;
pub mod image_proc;
pub mod photometry;
pub mod shared_args;
pub mod star_math;

// Re-exports for easier access
pub use algo::icp::{iterative_closest_point, ICPResult};
pub use hardware::sensor::SensorConfig;
pub use hardware::star_projection::{
    field_diameter, filter_stars_in_field, magnitude_to_electrons, pixel_scale,
};
pub use hardware::telescope::TelescopeConfig;
pub use image_proc::histogram_stretch::stretch_histogram;
pub use photometry::quantum_efficiency::QuantumEfficiency;
pub use photometry::spectrum::{Spectrum, CGS};
pub use photometry::trapezoid::trap_integrate;
pub use starfield::catalogs::StarPosition;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

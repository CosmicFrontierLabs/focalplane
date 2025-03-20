//! Space telescope optical and sensor simulation
//!
//! This crate provides functionality for simulating the optical system
//! and sensor hardware of a space telescope, as well as image processing
//! routines for analyzing the resulting images.

pub mod hardware;
pub mod image_proc;

// Re-exports for easier access
pub use hardware::sensor::SensorConfig;
pub use hardware::star_projection::{
    field_diameter, filter_stars_in_field, magnitude_to_photon_flux, pixel_scale,
};
pub use hardware::telescope::TelescopeConfig;
pub use image_proc::histogram_stretch::stretch_histogram;
pub use starfield::catalogs::StarPosition;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

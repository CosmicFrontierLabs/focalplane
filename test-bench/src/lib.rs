mod calibration;
pub mod calibration_controller;
pub mod calibration_overlay;
pub mod camera_init;
pub mod camera_server;
pub mod display_patterns;

#[cfg(feature = "sdl2")]
pub mod calibrate;

#[cfg(feature = "sdl2")]
pub mod display_utils;

#[cfg(target_os = "linux")]
pub mod gpio;

#[cfg(target_os = "linux")]
pub mod orin_monitoring;

#[cfg(feature = "playerone")]
pub use hardware::poa;

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        assert_eq!(2 + 2, 4);
    }
}

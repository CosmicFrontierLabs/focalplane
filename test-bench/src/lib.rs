pub mod calibration_controller;
pub mod camera_init;
pub mod display_patterns;
pub mod embedded_assets;
pub mod tracking_collector;
pub mod ws_stream;

#[cfg(feature = "pi-fsm")]
pub mod camera_server;

#[cfg(feature = "pi-fsm")]
pub mod fsm_calibration;
pub mod mjpeg;

#[cfg(feature = "sdl2")]
pub mod calibrate;

#[cfg(feature = "sdl2")]
pub mod display_utils;

#[cfg(target_os = "linux")]
pub mod gpio;

#[cfg(all(target_os = "linux", feature = "orin"))]
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

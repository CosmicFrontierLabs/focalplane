mod calibration;
pub mod calibration_overlay;
pub mod camera_init;
pub mod camera_server;
pub mod display_patterns;

#[cfg(feature = "sdl2")]
pub mod display_utils;

pub mod gpio;

#[cfg(feature = "playerone")]
pub mod poa;

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_test() {
        assert_eq!(2 + 2, 4);
    }
}

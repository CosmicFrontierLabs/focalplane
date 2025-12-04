use std::ffi::CStr;
use std::os::unix::io::AsRawFd;
use v4l::prelude::*;
use v4l::video::Capture;

#[derive(Debug, Clone)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
    pub fps: Option<f64>,
}

pub fn get_available_resolutions(device_path: &str) -> anyhow::Result<Vec<Resolution>> {
    let device = Device::with_path(device_path)?;
    let mut resolutions = Vec::new();

    // Get first format (RG16)
    if let Some(fmt) = device.enum_formats()?.into_iter().next() {
        if let Ok(framesizes) = device.enum_framesizes(fmt.fourcc) {
            for size in framesizes {
                if let v4l::framesize::FrameSizeEnum::Discrete(discrete) = size.size {
                    // Get frame rate for this resolution
                    let fps = if let Ok(intervals) =
                        device.enum_frameintervals(fmt.fourcc, discrete.width, discrete.height)
                    {
                        if let Some(interval) = intervals.into_iter().next() {
                            match interval.interval {
                                v4l::frameinterval::FrameIntervalEnum::Discrete(disc) => {
                                    Some(disc.denominator as f64 / disc.numerator as f64)
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    resolutions.push(Resolution {
                        width: discrete.width,
                        height: discrete.height,
                        fps,
                    });
                }
            }
        }
    }

    Ok(resolutions)
}

pub fn query_menu_item(fd: i32, ctrl_id: u32, index: u32) -> Option<String> {
    unsafe {
        #[repr(C)]
        struct v4l2_querymenu {
            id: u32,
            index: u32,
            name: [u8; 32],
            reserved: u32,
        }

        let mut querymenu: v4l2_querymenu = std::mem::zeroed();
        querymenu.id = ctrl_id;
        querymenu.index = index;

        // VIDIOC_QUERYMENU ioctl value
        const VIDIOC_QUERYMENU: std::os::raw::c_ulong = 0xc02c5625;

        let ret = libc::ioctl(
            fd,
            VIDIOC_QUERYMENU,
            &mut querymenu as *mut _ as *mut std::os::raw::c_void,
        );

        if ret == 0 {
            // Check if it's a named menu item
            if querymenu.name[0] != 0 {
                let c_str = CStr::from_ptr(querymenu.name.as_ptr() as *const std::os::raw::c_char);
                c_str.to_str().ok().map(|s| s.to_string())
            } else {
                Some("(unnamed item)".to_string())
            }
        } else {
            None
        }
    }
}

#[derive(Default)]
pub struct CameraMetadata {
    pub driver: String,
    pub card: String,
    pub bus: String,
    pub formats: Vec<String>,
    pub resolutions: Vec<String>,
    pub controls: Vec<String>,
    pub test_patterns: Vec<String>,
}

pub fn collect_camera_metadata(device_path: &str) -> anyhow::Result<CameraMetadata> {
    let device = Device::with_path(device_path)?;
    let mut metadata = CameraMetadata::default();

    // Query device capabilities
    let caps = device.query_caps()?;
    metadata.driver = caps.driver.clone();
    metadata.card = caps.card.clone();
    metadata.bus = caps.bus.clone();

    // List supported formats
    let formats = device.enum_formats()?;
    for fmt in formats {
        let fourcc_bytes = fmt.fourcc.repr;
        let fourcc_str = std::str::from_utf8(&fourcc_bytes).unwrap_or("????");
        metadata
            .formats
            .push(format!("{} ({})", fourcc_str, fmt.description));

        // List frame sizes for this format
        if let Ok(framesizes) = device.enum_framesizes(fmt.fourcc) {
            for size in framesizes.into_iter().take(5) {
                // Limit to first 5
                if let v4l::framesize::FrameSizeEnum::Discrete(discrete) = size.size {
                    metadata
                        .resolutions
                        .push(format!("{}x{}", discrete.width, discrete.height));
                }
            }
        }
    }

    // List camera controls
    if let Ok(controls) = device.query_controls() {
        for ctrl in controls {
            let control_info = format!(
                "{}: {} (ID: {})",
                ctrl.name,
                match ctrl.typ {
                    v4l::control::Type::Integer => "Integer",
                    v4l::control::Type::Boolean => "Boolean",
                    v4l::control::Type::Menu => "Menu",
                    v4l::control::Type::Integer64 => "Integer64",
                    _ => "Unknown",
                },
                ctrl.id
            );
            metadata.controls.push(control_info);

            // Special handling for Test Pattern
            if ctrl.name == "Test Pattern" && ctrl.typ == v4l::control::Type::Menu {
                if let Ok(fd) = std::fs::File::open(device_path) {
                    let raw_fd = fd.as_raw_fd();
                    let mut index = ctrl.minimum as u32;
                    let max = ctrl.maximum as u32;
                    while index <= max {
                        if let Some(menu_name) = query_menu_item(raw_fd, ctrl.id, index) {
                            metadata
                                .test_patterns
                                .push(format!("[{index}] {menu_name}"));
                        }
                        index += 1;
                    }
                }
            }
        }
    }

    Ok(metadata)
}

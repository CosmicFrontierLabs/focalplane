use log::{debug, info, warn};
use std::os::unix::io::AsRawFd;
use v4l::prelude::*;
use v4l::video::Capture;

use crate::camera::v4l2_utils::query_menu_item;

pub fn show_camera_info(device_path: &str) -> anyhow::Result<()> {
    let device = Device::with_path(device_path)?;

    info!("=== Camera Information ===");
    info!("Device: {device_path}");

    // Query device capabilities
    let caps = device.query_caps()?;
    info!("Driver: {}", caps.driver);
    info!("Card: {}", caps.card);
    info!("Bus: {}", caps.bus);

    // List supported formats
    info!("=== Supported Formats ===");
    let formats = device.enum_formats()?;

    for fmt in formats {
        let fourcc_bytes = fmt.fourcc.repr;
        let fourcc_str = std::str::from_utf8(&fourcc_bytes).unwrap_or("????");
        info!("Format: {} ({})", fourcc_str, fmt.description);
        debug!("  FourCC bytes: {fourcc_bytes:?}");

        // List frame sizes for this format
        if let Ok(framesizes) = device.enum_framesizes(fmt.fourcc) {
            info!("  Supported resolutions:");
            for (i, size) in framesizes.into_iter().enumerate() {
                match size.size {
                    v4l::framesize::FrameSizeEnum::Discrete(discrete) => {
                        info!("    {}. {}x{}", i + 1, discrete.width, discrete.height);

                        // Try to get frame intervals for this resolution
                        if let Ok(intervals) =
                            device.enum_frameintervals(fmt.fourcc, discrete.width, discrete.height)
                        {
                            for interval in intervals {
                                match interval.interval {
                                    v4l::frameinterval::FrameIntervalEnum::Discrete(disc) => {
                                        let fps = disc.denominator as f64 / disc.numerator as f64;
                                        info!("       - {fps:.2} fps");
                                    }
                                    v4l::frameinterval::FrameIntervalEnum::Stepwise(step) => {
                                        let min_fps =
                                            step.min.denominator as f64 / step.min.numerator as f64;
                                        let max_fps =
                                            step.max.denominator as f64 / step.max.numerator as f64;
                                        info!("       - {min_fps:.2} to {max_fps:.2} fps");
                                    }
                                }
                            }
                        }
                    }
                    v4l::framesize::FrameSizeEnum::Stepwise(stepwise) => {
                        info!(
                            "    Stepwise from {}x{} to {}x{} (step: {}x{})",
                            stepwise.min_width,
                            stepwise.min_height,
                            stepwise.max_width,
                            stepwise.max_height,
                            stepwise.step_width,
                            stepwise.step_height
                        );
                    }
                }
            }
        }
    }

    // List camera controls
    info!("=== Camera Controls ===");
    if let Ok(controls) = device.query_controls() {
        for ctrl in controls {
            info!(
                "  {}: {} (ID: {})",
                ctrl.name,
                match ctrl.typ {
                    v4l::control::Type::Integer => "Integer",
                    v4l::control::Type::Boolean => "Boolean",
                    v4l::control::Type::Menu => "Menu",
                    v4l::control::Type::Button => "Button",
                    v4l::control::Type::Integer64 => "Integer64",
                    v4l::control::Type::String => "String",
                    _ => "Unknown",
                },
                ctrl.id
            );

            // Show range for integer controls
            if matches!(
                ctrl.typ,
                v4l::control::Type::Integer | v4l::control::Type::Integer64
            ) {
                info!(
                    "    Range: {} to {} (step: {})",
                    ctrl.minimum, ctrl.maximum, ctrl.step
                );
                info!("    Default: {}", ctrl.default);
            }

            // Show menu items for menu controls
            if ctrl.typ == v4l::control::Type::Menu {
                info!("    Menu items:");
                // Open device separately for raw fd access
                if let Ok(fd) = std::fs::File::open(device_path) {
                    let raw_fd = fd.as_raw_fd();
                    let mut index = ctrl.minimum as u32;
                    let max = ctrl.maximum as u32;
                    while index <= max {
                        if let Some(menu_name) = query_menu_item(raw_fd, ctrl.id, index) {
                            info!("      [{index}] {menu_name}");
                        } else {
                            // Skip invalid indices silently
                        }
                        index += 1;
                    }
                } else {
                    warn!("    (Could not open device for menu enumeration)");
                }
            }
        }
    }

    Ok(())
}

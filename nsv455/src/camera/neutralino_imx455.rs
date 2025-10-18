use log::debug;
use v4l::prelude::*;

/// Calculate the stride (bytes per row) for the Neutralino IMX455 sensor.
/// The sensor uses specific padding patterns based on resolution.
pub fn calculate_stride(width: u32, height: u32, frame_size: usize) -> usize {
    if width == 8096 && height == 6324 {
        // Max resolution has 96 pixel padding (192 bytes)
        (width as usize + 96) * 2
    } else {
        // For other resolutions, calculate stride from actual data size
        frame_size / height as usize
    }
}

/// Calculate padding in pixels for a given resolution
pub fn calculate_padding_pixels(width: u32, height: u32, frame_size: usize) -> usize {
    let stride = calculate_stride(width, height, frame_size);
    let stride_pixels = stride / 2; // 16-bit pixels = 2 bytes each
    stride_pixels - width as usize
}

pub fn read_sensor_temperatures(device_path: &str) -> (Option<f32>, Option<f32>) {
    let mut fpga_temp = None;
    let mut pcb_temp = None;

    if let Ok(device) = Device::with_path(device_path) {
        if let Ok(controls) = device.query_controls() {
            debug!("Enumerating controls for temperature readings...");
            for ctrl in controls {
                let name_lower = ctrl.name.to_lowercase();
                debug!("Found control: '{}' (type: {:?})", ctrl.name, ctrl.typ);

                if name_lower.contains("fpga") && name_lower.contains("temperature") {
                    debug!("Matched FPGA temperature control");
                    if let Ok(control) = device.control(ctrl.id) {
                        match control.value {
                            v4l::control::Value::Integer(val) => {
                                fpga_temp = Some(val as f32 / 1000.0);
                                debug!(
                                    "FPGA temp (Integer): {} raw, {:.3} C",
                                    val,
                                    val as f32 / 1000.0
                                );
                            }
                            _ => log::warn!("FPGA temp unexpected type: {:?}", control.value),
                        }
                    }
                }

                if name_lower.contains("sensor")
                    && name_lower.contains("pcb")
                    && name_lower.contains("temperature")
                {
                    debug!("Matched PCB temperature control");
                    if let Ok(control) = device.control(ctrl.id) {
                        match control.value {
                            v4l::control::Value::Integer(val) => {
                                pcb_temp = Some(val as f32 / 1000.0);
                                debug!(
                                    "PCB temp (Integer): {} raw, {:.3} C",
                                    val,
                                    val as f32 / 1000.0
                                );
                            }
                            _ => log::warn!("PCB temp unexpected type: {:?}", control.value),
                        }
                    }
                }
            }
        }
    }

    debug!("Final temperatures - FPGA: {fpga_temp:?} C, PCB: {pcb_temp:?} C");
    (fpga_temp, pcb_temp)
}

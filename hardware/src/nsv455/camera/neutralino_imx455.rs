use log::{debug, trace};
use std::sync::OnceLock;
use v4l::prelude::*;

/// Cached temperature control IDs for efficient repeated reads.
/// These IDs are discovered once on first use and reused for all subsequent reads.
#[derive(Debug, Clone)]
struct TemperatureControlIds {
    fpga_temp_id: Option<u32>,
    pcb_temp_id: Option<u32>,
}

/// Global cache for temperature control IDs.
/// Using OnceLock since control IDs don't change during device lifetime.
static TEMP_CONTROL_CACHE: OnceLock<TemperatureControlIds> = OnceLock::new();

/// Discover temperature control IDs by scanning device controls.
/// This is only called once, with results cached globally.
fn discover_temperature_controls(device: &Device) -> TemperatureControlIds {
    let mut fpga_temp_id = None;
    let mut pcb_temp_id = None;

    if let Ok(controls) = device.query_controls() {
        debug!("Discovering temperature control IDs (one-time scan)...");
        for ctrl in controls {
            let name_lower = ctrl.name.to_lowercase();
            trace!(
                "Found control: '{}' (id: {}, type: {:?})",
                ctrl.name,
                ctrl.id,
                ctrl.typ
            );

            if name_lower.contains("fpga") && name_lower.contains("temperature") {
                debug!("Matched FPGA temperature control: id={}", ctrl.id);
                fpga_temp_id = Some(ctrl.id);
            }

            if name_lower.contains("sensor")
                && name_lower.contains("pcb")
                && name_lower.contains("temperature")
            {
                debug!("Matched PCB temperature control: id={}", ctrl.id);
                pcb_temp_id = Some(ctrl.id);
            }
        }
    }

    debug!(
        "Temperature control discovery complete: FPGA={:?}, PCB={:?}",
        fpga_temp_id, pcb_temp_id
    );
    TemperatureControlIds {
        fpga_temp_id,
        pcb_temp_id,
    }
}

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

/// Read sensor temperatures using cached control IDs.
/// On first call, discovers and caches the control IDs.
/// Subsequent calls use the cached IDs for direct reads without scanning.
pub fn read_sensor_temperatures(device_path: &str) -> (Option<f32>, Option<f32>) {
    let mut fpga_temp = None;
    let mut pcb_temp = None;

    if let Ok(device) = Device::with_path(device_path) {
        let ids = TEMP_CONTROL_CACHE.get_or_init(|| discover_temperature_controls(&device));

        if let Some(id) = ids.fpga_temp_id {
            if let Ok(control) = device.control(id) {
                match control.value {
                    v4l::control::Value::Integer(val) => {
                        fpga_temp = Some(val as f32 / 1000.0);
                        trace!("FPGA temp: {:.3} C", val as f32 / 1000.0);
                    }
                    _ => log::warn!("FPGA temp unexpected type: {:?}", control.value),
                }
            }
        }

        if let Some(id) = ids.pcb_temp_id {
            if let Ok(control) = device.control(id) {
                match control.value {
                    v4l::control::Value::Integer(val) => {
                        pcb_temp = Some(val as f32 / 1000.0);
                        trace!("PCB temp: {:.3} C", val as f32 / 1000.0);
                    }
                    _ => log::warn!("PCB temp unexpected type: {:?}", control.value),
                }
            }
        }
    }

    (fpga_temp, pcb_temp)
}

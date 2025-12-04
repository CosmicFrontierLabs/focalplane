use super::controls::{ControlMap, ControlType};
use shared::camera_interface::{CameraError, CameraResult};
use v4l::prelude::*;
use v4l::video::Capture;

#[derive(Debug, Clone)]
pub struct OffsetConstraints {
    pub min: usize,
    pub max: usize,
    pub step: usize,
}

#[derive(Debug, Clone)]
pub struct RoiConstraints {
    pub h_offset: OffsetConstraints,
    pub v_offset: OffsetConstraints,
    pub supported_sizes: Vec<(usize, usize)>,
}

impl RoiConstraints {
    pub fn from_device(device: &Device, control_map: &ControlMap) -> CameraResult<Self> {
        let h_offset_info = control_map.get(ControlType::ROIHOffset).ok_or_else(|| {
            CameraError::HardwareError("ROIHOffset control not found".to_string())
        })?;

        let v_offset_info = control_map.get(ControlType::ROIVOffset).ok_or_else(|| {
            CameraError::HardwareError("ROIVOffset control not found".to_string())
        })?;

        let control_descs = device
            .query_controls()
            .map_err(|e| CameraError::HardwareError(format!("Failed to query controls: {e}")))?;

        let mut h_offset_desc = None;
        let mut v_offset_desc = None;

        for desc in control_descs {
            if desc.id == h_offset_info.id {
                h_offset_desc = Some(desc);
            } else if desc.id == v_offset_info.id {
                v_offset_desc = Some(desc);
            }
            if h_offset_desc.is_some() && v_offset_desc.is_some() {
                break;
            }
        }

        let h_offset_desc = h_offset_desc.ok_or_else(|| {
            CameraError::HardwareError("Failed to get ROIHOffset control description".to_string())
        })?;

        let v_offset_desc = v_offset_desc.ok_or_else(|| {
            CameraError::HardwareError("Failed to get ROIVOffset control description".to_string())
        })?;

        let h_offset = OffsetConstraints {
            min: h_offset_desc.minimum as usize,
            max: h_offset_desc.maximum as usize,
            step: h_offset_desc.step as usize,
        };

        let v_offset = OffsetConstraints {
            min: v_offset_desc.minimum as usize,
            max: v_offset_desc.maximum as usize,
            step: v_offset_desc.step as usize,
        };

        let mut supported_sizes = Vec::new();

        if let Ok(formats) = device.enum_formats() {
            for format in formats {
                let fourcc_bytes = format.fourcc.repr;
                let fourcc_str = std::str::from_utf8(&fourcc_bytes).unwrap_or("????");

                // NOTE: The NSV455/IMX455 sensor reports RG16 (Bayer) format in capabilities,
                // but the actual data is always Y16 (monochrome). This is a quirk of the driver.
                if fourcc_str == "Y16 " || fourcc_str == "RG16" {
                    if let Ok(framesizes) = device.enum_framesizes(format.fourcc) {
                        for framesize in framesizes {
                            if let v4l::framesize::FrameSizeEnum::Discrete(discrete) =
                                framesize.size
                            {
                                supported_sizes
                                    .push((discrete.width as usize, discrete.height as usize));
                            }
                        }
                    }
                    break;
                }
            }
        }

        if supported_sizes.is_empty() {
            return Err(CameraError::HardwareError(
                "Failed to query supported frame sizes from device".to_string(),
            ));
        }

        Ok(Self {
            h_offset,
            v_offset,
            supported_sizes,
        })
    }
}

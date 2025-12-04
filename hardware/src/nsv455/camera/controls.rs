use shared::camera_interface::{CameraError, CameraResult};
use std::collections::HashMap;
use v4l::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlType {
    Gain,
    Exposure,
    BlackLevel,
    FrameRate,
    LowLatencyMode,
    TestPattern,
    ROIHOffset,
    ROIVOffset,
}

#[derive(Debug, Clone)]
pub struct ControlInfo {
    pub id: u32,
    pub control_type: v4l::control::Type,
    pub units: Option<&'static str>,
    pub description: &'static str,
}

impl ControlInfo {
    fn new(
        id: u32,
        control_type: v4l::control::Type,
        units: Option<&'static str>,
        description: &'static str,
    ) -> Self {
        Self {
            id,
            control_type,
            units,
            description,
        }
    }
}

pub struct ControlMap {
    controls: HashMap<ControlType, ControlInfo>,
}

impl ControlMap {
    pub fn from_device(device: &Device) -> CameraResult<Self> {
        let mut controls = HashMap::new();

        let control_descs = device
            .query_controls()
            .map_err(|e| CameraError::HardwareError(format!("Failed to query controls: {e}")))?;

        for desc in control_descs {
            let control_type = match desc.name.as_str() {
                "Gain" | "gain" => Some(ControlType::Gain),
                "Exposure" | "exposure" => Some(ControlType::Exposure),
                "Black Level" | "black_level" => Some(ControlType::BlackLevel),
                "Frame Rate" | "frame_rate" => Some(ControlType::FrameRate),
                "Low Latency Mode" => Some(ControlType::LowLatencyMode),
                "Test Pattern" => Some(ControlType::TestPattern),
                "ROI hor. start pos" => Some(ControlType::ROIHOffset),
                "ROI ver. start pos" => Some(ControlType::ROIVOffset),
                _ => None,
            };

            if let Some(ct) = control_type {
                let (units, description) = match ct {
                    ControlType::Gain => (Some("0.1dB?"), "Camera sensor gain"),
                    ControlType::Exposure => (Some("microseconds"), "Exposure time"),
                    ControlType::BlackLevel => (Some("DN"), "Black level offset"),
                    ControlType::FrameRate => (Some("??"), "Frame rate"),
                    ControlType::LowLatencyMode => (None, "Low latency streaming mode"),
                    ControlType::TestPattern => (
                        None,
                        "Test pattern mode (0=None, 1=Vertical, 2=Horizontal, 3=Gradient)",
                    ),
                    ControlType::ROIHOffset => (Some("pixels"), "ROI horizontal offset"),
                    ControlType::ROIVOffset => (Some("pixels"), "ROI vertical offset"),
                };

                controls.insert(ct, ControlInfo::new(desc.id, desc.typ, units, description));
            }
        }

        let map = Self { controls };
        map.validate_required_controls()?;
        Ok(map)
    }

    fn validate_required_controls(&self) -> CameraResult<()> {
        let required = [
            ControlType::Gain,
            ControlType::Exposure,
            ControlType::BlackLevel,
            ControlType::FrameRate,
            ControlType::LowLatencyMode,
            ControlType::TestPattern,
            ControlType::ROIHOffset,
            ControlType::ROIVOffset,
        ];

        let mut missing = Vec::new();
        for control_type in &required {
            if !self.controls.contains_key(control_type) {
                missing.push(format!("{control_type:?}"));
            }
        }

        if !missing.is_empty() {
            return Err(CameraError::HardwareError(format!(
                "Device missing required controls: {}",
                missing.join(", ")
            )));
        }

        Ok(())
    }

    pub fn get(&self, control_type: ControlType) -> Option<&ControlInfo> {
        self.controls.get(&control_type)
    }

    pub fn set_control(
        &self,
        device: &mut Device,
        control_type: ControlType,
        value: i64,
    ) -> CameraResult<()> {
        let info = self.get(control_type).ok_or_else(|| {
            CameraError::ConfigError(format!("Control {control_type:?} not found"))
        })?;

        let control_value = match info.control_type {
            v4l::control::Type::Boolean => v4l::control::Value::Boolean(value != 0),
            v4l::control::Type::Menu => v4l::control::Value::Integer(value),
            v4l::control::Type::Integer => v4l::control::Value::Integer(value),
            v4l::control::Type::Integer64 => v4l::control::Value::Integer(value),
            _ => v4l::control::Value::Integer(value),
        };

        let ctrl = v4l::Control {
            id: info.id,
            value: control_value,
        };

        device
            .set_control(ctrl)
            .map_err(|e| CameraError::ConfigError(format!("Failed to set control: {e}")))?;

        Ok(())
    }
}

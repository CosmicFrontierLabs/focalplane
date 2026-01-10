use super::controls::{ControlMap, ControlType};
use super::neutralino_imx455::read_sensor_temperatures;
use super::roi_constraints::RoiConstraints;
use ndarray::Array2;
use once_cell::sync::OnceCell;
use shared::camera_interface::{
    CameraConfig, CameraError, CameraInterface, CameraResult, FrameMetadata, SensorBitDepth,
    SensorGeometry, Timestamp,
};
use shared::image_proc::detection::aabb::AABB;
use shared::image_size::PixelShape;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;

pub struct NSV455Camera {
    device_path: String,
    config: CameraConfig,
    frame_number: Arc<AtomicU64>,
    roi: Option<AABB>,
    gain: i32,
    black_level: i32,
    framerate: u32,
    low_latency_mode: i32,
    test_pattern: i32,
    roi_constraints: RoiConstraints,
    control_map: OnceCell<ControlMap>,
}

impl NSV455Camera {
    /// IMX455 sensor fixed dimensions (from sensor.rs specifications)
    pub const SENSOR_WIDTH: u32 = 9568;
    pub const SENSOR_HEIGHT: u32 = 6380;
    pub const PIXEL_SIZE_MICRONS: f64 = 3.76;

    pub fn from_device(device_path: String) -> CameraResult<Self> {
        let device = Device::with_path(&device_path)
            .map_err(|e| CameraError::HardwareError(format!("Failed to open device: {e}")))?;

        // Query the actual format from the device to get real dimensions
        let format = device
            .format()
            .map_err(|e| CameraError::HardwareError(format!("Failed to get format: {e}")))?;

        let actual_width = format.width as usize;
        let actual_height = format.height as usize;

        tracing::info!(
            "NSV455 device reports format: {}x{} (expected IMX455: {}x{})",
            actual_width,
            actual_height,
            Self::SENSOR_WIDTH,
            Self::SENSOR_HEIGHT
        );

        let config = CameraConfig::new(
            actual_width,
            actual_height,
            Duration::from_millis(100),
            SensorBitDepth::Bits16,
            26_000.0,  // Max well depth in electrons (IMX455)
            1.0 / 0.4, // Conversion gain (DN/e-) ~2.5
        );

        let control_map = ControlMap::from_device(&device)?;
        let roi_constraints = RoiConstraints::from_device(&device, &control_map)?;

        Ok(Self {
            device_path,
            config,
            frame_number: Arc::new(AtomicU64::new(0)),
            roi: None,
            gain: 360,
            black_level: 0,
            framerate: 23_000_000,
            low_latency_mode: 1,
            test_pattern: 0,
            roi_constraints,
            control_map: OnceCell::from(control_map),
        })
    }

    fn open_device(&self) -> CameraResult<Device> {
        Device::with_path(&self.device_path)
            .map_err(|e| CameraError::HardwareError(format!("Failed to open device: {e}")))
    }

    fn configure_device(&mut self, device: &mut Device) -> CameraResult<()> {
        let control_map = self
            .control_map
            .get()
            .expect("Control map should be initialized in from_device()");

        let mut format = device
            .format()
            .map_err(|e| CameraError::HardwareError(format!("Failed to get format: {e}")))?;

        let initial_fourcc_bytes = format.fourcc.repr;
        let initial_fourcc_str = std::str::from_utf8(&initial_fourcc_bytes).unwrap_or("????");

        tracing::info!(
            "Initial format: {}x{} {} ({:?})",
            format.width,
            format.height,
            initial_fourcc_str,
            initial_fourcc_bytes
        );

        let requested_width = self
            .roi
            .as_ref()
            .map_or(self.config.size.width, |roi| roi.max_col - roi.min_col + 1)
            as u32;
        let requested_height = self
            .roi
            .as_ref()
            .map_or(self.config.size.height, |roi| roi.max_row - roi.min_row + 1)
            as u32;

        format.width = requested_width;
        format.height = requested_height;
        format.fourcc = v4l::FourCC::new(b"Y16 ");

        tracing::info!(
            "Requesting format: {}x{} Y16",
            requested_width,
            requested_height
        );

        device
            .set_format(&format)
            .map_err(|e| CameraError::ConfigError(format!("Failed to set format: {e}")))?;

        let actual_format = device
            .format()
            .map_err(|e| CameraError::HardwareError(format!("Failed to get format: {e}")))?;
        let fourcc_bytes = actual_format.fourcc.repr;
        let fourcc_str = std::str::from_utf8(&fourcc_bytes).unwrap_or("????");

        let stride = actual_format.stride;
        let expected_stride = actual_format.width * 2;

        // NOTE: The V4L2 driver reports RG16 (Bayer) format, but the actual data is Y16 (monochrome).
        // This is a quirk of the NSV455 camera/driver - the format string in the logs will show
        // RG16 but the pixel data is actually raw 16-bit monochrome. Don't worry about the mismatch.
        tracing::info!(
            "Format negotiated: {}x{} {} ({:?}), stride: {} bytes (expected: {} bytes)",
            actual_format.width,
            actual_format.height,
            fourcc_str,
            fourcc_bytes,
            stride,
            expected_stride
        );

        if stride != expected_stride {
            tracing::warn!(
                "Row padding detected: {} bytes per row ({}px × 2 = {} expected, {} padding bytes)",
                stride,
                actual_format.width,
                expected_stride,
                stride - expected_stride
            );
        }

        if actual_format.width != requested_width || actual_format.height != requested_height {
            tracing::warn!(
                "Camera did not accept requested resolution. Using {}x{} instead of {}x{}",
                actual_format.width,
                actual_format.height,
                requested_width,
                requested_height
            );
        }

        // Update stored dimensions to match actual negotiated format
        self.config.size = PixelShape::with_width_height(
            actual_format.width as usize,
            actual_format.height as usize,
        );

        control_map.set_control(device, ControlType::Gain, self.gain as i64)?;

        control_map.set_control(
            device,
            ControlType::Exposure,
            self.config.exposure.as_micros() as i64,
        )?;

        control_map.set_control(device, ControlType::BlackLevel, self.black_level as i64)?;

        control_map.set_control(device, ControlType::FrameRate, self.framerate as i64)?;

        control_map.set_control(
            device,
            ControlType::LowLatencyMode,
            self.low_latency_mode as i64,
        )?;

        control_map.set_control(device, ControlType::TestPattern, self.test_pattern as i64)?;

        if let Some(roi) = self.roi {
            control_map.set_control(device, ControlType::ROIHOffset, roi.min_col as i64)?;
            control_map.set_control(device, ControlType::ROIVOffset, roi.min_row as i64)?;
        }

        Ok(())
    }

    fn get_roi_constraints(&self) -> &RoiConstraints {
        &self.roi_constraints
    }

    pub fn roi_constraints(&self) -> &RoiConstraints {
        self.get_roi_constraints()
    }
}

impl CameraInterface for NSV455Camera {
    fn check_roi_size(&self, size: PixelShape) -> CameraResult<()> {
        let constraints = self.get_roi_constraints();

        if !constraints
            .supported_sizes
            .iter()
            .any(|(w, h)| *w == size.width && *h == size.height)
        {
            let sizes_str = constraints
                .supported_sizes
                .iter()
                .map(|(w, h)| format!("{w}x{h}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(CameraError::InvalidROI(format!(
                "ROI size {}x{} is not supported. Supported sizes: {sizes_str}",
                size.width, size.height
            )));
        }
        Ok(())
    }

    fn get_roi_offset_alignment(&self) -> (usize, usize) {
        let constraints = self.get_roi_constraints();
        (constraints.h_offset.step, constraints.v_offset.step)
    }

    fn set_roi(&mut self, roi: AABB) -> CameraResult<()> {
        let constraints = self.get_roi_constraints();
        let width = roi.max_col - roi.min_col + 1;
        let height = roi.max_row - roi.min_row + 1;

        // Check horizontal offset alignment and range
        if roi.min_col % constraints.h_offset.step != 0 {
            return Err(CameraError::InvalidROI(format!(
                "ROI horizontal offset {} must be aligned to {} pixels",
                roi.min_col, constraints.h_offset.step
            )));
        }
        if roi.min_col < constraints.h_offset.min || roi.min_col > constraints.h_offset.max {
            return Err(CameraError::InvalidROI(format!(
                "ROI horizontal offset {} out of range {}-{}",
                roi.min_col, constraints.h_offset.min, constraints.h_offset.max
            )));
        }

        // Check vertical offset alignment and range
        if roi.min_row % constraints.v_offset.step != 0 {
            return Err(CameraError::InvalidROI(format!(
                "ROI vertical offset {} must be aligned to {} pixels",
                roi.min_row, constraints.v_offset.step
            )));
        }
        if roi.min_row < constraints.v_offset.min || roi.min_row > constraints.v_offset.max {
            return Err(CameraError::InvalidROI(format!(
                "ROI vertical offset {} out of range {}-{}",
                roi.min_row, constraints.v_offset.min, constraints.v_offset.max
            )));
        }

        // Check ROI size against supported sizes
        if !constraints
            .supported_sizes
            .iter()
            .any(|(w, h)| *w == width && *h == height)
        {
            let sizes_str = constraints
                .supported_sizes
                .iter()
                .map(|(w, h)| format!("{w}x{h}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(CameraError::InvalidROI(format!(
                "ROI size {width}x{height} is not supported. Supported sizes: {sizes_str}"
            )));
        }

        self.roi = Some(roi);
        Ok(())
    }

    fn clear_roi(&mut self) -> CameraResult<()> {
        self.roi = None;
        Ok(())
    }

    fn set_exposure(&mut self, exposure: Duration) -> CameraResult<()> {
        self.config.exposure = exposure;
        Ok(())
    }

    fn get_exposure(&self) -> Duration {
        self.config.exposure
    }

    fn geometry(&self) -> SensorGeometry {
        SensorGeometry::new(
            self.config.size.width,
            self.config.size.height,
            Self::PIXEL_SIZE_MICRONS,
        )
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn get_roi(&self) -> Option<AABB> {
        self.roi
    }

    fn stream(
        &mut self,
        callback: &mut dyn FnMut(&Array2<u16>, &FrameMetadata) -> bool,
    ) -> CameraResult<()> {
        let mut device = self.open_device()?;
        self.configure_device(&mut device)?;

        let actual_format = device
            .format()
            .map_err(|e| CameraError::HardwareError(format!("Failed to get format: {e}")))?;

        let width = actual_format.width as usize;
        let height = actual_format.height as usize;

        let mut stream = MmapStream::new(&device, Type::VideoCapture)
            .map_err(|e| CameraError::HardwareError(format!("Failed to create stream: {e}")))?;

        loop {
            let (buf, meta) = stream
                .next()
                .map_err(|e| CameraError::CaptureError(format!("Failed to get frame: {e}")))?;

            let frame_data = buf.to_vec();
            // let stride =
            //     calculate_stride(actual_format.width, actual_format.height, frame_data.len());
            let stride = actual_format.stride as usize;

            // TODO: Extra memcopy in the inner loop that could be avoided with a cast.
            // We're returning a pointer to a strided 2D array, so we could potentially
            // cast directly to Array2<u16> with custom strides instead of copying pixels.
            let mut pixels = Vec::with_capacity(width * height);

            for row in 0..height {
                let row_start = row * stride;
                let row_end = row_start + (width * 2);

                if row_end <= frame_data.len() {
                    for col in 0..width {
                        let pixel_start = row_start + (col * 2);
                        if pixel_start + 1 < frame_data.len() {
                            let pixel_value = u16::from_le_bytes([
                                frame_data[pixel_start],
                                frame_data[pixel_start + 1],
                            ]);
                            pixels.push(pixel_value);
                        }
                    }
                }
            }

            if pixels.len() != width * height {
                continue;
            }

            let array = Array2::from_shape_vec((height, width), pixels)
                .map_err(|e| CameraError::CaptureError(format!("Failed to create array: {e}")))?;

            let (fpga_temp, pcb_temp) = read_sensor_temperatures(&self.device_path);

            let timestamp = Timestamp::new(
                meta.timestamp.sec as u64,
                (meta.timestamp.usec * 1000) as u64,
            );

            let frame_num = self.frame_number.fetch_add(1, Ordering::SeqCst);

            let mut temperatures = HashMap::new();
            if let Some(temp) = fpga_temp {
                temperatures.insert("fpga".to_string(), temp as f64);
            }
            if let Some(temp) = pcb_temp {
                temperatures.insert("pcb".to_string(), temp as f64);
            }

            let metadata = FrameMetadata {
                frame_number: frame_num,
                exposure: self.config.exposure,
                timestamp,
                pointing: None,
                roi: self.roi,
                temperatures,
            };

            if !callback(&array, &metadata) {
                break;
            }
        }

        Ok(())
    }

    fn saturation_value(&self) -> f64 {
        self.config.get_saturation()
    }

    fn name(&self) -> &str {
        "NSV455"
    }

    fn get_bit_depth(&self) -> SensorBitDepth {
        self.config.bit_depth
    }

    fn set_bit_depth(&mut self, bit_depth: SensorBitDepth) -> CameraResult<()> {
        self.config.bit_depth = bit_depth;
        Ok(())
    }

    fn get_gain(&self) -> f64 {
        self.gain as f64
    }

    fn set_gain(&mut self, gain: f64) -> CameraResult<()> {
        self.gain = gain.round() as i32;
        Ok(())
    }

    fn get_serial(&self) -> String {
        // TODO: Get proper serial from Neutralino - awaiting their guidance on how to query it
        // V4L2 doesn't provide a standard serial, and device path is not camera-specific
        tracing::warn!("NSV455 serial number is hardcoded - awaiting Neutralino guidance");
        "NSV455_UNKNOWN".to_string()
    }
}

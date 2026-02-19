use super::controls::{ControlMap, ControlType, TestPattern};
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

/// Error returned when a frame buffer is too small to unpack.
#[derive(Debug, thiserror::Error)]
#[error(
    "frame buffer too small: need {needed_bytes} bytes for {width}x{height} frame \
     (stride={stride}), got {actual_bytes}"
)]
pub struct UnpackFrameError {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub needed_bytes: usize,
    pub actual_bytes: usize,
}

/// Convert a raw V4L2 frame buffer of little-endian `u16` pixels into a flat `Vec<u16>`.
///
/// Handles stride padding: when `stride > width * 2`, the extra bytes at the
/// end of each row are skipped.
pub fn unpack_u16_frame(
    frame_data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Vec<u16>, UnpackFrameError> {
    let row_bytes = width * 2;
    let needed = if stride == row_bytes {
        height * row_bytes
    } else {
        (height - 1) * stride + row_bytes
    };

    if frame_data.len() < needed {
        return Err(UnpackFrameError {
            width,
            height,
            stride,
            needed_bytes: needed,
            actual_bytes: frame_data.len(),
        });
    }

    // Fast path: no row padding, so the pixel data is contiguous and we can
    // reinterpret the entire byte slice as u16 in one shot.
    if stride == row_bytes {
        return Ok(bytemuck::cast_slice::<u8, u16>(&frame_data[..needed]).to_vec());
    }

    // Slow path: stride includes padding bytes at the end of each row (e.g.
    // alignment to a cache line or DMA boundary). Copy pixel data row-by-row,
    // skipping the padding between rows.
    let mut pixels = Vec::with_capacity(width * height);
    for row in 0..height {
        let row_start = row * stride;
        let row_u16 =
            bytemuck::cast_slice::<u8, u16>(&frame_data[row_start..row_start + row_bytes]);
        pixels.extend_from_slice(row_u16);
    }
    Ok(pixels)
}

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

    /// Empirical vertical offset correction for ROI placement.
    ///
    /// The NSV455 negotiates a 9576x6388 readout (8 pixels larger than the
    /// 9568x6380 active area in each dimension). ROI V-offset controls appear
    /// to be indexed relative to the active area origin, not the full readout.
    /// Star coordinates detected in the full frame include the 8-row border,
    /// so we add 8 to the V-offset to align the ROI with full-frame coordinates.
    ///
    /// Measured via random-pixel tracking: without correction the centroid
    /// lands ~8px below expected position in the ROI.
    const ROI_V_OFFSET_CORRECTION: usize = 8;

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
        let configure_start = std::time::Instant::now();
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

        // When ROI is None, request full sensor dimensions (not config.size which may be stale)
        let requested_width = self.roi.as_ref().map_or(Self::SENSOR_WIDTH, |roi| {
            (roi.max_col - roi.min_col + 1) as u32
        });
        let requested_height = self.roi.as_ref().map_or(Self::SENSOR_HEIGHT, |roi| {
            (roi.max_row - roi.min_row + 1) as u32
        });

        format.width = requested_width;
        format.height = requested_height;
        format.fourcc = v4l::FourCC::new(b"Y16 ");

        tracing::info!(
            "Requesting format: {}x{} Y16",
            requested_width,
            requested_height
        );

        let set_format_start = std::time::Instant::now();
        device
            .set_format(&format)
            .map_err(|e| CameraError::ConfigError(format!("Failed to set format: {e}")))?;
        tracing::info!(
            "set_format took {:.1}ms",
            set_format_start.elapsed().as_secs_f64() * 1000.0
        );

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

        let controls_start = std::time::Instant::now();
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
            let roi_start = std::time::Instant::now();
            let corrected_v_offset = roi.min_row + Self::ROI_V_OFFSET_CORRECTION;
            control_map.set_control(device, ControlType::ROIHOffset, roi.min_col as i64)?;
            control_map.set_control(device, ControlType::ROIVOffset, corrected_v_offset as i64)?;
            tracing::info!(
                "ROI offset controls took {:.1}ms (v_offset: {} + {} = {})",
                roi_start.elapsed().as_secs_f64() * 1000.0,
                roi.min_row,
                Self::ROI_V_OFFSET_CORRECTION,
                corrected_v_offset
            );
        }
        tracing::info!(
            "All controls took {:.1}ms",
            controls_start.elapsed().as_secs_f64() * 1000.0
        );

        tracing::info!(
            "configure_device total: {:.1}ms",
            configure_start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(())
    }

    fn get_roi_constraints(&self) -> &RoiConstraints {
        &self.roi_constraints
    }

    pub fn roi_constraints(&self) -> &RoiConstraints {
        self.get_roi_constraints()
    }

    /// Set the test pattern mode for the sensor.
    ///
    /// Test patterns are useful for validating the camera pipeline without
    /// requiring an optical input. The pattern is generated by the sensor itself.
    pub fn set_test_pattern(&mut self, pattern: TestPattern) {
        self.test_pattern = pattern.as_i32();
    }

    /// Get the current test pattern mode.
    pub fn get_test_pattern(&self) -> TestPattern {
        TestPattern::from_i32(self.test_pattern).unwrap_or_default()
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

        // Check vertical offset alignment and range (including V-offset correction)
        if roi.min_row % constraints.v_offset.step != 0 {
            return Err(CameraError::InvalidROI(format!(
                "ROI vertical offset {} must be aligned to {} pixels",
                roi.min_row, constraints.v_offset.step
            )));
        }
        let corrected_v = roi.min_row + Self::ROI_V_OFFSET_CORRECTION;
        if corrected_v > constraints.v_offset.max {
            return Err(CameraError::InvalidROI(format!(
                "ROI vertical offset {} + {} correction = {} exceeds max {}",
                roi.min_row,
                Self::ROI_V_OFFSET_CORRECTION,
                corrected_v,
                constraints.v_offset.max
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

    fn get_roi_offset_alignment(&self) -> (usize, usize) {
        let constraints = self.get_roi_constraints();
        (constraints.h_offset.step, constraints.v_offset.step)
    }

    fn stream(
        &mut self,
        callback: &mut dyn FnMut(&Array2<u16>, &FrameMetadata) -> bool,
    ) -> CameraResult<()> {
        let stream_setup_start = std::time::Instant::now();

        let open_start = std::time::Instant::now();
        let mut device = self.open_device()?;
        tracing::info!(
            "open_device took {:.1}ms",
            open_start.elapsed().as_secs_f64() * 1000.0
        );

        self.configure_device(&mut device)?;

        let actual_format = device
            .format()
            .map_err(|e| CameraError::HardwareError(format!("Failed to get format: {e}")))?;

        let width = actual_format.width as usize;
        let height = actual_format.height as usize;

        let mmap_start = std::time::Instant::now();
        let mut stream = MmapStream::new(&device, Type::VideoCapture)
            .map_err(|e| CameraError::HardwareError(format!("Failed to create stream: {e}")))?;
        tracing::info!(
            "MmapStream::new took {:.1}ms",
            mmap_start.elapsed().as_secs_f64() * 1000.0
        );

        tracing::info!(
            "Total stream setup: {:.1}ms (before first frame)",
            stream_setup_start.elapsed().as_secs_f64() * 1000.0
        );

        loop {
            let (buf, meta) = stream
                .next()
                .map_err(|e| CameraError::CaptureError(format!("Failed to get frame: {e}")))?;

            let stride = actual_format.stride as usize;
            let pixels = match unpack_u16_frame(&buf, width, height, stride) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("{e}");
                    continue;
                }
            };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_no_padding() {
        // 3x2 image, stride == width*2 (no padding)
        let pixels: Vec<u16> = vec![1, 2, 3, 4, 5, 6];
        let bytes: Vec<u8> = pixels.iter().flat_map(|p| p.to_le_bytes()).collect();
        let result = unpack_u16_frame(&bytes, 3, 2, 6).unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn unpack_with_stride_padding() {
        // 3x2 image with stride=8 (2 padding bytes per row)
        let row0: Vec<u16> = vec![10, 20, 30];
        let row1: Vec<u16> = vec![40, 50, 60];
        let mut bytes = Vec::new();
        for p in &row0 {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        bytes.extend_from_slice(&[0xAA, 0xBB]); // padding
        for p in &row1 {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        bytes.extend_from_slice(&[0xCC, 0xDD]); // padding

        let result = unpack_u16_frame(&bytes, 3, 2, 8).unwrap();
        assert_eq!(result, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn unpack_buffer_too_small() {
        let bytes = vec![0u8; 4]; // only 2 pixels worth of data
        let err = unpack_u16_frame(&bytes, 3, 2, 6).unwrap_err();
        assert_eq!(err.width, 3);
        assert_eq!(err.height, 2);
        assert_eq!(err.stride, 6);
        assert_eq!(err.needed_bytes, 12);
        assert_eq!(err.actual_bytes, 4);
    }

    #[test]
    fn unpack_single_pixel() {
        let bytes = 42u16.to_le_bytes().to_vec();
        let result = unpack_u16_frame(&bytes, 1, 1, 2).unwrap();
        assert_eq!(result, vec![42]);
    }

    #[test]
    fn unpack_large_stride_buffer_too_small() {
        // stride=16, but buffer only has enough for no-padding layout
        let bytes = vec![0u8; 12]; // 3*2*2 = 12, but need (1*16 + 6) = 22
        let err = unpack_u16_frame(&bytes, 3, 2, 16).unwrap_err();
        assert_eq!(err.needed_bytes, 22);
        assert_eq!(err.actual_bytes, 12);
    }

    #[test]
    fn unpack_error_message_is_descriptive() {
        let err = unpack_u16_frame(&[0u8; 4], 100, 200, 200).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("100x200"), "missing dimensions: {msg}");
        assert!(msg.contains("stride=200"), "missing stride: {msg}");
        assert!(msg.contains("got 4"), "missing actual size: {msg}");
    }
}

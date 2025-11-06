use crate::camera::neutralino_imx455::{calculate_stride, read_sensor_temperatures};
use crate::v4l2_capture::{CameraConfig as V4L2Config, V4L2Capture};
use ndarray::Array2;
use shared::camera_interface::{
    CameraConfig, CameraError, CameraInterface, CameraResult, FrameMetadata, SensorGeometry,
    Timestamp,
};
use shared::image_proc::detection::aabb::AABB;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;

pub struct NSV455Camera {
    device_path: String,
    v4l2_config: V4L2Config,
    config: CameraConfig,
    frame_number: Arc<AtomicU64>,
    roi: Option<AABB>,
}

impl NSV455Camera {
    /// IMX455 sensor fixed dimensions (from sensor.rs specifications)
    pub const SENSOR_WIDTH: u32 = 9568;
    pub const SENSOR_HEIGHT: u32 = 6380;
    pub const PIXEL_SIZE_MICRONS: f64 = 3.76;

    pub fn new(device_path: String) -> CameraResult<Self> {
        let v4l2_config = V4L2Config {
            device_path: device_path.clone(),
            width: Self::SENSOR_WIDTH,
            height: Self::SENSOR_HEIGHT,
            framerate: 23_000_000,
            gain: 360,
            exposure: 140,
            black_level: 4095,
        };

        let config = CameraConfig {
            width: Self::SENSOR_WIDTH as usize,
            height: Self::SENSOR_HEIGHT as usize,
            exposure: Duration::from_millis(100),
            bit_depth: 16,
        };

        Ok(Self {
            device_path,
            v4l2_config,
            config,
            frame_number: Arc::new(AtomicU64::new(0)),
            roi: None,
        })
    }
}

impl CameraInterface for NSV455Camera {
    fn check_roi_size(&self, _width: usize, _height: usize) -> CameraResult<()> {
        Ok(())
    }

    fn set_roi(&mut self, roi: AABB) -> CameraResult<()> {
        if roi.max_col >= self.config.width || roi.max_row >= self.config.height {
            return Err(CameraError::InvalidROI(
                "ROI extends beyond sensor bounds".to_string(),
            ));
        }

        let roi_width = (roi.max_col - roi.min_col + 1) as u32;
        let roi_height = (roi.max_row - roi.min_row + 1) as u32;

        self.v4l2_config.width = roi_width;
        self.v4l2_config.height = roi_height;
        self.roi = Some(roi);

        Ok(())
    }

    fn clear_roi(&mut self) -> CameraResult<()> {
        self.v4l2_config.width = self.config.width as u32;
        self.v4l2_config.height = self.config.height as u32;
        self.roi = None;
        Ok(())
    }

    fn set_exposure(&mut self, exposure: Duration) -> CameraResult<()> {
        self.config.exposure = exposure;
        self.v4l2_config.exposure = exposure.as_micros() as i32;
        Ok(())
    }

    fn get_exposure(&self) -> Duration {
        self.config.exposure
    }

    fn geometry(&self) -> SensorGeometry {
        SensorGeometry {
            width: self.config.width,
            height: self.config.height,
            pixel_size_microns: Self::PIXEL_SIZE_MICRONS,
        }
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
        let capture = V4L2Capture::new(self.v4l2_config.clone()).map_err(|e| {
            CameraError::HardwareError(format!("Failed to create V4L2 capture: {e}"))
        })?;

        let mut device = capture
            .open_device()
            .map_err(|e| CameraError::HardwareError(format!("Failed to open device: {e}")))?;

        capture
            .configure_device(&mut device)
            .map_err(|e| CameraError::ConfigError(format!("Failed to configure device: {e}")))?;

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
            let stride =
                calculate_stride(actual_format.width, actual_format.height, frame_data.len());

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
        65535.0
    }

    fn name(&self) -> &str {
        "NSV455"
    }

    fn get_bit_depth(&self) -> u8 {
        self.config.bit_depth
    }

    fn set_bit_depth(&mut self, bit_depth: u8) -> CameraResult<()> {
        self.config.bit_depth = bit_depth;
        Ok(())
    }

    fn get_gain(&self) -> f64 {
        self.v4l2_config.gain as f64
    }

    fn set_gain(&mut self, gain: f64) -> CameraResult<()> {
        self.v4l2_config.gain = gain.round() as i32;
        Ok(())
    }

    fn get_serial(&self) -> String {
        self.device_path.clone()
    }
}

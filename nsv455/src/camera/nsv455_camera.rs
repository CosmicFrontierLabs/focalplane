use crate::camera::neutralino_imx455::{
    calculate_padding_pixels, calculate_stride, read_sensor_temperatures,
};
use crate::v4l2_capture::{CameraConfig as V4L2Config, V4L2Capture};
use ndarray::Array2;
use shared::camera_interface::{
    CameraConfig, CameraError, CameraInterface, CameraResult, FrameMetadata, Timestamp,
};
use shared::image_proc::detection::aabb::AABB;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;

type FrameBuffer = Arc<Mutex<Option<(Array2<u16>, FrameMetadata)>>>;

pub struct NSV455Camera {
    device_path: String,
    v4l2_config: V4L2Config,
    config: CameraConfig,
    frame_number: Arc<AtomicU64>,
    is_capturing: Arc<AtomicBool>,
    latest_frame: FrameBuffer,
    capture_thread: Option<std::thread::JoinHandle<()>>,
    roi: Option<AABB>,
}

impl NSV455Camera {
    pub fn new(device_path: String, width: u32, height: u32) -> CameraResult<Self> {
        let v4l2_config = V4L2Config {
            device_path: device_path.clone(),
            width,
            height,
            framerate: 23_000_000,
            gain: 360,
            exposure: 140,
            black_level: 4095,
        };

        let config = CameraConfig {
            width: width as usize,
            height: height as usize,
            exposure: Duration::from_millis(100),
            bit_depth: 16,
        };

        Ok(Self {
            device_path,
            v4l2_config,
            config,
            frame_number: Arc::new(AtomicU64::new(0)),
            is_capturing: Arc::new(AtomicBool::new(false)),
            latest_frame: Arc::new(Mutex::new(None)),
            capture_thread: None,
            roi: None,
        })
    }

    fn capture_internal(&mut self) -> CameraResult<(Array2<u16>, FrameMetadata)> {
        let capture = V4L2Capture::new(self.v4l2_config.clone()).map_err(|e| {
            CameraError::HardwareError(format!("Failed to create V4L2Capture: {e}"))
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

        let mut stream = MmapStream::new(&device, Type::VideoCapture)
            .map_err(|e| CameraError::CaptureError(format!("Failed to create stream: {e}")))?;

        let (buf, meta) = stream
            .next()
            .map_err(|e| CameraError::CaptureError(format!("Failed to capture frame: {e}")))?;

        let frame_data = buf.to_vec();
        let width = actual_format.width as usize;
        let height = actual_format.height as usize;
        let stride = calculate_stride(actual_format.width, actual_format.height, frame_data.len());
        let padding_pixels =
            calculate_padding_pixels(actual_format.width, actual_format.height, frame_data.len());

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
            return Err(CameraError::CaptureError(format!(
                "Pixel count mismatch: expected {}, got {}. Stride: {}, padding: {} pixels",
                width * height,
                pixels.len(),
                stride,
                padding_pixels
            )));
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

        Ok((array, metadata))
    }
}

impl CameraInterface for NSV455Camera {
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

    fn capture_frame(&mut self) -> CameraResult<(Array2<u16>, FrameMetadata)> {
        self.capture_internal()
    }

    fn set_exposure(&mut self, exposure: Duration) -> CameraResult<()> {
        self.config.exposure = exposure;
        self.v4l2_config.exposure = exposure.as_micros() as i32;
        Ok(())
    }

    fn get_exposure(&self) -> Duration {
        self.config.exposure
    }

    fn get_config(&self) -> &CameraConfig {
        &self.config
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn get_roi(&self) -> Option<AABB> {
        self.roi
    }

    fn start_continuous_capture(&mut self) -> CameraResult<()> {
        if self.is_capturing.load(Ordering::SeqCst) {
            return Ok(());
        }

        self.is_capturing.store(true, Ordering::SeqCst);

        let v4l2_config = self.v4l2_config.clone();
        let device_path = self.device_path.clone();
        let config = self.config.clone();
        let is_capturing = Arc::clone(&self.is_capturing);
        let frame_number = Arc::clone(&self.frame_number);
        let latest_frame = Arc::clone(&self.latest_frame);
        let roi = self.roi;

        let handle = std::thread::spawn(move || {
            let capture = match V4L2Capture::new(v4l2_config.clone()) {
                Ok(cap) => cap,
                Err(_) => return,
            };

            let mut device = match capture.open_device() {
                Ok(dev) => dev,
                Err(_) => return,
            };

            if capture.configure_device(&mut device).is_err() {
                return;
            }

            let actual_format = match device.format() {
                Ok(fmt) => fmt,
                Err(_) => return,
            };

            let width = actual_format.width as usize;
            let height = actual_format.height as usize;

            let mut stream = match MmapStream::new(&device, Type::VideoCapture) {
                Ok(s) => s,
                Err(_) => return,
            };

            while is_capturing.load(Ordering::SeqCst) {
                let (buf, meta) = match stream.next() {
                    Ok(frame) => frame,
                    Err(_) => break,
                };

                let frame_data = buf.to_vec();
                let stride =
                    calculate_stride(actual_format.width, actual_format.height, frame_data.len());

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

                let array = match Array2::from_shape_vec((height, width), pixels) {
                    Ok(arr) => arr,
                    Err(_) => continue,
                };

                let (fpga_temp, pcb_temp) = read_sensor_temperatures(&device_path);

                let timestamp = Timestamp::new(
                    meta.timestamp.sec as u64,
                    (meta.timestamp.usec * 1000) as u64,
                );

                let frame_num = frame_number.fetch_add(1, Ordering::SeqCst);

                let mut temperatures = HashMap::new();
                if let Some(temp) = fpga_temp {
                    temperatures.insert("fpga".to_string(), temp as f64);
                }
                if let Some(temp) = pcb_temp {
                    temperatures.insert("pcb".to_string(), temp as f64);
                }

                let metadata = FrameMetadata {
                    frame_number: frame_num,
                    exposure: config.exposure,
                    timestamp,
                    pointing: None,
                    roi,
                    temperatures,
                };

                if let Ok(mut frame) = latest_frame.lock() {
                    *frame = Some((array, metadata));
                }
            }
        });

        self.capture_thread = Some(handle);
        Ok(())
    }

    fn stop_continuous_capture(&mut self) -> CameraResult<()> {
        self.is_capturing.store(false, Ordering::SeqCst);
        if let Some(handle) = self.capture_thread.take() {
            let _ = handle.join();
        }
        Ok(())
    }

    fn get_latest_frame(&mut self) -> Option<(Array2<u16>, FrameMetadata)> {
        self.latest_frame.lock().ok()?.take()
    }

    fn is_capturing(&self) -> bool {
        self.is_capturing.load(Ordering::SeqCst)
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

impl Drop for NSV455Camera {
    fn drop(&mut self) {
        let _ = self.stop_continuous_capture();
    }
}

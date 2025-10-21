use ndarray::Array2;
use playerone_sdk::{Camera, CameraDescription, ImageFormat};
use shared::camera_interface::{
    CameraConfig, CameraError, CameraInterface, CameraResult, FrameMetadata, Timestamp,
};
use shared::image_proc::detection::aabb::AABB;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type FrameBuffer = Arc<Mutex<Option<(Array2<u16>, FrameMetadata)>>>;

pub struct PlayerOneCamera {
    camera: Arc<Mutex<Camera>>,
    config: CameraConfig,
    exposure: Duration,
    roi: Option<AABB>,
    frame_number: Arc<AtomicU64>,
    is_capturing: Arc<AtomicBool>,
    latest_frame: FrameBuffer,
    capture_thread: Option<std::thread::JoinHandle<()>>,
    name: String,
}

impl PlayerOneCamera {
    pub fn new(camera_id: i32) -> CameraResult<Self> {
        let cameras = Camera::all_cameras();
        let descriptor = cameras
            .into_iter()
            .find(|desc| desc.camera_id() == camera_id)
            .ok_or_else(|| {
                CameraError::HardwareError(format!("Camera with ID {camera_id} not found"))
            })?;

        Self::from_descriptor(descriptor)
    }

    pub fn from_descriptor(descriptor: CameraDescription) -> CameraResult<Self> {
        let max_width = descriptor.properties().max_width as usize;
        let max_height = descriptor.properties().max_height as usize;
        let camera_name = descriptor.properties().camera_model_name.clone();

        let mut camera = descriptor
            .open()
            .map_err(|e| CameraError::HardwareError(format!("Failed to open camera: {e}")))?;

        camera
            .set_image_format(ImageFormat::RAW16)
            .map_err(|e| CameraError::ConfigError(format!("Failed to set RAW16 format: {e}")))?;

        let config = CameraConfig {
            width: max_width,
            height: max_height,
            exposure: Duration::from_millis(100),
        };

        Ok(Self {
            camera: Arc::new(Mutex::new(camera)),
            config,
            exposure: Duration::from_millis(100),
            roi: None,
            frame_number: Arc::new(AtomicU64::new(0)),
            is_capturing: Arc::new(AtomicBool::new(false)),
            latest_frame: Arc::new(Mutex::new(None)),
            capture_thread: None,
            name: camera_name,
        })
    }

    fn capture_internal(&mut self) -> CameraResult<(Array2<u16>, FrameMetadata)> {
        let exposure_us = self
            .exposure
            .as_micros()
            .try_into()
            .map_err(|_| CameraError::ConfigError("Exposure time too large".to_string()))?;

        let mut camera = self
            .camera
            .lock()
            .map_err(|_| CameraError::HardwareError("Camera mutex poisoned".to_string()))?;

        camera
            .set_exposure(exposure_us, false)
            .map_err(|e| CameraError::ConfigError(format!("Failed to set exposure: {e}")))?;

        let (width, height) = if let Some(roi) = &self.roi {
            let roi_width = (roi.max_col - roi.min_col + 1) as u32;
            let roi_height = (roi.max_row - roi.min_row + 1) as u32;

            camera
                .set_image_start_pos(roi.min_col as u32, roi.min_row as u32)
                .map_err(|e| CameraError::ConfigError(format!("Failed to set ROI start: {e}")))?;

            camera
                .set_image_size(roi_width, roi_height)
                .map_err(|e| CameraError::ConfigError(format!("Failed to set ROI size: {e}")))?;

            (roi_width as usize, roi_height as usize)
        } else {
            (self.config.width, self.config.height)
        };

        let mut buffer = camera.create_image_buffer();

        camera
            .capture(&mut buffer, Some(5000))
            .map_err(|e| CameraError::CaptureError(format!("Capture failed: {e}")))?;

        let u16_data: Vec<u16> = buffer
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        let array = Array2::from_shape_vec((height, width), u16_data)
            .map_err(|e| CameraError::CaptureError(format!("Failed to create array: {e}")))?;

        let temperature = camera
            .temperature()
            .map_err(|e| CameraError::HardwareError(format!("Failed to read temperature: {e}")))?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0));
        let timestamp = Timestamp::from_duration(now);

        let frame_num = self.frame_number.fetch_add(1, Ordering::SeqCst);

        let mut temperatures = HashMap::new();
        temperatures.insert("sensor".to_string(), temperature);

        let metadata = FrameMetadata {
            frame_number: frame_num,
            exposure: self.exposure,
            timestamp,
            pointing: None,
            roi: self.roi,
            temperatures,
        };

        Ok((array, metadata))
    }
}

impl CameraInterface for PlayerOneCamera {
    fn set_roi(&mut self, roi: AABB) -> CameraResult<()> {
        let roi_width = (roi.max_col - roi.min_col + 1) as u32;
        let roi_height = (roi.max_row - roi.min_row + 1) as u32;

        if roi_width % 4 != 0 {
            return Err(CameraError::InvalidROI(
                "ROI width must be divisible by 4".to_string(),
            ));
        }
        if roi_height % 2 != 0 {
            return Err(CameraError::InvalidROI(
                "ROI height must be divisible by 2".to_string(),
            ));
        }

        if roi.max_col >= self.config.width || roi.max_row >= self.config.height {
            return Err(CameraError::InvalidROI(
                "ROI extends beyond sensor bounds".to_string(),
            ));
        }

        self.roi = Some(roi);
        Ok(())
    }

    fn clear_roi(&mut self) -> CameraResult<()> {
        self.roi = None;
        let mut camera = self
            .camera
            .lock()
            .map_err(|_| CameraError::HardwareError("Camera mutex poisoned".to_string()))?;
        camera
            .set_image_size(self.config.width as u32, self.config.height as u32)
            .map_err(|e| CameraError::ConfigError(format!("Failed to clear ROI: {e}")))?;
        Ok(())
    }

    fn capture_frame(&mut self) -> CameraResult<(Array2<u16>, FrameMetadata)> {
        self.capture_internal()
    }

    fn set_exposure(&mut self, exposure: Duration) -> CameraResult<()> {
        self.exposure = exposure;
        Ok(())
    }

    fn get_exposure(&self) -> Duration {
        self.exposure
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

        let camera = Arc::clone(&self.camera);
        let exposure = self.exposure;
        let roi = self.roi;
        let config = self.config.clone();
        let is_capturing = Arc::clone(&self.is_capturing);
        let frame_number = Arc::clone(&self.frame_number);
        let latest_frame = Arc::clone(&self.latest_frame);

        let handle = std::thread::spawn(move || {
            while is_capturing.load(Ordering::SeqCst) {
                let exposure_us: i64 = match exposure.as_micros().try_into() {
                    Ok(v) => v,
                    Err(_) => break,
                };

                let mut camera = match camera.lock() {
                    Ok(cam) => cam,
                    Err(_) => break,
                };

                if camera.set_exposure(exposure_us, false).is_err() {
                    break;
                }

                let (width, height) = if let Some(roi) = roi {
                    let roi_width = (roi.max_col - roi.min_col + 1) as u32;
                    let roi_height = (roi.max_row - roi.min_row + 1) as u32;

                    if camera
                        .set_image_start_pos(roi.min_col as u32, roi.min_row as u32)
                        .is_err()
                    {
                        break;
                    }
                    if camera.set_image_size(roi_width, roi_height).is_err() {
                        break;
                    }

                    (roi_width as usize, roi_height as usize)
                } else {
                    (config.width, config.height)
                };

                let mut buffer = camera.create_image_buffer();

                if camera.capture(&mut buffer, Some(5000)).is_err() {
                    break;
                }

                let u16_data: Vec<u16> = buffer
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();

                if let Ok(array) = Array2::from_shape_vec((height, width), u16_data) {
                    let temperature = camera.temperature().unwrap_or(0.0);

                    drop(camera);

                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or(Duration::from_secs(0));
                    let timestamp = Timestamp::from_duration(now);

                    let frame_num = frame_number.fetch_add(1, Ordering::SeqCst);

                    let mut temperatures = HashMap::new();
                    temperatures.insert("sensor".to_string(), temperature);

                    let metadata = FrameMetadata {
                        frame_number: frame_num,
                        exposure,
                        timestamp,
                        pointing: None,
                        roi,
                        temperatures,
                    };

                    if let Ok(mut frame) = latest_frame.lock() {
                        *frame = Some((array, metadata));
                    }
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
        &self.name
    }
}

impl Drop for PlayerOneCamera {
    fn drop(&mut self) {
        let _ = self.stop_continuous_capture();
    }
}

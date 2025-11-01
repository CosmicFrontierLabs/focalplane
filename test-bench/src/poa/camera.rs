use ndarray::Array2;
use playerone_sdk::{Camera, CameraDescription, ImageFormat, ROI};
use shared::camera_interface::{
    CameraConfig, CameraError, CameraInterface, CameraResult, FrameMetadata, Timestamp,
};
use shared::image_proc::detection::aabb::AABB;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn aabb_to_roi(aabb: &AABB) -> ROI {
    let width = (aabb.max_col - aabb.min_col + 1) as u32;
    let height = (aabb.max_row - aabb.min_row + 1) as u32;
    ROI {
        start_x: aabb.min_col as u32,
        start_y: aabb.min_row as u32,
        width,
        height,
    }
}

pub struct PlayerOneCamera {
    camera: Arc<Mutex<Camera>>,
    config: CameraConfig,
    frame_number: Arc<AtomicU64>,
    name: String,
    serial_number: String,
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
        let serial_number = descriptor.properties().serial_number.clone();

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
            bit_depth: 16,
        };

        Ok(Self {
            camera: Arc::new(Mutex::new(camera)),
            config,
            frame_number: Arc::new(AtomicU64::new(0)),
            name: camera_name,
            serial_number,
        })
    }
}

impl CameraInterface for PlayerOneCamera {
    fn check_roi_size(&self, width: usize, height: usize) -> CameraResult<()> {
        if width % 4 != 0 {
            return Err(CameraError::InvalidROI(
                "ROI width must be divisible by 4".to_string(),
            ));
        }
        if height % 2 != 0 {
            return Err(CameraError::InvalidROI(
                "ROI height must be divisible by 2".to_string(),
            ));
        }
        Ok(())
    }

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

        let poa_roi = aabb_to_roi(&roi);
        let mut camera = self
            .camera
            .lock()
            .map_err(|_| CameraError::HardwareError("Camera mutex poisoned".to_string()))?;

        let current_hw_roi = camera.roi();
        if current_hw_roi.start_x != poa_roi.start_x
            || current_hw_roi.start_y != poa_roi.start_y
            || current_hw_roi.width != poa_roi.width
            || current_hw_roi.height != poa_roi.height
        {
            let roi_set_start = std::time::Instant::now();
            camera
                .set_roi(&poa_roi)
                .map_err(|e| CameraError::ConfigError(format!("Failed to set ROI: {e}")))?;
            tracing::debug!(
                "ROI set took {:.2}ms",
                roi_set_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }

    fn clear_roi(&mut self) -> CameraResult<()> {
        let mut camera = self
            .camera
            .lock()
            .map_err(|_| CameraError::HardwareError("Camera mutex poisoned".to_string()))?;
        camera
            .set_image_size(self.config.width as u32, self.config.height as u32)
            .map_err(|e| CameraError::ConfigError(format!("Failed to clear ROI: {e}")))?;
        Ok(())
    }

    fn set_exposure(&mut self, exposure: Duration) -> CameraResult<()> {
        let mut camera = self
            .camera
            .lock()
            .map_err(|_| CameraError::HardwareError("Camera mutex poisoned".to_string()))?;

        camera
            .set_exposure(
                exposure
                    .as_micros()
                    .try_into()
                    .map_err(|_| CameraError::ConfigError("Exposure time too large".to_string()))?,
                false,
            )
            .map_err(|e| CameraError::ConfigError(format!("Failed to set exposure: {e}")))
    }

    fn get_exposure(&self) -> Duration {
        self.camera
            .lock()
            .ok()
            .and_then(|camera| camera.exposure().ok())
            .map(|(exposure_us, _auto)| Duration::from_micros(exposure_us as u64))
            .unwrap_or(Duration::from_millis(100))
    }

    fn get_config(&self) -> &CameraConfig {
        &self.config
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn get_roi(&self) -> Option<AABB> {
        let camera = self.camera.lock().ok()?;
        let hw_roi = camera.roi();

        if hw_roi.width == self.config.width as u32
            && hw_roi.height == self.config.height as u32
            && hw_roi.start_x == 0
            && hw_roi.start_y == 0
        {
            None
        } else {
            Some(AABB {
                min_col: hw_roi.start_x as usize,
                min_row: hw_roi.start_y as usize,
                max_col: (hw_roi.start_x + hw_roi.width - 1) as usize,
                max_row: (hw_roi.start_y + hw_roi.height - 1) as usize,
            })
        }
    }

    fn stream(
        &mut self,
        callback: &mut dyn FnMut(&Array2<u16>, &FrameMetadata) -> bool,
    ) -> CameraResult<()> {
        let mut camera = self
            .camera
            .lock()
            .map_err(|_| CameraError::HardwareError("Camera mutex poisoned".to_string()))?;

        let config = self.config.clone();
        let frame_number = Arc::clone(&self.frame_number);

        let timeout_ms = Some(((config.exposure.as_millis() * 3) as u32).max(1000));

        camera
            .stream(timeout_ms, |cam, buffer| {
                let hw_roi = cam.roi();
                let (width, height) = (hw_roi.width as usize, hw_roi.height as usize);

                let (exposure_us, _auto) = match cam.exposure() {
                    Ok(exp) => exp,
                    Err(_) => return false,
                };

                let u16_data: Vec<u16> = buffer
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();

                let array = match Array2::from_shape_vec((height, width), u16_data) {
                    Ok(arr) => arr,
                    Err(_) => return false,
                };

                let temperature = cam.temperature().unwrap_or(f64::NAN);

                let roi_aabb = if hw_roi.width == config.width as u32
                    && hw_roi.height == config.height as u32
                    && hw_roi.start_x == 0
                    && hw_roi.start_y == 0
                {
                    None
                } else {
                    Some(AABB {
                        min_col: hw_roi.start_x as usize,
                        min_row: hw_roi.start_y as usize,
                        max_col: (hw_roi.start_x + hw_roi.width - 1) as usize,
                        max_row: (hw_roi.start_y + hw_roi.height - 1) as usize,
                    })
                };

                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::from_secs(0));
                let timestamp = Timestamp::from_duration(now);

                let frame_num = frame_number.fetch_add(1, Ordering::SeqCst);

                let mut temperatures = HashMap::new();
                temperatures.insert("sensor".to_string(), temperature);

                let metadata = FrameMetadata {
                    frame_number: frame_num,
                    exposure: Duration::from_micros(exposure_us as u64),
                    timestamp,
                    pointing: None,
                    roi: roi_aabb,
                    temperatures,
                };

                callback(&array, &metadata)
            })
            .map_err(|e| CameraError::CaptureError(format!("Stream failed: {e}")))
    }

    fn saturation_value(&self) -> f64 {
        65535.0
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn get_bit_depth(&self) -> u8 {
        self.config.bit_depth
    }

    fn set_bit_depth(&mut self, bit_depth: u8) -> CameraResult<()> {
        self.config.bit_depth = bit_depth;
        Ok(())
    }

    fn get_serial(&self) -> String {
        self.serial_number.clone()
    }

    fn get_gain(&self) -> f64 {
        self.camera
            .lock()
            .ok()
            .and_then(|camera| camera.gain().ok())
            .map(|(gain, _is_auto)| gain as f64)
            .unwrap_or(0.0)
    }

    fn set_gain(&mut self, gain: f64) -> CameraResult<()> {
        let mut camera = self
            .camera
            .lock()
            .map_err(|_| CameraError::HardwareError("Camera mutex poisoned".to_string()))?;

        camera
            .set_gain(gain as i64, false)
            .map_err(|e| CameraError::ConfigError(format!("Failed to set gain: {e}")))
    }
}

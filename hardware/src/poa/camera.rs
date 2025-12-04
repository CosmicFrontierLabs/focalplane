use bytemuck;
use ndarray::Array2;
use playerone_sdk::{Camera, CameraDescription, ImageFormat};
use shared::camera_interface::{
    CameraConfig, CameraError, CameraInterface, CameraResult, FrameMetadata, SensorBitDepth,
    SensorGeometry, Timestamp,
};
use shared::image_proc::detection::aabb::AABB;
use shared::image_size::PixelShape;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::roi::{aabb_to_roi, roi_to_aabb};

fn get_sensor_geometry_for_camera(
    camera_name: &str,
    width: usize,
    height: usize,
) -> SensorGeometry {
    if camera_name.to_lowercase().contains("zeus") {
        SensorGeometry::new(9568, 6380, 3.76)
    } else {
        tracing::warn!(
            "Unknown PlayerOne camera '{}' - using detected dimensions with NaN pixel size",
            camera_name
        );
        SensorGeometry::new(width, height, f64::NAN)
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

        let config = CameraConfig::new(
            max_width,
            max_height,
            Duration::from_millis(100),
            SensorBitDepth::Bits16,
            51_000.0, // Max well depth in electrons (IMX571)
            1.0,      // DN per electron (gain dependent, nominal value)
        );

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
    fn check_roi_size(&self, size: PixelShape) -> CameraResult<()> {
        if size.width % 4 != 0 {
            return Err(CameraError::InvalidROI(
                "ROI width must be divisible by 4".to_string(),
            ));
        }
        if size.height % 2 != 0 {
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

        if roi.max_col >= self.config.size.width || roi.max_row >= self.config.size.height {
            return Err(CameraError::InvalidROI(
                "ROI extends beyond sensor bounds".to_string(),
            ));
        }

        let poa_roi = aabb_to_roi(&roi);
        let mut camera = self
            .camera
            .lock()
            .map_err(|_| CameraError::HardwareError("Camera mutex poisoned".to_string()))?;

        let _ = camera.stop_exposure();

        let current_hw_roi = camera.roi();
        if !super::roi::roi_eq(&current_hw_roi, &poa_roi) {
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

        let _ = camera.stop_exposure();

        camera
            .set_image_size(
                self.config.size.width as u32,
                self.config.size.height as u32,
            )
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

    fn geometry(&self) -> SensorGeometry {
        get_sensor_geometry_for_camera(&self.name, self.config.size.width, self.config.size.height)
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn get_roi(&self) -> Option<AABB> {
        let camera = self.camera.lock().ok()?;
        let hw_roi = camera.roi();

        if hw_roi.width == self.config.size.width as u32
            && hw_roi.height == self.config.size.height as u32
            && hw_roi.start_x == 0
            && hw_roi.start_y == 0
        {
            None
        } else {
            Some(roi_to_aabb(&hw_roi))
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

                let u16_data = bytemuck::cast_slice::<u8, u16>(buffer);

                let array = match Array2::from_shape_vec((height, width), u16_data.to_vec()) {
                    Ok(arr) => arr,
                    Err(_) => return false,
                };

                let temperature = cam.temperature().unwrap_or(f64::NAN);

                let roi_aabb = if hw_roi.width == config.size.width as u32
                    && hw_roi.height == config.size.height as u32
                    && hw_roi.start_x == 0
                    && hw_roi.start_y == 0
                {
                    None
                } else {
                    Some(roi_to_aabb(&hw_roi))
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
            .map_err(|e| CameraError::CaptureError(format!("Stream failed: {e}")))?;

        Ok(())
    }

    fn saturation_value(&self) -> f64 {
        65535.0
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn get_bit_depth(&self) -> SensorBitDepth {
        self.config.bit_depth
    }

    fn set_bit_depth(&mut self, bit_depth: SensorBitDepth) -> CameraResult<()> {
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

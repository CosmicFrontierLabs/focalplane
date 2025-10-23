use ndarray::Array2;
use playerone_sdk::{Camera, CameraDescription, ImageFormat, ROI};
use shared::camera_interface::{
    CameraConfig, CameraError, CameraInterface, CameraResult, FrameMetadata, Timestamp,
};
use shared::image_proc::detection::aabb::AABB;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type FrameBuffer = Arc<Mutex<Option<(Array2<u16>, FrameMetadata)>>>;

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
    is_capturing: Arc<AtomicBool>,
    is_streaming: Arc<AtomicBool>,
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
            bit_depth: 16,
        };

        Ok(Self {
            camera: Arc::new(Mutex::new(camera)),
            config,
            frame_number: Arc::new(AtomicU64::new(0)),
            is_capturing: Arc::new(AtomicBool::new(false)),
            is_streaming: Arc::new(AtomicBool::new(false)),
            latest_frame: Arc::new(Mutex::new(None)),
            capture_thread: None,
            name: camera_name,
        })
    }

    fn capture_internal(&mut self) -> CameraResult<(Array2<u16>, FrameMetadata)> {
        let mut camera = self
            .camera
            .lock()
            .map_err(|_| CameraError::HardwareError("Camera mutex poisoned".to_string()))?;

        if !self.is_streaming.load(Ordering::SeqCst) {
            camera.start_exposure().map_err(|e| {
                CameraError::CaptureError(format!("Failed to start streaming: {e}"))
            })?;
            self.is_streaming.store(true, Ordering::SeqCst);
            tracing::info!("Started continuous streaming mode");
        }

        let current_hw_roi = camera.roi();
        let (width, height) = (
            current_hw_roi.width as usize,
            current_hw_roi.height as usize,
        );

        let mut buffer = camera.create_image_buffer();

        let (exposure_us, _auto) = camera
            .exposure()
            .map_err(|e| CameraError::HardwareError(format!("Failed to read exposure: {e}")))?;
        let timeout_ms = (exposure_us / 1000) as i32 + 500;

        camera
            .get_image_data(&mut buffer, Some(timeout_ms))
            .map_err(|e| CameraError::CaptureError(format!("Get image data failed: {e}")))?;

        let u16_data: Vec<u16> = buffer
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        let array = Array2::from_shape_vec((height, width), u16_data)
            .map_err(|e| CameraError::CaptureError(format!("Failed to create array: {e}")))?;

        let temp_start = std::time::Instant::now();
        let temperature = camera
            .temperature()
            .map_err(|e| CameraError::HardwareError(format!("Failed to read temperature: {e}")))?;
        let temp_elapsed = temp_start.elapsed();

        tracing::info!(
            "temperature() took {:.2}ms",
            temp_elapsed.as_secs_f64() * 1000.0
        );

        drop(camera);

        let exposure = Duration::from_micros(exposure_us as u64);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0));
        let timestamp = Timestamp::from_duration(now);

        let frame_num = self.frame_number.fetch_add(1, Ordering::SeqCst);

        let mut temperatures = HashMap::new();
        temperatures.insert("sensor".to_string(), temperature);

        let roi_aabb = if current_hw_roi.width == self.config.width as u32
            && current_hw_roi.height == self.config.height as u32
            && current_hw_roi.start_x == 0
            && current_hw_roi.start_y == 0
        {
            None
        } else {
            Some(AABB {
                min_col: current_hw_roi.start_x as usize,
                min_row: current_hw_roi.start_y as usize,
                max_col: (current_hw_roi.start_x + current_hw_roi.width - 1) as usize,
                max_row: (current_hw_roi.start_y + current_hw_roi.height - 1) as usize,
            })
        };

        let metadata = FrameMetadata {
            frame_number: frame_num,
            exposure,
            timestamp,
            pointing: None,
            roi: roi_aabb,
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
            tracing::info!(
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

    fn capture_frame(&mut self) -> CameraResult<(Array2<u16>, FrameMetadata)> {
        self.capture_internal()
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

    fn start_continuous_capture(&mut self) -> CameraResult<()> {
        if self.is_capturing.load(Ordering::SeqCst) {
            return Ok(());
        }

        self.is_capturing.store(true, Ordering::SeqCst);

        let camera = Arc::clone(&self.camera);
        let config = self.config.clone();
        let is_capturing = Arc::clone(&self.is_capturing);
        let frame_number = Arc::clone(&self.frame_number);
        let latest_frame = Arc::clone(&self.latest_frame);

        let handle = std::thread::spawn(move || {
            while is_capturing.load(Ordering::SeqCst) {
                let mut camera = match camera.lock() {
                    Ok(cam) => cam,
                    Err(_) => break,
                };

                let exposure = match camera.exposure() {
                    Ok((exposure_us, _auto)) => Duration::from_micros(exposure_us as u64),
                    Err(_) => break,
                };

                let hw_roi = camera.roi();
                let (width, height) = (hw_roi.width as usize, hw_roi.height as usize);

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
                        roi: roi_aabb,
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

    fn get_bit_depth(&self) -> u8 {
        self.config.bit_depth
    }

    fn set_bit_depth(&mut self, bit_depth: u8) -> CameraResult<()> {
        self.config.bit_depth = bit_depth;
        Ok(())
    }
}

impl Drop for PlayerOneCamera {
    fn drop(&mut self) {
        let _ = self.stop_continuous_capture();

        if self.is_streaming.load(Ordering::SeqCst) {
            if let Ok(mut camera) = self.camera.lock() {
                let _ = camera.stop_exposure();
                self.is_streaming.store(false, Ordering::SeqCst);
            }
        }
    }
}

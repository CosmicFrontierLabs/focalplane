use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use v4l::buffer::{Metadata, Type};
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;

#[derive(Debug, Clone)]
pub struct CameraConfig {
    /// Path to the V4L2 device (e.g., "/dev/video0")
    pub device_path: String,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Frame rate in Hz (e.g., 23000000 for 23 MHz)
    pub framerate: u32,
    /// Analog gain value (sensor-specific units, typically 0-1023)
    pub gain: i32,
    /// Exposure time (sensor-specific units, typically in lines or microseconds)
    pub exposure: i32,
    /// Black level offset (sensor-specific units, typically 12-bit range 0-4095)
    pub black_level: i32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            device_path: "/dev/video0".to_string(),
            width: 1024,
            height: 1024,
            framerate: 23_000_000,
            gain: 360,
            exposure: 140,
            black_level: 4095,
        }
    }
}

pub struct V4L2Capture {
    config: CameraConfig,
}

#[derive(Debug, Clone)]
pub struct CaptureResult {
    pub frame: Vec<u8>,
    pub actual_width: u32,
    pub actual_height: u32,
    pub fourcc: [u8; 4],
}

impl V4L2Capture {
    pub fn new(config: CameraConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn open_device(&self) -> Result<Device> {
        Device::with_path(&self.config.device_path)
            .with_context(|| format!("Failed to open device: {}", self.config.device_path))
    }

    pub fn configure_device(&self, device: &mut Device) -> Result<()> {
        let mut format = device.format()?;
        let initial_fourcc_bytes = format.fourcc.repr;
        let initial_fourcc_str = std::str::from_utf8(&initial_fourcc_bytes).unwrap_or("????");

        tracing::info!(
            "Initial format: {}x{} {} ({:?})",
            format.width,
            format.height,
            initial_fourcc_str,
            initial_fourcc_bytes
        );

        format.width = self.config.width;
        format.height = self.config.height;
        format.fourcc = v4l::FourCC::new(b"Y16 ");

        tracing::info!(
            "Requesting format: {}x{} Y16",
            self.config.width,
            self.config.height
        );

        device.set_format(&format)?;

        // Log the actual format after setting
        let actual_format = device.format()?;
        let fourcc_bytes = actual_format.fourcc.repr;
        let fourcc_str = std::str::from_utf8(&fourcc_bytes).unwrap_or("????");

        // Check if stride/bytesperline is available
        // V4L2 format includes bytesperline for stride information
        let stride = actual_format.stride;
        let expected_stride = actual_format.width * 2; // For 16-bit format

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

        // Store the actual dimensions for later use
        if actual_format.width != self.config.width || actual_format.height != self.config.height {
            tracing::warn!(
                "Camera did not accept requested resolution. Using {}x{} instead of {}x{}",
                actual_format.width,
                actual_format.height,
                self.config.width,
                self.config.height
            );
        }

        if let Ok(controls) = device.query_controls() {
            for control_desc in controls {
                match control_desc.name.as_str() {
                    "Gain" | "gain" => {
                        let ctrl = v4l::Control {
                            id: control_desc.id,
                            value: v4l::control::Value::Integer(self.config.gain as i64),
                        };
                        let _ = device.set_control(ctrl);
                    }
                    "Exposure" | "exposure" => {
                        let ctrl = v4l::Control {
                            id: control_desc.id,
                            value: v4l::control::Value::Integer(self.config.exposure as i64),
                        };
                        let _ = device.set_control(ctrl);
                    }
                    "Black Level" | "black_level" => {
                        let ctrl = v4l::Control {
                            id: control_desc.id,
                            value: v4l::control::Value::Integer(self.config.black_level as i64),
                        };
                        let _ = device.set_control(ctrl);
                    }
                    "Frame Rate" | "frame_rate" => {
                        let ctrl = v4l::Control {
                            id: control_desc.id,
                            value: v4l::control::Value::Integer(self.config.framerate as i64),
                        };
                        let _ = device.set_control(ctrl);
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    pub fn capture_single_frame(&self) -> Result<Vec<u8>> {
        let mut device = self.open_device()?;
        self.configure_device(&mut device)?;

        let mut stream = MmapStream::new(&device, Type::VideoCapture)?;
        let (buf, _meta) = stream.next()?;
        Ok(buf.to_vec())
    }

    pub fn capture_single_frame_with_info(&self) -> Result<CaptureResult> {
        let mut device = self.open_device()?;
        self.configure_device(&mut device)?;

        let actual_format = device.format()?;
        let mut stream = MmapStream::new(&device, Type::VideoCapture)?;
        let (buf, _meta) = stream.next()?;

        Ok(CaptureResult {
            frame: buf.to_vec(),
            actual_width: actual_format.width,
            actual_height: actual_format.height,
            fourcc: actual_format.fourcc.repr,
        })
    }

    pub fn capture_frames_with_skip(&self, count: usize, skip: usize) -> Result<Vec<Vec<u8>>> {
        let mut device = self.open_device()?;
        self.configure_device(&mut device)?;

        let mut stream = MmapStream::new(&device, Type::VideoCapture)?;
        let mut frames = Vec::new();

        for _ in 0..skip {
            let _ = stream.next()?;
        }

        for _ in 0..count {
            let (buf, _meta) = stream.next()?;
            frames.push(buf.to_vec());
        }

        Ok(frames)
    }
}

pub struct CaptureSession<'a> {
    device: Device,
    stream: Option<MmapStream<'a>>,
}

impl<'a> CaptureSession<'a> {
    pub fn new(config: &CameraConfig) -> Result<Self>
    where
        Self: 'a,
    {
        let capture = V4L2Capture::new(config.clone())?;
        let mut device = capture.open_device()?;
        capture.configure_device(&mut device)?;

        Ok(Self {
            device,
            stream: None,
        })
    }

    pub fn start_stream(&mut self) -> Result<()> {
        tracing::info!("Starting V4L2 stream...");
        let stream = MmapStream::new(&self.device, Type::VideoCapture)?;
        self.stream = Some(stream);
        tracing::info!("V4L2 stream started successfully");
        Ok(())
    }

    pub fn capture_frame(&mut self) -> Result<(Vec<u8>, Metadata)> {
        tracing::debug!("Attempting to capture frame...");
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Stream not started"))?;

        let (buf, meta) = stream.next()?;
        let frame_size = buf.len();
        tracing::info!(
            "Captured frame: {} bytes, timestamp: {:?}",
            frame_size,
            meta.timestamp
        );
        Ok((buf.to_vec(), *meta))
    }

    pub fn stop_stream(&mut self) {
        self.stream.take();
    }

    pub fn save_raw_frame(&mut self, path: &Path) -> Result<()> {
        let (frame, _meta) = self.capture_frame()?;
        std::fs::write(path, frame)?;
        Ok(())
    }
}

pub struct FrameBuffer {
    frames: Arc<Mutex<Vec<Vec<u8>>>>,
    max_frames: usize,
}

impl FrameBuffer {
    pub fn new(max_frames: usize) -> Self {
        Self {
            frames: Arc::new(Mutex::new(Vec::new())),
            max_frames,
        }
    }

    pub async fn push(&self, frame: Vec<u8>) {
        let mut frames = self.frames.lock().await;
        if frames.len() >= self.max_frames {
            frames.remove(0);
        }
        frames.push(frame);
    }

    pub async fn get_latest(&self) -> Option<Vec<u8>> {
        let frames = self.frames.lock().await;
        frames.last().cloned()
    }

    pub async fn get_all(&self) -> Vec<Vec<u8>> {
        self.frames.lock().await.clone()
    }

    pub async fn clear(&self) {
        self.frames.lock().await.clear();
    }

    pub async fn len(&self) -> usize {
        self.frames.lock().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.frames.lock().await.is_empty()
    }
}

pub struct ResolutionProfile {
    pub width: u32,
    pub height: u32,
    pub framerate: u32,
    pub test_frames: u32,
}

impl ResolutionProfile {
    pub fn standard_profiles() -> Vec<Self> {
        vec![
            Self {
                width: 128,
                height: 128,
                framerate: 133_000_000,
                test_frames: 134,
            },
            Self {
                width: 256,
                height: 256,
                framerate: 83_000_000,
                test_frames: 84,
            },
            Self {
                width: 512,
                height: 512,
                framerate: 44_000_000,
                test_frames: 45,
            },
            Self {
                width: 1024,
                height: 1024,
                framerate: 23_000_000,
                test_frames: 24,
            },
            Self {
                width: 2048,
                height: 2048,
                framerate: 12_000_000,
                test_frames: 13,
            },
            Self {
                width: 4096,
                height: 4096,
                framerate: 6_000_000,
                test_frames: 7,
            },
            Self {
                width: 8096,
                height: 6324,
                framerate: 3_700_000,
                test_frames: 4,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_config_default() {
        let config = CameraConfig::default();
        assert_eq!(config.device_path, "/dev/video0");
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 1024);
    }

    #[test]
    fn test_resolution_profiles() {
        let profiles = ResolutionProfile::standard_profiles();
        assert_eq!(profiles.len(), 7);
        assert_eq!(profiles[0].width, 128);
        assert_eq!(profiles[6].width, 8096);
    }

    #[tokio::test]
    async fn test_frame_buffer() {
        let buffer = FrameBuffer::new(3);

        buffer.push(vec![1, 2, 3]).await;
        buffer.push(vec![4, 5, 6]).await;
        buffer.push(vec![7, 8, 9]).await;

        assert_eq!(buffer.len().await, 3);

        buffer.push(vec![10, 11, 12]).await;
        assert_eq!(buffer.len().await, 3);

        let latest = buffer.get_latest().await;
        assert_eq!(latest, Some(vec![10, 11, 12]));
    }
}

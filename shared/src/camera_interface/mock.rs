use super::{AABBExt, CameraConfig, CameraInterface, CameraResult, FrameMetadata, Timestamp};
use crate::image_proc::detection::AABB;
use ndarray::Array2;
use std::collections::HashMap;
use std::time::Duration;

type FrameGenerator =
    Box<dyn FnMut(u64, Duration, Option<AABB>, &CameraConfig) -> Array2<u16> + Send + Sync>;

enum FrameSource {
    Pregenerated {
        frames: Vec<Array2<u16>>,
        index: usize,
    },
    Generated(FrameGenerator),
}

pub struct MockCameraInterface {
    config: CameraConfig,
    roi: Option<AABB>,
    frame_count: u64,
    saturation: f64,
    source: FrameSource,
    elapsed_time: Duration,
    gain: f64,
}

impl MockCameraInterface {
    pub fn new(config: CameraConfig, frames: Vec<Array2<u16>>) -> Self {
        Self {
            config,
            roi: None,
            frame_count: 0,
            saturation: 65535.0,
            source: FrameSource::Pregenerated { frames, index: 0 },
            elapsed_time: Duration::ZERO,
            gain: 0.0,
        }
    }

    pub fn new_repeating(config: CameraConfig, frame: Array2<u16>) -> Self {
        Self::new(config, vec![frame])
    }

    pub fn new_zeros(config: CameraConfig) -> Self {
        let frame_data = Array2::zeros((config.height, config.width));
        Self::new_repeating(config, frame_data)
    }

    pub fn with_generator<F>(config: CameraConfig, generator: F) -> Self
    where
        F: FnMut(u64, Duration, Option<AABB>, &CameraConfig) -> Array2<u16> + Send + Sync + 'static,
    {
        Self {
            config,
            roi: None,
            frame_count: 0,
            saturation: 65535.0,
            source: FrameSource::Generated(Box::new(generator)),
            elapsed_time: Duration::ZERO,
            gain: 0.0,
        }
    }

    pub fn with_saturation(mut self, saturation: f64) -> Self {
        self.saturation = saturation;
        self
    }

    pub fn reset(&mut self) {
        if let FrameSource::Pregenerated { index, .. } = &mut self.source {
            *index = 0;
        }
        self.frame_count = 0;
        self.elapsed_time = Duration::ZERO;
    }

    fn generate_frame(&mut self) -> CameraResult<Array2<u16>> {
        let full_frame = match &mut self.source {
            FrameSource::Pregenerated { frames, index } => {
                let frame_idx = if frames.len() == 1 {
                    0
                } else {
                    if *index >= frames.len() {
                        return Err(crate::camera_interface::CameraError::CaptureError(
                            "No more frames".to_string(),
                        ));
                    }
                    let current = *index;
                    *index += 1;
                    current
                };
                frames[frame_idx].clone()
            }
            FrameSource::Generated(generator) => {
                let next_frame_number = self.frame_count + 1;
                let next_timestamp = self.elapsed_time + self.config.exposure;
                generator(next_frame_number, next_timestamp, self.roi, &self.config)
            }
        };

        let output_frame = if let Some(roi) = &self.roi {
            roi.extract_from_frame(&full_frame.view())
        } else {
            full_frame
        };

        Ok(output_frame)
    }

    fn generate_metadata(&self) -> FrameMetadata {
        let timestamp = Timestamp::from_duration(self.elapsed_time);
        let mut temperatures = HashMap::new();
        temperatures.insert("sensor".to_string(), 20.0);

        FrameMetadata {
            frame_number: self.frame_count,
            exposure: self.config.exposure,
            timestamp,
            pointing: None,
            roi: self.roi,
            temperatures,
        }
    }
}

impl CameraInterface for MockCameraInterface {
    fn check_roi_size(&self, _width: usize, _height: usize) -> CameraResult<()> {
        Ok(())
    }

    fn set_roi(&mut self, roi: AABB) -> CameraResult<()> {
        roi.validate_for_sensor(self.config.width, self.config.height)?;
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

    fn get_config(&self) -> &CameraConfig {
        &self.config
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn get_roi(&self) -> Option<AABB> {
        self.roi
    }

    fn saturation_value(&self) -> f64 {
        self.saturation
    }

    fn name(&self) -> &str {
        "MockCamera"
    }

    fn get_bit_depth(&self) -> u8 {
        self.config.bit_depth
    }

    fn set_bit_depth(&mut self, bit_depth: u8) -> CameraResult<()> {
        self.config.bit_depth = bit_depth;
        Ok(())
    }

    fn get_serial(&self) -> String {
        "MOCK-00000".to_string()
    }

    fn get_gain(&self) -> f64 {
        self.gain
    }

    fn set_gain(&mut self, gain: f64) -> CameraResult<()> {
        self.gain = gain;
        Ok(())
    }

    fn stream(
        &mut self,
        callback: &mut dyn FnMut(&Array2<u16>, &FrameMetadata) -> bool,
    ) -> CameraResult<()> {
        loop {
            let frame = self.generate_frame()?;
            self.elapsed_time += self.config.exposure;
            self.frame_count += 1;
            let metadata = self.generate_metadata();

            if !callback(&frame, &metadata) {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_camera() -> MockCameraInterface {
        let config = CameraConfig {
            width: 640,
            height: 480,
            exposure: Duration::from_millis(100),
            bit_depth: 16,
        };
        MockCameraInterface::new_zeros(config)
    }

    fn create_config(width: usize, height: usize, exposure_ms: u64) -> CameraConfig {
        CameraConfig {
            width,
            height,
            exposure: Duration::from_millis(exposure_ms),
            bit_depth: 16,
        }
    }

    #[test]
    fn test_basic_capture() {
        let mut camera = create_test_camera();
        assert!(camera.is_ready());

        let (frame, metadata) = camera.capture_frame().unwrap();
        assert_eq!(frame.shape(), &[480, 640]);
        assert_eq!(metadata.frame_number, 1);
    }

    #[test]
    fn test_exposure_setting() {
        let mut camera = create_test_camera();
        assert_eq!(camera.get_exposure(), Duration::from_millis(100));
        let new_exposure = Duration::from_millis(200);
        camera.set_exposure(new_exposure).unwrap();
        assert_eq!(camera.get_exposure(), new_exposure);
    }

    #[test]
    fn test_roi_operations() {
        let mut camera = create_test_camera();

        let roi = AABB {
            min_col: 100,
            min_row: 100,
            max_col: 200,
            max_row: 200,
        };

        camera.set_roi(roi).unwrap();
        assert_eq!(camera.get_roi(), Some(roi));

        let (frame, metadata) = camera.capture_frame().unwrap();
        assert_eq!(frame.shape(), &[101, 101]);
        assert_eq!(metadata.roi, Some(roi));

        camera.clear_roi().unwrap();
        assert_eq!(camera.get_roi(), None);

        let (frame, _) = camera.capture_frame().unwrap();
        assert_eq!(frame.shape(), &[480, 640]);
    }

    #[test]
    fn test_invalid_roi() {
        let mut camera = create_test_camera();

        let roi = AABB {
            min_col: 600,
            min_row: 100,
            max_col: 700,
            max_row: 200,
        };

        assert!(camera.set_roi(roi).is_err());
    }

    #[test]
    fn test_stream() {
        let mut camera = create_test_camera();

        let mut frame_count = 0;
        let mut last_frame_shape = None;
        let mut last_metadata_frame_number = None;

        camera
            .stream(&mut |frame, metadata| {
                frame_count += 1;
                last_frame_shape = Some(frame.shape().to_vec());
                last_metadata_frame_number = Some(metadata.frame_number);
                frame_count < 3
            })
            .unwrap();

        assert_eq!(frame_count, 3);
        assert_eq!(last_frame_shape, Some(vec![480, 640]));
        assert_eq!(last_metadata_frame_number, Some(3));
    }

    #[test]
    fn test_frame_counting() {
        let mut camera = create_test_camera();

        for i in 1..=5 {
            let (_, metadata) = camera.capture_frame().unwrap();
            assert_eq!(metadata.frame_number, i);
        }
    }

    #[test]
    fn test_saturation_value() {
        let camera = create_test_camera();
        assert_eq!(camera.saturation_value(), 65535.0);

        let custom_camera =
            MockCameraInterface::new_zeros(create_config(640, 480, 100)).with_saturation(16383.0);
        assert_eq!(custom_camera.saturation_value(), 16383.0);
    }

    #[test]
    fn test_camera_name() {
        let camera = create_test_camera();
        assert_eq!(camera.name(), "MockCamera");
    }

    #[test]
    fn test_multiple_frames() {
        let config = create_config(10, 10, 10);

        let mut frame1 = Array2::zeros((10, 10));
        frame1[[5, 5]] = 100;
        let mut frame2 = Array2::zeros((10, 10));
        frame2[[3, 3]] = 200;
        let mut frame3 = Array2::zeros((10, 10));
        frame3[[7, 7]] = 300;

        let mut camera = MockCameraInterface::new(config, vec![frame1, frame2, frame3]);

        let (f1, _) = camera.capture_frame().unwrap();
        assert_eq!(f1[[5, 5]], 100);

        let (f2, _) = camera.capture_frame().unwrap();
        assert_eq!(f2[[3, 3]], 200);

        let (f3, _) = camera.capture_frame().unwrap();
        assert_eq!(f3[[7, 7]], 300);

        assert!(camera.capture_frame().is_err());
    }

    #[test]
    fn test_repeating_mode() {
        let config = create_config(10, 10, 10);

        let mut frame = Array2::zeros((10, 10));
        frame[[5, 5]] = 42;

        let mut camera = MockCameraInterface::new_repeating(config, frame);

        for _ in 0..10 {
            let (f, _) = camera.capture_frame().unwrap();
            assert_eq!(f[[5, 5]], 42);
        }
    }

    #[test]
    fn test_elapsed_time_tracking() {
        let config = create_config(10, 10, 100);

        let mut camera = MockCameraInterface::new_zeros(config);

        let (_, meta1) = camera.capture_frame().unwrap();
        assert_eq!(meta1.timestamp.to_duration(), Duration::from_millis(100));

        let (_, meta2) = camera.capture_frame().unwrap();
        assert_eq!(meta2.timestamp.to_duration(), Duration::from_millis(200));

        let (_, meta3) = camera.capture_frame().unwrap();
        assert_eq!(meta3.timestamp.to_duration(), Duration::from_millis(300));
    }

    #[test]
    fn test_reset() {
        let config = create_config(10, 10, 50);

        let frame1 = Array2::ones((10, 10));
        let frame2 = Array2::ones((10, 10)) * 2;

        let mut camera = MockCameraInterface::new(config, vec![frame1, frame2]);

        camera.capture_frame().unwrap();
        let (_, meta) = camera.capture_frame().unwrap();
        assert_eq!(meta.frame_number, 2);
        assert_eq!(meta.timestamp.to_duration(), Duration::from_millis(100));

        camera.reset();

        let (_, meta) = camera.capture_frame().unwrap();
        assert_eq!(meta.frame_number, 1);
        assert_eq!(meta.timestamp.to_duration(), Duration::from_millis(50));
    }

    #[test]
    fn test_temperature_metadata() {
        let mut camera = create_test_camera();
        let (_, metadata) = camera.capture_frame().unwrap();
        assert_eq!(metadata.temperatures.get("sensor"), Some(&20.0));
    }

    #[test]
    fn test_bit_depth() {
        let mut camera = create_test_camera();
        assert_eq!(camera.get_bit_depth(), 16);

        camera.set_bit_depth(12).unwrap();
        assert_eq!(camera.get_bit_depth(), 12);

        camera.set_bit_depth(14).unwrap();
        assert_eq!(camera.get_bit_depth(), 14);
    }

    #[test]
    fn test_gain() {
        let mut camera = create_test_camera();
        assert_eq!(camera.get_gain(), 0.0);

        camera.set_gain(100.0).unwrap();
        assert_eq!(camera.get_gain(), 100.0);

        camera.set_gain(50.5).unwrap();
        assert_eq!(camera.get_gain(), 50.5);

        camera.set_gain(0.0).unwrap();
        assert_eq!(camera.get_gain(), 0.0);
    }
}

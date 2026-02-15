//! Asynchronous frame writer with worker thread pool.
//!
//! Provides a reusable component for writing frames to disk without blocking
//! the main capture loop. Uses a bounded channel and worker thread pool.

use anyhow::{Context, Result};
use crossbeam_channel::{bounded, Sender, TrySendError};
use fitsio::compat::fitsfile::FitsFile;
use fitsio::compat::images::{ImageDescription, ImageType, WriteImage};
use image::DynamicImage;
use ndarray::Array2;
use std::mem;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use tracing::{info, warn};

use crate::image_proc::{array2_to_gray16_image, array2_to_gray_image};

#[derive(Debug, Clone)]
pub enum ImagePayload {
    U16(Array2<u16>),
    U8(Array2<u8>),
    F64(Array2<f64>),
    Dynamic(DynamicImage),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameFormat {
    Png,
    Fits,
}

pub struct FrameWriterHandle {
    sender: Sender<FrameWriteTask>,
    workers: Vec<JoinHandle<()>>,
}

struct FrameWriteTask {
    payload: ImagePayload,
    filepath: PathBuf,
    format: FrameFormat,
}

impl FrameWriterHandle {
    pub fn new(num_workers: usize, buffer_size: usize) -> Result<Self> {
        let (sender, receiver) = bounded::<FrameWriteTask>(buffer_size);

        let mut workers = Vec::new();
        for worker_id in 0..num_workers {
            let receiver = receiver.clone();

            let handle = std::thread::spawn(move || {
                info!("Frame writer worker {} started", worker_id);
                while let Ok(task) = receiver.recv() {
                    if let Err(e) = save_frame(&task.payload, &task.filepath, task.format) {
                        warn!(
                            "Worker {} failed to save frame to {}: {}",
                            worker_id,
                            task.filepath.display(),
                            e
                        );
                    }
                }
                info!("Frame writer worker {} shutting down", worker_id);
            });

            workers.push(handle);
        }

        Ok(Self { sender, workers })
    }

    pub fn wait_for_completion(mut self) {
        mem::drop(self.sender);

        for (worker_id, handle) in self.workers.drain(..).enumerate() {
            if let Err(e) = handle.join() {
                warn!("Worker {} panicked: {:?}", worker_id, e);
            }
        }

        info!("All frame writer workers completed");
    }

    pub fn write_frame(
        &self,
        frame_data: &Array2<u16>,
        filepath: PathBuf,
        format: FrameFormat,
    ) -> Result<()> {
        let task = FrameWriteTask {
            payload: ImagePayload::U16(frame_data.clone()),
            filepath: filepath.clone(),
            format,
        };

        match self.sender.try_send(task) {
            Ok(_) => Ok(()),
            Err(TrySendError::Full(_)) => {
                anyhow::bail!(
                    "Frame writer queue full, cannot write to {}",
                    filepath.display()
                )
            }
            Err(TrySendError::Disconnected(_)) => {
                anyhow::bail!("Frame writer workers have shut down")
            }
        }
    }

    pub fn write_frame_nonblocking(
        &self,
        frame_data: &Array2<u16>,
        filepath: PathBuf,
        format: FrameFormat,
    ) -> bool {
        let task = FrameWriteTask {
            payload: ImagePayload::U16(frame_data.clone()),
            filepath,
            format,
        };

        self.sender.try_send(task).is_ok()
    }
    pub fn write_u8_frame(
        &self,
        frame_data: &Array2<u8>,
        filepath: PathBuf,
        format: FrameFormat,
    ) -> Result<()> {
        let task = FrameWriteTask {
            payload: ImagePayload::U8(frame_data.clone()),
            filepath,
            format,
        };
        self.send_task(task)
    }

    pub fn write_f64_frame(
        &self,
        frame_data: &Array2<f64>,
        filepath: PathBuf,
        format: FrameFormat,
    ) -> Result<()> {
        let task = FrameWriteTask {
            payload: ImagePayload::F64(frame_data.clone()),
            filepath,
            format,
        };
        self.send_task(task)
    }

    pub fn write_dynamic_image(
        &self,
        image: DynamicImage,
        filepath: PathBuf,
        format: FrameFormat,
    ) -> Result<()> {
        let task = FrameWriteTask {
            payload: ImagePayload::Dynamic(image),
            filepath,
            format,
        };
        self.send_task(task)
    }

    fn send_task(&self, task: FrameWriteTask) -> Result<()> {
        match self.sender.try_send(task) {
            Ok(_) => Ok(()),
            Err(TrySendError::Full(_)) => {
                anyhow::bail!("Frame writer queue full")
            }
            Err(TrySendError::Disconnected(_)) => {
                anyhow::bail!("Frame writer workers have shut down")
            }
        }
    }
}

fn save_frame(payload: &ImagePayload, filepath: &Path, format: FrameFormat) -> Result<()> {
    if let Some(parent) = filepath.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    match format {
        FrameFormat::Png => save_as_png(payload, filepath),
        FrameFormat::Fits => save_as_fits(payload, filepath),
    }
}

fn save_as_png(payload: &ImagePayload, filepath: &Path) -> Result<()> {
    match payload {
        ImagePayload::U16(frame) => {
            let img_buffer = array2_to_gray16_image(frame);
            img_buffer.save(filepath)?;
        }
        ImagePayload::U8(frame) => {
            let img_buffer = array2_to_gray_image(frame);
            img_buffer.save(filepath)?;
        }
        ImagePayload::Dynamic(image) => {
            image.save(filepath)?;
        }
        ImagePayload::F64(_) => {
            anyhow::bail!("Saving f64 arrays as PNG is not supported directly");
        }
    }
    Ok(())
}

fn save_as_fits(payload: &ImagePayload, filepath: &Path) -> Result<()> {
    let (width, height, data_type) = match payload {
        ImagePayload::U16(frame) => {
            let (h, w) = frame.dim();
            (w, h, ImageType::Long)
        }
        ImagePayload::F64(frame) => {
            let (h, w) = frame.dim();
            (w, h, ImageType::Double)
        }
        _ => anyhow::bail!("Unsupported payload type for FITS"),
    };

    let image_description = ImageDescription {
        data_type,
        dimensions: vec![width, height],
    };

    let mut fptr = FitsFile::create(filepath)
        .overwrite()
        .open()
        .map_err(|e| anyhow::anyhow!("Failed to create FITS file {}: {}", filepath.display(), e))?;

    let hdu = fptr
        .create_image("PRIMARY", &image_description)
        .map_err(|e| anyhow::anyhow!("Failed to create image HDU {}: {}", filepath.display(), e))?;

    match payload {
        ImagePayload::U16(frame) => {
            let data: Vec<i32> = frame.iter().map(|&v| v as i32).collect();
            i32::write_image(&mut fptr, &hdu, &data)?;
        }
        ImagePayload::F64(frame) => {
            let data: Vec<f64> = frame.iter().cloned().collect();
            f64::write_image(&mut fptr, &hdu, &data)?;
        }
        _ => unreachable!(),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use tempfile::TempDir;

    #[test]
    fn test_frame_writer_basic() {
        let temp_dir = TempDir::new().unwrap();
        let writer = FrameWriterHandle::new(2, 10).unwrap();

        let frame = Array2::from_shape_fn((64, 64), |(y, x)| ((x + y) * 100) as u16);

        let filepath = temp_dir.path().join("test_frame.png");
        writer
            .write_frame(&frame, filepath.clone(), FrameFormat::Png)
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        assert!(filepath.exists());
    }

    #[test]
    fn test_frame_writer_multiple_frames() {
        let temp_dir = TempDir::new().unwrap();
        let writer = FrameWriterHandle::new(2, 10).unwrap();

        for i in 0..5 {
            let frame = Array2::from_shape_fn((32, 32), |(y, x)| ((x + y + i) * 50) as u16);

            let filepath = temp_dir.path().join(format!("frame_{}.png", i));
            writer
                .write_frame(&frame, filepath, FrameFormat::Png)
                .unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(200));

        for i in 0..5 {
            let filepath = temp_dir.path().join(format!("frame_{}.png", i));
            assert!(filepath.exists(), "Frame {} should exist", i);
        }
    }

    #[test]
    fn test_frame_writer_nonblocking() {
        let temp_dir = TempDir::new().unwrap();
        let writer = FrameWriterHandle::new(1, 2).unwrap();

        let frame = Array2::from_shape_fn((16, 16), |(y, x)| ((x + y) * 200) as u16);

        let success = writer.write_frame_nonblocking(
            &frame,
            temp_dir.path().join("nonblock_frame.png"),
            FrameFormat::Png,
        );
        assert!(success);
    }

    #[test]
    fn test_save_frame_png() {
        let temp_dir = TempDir::new().unwrap();
        let filepath = temp_dir.path().join("frame.png");

        let frame = Array2::from_shape_fn((8, 8), |(y, x)| ((x + y) * 10) as u16);

        save_frame(&ImagePayload::U16(frame), &filepath, FrameFormat::Png).unwrap();
        assert!(filepath.exists());
    }

    #[test]
    fn test_frame_writer_creates_nested_directories() {
        let temp_dir = TempDir::new().unwrap();
        let writer = FrameWriterHandle::new(1, 5).unwrap();

        let frame = Array2::from_shape_fn((16, 16), |(y, x)| ((x + y) * 150) as u16);

        let nested_path = temp_dir.path().join("subdir1/subdir2/nested_frame.png");
        writer
            .write_frame(&frame, nested_path.clone(), FrameFormat::Png)
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        assert!(nested_path.exists());
        assert!(nested_path.parent().unwrap().exists());
    }

    #[test]
    fn test_frame_writer_wait_for_completion() {
        let temp_dir = TempDir::new().unwrap();
        let writer = FrameWriterHandle::new(2, 10).unwrap();

        let mut paths = Vec::new();
        for i in 0..10 {
            let frame = Array2::from_shape_fn((32, 32), |(y, x)| ((x + y + i) * 20) as u16);
            let path = temp_dir.path().join(format!("wait_test_{}.png", i));
            paths.push(path.clone());
            writer.write_frame(&frame, path, FrameFormat::Png).unwrap();
        }

        writer.wait_for_completion();

        for path in paths {
            assert!(path.exists(), "Frame at {} should exist", path.display());
        }
    }

    #[test]
    fn test_save_fits_file() {
        let temp_dir = TempDir::new().unwrap();
        let filepath = temp_dir.path().join("test_frame.fits");

        let frame = Array2::from_shape_fn((16, 16), |(y, x)| ((x + y) * 100) as u16);

        save_frame(&ImagePayload::U16(frame), &filepath, FrameFormat::Fits).unwrap();
        assert!(filepath.exists());
    }

    #[test]
    fn test_frame_writer_fits() {
        let temp_dir = TempDir::new().unwrap();
        let writer = FrameWriterHandle::new(2, 10).unwrap();

        let frame = Array2::from_shape_fn((32, 32), |(y, x)| ((x + y) * 50) as u16);

        let filepath = temp_dir.path().join("test_frame.fits");
        writer
            .write_frame(&frame, filepath.clone(), FrameFormat::Fits)
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        assert!(filepath.exists());
    }

    #[test]
    fn test_frame_writer_mixed_formats() {
        let temp_dir = TempDir::new().unwrap();
        let writer = FrameWriterHandle::new(2, 10).unwrap();

        let frame = Array2::from_shape_fn((24, 24), |(y, x)| ((x + y) * 75) as u16);

        let png_path = temp_dir.path().join("frame.png");
        let fits_path = temp_dir.path().join("frame.fits");
        let fit_path = temp_dir.path().join("frame.fit");

        writer
            .write_frame(&frame, png_path.clone(), FrameFormat::Png)
            .unwrap();
        writer
            .write_frame(&frame, fits_path.clone(), FrameFormat::Fits)
            .unwrap();
        writer
            .write_frame(&frame, fit_path.clone(), FrameFormat::Fits)
            .unwrap();

        writer.wait_for_completion();

        assert!(png_path.exists(), "PNG file should exist");
        assert!(fits_path.exists(), "FITS file should exist");
        assert!(fit_path.exists(), "FIT file should exist");
    }
}

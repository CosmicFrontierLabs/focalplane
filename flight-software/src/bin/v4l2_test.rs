use anyhow::Result;
use flight_software::v4l2_capture::{
    CameraConfig, CaptureResult, CaptureSession, ResolutionProfile, V4L2Capture,
};
use image::{ImageBuffer, Luma};
use std::time::Instant;
use tracing::{error, info};

fn save_frame_as_png_from_capture(capture: &CaptureResult, filename: &str) -> Result<()> {
    let fourcc_str = std::str::from_utf8(&capture.fourcc).unwrap_or("????");
    info!(
        "Converting frame: {} bytes, {}x{}, format: {}",
        capture.frame.len(),
        capture.actual_width,
        capture.actual_height,
        fourcc_str
    );

    let pixels_8bit = match &capture.fourcc {
        // RG16 format - 16-bit Bayer pattern
        [82, 71, 49, 54] => {
            info!("Processing as RG16 format (16-bit Bayer pattern)");

            // RG16 appears to have extra data - calculate actual image data size
            let expected_raw_size = (capture.actual_width * capture.actual_height * 2) as usize;
            let extra_bytes = capture.frame.len() - expected_raw_size;

            if extra_bytes > 0 {
                info!(
                    "Frame has {} extra bytes beyond expected {}x{}x2 = {} bytes",
                    extra_bytes, capture.actual_width, capture.actual_height, expected_raw_size
                );

                // Calculate stride - seems to be padded rows
                let bytes_per_row = capture.frame.len() / capture.actual_height as usize;
                let expected_bytes_per_row = (capture.actual_width * 2) as usize;
                info!(
                    "Bytes per row: {} (expected: {}), padding: {} bytes/row",
                    bytes_per_row,
                    expected_bytes_per_row,
                    bytes_per_row - expected_bytes_per_row
                );
            }

            // For Bayer pattern, we need to demosaic, but for now just take the red channel
            // RG16 layout: R G R G...
            //              G B G B...
            // We'll take every other pixel from every other row (red pixels)
            let mut pixels = Vec::new();
            let stride = capture.frame.len() / capture.actual_height as usize;

            for y in 0..capture.actual_height {
                let row_start = (y as usize) * stride;
                for x in 0..capture.actual_width {
                    let pixel_offset = row_start + (x as usize) * 2;
                    if pixel_offset + 1 < capture.frame.len() {
                        let value = u16::from_le_bytes([
                            capture.frame[pixel_offset],
                            capture.frame[pixel_offset + 1],
                        ]);
                        pixels.push((value >> 8) as u8);
                    } else {
                        pixels.push(0);
                    }
                }
            }
            pixels
        }
        // Y16 format - 16-bit grayscale
        [89, 49, 54, 32] | [89, 49, 54, 0] => {
            info!("Processing as Y16 format (16-bit grayscale)");
            let expected_size = (capture.actual_width * capture.actual_height * 2) as usize;
            if capture.frame.len() != expected_size {
                error!(
                    "Y16 size mismatch: got {} bytes, expected {}",
                    capture.frame.len(),
                    expected_size
                );
            }

            capture
                .frame
                .chunks(2)
                .map(|chunk| {
                    if chunk.len() == 2 {
                        (u16::from_le_bytes([chunk[0], chunk[1]]) >> 8) as u8
                    } else {
                        chunk[0]
                    }
                })
                .collect()
        }
        // YUYV format - YUV 4:2:2
        [89, 85, 89, 86] => {
            info!("Processing as YUYV format (YUV 4:2:2)");
            let expected_size = (capture.actual_width * capture.actual_height * 2) as usize;
            if capture.frame.len() != expected_size {
                error!(
                    "YUYV size mismatch: got {} bytes, expected {}",
                    capture.frame.len(),
                    expected_size
                );
            }

            // Extract Y (luminance) from YUYV
            // YUYV format: Y0 U0 Y1 V0 | Y2 U1 Y3 V1 | ...
            capture
                .frame
                .chunks(2)
                .step_by(1)
                .enumerate()
                .filter(|(i, _)| i % 2 == 0)
                .map(|(_, chunk)| chunk[0])
                .collect()
        }
        // Other formats - try to interpret as 8-bit grayscale
        _ => {
            info!(
                "Unknown format {:?}, attempting 8-bit grayscale interpretation",
                capture.fourcc
            );
            let expected_size = (capture.actual_width * capture.actual_height) as usize;
            if capture.frame.len() == expected_size {
                capture.frame.clone()
            } else if capture.frame.len() == expected_size * 2 {
                // Might be 16-bit, take every other byte
                capture.frame.iter().step_by(2).copied().collect()
            } else {
                error!(
                    "Cannot determine pixel format. Frame size {} doesn't match {}x{}",
                    capture.frame.len(),
                    capture.actual_width,
                    capture.actual_height
                );
                return Err(anyhow::anyhow!("Cannot determine pixel format"));
            }
        }
    };

    let expected_pixels = (capture.actual_width * capture.actual_height) as usize;
    if pixels_8bit.len() != expected_pixels {
        return Err(anyhow::anyhow!(
            "Pixel count mismatch after conversion: got {} pixels, expected {}",
            pixels_8bit.len(),
            expected_pixels
        ));
    }

    let img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
        capture.actual_width,
        capture.actual_height,
        pixels_8bit,
    )
    .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    img.save(filename)?;
    info!("Saved image as {}", filename);
    Ok(())
}

fn save_frame_as_png(frame: &[u8], width: u32, height: u32, filename: &str) -> Result<()> {
    let expected_size = (width * height * 2) as usize;
    info!(
        "Frame size: {} bytes, expected: {} bytes ({}x{}x2)",
        frame.len(),
        expected_size,
        width,
        height
    );

    // Check if we have the right amount of data
    if frame.len() != expected_size {
        error!(
            "Frame size mismatch! Got {} bytes but expected {} bytes for {}x{} Y16 format",
            frame.len(),
            expected_size,
            width,
            height
        );
    }

    // Assuming 16-bit raw data from camera, convert to 8-bit for PNG
    // Using chunks instead of chunks_exact to handle size mismatches
    let mut pixels_16bit = Vec::new();
    for chunk in frame.chunks(2) {
        if chunk.len() == 2 {
            pixels_16bit.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        } else if chunk.len() == 1 {
            // Handle odd byte at end by padding with 0
            pixels_16bit.push(u16::from_le_bytes([chunk[0], 0]));
        }
    }

    // Scale 16-bit values to 8-bit (divide by 256)
    let pixels_8bit: Vec<u8> = pixels_16bit.iter().map(|&val| (val >> 8) as u8).collect();

    // Check pixel count
    let expected_pixels = (width * height) as usize;
    if pixels_8bit.len() != expected_pixels {
        return Err(anyhow::anyhow!(
            "Pixel count mismatch! Got {} pixels but expected {} for {}x{}",
            pixels_8bit.len(),
            expected_pixels,
            width,
            height
        ));
    }

    let img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, pixels_8bit)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    img.save(filename)?;
    Ok(())
}

fn test_resolution_profile(device: &str, profile: &ResolutionProfile) -> Result<()> {
    info!(
        "Testing {} - Resolution {}x{} @ {} Hz",
        device, profile.width, profile.height, profile.framerate
    );

    let config = CameraConfig {
        device_path: device.to_string(),
        width: profile.width,
        height: profile.height,
        framerate: profile.framerate,
        gain: 360,
        exposure: 140,
        black_level: 4095,
    };

    let capture = V4L2Capture::new(config)?;

    let start = Instant::now();
    let capture_result = capture.capture_single_frame_with_info()?;
    let elapsed = start.elapsed();

    let fourcc_str = std::str::from_utf8(&capture_result.fourcc).unwrap_or("????");
    info!(
        "Captured frame: {} bytes, actual resolution: {}x{}, format: {}",
        capture_result.frame.len(),
        capture_result.actual_width,
        capture_result.actual_height,
        fourcc_str
    );

    // Save as raw
    let raw_filename = format!(
        "test_{}x{}.raw",
        capture_result.actual_width, capture_result.actual_height
    );
    std::fs::write(&raw_filename, &capture_result.frame)?;

    // Save as PNG with actual dimensions
    let png_filename = format!(
        "test_{}x{}.png",
        capture_result.actual_width, capture_result.actual_height
    );
    save_frame_as_png_from_capture(&capture_result, &png_filename)?;

    let raw_size = std::fs::metadata(&raw_filename)?.len();
    let png_size = std::fs::metadata(&png_filename)?.len();
    info!(
        "Captured frame saved as {} ({} bytes) and {} ({} bytes) in {:?}",
        raw_filename, raw_size, png_filename, png_size, elapsed
    );

    Ok(())
}

fn test_single_capture(device: &str) -> Result<()> {
    info!("Testing single frame capture from {}", device);

    let config = CameraConfig {
        device_path: device.to_string(),
        ..Default::default()
    };

    info!(
        "Requesting config: {}x{} @ {} Hz, gain={}, exposure={}",
        config.width, config.height, config.framerate, config.gain, config.exposure
    );

    let capture = V4L2Capture::new(config)?;

    let start = Instant::now();
    let capture_result = capture.capture_single_frame_with_info()?;
    let elapsed = start.elapsed();

    let fourcc_str = std::str::from_utf8(&capture_result.fourcc).unwrap_or("????");
    info!(
        "Captured frame: {} bytes, actual resolution: {}x{}, format: {}",
        capture_result.frame.len(),
        capture_result.actual_width,
        capture_result.actual_height,
        fourcc_str
    );

    // Save as raw
    let raw_filename = format!(
        "single_capture_{}x{}.raw",
        capture_result.actual_width, capture_result.actual_height
    );
    std::fs::write(&raw_filename, &capture_result.frame)?;
    info!("Saved raw frame as {}", raw_filename);

    // Save as PNG with actual dimensions
    let png_filename = format!(
        "single_capture_{}x{}.png",
        capture_result.actual_width, capture_result.actual_height
    );
    save_frame_as_png_from_capture(&capture_result, &png_filename)?;

    info!(
        "Single frame captured: {} bytes in {:?}, saved as {} and {}",
        capture_result.frame.len(),
        elapsed,
        raw_filename,
        png_filename
    );

    Ok(())
}

fn test_continuous_capture(device: &str, count: usize) -> Result<()> {
    info!(
        "Testing continuous capture of {} frames from {}",
        count, device
    );

    let config = CameraConfig {
        device_path: device.to_string(),
        ..Default::default()
    };

    let width = config.width;
    let height = config.height;
    let mut session = CaptureSession::new(&config)?;
    session.start_stream()?;

    let start = Instant::now();

    for i in 0..count {
        let frame_start = Instant::now();
        let (frame, _meta) = session.capture_frame()?;
        let frame_time = frame_start.elapsed();

        // Save first frame as PNG for verification
        if i == 0 {
            let png_filename = "continuous_first_frame.png";
            save_frame_as_png(&frame, width, height, png_filename)?;
            info!("First frame saved as {}", png_filename);
        }

        info!(
            "Frame {}/{}: {} bytes captured in {:?}",
            i + 1,
            count,
            frame.len(),
            frame_time
        );
    }

    let total_time = start.elapsed();
    let fps = count as f64 / total_time.as_secs_f64();

    info!(
        "Captured {} frames in {:?} ({:.2} fps)",
        count, total_time, fps
    );

    session.stop_stream();
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let device = std::env::var("VIDEO_DEVICE").unwrap_or_else(|_| "/dev/video0".to_string());

    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "single".to_string());

    match mode.as_str() {
        "single" => {
            test_single_capture(&device)?;
        }
        "continuous" => {
            let count = std::env::args()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);
            test_continuous_capture(&device, count)?;
        }
        "profiles" => {
            let profiles = ResolutionProfile::standard_profiles();
            for profile in &profiles {
                if let Err(e) = test_resolution_profile(&device, profile) {
                    error!("Failed to test profile: {}", e);
                }
            }
        }
        "custom" => {
            let width: u32 = std::env::args()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1024);
            let height: u32 = std::env::args()
                .nth(3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1024);
            let framerate: u32 = std::env::args()
                .nth(4)
                .and_then(|s| s.parse().ok())
                .unwrap_or(23_000_000);

            let profile = ResolutionProfile {
                width,
                height,
                framerate,
                test_frames: 10,
            };

            test_resolution_profile(&device, &profile)?;
        }
        _ => {
            eprintln!(
                "Usage: {} [mode] [options]",
                std::env::args().next().unwrap()
            );
            eprintln!("Modes:");
            eprintln!("  single            - Capture single frame");
            eprintln!("  continuous [n]    - Capture n frames (default: 10)");
            eprintln!("  profiles          - Test all standard resolution profiles");
            eprintln!("  custom w h fps    - Test custom resolution");
            eprintln!();
            eprintln!("Environment:");
            eprintln!("  VIDEO_DEVICE      - Device path (default: /dev/video0)");
        }
    }

    Ok(())
}

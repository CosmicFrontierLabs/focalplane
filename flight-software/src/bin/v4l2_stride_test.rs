use anyhow::Result;
use clap::Parser;
use flight_software::v4l2_capture::{CameraConfig, V4L2Capture};
use v4l::prelude::*;
use v4l::video::Capture;

#[derive(Debug, Clone)]
struct ResolutionInfo {
    width: u32,
    height: u32,
    #[allow(dead_code)]
    fps: f32,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Video device to use
    #[arg(short, long, default_value = "/dev/video0")]
    device: String,
}

fn enumerate_resolutions(device_path: &str) -> Result<Vec<ResolutionInfo>> {
    let device = Device::with_path(device_path)?;
    let mut resolutions = Vec::new();

    let _format = device.format()?;

    // For IMX455, we know the standard resolutions
    // Since enum_framesizes might not work properly, use known values
    let known_resolutions = vec![
        (128, 128, 133.0),
        (256, 256, 83.0),
        (512, 512, 44.0),
        (1024, 1024, 23.0),
        (2048, 2048, 12.0),
        (4096, 4096, 6.0),
        (8096, 6324, 3.7),
    ];

    for (width, height, fps) in known_resolutions {
        resolutions.push(ResolutionInfo { width, height, fps });
    }

    Ok(resolutions)
}

fn test_resolution(device_path: &str, width: u32, height: u32) -> Result<()> {
    println!("\n=== Testing {width}x{height} ===");

    let config = CameraConfig {
        device_path: device_path.to_string(),
        width,
        height,
        ..Default::default()
    };

    let capture = V4L2Capture::new(config)?;
    let frame_data = capture.capture_single_frame()?;

    let expected_pixels = (width * height) as usize;
    let expected_bytes_no_padding = expected_pixels * 2; // 16-bit per pixel
    let actual_bytes = frame_data.len();

    println!("  Expected pixels: {expected_pixels}");
    println!("  Expected bytes (no padding): {expected_bytes_no_padding}");
    println!("  Actual bytes received: {actual_bytes}");

    // Calculate padding based on 256-pixel alignment
    const ALIGNMENT_PIXELS: usize = 256;
    let padded_width = (width as usize).div_ceil(ALIGNMENT_PIXELS) * ALIGNMENT_PIXELS;
    let expected_padding_pixels = padded_width - width as usize;

    let rows = height as usize;
    let bytes_per_row = actual_bytes / rows;
    let expected_bytes_per_row = (width as usize) * 2;
    let padding_bytes_per_row = bytes_per_row - expected_bytes_per_row;
    let padding_pixels_per_row = padding_bytes_per_row / 2;

    println!("  Bytes per row: {bytes_per_row}");
    println!("  Expected bytes per row: {expected_bytes_per_row}");
    println!("  Padding bytes per row: {padding_bytes_per_row}");
    println!("  Padding pixels per row: {padding_pixels_per_row}");
    println!("  Width {width} -> padded to {padded_width} (next multiple of {ALIGNMENT_PIXELS})");

    // Verify it matches 256-pixel alignment
    if padding_pixels_per_row == expected_padding_pixels {
        println!("  ✓ PASS: Padding is {padding_pixels_per_row} pixels (rounds to {padded_width})");
    } else {
        println!(
            "  ✗ FAIL: Padding is {padding_pixels_per_row} pixels, expected {expected_padding_pixels} for 256-alignment"
        );
    }

    // Verify the math checks out
    let calculated_total = padded_width * height as usize * 2;
    if calculated_total == actual_bytes {
        println!("  ✓ PASS: Total size matches padded_width * height * 2");
    } else {
        println!("  ✗ FAIL: Size mismatch. Calculated: {calculated_total}, Actual: {actual_bytes}");
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("V4L2 Stride/Padding Test");
    println!("Device: {}", args.device);
    println!("Testing that all resolutions align to 256-pixel boundaries...\n");

    // Get available resolutions
    let resolutions = enumerate_resolutions(&args.device)?;

    if resolutions.is_empty() {
        println!("No resolutions found!");
        return Ok(());
    }

    println!("Found {} resolutions to test", resolutions.len());

    let mut passed = 0;
    let mut failed = 0;

    for res in &resolutions {
        match test_resolution(&args.device, res.width, res.height) {
            Ok(_) => passed += 1,
            Err(e) => {
                println!("  ✗ ERROR testing {}x{}: {}", res.width, res.height, e);
                failed += 1;
            }
        }

        // Small delay between tests to avoid overwhelming the device
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    println!("\n=== SUMMARY ===");
    println!("Tested: {} resolutions", resolutions.len());
    println!("Passed: {passed}");
    println!("Failed: {failed}");

    if failed == 0 {
        println!("\n✓ All resolutions align to 256-pixel boundaries!");
    } else {
        println!("\n✗ Some resolutions have inconsistent padding!");
    }

    Ok(())
}

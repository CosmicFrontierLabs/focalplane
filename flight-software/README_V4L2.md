# V4L2 Frame Capture Module

## Overview
V4L2 frame capture module for flight software. Provides interface to camera devices using Video4Linux2 API.

## Hardware Configuration
**IMPORTANT**: The Nvidia Orin system has a single IMX455 sensor at `/dev/video0`. Only one process can access the sensor at a time - ensure no other camera applications are running before starting capture.

### IMX455 Sensor Specifications
- **Sensor**: Sony IMX455 CMOS
- **Device Path**: `/dev/video0` (exclusive access)
- **Format**: RG16 (16-bit Bayer RGRG/GBGB pattern)
- **Max Resolution**: 8096x6324 @ 4 fps
- **Test Patterns**: Available via control ID 10100848

## Features
- Single frame capture
- Continuous streaming with frame buffering  
- Multiple resolution/framerate profiles
- Async frame buffer management
- Raw frame data output (16-bit grayscale)

## Usage

### Basic Frame Capture
```rust
use flight_software::v4l2_capture::{CameraConfig, V4L2Capture};

let config = CameraConfig::default();
let capture = V4L2Capture::new(config)?;
let frame = capture.capture_single_frame()?;
```

### Streaming Session
```rust
use flight_software::v4l2_capture::{CameraConfig, CaptureSession};

let config = CameraConfig::default();
let mut session = CaptureSession::new(&config)?;
session.start_stream()?;

let frame = session.capture_frame()?;
session.stop_stream();
```

### Test Binary
```bash
# Single frame capture
cargo run --bin v4l2_test single

# Continuous capture (10 frames)
cargo run --bin v4l2_test continuous 10

# Test all resolution profiles
cargo run --bin v4l2_test profiles

# Custom resolution test
cargo run --bin v4l2_test custom 2048 2048 12000000

# Use different device
VIDEO_DEVICE=/dev/video1 cargo run --bin v4l2_test single
```

## Resolution Profiles
Standard profiles matching your bash script:
- 128x128 @ 133MHz
- 256x256 @ 83MHz
- 512x512 @ 44MHz
- 1024x1024 @ 23MHz (default)
- 2048x2048 @ 12MHz
- 4096x4096 @ 6MHz
- 8096x6324 @ 3.7MHz

## Camera Controls
Configurable parameters:
- Gain (default: 360)
- Exposure (default: 140)
- Black level (default: 4095)
- Frame rate (device-specific)

## Frame Buffer
Async circular buffer for frame management:
```rust
use flight_software::v4l2_capture::FrameBuffer;

let buffer = FrameBuffer::new(100);
buffer.push(frame_data).await;
let latest = buffer.get_latest().await;
```
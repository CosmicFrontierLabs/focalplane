# Remote Testing on Nvidia Orin

This document describes how to run tests remotely on the Nvidia Orin device.

## Prerequisites

1. **Local machine requirements:**
   - Rust toolchain with `rustup`
   - ARM64 cross-compilation tools: `sudo apt install gcc-aarch64-linux-gnu`
   - SSH access to the Orin device

2. **Remote Orin requirements:**
   - SSH server running
   - User account with appropriate permissions
   - V4L2 camera device available (for V4L2 tests)

## Configuration

The test script connects to: `cosmicfrontiers@192.168.15.229`

Modify the `REMOTE_HOST` variable in `test-remote-orin.sh` if your device has a different address.

## Usage

### Basic usage (run all tests):
```bash
./test-remote-orin.sh
```

### Test modes:
```bash
# Run only unit tests
./test-remote-orin.sh --mode unit

# Run only V4L2 camera tests
./test-remote-orin.sh --mode v4l2

# Run only binary startup tests
./test-remote-orin.sh --mode bin

# Run all tests (default)
./test-remote-orin.sh --mode all
```

### Additional options:
```bash
# Skip building (use existing binaries)
./test-remote-orin.sh --skip-build

# Keep remote test directory for debugging
./test-remote-orin.sh --keep-remote

# Use a different video device
./test-remote-orin.sh --video-device /dev/video1

# Combine options
./test-remote-orin.sh --mode v4l2 --video-device /dev/video1 --keep-remote
```

## What the script does

1. **Build Phase**: Cross-compiles all binaries and tests for ARM64
2. **Setup Phase**: Creates a temporary directory on the remote Orin
3. **Transfer Phase**: Copies compiled binaries via SCP
4. **Test Phase**: Runs tests remotely via SSH
   - Unit tests (if any)
   - V4L2 camera tests (single capture, continuous capture, resolution profiles)
   - Binary startup tests
5. **Retrieval Phase**: Downloads any generated test artifacts
6. **Cleanup Phase**: Removes remote test directory (unless `--keep-remote` is used)

## Test Results

- Test artifacts (captured images, etc.) are saved to: `./remote-test-results/`
- Exit code indicates success (0) or failure (non-zero)
- Colored output shows progress and results

## V4L2 Tests

The V4L2 tests specifically test camera frame grabbing functionality:

1. **Single capture**: Captures one frame from the camera
2. **Continuous capture**: Captures multiple frames in sequence
3. **Resolution profiles**: Tests various standard resolutions

These tests are critical for verifying the integrated camera functionality on the Orin.

## Troubleshooting

### SSH connection fails
- Verify the Orin is reachable: `ping 192.168.15.229`
- Test SSH manually: `ssh cosmicfrontiers@192.168.15.229`
- Ensure SSH keys are set up for passwordless access

### Build fails
- Ensure cross-compilation tools are installed: `sudo apt install gcc-aarch64-linux-gnu`
- Run the setup script first: `./setup-arm64-cross-compilation.sh`

### V4L2 tests fail
- Verify camera device exists on Orin: `ssh cosmicfrontiers@192.168.15.229 "ls -la /dev/video*"`
- Check camera permissions: User must have access to video devices
- Try different video device: `--video-device /dev/video1`

### Tests timeout
- The Orin might be under heavy load
- Network latency might be high
- Consider running tests individually with `--mode` option

## Manual Testing

To manually test on the Orin after transferring files:

```bash
# SSH into the device
ssh cosmicfrontiers@192.168.15.229

# Navigate to test directory (check script output for exact path)
cd /home/cosmicfrontiers/flight-software-test-*

# Run V4L2 test manually
VIDEO_DEVICE=/dev/video0 ./v4l2_test single

# Run with different modes
./v4l2_test continuous 5
./v4l2_test profiles
./v4l2_test custom 1920 1080 30
```
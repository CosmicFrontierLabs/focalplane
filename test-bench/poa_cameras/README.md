# POA Cameras - PlayerOne Astronomy Camera Support

PlayerOne Astronomy camera support for the test bench. This package contains experimental and development software that is **not intended for flight**, unlike the `flight-software` package.

## Purpose

This package provides tools for:
- Testing and evaluating camera hardware (PlayerOne astronomy cameras, etc.)
- Prototyping new sensor integrations
- Development utilities and diagnostics
- Performance testing and benchmarking

## Binaries

### playerone_info

Enumerate and display properties of connected PlayerOne astronomy cameras.

```bash
# List all connected PlayerOne cameras
cargo run --bin playerone_info

# Show detailed properties
cargo run --bin playerone_info -- --detailed
```

## Building

### Prerequisites

The PlayerOne SDK native libraries are vendored in `../third_party/playerone-sdk/` for both ARM64 and x86_64 architectures.

### Local Build (x86_64)

For development/testing on x86_64:

```bash
# Ensure libusb-1.0 is installed
sudo apt-get install libusb-1.0-0-dev

# Build with x64 libs
RUSTFLAGS="-L $(pwd)/../../third_party/playerone-sdk/lib/x64" \
cargo build --release --package poa_cameras
```

### ARM64 Cross-Compilation (for Jetson Orin)

**Note**: Cross-compilation is complex due to ARM64 dependency requirements. For simplicity, consider building natively on the Orin (see "Native Build on Orin" below).

If you want to cross-compile from x86_64:

**Requirements:**
1. ARM64 cross-compiler toolchain
2. ARM64 libusb-1.0 (challenging on Ubuntu x86_64 systems)

```bash
# Install cross-compiler
sudo apt-get install gcc-aarch64-linux-gnu

# Attempt to install ARM64 libusb (may fail on x86_64-only repos)
sudo dpkg --add-architecture arm64
sudo apt-get update
sudo apt-get install libusb-1.0-0:arm64

# Build (if dependencies satisfied)
../../scripts/build-arm64.sh poa_cameras playerone_info
```

**Known Issue**: Ubuntu x86_64 repositories typically don't provide ARM64 packages, causing `apt-get install libusb-1.0-0:arm64` to fail with 404 errors.

### Native Build on Orin (Recommended)

Building directly on the Jetson Orin is simpler and avoids cross-compilation complexity:

```bash
# On Orin: Install Rust if not present
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# On Orin: Install libusb
sudo apt-get install libusb-1.0-0-dev

# Copy project to Orin
rsync -avz --exclude target --exclude .git \
  /path/to/meter-sim/ user@orin-ip:~/meter-sim/

# On Orin: Build
cd ~/meter-sim
cargo build --release --package poa_cameras
```

## Deployment to Jetson Orin

### Deploy and Run

```bash
# Deploy and run playerone_info
../../scripts/deploy-to-orin.sh --package poa_cameras --binary playerone_info --run './playerone_info --detailed'

# Deploy all binaries and keep on remote
../../scripts/deploy-to-orin.sh --package poa_cameras --keep-remote
```

### Environment Variables

- `ORIN_HOST` - Remote Orin hostname/IP (default: cosmicfrontiers@192.168.15.229)

## Dependencies

### PlayerOne SDK

**SDK Version**: 3.9.0 (vendored in `../../third_party/playerone-sdk/`)

The PlayerOne SDK native libraries are included in the repository under `third_party/playerone-sdk/` for:
- ARM64 (Jetson Orin, Raspberry Pi 4+)
- x86_64 (development machines)

**Runtime on Jetson Orin:**

Option 1: System installation (recommended for production):
```bash
# Install PlayerOne SDK libraries to system path
sudo cp third_party/playerone-sdk/lib/arm64/libPlayerOneCamera.so* /usr/local/lib/
sudo ldconfig

# Install USB permissions
sudo cp third_party/playerone-sdk/udev/*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Option 2: Runtime library path (for testing):
```bash
export LD_LIBRARY_PATH=/path/to/meter-sim/third_party/playerone-sdk/lib/arm64:$LD_LIBRARY_PATH
```

### System Dependencies

**libusb-1.0**: Required by PlayerOne SDK for USB camera communication

```bash
# On Orin or development machine
sudo apt-get install libusb-1.0-0-dev
```

## Non-Flight Software

**WARNING**: This package is for development and testing only. Code here should **not** be deployed to flight systems. For flight-qualified software, use the `flight-software` package.

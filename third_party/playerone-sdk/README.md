# PlayerOne Camera SDK

This directory contains the PlayerOne Camera SDK native libraries for Linux.

## Version

**SDK Version:** 3.9.0

## Source

Downloaded from [Player One Astronomy](https://player-one-astronomy.com/)

Original package: `PlayerOne_Camera_SDK_Linux_V3.9.0.tar.gz`

## Contents

```
playerone-sdk/
├── lib/
│   ├── arm64/          # ARM64 libs for Jetson Orin / Raspberry Pi 4+
│   │   └── libPlayerOneCamera.so.3.9.0
│   └── x64/            # x86_64 libs for development machines
│       └── libPlayerOneCamera.so.3.9.0
├── include/
│   ├── PlayerOneCamera.h      # Main C API header
│   └── ConvFuncs.h           # Conversion utilities
├── udev/               # USB device rules for Linux
├── license.txt         # SDK license terms
└── changelog.txt       # Version history
```

## Usage

### Building for ARM64 (Jetson Orin)

The build system will automatically use the ARM64 libraries when cross-compiling:

```bash
PLAYERONE_LIB_PATH=$(pwd)/third_party/playerone-sdk/lib/arm64 \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
cargo build --target aarch64-unknown-linux-gnu --release --package poa_cameras
```

### Building for x86_64 (Development)

```bash
PLAYERONE_LIB_PATH=$(pwd)/third_party/playerone-sdk/lib/x64 \
cargo build --release --package poa_cameras
```

### Runtime Setup

On the target system (Jetson Orin), either:

**Option 1: Install to system path**
```bash
sudo cp third_party/playerone-sdk/lib/arm64/libPlayerOneCamera.so* /usr/local/lib/
sudo ldconfig
```

**Option 2: Use LD_LIBRARY_PATH**
```bash
export LD_LIBRARY_PATH=/path/to/meter-sim/third_party/playerone-sdk/lib/arm64:$LD_LIBRARY_PATH
```

**Option 3: Copy binary with libraries**
When deploying binaries, copy the .so files alongside them and use `rpath` or `LD_LIBRARY_PATH`.

### USB Permissions (Linux)

To allow non-root access to PlayerOne cameras:

```bash
sudo cp udev/*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then reconnect the camera.

## License

See `license.txt` for the PlayerOne SDK license terms.

## Notes

- The SDK provides pre-compiled binaries only - no source code is available
- The `playerone-sdk` Rust crate (from crates.io) provides safe Rust bindings
- Version 3.9.0 includes support for latest PlayerOne camera models

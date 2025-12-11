# meter-sim

A space telescope simulation system written in Rust, focusing on star tracking and attitude determination for prototype space telescope stabilization.

## Mission

Meter-sim provides a simulator harness for satellite attitude determination algorithms that stabilize prototype space telescopes. The primary goal is to develop and test robust algorithms for maintaining precise telescope orientation using star tracking in various orbital conditions.

The simulator allows quantitative evaluation of tracking accuracy under realistic motion models, including spacecraft tumble, sensor noise, and optical distortions. This enables development of algorithms that can maintain sub-arcsecond pointing accuracy even under challenging conditions.

## Structure

The project is organized as a Rust workspace with multiple packages:

### Core Simulation & Analysis
- **simulator**: Space telescope optical and sensor simulation with realistic noise models, star field rendering, and sensor performance analysis tools
- **monocle**: Modular Orientation, Navigation & Optical Control Logic Engine - implements Fine Guidance System (FGS) tracking algorithms and attitude determination
- **monocle_harness**: Test harness and benchmarking tools for monocle tracking algorithms

### Hardware Drivers
- **hardware**: Hardware drivers for test bench equipment and flight hardware (Jetson Orin, cameras, gyros, PI stages)

### Test Infrastructure
- **test-bench**: AprilTag detection and calibration test bench
  - **poa_cameras**: PlayerOne Astronomy camera support for hardware testing (non-flight software)

### Test Infrastructure
- **test-bench**: AprilTag detection and calibration test bench
  - **display-util**: Display and visualization utilities for fullscreen pattern output

### Utilities
- **shared**: Shared utilities and common code across packages
- **test_helpers**: Common test utilities and fixtures

## Features

- Load and use the Hipparcos and Gaia star catalogs
- Simulate telescope optics and sensor behavior with realistic noise models
- Implement and benchmark real-time star tracking algorithms
- Test attitude determination accuracy under various conditions
- Estimate segmentation and algorithm accuracy under different simulated motion models
- Evaluate performance with various jitter, drift, and tumble scenarios
- Iterative Closest Point (ICP) implementation for pattern matching

## Interesting Reference Implementations

- ROMAN Telescope Simulation Project: https://github.com/spacetelescope/romanisim
- Pandora Telescope Simulator: https://github.com/PandoraMission/pandora-sim

## Getting Started

### Building
```bash
# Build all packages
cargo build --release

# Build specific package
cargo build --release --package simulator
cargo build --release --package monocle
```

### Simulation & Analysis

```bash
# Compare sensor performance across different models
cargo run --bin sensor_shootout -- --experiments 100

# Single-shot debug mode pointing at the Pleiades cluster
cargo run --bin sensor_shootout -- --single-shot-debug "56.75,24.12" --experiments 1

# Estimate detection performance across magnitudes and sensors
cargo run --bin sensor_floor_est --package simulator

# Test centroid accuracy across sub-pixel positions
cargo run --bin centroid_accuracy_test --package simulator

# Generate analysis plots
cargo run --bin camera_qe_plot --package simulator
cargo run --bin stellar_color_plot --package simulator
```

### Tracking & FGS

```bash
# Run FGS tracking demo with different motion patterns
cargo run --bin tracking_demo --motion sine_ra --duration 20
cargo run --bin tracking_demo --motion circular --frame-rate 30
cargo run --bin tracking_demo --motion chaotic --duration 30 --verbose

# Benchmark FGS performance
cargo run --bin fgs_shootout --package monocle_harness
```

### Orin Monitoring (Test Infrastructure)

```bash
# Run Orin monitoring server with Prometheus metrics
cargo run --bin orin_monitor --package test-bench

# Deploy to Orin
scripts/deploy-to-orin.sh --package test-bench --binary orin_monitor

# PlayerOne camera tools
cargo run --bin playerone_info --package test-bench --features playerone
```

## Logging

The project uses `env_logger` for configurable logging output. Control log levels by setting the `RUST_LOG` environment variable:

```bash
# Show all logs (very verbose)
RUST_LOG=debug cargo run --bin sensor_shootout

# Show info and higher priority logs (recommended)
RUST_LOG=info cargo run --bin sensor_shootout

# Show only warnings and errors (minimal output)
RUST_LOG=warn cargo run --bin sensor_shootout

# Show logs for specific modules only
RUST_LOG=sensor_shootout=debug cargo run --bin sensor_shootout

# Multiple modules with different levels
RUST_LOG=sensor_shootout=info,simulator::image_proc=debug cargo run --bin sensor_shootout
```

Available log levels (in order of priority):
- `error`: Critical errors only
- `warn`: Warnings and errors
- `info`: General information, warnings, and errors (recommended default)
- `debug`: Detailed debugging information (very verbose)
- `trace`: Extremely detailed tracing information (rarely needed)

### Test Bench

The test bench includes web-based frontends for camera monitoring and calibration pattern generation. These frontends are built with Yew (Rust WebAssembly framework).

#### Building the Frontends

**First time setup:**
```bash
# Install trunk (WASM build tool) - only needed once
cargo install --locked trunk

# Build the frontends
./scripts/build-yew-frontends.sh
```

The build script compiles the Yew applications in `test-bench-frontend/` and outputs to `test-bench-frontend/dist/`. Both servers will automatically serve these files.

**If you see errors about missing WASM files**, run the build script above.

#### Running the Servers

```bash
# Calibration pattern server
cargo run --bin calibrate_serve
# Opens web UI at http://localhost:3001

# Camera monitoring server with mock camera
cargo run --bin cam_serve -- --camera-type mock
# Opens web UI at http://localhost:3000

# Camera server with real hardware
cargo run --bin cam_serve -- --camera-type playerone  # Or other camera types
```

See [Yew Frontend Migration Guide](docs/yew-frontend-migration.md) for details on the WebAssembly-based frontends.

## Documentation

Generate and view the documentation with:

```bash
cargo doc --open
```
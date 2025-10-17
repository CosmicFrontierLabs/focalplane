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

### Flight Hardware
- **flight-software**: Production monitoring system for NVIDIA Jetson Orin flight computers with Prometheus/Grafana integration for power, thermal, and system telemetry
- **orin-dev**: Development and testing tools for Jetson Orin platform including PlayerOne camera integration (non-flight software)

### Utilities
- **shared**: Shared utilities and common code across packages
- **display-util**: Display and visualization utilities for fullscreen output
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

### Flight Hardware (Jetson Orin)

```bash
# Run flight monitor locally
cargo run --bin flight_monitor --package flight-software

# Or use Docker stack with Prometheus/Grafana
cd flight-software && docker-compose up -d

# PlayerOne camera tools (orin-dev package)
cargo run --bin playerone_info --package orin-dev

# Deploy to Orin
scripts/deploy-to-orin.sh --package flight-software --binary flight_monitor
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

## Documentation

Generate and view the documentation with:

```bash
cargo doc --open
```
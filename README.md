# meter-sim

A space telescope simulation system written in Rust, focusing on star tracking and attitude determination for prototype space telescope stabilization.

## Mission

Meter-sim provides a simulator harness for satellite attitude determination algorithms that stabilize prototype space telescopes. The primary goal is to develop and test robust algorithms for maintaining precise telescope orientation using star tracking in various orbital conditions.

The simulator allows quantitative evaluation of tracking accuracy under realistic motion models, including spacecraft tumble, sensor noise, and optical distortions. This enables development of algorithms that can maintain sub-arcsecond pointing accuracy even under challenging conditions.

## Structure

The project is organized as a Rust workspace with multiple packages:

- **simulator**: Space telescope optical and sensor simulation
- **track**: Star tracking and attitude determination algorithms
- **viz**: Visualization tools for simulation output

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

```bash
# Build all packages
cargo build

# Run scope view example
cargo run --example scope_view --package simulator
```

## Documentation

Generate and view the documentation with:

```bash
cargo doc --open
```
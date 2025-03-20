# meter-sim

A space telescope simulation system written in Rust, focusing on star tracking and attitude determination.

## Structure

The project is organized as a Rust workspace with multiple packages:

- **starfield**: Astronomical calculations library (inspired by Python's skyfield)
- **ephemeris**: Celestial body position and motion calculations
- **simulator**: Space telescope optical and sensor simulation
- **track**: Star tracking and attitude determination algorithms

## Features

- Load and use the Hipparcos star catalog
- Perform celestial mechanics calculations
- Simulate telescope optics and sensor behavior
- Implement real-time star tracking algorithms

## Interesting Reference Implementations

- ROMAN Telescope Simulation Project: https://github.com/spacetelescope/romanisim
- Pandora Telescope Simulator: https://github.com/PandoraMission/pandora-sim

## Getting Started

```bash
# Build all packages
cargo build

# Run the Hipparcos catalog example
cargo run --example hipparcos --package starfield
```

## Documentation

Generate and view the documentation with:

```bash
cargo doc --open
```
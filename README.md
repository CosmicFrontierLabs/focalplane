# focalplane

Space telescope imaging and sensor simulation, written in Rust.

Simulates the full optical path from star catalog to detector output, including:
- Telescope optics and point spread functions
- Sensor noise models (read noise, dark current, photon noise)
- Star field rendering from Hipparcos/Gaia catalogs
- Photometry and spectral response
- FITS file I/O

## Building

```bash
# Build
cargo build --release

# Test
cargo test
```

## Binaries

```bash
# Compare sensor performance across different models
cargo run --release --bin sensor_shootout -- --experiments 100

# Single-shot debug mode pointing at the Pleiades
cargo run --release --bin sensor_shootout -- --single-shot-debug "56.75,24.12" --experiments 1
```

## Dependencies

Foundation crates from [cfl-foundations](https://github.com/CosmicFrontierLabs/cfl-foundations):
- `meter-math` - quaternions, ICP, interpolation
- `shared` - image processing, camera traits, frame writing
- `shared-wasm` - shared serialization types

# Repository Split Plan

This document outlines a potential strategy for splitting the meter-sim monorepo into multiple repositories or published crates.

## Current Architecture

The workspace contains 8 crates with the following dependency structure:

```
                    ┌─────────────────────────────────────────────────────┐
                    │                    shared (HUB)                      │
                    │  23K LOC - Image proc, algorithms, camera interface  │
                    │  No internal deps, only external crates              │
                    └──────────┬──────────────────────┬────────────────────┘
                               │                      │
                         ┌─────▼─────┐         ┌──────▼──────┐
                         │ simulator │         │  monocle    │
                         │ 19K LOC   │         │  2.7K LOC   │
                         │ (optical) │         │  (FGS algo) │
                         └─────┬─────┘         └──────┬──────┘
                               │                      │
                         ┌─────▼──────────────────────▼─────┐
                         │           test-bench             │
                         │           10.6K LOC              │
                         │  (server, calibration, binaries) │
                         └──────────────┬───────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
              ┌─────▼────┐    ┌─────────▼─────────┐   ┌─────▼─────┐
              │ hardware │    │ test-bench-shared │   │ frontend  │
              │ 10K LOC  │    │    (minimal)      │   │  (WASM)   │
              └──────────┘    └───────────────────┘   └───────────┘

Additional:
- monocle_harness: bridges simulator + monocle for testing, includes protocol definitions (proto_control module)
```

---

## The `shared` Problem: A Hub That Does Too Much

The `shared` crate is the largest coupling point in the codebase. At 23K LOC, it contains:

| Module | LOC | Users | Purpose | Extraction Difficulty |
|--------|-----|-------|---------|----------------------|
| `algo/icp.rs` | 2K | monocle, simulator | Iterative Closest Point pattern matching | Easy |
| `algo/quaternion.rs` | 1K | simulator | Rotation math | Easy |
| `algo/spline.rs` | 1K | simulator | Interpolation | Easy |
| `algo/matrix2.rs` | 500 | test-bench | 2D transform math | Easy |
| `algo/psd.rs` | 2K | ? | Power spectral density | Easy |
| `image_proc/detection/` | 4K | monocle, simulator | Star detection | Medium |
| `image_proc/centroid.rs` | 1K | monocle, test-bench | Sub-pixel centroiding | Medium |
| `image_proc/airy.rs` | 2K | simulator, test-bench | Point Spread Function | Medium |
| `image_proc/noise/` | 2K | simulator | Sensor noise models | Medium |
| `camera_interface/` | 2K | monocle, test-bench | Camera abstraction traits | Medium |
| `units.rs` | 500 | simulator | Physical units (via UOM) | Easy |
| `zmq.rs` | 500 | test-bench only | ZeroMQ helpers | Easy (feature gate) |
| `pattern_client.rs` | 500 | test-bench only | Pattern generator client | Easy (feature gate) |
| `frame_writer.rs` | 500 | test-bench only | FITS file writing | Easy (feature gate) |
| `tracking_*` | 1K | test-bench only | Telemetry collection | Easy (feature gate) |

### Key Insight: Different Consumers Need Different Subsets

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WHO NEEDS WHAT FROM SHARED                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  monocle (FGS Algorithm)                                                    │
│  ├── camera_interface (traits only)                                         │
│  ├── image_proc/detection                                                   │
│  ├── image_proc/centroid                                                    │
│  └── bad_pixel_map                                                          │
│                                                                             │
│  simulator (Optics Simulation)                                              │
│  ├── units                                                                  │
│  ├── algo/* (ICP, quaternion, spline, bilinear)                             │
│  ├── image_proc/airy                                                        │
│  ├── image_proc/noise                                                       │
│  ├── image_proc/detection (for validation)                                  │
│  └── frame_writer (for output)                                              │
│                                                                             │
│  test-bench (Integration)                                                   │
│  ├── EVERYTHING (it's the integration layer)                                │
│  └── Plus ZMQ, tracking_collector, pattern_client                           │
│                                                                             │
│  hardware (Drivers)                                                         │
│  └── camera_interface (trait impl only)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Natural Domain Boundaries

### Domain 1: Physics Simulation (`simulator`)
**What it is**: Optical system simulation for testing algorithms without real hardware.

**Contains**:
- Telescope optics model (aperture, focal length, obscuration)
- Sensor physics (quantum efficiency, read noise, dark current)
- Photometry (star magnitudes, atmospheric effects)
- Scene rendering (star fields, satellites, cosmic rays)
- 13 analysis binaries for research

**Dependencies**: `shared` (algo, image_proc, units)

**Publishability**: Medium - Useful for optical system design, but tightly coupled to `shared`.

### Domain 2: Tracking Algorithm (`monocle`)
**What it is**: Fine Guidance System state machine and tracking algorithms.

**Contains**:
- Star detection and guide star selection
- Kalman filtering for position estimation
- Multi-star tracking state machine
- SNR-based filtering and quality metrics

**Dependencies**: `shared` (camera_interface traits, image_proc, bad_pixel_map)

**Publishability**: High - Clean algorithm boundary, valuable standalone. Could be `meter-fgs` on crates.io.

### Domain 3: Hardware Drivers (`hardware`)
**What it is**: Vendor-specific hardware interfaces.

**Structure by vendor**:
```
hardware/
├── exail/          # Exail gyroscope (serial protocol)
├── pi/             # PI GCS controllers (FSM, stages)
│   ├── e727/       # E-727 piezo controller
│   └── s330/       # S-330 tip-tilt stage
├── nsv455/         # NSV455 camera (V4L2)
├── poa/            # PlayerOne astronomy camera (SDK)
├── orin/           # Jetson Orin GPIO/monitoring
├── ftdi/           # FTDI serial adapters
└── exolambda/      # ExoLambda packet protocol
```

**Current features**:
- `playerone` - PlayerOne SDK (optional)
- `linux-bins` - Linux-only test binaries

**Proposed features** (not yet implemented):
- `nsv455` - NSV455 V4L2 camera
- `exail` - Exail gyroscope
- `pi-stage` - PI controllers

**Publishability**: Low individually - Very hardware-specific, but feature-gating helps.

### Domain 4: Integration (`test-bench`)
**What it is**: Ties everything together for real hardware testing.

**Contains**:
- `fgs_server` - Main Fine Guidance System server binary
- `calibration_controller` - Optical calibration routines
- Camera initialization and server infrastructure
- AprilTag detection for alignment
- FSM calibration (new)

**Dependencies**: ALL other crates

**Publishability**: Low - This is the application layer, not a library.

### Domain 5: Protocol Types (now in `monocle_harness::proto_control`)
**What it is**: Message types and coordinate definitions.

**Contains**:
- Timestamp types
- Gyro data structures
- FSM command types
- FGS message types
- StateEstimator trait

**Dependencies**: ZERO external deps (pure Rust types)

**Location**: Merged into `monocle_harness` as the `proto_control` module. Re-exported at crate root for convenience.

---

## Recommended Strategy: Layered Architecture

### Layer 0: Primitives (publish to crates.io)

These are self-contained, reusable components with no internal dependencies.

```
meter-types                          meter-math
├── units.rs (via UOM wrapper)       ├── quaternion.rs
├── coordinates.rs                   ├── icp.rs (pattern matching)
└── timestamps.rs                    ├── spline.rs
                                     ├── bilinear.rs
                                     ├── matrix2.rs
                                     └── stats.rs

meter-image
├── detection/
│   ├── unified.rs (main detector)
│   ├── thresholding.rs
│   └── aabb.rs
├── centroid.rs
├── convolve2d.rs
├── airy.rs (PSF)
└── noise/
    ├── generate.rs
    └── quantify.rs
```

### Layer 1: Domain Crates

```
meter-optics (from simulator)
├── Depends on: meter-types, meter-math, meter-image
├── Contains: telescope, sensor, photometry, scene rendering
└── Binaries: sensor_shootout, etc.

meter-fgs (from monocle)
├── Depends on: meter-types, meter-math, meter-image
├── Contains: state machine, filters, selection
└── API: FGS tracking algorithm only

meter-drivers (from hardware)
├── Depends on: meter-types
├── Feature flags: exail, pi-stage, nsv455, playerone, jetson
└── Each driver optional
```

### Layer 2: Integration (stays as application repo)

```
meter-testbench
├── Depends on: ALL layer 0+1 crates
├── Contains: servers, calibration, harness
└── Binaries: fgs_server, calibration_controller
```

---

## Migration Path

### Phase 1: Feature-Gate `shared` (Immediate, Low Risk)

This reduces compile times and makes dependencies explicit without restructuring.

**Step 1.1**: Add feature flags to `shared/Cargo.toml`:
```toml
[features]
default = ["full"]
full = ["zmq-support", "tracking", "pattern-client", "frame-writer"]
zmq-support = ["zmq"]
tracking = []
pattern-client = ["zmq-support"]
frame-writer = []
image-proc = []
algo = []
```

**Step 1.2**: Gate modules behind features:
```rust
#[cfg(feature = "zmq-support")]
pub mod zmq;

#[cfg(feature = "tracking")]
pub mod tracking_collector;
pub mod tracking_message;

#[cfg(feature = "pattern-client")]
pub mod pattern_client;
```

**Step 1.3**: Update dependent crates:
```toml
# monocle/Cargo.toml - only needs image processing
shared = { version = "0.1.0", path = "../shared", default-features = false, features = ["image-proc", "algo"] }

# simulator/Cargo.toml - needs more
shared = { version = "0.1.0", path = "../shared", default-features = false, features = ["image-proc", "algo", "frame-writer"] }

# test-bench/Cargo.toml - needs everything
shared = { version = "0.1.0", path = "../shared" }
```

**Outcome**: Faster incremental builds, explicit dependencies, groundwork for extraction.

### Phase 2: Feature-Gate Hardware Drivers (Medium Risk)

**Current state**: NSV455 driver always compiled on Linux, PlayerOne is optional.

**Step 2.1**: Add features to `hardware/Cargo.toml`:
```toml
[features]
default = []
nsv455 = []  # NEW: Make NSV455 optional
playerone = ["dep:playerone-sdk"]  # Existing
exail = []   # NEW: Make Exail gyro optional
pi-stage = [] # NEW: Make PI controllers optional
```

**Step 2.2**: Gate driver modules:
```rust
#[cfg(feature = "nsv455")]
pub mod nsv455;

#[cfg(feature = "exail")]
pub mod exail;

#[cfg(feature = "pi-stage")]
pub mod pi;
```

**Outcome**: Can build for specific hardware targets, smaller binaries.

### Phase 3: ~~Extract `proto-control`~~ (COMPLETED)

**Status**: Merged into `monocle_harness::proto_control` module.

The protocol types are now available via:
```rust
use monocle_harness::{Timestamp, GyroReadout, FgsReadout, FsmCommand, StateEstimator};
// or
use monocle_harness::proto_control::*;
```

**Outcome**: Simplified workspace, protocol types co-located with harness.

### Phase 4: Extract Math Primitives (Medium Risk)

**Step 4.1**: Create `meter-math` crate with:
- `quaternion.rs` from `shared/src/algo/`
- `icp.rs` (pattern matching algorithm)
- `spline.rs`, `bilinear.rs`
- `matrix2.rs`, `stats.rs`

**Step 4.2**: Publish with stable API.

**Step 4.3**: Update `shared` to re-export from `meter-math`:
```rust
pub use meter_math as algo;
```

**Outcome**: Core algorithms available to external projects.

### Phase 5: Extract Image Processing (Higher Risk)

**Step 5.1**: Create `meter-image` crate with:
- `detection/` module
- `centroid.rs`
- `airy.rs` (PSF)
- `noise/` module

**Step 5.2**: Handle dependencies carefully - this touches monocle and simulator.

**Outcome**: Image processing algorithms available standalone.

### Phase 6: Extract FGS Algorithm (Highest Value)

**Step 6.1**: Publish `monocle` as `meter-fgs`:
```toml
[package]
name = "meter-fgs"
description = "Fine Guidance System tracking algorithm"
```

**Step 6.2**: Define stable public API:
```rust
// Public API surface
pub struct FGS { ... }
pub struct FGSConfig { ... }
pub trait CameraSource { ... }
pub enum FGSState { ... }
pub struct TrackingResult { ... }
```

**Outcome**: High-value IP available for external integrations.

---

## Decision Matrix

| Factor | Keep Monorepo | Split Repos | Publish Crates |
|--------|--------------|-------------|----------------|
| Team size | Small (1-3) | Multiple teams | Community |
| Release cadence | Tight coupling | Independent | Semver required |
| External users | None | Forks needed | Yes |
| Build times | Acceptable | Slow CI | Faster downstream |
| Code sharing | Maximum | Via packages | Via crates.io |

### Current Recommendation: Hybrid Approach

1. **Stay monorepo** for tight iteration (test-bench + hardware)
2. **Feature-gate** for modularity without splitting
3. **Publish** `meter-fgs` (monocle) and `meter-math` for external value
4. **Defer** full split until there are actual external consumers

---

## Immediate Low-Hanging Fruit

These can be done today with minimal risk:

1. ~~**Feature-gate ZMQ in `shared`**~~ ✅ Done (PR #580)
2. **Feature-gate NSV455 in `hardware`** - Not needed for simulation
3. ~~**Extract `proto-control`**~~ ✅ Merged into `monocle_harness`
4. **Document public API** for `monocle` - Prepare for publication

---

## Open Questions

1. **Naming convention**: `meter-*` vs `cfl-*` vs something else?
2. **Registry choice**: crates.io (public) vs private registry (internal)?
3. **Versioning policy**: Lockstep versions or independent semver?
4. **Monorepo tooling**: If keeping together, use `cargo-workspaces`?
5. **CI implications**: How to handle optional features in GitHub Actions?

---

## Appendix: Dependency Graph Detail

```
shared (23K LOC)
├── algo/ (12K LOC)
│   ├── icp.rs ─────────────► monocle, simulator
│   ├── quaternion.rs ──────► simulator
│   ├── spline.rs ──────────► simulator
│   ├── bilinear.rs ────────► simulator
│   ├── matrix2.rs ─────────► test-bench (FSM calibration)
│   ├── motion.rs ──────────► simulator
│   ├── psd.rs ─────────────► (analysis tools)
│   └── stats.rs ───────────► simulator
│
├── image_proc/ (10K LOC)
│   ├── detection/ ─────────► monocle, simulator
│   ├── centroid.rs ────────► monocle, test-bench
│   ├── airy.rs ────────────► simulator, test-bench
│   ├── noise/ ─────────────► simulator
│   ├── convolve2d.rs ──────► simulator
│   └── overlay.rs ─────────► test-bench
│
├── camera_interface/ ──────► monocle, hardware (traits)
├── units.rs ───────────────► simulator
├── zmq.rs ─────────────────► test-bench ONLY
├── pattern_client.rs ──────► test-bench ONLY
├── frame_writer.rs ────────► simulator, test-bench
└── tracking_*.rs ──────────► test-bench ONLY
```

This graph shows that `test-bench` is the only consumer of several modules, making them good candidates for feature-gating or moving into `test-bench` itself.

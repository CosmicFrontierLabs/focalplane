# Repository Split Plan

This document outlines a potential strategy for splitting the meter-sim monorepo into multiple repositories or published crates.

## Current Architecture

The workspace contains 9 crates with the following dependency structure:

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
- monocle_harness: bridges simulator + monocle for testing
- proto-control: standalone protocol definitions (~100 LOC, zero deps)
```

## Natural Domain Boundaries

### Domain 1: Physics Simulation (`simulator`)
- Optics, sensors, photometry, celestial mechanics
- 13 analysis binaries for research/tuning
- Self-contained with only `shared` dependency

### Domain 2: Tracking Algorithm (`monocle`)
- Fine Guidance System state machine
- Kalman filtering, guide star selection
- Pure algorithm, minimal dependencies

### Domain 3: Hardware Drivers (`hardware`)
- Exail gyroscope, PI FSM/stage, NSV455 camera, PlayerOne camera, Jetson GPIO
- Platform-specific (Linux V4L2, USB serial)
- Modular per hardware vendor

### Domain 4: Integration (`test-bench`)
- Server binaries, calibration routines
- Ties everything together
- Highest coupling, most dependencies

### Domain 5: Protocol (`proto-control`)
- Timestamp, gyro, FSM, FGS message types
- Zero internal dependencies
- Could be extracted immediately

## The `shared` Problem

The `shared` crate is doing too much:

| Module | Used By | Could Be Separate |
|--------|---------|-------------------|
| `image_proc/` | simulator, test-bench | Yes - `meter-sim-image` |
| `algo/` | monocle, simulator | Yes - `meter-sim-math` |
| `camera_interface/` | monocle, test-bench | Yes - trait crate |
| `zmq.rs`, `pattern_client.rs` | test-bench only | Yes - feature flag |
| `frame_writer.rs` | test-bench only | Yes - feature flag |

## Recommended Strategy: Layered Architecture

### Layer 0: Primitives (publish to crates.io)

```
meter-sim-types
├── units.rs (Length, Temperature, etc.)
├── coordinates (from proto-control)
└── timestamps
```

```
meter-sim-math
├── quaternion.rs
├── icp.rs (pattern matching)
├── spline.rs, bilinear.rs
└── matrix2.rs
```

```
meter-sim-image
├── detection/ (star finding)
├── centroid.rs
├── convolve2d.rs
└── airy.rs (PSF)
```

### Layer 1: Domain Crates

```
meter-sim-optics (simulator)
├── Depends on: meter-sim-types, meter-sim-math, meter-sim-image
└── Contains: telescope, sensor, photometry, scene rendering

meter-sim-fgs (monocle)
├── Depends on: meter-sim-types, meter-sim-math
└── Contains: state machine, filters, controllers

meter-sim-drivers (hardware)
├── Depends on: meter-sim-types
├── Feature flags per hardware: exail, pi-stage, nsv455, playerone, jetson
└── Each driver optional
```

### Layer 2: Integration (stays as monorepo or single repo)

```
meter-sim-testbench
├── Depends on: ALL layer 0+1 crates
└── Contains: servers, calibration, harness
```

## Migration Path

### Phase 1: Extract Primitives (Low Risk)
1. Create `meter-sim-types` from `shared/units.rs` + `proto-control`
2. Create `meter-sim-math` from `shared/algo/`
3. Publish to crates.io or internal registry
4. Update internal imports

### Phase 2: Feature-Gate Hardware (Medium Risk)
1. Make hardware drivers optional features in `hardware` crate
2. `nsv455` feature for NSV455 camera
3. `playerone` feature for PlayerOne camera
4. `exail` feature for Exail gyro
5. `pi-stage` feature for PI equipment

### Phase 3: Publish Algorithm (Medium Risk)
1. Extract `monocle` as `meter-sim-fgs`
2. Publish with stable API
3. Keep `monocle_harness` in monorepo (test tooling)

### Phase 4: Split Simulation (Higher Risk)
1. Extract `simulator` as `meter-sim-optics`
2. Extract `shared/image_proc/` as `meter-sim-image`
3. Requires careful API design

## Decision Criteria

### Keep as Monorepo If:
- Single team with tight iteration cycles
- All code changes together frequently
- No external consumers of library code

### Split Repos If:
- Multiple teams need independent release cycles
- External projects want to use FGS algorithm
- Hardware drivers need different CI/build requirements
- Simulation needs to run without hardware dependencies

### Publish Crates If:
- Want to share algorithms with broader community
- Need semver guarantees for internal consumers
- Want to reduce build times for downstream users

## Immediate Low-Hanging Fruit

1. **Feature-gate ZMQ in `shared`** - Only test-bench uses it
2. **Extract `proto-control`** - Zero deps, immediate extraction
3. **Feature-gate hardware drivers** - Avoid pulling camera SDKs for simulation-only builds
4. **Publish `monocle` as library** - Clean API, valuable IP

## Open Questions

1. Should hardware drivers be one crate with features or separate crates?
2. Internal registry (Artifactory/GitLab) vs crates.io for publishing?
3. Semver policy for internal crates?
4. Monorepo tooling (cargo-workspaces, release-plz) if staying together?

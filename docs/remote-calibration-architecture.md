# Remote Calibration System Architecture

## Overview

Desktop-driven optical calibration system that remotely controls the OLED display pattern while collecting tracking feedback from the camera system. Produces real-time visualization of alignment, sensor coverage, and focus quality.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              DESKTOP                                    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                  calibration_controller (NEW)                     │  │
│  │                                                                   │  │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │  │
│  │   │ Pattern     │    │ Tracking    │    │ Calibration Engine  │  │  │
│  │   │ Commander   │    │ Listener    │    │                     │  │  │
│  │   │ (ZMQ PUB)   │    │ (ZMQ SUB)   │    │ - Point collection  │  │  │
│  │   └──────┬──────┘    └──────┬──────┘    │ - Transform est.    │  │  │
│  │          │                  │           │ - Coverage calc     │  │  │
│  │          │                  │           │ - Defocus mapping   │  │  │
│  │          │                  │           └──────────┬──────────┘  │  │
│  │          │                  │                      │             │  │
│  │          │                  └──────────────────────┤             │  │
│  │          │                                         │             │  │
│  │          │              ┌──────────────────────────┴──────────┐  │  │
│  │          │              │        egui Visualization           │  │  │
│  │          │              │                                     │  │  │
│  │          │              │  - Live spot overlay                │  │  │
│  │          │              │  - Transform parameters             │  │  │
│  │          │              │  - Sensor coverage polygon          │  │  │
│  │          │              │  - Defocus heatmap                  │  │  │
│  │          │              │  - Residual error plot              │  │  │
│  │          │              └─────────────────────────────────────┘  │  │
│  └──────────┼───────────────────────────────────────────────────────┘  │
│             │                                                           │
└─────────────┼───────────────────────────────────────────────────────────┘
              │ PatternCommand (ZMQ)
              ▼
┌─────────────────────────┐
│      TEST-BENCH-PI      │
│                         │
│  ┌───────────────────┐  │
│  │  calibrate_serve  │  │
│  │                   │  │
│  │  ZMQ SUB ◄────────┼──┼─── PatternCommand
│  │                   │  │
│  │  ┌─────────────┐  │  │
│  │  │RemoteCtrl'd │  │  │
│  │  │Pattern (NEW)│  │  │
│  │  └─────────────┘  │  │
│  │                   │  │
│  │  HTTP API (existing)  │
│  └───────────────────┘  │
│           │             │
│        OLED 2560x2560   │
└───────────┬─────────────┘
            │ (optical path)
            ▼
┌─────────────────────────┐
│         ORIN            │
│                         │
│  ┌───────────────────┐  │
│  │    cam_track      │  │
│  │                   │  │
│  │  ZMQ PUB ─────────┼──┼───► TrackingMessage
│  │                   │  │
│  │  (future: add     │  │
│  │   FWHM to msg)    │  │
│  └───────────────────┘  │
│           ▲             │
│        Camera           │
└─────────────────────────┘
```

## Components

### Track 1: Shared Message Types (shared crate)

**Files**: `shared/src/pattern_command.rs`

**Dependencies**: None (can start immediately)

```rust
/// Commands sent from desktop to control OLED pattern display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternCommand {
    /// Display a single Gaussian spot
    Spot {
        x: f64,
        y: f64,
        fwhm: f64,
        intensity: f64,
    },
    /// Display multiple spots simultaneously
    SpotGrid {
        positions: Vec<(f64, f64)>,
        fwhm: f64,
        intensity: f64,
    },
    /// Uniform gray level
    Uniform { level: u8 },
    /// Clear to black
    Clear,
}

/// Response from calibrate_serve acknowledging command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAck {
    pub command_id: u64,
    pub timestamp_us: u64,
}
```

**Deliverable**: Message types in shared crate with serialization tests

---

### Track 2: Remote-Controlled Pattern (test-bench crate)

**Files**:
- `test-bench/src/display_patterns/remote_controlled.rs`
- `test-bench/src/calibrate/pattern.rs` (add variant)
- `test-bench/src/bin/calibrate_serve.rs` (add ZMQ listener)

**Dependencies**: Track 1 (PatternCommand types)

**New Pattern Variant**:
```rust
// In PatternConfig enum
#[serde(skip)]
RemoteControlled {
    command_receiver: Arc<Mutex<RemotePatternState>>,
    display_size: PixelShape,
}
```

**RemotePatternState**:
```rust
pub struct RemotePatternState {
    subscriber: TypedZmqSubscriber<PatternCommand>,
    current_command: PatternCommand,
    last_update: Instant,
}

impl RemotePatternState {
    pub fn poll(&mut self) {
        // Drain ZMQ, keep latest command
        for cmd in self.subscriber.drain() {
            self.current_command = cmd;
            self.last_update = Instant::now();
        }
    }

    pub fn current(&self) -> &PatternCommand {
        &self.current_command
    }
}
```

**calibrate_serve changes**:
- Add `--pattern-zmq-sub` argument for command endpoint
- When RemoteControlled pattern active, poll for commands each frame
- Render based on current command

**Deliverable**: calibrate_serve can display spots at positions commanded via ZMQ

---

### Track 3: Extended Tracking Message (optional, for defocus)

**Files**:
- `shared/src/tracking_message.rs`
- `test-bench/src/bin/cam_track.rs`

**Dependencies**: None (can start immediately)

**Extended TrackingMessage**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingMessage {
    pub track_id: u32,
    pub x: f64,
    pub y: f64,
    pub timestamp: Timestamp,

    // NEW: spot shape for defocus mapping
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fwhm_x: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fwhm_y: Option<f64>,
}
```

**cam_track changes**:
- Extract spot FWHM from FGS or centroiding algorithm
- Populate fwhm_x/fwhm_y in TrackingMessage

**Deliverable**: TrackingMessage includes measured spot size

---

### Track 4: Calibration Controller Binary (new)

**Files**: `test-bench/src/bin/calibration_controller.rs`

**Dependencies**:
- Track 1 (PatternCommand)
- Track 2 (for testing, but can mock)

**Core Structure**:
```rust
struct CalibrationController {
    // Communication
    pattern_publisher: TypedZmqPublisher<PatternCommand>,
    tracking_subscriber: TrackingCollector,

    // Calibration state
    mode: CalibrationMode,
    spots: Vec<SpotParams>,
    spot_index: usize,
    measurements: Vec<(f64, f64)>,
    correspondences: Vec<PointCorrespondence>,

    // Results (updated in real-time)
    current_alignment: Option<OpticalAlignment>,
    sensor_coverage: Option<SensorCoverage>,
    defocus_map: Option<DefocusMap>,

    // Timing
    settle_duration: Duration,
    last_move: Instant,
}

enum CalibrationMode {
    Idle,
    GridCalibration { grid_size: usize, spacing: f64 },
    ManualSpot,
    DefocusScan,
}
```

**Command-line interface**:
```bash
calibration_controller \
    --pattern-pub tcp://test-bench-pi:5556 \
    --tracking-sub tcp://orin:5555 \
    --grid-size 5 \
    --grid-spacing 200.0 \
    --settle-secs 1.0
```

**Deliverable**: CLI tool that drives calibration and outputs results

---

### Track 5: Visualization UI (egui)

**Files**: `test-bench/src/bin/calibration_controller.rs` (integrated)

**Dependencies**: Track 4 (controller logic)

**UI Panels**:

1. **Control Panel**
   - Start/stop calibration
   - Grid size, spacing controls
   - Settle time slider
   - Manual spot position (for debugging)

2. **Live View**
   - OLED coordinate system with current spot position
   - Sensor coordinate system with measured positions
   - Overlay showing correspondence lines

3. **Transform Panel**
   - Scale X/Y
   - Rotation (degrees)
   - Translation X/Y
   - RMS error
   - Point count

4. **Coverage View**
   - Sensor frame outline
   - OLED projection polygon (computed from transform)
   - Coverage percentage

5. **Defocus Heatmap** (if FWHM available)
   - Grid of measured FWHM values
   - Color-coded by deviation from expected
   - Best focus region highlighted

**Deliverable**: Interactive egui application

---

### Track 6: Calibration Result Types (shared crate)

**Files**: `shared/src/calibration_results.rs`

**Dependencies**: None

```rust
/// Sensor coverage computed from optical alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorCoverage {
    /// OLED corners mapped to sensor coordinates
    /// Order: top-left, top-right, bottom-right, bottom-left
    pub corners: [(f64, f64); 4],

    /// Bounding box in sensor pixels
    pub bounds: SensorBounds,

    /// Fraction of sensor area covered by OLED projection
    pub coverage_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorBounds {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl SensorCoverage {
    pub fn from_alignment(
        alignment: &OpticalAlignment,
        oled_width: u32,
        oled_height: u32,
        sensor_width: u32,
        sensor_height: u32,
    ) -> Self {
        // Map OLED corners through transform
        let corners = [
            alignment.display_to_sensor(0.0, 0.0),
            alignment.display_to_sensor(oled_width as f64, 0.0),
            alignment.display_to_sensor(oled_width as f64, oled_height as f64),
            alignment.display_to_sensor(0.0, oled_height as f64),
        ];

        // Compute bounds and coverage...
    }
}

/// Defocus measurements across the field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefocusMap {
    pub points: Vec<DefocusPoint>,
    pub grid_size: (usize, usize),
    pub expected_fwhm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefocusPoint {
    pub display_x: f64,
    pub display_y: f64,
    pub sensor_x: f64,
    pub sensor_y: f64,
    pub measured_fwhm_x: f64,
    pub measured_fwhm_y: f64,
    /// Ratio of measured to expected FWHM (1.0 = perfect focus)
    pub defocus_ratio: f64,
}
```

**Deliverable**: Types for calibration outputs with save/load

---

## Parallel Development Tracks

| Track | Component | Dependencies | Estimated Size |
|-------|-----------|--------------|----------------|
| 1 | PatternCommand types | None | ~50 lines |
| 2 | RemoteControlled pattern | Track 1 | ~200 lines |
| 3 | Extended TrackingMessage | None | ~50 lines |
| 4 | CalibrationController core | Track 1, 2 | ~400 lines |
| 5 | egui Visualization | Track 4 | ~300 lines |
| 6 | Result types (Coverage, Defocus) | None | ~150 lines |

**Parallelization**:
- Tracks 1, 3, 6 can all start immediately (no dependencies)
- Track 2 needs Track 1
- Track 4 needs Tracks 1, 2
- Track 5 needs Track 4

```
Timeline:

    Track 1 ─────┐
                 ├──► Track 2 ─────┐
    Track 3 ─────┤                 ├──► Track 4 ──► Track 5
                 │                 │
    Track 6 ─────┴─────────────────┘
```

## Interface Contracts

### ZMQ Endpoints

| Endpoint | Direction | Message Type | Port (suggested) |
|----------|-----------|--------------|------------------|
| Pattern commands | Desktop → Pi | PatternCommand | tcp://*:5556 |
| Tracking updates | Orin → Desktop | TrackingMessage | tcp://*:5555 |

### Message Flow

1. Desktop publishes `PatternCommand::Spot { x, y, fwhm, intensity }`
2. Pi receives, renders spot on OLED
3. Camera sees spot, cam_track detects it
4. cam_track publishes `TrackingMessage { x, y, fwhm_x, fwhm_y, ... }`
5. Desktop receives, accumulates measurements
6. After N measurements, Desktop averages and stores correspondence
7. Desktop advances to next spot position
8. After all positions, Desktop computes `OpticalAlignment`
9. Desktop computes `SensorCoverage` from alignment
10. Desktop optionally runs defocus scan using measured FWHM values

## Testing Strategy

### Unit Tests
- PatternCommand serialization roundtrip
- SensorCoverage computation from known alignment
- DefocusMap statistics

### Integration Tests
- ZMQ pub/sub with PatternCommand
- RemoteControlled pattern renders correct spots
- Full loop with simulated tracking (like existing optical_calibration tests)

### Manual Testing
- Deploy to real hardware
- Verify spot appears at commanded position
- Verify tracking messages arrive
- Verify calibration converges

## Future Extensions

1. **Continuous mode**: Keep running after initial calibration, update transform as system drifts
2. **Distortion mapping**: Higher-order polynomial instead of affine
3. **Multi-spot tracking**: Display and track multiple spots simultaneously for faster calibration
4. **Thermal compensation**: Track drift vs temperature
5. **Auto-focus**: Use defocus map to command focus actuator

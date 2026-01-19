# Remote Calibration System Architecture

## Overview

Desktop-driven optical calibration system that remotely controls the OLED display pattern while collecting tracking feedback from the camera system. Produces real-time visualization of alignment, sensor coverage, and focus quality.

## System Diagram

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                       DESKTOP                                 │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        calibration_controller (NEW)                     │  │
│  │                                                                         │  │
│  │  ┌─────────────┐    ┌──────────────────────────────┐   ┌─────────────┐  │  │
│  │  │ Pattern     │    │     Calibration Engine       │   │ Tracking    │  │  │
│  │  │ Commander   │    │  - Point collection          │   │ Listener    │  │  │
│  │  │ (ZMQ REQ)   │    │  - Transform estimation      │   │ (ZMQ SUB)   │  │  │
│  │  └──────┬──────┘    │  - Coverage calculation      │   └──────┬──────┘  │  │
│  │         │           │  - Defocus mapping           │          │         │  │
│  │         │           └──────────────────────────────┘          │         │  │
│  └─────────┼─────────────────────────────────────────────────────┼─────────┘  │
│            │                                                     │            │
│            │ PatternCommand                      TrackingMessage │            │
│            │ (REQ/REP)                           (PUB/SUB)       │            │
└────────────┼─────────────────────────────────────────────────────┼────────────┘
             │                                                     │
             │                                                     │
             ▼                                                     ▼
┌────────────────────────┐                             ┌────────────────────────┐
│     TEST-BENCH-PI      │                             │         NSV-211        │
│                        │                             │                        │
│  ┌──────────────────┐  │                             │  ┌──────────────────┐  │
│  │  calibrate_serve │  │                             │  │    cam_track     │  │
│  │                  │  │                             │  │                  │  │
│  │  ZMQ REP         │  │                             │  │  ZMQ PUB         │  │
│  │  tcp://*:5556    │  │                             │  │  tcp://*:5555    │  |
│  └──────────────────┘  │                             │  └──────────────────┘  |
│            ▼           │        (optical path)       │          ▲             |
│    OLED 2560x2560 ─────┼─────────────────────────────┼───────►Camera          |
│                        │                             │                        │
└────────────────────────┘                             └────────────────────────┘
```

## Components

### Track 1: Shared Message Types (shared crate) ✅ COMPLETED

**Files**: `shared/src/pattern_command.rs`

**Dependencies**: None

**Implementation**:
```rust
/// Commands sent from desktop to control OLED pattern display
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type")]
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
    #[default]
    Clear,
}

// Helper functions included:
// - PatternCommand::centered_spot(width, height, fwhm, intensity)
// - PatternCommand::centered_grid(width, height, grid_size, spacing, fwhm, intensity)
```

**Acknowledgment**: Simple string response `"ok"` or `"error: <message>"`

**Deliverable**: ✅ Message types in shared crate with serialization tests

---

### Track 2: Remote-Controlled Pattern (test-bench crate) ✅ COMPLETED

**Files**:
- `test-bench/src/display_patterns/remote_controlled.rs`
- `test-bench/src/calibrate/pattern.rs` (PatternConfig::RemoteControlled variant)
- `test-bench/src/calibrate/schema.rs` (web UI schema + Text control type)
- `test-bench/src/bin/calibrate_serve.rs` (ZMQ REP handler)

**Dependencies**: Track 1 (PatternCommand types)

**Architecture**: Uses ZMQ REQ/REP pattern for reliable command delivery:
- calibrate_serve **binds** REP socket at startup (always listening)
- Desktop **connects** with REQ socket, sends command, waits for ack
- Commands update shared state regardless of active pattern

**PatternConfig Variant**:
```rust
#[serde(skip)]
RemoteControlled {
    state: Arc<Mutex<RemotePatternState>>,
    pattern_size: PixelShape,
}
```

**RemotePatternState** (simplified, no ZMQ polling):
```rust
pub struct RemotePatternState {
    current_command: PatternCommand,
    last_update: Instant,
    cached_norm_factor: f64,
    cached_fwhm: f64,
}

impl RemotePatternState {
    pub fn set_command(&mut self, cmd: PatternCommand) {
        self.current_command = cmd;
        self.last_update = Instant::now();
    }
    pub fn current(&self) -> &PatternCommand { &self.current_command }
}
```

**calibrate_serve changes**:
- `--zmq-bind` argument (default: `tcp://*:5556`)
- ZMQ REP handler thread spawned at startup
- Shared `RemotePatternState` updated on command receipt
- Pattern selectable via HTTP POST `/config` with `{"pattern_id": "RemoteControlled", "values": {}}`

**Usage**:
```python
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect('tcp://cfl-test-bench.tail944341.ts.net:5556')
sock.send_json({'type': 'Spot', 'x': 1280, 'y': 1280, 'fwhm': 5.0, 'intensity': 1.0})
print(sock.recv_string())  # "ok"
```

**Deliverable**: ✅ calibrate_serve displays spots at positions commanded via ZMQ REQ/REP

---

### Track 3: Extended Tracking Message ✅ COMPLETED

**Files**:
- `shared/src/image_proc/centroid.rs` - `SpotShape` struct definition
- `shared/src/tracking_message.rs` - `TrackingMessage` with SpotShape
- `monocle/src/lib.rs` - `GuideStar`, `GuidanceUpdate` integration

**Dependencies**: None

**Summary**: TrackingMessage now includes a `SpotShape` struct with flux, second moments (m_xx, m_yy, m_xy), aspect ratio, and diameter. This provides full shape characterization for defocus mapping and radiometric calibration.

**Deliverable**: ✅ TrackingMessage includes spot shape with moments and diameter

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
    --pattern-pub tcp://cfl-test-bench:5556 \
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

### Track 7: System Info Auto-Discovery (no new endpoints)

**Problem**: calibration_controller currently requires manual `--sensor-width`, `--sensor-height`, `--display-width`, `--display-height` flags. These should be auto-discovered from existing services.

**Approach**: Piggyback on existing interfaces - no new ZMQ endpoints needed.

**Files**:
- `shared/src/system_info.rs` (new) - Message types
- `test-bench/src/bin/calibrate_serve.rs` - Add HTTP `/info` endpoint
- `shared/src/tracking_message.rs` - Add optional SensorInfo to TrackingMessage

**Dependencies**: None

---

#### Display Info via HTTP (calibrate_serve)

calibrate_serve already has an HTTP server. Add a simple `/info` endpoint:

```rust
/// Display system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayInfo {
    pub width: u32,
    pub height: u32,
    /// Pixel pitch in microns
    pub pixel_pitch_um: f64,
    pub name: String,
}

// Add route in calibrate_serve HTTP server:
// GET /info -> DisplayInfo JSON
async fn get_display_info(State(state): State<AppState>) -> Json<DisplayInfo> {
    Json(DisplayInfo {
        width: state.display_width,
        height: state.display_height,
        pixel_pitch_um: state.pixel_pitch_um,
        name: "OLED".to_string(),
    })
}
```

**Usage**:
```bash
curl http://cfl-test-bench:8080/info
# {"width":2560,"height":2560,"pixel_pitch_um":9.6,"name":"OLED"}
```

---

#### Sensor Info via TrackingMessage (cam_track)

Extend TrackingMessage to include optional sensor metadata. The first message (or periodic messages) include this info:

```rust
/// Sensor system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorInfo {
    pub width: u32,
    pub height: u32,
    /// Pixel pitch in microns
    pub pixel_pitch_um: f64,
    pub name: String,
}

/// Extended tracking message with optional sensor info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingMessage {
    pub track_id: u32,
    pub x: f64,
    pub y: f64,
    pub timestamp: Timestamp,
    pub shape: SpotShape,
    /// Sensor info - included on first message and periodically thereafter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensor_info: Option<SensorInfo>,
}
```

**cam_track behavior**:
- Include `sensor_info` on the first TrackingMessage after startup
- Include it again every N messages (e.g., every 100) for late joiners
- Skip it on most messages to save bandwidth

```rust
// In cam_track publish loop:
let include_info = frame_count == 0 || frame_count % 100 == 0;
let msg = TrackingMessage {
    track_id,
    x, y,
    timestamp,
    shape,
    sensor_info: if include_info {
        Some(SensorInfo {
            width: sensor_width,
            height: sensor_height,
            pixel_pitch_um: camera.pixel_pitch_um(),
            name: camera_name.clone(),
        })
    } else {
        None
    },
};
```

**calibration_controller behavior**:
- On startup, wait for first TrackingMessage with `sensor_info`
- Cache the sensor dimensions
- Use cached values for visualization

---

**CLI changes to calibration_controller**:
```bash
# Display info fetched via HTTP (existing server, new endpoint)
--display-info-url http://cfl-test-bench.tail944341.ts.net:8080/info

# Sensor info auto-discovered from TrackingMessage stream
# (no new endpoint needed)

# Manual overrides still available if needed
--display-width 2560  # optional override
--sensor-width 9576   # optional override
```

**Deliverable**: calibration_controller auto-discovers dimensions without new ZMQ endpoints

---

## Parallel Development Tracks

| Track | Component | Dependencies | Status |
|-------|-----------|--------------|--------|
| 1 | PatternCommand types | None | ✅ Complete |
| 2 | RemoteControlled pattern + ZMQ REP | Track 1 | ✅ Complete |
| 3 | Extended TrackingMessage (SpotShape) | None | ✅ Complete |
| 4 | CalibrationController core | Track 1, 2 | ✅ Complete |
| 5 | egui Visualization | Track 4 | Pending |
| 6 | Result types (Coverage, Defocus) | None | Pending |
| 7 | System Info Endpoints | None | Pending |

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

| Endpoint | Pattern | Direction | Message Type | Port |
|----------|---------|-----------|--------------|------|
| Pattern commands | REQ/REP | Desktop (REQ) → Pi (REP) | PatternCommand JSON | tcp://*:5556 |
| Tracking updates | PUB/SUB | Orin (PUB) → Desktop (SUB) | TrackingMessage JSON | tcp://*:5555 |

### HTTP Endpoints

| Service | Endpoint | Method | Response |
|---------|----------|--------|----------|
| calibrate_serve | `/info` | GET | DisplayInfo JSON |
| calibrate_serve | `/config` | POST | Pattern configuration |

**REQ/REP Benefits**:
- Desktop knows when command was received (waits for "ok" reply)
- No missed commands (unlike PUB/SUB which can drop if subscriber not ready)
- Simple error handling ("error: ..." replies)

### Message Flow

1. Desktop sends `PatternCommand::Spot { x, y, fwhm, intensity }` via REQ socket
2. Pi receives on REP socket, updates shared state, replies `"ok"`
3. Desktop receives `"ok"`, knows spot will be rendered
4. Camera sees spot, cam_track detects it
5. cam_track publishes `TrackingMessage { x, y, shape: SpotShape { flux, diameter, ... } }`
6. Desktop receives via SUB, accumulates measurements
7. After N measurements, Desktop averages and stores correspondence
8. Desktop advances to next spot position
9. After all positions, Desktop computes `OpticalAlignment`
10. Desktop computes `SensorCoverage` from alignment
11. Desktop optionally runs defocus scan using measured FWHM values

## Deployment

### Building and Deploying to cfl-test-bench

```bash
# Build and deploy calibrate_serve to Pi
./scripts/build-remote.sh --package test-bench --binary calibrate_serve --features sdl2 --cfl-test-bench
```

### Managing the systemd Service

calibrate_serve runs as a systemd service on cfl-test-bench:

```bash
# View logs (use this, not manual nohup!)
sudo journalctl -u calibrate-serve -f

# Restart service (picks up new binary after deploy)
sudo systemctl restart calibrate-serve

# Check status
sudo systemctl status calibrate-serve

# Stop service
sudo systemctl stop calibrate-serve
```

**Important**: Always use the systemd service, not manual `nohup` commands. The service:
- Starts automatically on boot
- Uses `--wait-for-oled` to detect the OLED display
- Logs to journald (accessible via `journalctl`)
- Restarts on failure

Service file location: `/etc/systemd/system/calibrate-serve.service`

---

## Testing Strategy

### Unit Tests
- PatternCommand serialization roundtrip
- SensorCoverage computation from known alignment
- DefocusMap statistics

### Integration Tests
- ZMQ REQ/REP with PatternCommand
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

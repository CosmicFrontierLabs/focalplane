# Guiding System Coordinate Systems

This document describes the coordinate systems, transforms, and data flow in the closed-loop guiding system from centroid measurement through to FSM command.

## Overview

The guiding system stabilizes a star image on the sensor by commanding a Fast Steering Mirror (FSM) to counteract motion. The data flow involves several coordinate transformations:

```
┌─────────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│ Sensor Centroid │ ──► │ LOS          │ ──► │ Calibration   │ ──► │ FSM Command │
│ (pixels)        │     │ Controller   │     │ Transform     │     │ (µrad)      │
└─────────────────┘     └──────────────┘     └───────────────┘     └─────────────┘
```

## 1. Sensor Coordinates (Pixels)

### Measurement
- **Source**: Camera centroid detection via monocle FGS
- **Units**: Pixels
- **Frame**: Sensor-relative, origin at (0, 0) corner
- **Absolute vs Relative**: Absolute position on sensor
- **Typical values**: 0 to ~9500 px for NSV455, 0 to ~1024 px for smaller sensors

### Setpoint
- **Source**: User-defined or auto-acquired lock position
- **Units**: Pixels
- **Frame**: Same sensor coordinate frame as measurement
- **Set via**: `GuidingController::set_setpoint(x, y)`

### Error Calculation
The tracking error is computed internally by the LOS controller:
```
error_x = command_x - meas_x
error_y = command_y - meas_y
```

A positive error means the centroid is below/left of the setpoint.

**Location**: `monocle/src/controllers/los_controller.rs:284-285`

## 2. LOS Controller

### Input
- **Type**: Centroid position (not error directly)
- **Units**: Pixels [absolute on sensor]
- **Source**: Centroid tracker output

### Internal Operation
The LOS controller (`monocle/src/controllers/los_controller.rs`) is a discrete-time state-space compensator:

1. Computes error internally: `error = command - measurement` [pixels]
2. Updates 5-state internal compensator
3. Produces correction signal [sensor-frame]

**Design rate**: 40 Hz

### Output
- **Name**: `LosControlOutput { u_x, u_y }`
- **Units**: Sensor-frame correction signal
- **Frame**: Sensor-aligned (X/Y correspond to sensor X/Y)
- **Relative**: Correction signal, not absolute position

The output is in a "sensor-aligned frame" - the axes correspond to sensor X/Y. The controller's gain coefficients encode both control dynamics and an implicit scale factor from the original design. The calibration transform converts this to physical FSM coordinates.

**Location**: `monocle/src/controllers/los_controller.rs:145-150`

### Key Behavior
- **Integrating dynamics**: First state acts as integrator (A[0,0] ≈ 1.0)
- **D = 0**: No direct feedthrough, output depends only on accumulated state
- **Do not accumulate externally**: The LOS controller integrates internally

## 3. Calibration Transform

The calibration transform converts from pixel-space corrections to FSM angle commands.

### Why This Is Needed
The relationship between FSM angles and sensor centroid motion depends on:
- Physical mounting orientation of the FSM
- Optical path (fold mirrors, lenses)
- Detector orientation
- Scale factor (pixels per µrad)

The transform can include:
- Rotation (FSM axes may not align with sensor axes)
- Scaling (different sensitivity per axis)
- Sign inversion (some axes may be inverted)

### Transform Matrix

The calibration produces two related matrices:

**FSM → Sensor** (`fsm_to_sensor`): Maps FSM motion to pixel motion
```
[ Δpix_x ]   [ m00  m01 ]   [ Δangle_x ]
[        ] = [          ] × [          ]
[ Δpix_y ]   [ m10  m11 ]   [ Δangle_y ]
```
Units: pixels per µrad

**Sensor → FSM** (`sensor_to_fsm`): Inverse, maps pixel motion to required FSM motion
```
[ Δangle_x ]             [ Δpix_x ]
[          ] = M^(-1) ×  [        ]
[ Δangle_y ]             [ Δpix_y ]
```
Units: µrad per pixel

**Location**: `test-bench/src/fsm_calibration/transform.rs:249-259`

### Example Calibration Values
From the NSV455 on orin-005:
```json
{
  "fsm_to_sensor": [0.028329, 0.001604, 0.000027, -0.020555],
  "intercept_pixels": [2993.07, 3531.09]
}
```

Matrix form:
```
fsm_to_sensor = [ 0.0283   0.0016 ]
                [ 0.00003 -0.0206 ]
```

The negative value at position (1,1) = -0.0206 means:
- FSM axis 2 and sensor Y move in **opposite directions**
- Without the transform, the controller would drive the wrong way on Y

### Transform Application

In `GuidingController::update()`:
```rust
// LOS controller: pixels → sensor-frame correction (has integrating dynamics)
let LosControlOutput { u_x, u_y } = self.los_controller.update(centroid_x, centroid_y);

// Calibration transform: sensor-frame → FSM µrad (rotation, scale, sign)
let (offset_x, offset_y) = self.transform.pix_delta_to_angle_delta(u_x, u_y);
```

**Location**: `test-bench/src/guiding/mod.rs:227-231`

## 4. FSM Offset Calculation

### Current Offset
The guiding controller maintains an FSM offset from center:
```rust
self.fsm_offset = (offset_x, offset_y);
```

This is **not accumulated** - the LOS controller has integrating dynamics internally.

### Absolute FSM Position
```rust
let center = self.range_limits.center();
let raw_x = center.0 + offset_x;
let raw_y = center.1 + offset_y;
```

**Location**: `test-bench/src/guiding/mod.rs:234-237`

## 5. Safety Clamping

### Range Limits
The FSM has physical travel limits:
```rust
pub struct FsmRangeLimits {
    pub x_min: f64,  // Typically ~0 µrad
    pub x_max: f64,  // Typically ~2000 µrad
    pub y_min: f64,
    pub y_max: f64,
}
```

### Clamping
```rust
let (clamped_x, clamped_y) = self.range_limits.clamp(raw_x, raw_y);
let was_clamped = clamped_x != raw_x || clamped_y != raw_y;
```

If clamped, the offset is updated to reflect actual position to prevent integrator windup:
```rust
if was_clamped {
    self.fsm_offset = (clamped_x - center.0, clamped_y - center.1);
}
```

**Location**: `test-bench/src/guiding/mod.rs:239-253`

## 6. FSM Command

### Output
The `GuidingOutput` structure contains:
```rust
pub struct GuidingOutput {
    pub fsm_x_urad: f64,    // Clamped command
    pub fsm_y_urad: f64,
    pub raw_x_urad: f64,    // Before clamping
    pub raw_y_urad: f64,
    pub was_clamped: bool,
}
```

### Command Execution
The FSM command is sent via the hardware interface:
```rust
fsm.move_to(output.fsm_x_urad, output.fsm_y_urad)?;
```

**Location**: `hardware/src/pi/s330.rs:195-197`

### Units
- **FSM Position**: Microradians (µrad) of tip/tilt angle
- **Typical range**: 0 to ~2000 µrad per axis
- **Center**: ~1000 µrad (middle of travel)

## Complete Data Flow Example

1. **Sensor**: Measures centroid at (3010.5, 3515.2) pixels
2. **Setpoint**: Target is (3000.0, 3530.0) pixels
3. **Error**: (-10.5, 14.8) pixels (centroid right and below target)
4. **LOS Controller**: Produces (u_x, u_y) correction signal
5. **Transform**: Applies `sensor_to_fsm` matrix inverse:
   ```
   fsm_delta = sensor_to_fsm × [u_x, u_y]
   ```
   Accounting for axis rotation and sign inversion
6. **Offset**: FSM position = center + fsm_delta
7. **Clamp**: Ensure within [0, 2000] µrad
8. **Command**: Send to PI E-727 controller

## Coordinate System Summary

| Stage | Coordinate Frame | Units | Absolute/Relative |
|-------|-----------------|-------|-------------------|
| Centroid measurement | Sensor | pixels | Absolute |
| Setpoint | Sensor | pixels | Absolute |
| Tracking error | Sensor | pixels | Relative (delta) |
| LOS controller output | Sensor-aligned | sensor-frame | Relative (correction) |
| Post-transform output | FSM-aligned | µrad | Relative (offset from center) |
| FSM offset | FSM | µrad | Relative to center |
| FSM command | FSM | µrad | Absolute |

**Note on "sensor-frame" units**: The LOS controller output has the control dynamics and gains needed for the feedback loop, but its coordinate axes correspond to sensor X/Y. The numerical values encode both the control response and an implicit pixel-to-angle relationship from the original design. The calibration transform rotates and scales this into physical FSM coordinates.

## Key Files

- `test-bench/src/guiding/mod.rs` - GuidingController implementation
- `test-bench/src/fsm_calibration/transform.rs` - FsmTransform with conversion methods
- `monocle/src/controllers/los_controller.rs` - LOS feedback controller
- `hardware/src/pi/s330.rs` - S-330 FSM driver
- `docs/fsm-axis-calibration.md` - Calibration procedure details

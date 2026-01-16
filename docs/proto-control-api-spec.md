# Proto-Control API Specification

Line-of-Sight Control Interface for the meter-sim spacecraft pointing control system.

> **Note**: These types are now located in `monocle_harness::proto_control`.
> Use `use monocle_harness::{Timestamp, GyroReadout, FgsReadout, FsmCommand, StateEstimator};`

## Overview

This module defines the protocol and data structures for the line-of-sight (LOS)
control algorithm to interface with the spacecraft simulation system. The
`StateEstimator` trait provides the interface for state estimation and control logic.

## Architecture

### Control Loop Timing

- **`GyroTick`** - 500Hz tick counter synchronized with gyroscope measurements
- **`Timestamp`** - Microseconds (u64) for sensor timing precision

### Sensor Inputs

| Struct | Description | Rate |
|--------|-------------|------|
| `GyroReadout` | 3-axis integrated angles (radians) from Exail gyroscope | Every gyro tick (500Hz) |
| `FgsReadout` | Fine Guidance System 2D angular position with variance (arcseconds) | Lower rate (camera-based) |
| `FsmReadout` | Fast Steering Mirror voltage feedback (vx/vy volts) | Every gyro tick |

### Control Output

- **`FsmCommand`** - Voltage commands to drive the Fast Steering Mirror (vx/vy volts)

### State Container

- **`EstimatorState`** - Packages one complete control cycle: gyro tick, all sensor
  readings, and the computed FSM command output

## The StateEstimator Trait

This is the interface for the LOS control algorithm:

```text
f: (state_history, gyro, fsm_readout, fgs_readout?) → EstimatorState
```

Key design points:

- Receives FIFO history of previous outputs (oldest → newest)
- FGS readout is optional (camera rate slower than 500Hz control loop)
- Caller handles FSM command execution timing and LOS updates to payload computer

## System Context

The system implements classic sensor fusion for spacecraft fine pointing:

1. **High-rate gyro measurement** - Exail gyroscope provides integrated angles at 500Hz
2. **Periodic FGS corrections** - Camera-based Fine Guidance System provides absolute reference
3. **FSM actuation** - Fast Steering Mirror commands for fine pointing stabilization

```text
+-------------+     +-----------------+     +-------------+
| Exail Gyro  |---->|                 |---->|     FSM     |
|   (500Hz)   |     |  StateEstimator |     |  (actuator) |
+-------------+     |                 |     +-------------+
                    |  (LOS control   |
+-------------+     |   algorithm)    |     +-------------+
|     FGS     |---->|                 |---->|   Payload   |
|  (camera)   |     +-----------------+     |  Computer   |
+-------------+           ^                 | (LOS update)|
                          |                 +-------------+
                    +-----+-----+
                    |   state   |
                    |  history  |
                    +-----------+
```


\newpage

## Timestamp

*Struct*

Timestamp in microseconds since the initialization of the control loop.

Maximum representable time: ~584,942 years. If your mission exceeds this
duration, congratulations on the interstellar voyage and/or achieving
functional immortality. Please file a bug report from Alpha Centauri.

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timestamp(pub u64);
```

### Methods

#### `from_micros`

Create timestamp from microseconds.

```rust
pub fn from_micros(micros: u64) -> Self
```

#### `as_micros`

Get timestamp as microseconds.

```rust
pub fn as_micros(&self) -> u64
```

\newpage

## GyroTick

*Struct*

Gyro tick counter at 500Hz.

Represents discrete timing ticks synchronized with gyroscope measurements.
Maximum representable time: ~99.42 days (2^32 ticks at 500Hz)

# Behavior

- Increments by exactly 1 for each `GyroReadout` result
- Strictly monotonic (always increasing, never decreases or repeats)
- Provides consistent time reference across all sensor measurements

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GyroTick(pub u32);
```

\newpage

## GyroReadout

*Struct*

Gyroscope readout representing integrated angle on three axes.

All angular values are in radians, representing the integrated angle
as reported by the Exail gyroscope hardware.

# Timing

The `timestamp` field represents the time difference between the current
XYZ angle measurement and the previous measurement, as reported by the
Exail gyroscope hardware. The timestamp is aligned to the first moment
the measurement was computed by the gyro.

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GyroReadout {
    /// X-axis angle in radians
    pub x: f64,
    /// Y-axis angle in radians
    pub y: f64,
    /// Z-axis angle in radians
    pub z: f64,
    /// Timestamp representing time difference from previous measurement,
    /// aligned to the first moment the measurement was computed by the gyro
    pub timestamp: Timestamp,
}
```

\newpage

## FgsReadout

*Struct*

Fine Guidance System 2D angular estimate with uncertainty.

Represents pointing direction in two angular dimensions with
variance estimates for each axis.

# Timing

The `timestamp` field corresponds to the center of the image exposure.

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FgsReadout {
    /// X-axis angular position in arcseconds
    pub x: f64,
    /// Y-axis angular position in arcseconds
    pub y: f64,
    /// Variance of x-axis measurement in arcseconds²
    pub x_variance: f64,
    /// Variance of y-axis measurement in arcseconds²
    pub y_variance: f64,
    /// Timestamp corresponding to center of image exposure
    pub timestamp: Timestamp,
}
```

\newpage

## FsmReadout

*Struct*

FSM (Fast Steering Mirror) readout with X and Y axis feedback.

Represents voltage feedback from the fast steering mirror on two orthogonal axes.

# Timing

The `timestamp` field is intended to indicate the center of the ADC
(Analog-to-Digital Converter) window from the ExoLambda board.

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsmReadout {
    /// X-axis voltage readout in volts
    pub vx: f64,
    /// Y-axis voltage readout in volts
    pub vy: f64,
    /// Timestamp indicating center of ADC window from ExoLambda board
    pub timestamp: Timestamp,
}
```

\newpage

## FsmCommand

*Struct*

FSM (Fast Steering Mirror) command with X and Y axis voltages.

Represents commanded voltages for the fast steering mirror on two orthogonal axes.

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsmCommand {
    /// X-axis voltage command in volts
    pub vx: f64,
    /// Y-axis voltage command in volts
    pub vy: f64,
}
```

\newpage

## EstimatorState

*Struct*

Estimator state containing sensor measurements and control outputs.

Represents the complete state used by the estimation and control system.

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimatorState {
    /// Gyro tick for this state
    pub gyro_tick: GyroTick,
    /// Gyroscope readout
    pub gyro: GyroReadout,
    /// FSM voltage readout
    pub fsm_readout: FsmReadout,
    /// Optional fine guidance system readout
    pub fgs_readout: Option<FgsReadout>,
    /// FSM voltage command
    pub fsm_command: FsmCommand,
}
```

\newpage

## StateEstimator

*Trait*

Trait for line-of-sight state estimator implementation.

Implement this trait to provide state estimation and control logic
for the LOS control algorithm. The estimator processes sensor measurements
and state history to compute the next FSM command.

# Function Signature (Conceptual)

```text
f: (&[EstimatorState], &GyroReadout, &FsmReadout, Option<&FgsReadout>) → FsmCommand
```

# State History

The `state_history` parameter contains previous `EstimatorState` values
assembled by the caller from prior calls. This vector provides the estimator
with access to historical sensor readings and commands for filtering,
prediction, and state estimation purposes.

**Ordering Requirements:**

- Elements are ordered chronologically: `state_history[0]` is the oldest state
- `state_history[n-1]` is the most recent state (where n = length)
- On the first call, `state_history` will be empty (`&[]`)

**Length Constraints:**

- The vector will retain a fixed maximum number of previous states
- When the maximum length is reached, oldest states are removed as new ones are added
- The exact maximum length is implementation-defined (FIFO buffer behavior)
- The estimator should not assume any specific history length

# Parameters

- `state_history`: Previous estimator states assembled by caller (ordered oldest to newest)
- `gyro_readout`: Current gyroscope angle readout
- `fsm_readout`: Current FSM voltage readout
- `fgs_readout`: Optional fine guidance system readout. None if FGS
  data is not available for this cycle.

# Returns

The computed FSM voltage command for this cycle.

# Caller Responsibilities

Upon return of the `FsmCommand`, it is the caller's responsibility to:

- Assemble an `EstimatorState` from the passed sensor readings and returned command
- Append the new state to the history for subsequent calls
- Perform the `FsmCommand` adjustment to the FSM hardware at the appropriate time
- Push a Line-of-Sight (LOS) update to the payload computer at the required interval

### Definition

```rust
pub trait StateEstimator {
    /// Compute FSM command from sensor measurements and history.
    ///
    /// # Parameters
    ///
    /// - `state_history`: Previous estimator states assembled by caller,
    ///   ordered oldest to newest (index 0 = oldest). Empty on first call.
    ///   See trait-level documentation for detailed ordering requirements.
    /// - `gyro_readout`: Current gyroscope angle readout
    /// - `fsm_readout`: Current FSM voltage readout
    /// - `fgs_readout`: Optional fine guidance system update. None if FGS
    ///   update is not available for this cycle. When present, represents
    ///   a new fine guidance measurement.
    ///
    /// # Returns
    ///
    /// The computed FSM voltage command for this cycle.
    fn estimate(
        &self,
        state_history: &[EstimatorState],
        gyro_readout: &GyroReadout,
        fsm_readout: &FsmReadout,
        fgs_readout: Option<&FgsReadout>,
    ) -> FsmCommand;
}
```


# Proto-Control API Specification

**External Integrator Interface Documentation**

# Proto-Control: External Integrator Interface Specification

Protocol definitions and data structures for external state propagation
integrators to interface with the meter-sim spacecraft simulation system.

\newpage

## Timestamp

*Struct*

Timestamp in microseconds since the initialization of the control loop.

Maximum representable time: ~71.58 minutes (2^32 microseconds)

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timestamp(pub u32);
```

### Methods

#### `from_micros`

Create timestamp from microseconds.

```rust
pub fn from_micros(micros: u32) -> Self
```

#### `as_micros`

Get timestamp as microseconds.

```rust
pub fn as_micros(&self) -> u32
```

\newpage

## RtiTick

*Struct*

RTI (Real-Time Interrupt) tick counter at 500Hz.

Represents discrete timing ticks from the control system clock.
Maximum representable time: ~99.42 days (2^32 ticks at 500Hz)

# Behavior

- Increments by exactly 1 for each `GyroReadout` result
- Strictly monotonic (always increasing, never decreases or repeats)
- Provides consistent time reference across all sensor measurements

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RtiTick(pub u32);
```

\newpage

## GyroReadout

*Struct*

Gyroscope readout representing angular change on three axes.

All angular values are in arcseconds, representing the delta angle
since the previous measurement (not absolute position or rates).

# Timing

The `timestamp` field represents the time difference between the current
XYZ angle measurement and the previous measurement, as reported by the
Exail gyroscope hardware.

**Note:** The alignment of this timestamp with respect to the actual time
of the angle measurements is currently TBD (to be determined).

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GyroReadout {
    /// X-axis delta angle in arcseconds
    pub x: f64,
    /// Y-axis delta angle in arcseconds
    pub y: f64,
    /// Z-axis delta angle in arcseconds
    pub z: f64,
    /// Timestamp representing time difference from previous measurement
    /// as reported by Exail gyro (alignment TBD)
    pub timestamp: Timestamp,
}
```

\newpage

## FgsReadout

*Struct*

Fine Guidance System 2D angular estimate with uncertainty.

Represents pointing direction in two angular dimensions with
variance estimates for each axis.

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
    /// Timestamp of measurement
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
    /// RTI tick for this state
    pub rti_tick: RtiTick,
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

Trait for external state estimator implementation.

The external party implements this trait to provide state estimation
and control logic. The estimator processes sensor measurements and
command history to produce the next estimator state.

# Function Signature (Conceptual)

```text
f: (Vec<EstimatorState>, GyroReadout, FsmReadout, Option<FgsReadout>) → EstimatorState
```

# State History

The `state_history` parameter contains previous `EstimatorState` values
that were returned by prior calls to this `estimate()` function. This vector
provides the estimator with access to its own historical outputs for filtering,
prediction, and state estimation purposes.

**Ordering Requirements:**
- Elements are ordered chronologically: `state_history[0]` is the oldest state
- `state_history[n-1]` is the most recent state (where n = length)
- On the first call, `state_history` will be empty (`&[]`)
- Each subsequent call appends the previous return value to the end of the vector

**Length Constraints:**
- The vector will retain a fixed maximum number of previous states
- When the maximum length is reached, oldest states are removed as new ones are added
- The exact maximum length is implementation-defined (FIFO buffer behavior)
- The estimator should not assume any specific history length

**Example Timeline:**
```text
Call 1: state_history = []           → returns state_1
Call 2: state_history = [state_1]    → returns state_2
Call 3: state_history = [state_1, state_2] → returns state_3
...
Call N: state_history = [state_k, ..., state_N-1] → returns state_N
        (length capped at maximum, oldest states dropped)
```

# Parameters

- `state_history`: Previous estimator states returned by this function (ordered oldest to newest)
- `gyro`: New gyroscope delta angle readout since previous measurement
- `fsm_readout`: Current FSM voltage readout
- `fgs_readout`: Optional fine guidance system readout. None if FGS
  data is not available for this cycle.

# Returns

Next estimator state containing updated gyro, FSM readout, and FSM command.

# Caller Responsibilities

Upon return of the next `EstimatorState`, it is the caller's responsibility to:
- Perform the `FsmCommand` adjustment to the FSM hardware at the appropriate time
- Push a Line-of-Sight (LOS) update to the payload computer at the required interval

The estimator only computes the desired FSM command; execution timing and
LOS update delivery are handled externally by the calling system.

### Definition

```rust
pub trait StateEstimator {
    /// Estimate next state from sensor measurements and history.
    ///
    /// # Parameters
    ///
    /// - `state_history`: Previous estimator states returned by this function,
    ///   ordered oldest to newest (index 0 = oldest). Empty on first call.
    ///   See trait-level documentation for detailed ordering requirements.
    /// - `gyro_readout`: New gyroscope delta angle readout since previous measurement
    /// - `fsm_readout`: Current FSM voltage readout
    /// - `fgs_readout`: Optional fine guidance system update. None if FGS
    ///   update is not available for this cycle. When present, represents
    ///   a new fine guidance measurement.
    ///
    /// # Returns
    ///
    /// New estimator state containing the input measurements and computed
    /// FSM voltage command for this cycle.
    fn estimate(
        &self,
        state_history: &[EstimatorState],
        gyro_readout: &GyroReadout,
        fsm_readout: &FsmReadout,
        fgs_readout: Option<&FgsReadout>,
    ) -> EstimatorState;
}
```


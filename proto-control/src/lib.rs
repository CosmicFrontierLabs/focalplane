//! # Proto-Control: External Integrator Interface Specification
//!
//! Protocol definitions and data structures for external state propagation
//! integrators to interface with the meter-sim spacecraft simulation system.

/// Timestamp in microseconds since the initialization of the control loop.
///
/// Maximum representable time: ~71.58 minutes (2^32 microseconds)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timestamp(pub u32);

impl Timestamp {
    /// Create timestamp from microseconds.
    pub fn from_micros(micros: u32) -> Self {
        Self(micros)
    }

    /// Get timestamp as microseconds.
    pub fn as_micros(&self) -> u32 {
        self.0
    }
}

/// RTI (Real-Time Interrupt) tick counter at 500Hz.
///
/// Represents discrete timing ticks from the control system clock.
/// Maximum representable time: ~99.42 days (2^32 ticks at 500Hz)
///
/// # Behavior
///
/// - Increments by exactly 1 for each `GyroReadout` result
/// - Strictly monotonic (always increasing, never decreases or repeats)
/// - Provides consistent time reference across all sensor measurements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RtiTick(pub u32);

/// Gyroscope readout representing angular change on three axes.
///
/// All angular values are in arcseconds, representing the delta angle
/// since the previous measurement (not absolute position or rates).
///
/// # Timing
///
/// The `timestamp` field represents the time difference between the current
/// XYZ angle measurement and the previous measurement, as reported by the
/// Exail gyroscope hardware.
///
/// **Note:** The alignment of this timestamp with respect to the actual time
/// of the angle measurements is currently TBD (to be determined).
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

impl GyroReadout {
    /// Create new gyro readout.
    pub fn new(x: f64, y: f64, z: f64, timestamp: Timestamp) -> Self {
        Self { x, y, z, timestamp }
    }

    /// Convert delta angles to radians.
    pub fn to_radians(&self) -> [f64; 3] {
        const ARCSEC_TO_RAD: f64 = 4.84813681109536e-6;
        [
            self.x * ARCSEC_TO_RAD,
            self.y * ARCSEC_TO_RAD,
            self.z * ARCSEC_TO_RAD,
        ]
    }

    /// Create from delta angles in radians.
    pub fn from_radians(x_rad: f64, y_rad: f64, z_rad: f64, timestamp: Timestamp) -> Self {
        const RAD_TO_ARCSEC: f64 = 206264.80624709636;
        Self {
            x: x_rad * RAD_TO_ARCSEC,
            y: y_rad * RAD_TO_ARCSEC,
            z: z_rad * RAD_TO_ARCSEC,
            timestamp,
        }
    }
}

/// Fine Guidance System 2D angular estimate with uncertainty.
///
/// Represents pointing direction in two angular dimensions with
/// variance estimates for each axis.
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

impl FgsReadout {
    /// Create new FGS readout.
    pub fn new(x: f64, y: f64, x_variance: f64, y_variance: f64, timestamp: Timestamp) -> Self {
        Self {
            x,
            y,
            x_variance,
            y_variance,
            timestamp,
        }
    }

    /// Convert angular positions to radians.
    pub fn to_radians(&self) -> [f64; 2] {
        const ARCSEC_TO_RAD: f64 = 4.84813681109536e-6;
        [self.x * ARCSEC_TO_RAD, self.y * ARCSEC_TO_RAD]
    }

    /// Get variances in radians².
    pub fn variance_radians(&self) -> [f64; 2] {
        const ARCSEC_TO_RAD: f64 = 4.84813681109536e-6;
        const ARCSEC2_TO_RAD2: f64 = ARCSEC_TO_RAD * ARCSEC_TO_RAD;
        [
            self.x_variance * ARCSEC2_TO_RAD2,
            self.y_variance * ARCSEC2_TO_RAD2,
        ]
    }

    /// Get standard deviations in arcseconds.
    pub fn std_dev(&self) -> [f64; 2] {
        [self.x_variance.sqrt(), self.y_variance.sqrt()]
    }
}

/// FSM (Fast Steering Mirror) readout with X and Y axis feedback.
///
/// Represents voltage feedback from the fast steering mirror on two orthogonal axes.
///
/// # Timing
///
/// The `timestamp` field is intended to indicate the center of the ADC
/// (Analog-to-Digital Converter) window from the ExoLambda board.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsmReadout {
    /// X-axis voltage readout in volts
    pub vx: f64,
    /// Y-axis voltage readout in volts
    pub vy: f64,
    /// Timestamp indicating center of ADC window from ExoLambda board
    pub timestamp: Timestamp,
}

impl FsmReadout {
    /// Create new FSM readout.
    pub fn new(vx: f64, vy: f64, timestamp: Timestamp) -> Self {
        Self { vx, vy, timestamp }
    }
}

/// FSM (Fast Steering Mirror) command with X and Y axis voltages.
///
/// Represents commanded voltages for the fast steering mirror on two orthogonal axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsmCommand {
    /// X-axis voltage command in volts
    pub vx: f64,
    /// Y-axis voltage command in volts
    pub vy: f64,
}

impl FsmCommand {
    /// Create new FSM command.
    pub fn new(vx: f64, vy: f64) -> Self {
        Self { vx, vy }
    }
}

/// Estimator state containing sensor measurements and control outputs.
///
/// Represents the complete state used by the estimation and control system.
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

impl EstimatorState {
    /// Create new estimator state.
    pub fn new(
        rti_tick: RtiTick,
        gyro: GyroReadout,
        fsm_readout: FsmReadout,
        fgs_readout: Option<FgsReadout>,
        fsm_command: FsmCommand,
    ) -> Self {
        Self {
            rti_tick,
            gyro,
            fsm_readout,
            fgs_readout,
            fsm_command,
        }
    }
}

/// Trait for external state estimator implementation.
///
/// The external party implements this trait to provide state estimation
/// and control logic. The estimator processes sensor measurements and
/// command history to produce the next estimator state.
///
/// # Function Signature (Conceptual)
///
/// ```text
/// f: (Vec<EstimatorState>, GyroReadout, FsmReadout, Option<FgsReadout>) → EstimatorState
/// ```
///
/// # State History
///
/// The `state_history` parameter contains previous `EstimatorState` values
/// that were returned by prior calls to this `estimate()` function. This vector
/// provides the estimator with access to its own historical outputs for filtering,
/// prediction, and state estimation purposes.
///
/// **Ordering Requirements:**
/// - Elements are ordered chronologically: `state_history[0]` is the oldest state
/// - `state_history[n-1]` is the most recent state (where n = length)
/// - On the first call, `state_history` will be empty (`&[]`)
/// - Each subsequent call appends the previous return value to the end of the vector
///
/// **Length Constraints:**
/// - The vector will retain a fixed maximum number of previous states
/// - When the maximum length is reached, oldest states are removed as new ones are added
/// - The exact maximum length is implementation-defined (FIFO buffer behavior)
/// - The estimator should not assume any specific history length
///
/// **Example Timeline:**
/// ```text
/// Call 1: state_history = []           → returns state_1
/// Call 2: state_history = [state_1]    → returns state_2
/// Call 3: state_history = [state_1, state_2] → returns state_3
/// ...
/// Call N: state_history = [state_k, ..., state_N-1] → returns state_N
///         (length capped at maximum, oldest states dropped)
/// ```
///
/// # Parameters
///
/// - `state_history`: Previous estimator states returned by this function (ordered oldest to newest)
/// - `gyro`: New gyroscope delta angle readout since previous measurement
/// - `fsm_readout`: Current FSM voltage readout
/// - `fgs_readout`: Optional fine guidance system readout. None if FGS
///   data is not available for this cycle.
///
/// # Returns
///
/// Next estimator state containing updated gyro, FSM readout, and FSM command.
///
/// # Caller Responsibilities
///
/// Upon return of the next `EstimatorState`, it is the caller's responsibility to:
/// - Perform the `FsmCommand` adjustment to the FSM hardware at the appropriate time
/// - Push a Line-of-Sight (LOS) update to the payload computer at the required interval
///
/// The estimator only computes the desired FSM command; execution timing and
/// LOS update delivery are handled externally by the calling system.
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

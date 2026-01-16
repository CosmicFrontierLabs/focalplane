//! Protocol definitions and control interfaces for external integrators.
//!
//! Line-of-Sight Control Interface for the meter-sim spacecraft pointing control system.
//!
//! # Overview
//!
//! This module defines the protocol and data structures for the line-of-sight (LOS)
//! control algorithm to interface with the spacecraft simulation system. The
//! [`StateEstimator`] trait provides the interface for state estimation and control logic.
//!
//! # Architecture
//!
//! ## Control Loop Timing
//!
//! - [`GyroTick`] - 500Hz tick counter synchronized with gyroscope measurements
//! - [`Timestamp`] - Microseconds (u64) for sensor timing precision
//!
//! ## Sensor Inputs
//!
//! | Struct | Description | Rate |
//! |--------|-------------|------|
//! | [`GyroReadout`] | 3-axis integrated angles (radians) from Exail gyroscope | Every gyro tick (500Hz) |
//! | [`FgsReadout`] | Fine Guidance System 2D angular position with variance (arcseconds) | Lower rate (camera-based) |
//! | [`FsmReadout`] | Fast Steering Mirror voltage feedback (vx/vy volts) | Every gyro tick |
//!
//! ## Control Output
//!
//! - [`FsmCommand`] - Voltage commands to drive the Fast Steering Mirror (vx/vy volts)
//!
//! ## State Container
//!
//! - [`EstimatorState`] - Packages one complete control cycle: gyro tick, all sensor
//!   readings, and the computed FSM command output
//!
//! # The StateEstimator Trait
//!
//! This is the interface for the LOS control algorithm:
//!
//! ```text
//! f: (state_history, gyro, fsm_readout, fgs_readout?) → FsmCommand
//! ```
//!
//! Key design points:
//!
//! - Receives FIFO history of previous outputs (oldest → newest)
//! - FGS readout is optional (camera rate slower than 500Hz control loop)
//! - Caller handles FSM command execution timing and LOS updates to payload computer
//!
//! # System Context
//!
//! The system implements classic sensor fusion for spacecraft fine pointing:
//!
//! 1. **High-rate gyro measurement** - Exail gyroscope provides integrated angles at 500Hz
//! 2. **Periodic FGS corrections** - Camera-based Fine Guidance System provides absolute reference
//! 3. **FSM actuation** - Fast Steering Mirror commands for fine pointing stabilization
//!
//! ```text
//! +-------------+     +-----------------+     +-------------+
//! | Exail Gyro  |---->|                 |---->|     FSM     |
//! |   (500Hz)   |     |  StateEstimator |     |  (actuator) |
//! +-------------+     |                 |     +-------------+
//!                     |  (LOS control   |
//! +-------------+     |   algorithm)    |     +-------------+
//! |     FGS     |---->|                 |---->|   Payload   |
//! |  (camera)   |     +-----------------+     |  Computer   |
//! +-------------+           ^                 | (LOS update)|
//!                           |                 +-------------+
//!                     +-----+-----+
//!                     |   state   |
//!                     |  history  |
//!                     +-----------+
//! ```

/// Timestamp in microseconds since the initialization of the control loop.
///
/// Maximum representable time: ~584,942 years. If your mission exceeds this
/// duration, congratulations on the interstellar voyage and/or achieving
/// functional immortality. Please file a bug report from Alpha Centauri.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Create timestamp from microseconds.
    pub fn from_micros(micros: u64) -> Self {
        Self(micros)
    }

    /// Get timestamp as microseconds.
    pub fn as_micros(&self) -> u64 {
        self.0
    }
}

/// Gyro tick counter at 500Hz.
///
/// Represents discrete timing ticks synchronized with gyroscope measurements.
/// Maximum representable time: ~99.42 days (2^32 ticks at 500Hz)
///
/// # Behavior
///
/// - Increments by exactly 1 for each `GyroReadout` result
/// - Strictly monotonic (always increasing, never decreases or repeats)
/// - Provides consistent time reference across all sensor measurements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GyroTick(pub u32);

/// Gyroscope readout representing integrated angle on three axes.
///
/// All angular values are in radians, representing the integrated angle
/// as reported by the Exail gyroscope hardware.
///
/// # Timing
///
/// The `timestamp` field represents the time difference between the current
/// XYZ angle measurement and the previous measurement, as reported by the
/// Exail gyroscope hardware. The timestamp is aligned to the first moment
/// the measurement was computed by the gyro.
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

impl GyroReadout {
    /// Create new gyro readout with angles in radians.
    pub fn new(x: f64, y: f64, z: f64, timestamp: Timestamp) -> Self {
        Self { x, y, z, timestamp }
    }

    /// Get angles as an array [x, y, z] in radians.
    pub fn as_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Convert angles to arcseconds.
    pub fn to_arcseconds(&self) -> [f64; 3] {
        const RAD_TO_ARCSEC: f64 = 206264.80624709636;
        [
            self.x * RAD_TO_ARCSEC,
            self.y * RAD_TO_ARCSEC,
            self.z * RAD_TO_ARCSEC,
        ]
    }
}

/// Fine Guidance System 2D angular estimate with uncertainty.
///
/// Represents pointing direction in two angular dimensions with
/// variance estimates for each axis.
///
/// # Timing
///
/// The `timestamp` field corresponds to the center of the image exposure.
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

impl EstimatorState {
    /// Create new estimator state.
    pub fn new(
        gyro_tick: GyroTick,
        gyro: GyroReadout,
        fsm_readout: FsmReadout,
        fgs_readout: Option<FgsReadout>,
        fsm_command: FsmCommand,
    ) -> Self {
        Self {
            gyro_tick,
            gyro,
            fsm_readout,
            fgs_readout,
            fsm_command,
        }
    }
}

/// Trait for line-of-sight state estimator implementation.
///
/// Implement this trait to provide state estimation and control logic
/// for the LOS control algorithm. The estimator processes sensor measurements
/// and state history to compute the next FSM command.
///
/// # Function Signature (Conceptual)
///
/// ```text
/// f: (&[EstimatorState], &GyroReadout, &FsmReadout, Option<&FgsReadout>) → FsmCommand
/// ```
///
/// # State History
///
/// The `state_history` parameter contains previous `EstimatorState` values
/// assembled by the caller from prior calls. This vector provides the estimator
/// with access to historical sensor readings and commands for filtering,
/// prediction, and state estimation purposes.
///
/// **Ordering Requirements:**
///
/// - Elements are ordered chronologically: `state_history[0]` is the oldest state
/// - `state_history[n-1]` is the most recent state (where n = length)
/// - On the first call, `state_history` will be empty (`&[]`)
///
/// **Length Constraints:**
///
/// - The vector will retain a fixed maximum number of previous states
/// - When the maximum length is reached, oldest states are removed as new ones are added
/// - The exact maximum length is implementation-defined (FIFO buffer behavior)
/// - The estimator should not assume any specific history length
///
/// # Parameters
///
/// - `state_history`: Previous estimator states assembled by caller (ordered oldest to newest)
/// - `gyro_readout`: Current gyroscope angle readout
/// - `fsm_readout`: Current FSM voltage readout
/// - `fgs_readout`: Optional fine guidance system readout. None if FGS
///   data is not available for this cycle.
///
/// # Returns
///
/// The computed FSM voltage command for this cycle.
///
/// # Caller Responsibilities
///
/// Upon return of the `FsmCommand`, it is the caller's responsibility to:
///
/// - Assemble an `EstimatorState` from the passed sensor readings and returned command
/// - Append the new state to the history for subsequent calls
/// - Perform the `FsmCommand` adjustment to the FSM hardware at the appropriate time
/// - Push a Line-of-Sight (LOS) update to the payload computer at the required interval
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_roundtrip() {
        let ts = Timestamp::from_micros(12345678);
        assert_eq!(ts.as_micros(), 12345678);
    }

    #[test]
    fn test_gyro_readout_conversions() {
        let gyro = GyroReadout::new(0.001, 0.002, 0.003, Timestamp::from_micros(1000));
        let arr = gyro.as_array();
        assert_eq!(arr, [0.001, 0.002, 0.003]);

        let arcsec = gyro.to_arcseconds();
        // 1 radian ≈ 206264.8 arcseconds
        assert!((arcsec[0] - 206.26480624709636).abs() < 0.001);
    }

    #[test]
    fn test_fgs_readout_conversions() {
        let fgs = FgsReadout::new(1.0, 2.0, 0.01, 0.04, Timestamp::from_micros(1000));
        let radians = fgs.to_radians();
        // 1 arcsec ≈ 4.848e-6 radians
        assert!((radians[0] - 4.84813681109536e-6).abs() < 1e-12);

        let std = fgs.std_dev();
        assert!((std[0] - 0.1).abs() < 1e-10);
        assert!((std[1] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_fsm_types() {
        let readout = FsmReadout::new(1.5, 2.5, Timestamp::from_micros(1000));
        assert_eq!(readout.vx, 1.5);
        assert_eq!(readout.vy, 2.5);

        let cmd = FsmCommand::new(3.0, 4.0);
        assert_eq!(cmd.vx, 3.0);
        assert_eq!(cmd.vy, 4.0);
    }

    #[test]
    fn test_estimator_state() {
        let gyro = GyroReadout::new(0.0, 0.0, 0.0, Timestamp::from_micros(1000));
        let fsm_readout = FsmReadout::new(0.0, 0.0, Timestamp::from_micros(1000));
        let fsm_cmd = FsmCommand::new(0.0, 0.0);

        let state = EstimatorState::new(GyroTick(1), gyro, fsm_readout, None, fsm_cmd);
        assert_eq!(state.gyro_tick, GyroTick(1));
        assert!(state.fgs_readout.is_none());
    }

    /// Dummy estimator for testing the trait
    struct ZeroEstimator;

    impl StateEstimator for ZeroEstimator {
        fn estimate(
            &self,
            _state_history: &[EstimatorState],
            _gyro_readout: &GyroReadout,
            _fsm_readout: &FsmReadout,
            _fgs_readout: Option<&FgsReadout>,
        ) -> FsmCommand {
            FsmCommand::new(0.0, 0.0)
        }
    }

    #[test]
    fn test_state_estimator_trait() {
        let estimator = ZeroEstimator;
        let gyro = GyroReadout::new(0.0, 0.0, 0.0, Timestamp::from_micros(1000));
        let fsm = FsmReadout::new(0.0, 0.0, Timestamp::from_micros(1000));

        let cmd = estimator.estimate(&[], &gyro, &fsm, None);
        assert_eq!(cmd.vx, 0.0);
        assert_eq!(cmd.vy, 0.0);
    }
}

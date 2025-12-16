//! Line of Sight (LOS) Feedback Controller
//!
//! A discrete-time state-space feedback controller for LOS stabilization.
//! Designed to run at 40Hz, uses a 5-state compensator for X and Y channels independently.
//!
//! # Operation
//! 1. Create controller with `LosController::new()`
//! 2. Enable control with `set_enabled(true)`
//! 3. Call `update()` each frame with centroid measurements
//! 4. Use `set_hold(true)` to pause control (maintains last output)
//! 5. Set command setpoint with `set_command()` to jog the centroid
//!
//! # Units
//! - Input: centroid position in pixels
//! - Output: FSM command in microradians

use nalgebra::{Matrix5, RowVector5, Vector5};

/// State-space matrices for the feedback compensator (40Hz design)
/// These are identical for X and Y channels
/// Coefficients copied verbatim from CF_LOS_FB_40Hz.h to ensure numerical equivalence
#[allow(clippy::excessive_precision)]
mod coefficients {
    use nalgebra::{Matrix5, RowVector5, Vector5};

    /// State transition matrix A (5x5)
    pub fn a_matrix() -> Matrix5<f64> {
        #[rustfmt::skip]
        let a = Matrix5::new(
            9.999999999999523e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 5.819362123130311e-01, 3.497029269294699e-01, -1.677023981165541e-01, -5.351926461854359e-02,
            0.000000000000000e+00, 3.497029269294702e-01, 5.558793564483747e-01, 4.869139212556047e-01, 7.259421051576656e-02,
            0.000000000000000e+00, 1.677023981165544e-01, -4.869139212556047e-01, 5.476727449843610e-01, -1.745555043235154e-01,
            0.000000000000000e+00, 5.351926461854263e-02, -7.259421051576770e-02, -1.745555043235147e-01, 8.588662137791048e-01,
        );
        a
    }

    /// Input matrix B (5x1)
    pub fn b_vector() -> Vector5<f64> {
        Vector5::new(
            1.769322751627364e+03,
            -9.711041417320714e-01,
            4.306078653167560e-01,
            1.783269216694375e-02,
            3.846046698903809e-02,
        )
    }

    /// Output matrix C (1x5)
    pub fn c_vector() -> RowVector5<f64> {
        RowVector5::new(
            7.064850089399111e-04,
            -9.711041417320715e-01,
            4.306078653167552e-01,
            -1.783269216694382e-02,
            -3.846046698903713e-02,
        )
    }

    /// Feedthrough scalar D
    pub const D: f64 = 0.0;
}

/// Single-axis state-space controller using nalgebra
#[derive(Debug, Clone)]
struct AxisController {
    /// Current state vector
    state: Vector5<f64>,
    /// State transition matrix
    a: Matrix5<f64>,
    /// Input matrix
    b: Vector5<f64>,
    /// Output matrix
    c: RowVector5<f64>,
}

impl Default for AxisController {
    fn default() -> Self {
        Self::new()
    }
}

impl AxisController {
    /// Create a new axis controller with zero initial state
    fn new() -> Self {
        Self {
            state: Vector5::zeros(),
            a: coefficients::a_matrix(),
            b: coefficients::b_vector(),
            c: coefficients::c_vector(),
        }
    }

    /// Reset state to zero
    fn reset(&mut self) {
        self.state = Vector5::zeros();
    }

    /// Compute one step of the controller
    ///
    /// # Arguments
    /// * `error` - The tracking error (command - measurement) in pixels
    ///
    /// # Returns
    /// The control output in microradians
    fn update(&mut self, error: f64) -> f64 {
        // Compute output: u = C * x + D * error
        let output = (self.c * self.state)[0] + coefficients::D * error;

        // Compute next state: x[k+1] = A * x[k] + B * error
        self.state = self.a * self.state + self.b * error;

        output
    }
}

/// LOS Feedback Controller output
#[derive(Debug, Clone, Copy, Default)]
pub struct LosControlOutput {
    /// X-axis FSM command in microradians
    pub u_x: f64,
    /// Y-axis FSM command in microradians
    pub u_y: f64,
}

/// Line of Sight Feedback Controller
///
/// Implements a discrete-time state-space feedback controller for stabilizing
/// the line of sight. Uses separate X and Y axis controllers.
#[derive(Debug, Clone)]
pub struct LosController {
    /// X-axis controller
    x_controller: AxisController,
    /// Y-axis controller
    y_controller: AxisController,
    /// Control enable flag
    enabled: bool,
    /// Hold flag - when true, output is frozen at last value
    hold: bool,
    /// Command setpoint X in pixels
    command_x: f64,
    /// Command setpoint Y in pixels
    command_y: f64,
    /// Previous output (used for hold mode)
    previous_output: LosControlOutput,
}

impl Default for LosController {
    fn default() -> Self {
        Self::new()
    }
}

impl LosController {
    /// Create a new LOS controller
    pub fn new() -> Self {
        Self {
            x_controller: AxisController::new(),
            y_controller: AxisController::new(),
            enabled: false,
            hold: false,
            command_x: 0.0,
            command_y: 0.0,
            previous_output: LosControlOutput::default(),
        }
    }

    /// Enable or disable the controller
    ///
    /// When disabled, outputs zero and resets internal state.
    /// When enabled, begins tracking from zero state.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.reset();
        }
    }

    /// Check if controller is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set hold mode
    ///
    /// When hold is true, the controller freezes its output at the last computed value.
    /// Internal state is preserved, so control can resume where it left off.
    pub fn set_hold(&mut self, hold: bool) {
        self.hold = hold;
    }

    /// Check if controller is in hold mode
    pub fn is_hold(&self) -> bool {
        self.hold
    }

    /// Set the command setpoint in pixels
    ///
    /// The controller will try to drive the measured centroid to this position.
    pub fn set_command(&mut self, x: f64, y: f64) {
        self.command_x = x;
        self.command_y = y;
    }

    /// Get current command setpoint
    pub fn command(&self) -> (f64, f64) {
        (self.command_x, self.command_y)
    }

    /// Reset the controller state
    ///
    /// Clears internal state and previous output.
    pub fn reset(&mut self) {
        self.x_controller.reset();
        self.y_controller.reset();
        self.previous_output = LosControlOutput::default();
    }

    /// Update the controller with new measurements
    ///
    /// # Arguments
    /// * `meas_x` - Measured centroid X position in pixels
    /// * `meas_y` - Measured centroid Y position in pixels
    ///
    /// # Returns
    /// Control output with FSM commands in microradians
    pub fn update(&mut self, meas_x: f64, meas_y: f64) -> LosControlOutput {
        // Hold mode: return previous output without updating state
        if self.hold {
            return self.previous_output;
        }

        // Disabled: reset and return zero
        if !self.enabled {
            self.reset();
            return LosControlOutput::default();
        }

        // Compute errors
        let error_x = self.command_x - meas_x;
        let error_y = self.command_y - meas_y;

        // Update controllers
        let u_x = self.x_controller.update(error_x);
        let u_y = self.y_controller.update(error_y);

        let output = LosControlOutput { u_x, u_y };
        self.previous_output = output;

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_controller_default_disabled() {
        let controller = LosController::new();
        assert!(!controller.is_enabled());
        assert!(!controller.is_hold());
    }

    #[test]
    fn test_controller_outputs_zero_when_disabled() {
        let mut controller = LosController::new();
        let output = controller.update(10.0, 20.0);
        assert_eq!(output.u_x, 0.0);
        assert_eq!(output.u_y, 0.0);
    }

    #[test]
    fn test_controller_outputs_nonzero_when_enabled_with_error() {
        let mut controller = LosController::new();
        controller.set_enabled(true);
        controller.set_command(0.0, 0.0);

        // First call: output uses x[k] which is zero, so output is zero
        // But state gets updated with the error
        let _output1 = controller.update(1.0, 1.0);

        // Second call: now x[k] has values from previous iteration
        let output2 = controller.update(1.0, 1.0);

        // With accumulated state, we expect nonzero output
        assert_ne!(output2.u_x, 0.0);
        assert_ne!(output2.u_y, 0.0);
    }

    #[test]
    fn test_controller_hold_freezes_output() {
        let mut controller = LosController::new();
        controller.set_enabled(true);

        // Get some output
        let output1 = controller.update(1.0, 1.0);

        // Enable hold
        controller.set_hold(true);

        // Output should be frozen even with different input
        let output2 = controller.update(100.0, 100.0);
        assert_eq!(output1.u_x, output2.u_x);
        assert_eq!(output1.u_y, output2.u_y);
    }

    #[test]
    fn test_controller_reset_clears_state() {
        let mut controller = LosController::new();
        controller.set_enabled(true);

        // Build up some state
        for _ in 0..10 {
            controller.update(1.0, 1.0);
        }

        // Reset
        controller.reset();
        controller.set_enabled(true);

        // First output after reset should match fresh controller
        let mut fresh = LosController::new();
        fresh.set_enabled(true);

        let output_reset = controller.update(1.0, 1.0);
        let output_fresh = fresh.update(1.0, 1.0);

        assert_eq!(output_reset.u_x, output_fresh.u_x);
        assert_eq!(output_reset.u_y, output_fresh.u_y);
    }

    #[test]
    fn test_axis_symmetry() {
        // X and Y controllers use same coefficients, so symmetric input should give symmetric output
        let mut controller = LosController::new();
        controller.set_enabled(true);

        let output = controller.update(5.0, 5.0);
        assert_eq!(output.u_x, output.u_y);
    }

    /// Reference values from C implementation (main.c in reference/ directory)
    /// Test 1: Step response with constant error of -1 (cmd=0, meas=1)
    /// These are the first 20 output values from the C reference implementation
    #[rustfmt::skip]
    const C_REFERENCE_STEP_RESPONSE: [(f64, f64); 20] = [
        (0.000000000000000e+00, 0.000000000000000e+00),
        (-2.376669175331946e+00, -2.376669175331946e+00),
        (-4.004549395017822e+00, -4.004549395017822e+00),
        (-5.299126433152320e+00, -5.299126433152320e+00),
        (-6.589907107464914e+00, -6.589907107464914e+00),
        (-7.979959624053390e+00, -7.979959624053390e+00),
        (-9.430870714191737e+00, -9.430870714191737e+00),
        (-1.087799269322810e+01, -1.087799269322810e+01),
        (-1.228753789339230e+01, -1.228753789339230e+01),
        (-1.365851390494384e+01, -1.365851390494384e+01),
        (-1.500363659158541e+01, -1.500363659158541e+01),
        (-1.633411736982637e+01, -1.633411736982637e+01),
        (-1.765497136950826e+01, -1.765497136950826e+01),
        (-1.896693536970881e+01, -1.896693536970881e+01),
        (-2.026962511697034e+01, -2.026962511697034e+01),
        (-2.156325446590014e+01, -2.156325446590014e+01),
        (-2.284881347163102e+01, -2.284881347163102e+01),
        (-2.412759164289418e+01, -2.412759164289418e+01),
        (-2.540075769944627e+01, -2.540075769944627e+01),
        (-2.666921783953972e+01, -2.666921783953972e+01),
    ];

    #[test]
    fn test_matches_c_reference_step_response() {
        let mut controller = LosController::new();
        controller.set_enabled(true);
        controller.set_command(0.0, 0.0);

        for (expected_u_x, expected_u_y) in C_REFERENCE_STEP_RESPONSE.iter() {
            let output = controller.update(1.0, 1.0);
            assert_relative_eq!(output.u_x, *expected_u_x, epsilon = 1e-10);
            assert_relative_eq!(output.u_y, *expected_u_y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_matches_c_reference_hold_mode() {
        let mut controller = LosController::new();
        controller.set_enabled(true);
        controller.set_command(0.0, 0.0);

        // Run for 5 steps to build up state (matches C reference Test 3)
        for _ in 0..5 {
            controller.update(1.0, 1.0);
        }

        // Get the output just before hold - should be step 5's output
        let last_output = controller.update(1.0, 1.0);
        let expected_hold_value = -7.979959624053390e+00;
        assert_relative_eq!(last_output.u_x, expected_hold_value, epsilon = 1e-10);

        // Enable hold
        controller.set_hold(true);

        // Output should stay frozen at previous value even with different input
        for _ in 0..5 {
            let output = controller.update(5.0, 5.0);
            assert_eq!(output.u_x, last_output.u_x);
            assert_eq!(output.u_y, last_output.u_y);
        }

        // Release hold - controller should resume from preserved state
        controller.set_hold(false);
        let resumed = controller.update(1.0, 1.0);

        // After hold release, should continue as if hold never happened
        let expected_after_hold = -9.430870714191737e+00;
        assert_relative_eq!(resumed.u_x, expected_after_hold, epsilon = 1e-10);
    }

    /// Test sinusoidal disturbance response matches C reference
    /// Uses first few points from Test 2 in C reference
    #[test]
    fn test_matches_c_reference_sinusoidal() {
        let mut controller = LosController::new();
        controller.set_enabled(true);
        controller.set_command(0.0, 0.0);

        // Expected values from C reference Test 2 (first 10 steps)
        #[rustfmt::skip]
        let expected: [(f64, f64, f64, f64); 10] = [
            // (meas_x, meas_y, u_x, u_y)
            (0.000000000000000e+00, 1.500000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00),
            (3.128689300804615e-01, 1.495376000599692e+00, 0.000000000000000e+00, -3.565003762997919e+00),
            (6.180339887498943e-01, 1.481532510892707e+00, -7.435859420413187e-01, -5.995834375685266e+00),
            (9.079809994790926e-01, 1.458554880596515e+00, -1.978175473001591e+00, -7.897271220461795e+00),
            (1.175570504584945e+00, 1.426584774442730e+00, -3.569088691456440e+00, -9.750310339770142e+00),
            (1.414213562373094e+00, 1.385819298766930e+00, -5.475964269742095e+00, -1.169811188600438e+01),
            (1.618033988749894e+00, 1.336509786282552e+00, -7.682907677643916e+00, -1.367150614477236e+01),
            (1.782013048376735e+00, 1.278960246531139e+00, -1.015461740082689e+01, -1.556163636727841e+01),
            (1.902113032590306e+00, 1.213525491562422e+00, -1.282904624763556e+01, -1.730615024966599e+01),
            (1.975376681190275e+00, 1.140608948400047e+00, -1.562858429733942e+01, -1.889262991412671e+01),
        ];

        for (meas_x, meas_y, expected_u_x, expected_u_y) in expected.iter() {
            let output = controller.update(*meas_x, *meas_y);
            assert_relative_eq!(output.u_x, *expected_u_x, epsilon = 1e-10);
            assert_relative_eq!(output.u_y, *expected_u_y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_extended_step_response_100_iterations() {
        // Run 100 iterations and check final value matches C reference
        let mut controller = LosController::new();
        controller.set_enabled(true);
        controller.set_command(0.0, 0.0);

        let mut last_output = LosControlOutput::default();
        for _ in 0..100 {
            last_output = controller.update(1.0, 1.0);
        }

        // Step 99 from C reference
        let expected_u_x = -1.266455538403867e+02;
        assert_relative_eq!(last_output.u_x, expected_u_x, epsilon = 1e-10);
    }
}

/// Tests that compare against the actual C implementation (linked at build time)
#[cfg(test)]
mod c_reference_tests {
    use super::*;
    use approx::assert_relative_eq;

    extern "C" {
        fn CF_LOS_FB_function(
            u_x: *mut f64,
            u_y: *mut f64,
            pc_onoff_flag: i32,
            hold_flag: i32,
            los_cmd_x: f64,
            los_cmd_y: f64,
            los_meas_x: f64,
            los_meas_y: f64,
        );
    }

    /// Wrapper around the C function for easier use
    struct CController;

    impl CController {
        fn reset() {
            let mut u_x = 0.0;
            let mut u_y = 0.0;
            unsafe {
                CF_LOS_FB_function(&mut u_x, &mut u_y, 0, 0, 0.0, 0.0, 0.0, 0.0);
            }
        }

        fn update(cmd_x: f64, cmd_y: f64, meas_x: f64, meas_y: f64) -> (f64, f64) {
            let mut u_x = 0.0;
            let mut u_y = 0.0;
            unsafe {
                CF_LOS_FB_function(&mut u_x, &mut u_y, 1, 0, cmd_x, cmd_y, meas_x, meas_y);
            }
            (u_x, u_y)
        }
    }

    /// Single test that runs both step response and sinusoidal to avoid
    /// issues with C static state persisting between parallel test runs
    #[test]
    fn test_rust_matches_c_implementation() {
        // Part 1: Step response
        {
            CController::reset();
            let mut rust_controller = LosController::new();
            rust_controller.set_enabled(true);
            rust_controller.set_command(0.0, 0.0);

            for _ in 0..100 {
                let (c_u_x, c_u_y) = CController::update(0.0, 0.0, 1.0, 1.0);
                let rust_output = rust_controller.update(1.0, 1.0);
                assert_relative_eq!(rust_output.u_x, c_u_x, epsilon = 1e-14);
                assert_relative_eq!(rust_output.u_y, c_u_y, epsilon = 1e-14);
            }
        }

        // Part 2: Sinusoidal
        {
            CController::reset();
            let mut rust_controller = LosController::new();
            rust_controller.set_enabled(true);
            rust_controller.set_command(0.0, 0.0);

            for i in 0..200 {
                let t = i as f64 / 40.0;
                let meas_x = 2.0 * (2.0 * std::f64::consts::PI * 1.0 * t).sin();
                let meas_y = 1.5 * (2.0 * std::f64::consts::PI * 0.5 * t).cos();

                let (c_u_x, c_u_y) = CController::update(0.0, 0.0, meas_x, meas_y);
                let rust_output = rust_controller.update(meas_x, meas_y);
                assert_relative_eq!(rust_output.u_x, c_u_x, epsilon = 1e-14);
                assert_relative_eq!(rust_output.u_y, c_u_y, epsilon = 1e-14);
            }
        }
    }
}

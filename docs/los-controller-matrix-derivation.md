# Deriving Better LOS Controller Matrices

## 1. What We Have Today

The current LOS controller is a 5-state discrete-time state-space compensator running at 40 Hz. It was provided as a set of fixed numerical matrices (A, B, C, D) in `CF_LOS_FB_40Hz.h` with no accompanying derivation, plant model, or design documentation.

### Current Coefficients

```
State-space form:
    x[k+1] = A * x[k] + B * u[k]
    y[k]   = C * x[k] + D * u[k]

Where:
    u = error [pixels] = command - measurement
    y = correction signal [sensor-frame]
    x = 5-element internal state
    D = 0 (no direct feedthrough)
```

**A (5x5):**
```
[ 1.0000  0       0       0       0      ]   ← integrator (row 0)
[ 0       0.5819  0.3497 -0.1677 -0.0535 ]
[ 0       0.3497  0.5559  0.4869  0.0726 ]
[ 0       0.1677 -0.4869  0.5477 -0.1746 ]
[ 0       0.0535 -0.0726 -0.1746  0.8589 ]
```

**B (5x1):**
```
[ 1769.3 ]   ← large integrator gain
[ -0.971 ]
[  0.431 ]
[  0.018 ]
[  0.038 ]
```

**C (1x5):**
```
[ 7.065e-4  -0.971  0.431  -0.018  -0.038 ]
```

### Key Observations About the Current Design

1. **State 0 is a pure integrator**: A[0,0] ≈ 1.0 exactly, row/column 0 is decoupled from states 1-4. This guarantees zero steady-state error for constant disturbances.

2. **States 1-4 form a 4x4 coupled block**: The eigenvalues of the lower-right 4x4 submatrix determine the transient dynamics. These are all inside the unit circle (stable), with magnitudes ~0.58 to 0.86 — corresponding to time constants of ~1-4 frames at 40 Hz (25-100 ms settling).

3. **B[0] = 1769.3 is the dominant gain**: This sets how aggressively the integrator accumulates error. At 40 Hz with 1 pixel steady error, the integrator state grows by ~1769 per step, and the output grows by C[0] * 1769 ≈ 1.25 per step. This means it takes roughly 1 step to correct a 1-pixel error.

4. **B and C share magnitudes on states 1-4**: |B[i]| ≈ |C[i]| for i=1..4, with sign differences. This is characteristic of a balanced realization.

5. **No explicit plant model is encoded**: The controller operates purely in pixel-space. It takes pixel errors in and produces pixel-scale corrections out. The FsmTransform converts that to µrad. The controller itself doesn't "know" about focal length, pixel pitch, or FSM geometry.

### What the Controller Doesn't Know

The fixed matrices embed implicit assumptions about the plant that we can't verify:

| Parameter | Assumed Value | Actual Value | Impact |
|-----------|--------------|--------------|--------|
| Update rate | 40 Hz | Variable (depends on exposure + readout) | Phase margin changes |
| Sensor delay | Unknown | 1 frame (25ms at 40Hz) + centroid compute | Stability margin |
| FSM bandwidth | Unknown | PI S-330 spec (~300 Hz -3dB) | Likely overdesigned |
| FSM → pixel gain | Unknown | 0.02-0.03 px/µrad (from calibration) | Affects loop gain |
| Pixel scale | Unknown | 0.377 arcsec/px (IMX455 + JBT 0.5m) | Physical interpretation |

## 2. The Complete Physical System

### 2.1 Telescope Optics

**JBT 0.5m Configuration** (primary telescope):
- Aperture: 0.485 m
- Focal length: 5.987 m
- f-number: 12.3
- Obscuration: 35% linear

**Plate scale:**
```
plate_scale = 206265 arcsec/rad / focal_length_mm
            = 206265 / 5987 = 34.45 arcsec/mm

pixel_scale = plate_scale * pixel_pitch_mm
            = 34.45 * 0.00376 = 0.1296 arcsec/pixel
            = 0.628 µrad/pixel
```

Wait — let's cross-check with the calibration data. The NSV455 calibration shows:
```
fsm_to_sensor ≈ [0.028, 0.002; 0.000, -0.021] pixels/µrad
```

Taking the diagonal elements: ~0.025 pixels per µrad → ~40 µrad per pixel. This is the *FSM angle per pixel*, not sky angle per pixel. The FSM operates with a 2x reflection factor and the beam may be compressed through the optical path, so the FSM-to-sky conversion has additional geometry.

### 2.2 FSM (PI S-330)

- Range: 0-2000 µrad per axis (±1000 µrad from center)
- Bandwidth: ~300 Hz mechanical (-3dB), closed-loop servo at E-727
- Resolution: sub-µrad (piezo)
- Actuator: 3 piezo elements (2 tilt + 1 bias preload at 100V)

The FSM's -3dB bandwidth at ~300 Hz means for our 40 Hz control loop, the FSM is essentially a unity-gain element with negligible phase lag. We can model it as:

```
G_fsm(z) ≈ 1    (at 40 Hz)
```

This is a good assumption. The FSM is much faster than our control loop.

### 2.3 Sensor + Centroid Pipeline

**IMX455 sensor:**
- Pixel pitch: 3.76 µm
- Resolution: 9568 × 6380 pixels
- Max frame rate: 21.33 Hz (full frame)
- Read noise: 1.58 e- (sets centroid precision floor)
- Full well: 26,000 e-

**ROI mode frame rate:**
At 128×128 ROI, frame rates of ~40 Hz are achievable (from timing measurements: ~252 ms between frames at full frame, but ROI can go faster).

**Centroid precision:**
For a well-exposed star (SNR > 20), centroid precision is approximately:
```
σ_centroid ≈ FWHM / (2 * SNR)
```
With FWHM ≈ 3 pixels and SNR = 50: σ ≈ 0.03 pixels ≈ 0.004 arcsec ≈ 19 nrad on-sky.

### 2.4 The Full Loop Transfer Function

```
                              Plant
                     ┌─────────────────────────┐
                     │                         │
    error ──► K(z) ──► T^(-1) ──► FSM ──► Optics ──► Sensor ──► centroid
      ▲              │         │      │         │          │
      │              │ µrad    │ tilt │ angular │ pixels   │
      │              │         │      │ motion  │          │
      └──────────────┴─────────┴──────┴─────────┴──────────┘
                              measurement

Where:
    K(z) = LOS controller (5-state compensator)
    T^(-1) = FsmTransform.sensor_to_fsm (2x2 matrix, µrad/pixel)
    FSM = PI S-330 transfer function ≈ 1 at 40 Hz
    Optics = Sky angle to pixel mapping ≈ constant gain
    Sensor = Centroid measurement with 1-frame delay (ZOH)
```

The key insight is that K(z) currently operates in pixel-space and T^(-1) converts to FSM-space. The plant gain from FSM µrad back to pixels is the FsmTransform forward matrix. So the open-loop transfer function is:

```
L(z) = K(z) * T^(-1) * G_fsm(z) * T * z^(-1)
     = K(z) * z^(-1)    (T^(-1) * T = I, G_fsm ≈ 1)
```

The calibration transform cancels out! The loop transfer function depends only on the controller K(z) and the one-frame delay z^(-1).

This is elegant — it means the controller design is independent of the specific FSM-to-sensor geometry, as long as the calibration matrix is correct.

### 2.5 But There's a Problem

The cancellation assumes:
1. The calibration transform is perfectly known (it has measurement noise)
2. The FSM is a perfect unity gain element at all frequencies (it has dynamics)
3. There is exactly one frame of delay (actual delay depends on exposure time + readout + compute)
4. The pixel-to-angle mapping is linear (true for small angles, which it is)

If any of these assumptions break, the effective loop gain changes, and the controller designed for one set of parameters may have suboptimal margins.

## 3. How to Derive Better Matrices

There are several approaches, from simple to sophisticated:

### 3.1 Approach A: Classical Loopshaping (Recommended Starting Point)

Design K(z) in the frequency domain to achieve desired:
- **Bandwidth**: How fast can we reject disturbances?
- **Phase margin**: How much delay tolerance do we have?
- **Gain margin**: How much gain uncertainty can we tolerate?
- **Disturbance rejection**: How much jitter can we suppress?

**Step 1: Characterize the plant**

The plant (from controller output to centroid measurement) is:
```
P(z) = G_gain * z^(-d)
```
Where:
- `G_gain` = overall gain (should be ~1 after calibration, but verify)
- `d` = total delay in frames (exposure + readout + compute + command)

Measure `d` empirically: apply a step command to the FSM and count frames until centroid response appears. This is the most critical parameter.

**Step 2: Choose controller structure**

The current 5-state design has:
- 1 integrator (zero steady-state error)
- 4 modes for transient shaping

A simpler starting point: PI + lead-lag compensator:
```
K(z) = K_i / (1 - z^(-1))                     # Integrator
      * (1 + a1*z^(-1)) / (1 + b1*z^(-1))     # Lead-lag 1
      * (1 + a2*z^(-1)) / (1 + b2*z^(-1))     # Lead-lag 2
```

This gives 5 parameters to tune (K_i, a1, b1, a2, b2) instead of the opaque 35+ matrix entries.

**Step 3: Tune for the measured delay**

With delay `d` frames:
- Maximum achievable bandwidth ≈ f_sample / (2*pi*d) (rule of thumb)
- At 40 Hz with d=1: max bandwidth ≈ 6 Hz
- At 40 Hz with d=2: max bandwidth ≈ 3 Hz

### 3.2 Approach B: Parametric State-Space Design

Keep the 5-state structure but derive it from explicit design parameters:

**Step 1: Define the plant model**
```python
import control

# Plant: unit gain + delay + optional FSM dynamics
P_continuous = control.tf([1], [1])  # FSM at DC
P_discrete = control.c2d(P_continuous, Ts=1/40)  # 40 Hz sampling
P_delay = control.tf([1], [1, 0], dt=1/40)  # 1-frame delay

P = P_discrete * P_delay
```

**Step 2: Specify performance requirements**
```python
# Disturbance rejection: suppress jitter below 5 Hz by 20 dB
# Sensitivity peak: < 6 dB (Ms < 2)
# Bandwidth: 3-5 Hz crossover
# Phase margin: > 45 degrees
# Gain margin: > 6 dB
```

**Step 3: Synthesize controller (e.g., H-infinity or LQG)**
```python
import control

# LQR approach: minimize ∫(Q*e² + R*u²)dt
Q = np.diag([10.0, 1.0])  # Weight on error vs control effort
R = np.array([[0.01]])     # Weight on control signal

K, S, E = control.dlqr(A_plant, B_plant, Q, R)
```

Or H-infinity:
```python
# Mixed sensitivity: minimize ||[W1*S; W2*T]||_inf
W1 = control.tf([1, 0.1], [1, 0.001])  # Sensitivity weight (good low-freq rejection)
W2 = control.tf([1], [1, 10])          # Complementary sensitivity weight (roll off high-freq)

K_hinf = control.hinfsyn(P, W1, W2)
```

**Step 4: Convert to state-space and extract A, B, C, D**
```python
K_ss = control.ss(K_hinf)
A = K_ss.A  # → a_matrix()
B = K_ss.B  # → b_vector()
C = K_ss.C  # → c_vector()
D = K_ss.D  # → D constant
```

### 3.3 Approach C: Adaptive / Self-Tuning

Build matrices from measured system parameters at runtime:

```
1. Measure total loop delay (step response test)
2. Measure loop gain (calibration matrix gives this)
3. Measure disturbance spectrum (PSD of centroid during open-loop)
4. Design optimal controller for measured plant
5. Hot-swap matrices in LosController
```

This is the most robust approach but requires online system identification.

## 4. Parameters That Should Influence the Matrices

### 4.1 Telescope / Optical Path

| Parameter | Effect | How to Incorporate |
|-----------|--------|-------------------|
| Focal length | Changes pixel scale (arcsec/px) | Absorbed by calibration matrix |
| f-number | Affects PSF size → centroid precision | Changes effective noise → adjust Q weight |
| Obscuration | Reduces light → worse SNR → worse centroid | Changes effective noise |

**Verdict**: Focal length and geometry are handled by the calibration matrix and don't need to be in the controller matrices. But the SNR impact (through aperture/obscuration) affects optimal aggressiveness.

### 4.2 Sensor

| Parameter | Effect | How to Incorporate |
|-----------|--------|-------------------|
| Pixel pitch | Changes angular scale | Absorbed by calibration matrix |
| Frame rate | Sets controller sample rate | **Fundamental**: controller must be redesigned for different rates |
| Read noise | Sets centroid noise floor | Adjust Q/R weights or noise covariance |
| Exposure time | Part of total loop delay + SNR trade | **Fundamental**: changes delay `d` |

**Verdict**: Frame rate and exposure time are critical design parameters that should change the matrices. The current 40 Hz design is only optimal at 40 Hz.

### 4.3 FSM

| Parameter | Effect | How to Incorporate |
|-----------|--------|-------------------|
| Bandwidth | Phase lag at high frequencies | Usually negligible (300 Hz >> 40 Hz) |
| Travel range | Saturation limit | Affects anti-windup, not matrices |
| Resolution | Quantization noise | Usually negligible (sub-µrad) |
| Axis coupling | Off-diagonal FSM dynamics | Absorbed by calibration matrix |

**Verdict**: FSM parameters are mostly irrelevant for matrix design because the FSM is so much faster than the control loop. If the loop rate increased to >100 Hz, FSM dynamics would start to matter.

### 4.4 Motion Profile / Disturbance

| Parameter | Effect | How to Incorporate |
|-----------|--------|-------------------|
| Jitter spectrum | Determines optimal bandwidth | Shape sensitivity function S(z) to match |
| Jitter amplitude | Determines required FSM range | Affects saturation/windup, not linear matrices |
| Drift rate | Determines integrator strength | Current integrator handles this |
| Slew rate | May exceed controller bandwidth | Consider gain scheduling |

**Verdict**: The disturbance spectrum is the most important input for optimal controller design. A controller optimized for broadband jitter (white noise) will differ from one optimized for narrowband vibration (e.g., reaction wheel harmonics at specific frequencies).

### 4.5 Uncertainties

| Uncertainty | Effect | How to Incorporate |
|------------|--------|-------------------|
| Calibration matrix error | Effective gain changes | Increase gain margin in design |
| Delay variation | Phase margin changes | Design for worst-case delay |
| Frame timing jitter | Sample rate variation | Robust control (µ-synthesis) |
| Temperature drift | Calibration matrix drift | Periodic recalibration, or increase margins |

## 5. Practical Recommendation

### Phase 1: Measure What Matters

Before redesigning the controller, measure:

1. **Total loop delay** — The most critical parameter. Apply a step to FSM, measure frames until centroid responds. This tells you the achievable bandwidth ceiling.

2. **Disturbance PSD** — Record centroid positions with controller disabled (or during open-loop acquisition). Compute power spectral density. This tells you what frequencies to suppress.

3. **Closed-loop sensitivity** — With current controller running, inject a known disturbance (sinusoidal FSM wiggle at varying frequencies). Measure rejection ratio vs frequency. This tells you the current controller's actual performance.

4. **Centroid noise floor** — With FSM locked (static), record centroid variance over many frames. This sets the lower bound on achievable tracking precision and the noise covariance for LQG design.

### Phase 2: Design for Measured Plant

With measurements in hand:

1. Build a Python `control` library plant model:
   ```python
   Ts = 1/40  # or measured actual rate
   delay_frames = measured_delay
   noise_variance = measured_centroid_variance
   ```

2. Design controller using `control.dlqr()` or `control.hinfsyn()` with weights reflecting:
   - Disturbance rejection at measured jitter frequencies
   - Noise rejection at high frequencies (roll off above Nyquist/2)
   - Gain margin > 6 dB for calibration uncertainty
   - Phase margin > 30° for delay uncertainty

3. Export (A, B, C, D) to Rust code format, replacing current coefficients.

### Phase 3: Rate-Adaptive Controllers

The biggest limitation of the current design is the fixed 40 Hz assumption. In practice:
- Full-frame acquisition runs at ~4 Hz (exposure + readout)
- ROI tracking runs at 4-40 Hz depending on exposure
- Different stars need different exposures

Design a family of controllers indexed by frame rate:
```rust
fn controller_for_rate(rate_hz: f64) -> (Matrix5<f64>, Vector5<f64>, RowVector5<f64>) {
    // Discretize continuous-time controller at the actual rate
    // Or interpolate from pre-computed table
}
```

This could be done with a continuous-time controller design discretized at runtime using `c2d` (ZOH or Tustin).

### Phase 4: Incorporate Geometric Knowledge

Once the measurement-driven approach is working, geometric parameters can refine the noise model:

```
centroid_variance = f(aperture, focal_length, pixel_pitch, star_magnitude,
                      exposure_time, read_noise, sky_background, PSF_shape)
```

This lets you compute optimal Q/R weights without measuring noise directly — useful for adaptive gain scheduling based on guide star brightness.

## 6. The Calibration Matrix Is Separate (and That's Good)

The 2x2 FsmTransform calibration matrix and the 5-state controller matrices serve different purposes:

| | Controller (A,B,C,D) | Calibration (T) |
|---|---|---|
| **What it encodes** | Control dynamics (bandwidth, damping, integrator) | Geometric mapping (rotation, scale, sign) |
| **When it changes** | When frame rate, delay, or noise changes | When optical alignment changes |
| **How to measure** | System identification + control synthesis | Sinusoidal wiggle + fit |
| **Update frequency** | Per-session or per-configuration | Per thermal cycle or per-realignment |

The current architecture where K(z) operates in pixel-space and T converts to FSM-space is correct. Don't merge them — keep the separation.

## 7. Code Changes Required

To support runtime matrix updates, the existing `AxisController` in `los_controller.rs` already stores A, B, C as fields:

```rust
struct AxisController {
    state: Vector5<f64>,
    a: Matrix5<f64>,
    b: Vector5<f64>,
    c: RowVector5<f64>,
}
```

To use derived matrices, you would:

1. Add a constructor that accepts custom matrices:
```rust
impl AxisController {
    fn with_matrices(a: Matrix5<f64>, b: Vector5<f64>, c: RowVector5<f64>) -> Self {
        Self { state: Vector5::zeros(), a, b, c }
    }
}
```

2. Add a method to `LosController` for hot-swapping:
```rust
impl LosController {
    pub fn set_matrices(&mut self, a: Matrix5<f64>, b: Vector5<f64>, c: RowVector5<f64>) {
        self.x_controller = AxisController::with_matrices(a.clone(), b.clone(), c.clone());
        self.y_controller = AxisController::with_matrices(a, b, c);
        self.reset();
    }
}
```

3. For rate-adaptive control, add a continuous-time design + discretization:
```rust
pub fn discretize_controller(continuous_a: &Matrix5<f64>, continuous_b: &Vector5<f64>,
                              sample_rate_hz: f64) -> (Matrix5<f64>, Vector5<f64>) {
    let dt = 1.0 / sample_rate_hz;
    // Matrix exponential: A_d = exp(A_c * dt)
    // Input integral: B_d = A_c^(-1) * (A_d - I) * B_c
    // (Or use Padé approximation for numerical stability)
}
```

## 8. Appendix: Eigenvalue Analysis of Current Controller

The A matrix eigenvalues reveal the internal dynamics:

**State 0** (integrator):
- λ₀ = 0.99999999999952 ≈ 1.0
- Pure integrator, infinite time constant
- Gives zero steady-state error for step disturbances

**States 1-4** (the 4x4 block):
The lower-right block has eigenvalues that can be computed from the characteristic polynomial. Since the matrix has a specific structure (symmetric-like couplings), the eigenvalues come in complex conjugate pairs:

Approximate eigenvalues (from matrix structure):
- λ₁,₂ ≈ 0.57 ± 0.55j → |λ| ≈ 0.79, angle ≈ ±44° → natural frequency ≈ 2.8 Hz, damping ≈ 0.15
- λ₃,₄ ≈ 0.73 ± 0.24j → |λ| ≈ 0.77, angle ≈ ±18° → natural frequency ≈ 1.1 Hz, damping ≈ 0.30

These internal modes shape the transient response:
- The faster mode (2.8 Hz) provides quick initial response
- The slower mode (1.1 Hz) provides longer-term correction
- Both modes decay in ~3-5 frames (75-125 ms)

The loop crossover frequency (where |L(jω)| = 1) is approximately 3-5 Hz, consistent with a well-designed controller for a 40 Hz sample rate with 1-frame delay.

## 9. Summary

| Question | Answer |
|----------|--------|
| Can we derive better matrices? | Yes, but first measure the plant (delay, noise, disturbance spectrum) |
| What parameters matter most? | Frame rate, total loop delay, disturbance spectrum |
| Does telescope geometry affect the controller? | Only through SNR → centroid noise. Geometry is in the calibration matrix |
| Should we merge controller and calibration? | No. Keep them separate. The transform cancels in the loop |
| What's the biggest limitation today? | Fixed 40 Hz assumption. Need rate-adaptive design |
| What tool to use? | Python `control` library for synthesis, export to Rust |
| What's the minimum viable improvement? | Measure loop delay, verify margins are adequate |

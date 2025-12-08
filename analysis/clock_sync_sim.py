#!/usr/bin/env python3
"""
Clock synchronization simulation.

Simulates two clocks:
1. Free-running 500 Hz clock (not adjustable) with ppm drift
2. Adjustable clock: 75 MHz master divided down to 500 Hz

Goal: Use phase measurements at each free-running tick to adjust the
divisor once per second, converging to a target phase offset.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Callable


@dataclass
class FreeRunningClock:
    """A simple clock with fixed frequency and ppm error."""
    nominal_freq: float = 500.0  # Hz
    ppm_error: float = 0.0  # parts per million
    phase: float = 0.0  # cycles (accumulated)

    @property
    def true_freq(self) -> float:
        return self.nominal_freq * (1 + self.ppm_error / 1e6)

    def advance(self, true_time_delta: float) -> float:
        """Advance clock by true_time_delta seconds, return new phase."""
        self.phase += self.true_freq * true_time_delta
        return self.phase

    def get_tick_times(self, start_time: float, end_time: float) -> np.ndarray:
        """Get all tick times in the interval [start_time, end_time)."""
        period = 1.0 / self.true_freq
        # Find first tick after start_time
        start_tick = int(np.ceil(start_time * self.true_freq))
        end_tick = int(np.floor(end_time * self.true_freq))
        tick_indices = np.arange(start_tick, end_tick + 1)
        return tick_indices * period


@dataclass
class AdjustableClock:
    """
    Clock based on 75 MHz master, divided down to ~500 Hz.
    Divisor can be adjusted to tune output frequency.
    """
    master_freq_nominal: float = 75e6  # Hz
    master_ppm_error: float = 0.0  # ppm error on master oscillator
    divisor: int = 150_000  # 75e6 / 150000 = 500 Hz nominal
    phase: float = 0.0  # cycles (accumulated)

    # For 2 kHz output, would use divisor of 37,500

    @property
    def true_master_freq(self) -> float:
        return self.master_freq_nominal * (1 + self.master_ppm_error / 1e6)

    @property
    def output_freq(self) -> float:
        return self.true_master_freq / self.divisor

    def advance(self, true_time_delta: float) -> float:
        """Advance clock by true_time_delta seconds, return new phase."""
        self.phase += self.output_freq * true_time_delta
        return self.phase

    def adjust_divisor(self, delta: int):
        """Adjust divisor by delta counts."""
        self.divisor += delta

    def freq_change_per_divisor_step(self) -> float:
        """Approximate frequency change for divisor ±1."""
        # df/dN = -f_master / N^2 = -f_out / N
        return -self.output_freq / self.divisor


@dataclass
class PIDController:
    """PID controller for clock synchronization."""
    kp: float = 0.0  # Proportional gain
    ki: float = 0.0  # Integral gain
    kd: float = 0.0  # Derivative gain

    integral: float = 0.0
    last_error: float = 0.0
    first_update: bool = True

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.first_update = True

    def update(self, error: float, dt: float) -> float:
        """
        Update PID controller with new error measurement.
        Returns control output.
        """
        # Proportional term
        p_term = self.kp * error

        # Integral term (with anti-windup would be nice but keeping simple)
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        if self.first_update:
            d_term = 0.0
            self.first_update = False
        else:
            d_term = self.kd * (error - self.last_error) / dt

        self.last_error = error

        return p_term + i_term + d_term


@dataclass
class SimulationResult:
    """Container for simulation results."""
    times: List[float] = field(default_factory=list)
    phase_errors: List[float] = field(default_factory=list)  # In cycles
    divisor_history: List[int] = field(default_factory=list)
    freq_error_ppm: List[float] = field(default_factory=list)
    adjustment_times: List[float] = field(default_factory=list)
    adjustment_values: List[int] = field(default_factory=list)


def wrap_phase_error(error: float) -> float:
    """Wrap phase error to [-0.5, 0.5) cycles."""
    return ((error + 0.5) % 1.0) - 0.5


def run_simulation(
    free_clock_ppm: float,
    adj_clock_ppm: float,
    rng: np.random.Generator,
    target_phase_offset: float = 0.0,  # cycles
    simulation_duration: float = 60.0,  # seconds
    adjustment_interval: float = 1.0,  # seconds
    pid_kp: float = 1.0,
    pid_ki: float = 0.1,
    pid_kd: float = 0.0,
    use_frequency_feedforward: bool = True,
    random_initial_phase: bool = True,
) -> SimulationResult:
    """
    Run clock synchronization simulation.

    Args:
        free_clock_ppm: PPM error on free-running clock
        adj_clock_ppm: PPM error on adjustable clock's master oscillator
        rng: Random number generator (required, always injected)
        target_phase_offset: Desired phase offset (cycles)
        simulation_duration: How long to simulate (seconds)
        adjustment_interval: How often to adjust divisor (seconds)
        pid_kp: Proportional gain
        pid_ki: Integral gain
        pid_kd: Derivative gain
        use_frequency_feedforward: Estimate frequency offset and feedforward
        random_initial_phase: If True, start both clocks at random phases [0, 1)

    Returns:
        SimulationResult with time series data
    """
    # Initialize clocks
    free_clock = FreeRunningClock(nominal_freq=500.0, ppm_error=free_clock_ppm)
    adj_clock = AdjustableClock(
        master_freq_nominal=75e6,
        master_ppm_error=adj_clock_ppm,
        divisor=150_000
    )

    # Set random initial phases if requested
    if random_initial_phase:
        free_clock.phase = rng.uniform(0.0, 1.0)
        adj_clock.phase = rng.uniform(0.0, 1.0)

    # Initialize PID controller
    pid = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd)

    # Results storage
    result = SimulationResult()

    # Simulation state
    current_time = 0.0
    next_adjustment_time = adjustment_interval

    # Phase measurements within current adjustment window
    window_times: List[float] = []
    window_phase_errors: List[float] = []

    # Main simulation loop - step at free clock tick rate
    free_clock_period = 1.0 / free_clock.true_freq

    while current_time < simulation_duration:
        # Advance both clocks to current time
        free_phase = free_clock.phase
        adj_phase = adj_clock.phase

        # Measure phase error (how far adj clock is from target relative to free)
        # Positive error means adj clock is ahead
        raw_phase_error = adj_phase - free_phase - target_phase_offset
        phase_error = wrap_phase_error(raw_phase_error)

        # Record measurement
        result.times.append(current_time)
        result.phase_errors.append(phase_error)
        result.divisor_history.append(adj_clock.divisor)

        # Calculate instantaneous frequency error in ppm
        freq_error = (adj_clock.output_freq - free_clock.true_freq) / free_clock.true_freq * 1e6
        result.freq_error_ppm.append(freq_error)

        # Store for window analysis
        window_times.append(current_time)
        window_phase_errors.append(phase_error)

        # Check if it's time to adjust
        if current_time >= next_adjustment_time:
            # Estimate frequency offset from phase slope over window
            if len(window_times) >= 10 and use_frequency_feedforward:
                # Linear regression to estimate phase rate (frequency offset)
                t_arr = np.array(window_times) - window_times[0]
                p_arr = np.array(window_phase_errors)

                # Unwrap phase errors for regression
                p_unwrapped = np.unwrap(p_arr * 2 * np.pi) / (2 * np.pi)

                # Linear fit: phase = phase_rate * t + phase_0
                if len(t_arr) > 1:
                    phase_rate, phase_0 = np.polyfit(t_arr, p_unwrapped, 1)
                else:
                    phase_rate = 0.0
                    phase_0 = p_unwrapped[0] if len(p_unwrapped) > 0 else 0.0

                # phase_rate is in cycles/second = Hz frequency offset
                freq_offset_hz = phase_rate
            else:
                freq_offset_hz = 0.0
                phase_0 = phase_error

            # Current phase error (use last measurement or fitted intercept)
            current_phase_error = window_phase_errors[-1] if window_phase_errors else 0.0

            # PID output for phase correction
            pid_output = pid.update(current_phase_error, adjustment_interval)

            # Convert to divisor adjustment
            # We want to change frequency to correct phase over next interval
            # AND compensate for ongoing frequency offset

            # Frequency change needed to correct phase error over one interval:
            # Δf_phase = -phase_error / adjustment_interval (in Hz, cycles/s)

            # Frequency change to compensate frequency offset:
            # Δf_freq = -freq_offset_hz

            # Total desired frequency change
            if use_frequency_feedforward:
                desired_freq_change = -pid_output / adjustment_interval - freq_offset_hz
            else:
                desired_freq_change = -pid_output / adjustment_interval

            # Convert to divisor change
            # Δf = -f_out * ΔN / N, so ΔN = -Δf * N / f_out
            freq_per_step = adj_clock.freq_change_per_divisor_step()
            divisor_change_float = desired_freq_change / freq_per_step
            divisor_change = int(round(divisor_change_float))

            # Apply divisor change (with limits)
            max_change = 100  # Limit step size
            divisor_change = max(-max_change, min(max_change, divisor_change))

            if divisor_change != 0:
                adj_clock.adjust_divisor(divisor_change)
                result.adjustment_times.append(current_time)
                result.adjustment_values.append(divisor_change)

            # Reset window
            window_times = []
            window_phase_errors = []
            next_adjustment_time += adjustment_interval

        # Advance time by one free clock tick
        current_time += free_clock_period
        free_clock.advance(free_clock_period)
        adj_clock.advance(free_clock_period)

    return result


def analyze_convergence(result: SimulationResult, threshold_percent: float = 1.0) -> dict:
    """
    Analyze convergence of phase error.

    Args:
        result: Simulation result
        threshold_percent: Target phase error threshold (% of cycle)

    Returns:
        Dictionary with convergence metrics
    """
    phase_errors = np.array(result.phase_errors)
    times = np.array(result.times)

    threshold_cycles = threshold_percent / 100.0

    # Find first time we get below threshold and stay there
    below_threshold = np.abs(phase_errors) < threshold_cycles

    convergence_time = None
    for i in range(len(below_threshold)):
        if below_threshold[i:].all():
            convergence_time = times[i]
            break

    # If never converged, find when we first got close
    if convergence_time is None:
        first_below = np.where(below_threshold)[0]
        if len(first_below) > 0:
            convergence_time = times[first_below[0]]

    # Final phase error statistics
    final_window = phase_errors[-500:] if len(phase_errors) >= 500 else phase_errors

    return {
        'convergence_time': convergence_time,
        'final_mean_error_percent': np.mean(np.abs(final_window)) * 100,
        'final_std_error_percent': np.std(final_window) * 100,
        'final_max_error_percent': np.max(np.abs(final_window)) * 100,
        'threshold_percent': threshold_percent,
    }


def plot_results(result: SimulationResult, title: str = "Clock Synchronization"):
    """Plot simulation results."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    times = np.array(result.times)

    # Phase error plot
    ax1 = axes[0]
    phase_error_percent = np.array(result.phase_errors) * 100
    ax1.plot(times, phase_error_percent, 'b-', linewidth=0.5, alpha=0.7)
    ax1.axhline(y=1.0, color='r', linestyle='--', label='±1% threshold')
    ax1.axhline(y=-1.0, color='r', linestyle='--')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Phase Error (%)')
    ax1.set_title(f'{title}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-10, 10])

    # Divisor plot
    ax2 = axes[1]
    ax2.plot(times, result.divisor_history, 'g-', linewidth=1)
    ax2.axhline(y=150_000, color='gray', linestyle='--', alpha=0.5, label='Nominal')
    ax2.set_ylabel('Divisor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Frequency error plot
    ax3 = axes[2]
    ax3.plot(times, result.freq_error_ppm, 'm-', linewidth=0.5, alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_ylabel('Freq Error (ppm)')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_parameter_sweep(seed: int = 42):
    """Sweep parameters to understand convergence behavior."""
    rng = np.random.default_rng(seed)

    # Test different PPM combinations
    ppm_combos = [
        (10, -10),   # Small drift, opposite signs
        (50, -30),   # Moderate drift
        (100, 50),   # Large drift, same sign
        (100, -100), # Large drift, opposite signs
    ]

    results = []

    for free_ppm, adj_ppm in ppm_combos:
        print(f"\nTesting: free_clock={free_ppm} ppm, adj_clock={adj_ppm} ppm")

        result = run_simulation(
            free_clock_ppm=free_ppm,
            adj_clock_ppm=adj_ppm,
            rng=rng,
            simulation_duration=30.0,
            pid_kp=0.8,
            pid_ki=0.2,
            pid_kd=0.1,
            use_frequency_feedforward=True,
            random_initial_phase=True,
        )

        analysis = analyze_convergence(result, threshold_percent=1.0)
        analysis['free_ppm'] = free_ppm
        analysis['adj_ppm'] = adj_ppm
        results.append(analysis)

        print(f"  Convergence time: {analysis['convergence_time']:.2f}s"
              if analysis['convergence_time'] else "  Did not converge")
        print(f"  Final mean error: {analysis['final_mean_error_percent']:.3f}%")
        print(f"  Final std error: {analysis['final_std_error_percent']:.3f}%")

    return results


def run_full_ppm_sweep(
    ppm_range: float = 100.0,
    ppm_steps: int = 11,
    simulation_duration: float = 20.0,
    random_initial_phase: bool = True,
    seed: int = 42,
    adjustment_interval: float = 1.0,
) -> Tuple[List[dict], List[SimulationResult]]:
    """
    Sweep both clocks over ±ppm_range and collect all results.

    Args:
        ppm_range: Range of PPM values to sweep (±ppm_range)
        ppm_steps: Number of steps in each dimension
        simulation_duration: How long to simulate each experiment
        random_initial_phase: Start clocks at random phases
        seed: Random seed for reproducibility
        adjustment_interval: How often to adjust divisor (seconds)

    Returns:
        Tuple of (analysis_results, simulation_results)
    """
    rng = np.random.default_rng(seed)
    ppm_values = np.linspace(-ppm_range, ppm_range, ppm_steps)

    all_analyses = []
    all_results = []

    total_runs = len(ppm_values) ** 2
    run_idx = 0

    for free_ppm in ppm_values:
        for adj_ppm in ppm_values:
            run_idx += 1
            print(f"\r  Running {run_idx}/{total_runs}: "
                  f"free={free_ppm:+.0f}ppm, adj={adj_ppm:+.0f}ppm", end="")

            result = run_simulation(
                free_clock_ppm=free_ppm,
                adj_clock_ppm=adj_ppm,
                rng=rng,
                simulation_duration=simulation_duration,
                adjustment_interval=adjustment_interval,
                pid_kp=0.9,
                pid_ki=0.2,
                pid_kd=0.1,
                use_frequency_feedforward=True,
                random_initial_phase=random_initial_phase,
            )

            analysis = analyze_convergence(result, threshold_percent=1.0)
            analysis['free_ppm'] = free_ppm
            analysis['adj_ppm'] = adj_ppm
            analysis['net_ppm'] = adj_ppm - free_ppm  # Effective frequency offset

            all_analyses.append(analysis)
            all_results.append(result)

    print()  # Newline after progress
    return all_analyses, all_results


def plot_convergence_sweep(
    analyses: List[dict],
    results: List[SimulationResult],
    output_path: str = "analysis/clock_sync_sweep.png",
    save: bool = False,
):
    """
    Create comprehensive visualization of the parameter sweep.
    """
    fig = plt.figure(figsize=(16, 12))

    # Extract unique ppm values
    free_ppms = sorted(set(a['free_ppm'] for a in analyses))
    adj_ppms = sorted(set(a['adj_ppm'] for a in analyses))

    # --- Plot 1: Convergence time heatmap ---
    ax1 = fig.add_subplot(2, 2, 1)
    conv_times = np.zeros((len(free_ppms), len(adj_ppms)))
    for a in analyses:
        i = free_ppms.index(a['free_ppm'])
        j = adj_ppms.index(a['adj_ppm'])
        ct = a['convergence_time']
        conv_times[i, j] = ct if ct is not None else np.nan

    im1 = ax1.imshow(conv_times, origin='lower', aspect='auto',
                     extent=[adj_ppms[0], adj_ppms[-1], free_ppms[0], free_ppms[-1]],
                     cmap='viridis_r')
    ax1.set_xlabel('Adjustable Clock PPM')
    ax1.set_ylabel('Free-Running Clock PPM')
    ax1.set_title('Convergence Time to <1% Phase Error (seconds)')
    plt.colorbar(im1, ax=ax1, label='Time (s)')

    # Add contour lines
    X, Y = np.meshgrid(adj_ppms, free_ppms)
    contours = ax1.contour(X, Y, conv_times, levels=[2, 5, 10], colors='white', linewidths=1)
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%.0fs')

    # --- Plot 2: Final error heatmap ---
    ax2 = fig.add_subplot(2, 2, 2)
    final_errors = np.zeros((len(free_ppms), len(adj_ppms)))
    for a in analyses:
        i = free_ppms.index(a['free_ppm'])
        j = adj_ppms.index(a['adj_ppm'])
        final_errors[i, j] = a['final_mean_error_percent']

    im2 = ax2.imshow(final_errors, origin='lower', aspect='auto',
                     extent=[adj_ppms[0], adj_ppms[-1], free_ppms[0], free_ppms[-1]],
                     cmap='plasma', vmin=0, vmax=0.5)
    ax2.set_xlabel('Adjustable Clock PPM')
    ax2.set_ylabel('Free-Running Clock PPM')
    ax2.set_title('Final Mean Phase Error (%)')
    plt.colorbar(im2, ax=ax2, label='Error (%)')

    # --- Plot 3: All convergence curves (signed phase error) ---
    ax3 = fig.add_subplot(2, 2, 3)

    # Color by net ppm offset
    net_ppms = [a['net_ppm'] for a in analyses]
    min_net, max_net = min(net_ppms), max(net_ppms)

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=min_net, vmax=max_net)

    for analysis, result in zip(analyses, results):
        times = np.array(result.times)
        phase_errors = np.array(result.phase_errors) * 100  # Signed, in percent

        # Downsample for plotting (every 50th point)
        step = 50
        color = cmap(norm(analysis['net_ppm']))
        ax3.plot(times[::step], phase_errors[::step], color=color, alpha=0.3, linewidth=0.5)

    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=-1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='±1% threshold')
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Phase Error (%)')
    ax3.set_title('All Convergence Curves (colored by net PPM offset)')
    ax3.set_ylim([-15, 15])
    ax3.set_xlim([0, 20])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add colorbar for net ppm
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label('Net PPM Offset (adj - free)')

    # --- Plot 4: Convergence time vs net PPM ---
    ax4 = fig.add_subplot(2, 2, 4)

    net_ppms_arr = np.array([a['net_ppm'] for a in analyses])
    conv_times_arr = np.array([a['convergence_time'] if a['convergence_time'] else np.nan
                               for a in analyses])

    # Scatter plot
    colors = [cmap(norm(n)) for n in net_ppms_arr]
    ax4.scatter(np.abs(net_ppms_arr), conv_times_arr, c=colors, alpha=0.6, s=50)

    ax4.set_xlabel('|Net PPM Offset| = |adj_ppm - free_ppm|')
    ax4.set_ylabel('Convergence Time (s)')
    ax4.set_title('Convergence Time vs Absolute Frequency Offset')
    ax4.grid(True, alpha=0.3)

    # Fit a trend line
    valid = ~np.isnan(conv_times_arr)
    if np.sum(valid) > 2:
        z = np.polyfit(np.abs(net_ppms_arr[valid]), conv_times_arr[valid], 1)
        p = np.poly1d(z)
        x_fit = np.linspace(0, 200, 100)
        ax4.plot(x_fit, p(x_fit), 'k--', alpha=0.5, label=f'Linear fit: {z[0]:.3f}s/ppm')
        ax4.legend()

    plt.tight_layout()
    if save:
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved: {output_path}")

    return fig


def plot_phase_and_divisor(
    analyses: List[dict],
    results: List[SimulationResult],
    output_path: str = "analysis/clock_sync_phase_divisor.png",
    save: bool = False,
):
    """Plot phase error and divisor for all experiments, colored by net PPM."""
    fig = plt.figure(figsize=(15, 8))

    # Create gridspec with space for colorbar
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], wspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    cax = fig.add_subplot(gs[:, 1])

    # Color by net PPM offset
    net_ppms = [a['net_ppm'] for a in analyses]
    min_net, max_net = min(net_ppms), max(net_ppms)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=min_net, vmax=max_net)

    for analysis, result in zip(analyses, results):
        times = np.array(result.times)
        phase_errors = np.array(result.phase_errors) * 100  # Signed, in percent
        divisors = np.array(result.divisor_history)

        color = cmap(norm(analysis['net_ppm']))
        step = 25  # Downsample for plotting

        ax0.plot(times[::step], phase_errors[::step], color=color, alpha=0.4, linewidth=0.7)
        ax1.plot(times[::step], divisors[::step], color=color, alpha=0.4, linewidth=0.7)

    # Phase error plot
    ax0.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax0.axhline(y=-1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='±1% threshold')
    ax0.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax0.set_ylabel('Phase Error (%)')
    ax0.set_title('Phase Error vs Time (colored by net PPM = adj_ppm - free_ppm)')
    ax0.set_ylim([-15, 15])
    ax0.legend(loc='upper right')
    ax0.grid(True, alpha=0.3)

    # Divisor plot
    ax1.axhline(y=150_000, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Nominal (150000)')
    ax1.set_ylabel('Divisor')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Clock Divisor vs Time (colored by net PPM)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add colorbar in dedicated axis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Net PPM (adj - free)')

    plt.tight_layout()
    if save:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

    return fig


def plot_selected_convergence_curves(
    analyses: List[dict],
    results: List[SimulationResult],
    output_path: str = "analysis/clock_sync_curves.png",
    save: bool = False,
):
    """Plot a selection of convergence curves for clarity (signed phase error)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Select interesting cases
    cases = [
        ("Small offset (±20 ppm)", lambda a: abs(a['net_ppm']) <= 25),
        ("Medium offset (±50-80 ppm)", lambda a: 40 <= abs(a['net_ppm']) <= 90),
        ("Large offset (±100-150 ppm)", lambda a: 90 <= abs(a['net_ppm']) <= 160),
        ("Extreme offset (>150 ppm)", lambda a: abs(a['net_ppm']) > 150),
    ]

    cmap = plt.cm.viridis

    for ax, (title, filter_fn) in zip(axes.flat, cases):
        matching = [(a, r) for a, r in zip(analyses, results) if filter_fn(a)]

        if not matching:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Sort by net ppm for coloring
        matching.sort(key=lambda x: x[0]['net_ppm'])

        for idx, (analysis, result) in enumerate(matching):
            times = np.array(result.times)
            phase_errors = np.array(result.phase_errors) * 100  # Signed

            color = cmap(idx / len(matching))
            step = 25
            label = f"{analysis['net_ppm']:+.0f} ppm" if idx % max(1, len(matching)//5) == 0 else None
            ax.plot(times[::step], phase_errors[::step], color=color, alpha=0.7,
                    linewidth=1, label=label)

        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=-1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase Error (%)')
        ax.set_title(title)
        ax.set_ylim([-15, 15])
        ax.set_xlim([0, 15])
        ax.grid(True, alpha=0.3)
        if matching:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Clock Synchronization Convergence by Frequency Offset Magnitude', fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

    return fig


def tune_pid_gains(
    adjustment_interval: float = 0.5,
    kp_range: tuple = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3),
    ki_range: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    kd_range: tuple = (0.0, 0.05, 0.1, 0.15, 0.2),
    n_trials: int = 25,
    seed: int = 42,
    save: bool = False,
):
    """
    Sweep PID gains and find optimal combination for given update interval.
    """
    from itertools import product

    rng = np.random.default_rng(seed)
    results = []
    total_combos = len(kp_range) * len(ki_range) * len(kd_range)
    combo_idx = 0

    for kp, ki, kd in product(kp_range, ki_range, kd_range):
        combo_idx += 1
        print(f"\rTesting {combo_idx}/{total_combos}: Kp={kp:.1f}, Ki={ki:.1f}, Kd={kd:.2f}    ", end="")

        conv_times = []
        final_errors = []

        for trial in range(n_trials):
            free_ppm = rng.uniform(-100, 100)
            adj_ppm = rng.uniform(-100, 100)

            result = run_simulation(
                free_clock_ppm=free_ppm,
                adj_clock_ppm=adj_ppm,
                rng=rng,
                simulation_duration=20.0,
                adjustment_interval=adjustment_interval,
                pid_kp=kp,
                pid_ki=ki,
                pid_kd=kd,
                use_frequency_feedforward=True,
                random_initial_phase=True,
            )

            analysis = analyze_convergence(result, threshold_percent=1.0)
            conv_times.append(analysis['convergence_time'] if analysis['convergence_time'] else 20.0)
            final_errors.append(analysis['final_mean_error_percent'])

        results.append({
            'kp': kp, 'ki': ki, 'kd': kd,
            'mean_conv_time': np.mean(conv_times),
            'median_conv_time': np.median(conv_times),
            'max_conv_time': np.max(conv_times),
            'mean_final_error': np.mean(final_errors),
            'convergence_rate': sum(1 for t in conv_times if t < 20.0) / n_trials,
        })

    print()

    # Find best combinations
    converged = [r for r in results if r['convergence_rate'] == 1.0] or results
    best_mean = min(converged, key=lambda r: r['mean_conv_time'])
    best_worst = min(converged, key=lambda r: r['max_conv_time'])

    print("\n" + "=" * 60)
    print(f"PID Tuning Results (interval={adjustment_interval}s)")
    print("=" * 60)
    print(f"\nBest by MEAN convergence time:")
    print(f"  Kp={best_mean['kp']:.2f}, Ki={best_mean['ki']:.2f}, Kd={best_mean['kd']:.2f}")
    print(f"  Mean: {best_mean['mean_conv_time']:.2f}s, Max: {best_mean['max_conv_time']:.2f}s")
    print(f"\nBest by WORST-CASE convergence time:")
    print(f"  Kp={best_worst['kp']:.2f}, Ki={best_worst['ki']:.2f}, Kd={best_worst['kd']:.2f}")
    print(f"  Mean: {best_worst['mean_conv_time']:.2f}s, Max: {best_worst['max_conv_time']:.2f}s")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top 15 best combinations
    ax1 = axes[0]
    sorted_results = sorted(results, key=lambda r: r['mean_conv_time'])[:15]
    labels = [f"Kp={r['kp']:.1f}, Ki={r['ki']:.1f}, Kd={r['kd']:.2f}" for r in sorted_results]
    times = [r['mean_conv_time'] for r in sorted_results]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_results)))
    ax1.barh(range(len(sorted_results)), times, color=colors)
    ax1.set_yticks(range(len(sorted_results)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Mean Convergence Time (s)')
    ax1.set_title('Top 15 PID Gain Combinations')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    # Speed vs accuracy tradeoff
    ax2 = axes[1]
    conv_times_plot = [r['mean_conv_time'] for r in results]
    final_errors_plot = [r['mean_final_error'] for r in results]
    conv_rates = [r['convergence_rate'] for r in results]
    scatter = ax2.scatter(conv_times_plot, final_errors_plot, c=conv_rates,
                          cmap='RdYlGn', s=50, alpha=0.7, vmin=0.5, vmax=1.0)
    ax2.scatter([best_mean['mean_conv_time']], [best_mean['mean_final_error']],
                marker='*', s=300, c='red', edgecolors='black', zorder=10, label='Best')
    ax2.set_xlabel('Mean Convergence Time (s)')
    ax2.set_ylabel('Mean Final Error (%)')
    ax2.set_title('Convergence Speed vs Final Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Convergence Rate')

    plt.tight_layout()
    if save:
        plt.savefig("analysis/pid_tuning_results.png", dpi=150)
        print(f"\nSaved: analysis/pid_tuning_results.png")

    return results, best_mean


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clock synchronization simulation with PID control"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save plots to disk (default: off)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (default: show)"
    )
    parser.add_argument(
        "--ppm-range",
        type=float,
        default=100.0,
        help="PPM range to sweep (default: 100)"
    )
    parser.add_argument(
        "--ppm-steps",
        type=int,
        default=11,
        help="Number of PPM steps in sweep (default: 11)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Simulation duration in seconds (default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--adjustment-interval",
        type=float,
        default=1.0,
        help="How often to adjust divisor in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--tune-pid",
        action="store_true",
        help="Run PID gain tuning instead of PPM sweep"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # PID tuning mode
    if args.tune_pid:
        print("=" * 60)
        print("PID Gain Tuning Mode")
        print(f"Adjustment interval: {args.adjustment_interval:.2f}s")
        print("=" * 60)
        tune_pid_gains(
            adjustment_interval=args.adjustment_interval,
            seed=args.seed,
            save=args.save,
        )
        if not args.no_show:
            plt.show()
        return

    print("=" * 60)
    print("Clock Synchronization Simulation")
    print("=" * 60)
    print(f"Full Parameter Sweep: ±{args.ppm_range:.0f} ppm on both clocks")
    print(f"Adjustment interval: {args.adjustment_interval:.2f}s")
    print("=" * 60)

    # Run full sweep
    total_runs = args.ppm_steps ** 2
    print(f"\nRunning full parameter sweep ({args.ppm_steps}x{args.ppm_steps} = {total_runs} experiments)...")
    analyses, results = run_full_ppm_sweep(
        ppm_range=args.ppm_range,
        ppm_steps=args.ppm_steps,
        simulation_duration=args.duration,
        seed=args.seed,
        adjustment_interval=args.adjustment_interval,
    )

    # Summary statistics
    conv_times = [a['convergence_time'] for a in analyses if a['convergence_time'] is not None]
    print(f"\n--- Summary ---")
    print(f"Total experiments: {len(analyses)}")
    print(f"Converged to <1%: {len(conv_times)}/{len(analyses)}")
    if conv_times:
        print(f"Convergence time: min={min(conv_times):.2f}s, "
              f"max={max(conv_times):.2f}s, mean={np.mean(conv_times):.2f}s")

    # Generate plots
    print("\n--- Generating plots ---")
    plot_convergence_sweep(analyses, results, "analysis/clock_sync_sweep.png", save=args.save)
    plot_selected_convergence_curves(analyses, results, "analysis/clock_sync_curves.png", save=args.save)
    plot_phase_and_divisor(analyses, results, "analysis/clock_sync_phase_divisor.png", save=args.save)

    # Find worst case
    worst = max(analyses, key=lambda a: a['convergence_time'] if a['convergence_time'] else 0)
    print(f"\nWorst case: free={worst['free_ppm']:.0f}ppm, adj={worst['adj_ppm']:.0f}ppm")
    print(f"  Net offset: {worst['net_ppm']:.0f} ppm")
    print(f"  Convergence time: {worst['convergence_time']:.2f}s")

    # Find best case (non-zero offset)
    non_trivial = [a for a in analyses if abs(a['net_ppm']) > 10]
    if non_trivial:
        best = min(non_trivial, key=lambda a: a['convergence_time'] if a['convergence_time'] else float('inf'))
        print(f"\nBest non-trivial case: free={best['free_ppm']:.0f}ppm, adj={best['adj_ppm']:.0f}ppm")
        print(f"  Net offset: {best['net_ppm']:.0f} ppm")
        print(f"  Convergence time: {best['convergence_time']:.2f}s")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

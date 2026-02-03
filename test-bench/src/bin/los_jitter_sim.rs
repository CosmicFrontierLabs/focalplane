//! LOS Jitter Simulation
//!
//! Simulates closed-loop FSM control response to Gaussian noise jitter.
//! Uses the LOS feedback controller and FSM calibration transform to model
//! the complete control loop.
//!
//! # Outputs
//! - Time series plots (jitter, centroid error, FSM commands)
//! - Power spectral density (frequency content before/after correction)
//! - RMS statistics (input jitter, residual error, rejection ratio)

use anyhow::{Context, Result};
use clap::Parser;
use monocle::controllers::{LosControlOutput, LosController};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rustfft::{num_complex::Complex, FftPlanner};
use shared::fsm_transform::FsmTransform;
use std::path::{Path, PathBuf};
use test_bench::camera_init::CalibrationArgs;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "LOS jitter closed-loop simulation",
    long_about = "Simulates the FSM closed-loop response to Gaussian noise jitter.\n\n\
        Uses the LOS feedback controller and FSM calibration transform to model\n\
        the complete control loop. Outputs time series, PSD, and RMS statistics."
)]
struct Args {
    /// Duration of simulation in seconds
    #[arg(long, default_value = "10.0")]
    duration: f64,

    /// Frame rate in Hz (controller update rate)
    #[arg(long, default_value = "40.0")]
    framerate: f64,

    /// RMS jitter amplitude in pixels (per axis)
    #[arg(long, default_value = "1.0")]
    jitter_rms: f64,

    #[command(flatten)]
    calibration: CalibrationArgs,

    /// Output directory for plots (default: current directory)
    #[arg(long, default_value = ".")]
    output_dir: PathBuf,

    /// Random seed for reproducibility (optional)
    #[arg(long)]
    seed: Option<u64>,
}

/// Simulation state and results
struct SimulationResults {
    /// Time stamps in seconds
    time: Vec<f64>,
    /// Input jitter X (pixels)
    jitter_x: Vec<f64>,
    /// Input jitter Y (pixels)
    jitter_y: Vec<f64>,
    /// Centroid error X after FSM correction (pixels)
    error_x: Vec<f64>,
    /// Centroid error Y after FSM correction (pixels)
    error_y: Vec<f64>,
    /// FSM command X (µrad)
    fsm_x: Vec<f64>,
    /// FSM command Y (µrad)
    fsm_y: Vec<f64>,
    /// Sample rate
    sample_rate: f64,
}

impl SimulationResults {
    fn new(capacity: usize, sample_rate: f64) -> Self {
        Self {
            time: Vec::with_capacity(capacity),
            jitter_x: Vec::with_capacity(capacity),
            jitter_y: Vec::with_capacity(capacity),
            error_x: Vec::with_capacity(capacity),
            error_y: Vec::with_capacity(capacity),
            fsm_x: Vec::with_capacity(capacity),
            fsm_y: Vec::with_capacity(capacity),
            sample_rate,
        }
    }

    fn rms(data: &[f64]) -> f64 {
        let sum_sq: f64 = data.iter().map(|x| x * x).sum();
        (sum_sq / data.len() as f64).sqrt()
    }

    fn print_statistics(&self) {
        let jitter_rms_x = Self::rms(&self.jitter_x);
        let jitter_rms_y = Self::rms(&self.jitter_y);
        let error_rms_x = Self::rms(&self.error_x);
        let error_rms_y = Self::rms(&self.error_y);
        let fsm_rms_x = Self::rms(&self.fsm_x);
        let fsm_rms_y = Self::rms(&self.fsm_y);

        let rejection_x = if error_rms_x > 0.0 {
            jitter_rms_x / error_rms_x
        } else {
            f64::INFINITY
        };
        let rejection_y = if error_rms_y > 0.0 {
            jitter_rms_y / error_rms_y
        } else {
            f64::INFINITY
        };

        println!("\n=== RMS Statistics ===");
        println!("Input Jitter:");
        println!("  X: {jitter_rms_x:.4} px RMS");
        println!("  Y: {jitter_rms_y:.4} px RMS");
        println!("\nResidual Error (after correction):");
        println!("  X: {error_rms_x:.4} px RMS");
        println!("  Y: {error_rms_y:.4} px RMS");
        println!("\nFSM Commands:");
        println!("  X: {fsm_rms_x:.2} µrad RMS");
        println!("  Y: {fsm_rms_y:.2} µrad RMS");
        println!("\nRejection Ratio (jitter/residual):");
        println!(
            "  X: {:.1}x ({:.1} dB)",
            rejection_x,
            20.0 * rejection_x.log10()
        );
        println!(
            "  Y: {:.1}x ({:.1} dB)",
            rejection_y,
            20.0 * rejection_y.log10()
        );
    }

    fn compute_psd(data: &[f64], sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        let mut buffer: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut buffer);

        // Compute single-sided PSD
        let n_freq = n / 2 + 1;
        let df = sample_rate / n as f64;
        let frequencies: Vec<f64> = (0..n_freq).map(|i| i as f64 * df).collect();

        let psd: Vec<f64> = buffer[..n_freq]
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let power = (c.norm_sqr()) / (n as f64 * sample_rate);
                // Double power for single-sided (except DC and Nyquist)
                if i == 0 || i == n_freq - 1 {
                    power
                } else {
                    2.0 * power
                }
            })
            .collect();

        (frequencies, psd)
    }

    fn write_csv(&self, path: &Path) -> Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record([
            "time", "jitter_x", "jitter_y", "error_x", "error_y", "fsm_x", "fsm_y",
        ])?;

        for i in 0..self.time.len() {
            wtr.write_record(&[
                format!("{:.6}", self.time[i]),
                format!("{:.6}", self.jitter_x[i]),
                format!("{:.6}", self.jitter_y[i]),
                format!("{:.6}", self.error_x[i]),
                format!("{:.6}", self.error_y[i]),
                format!("{:.6}", self.fsm_x[i]),
                format!("{:.6}", self.fsm_y[i]),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }

    fn write_psd_csv(&self, path: &Path) -> Result<()> {
        let (freq_jitter, psd_jitter_x) = Self::compute_psd(&self.jitter_x, self.sample_rate);
        let (_, psd_jitter_y) = Self::compute_psd(&self.jitter_y, self.sample_rate);
        let (_, psd_error_x) = Self::compute_psd(&self.error_x, self.sample_rate);
        let (_, psd_error_y) = Self::compute_psd(&self.error_y, self.sample_rate);

        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record([
            "frequency",
            "psd_jitter_x",
            "psd_jitter_y",
            "psd_error_x",
            "psd_error_y",
        ])?;

        for i in 0..freq_jitter.len() {
            wtr.write_record(&[
                format!("{:.6}", freq_jitter[i]),
                format!("{:.10e}", psd_jitter_x[i]),
                format!("{:.10e}", psd_jitter_y[i]),
                format!("{:.10e}", psd_error_x[i]),
                format!("{:.10e}", psd_error_y[i]),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }
}

fn run_simulation(
    duration: f64,
    framerate: f64,
    jitter_rms: f64,
    transform: &FsmTransform,
    seed: Option<u64>,
) -> SimulationResults {
    let dt = 1.0 / framerate;
    let n_samples = (duration * framerate).ceil() as usize;

    let mut results = SimulationResults::new(n_samples, framerate);

    // Initialize random number generator
    let mut rng: rand::rngs::StdRng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_os_rng(),
    };
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Initialize LOS controller
    let mut controller = LosController::new();
    controller.set_enabled(true);
    controller.set_command(0.0, 0.0); // Track origin

    // FSM state (accumulated position in µrad)
    let mut fsm_x = 0.0;
    let mut fsm_y = 0.0;

    for i in 0..n_samples {
        let t = i as f64 * dt;

        // Generate Gaussian jitter (in pixels)
        let jitter_x: f64 = normal.sample(&mut rng) * jitter_rms;
        let jitter_y: f64 = normal.sample(&mut rng) * jitter_rms;

        // Convert PREVIOUS FSM position to pixel displacement using calibration transform
        // This breaks the algebraic loop - FSM correction from previous sample
        let (fsm_pix_x, fsm_pix_y) = transform.angle_delta_to_pix_delta(fsm_x, fsm_y);

        // Centroid position = jitter + FSM effect (from previous command)
        // Negative FSM command shifts centroid negative (same direction as command)
        // This creates proper negative feedback: positive error → negative command → negative shift
        let centroid_x = jitter_x + fsm_pix_x;
        let centroid_y = jitter_y + fsm_pix_y;

        // Record centroid error BEFORE updating FSM
        results.time.push(t);
        results.jitter_x.push(jitter_x);
        results.jitter_y.push(jitter_y);
        results.error_x.push(centroid_x);
        results.error_y.push(centroid_y);

        // LOS controller sees centroid position and outputs FSM command (absolute, not delta)
        // The controller has internal integrating dynamics
        let LosControlOutput { u_x, u_y } = controller.update(centroid_x, centroid_y);

        // FSM position IS the controller output (applied next sample due to loop delay)
        fsm_x = u_x;
        fsm_y = u_y;

        // Record FSM command
        results.fsm_x.push(fsm_x);
        results.fsm_y.push(fsm_y);
    }

    results
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("LOS Jitter Closed-Loop Simulation");
    println!("==================================");
    println!("Duration: {:.1} s", args.duration);
    println!("Frame rate: {:.1} Hz", args.framerate);
    println!("Jitter RMS: {:.3} px/axis", args.jitter_rms);

    let calibration_path = args
        .calibration
        .calibration
        .as_ref()
        .context("--calibration is required")?;
    println!("Loading calibration from {calibration_path:?}");
    let transform =
        FsmTransform::load(calibration_path).context("Failed to load calibration file")?;

    if let Some(desc) = transform.description() {
        println!("Transform: {desc}");
    }

    // Run simulation
    println!("\nRunning simulation...");
    let results = run_simulation(
        args.duration,
        args.framerate,
        args.jitter_rms,
        &transform,
        args.seed,
    );

    // Print statistics
    results.print_statistics();

    // Write output files
    std::fs::create_dir_all(&args.output_dir)?;

    let timeseries_path = args.output_dir.join("los_sim_timeseries.csv");
    println!("\nWriting time series to {timeseries_path:?}");
    results.write_csv(&timeseries_path)?;

    let psd_path = args.output_dir.join("los_sim_psd.csv");
    println!("Writing PSD data to {psd_path:?}");
    results.write_psd_csv(&psd_path)?;

    // Generate simple ASCII plots
    print_ascii_plot("Centroid Error X (px)", &results.error_x, 60, 15);
    print_ascii_plot("FSM Command X (µrad)", &results.fsm_x, 60, 15);

    println!("\n=== Output Files ===");
    println!("Time series: {timeseries_path:?}");
    println!("PSD data: {psd_path:?}");
    println!("\nTo plot with Python:");
    println!("  python analysis/plot_los_sim.py {:?}", args.output_dir);

    Ok(())
}

fn print_ascii_plot(title: &str, data: &[f64], width: usize, height: usize) {
    if data.is_empty() {
        return;
    }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range < 1e-10 {
        println!("\n{title}: constant at {min:.4}");
        return;
    }

    // Downsample data to fit width
    let step = data.len().max(1) / width.max(1);
    let step = step.max(1);
    let downsampled: Vec<f64> = data.iter().step_by(step).take(width).cloned().collect();

    // Create plot grid
    let mut grid = vec![vec![' '; downsampled.len()]; height];

    for (x, &val) in downsampled.iter().enumerate() {
        let y = ((val - min) / range * (height - 1) as f64).round() as usize;
        let y = y.min(height - 1);
        grid[height - 1 - y][x] = '•';
    }

    println!("\n{title}");
    println!("{:>10.3} ┤{}", max, grid[0].iter().collect::<String>());
    for row in grid.iter().skip(1).take(height - 2) {
        println!("           │{}", row.iter().collect::<String>());
    }
    println!(
        "{:>10.3} ┤{}",
        min,
        grid[height - 1].iter().collect::<String>()
    );
    println!("           └{}", "─".repeat(downsampled.len().min(width)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_simulation_produces_expected_output_count() {
        let transform =
            FsmTransform::new([0.02, 0.0, 0.0, 0.02], [0.0, 0.0]).expect("valid transform");

        let duration = 1.0;
        let framerate = 40.0;
        let jitter_rms = 1.0;
        let seed = Some(42);

        let results = run_simulation(duration, framerate, jitter_rms, &transform, seed);

        let expected_samples = (duration * framerate).ceil() as usize;
        assert_eq!(results.time.len(), expected_samples);
        assert_eq!(results.jitter_x.len(), expected_samples);
        assert_eq!(results.jitter_y.len(), expected_samples);
        assert_eq!(results.error_x.len(), expected_samples);
        assert_eq!(results.error_y.len(), expected_samples);
        assert_eq!(results.fsm_x.len(), expected_samples);
        assert_eq!(results.fsm_y.len(), expected_samples);
    }

    #[test]
    fn test_run_simulation_deterministic_with_seed() {
        let transform =
            FsmTransform::new([0.02, 0.0, 0.0, 0.02], [0.0, 0.0]).expect("valid transform");

        let results1 = run_simulation(0.5, 40.0, 1.0, &transform, Some(123));
        let results2 = run_simulation(0.5, 40.0, 1.0, &transform, Some(123));

        assert_eq!(results1.jitter_x, results2.jitter_x);
        assert_eq!(results1.jitter_y, results2.jitter_y);
        assert_eq!(results1.error_x, results2.error_x);
        assert_eq!(results1.fsm_x, results2.fsm_x);
    }
}

//! LED Rolling Shutter Latency Test
//!
//! # Experimental Setup
//!
//! This binary measures rolling shutter timing characteristics by precisely controlling
//! an LED and observing which row of the sensor first detects the brightness change.
//!
//! ## Hardware Configuration
//!
//! ```text
//! ┌─────────────────┐          ┌──────────────────┐
//! │   Test LED      │          │   GPIO LED       │
//! │ (in front of    │          │ (timing signal)  │
//! │  camera FOV)    │          │                  │
//! └────────┬────────┘          └────────┬─────────┘
//!          │                            │
//!          │  Illuminates               │  Control
//!          │  sensor                    │  signal
//!          ▼                            ▼
//! ┌─────────────────────────────────────────────┐
//! │  Rolling Shutter Camera (POA/NVS455)        │
//! │                                             │
//! │  Row 0    ████████████████████████  t=0    │
//! │  Row 1    ████████████████████████  t=Δ    │
//! │  Row 2    ████████████████████████  t=2Δ   │
//! │  ...                                        │
//! │  Row N    ████████████████████████  t=NΔ   │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! ## Experiment Phases
//!
//! ### Phase 1: Dark Frame Baseline Measurement
//! - LED stays OFF for multiple complete frames
//! - Measure dark frame row mean values
//! - Establish dark baseline for each row
//! - Characterize sensor dark current and read noise
//!
//! ### Phase 2: Light Frame Baseline Measurement
//! - LED stays ON for multiple complete frames
//! - Measure steady-state bright row mean values
//! - Establish brightness baseline for each row
//! - Verify row-to-row consistency
//!
//! ### Phase 3: Latency Characterization
//! - Capture frame N (LED off)
//! - At t = frame_end + delay, blink LED ON
//! - Capture frame N+1, N+2, etc.
//! - Analyze which row first shows brightness increase
//! - Repeat with increasing delay values
//!
//! ## Data Collection
//!
//! For each delay value, record:
//! - Delay after frame completion (microseconds)
//! - First frame showing change
//! - Row number where change first appears
//! - Row mean brightness delta
//! - Timestamp of frame capture
//!
//! ## Expected Results
//!
//! Plot: Row number vs Blink delay
//! - Should show linear relationship
//! - Slope = rolling shutter readout rate
//! - Intercept = frame start timing
//! - Reveals per-row exposure timing
//!
//! ## Output
//!
//! CSV format: `delay_us,frame_num,row_num,brightness_delta,timestamp`

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::{Array1, Array2, Axis};
use shared::camera_interface::{CameraInterface, FrameMetadata};
use shared::frame_writer::{FrameFormat, FrameWriterHandle};
use shared::image_proc::detection::AABB;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;
use test_bench::camera_init;
use test_bench::gpio::{detect_gpio_config, GpioConfig, GpioController};
use tracing::{debug, info};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "LED rolling shutter latency characterization",
    long_about = "Measures rolling shutter timing characteristics by controlling an LED and \
        observing which sensor row first detects the brightness change.\n\n\
        Experimental Setup:\n  \
        - Test LED illuminates camera field of view\n  \
        - GPIO pin controls LED timing with microsecond precision\n  \
        - Camera captures frames during LED state transitions\n\n\
        Phases:\n  \
        1. Dark Frame Baseline: LED OFF, measure dark row means\n  \
        2. Light Frame Baseline: LED ON, measure bright row means\n  \
        3. Latency Characterization: Pulse LED and detect edges\n\n\
        The output CSV contains rising/falling edge row numbers and calculated line rates."
)]
struct Args {
    #[command(flatten)]
    camera: camera_init::CameraArgs,

    #[command(flatten)]
    exposure: camera_init::ExposureArgs,

    #[arg(
        short = 'g',
        long,
        default_value = "100",
        help = "Camera gain",
        long_help = "Camera analog gain setting. Higher gain increases LED signal but also \
            noise. Should provide good separation between dark and light baselines."
    )]
    gain: f64,

    #[arg(
        long,
        default_value = "128",
        help = "ROI width (centered on sensor)",
        long_help = "Width of the region-of-interest in pixels, centered on the sensor. \
            A narrow ROI enables faster readout. Must cover the LED illumination area."
    )]
    roi_width: usize,

    #[arg(
        long,
        default_value = "128",
        help = "ROI height (centered on sensor)",
        long_help = "Height of the region-of-interest in pixels, centered on the sensor. \
            More rows provide better line rate statistics but increase readout time."
    )]
    roi_height: usize,

    #[arg(
        long,
        default_value = "10",
        help = "Number of baseline frames (LED continuously on/off)",
        long_help = "Number of frames to capture for each baseline measurement. \
            More frames improve statistical stability of the baseline."
    )]
    baseline_frames: usize,

    #[arg(
        long,
        default_value = "100",
        help = "Minimum delay between frames (microseconds)",
        long_help = "Minimum delay value for the latency sweep. The test varies delay from \
            --min-delay-us to --max-delay-us in steps of --delay-step-us."
    )]
    min_delay_us: u64,

    #[arg(
        long,
        default_value = "10000",
        help = "Maximum delay between frames (microseconds)",
        long_help = "Maximum delay value for the latency sweep. Should be set based on \
            expected frame exposure time."
    )]
    max_delay_us: u64,

    #[arg(
        long,
        default_value = "100",
        help = "Delay step size (microseconds)",
        long_help = "Step size for delay sweep in microseconds. Smaller steps give finer \
            timing resolution but take longer to complete."
    )]
    delay_step_us: u64,

    #[arg(
        long,
        default_value = "led_latency_results.csv",
        help = "Output CSV file path",
        long_help = "Path for the main results CSV file. Additional files will be created \
            with _dark_baseline.csv, _light_baseline.csv, and _row_means.csv suffixes."
    )]
    output: String,

    #[arg(
        long,
        help = "GPIO pin number (BOARD mode, e.g., 33 for Orin)",
        long_help = "GPIO pin number in BOARD numbering mode (physical pin on connector). \
            Mutually exclusive with --gpio-line. If neither specified, auto-detected.",
        conflicts_with = "gpio_line"
    )]
    gpio_pin: Option<u32>,

    #[arg(
        long,
        help = "GPIO line number (direct gpiochip0 offset)",
        long_help = "GPIO line number as direct offset on gpiochip0 (e.g., 127 for Neutralino). \
            Mutually exclusive with --gpio-pin. If neither specified, auto-detected.",
        conflicts_with = "gpio_pin"
    )]
    gpio_line: Option<u32>,

    #[arg(
        long,
        default_value = "5000",
        help = "LED on duration for blink tests (microseconds)",
        long_help = "Duration to keep LED on during each pulse. Determines how many rows \
            will be illuminated. Longer pulses span more rows but may exceed frame boundaries."
    )]
    led_on_duration_us: u64,

    #[arg(
        long,
        help = "Blink GPIO to identify pin (500ms on/off cycles)",
        long_help = "Enter identification mode where GPIO toggles every 500ms. Useful for \
            verifying correct GPIO pin connection. Press Ctrl+C to exit."
    )]
    identify: bool,

    #[arg(
        long,
        default_value = "100",
        help = "Number of frames to capture during latency test",
        long_help = "Number of frames to capture during latency characterization phase. \
            More frames provide better statistics on line rate variability."
    )]
    num_frames: usize,

    #[arg(
        long,
        default_value = "10000",
        help = "Delay before LED pulse (microseconds)",
        long_help = "Delay between receiving a frame and pulsing the LED. Shifts pulse \
            timing relative to the next frame's exposure window."
    )]
    pulse_delay_us: u64,

    #[arg(
        long,
        default_value = "5",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "Number of initial frames to discard during warm-up",
        long_help = "Number of initial frames to discard before analysis. Allows camera \
            and timing to stabilize before collecting data. Minimum: 1."
    )]
    warmup_drop_count: usize,

    #[arg(
        long,
        help = "Directory to save representative baseline frames",
        long_help = "If specified, saves one dark and one light baseline frame as PNG files \
            in this directory for visual inspection."
    )]
    save_frames_dir: Option<String>,
}

#[inline(always)]
fn spin_wait(duration_us: u64) {
    let start = std::time::Instant::now();
    let target = Duration::from_micros(duration_us);
    while start.elapsed() < target {
        std::hint::spin_loop();
    }
}

struct RowStats {
    row_means: Array1<f64>,
}

fn compute_row_means(frame: &Array2<u16>) -> Array1<f64> {
    frame.map_axis(Axis(1), |row| {
        let sum: u64 = row.iter().map(|&v| v as u64).sum();
        sum as f64 / row.len() as f64
    })
}

#[allow(clippy::too_many_arguments)]
fn run_baseline_measurement(
    camera: &mut dyn CameraInterface,
    led: &mut GpioController,
    baseline_frames: usize,
    roi_height: usize,
    frame_writer: Option<&FrameWriterHandle>,
    output_dir: Option<&Path>,
    led_on: bool,
    csv_path: Option<&Path>,
) -> Result<Array1<f64>> {
    let (phase_num, phase_name, led_state_str, filename) = if led_on {
        (2, "Light Frame", "ON", "light_baseline.png")
    } else {
        (1, "Dark Frame", "OFF", "dark_baseline.png")
    };

    info!(
        "=== Phase {}: {} Baseline Measurement ===",
        phase_num, phase_name
    );
    info!("LED {} for {} frames", led_state_str, baseline_frames);

    if led_on {
        led.on()?;
    } else {
        led.off()?;
    }
    thread::sleep(Duration::from_millis(100));

    let mut baseline_stats: Vec<RowStats> = Vec::new();
    let mut frame_count = 0;
    let mut csv_file = csv_path.and_then(|p| File::create(p).ok());

    if let Some(ref mut f) = csv_file {
        write!(f, "frame_num").ok();
        for row in 0..roi_height {
            write!(f, ",row_{row}").ok();
        }
        writeln!(f).ok();
    }

    camera.stream(&mut |frame_data: &Array2<u16>, _metadata: &FrameMetadata| {
        let row_means = compute_row_means(frame_data);

        if frame_count == 0 {
            if let (Some(writer), Some(dir)) = (frame_writer, output_dir) {
                let filepath = dir.join(filename);
                if let Err(e) = writer.write_frame(frame_data, filepath.clone(), FrameFormat::Png) {
                    tracing::warn!("Failed to queue frame: {}", e);
                } else {
                    info!(
                        "Queued {} baseline frame to {}",
                        phase_name.to_lowercase(),
                        filepath.display()
                    );
                }
            }
        }

        if let Some(ref mut f) = csv_file {
            write!(f, "{frame_count}").ok();
            for &val in row_means.iter() {
                write!(f, ",{val:.2}").ok();
            }
            writeln!(f).ok();
        }

        let (min_val, max_val) = row_means
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
                (min.min(v), max.max(v))
            });
        debug!(
            "{} baseline frame {}: row mean range [{:.1}, {:.1}]",
            phase_name, frame_count, min_val, max_val
        );

        baseline_stats.push(RowStats { row_means });
        frame_count += 1;

        frame_count < baseline_frames
    })?;

    let avg_baseline_per_row: Array1<f64> = baseline_stats
        .iter()
        .map(|s| s.row_means.view())
        .fold(Array1::zeros(roi_height), |acc, row| acc + row)
        / (baseline_frames as f64);

    let (min_val, max_val) = avg_baseline_per_row
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
    info!(
        "{} baseline established: mean brightness per row [{:.1}, {:.1}]",
        phase_name, min_val, max_val
    );

    Ok(avg_baseline_per_row)
}

fn timestamp_diff_s(
    first: &test_bench_shared::Timestamp,
    current: &test_bench_shared::Timestamp,
) -> f64 {
    let first_us = first
        .seconds
        .saturating_mul(1_000_000)
        .saturating_add(first.nanos / 1_000);
    let current_us = current
        .seconds
        .saturating_mul(1_000_000)
        .saturating_add(current.nanos / 1_000);
    (current_us - first_us) as f64 / 1_000_000.0
}

struct CapturedFrame {
    data: Array2<u16>,
    time_s: f64,
    time_s_instant: f64,
}

fn run_latency_characterization(
    camera: &mut dyn CameraInterface,
    led: &mut GpioController,
    dark_baseline: &Array1<f64>,
    light_baseline: &Array1<f64>,
    args: &Args,
) -> Result<()> {
    info!("=== Phase 3: Latency Characterization ===");
    info!(
        "Pulsing LED for {}μs to capture both edges and measure line rate",
        args.led_on_duration_us
    );

    led.off()?;
    thread::sleep(Duration::from_millis(100));

    let num_frames = args.num_frames;
    let warmup_drop = args.warmup_drop_count;
    let total_frames = num_frames + warmup_drop;
    info!(
        "Capturing {} frames with LED pulses ({} warmup + {} analysis)",
        total_frames, warmup_drop, num_frames
    );

    // Capture phase - prioritize timing, minimize processing
    let mut captured_frames: Vec<CapturedFrame> = Vec::with_capacity(num_frames);
    let mut first_timestamp: Option<test_bench_shared::Timestamp> = None;
    let mut first_instant: Option<std::time::Instant> = None;
    let mut frame_count = 0;

    camera.stream(&mut |frame_data: &Array2<u16>, metadata: &FrameMetadata| {
        let now = std::time::Instant::now();

        // Delay before pulse to shift it into middle of next frame's exposure
        if args.pulse_delay_us > 0 {
            spin_wait(args.pulse_delay_us);
        }

        // Pulse LED for next frame
        let _ = led.on();
        spin_wait(args.led_on_duration_us);
        let _ = led.off();

        // Store frame with minimal processing
        if first_timestamp.is_none() {
            first_timestamp = Some(metadata.timestamp);
            first_instant = Some(now);
        }

        // Skip warmup frames
        if frame_count < warmup_drop {
            frame_count += 1;
            return true;
        }

        let time_s = timestamp_diff_s(first_timestamp.as_ref().unwrap(), &metadata.timestamp);
        let time_s_instant = now
            .duration_since(*first_instant.as_ref().unwrap())
            .as_secs_f64();

        captured_frames.push(CapturedFrame {
            data: frame_data.clone(),
            time_s,
            time_s_instant,
        });

        frame_count += 1;
        captured_frames.len() < num_frames
    })?;

    info!(
        "Capture complete. Analyzing {} frames...",
        captured_frames.len()
    );

    // Analysis phase - process all frames after capture
    let mut csv_file = File::create(&args.output)?;
    writeln!(
        csv_file,
        "frame_num,rising_edge_row,falling_edge_row,bright_rows,line_rate_us,time_s,time_s_instant,delta_ms,delta_ms_instant"
    )?;

    let row_means_output = args.output.replace(".csv", "_row_means.csv");
    let mut row_means_file = File::create(&row_means_output)?;

    let num_rows = dark_baseline.len();
    write!(row_means_file, "frame_num,time_s")?;
    for row in 0..num_rows {
        write!(row_means_file, ",row_{row}")?;
    }
    writeln!(row_means_file)?;

    // Calculate thresholds for edge detection
    let rising_threshold: Array1<f64> = dark_baseline
        .iter()
        .zip(light_baseline.iter())
        .map(|(d, l)| d + (l - d) * 0.25)
        .collect();

    let falling_threshold: Array1<f64> = dark_baseline
        .iter()
        .zip(light_baseline.iter())
        .map(|(d, l)| l - (l - d) * 0.25)
        .collect();

    let mut prev_time_s = 0.0;
    let mut prev_time_s_instant = 0.0;
    for (frame_num, frame) in captured_frames.iter().enumerate() {
        let delta_ms = (frame.time_s - prev_time_s) * 1000.0;
        let delta_ms_instant = (frame.time_s_instant - prev_time_s_instant) * 1000.0;
        prev_time_s = frame.time_s;
        prev_time_s_instant = frame.time_s_instant;

        let row_means = compute_row_means(&frame.data);

        // Find rising edge: first row above threshold
        let rising_edge = row_means
            .iter()
            .zip(rising_threshold.iter())
            .enumerate()
            .find(|(_, (val, thresh))| *val > *thresh)
            .map(|(idx, _)| idx as i32)
            .unwrap_or(-1);

        // Find falling edge: first row after bright region that drops below threshold
        let falling_edge = if rising_edge >= 0 {
            let mut last_bright_row = rising_edge;
            for (idx, (val, thresh)) in row_means
                .iter()
                .zip(rising_threshold.iter())
                .enumerate()
                .skip(rising_edge as usize)
            {
                if *val > *thresh {
                    last_bright_row = idx as i32;
                }
            }

            if (last_bright_row as usize) < num_rows - 1 {
                row_means
                    .iter()
                    .zip(falling_threshold.iter())
                    .enumerate()
                    .skip(last_bright_row as usize + 1)
                    .find(|(_, (val, thresh))| *val < *thresh)
                    .map(|(idx, _)| idx as i32)
                    .unwrap_or(-1)
            } else {
                -1
            }
        } else {
            -1
        };

        let num_bright = if rising_edge >= 0 && falling_edge > rising_edge {
            (falling_edge - rising_edge) as usize
        } else if rising_edge >= 0 {
            (num_rows as i32 - rising_edge) as usize
        } else {
            0
        };

        let line_rate_us = if num_bright > 1 {
            args.led_on_duration_us as f64 / (num_bright - 1) as f64
        } else {
            0.0
        };

        let _ = writeln!(
            csv_file,
            "{frame_num},{rising_edge},{falling_edge},{num_bright},{line_rate_us:.6},{:.9},{:.9},{delta_ms:.9},{delta_ms_instant:.9}",
            frame.time_s,
            frame.time_s_instant
        );

        let _ = write!(row_means_file, "{frame_num},{:.6}", frame.time_s);
        for val in row_means.iter() {
            let _ = write!(row_means_file, ",{val:.2}");
        }
        let _ = writeln!(row_means_file);

        if num_bright > 0 {
            info!(
                "  Frame {}: edges at rows {}-{} ({} bright rows, {:.2}μs/line)",
                frame_num, rising_edge, falling_edge, num_bright, line_rate_us
            );
        }
    }

    info!("Results written to: {}", args.output);
    info!("Row means written to: {}", row_means_output);
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let gpio_config = if let Some(line) = args.gpio_line {
        GpioConfig::DirectLine(line)
    } else if let Some(pin) = args.gpio_pin {
        GpioConfig::BoardPin(pin)
    } else {
        detect_gpio_config().context("Failed to auto-detect GPIO configuration")?
    };

    let mut led = match gpio_config {
        GpioConfig::DirectLine(line) => {
            info!("Initializing GPIO LED on line {} (direct mode)", line);
            GpioController::new_from_line(line).context("Failed to initialize GPIO controller")?
        }
        GpioConfig::BoardPin(pin) => {
            info!("Initializing GPIO LED on pin {} (BOARD mode)", pin);
            GpioController::new(pin).context("Failed to initialize GPIO controller")?
        }
    };

    led.request_output("led_latency_test", 0)
        .context("Failed to configure GPIO as output")?;

    if args.identify {
        info!("\nGPIO Identification Mode");
        info!("========================");
        match gpio_config {
            GpioConfig::DirectLine(line) => info!("Blinking line {} every 500ms", line),
            GpioConfig::BoardPin(pin) => info!("Blinking pin {} every 500ms", pin),
        }
        info!("Press Ctrl+C to exit\n");

        loop {
            led.on()?;
            thread::sleep(Duration::from_millis(500));
            led.off()?;
            thread::sleep(Duration::from_millis(500));
        }
    }

    info!("Initializing camera...");
    let mut camera = camera_init::initialize_camera(&args.camera)?;

    let exposure = args.exposure.as_duration();
    info!("Setting camera exposure to {}ms", args.exposure.exposure_ms);
    camera.set_exposure(exposure)?;

    info!("Setting camera gain to {}", args.gain);
    camera.set_gain(args.gain)?;

    let full_size = camera.geometry().size;
    info!(
        "Camera full frame size: {}x{}",
        full_size.width, full_size.height
    );

    let roi_x = (full_size.width - args.roi_width) / 2;
    let roi_y = (full_size.height - args.roi_height) / 2;

    info!(
        "Setting ROI to {}x{} at ({}, {})",
        args.roi_width, args.roi_height, roi_x, roi_y
    );

    let roi = AABB::from_coords(
        roi_y,
        roi_x,
        roi_y + args.roi_height - 1,
        roi_x + args.roi_width - 1,
    );
    camera.set_roi(roi)?;

    thread::sleep(Duration::from_millis(500));

    let output_dir = args.save_frames_dir.as_ref().map(PathBuf::from);
    let frame_writer = if output_dir.is_some() {
        Some(FrameWriterHandle::new(4, 1000).context("Failed to create frame writer")?)
    } else {
        None
    };

    let dark_csv = args.output.replace(".csv", "_dark_baseline.csv");
    let light_csv = args.output.replace(".csv", "_light_baseline.csv");

    let dark_baseline = run_baseline_measurement(
        camera.as_mut(),
        &mut led,
        args.baseline_frames,
        args.roi_height,
        frame_writer.as_ref(),
        output_dir.as_deref(),
        false,
        Some(Path::new(&dark_csv)),
    )?;

    let light_baseline = run_baseline_measurement(
        camera.as_mut(),
        &mut led,
        args.baseline_frames,
        args.roi_height,
        frame_writer.as_ref(),
        output_dir.as_deref(),
        true,
        Some(Path::new(&light_csv)),
    )?;

    info!("Dark baseline CSV: {}", dark_csv);
    info!("Light baseline CSV: {}", light_csv);

    info!("Validating baseline separation...");
    let min_required_separation = 20.0;
    let delta: Array1<f64> = &light_baseline - &dark_baseline;
    let min_delta = delta.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_delta = delta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_delta = delta.iter().sum::<f64>() / delta.len() as f64;

    info!(
        "Baseline separation: min={:.1} DN, max={:.1} DN, mean={:.1} DN",
        min_delta, max_delta, mean_delta
    );

    if min_delta < min_required_separation {
        anyhow::bail!(
            "Baseline separation too small! Minimum delta is {min_delta:.1} DN, but need at least {min_required_separation:.1} DN. \
             Check LED brightness and camera settings."
        );
    }

    info!(
        "Baseline separation OK (min {:.1} DN >= {:.1} DN required)",
        min_delta, min_required_separation
    );

    run_latency_characterization(
        camera.as_mut(),
        &mut led,
        &dark_baseline,
        &light_baseline,
        &args,
    )?;

    info!("LED latency test complete");

    if let Some(writer) = frame_writer {
        info!("Waiting for frame writer to complete...");
        writer.wait_for_completion();
    }

    Ok(())
}

//! Unified CLI tool for PI E-727 fast steering mirror control.
//!
//! Combines functionality from multiple FSM binaries into a single tool with subcommands:
//! - `circle`: Trace circular patterns
//! - `steer`: Interactive steering with arrow keys
//! - `move`: Move to absolute/relative positions
//! - `resonance`: Excite and record resonance response
//! - `query`: Query axis positions and status
//! - `info`: Query device info and servo tuning parameters
//! - `off`: Disable servos

use std::f64::consts::PI;
use std::io::{self, Read, Write};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use hardware::pi::{Axis, GcsDevice, PiErrorCode, SpaParam, E727, RECORDER_SAMPLE_RATE_HZ, S330};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use strum::IntoEnumIterator;
use tracing::info;

/// Default E-727 IP address
const DEFAULT_IP: &str = "192.168.15.210";

/// Parse axis string (e.g. "1", "2") into Axis enum.
fn parse_axis(s: &str) -> Result<Axis> {
    s.parse().map_err(|e: String| anyhow::anyhow!(e))
}

/// PI E-727 Fast Steering Mirror Control Tool
#[derive(Parser, Debug)]
#[command(name = "fsm_tool")]
#[command(about = "Unified control tool for PI E-727 fast steering mirror")]
#[command(version)]
struct Args {
    /// E-727 IP address
    #[arg(long, global = true, default_value = DEFAULT_IP)]
    ip: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Trace a circular pattern with the FSM
    Circle {
        /// Radius as percentage of working FOV (0-100)
        #[arg(short, long, default_value = "95")]
        radius_percent: f64,

        /// Period of one revolution in seconds
        #[arg(short, long, default_value = "1.0")]
        period: f64,

        /// Number of revolutions (0 = infinite)
        #[arg(short, long, default_value = "0")]
        count: u32,

        /// Delay between commands in milliseconds
        #[arg(long, default_value = "1")]
        delay_ms: u64,
    },

    /// Interactive FSM steering with arrow keys
    Steer,

    /// Move to a specified position
    Move {
        /// Axis to move (1, 2, 3, or 4)
        #[arg(short, long)]
        axis: Option<String>,

        /// Absolute position to move to (in physical units)
        #[arg(short, long)]
        position: Option<f64>,

        /// Relative distance to move
        #[arg(short, long)]
        relative: Option<f64>,

        /// Move to center of axis range
        #[arg(short, long)]
        center: bool,

        /// Maximum step size per command (safety limit)
        #[arg(long, default_value = "10")]
        max_step: f64,

        /// Timeout in seconds for motion to complete
        #[arg(short, long, default_value = "5")]
        timeout: u64,

        /// Don't wait for motion to complete
        #[arg(long)]
        no_wait: bool,
    },

    /// Excite and record FSM resonance response
    Resonance {
        /// Axis to test (1 or 2)
        #[arg(short, long, default_value = "1")]
        axis: String,

        /// Step size in µrad for excitation
        #[arg(short, long, default_value = "100")]
        step: f64,

        /// Sample rate divider (1 = fastest)
        #[arg(long, default_value = "1")]
        rate: u32,

        /// Output CSV file
        #[arg(short, long, default_value = "resonance.csv")]
        output: String,
    },

    /// Query current positions and status
    Query {
        /// Specific axis to query (queries all if not specified)
        #[arg(short, long)]
        axis: Option<String>,
    },

    /// Disable servos
    Off {
        /// Specific axis to disable (disables all if not specified)
        #[arg(short, long)]
        axis: Option<String>,
    },

    /// Query device info and SPA parameters
    Info {
        /// Dump all SPA (Set Parameter Access) values from the controller
        #[arg(long)]
        dump_params: bool,

        /// Show data recorder configuration
        #[arg(long)]
        recorder: bool,

        /// List all available parameters (HPA? command)
        #[arg(long)]
        hpa: bool,

        /// Show all info (equivalent to --dump-params --recorder)
        #[arg(long)]
        all: bool,
    },

    /// Interactive GCS command REPL
    Repl,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Commands that use S330 interface (handles init, autozero, servo enable)
    match args.command {
        Command::Circle {
            radius_percent,
            period,
            count,
            delay_ms,
        } => {
            info!("Connecting to S-330 at {}...", args.ip);
            let mut fsm = S330::connect_ip(&args.ip)?;
            cmd_circle(&mut fsm, radius_percent, period, count, delay_ms)
        }
        Command::Steer => {
            info!("Connecting to S-330 at {}...", args.ip);
            let mut fsm = S330::connect_ip(&args.ip)?;
            cmd_steer(&mut fsm)
        }
        Command::Move {
            axis,
            position,
            relative,
            center,
            max_step,
            timeout,
            no_wait,
        } => cmd_move(
            &args.ip, axis, position, relative, center, max_step, timeout, no_wait,
        ),
        Command::Resonance {
            axis,
            step,
            rate,
            output,
        } => cmd_resonance(&args.ip, &axis, step, rate, &output),
        Command::Query { axis } => cmd_query(&args.ip, axis),
        Command::Off { axis } => cmd_off(&args.ip, axis),
        Command::Info {
            dump_params,
            recorder,
            hpa,
            all,
        } => cmd_info(&args.ip, dump_params || all, recorder || all, hpa),
        Command::Repl => cmd_repl(&args.ip),
    }
}

// ==================== Circle Command ====================

/// Linearly interpolate from one position to another over the given duration.
fn ramp_to(
    fsm: &mut S330,
    from_x: f64,
    from_y: f64,
    to_x: f64,
    to_y: f64,
    duration: Duration,
    delay_ms: u64,
) -> Result<()> {
    let start = Instant::now();
    while start.elapsed() < duration {
        let t = start.elapsed().as_secs_f64() / duration.as_secs_f64();
        let x = from_x + (to_x - from_x) * t;
        let y = from_y + (to_y - from_y) * t;
        fsm.move_to(x, y)?;
        std::thread::sleep(Duration::from_millis(delay_ms));
    }
    fsm.move_to(to_x, to_y)?;
    Ok(())
}

fn cmd_circle(
    fsm: &mut S330,
    radius_percent: f64,
    period: f64,
    count: u32,
    delay_ms: u64,
) -> Result<()> {
    let ((min_x, max_x), (min_y, max_y)) = fsm.get_travel_ranges()?;
    let unit = fsm.get_unit()?;
    let (center_x, center_y) = fsm.get_centers()?;

    info!(
        "Axis 1: center={:.1} {}, range=[{:.1}, {:.1}]",
        center_x, unit, min_x, max_x
    );
    info!(
        "Axis 2: center={:.1} {}, range=[{:.1}, {:.1}]",
        center_y, unit, min_y, max_y
    );

    let max_radius_x = (max_x - center_x).min(center_x - min_x);
    let max_radius_y = (max_y - center_y).min(center_y - min_y);
    let max_radius = max_radius_x.min(max_radius_y);

    let radius_pct = radius_percent.clamp(0.0, 100.0);
    let radius = max_radius * (radius_pct / 100.0);

    info!(
        "Circle: radius={:.1} {} ({:.0}% of {:.1} max), period={:.2}s",
        radius, unit, radius_pct, max_radius, period
    );

    let (cur_x, cur_y) = fsm.get_position()?;
    info!("Starting from: ({:.1}, {:.1})", cur_x, cur_y);

    // Move to starting position (right side of circle) over 1 second
    let start_x = center_x + radius;
    let start_y = center_y;
    info!("Moving to circle start ({:.1}, {:.1})...", start_x, start_y);
    ramp_to(
        fsm,
        cur_x,
        cur_y,
        start_x,
        start_y,
        Duration::from_secs(1),
        delay_ms,
    )?;

    let angular_velocity = 2.0 * PI / period;
    info!("Starting circular motion (Ctrl+C to stop)...");

    let start_time = Instant::now();
    let mut last_report_time = Instant::now();
    let mut pointings_since_report = 0u64;

    loop {
        let elapsed = start_time.elapsed().as_secs_f64();
        let angle = elapsed * angular_velocity;
        let revolutions = angle / (2.0 * PI);

        let target_x = center_x + radius * angle.cos();
        let target_y = center_y + radius * angle.sin();

        fsm.move_to(target_x, target_y)?;
        pointings_since_report += 1;

        if last_report_time.elapsed() >= Duration::from_millis(200) {
            info!(
                "{:.2} revs, {} pointings",
                revolutions, pointings_since_report
            );
            pointings_since_report = 0;
            last_report_time = Instant::now();
        }

        if count > 0 && revolutions >= count as f64 {
            break;
        }

        std::thread::sleep(Duration::from_millis(delay_ms));
    }

    // Return to center over 1 second
    let (cur_x, cur_y) = fsm.get_position()?;
    info!("Returning to center...");
    ramp_to(
        fsm,
        cur_x,
        cur_y,
        center_x,
        center_y,
        Duration::from_secs(1),
        delay_ms,
    )?;

    // S330 Drop will handle servo disable and shutdown
    info!("Done!");
    Ok(())
}

// ==================== Steer Command ====================

fn cmd_steer(fsm: &mut S330) -> Result<()> {
    let ((min_x, max_x), (min_y, max_y)) = fsm.get_travel_ranges()?;
    let (center_x, center_y) = fsm.get_centers()?;

    let (mut pos_x, mut pos_y) = fsm.get_position()?;
    println!("Position: ({pos_x:.1}, {pos_y:.1})");

    let step_sizes = [0.1, 1.0, 10.0, 100.0];
    let mut step_idx = 1;

    println!("\n=== FSM Steering ===");
    println!("Arrow keys: Move | S: Step size | C: Center | R: Record | Q: Quit\n");

    set_raw_mode()?;

    let result = (|| -> Result<()> {
        let mut stdin = io::stdin();
        let mut buf = [0u8; 3];

        loop {
            print!(
                "\rPos: ({:8.3}, {:8.3}) µrad | Step: {:5.1} µrad | ",
                pos_x, pos_y, step_sizes[step_idx]
            );
            io::stdout().flush()?;

            let n = stdin.read(&mut buf)?;
            if n == 0 {
                continue;
            }

            let mut dx = 0.0;
            let mut dy = 0.0;
            let step = step_sizes[step_idx];

            match &buf[..n] {
                [27, 91, 65] => dy = step,  // Up
                [27, 91, 66] => dy = -step, // Down
                [27, 91, 67] => dx = step,  // Right
                [27, 91, 68] => dx = -step, // Left
                [b'q'] | [b'Q'] | [27] => break,
                [b's'] | [b'S'] => {
                    step_idx = (step_idx + 1) % step_sizes.len();
                    continue;
                }
                [b'c'] | [b'C'] => {
                    pos_x = center_x;
                    pos_y = center_y;
                    fsm.move_to(pos_x, pos_y)?;
                    continue;
                }
                [b'r'] | [b'R'] => {
                    print!("\r Recording...                                        ");
                    io::stdout().flush()?;

                    let (errors, positions) = fsm
                        .e727_mut()
                        .record_position(Axis::Axis1, Duration::from_millis(200))?;

                    if let Some(filename) = write_capture_csv(&errors, &positions)? {
                        let n = errors.len().min(positions.len());
                        print!("\r Saved {filename} ({n} samples)                    ");
                        io::stdout().flush()?;
                    }
                    continue;
                }
                _ => continue,
            }

            if dx != 0.0 || dy != 0.0 {
                pos_x = (pos_x + dx).clamp(min_x, max_x);
                pos_y = (pos_y + dy).clamp(min_y, max_y);
                fsm.move_to(pos_x, pos_y)?;
            }
        }
        Ok(())
    })();

    restore_terminal()?;
    println!("\n\nDone! (S330 will shutdown on drop)");

    result
}

fn set_raw_mode() -> Result<()> {
    std::process::Command::new("stty")
        .arg("-echo")
        .arg("raw")
        .arg("-icanon")
        .status()?;
    Ok(())
}

fn restore_terminal() -> Result<()> {
    std::process::Command::new("stty")
        .arg("echo")
        .arg("cooked")
        .arg("icanon")
        .status()?;
    Ok(())
}

fn write_capture_csv(errors: &[f64], positions: &[f64]) -> Result<Option<String>> {
    if errors.is_empty() {
        return Ok(None);
    }

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let filename = format!("capture_{ts}.csv");
    let mut file = std::fs::File::create(&filename)?;
    writeln!(file, "time_us,error_x,pos_x")?;

    let n = errors.len().min(positions.len());
    for i in 0..n {
        let time_us = i as f64 * 20.0; // 50kHz = 20µs
        writeln!(file, "{:.1},{:.6},{:.6}", time_us, errors[i], positions[i])?;
    }

    Ok(Some(filename))
}

// ==================== Move Command ====================

#[allow(clippy::too_many_arguments)]
fn cmd_move(
    ip: &str,
    axis: Option<String>,
    position: Option<f64>,
    relative: Option<f64>,
    center: bool,
    max_step: f64,
    timeout: u64,
    no_wait: bool,
) -> Result<()> {
    if position.is_none() && relative.is_none() && !center {
        bail!("Must specify --position, --relative, or --center");
    }

    info!("Connecting to E-727 at {}...", ip);
    let mut fsm = E727::connect_ip(ip)?;

    let axis_str = axis.ok_or_else(|| anyhow::anyhow!("--axis is required for move operations"))?;
    let axis = parse_axis(&axis_str)?;

    let (min, max) = fsm.get_travel_range(axis)?;
    let unit = fsm.get_unit(axis)?;
    let current_pos = fsm.get_position(axis)?;
    let servo_on = fsm.get_servo(axis)?;

    info!(
        "Axis {axis}: current={current_pos:.3} {unit}, range=[{min:.3}, {max:.3}], servo={servo_on}"
    );

    let mut target = if let Some(pos) = position {
        pos
    } else if let Some(rel) = relative {
        current_pos + rel
    } else {
        fsm.get_center(axis)?
    };

    if target < min || target > max {
        bail!("Target position {target:.3} is outside range [{min:.3}, {max:.3}]");
    }

    let step = target - current_pos;
    if step.abs() > max_step {
        let clamped_step = step.signum() * max_step;
        let original_target = target;
        target = current_pos + clamped_step;
        info!(
            "Step size {:.3} exceeds max {max_step:.1}, clamping to {target:.3} (requested {original_target:.3})",
            step.abs()
        );
    }

    info!("Target position: {target:.3} {unit}");

    if !servo_on {
        info!("Enabling servo on axis {axis}...");
        fsm.set_servo(axis, true)?;
    }

    info!("Moving axis {axis} to {target:.3} {unit}...");
    // Build move command with target in correct axis slot
    let (a1, a2, a3, a4) = match axis {
        Axis::Axis1 => (Some(target), None, None, None),
        Axis::Axis2 => (None, Some(target), None, None),
        Axis::Axis3 => (None, None, Some(target), None),
        Axis::Axis4 => (None, None, None, Some(target)),
    };
    fsm.move_to(a1, a2, a3, a4)?;

    if !no_wait {
        let timeout_dur = Duration::from_secs(timeout);
        info!("Waiting for motion to complete (timeout: {timeout}s)...");

        let start = Instant::now();
        loop {
            if start.elapsed() > timeout_dur {
                bail!("Timeout waiting for motion to complete");
            }

            if fsm.is_on_target(axis)? {
                break;
            }

            std::thread::sleep(Duration::from_millis(10));
        }

        let final_pos = fsm.get_position(axis)?;
        let error = final_pos - target;
        info!("Motion complete! Final position: {final_pos:.3} {unit} (error: {error:.3} {unit})");
    } else {
        info!("Motion command sent (not waiting for completion)");
    }

    Ok(())
}

// ==================== Resonance Command ====================

fn cmd_resonance(ip: &str, axis: &str, step: f64, rate: u32, output: &str) -> Result<()> {
    const RECORD_POSITION_ERROR: u8 = 3;
    const RECORD_CURRENT_POSITION: u8 = 2;
    const TRIGGER_IMMEDIATE: u8 = 4;

    info!("Connecting to E-727 at {}...", ip);
    let mut gcs = GcsDevice::connect_default_port(ip)?;

    // Query axis range
    let response = gcs.query(&format!("TMN? {axis}"))?;
    let min: f64 = response.split('=').nth(1).unwrap().trim().parse()?;
    let response = gcs.query(&format!("TMX? {axis}"))?;
    let max: f64 = response.split('=').nth(1).unwrap().trim().parse()?;
    let center = (min + max) / 2.0;
    info!(
        "Axis {} range: [{:.1}, {:.1}], center: {:.1}",
        axis, min, max, center
    );

    let response = gcs.query(&format!("POS? {axis}"))?;
    let current: f64 = response.split('=').nth(1).unwrap().trim().parse()?;
    info!("Current position: {:.3}", current);

    info!("Configuring data recorder...");
    gcs.send(&format!("RTR {rate}"))?;
    info!("Sample rate divider: {}", rate);

    gcs.send(&format!("DRC 1 {axis} {RECORD_POSITION_ERROR}"))?;
    gcs.send(&format!("DRC 2 {axis} {RECORD_CURRENT_POSITION}"))?;

    gcs.send(&format!("DRT 1 {TRIGGER_IMMEDIATE} 0"))?;
    gcs.send(&format!("DRT 2 {TRIGGER_IMMEDIATE} 0"))?;

    let response = gcs.query("DRC? 1")?;
    info!("Table 1 config: {}", response.trim());
    let response = gcs.query("DRC? 2")?;
    info!("Table 2 config: {}", response.trim());

    info!("Enabling servo on axis {}...", axis);
    gcs.send(&format!("SVO {axis} 1"))?;
    std::thread::sleep(Duration::from_millis(100));

    let start_pos = center - step / 2.0;
    let end_pos = center + step / 2.0;

    info!("Moving to start position {:.1}...", start_pos);
    gcs.send(&format!("MOV {axis} {start_pos}"))?;
    std::thread::sleep(Duration::from_millis(500));

    info!("Starting data recording...");
    info!(
        "Applying step: {:.1} -> {:.1} µrad ({:.1} µrad step)",
        start_pos, end_pos, step
    );
    gcs.send(&format!("MOV {axis} {end_pos}"))?;

    info!("Waiting for transient response...");
    std::thread::sleep(Duration::from_millis(200));

    info!("Stopping recording...");
    gcs.send("DRT 1 0 0")?;
    gcs.send("DRT 2 0 0")?;

    let response = gcs.query("DRL? 1")?;
    let num_points: usize = response
        .split('=')
        .nth(1)
        .unwrap_or("0")
        .trim()
        .parse()
        .unwrap_or(0);
    info!("Recorded {} points in table 1", num_points);

    if num_points == 0 {
        info!("No data recorded!");
        return Ok(());
    }

    info!("Reading recorded data...");
    let response = gcs.query(&format!("DRR? 1 {num_points} 1"))?;
    let error_data: Vec<f64> = response
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                None
            } else {
                trimmed.parse().ok()
            }
        })
        .collect();

    let response = gcs.query(&format!("DRR? 1 {num_points} 2"))?;
    let position_data: Vec<f64> = response
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                None
            } else {
                trimmed.parse().ok()
            }
        })
        .collect();

    info!(
        "Got {} error samples and {} position samples",
        error_data.len(),
        position_data.len()
    );

    let sample_period_us = (rate as f64) / RECORDER_SAMPLE_RATE_HZ * 1e6;

    info!("Writing to {}...", output);
    let mut file = std::fs::File::create(output)?;
    writeln!(file, "time_us,position_error,position")?;

    let n_samples = error_data.len().min(position_data.len());
    for i in 0..n_samples {
        let time_us = i as f64 * sample_period_us;
        writeln!(
            file,
            "{:.3},{:.6},{:.6}",
            time_us, error_data[i], position_data[i]
        )?;
    }

    info!("Done! {} samples written", n_samples);

    if !error_data.is_empty() {
        let max_error = error_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_error = error_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let peak_to_peak = max_error - min_error;
        info!(
            "Position error: min={:.3}, max={:.3}, pk-pk={:.3}",
            min_error, max_error, peak_to_peak
        );
    }

    info!("Disabling servo...");
    gcs.send(&format!("SVO {axis} 0"))?;

    info!("Done!");
    Ok(())
}

// ==================== Query Command ====================

fn cmd_query(ip: &str, axis: Option<String>) -> Result<()> {
    info!("Connecting to E-727 at {}...", ip);
    let mut fsm = E727::connect_ip(ip)?;

    let axes: Vec<Axis> = if let Some(ax) = axis {
        vec![parse_axis(&ax)?]
    } else {
        fsm.connected_axes()?
    };

    for axis in axes {
        let (min, max) = fsm.get_travel_range(axis)?;
        let unit = fsm.get_unit(axis)?;
        let current_pos = fsm.get_position(axis)?;
        let servo_on = fsm.get_servo(axis)?;
        let on_target = fsm.is_on_target(axis)?;
        let autozeroed = fsm.is_autozeroed(axis)?;

        info!(
            "Axis {axis}: position={current_pos:.3} {unit}, range=[{min:.3}, {max:.3}], servo={servo_on}, on_target={on_target}, atz={autozeroed}"
        );
    }

    Ok(())
}

// ==================== Off Command ====================

fn cmd_off(ip: &str, axis: Option<String>) -> Result<()> {
    info!("Connecting to E-727 at {}...", ip);
    let mut fsm = E727::connect_ip(ip)?;

    let axes: Vec<Axis> = if let Some(ax) = axis {
        vec![parse_axis(&ax)?]
    } else {
        fsm.connected_axes()?
    };

    for axis in axes {
        info!("Disabling servo on axis {axis}...");
        fsm.set_servo(axis, false)?;
    }

    info!("Servos disabled");
    Ok(())
}

// ==================== Info Command ====================

fn cmd_info(ip: &str, dump_params: bool, show_recorder: bool, show_hpa: bool) -> Result<()> {
    info!("Connecting to E-727 at {}...", ip);
    let mut gcs = GcsDevice::connect_default_port(ip)?;

    // Basic device info
    info!("=== Device Info ===");
    let response = gcs.query("*IDN?")?;
    info!("IDN: {}", response.trim());

    let response = gcs.query("SAI?")?;
    let axes: Vec<&str> = response
        .lines()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    info!("Axes: {:?}", axes);

    let response = gcs.query("POS?")?;
    info!("Positions: {}", response.trim());

    let response = gcs.query("SVO?")?;
    info!("Servo states: {}", response.trim());

    // List all available parameters
    if show_hpa {
        info!("=== Available Parameters (HPA?) ===");
        let response = gcs.query("HPA?")?;
        // Print each parameter on its own line
        for line in response.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                info!("{}", trimmed);
            }
        }
    }

    // Data recorder info
    if show_recorder {
        info!("=== Data Recorder ===");
        let response = gcs.query("HDR?")?;
        info!("HDR? (available options):\n{}", response);

        let response = gcs.query("DRC?")?;
        info!("DRC? (current config): {}", response.trim());

        let response = gcs.query("DRT?")?;
        info!("DRT? (trigger): {}", response.trim());

        let response = gcs.query("RTR?")?;
        info!("RTR? (sample rate): {}", response.trim());

        let response = gcs.query("TNR?")?;
        info!("TNR? (num tables): {}", response.trim());
    }

    // All SPA parameters
    if dump_params {
        info!("=== SPA Parameters ===");

        // Calculate max name width for alignment
        let max_name_len = SpaParam::iter()
            .map(|p| p.to_string().len())
            .max()
            .unwrap_or(0);

        // First, query system-wide parameters (MaxItem=1)
        // Note: GCS requires axis identifier even for system-wide params, use axis 1
        info!("  System-wide:");
        for param in SpaParam::iter().filter(|p| p.is_system_wide()) {
            let addr = param.address();
            let response = gcs.query(&format!("SPA? 1 0x{addr:08X}"))?;
            // Parse value from response (format: "1=value")
            let value = response
                .split('=')
                .nth(1)
                .map(|s| s.trim())
                .unwrap_or(response.trim());
            info!(
                "    {:width$} (0x{:08X}) : {}",
                param,
                addr,
                value,
                width = max_name_len
            );
        }

        // Then query per-axis parameters (MaxItem=4) for all 4 axes
        for axis in ["1", "2", "3", "4"] {
            info!("  Axis {}:", axis);
            for param in SpaParam::iter().filter(|p| p.max_items() == 4) {
                let addr = param.address();
                match gcs.query(&format!("SPA? {axis} 0x{addr:08X}")) {
                    Ok(response) => {
                        let value = response
                            .split('=')
                            .nth(1)
                            .map(|s| s.trim())
                            .unwrap_or(response.trim());
                        info!(
                            "    {:width$} (0x{:08X}) : {}",
                            param,
                            addr,
                            value,
                            width = max_name_len
                        );
                    }
                    Err(e) => {
                        info!(
                            "    {:width$} (0x{:08X}) : <error: {}>",
                            param,
                            addr,
                            e,
                            width = max_name_len
                        );
                    }
                }
            }
        }

        // Query other multi-item parameters (MaxItem != 1 and != 4)
        // These are per-input-channel (1-7) or per-trigger (1-3) parameters
        let other_params: Vec<_> = SpaParam::iter()
            .filter(|p| !p.is_system_wide() && p.max_items() != 4)
            .collect();

        if !other_params.is_empty() {
            info!("  Other (per-channel/table):");
            for param in other_params {
                let addr = param.address();
                let n_items = param.max_items();
                // Query all items for this parameter
                for item in 1..=n_items {
                    match gcs.query(&format!("SPA? {item} 0x{addr:08X}")) {
                        Ok(response) => {
                            let value = response
                                .split('=')
                                .nth(1)
                                .map(|s| s.trim())
                                .unwrap_or(response.trim());
                            if item == 1 {
                                info!(
                                    "    {:width$} (0x{:08X}) : {}",
                                    param,
                                    addr,
                                    value,
                                    width = max_name_len
                                );
                            } else {
                                // Continuation lines for same param, pad to align
                                info!(
                                    "    {:width$}               : {}",
                                    "",
                                    value,
                                    width = max_name_len
                                );
                            }
                        }
                        Err(e) => {
                            if item == 1 {
                                info!(
                                    "    {:width$} (0x{:08X}) : <error: {}>",
                                    param,
                                    addr,
                                    e,
                                    width = max_name_len
                                );
                            } else {
                                info!(
                                    "    {:width$}               : <error: {}>",
                                    "",
                                    e,
                                    width = max_name_len
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    info!("Done!");
    Ok(())
}

// ==================== REPL Command ====================

const HISTORY_MAX_LINES: usize = 500;

fn get_history_path() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let config_dir = std::path::Path::new(&home).join(".config");
    std::fs::create_dir_all(&config_dir).ok()?;
    Some(config_dir.join("fsm_tool_hist.txt"))
}

fn truncate_history_file(path: &std::path::Path, max_lines: usize) {
    if let Ok(contents) = std::fs::read_to_string(path) {
        let lines: Vec<&str> = contents.lines().collect();
        if lines.len() > max_lines {
            let skip_count = lines.len() - max_lines;
            let truncated: Vec<&str> = lines.into_iter().skip(skip_count).collect();
            let _ = std::fs::write(path, truncated.join("\n") + "\n");
        }
    }
}

fn cmd_repl(ip: &str) -> Result<()> {
    println!("Connecting to E-727 at {ip}...");
    let mut fsm = E727::connect_ip(ip)?;

    {
        let gcs = fsm.device_mut();
        let response = gcs.query("*IDN?")?;
        println!("Connected: {}", response.trim());
    }

    println!();
    println!("GCS REPL - Enter commands (queries end with '?'), 'quit' to exit");
    println!("Use Up/Down arrows for command history");
    println!("Examples: *IDN?, POS?, SVO 1 1, MOV 1 1000");
    println!();

    let mut rl = DefaultEditor::new()?;

    let history_path = get_history_path();
    if let Some(ref path) = history_path {
        truncate_history_file(path, HISTORY_MAX_LINES);
        if path.exists() {
            let _ = rl.load_history(path);
        }
    }

    // Helper to handle connection errors with reconnect (exponential backoff)
    fn handle_connection_error(fsm: &mut E727, e: &dyn std::fmt::Display) {
        println!("Connection disrupted: {e}");
        let mut backoff_ms = 100u64;
        let mut attempts = 0u32;
        const MAX_BACKOFF_MS: u64 = 3000;

        loop {
            attempts += 1;
            print!("\rReconnecting... (attempt {attempts}, next retry in {backoff_ms}ms)      ");
            io::stdout().flush().ok();
            match fsm.reconnect() {
                Ok(()) => {
                    println!("\rReconnected successfully                                      ");
                    break;
                }
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(backoff_ms));
                    backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                }
            }
        }
    }

    loop {
        match rl.readline("> ") {
            Ok(line) => {
                let cmd = line.trim();
                if cmd.is_empty() {
                    continue;
                }

                // Add to history and save
                let _ = rl.add_history_entry(&line);
                if let Some(ref path) = history_path {
                    let _ = rl.save_history(path);
                }

                if cmd.eq_ignore_ascii_case("quit") || cmd.eq_ignore_ascii_case("exit") {
                    println!("Bye!");
                    break;
                }

                // Get mutable ref to device for raw commands
                let gcs = fsm.device_mut();

                // If command ends with ?, it's a query - expect response
                if cmd.ends_with('?') {
                    // Send query and read response
                    if let Err(e) = gcs.send(cmd) {
                        handle_connection_error(&mut fsm, &e);
                        continue;
                    }
                    match gcs.read() {
                        Ok(response) => {
                            for line in response.lines() {
                                println!("{line}");
                            }
                        }
                        Err(e) => {
                            handle_connection_error(&mut fsm, &e);
                            continue;
                        }
                    }
                } else {
                    // It's a command - send it
                    if let Err(e) = gcs.send(cmd) {
                        handle_connection_error(&mut fsm, &e);
                        continue;
                    }

                    // Try to read any response with 200ms timeout
                    let gcs = fsm.device_mut();
                    gcs.set_timeout(Duration::from_millis(200));
                    match gcs.read() {
                        Ok(response) => {
                            for line in response.lines() {
                                println!("{line}");
                            }
                        }
                        Err(_) => {
                            // Timeout is expected for commands with no response
                        }
                    }
                    // Restore normal timeout
                    gcs.set_timeout(Duration::from_secs(7));
                }

                // Query and decode ERR? after any command, only show if error
                match fsm.last_error_decoded() {
                    Ok((code, err)) => {
                        if err != PiErrorCode::NoError {
                            println!("ERR: {} ({code})", err.description());
                        }
                    }
                    Err(e) => {
                        handle_connection_error(&mut fsm, &e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                break;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                println!("Error: {err}");
                break;
            }
        }
    }

    Ok(())
}

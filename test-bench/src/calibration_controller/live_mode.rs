//! Live TUI mode for continuous calibration updates.
//!
//! Provides a real-time visualization of calibration progress using ratatui,
//! continuously cycling through grid positions and updating the transform estimate.

use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::Terminal;
use shared::optical_alignment::{estimate_affine_transform, OpticalAlignment, PointCorrespondence};

use super::init::{initialize, CalibrationContext};
use super::render_tui::ui;
use super::types::{App, CalibrationPoint, Measurement, MeasurementBuffer};
use super::Args;

/// Check if quit key was pressed. Returns true if should quit.
pub fn check_quit() -> bool {
    if event::poll(Duration::from_millis(0)).unwrap_or(false) {
        if let Ok(Event::Key(key)) = event::read() {
            return key.code == KeyCode::Char('q') || key.code == KeyCode::Esc;
        }
    }
    false
}

/// Sleep while checking for quit key. Returns true if should quit.
pub fn sleep_with_quit_check(duration: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < duration {
        if check_quit() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// Run the live TUI calibration mode.
///
/// Continuously cycles through grid positions, collecting measurements
/// and updating the display in real-time. Press 'q' to quit.
pub fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize connections and discover sensor
    let CalibrationContext {
        pattern_client,
        tracking_collector,
        sensor_width,
        sensor_height,
        positions,
    } = initialize(args, false)?;

    let total_positions = positions.len();

    // Initialize measurement buffers for each grid position
    let mut buffers: Vec<MeasurementBuffer> = positions
        .iter()
        .enumerate()
        .map(|(i, (dx, dy))| {
            let row = i / args.grid_size;
            let col = i % args.grid_size;
            MeasurementBuffer::new(*dx, *dy, row, col, args.history)
        })
        .collect();

    let settle_duration = Duration::from_secs_f64(args.settle_secs);
    let timeout = Duration::from_secs_f64(args.timeout_secs);

    // Set up ratatui terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::new(
        sensor_width,
        sensor_height,
        args.display_width,
        args.display_height,
        args.spot_fwhm,
    );

    let mut cycle_count: u32 = 0;
    let mut current_alignment: Option<OpticalAlignment> = None;

    // Main live loop
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        loop {
            cycle_count += 1;

            for position_idx in 0..total_positions {
                // Check for quit key
                if check_quit() {
                    return Ok(());
                }

                let buffer = &mut buffers[position_idx];
                let row = buffer.row;
                let col = buffer.col;
                let display_x = buffer.display_x;
                let display_y = buffer.display_y;

                app.log(format!(
                    "Spot [{row},{col}] → display ({display_x:.0}, {display_y:.0})"
                ));

                // Send spot command via REST
                if pattern_client
                    .spot(display_x, display_y, args.spot_fwhm, args.spot_intensity)
                    .is_err()
                {
                    app.log(format!("  ✗ HTTP send failed for [{row},{col}]"));
                    if check_quit() {
                        return Ok(());
                    }
                    continue;
                }

                // Wait for settle (with quit check)
                if sleep_with_quit_check(settle_duration) {
                    return Ok(());
                }

                // Collect measurements
                let start = Instant::now();
                let mut collected = 0;

                while collected < args.measurements_per_position {
                    if start.elapsed() > timeout {
                        break;
                    }

                    // Check quit during measurement collection
                    if check_quit() {
                        return Ok(());
                    }

                    let msgs = tracking_collector.poll();
                    for msg in msgs {
                        buffer.push(Measurement {
                            sensor_x: msg.x,
                            sensor_y: msg.y,
                            diameter: msg.shape.diameter,
                        });
                        collected += 1;
                    }

                    if collected < args.measurements_per_position {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }

                // Log measurement result
                if let Some((sx, sy, dia)) = buffer.average() {
                    app.log(format!(
                        "  ✓ sensor ({sx:.0}, {sy:.0}) dia={dia:.1}px [{collected} pts]"
                    ));
                } else if collected == 0 {
                    app.log("  ✗ timeout, no measurements".to_string());
                }

                // Build current calibration points from all buffers
                let calibration_points: Vec<CalibrationPoint> = buffers
                    .iter()
                    .filter_map(|b| {
                        b.average().map(|(sx, sy, dia)| CalibrationPoint {
                            row: b.row,
                            col: b.col,
                            display_x: b.display_x,
                            display_y: b.display_y,
                            sensor_x: sx,
                            sensor_y: sy,
                            avg_diameter: dia,
                        })
                    })
                    .collect();

                // Re-estimate transform if we have enough points
                if calibration_points.len() >= 3 {
                    let correspondences: Vec<PointCorrespondence> = calibration_points
                        .iter()
                        .map(|p| {
                            PointCorrespondence::new(
                                p.display_x,
                                p.display_y,
                                p.sensor_x,
                                p.sensor_y,
                            )
                        })
                        .collect();

                    current_alignment = estimate_affine_transform(&correspondences);
                }

                // Update app state and render
                app.update(
                    calibration_points,
                    current_alignment.clone(),
                    Some((row, col)),
                    Some((display_x, display_y)),
                    cycle_count,
                    collected,
                );

                terminal.draw(|frame| ui(frame, &app))?;
            }
        }
    })();

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

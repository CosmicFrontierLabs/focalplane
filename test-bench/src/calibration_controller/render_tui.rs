//! Ratatui TUI rendering for live calibration mode.

use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        canvas::{Canvas, Points},
        Block, Borders, Paragraph,
    },
    Frame,
};
use shared::algo::min_max_scan::MinMaxScan;
use shared::optical_alignment::OpticalAlignment;

use super::types::App;

/// Project display corners onto sensor using the affine transform.
/// Returns [(TL), (TR), (BR), (BL)] sensor coordinates.
pub fn project_display_corners(
    alignment: &OpticalAlignment,
    display_width: u32,
    display_height: u32,
) -> [(f64, f64); 4] {
    let w = display_width as f64;
    let h = display_height as f64;

    [
        alignment.display_to_sensor(0.0, 0.0), // Top-left
        alignment.display_to_sensor(w, 0.0),   // Top-right
        alignment.display_to_sensor(w, h),     // Bottom-right
        alignment.display_to_sensor(0.0, h),   // Bottom-left
    ]
}

/// Render the TUI using ratatui.
pub fn ui(frame: &mut Frame, app: &App) {
    let good_threshold = app.expected_diameter * 1.3;
    let moderate_threshold = app.expected_diameter * 2.0;

    // Main layout: header, canvas, info panel, log panel
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(15),   // Canvas
            Constraint::Length(6), // Info panel
            Constraint::Length(8), // Log panel
        ])
        .split(frame.area());

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "CALIBRATION CONTROLLER",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::raw(format!("Cycle: {:4}", app.cycle_count)),
        Span::raw("  "),
        Span::styled(
            format!("Track: {:4}", app.tracking_count),
            Style::default().fg(Color::Yellow),
        ),
    ]))
    .block(Block::default().borders(Borders::ALL).title("Live Mode"));
    frame.render_widget(header, chunks[0]);

    // Canvas for sensor view with 20% margin on each edge
    let sw = app.sensor_width as f64;
    let sh = app.sensor_height as f64;
    let margin = 0.2;
    let x_min = -margin * sw;
    let x_max = sw * (1.0 + margin);
    let y_min = -margin * sh;
    let y_max = sh * (1.0 + margin);

    let canvas = Canvas::default()
        .block(Block::default().borders(Borders::ALL).title(format!(
            "Sensor {}x{} → Display {}x{}",
            app.sensor_width, app.sensor_height, app.display_width, app.display_height
        )))
        .x_bounds([x_min, x_max])
        .y_bounds([y_min, y_max])
        .marker(symbols::Marker::Braille)
        .paint(|ctx| {
            // Draw display corners if we have alignment
            if let Some(ref align) = app.alignment {
                let corners = project_display_corners(align, app.display_width, app.display_height);
                // Draw corner box edges
                for i in 0..4 {
                    let (x1, y1) = corners[i];
                    let (x2, y2) = corners[(i + 1) % 4];
                    // Draw line as points
                    for j in 0..=20 {
                        let t = j as f64 / 20.0;
                        let x = x1 + t * (x2 - x1);
                        let y = sh - (y1 + t * (y2 - y1)); // Flip Y for canvas
                        ctx.draw(&Points {
                            coords: &[(x, y)],
                            color: Color::DarkGray,
                        });
                    }
                }
            }

            // Draw calibration points
            for point in &app.points {
                let is_current = app
                    .current_position
                    .map(|(r, c)| r == point.row && c == point.col)
                    .unwrap_or(false);

                let color = if is_current {
                    Color::Cyan
                } else if point.avg_diameter <= good_threshold {
                    Color::Green
                } else if point.avg_diameter <= moderate_threshold {
                    Color::Yellow
                } else {
                    Color::Red
                };

                // Flip Y for canvas (canvas Y=0 is bottom)
                ctx.draw(&Points {
                    coords: &[(point.sensor_x, sh - point.sensor_y)],
                    color,
                });
            }

            // Draw current position marker
            if let (Some(ref align), Some((dx, dy))) = (&app.alignment, app.current_display_xy) {
                let (sx, sy) = align.display_to_sensor(dx, dy);
                ctx.draw(&Points {
                    coords: &[(sx, sh - sy)],
                    color: Color::Cyan,
                });
            }
        });
    frame.render_widget(canvas, chunks[1]);

    // Info panel
    let mut info_lines = vec![Line::from(vec![
        Span::styled("●", Style::default().fg(Color::Green)),
        Span::raw("=good  "),
        Span::styled("●", Style::default().fg(Color::Yellow)),
        Span::raw("=moderate  "),
        Span::styled("●", Style::default().fg(Color::Red)),
        Span::raw("=poor  "),
        Span::styled("●", Style::default().fg(Color::Cyan)),
        Span::raw("=current"),
    ])];

    if let Some(ref align) = app.alignment {
        let (sx, sy) = align.scale();
        let rot = align.rotation_degrees();
        let rms = align.rms_error.unwrap_or(0.0);
        info_lines.push(Line::from(format!(
            "Transform: scale=({:.4}, {:.4}) rot={:.2}° rms={:.2}px  Points: {}",
            sx,
            sy,
            rot,
            rms,
            app.points.len()
        )));
    } else {
        info_lines.push(Line::from(Span::styled(
            "Transform: (need 3+ points)",
            Style::default().fg(Color::DarkGray),
        )));
    }

    // Diameter stats
    let diameters: Vec<f64> = app.points.iter().map(|p| p.avg_diameter).collect();
    if let Ok((min_dia, max_dia)) = MinMaxScan::new(&diameters).min_max() {
        let avg_dia: f64 = diameters.iter().sum::<f64>() / diameters.len() as f64;
        info_lines.push(Line::from(format!(
            "Diameter: min={min_dia:.1}px max={max_dia:.1}px avg={avg_dia:.1}px (good<{good_threshold:.1}px)"
        )));
    }

    info_lines.push(Line::from(Span::styled(
        "Press 'q' to quit",
        Style::default().fg(Color::DarkGray),
    )));

    let info =
        Paragraph::new(info_lines).block(Block::default().borders(Borders::ALL).title("Info"));
    frame.render_widget(info, chunks[2]);

    // Log panel - show last N log entries, scrolled to bottom
    let log_height = chunks[3].height.saturating_sub(2) as usize; // Account for borders
    let log_lines: Vec<Line> = app
        .logs
        .iter()
        .rev()
        .take(log_height)
        .rev()
        .map(|s| {
            Line::from(Span::styled(
                s.as_str(),
                Style::default().fg(Color::DarkGray),
            ))
        })
        .collect();

    let logs = Paragraph::new(log_lines).block(Block::default().borders(Borders::ALL).title("Log"));
    frame.render_widget(logs, chunks[3]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_display_corners() {
        // Identity-ish transform (scale 1, no rotation, offset 100,100)
        let alignment = OpticalAlignment {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx: 100.0,
            ty: 100.0,
            timestamp: 0,
            num_points: 4,
            rms_error: None,
        };

        let corners = project_display_corners(&alignment, 1000, 1000);

        // Top-left: (0,0) -> (100, 100)
        assert!((corners[0].0 - 100.0).abs() < 1e-10);
        assert!((corners[0].1 - 100.0).abs() < 1e-10);

        // Top-right: (1000,0) -> (1100, 100)
        assert!((corners[1].0 - 1100.0).abs() < 1e-10);
        assert!((corners[1].1 - 100.0).abs() < 1e-10);

        // Bottom-right: (1000,1000) -> (1100, 1100)
        assert!((corners[2].0 - 1100.0).abs() < 1e-10);
        assert!((corners[2].1 - 1100.0).abs() < 1e-10);

        // Bottom-left: (0,1000) -> (100, 1100)
        assert!((corners[3].0 - 100.0).abs() < 1e-10);
        assert!((corners[3].1 - 1100.0).abs() < 1e-10);
    }
}

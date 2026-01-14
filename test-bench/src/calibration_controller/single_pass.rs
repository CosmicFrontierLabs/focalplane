//! Single-pass calibration mode.
//!
//! Runs through each grid position once, collecting measurements and
//! estimating the display-to-sensor affine transform.

use std::time::{Duration, Instant};

use shared::optical_alignment::{estimate_affine_transform, PointCorrespondence};

use super::init::{initialize, CalibrationContext};
use super::types::CalibrationPoint;
use super::Args;

/// Run a single-pass calibration sequence.
///
/// Iterates through each grid position, sends spot commands, collects
/// tracking measurements, and estimates the affine transform.
pub fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    println!("Calibration Controller (Single Pass Mode)");
    println!("==========================================");

    // Initialize connections and discover sensor
    let CalibrationContext {
        pattern_client,
        tracking_collector,
        positions,
        ..
    } = initialize(args, true)?;

    let total_positions = positions.len();

    // Calibration state
    let mut calibration_points: Vec<CalibrationPoint> = Vec::with_capacity(total_positions);
    let settle_duration = Duration::from_secs_f64(args.settle_secs);
    let timeout = Duration::from_secs_f64(args.timeout_secs);

    // Run calibration
    for (i, (display_x, display_y)) in positions.iter().enumerate() {
        let row = i / args.grid_size;
        let col = i % args.grid_size;

        println!(
            "[{}/{}] Moving to display position ({:.1}, {:.1})",
            i + 1,
            total_positions,
            display_x,
            display_y
        );

        // Send spot command via REST
        pattern_client.spot(*display_x, *display_y, args.spot_fwhm, args.spot_intensity)?;

        // Wait for settle
        std::thread::sleep(settle_duration);

        // Collect measurements (x, y, diameter)
        let mut measurements: Vec<(f64, f64, f64)> = Vec::new();
        let start = Instant::now();

        while measurements.len() < args.measurements_per_position {
            if start.elapsed() > timeout {
                eprintln!(
                    "  Timeout waiting for measurements ({} of {} received)",
                    measurements.len(),
                    args.measurements_per_position
                );
                break;
            }

            let msgs = tracking_collector.poll();
            for msg in msgs {
                measurements.push((msg.x, msg.y, msg.shape.diameter));
            }

            if measurements.len() < args.measurements_per_position {
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        if measurements.is_empty() {
            eprintln!("  No measurements received, skipping position");
            continue;
        }

        // Average measurements
        let n = measurements.len() as f64;
        let (sum_x, sum_y, sum_dia) = measurements
            .iter()
            .fold((0.0, 0.0, 0.0), |(ax, ay, ad), (x, y, d)| {
                (ax + x, ay + y, ad + d)
            });
        let avg_sensor_x = sum_x / n;
        let avg_sensor_y = sum_y / n;
        let avg_diameter = sum_dia / n;

        println!(
            "  -> sensor ({:.2}, {:.2}) dia={:.2}px [{} measurements]",
            avg_sensor_x,
            avg_sensor_y,
            avg_diameter,
            measurements.len()
        );

        calibration_points.push(CalibrationPoint {
            row,
            col,
            display_x: *display_x,
            display_y: *display_y,
            sensor_x: avg_sensor_x,
            sensor_y: avg_sensor_y,
            avg_diameter,
        });
    }

    // Clear display
    println!();
    println!("Clearing display...");
    pattern_client.clear()?;

    // Convert to correspondences for affine estimation
    let correspondences: Vec<PointCorrespondence> = calibration_points
        .iter()
        .map(|p| PointCorrespondence::new(p.display_x, p.display_y, p.sensor_x, p.sensor_y))
        .collect();

    // Estimate transform
    println!();
    println!("=== Calibration Results ===");
    println!("Collected {} point correspondences", correspondences.len());

    if correspondences.len() < 3 {
        eprintln!("Error: Need at least 3 points for affine transform estimation");
        return Ok(());
    }

    match estimate_affine_transform(&correspondences) {
        Some(alignment) => {
            let (scale_x, scale_y) = alignment.scale();
            let rotation_deg = alignment.rotation_degrees();

            println!();
            println!("Affine Transform:");
            println!("  Scale X:     {scale_x:.6}");
            println!("  Scale Y:     {scale_y:.6}");
            println!("  Rotation:    {rotation_deg:.4} degrees");
            println!("  Translation: ({:.2}, {:.2})", alignment.tx, alignment.ty);
            if let Some(rms) = alignment.rms_error {
                println!("  RMS Error:   {rms:.4} pixels");
            }

            println!();
            println!("Matrix coefficients [a, b, c, d, tx, ty]:");
            println!(
                "  [{:.6}, {:.6}, {:.6}, {:.6}, {:.2}, {:.2}]",
                alignment.a, alignment.b, alignment.c, alignment.d, alignment.tx, alignment.ty
            );
        }
        None => {
            eprintln!("Error: Failed to estimate affine transform");
        }
    }

    Ok(())
}

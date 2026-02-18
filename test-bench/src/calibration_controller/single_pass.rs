//! Single-pass calibration mode.
//!
//! Runs through each grid position once, collecting measurements and
//! estimating the display-to-sensor affine transform.

use std::time::Duration;

use shared::optical_alignment::{estimate_affine_transform, PointCorrespondence};

use super::init::{initialize, CalibrationContext};
use super::types::CalibrationPoint;
use super::Args;

/// Run a single-pass calibration sequence.
///
/// Iterates through each grid position, sends spot commands, collects
/// tracking measurements, and estimates the affine transform.
pub async fn run(args: &Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Calibration Controller (Single Pass Mode)");
    println!("==========================================");

    // Initialize connections and discover sensor
    let CalibrationContext {
        pattern_client,
        tracking_collector,
        mut fgs,
        positions,
        ..
    } = initialize(args, true).await?;

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

        // Disable tracking before moving spot (forces reacquire on new position)
        fgs.disable_tracking().await?;

        // Send spot command via REST
        pattern_client
            .spot(*display_x, *display_y, args.spot_fwhm, args.spot_intensity)
            .await?;

        // Wait for settle
        tokio::time::sleep(settle_duration).await;

        // Enable tracking and wait for lock-on
        match fgs.enable_tracking(timeout).await {
            Ok(_) => println!("  Tracking locked on"),
            Err(e) => {
                eprintln!("  Tracking failed to lock: {e}, skipping position");
                continue;
            }
        }

        // Discard stale samples from acquisition, then collect measurements
        let measurements: Vec<(f64, f64, f64)> = match tracking_collector.collect_with_discard(
            args.discard_samples,
            args.measurements_per_position,
            timeout,
            timeout,
        ) {
            Ok(msgs) => msgs.iter().map(|m| (m.x, m.y, m.shape.diameter)).collect(),
            Err(e) => {
                eprintln!("  Collection incomplete: {e}");
                Vec::new()
            }
        };

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

    // Disable tracking
    fgs.disable_tracking().await?;

    // Clear display
    println!();
    println!("Clearing display...");
    pattern_client.clear().await?;

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

            let output_path = args
                .output_json
                .clone()
                .unwrap_or_else(|| "optical_alignment.json".into());
            match alignment.save_to_file(&output_path) {
                Ok(()) => println!("\nSaved to {}", output_path.display()),
                Err(e) => eprintln!("\nFailed to write {}: {e}", output_path.display()),
            }
        }
        None => {
            eprintln!("Error: Failed to estimate affine transform");
        }
    }

    Ok(())
}

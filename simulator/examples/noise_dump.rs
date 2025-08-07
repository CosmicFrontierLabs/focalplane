//! Sensor noise characteristics table generator and plotter.
//!
//! This example generates:
//! 1. A table showing read noise and dark current levels for all sensors
//! 2. A plot of dark current vs temperature curves for all sensors
//!
//! # Output Format
//!
//! - Markdown table showing read noise and dark current at key temperatures
//! - PNG plot showing dark current curves from -20°C to +20°C
//!
//! # Usage
//!
//! ```bash
//! cargo run --example noise_dump
//! ```

use plotters::prelude::*;
use simulator::hardware::dark_current::{MAX_TEMP_C, MIN_TEMP_C};
use simulator::hardware::sensor::models::ALL_SENSORS;
use simulator::units::{Temperature, TemperatureExt};

fn plot_dark_current_curves() -> Result<(), Box<dyn std::error::Error>> {
    // Create plot
    let root = BitMapBackend::new("dark_current_curves.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Dark Current vs Temperature for All Sensors",
            ("sans-serif", 40),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(MIN_TEMP_C..MAX_TEMP_C, (0.001f64..10f64).log_scale())?;

    chart
        .configure_mesh()
        .x_desc("Temperature (°C)")
        .y_desc("Dark Current (e⁻/pixel/second)")
        .y_label_formatter(&|y| format!("{y:.0e}"))
        .axis_desc_style(("sans-serif", 20))
        .draw()?;

    // Plot each sensor's dark current curve
    let sensors = &*ALL_SENSORS;
    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK, &YELLOW];

    for (idx, sensor) in sensors.iter().enumerate() {
        let color = colors[idx % colors.len()];

        // Generate temperature points for smooth curve
        let min_temp_int = (MIN_TEMP_C * 10.0) as i32;
        let max_temp_int = (MAX_TEMP_C * 10.0) as i32;
        let temps: Vec<f64> = (min_temp_int..=max_temp_int)
            .map(|t| t as f64 / 10.0)
            .collect();
        let mut curve_points = Vec::new();

        for temp in &temps {
            // Only plot points within the valid range
            let temperature = Temperature::from_celsius(*temp);
            if let Ok(dark_current) = sensor
                .dark_current_estimator
                .estimate_at_temperature(temperature)
            {
                curve_points.push((*temp, dark_current));
            }
        }

        // Draw the curve
        if !curve_points.is_empty() {
            // Calculate doubling temperature for this sensor
            let doubling_temp = sensor
                .dark_current_estimator
                .calculate_doubling_temperature();

            chart
                .draw_series(LineSeries::new(curve_points.clone(), color.stroke_width(1)))?
                .label(&sensor.name)
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3))
                });

            // Add annotation on the curve
            // Find a good position around the middle of the temperature range
            let annotation_temp = 10.0; // Place annotation at 10°C
            if let Ok(dark_current) = sensor
                .dark_current_estimator
                .estimate_at_temperature(Temperature::from_celsius(annotation_temp))
            {
                let annotation_text = format!("{doubling_temp:.1}°C/2×");
                chart.draw_series(std::iter::once(Text::new(
                    annotation_text,
                    (annotation_temp, dark_current),
                    ("sans-serif", 20).into_font().color(color),
                )))?;
            }
        }
    }

    // Configure and draw legend
    chart
        .configure_series_labels()
        .label_font(("sans-serif", 16))
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    println!("\nDark current curves saved to: dark_current_curves.png");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Sensor Noise Characteristics at Different Temperatures");
    println!("====================================================");
    println!("Assumptions: Frame rate = 10 Hz (100ms exposure)");
    println!();

    // Use all available sensors
    let sensors = &*ALL_SENSORS;

    // Temperature points to evaluate
    let temperatures = vec![-20.0, -10.0, 0.0, 10.0, 20.0];

    // Frame rate for read noise estimation (10 Hz = 100ms exposure)
    let exposure_duration = std::time::Duration::from_millis(100);

    // Print header row with temperatures
    print!("| Sensor | Parameter |");
    for temp in &temperatures {
        print!(" {temp}°C |");
    }
    println!();

    // Print separator (markdown style)
    print!("|--------|-----------|");
    for _ in &temperatures {
        print!("------|");
    }
    println!();

    // For each sensor, print read noise and dark current rows
    for config in sensors {
        // Sensor name row with read noise values
        print!("| {} | Read Noise (e⁻) |", config.name);
        for temp in &temperatures {
            let read_noise = config
                .read_noise_estimator
                .estimate(*temp, exposure_duration)
                .unwrap_or(0.0);
            print!(" {read_noise:.2} |");
        }
        println!();

        // Dark current row
        print!("| | Dark Current (e⁻/px/s) |");
        for temp in &temperatures {
            let dark_current = config
                .dark_current_estimator
                .estimate_at_temperature(Temperature::from_celsius(*temp))
                .expect("Temperature should be within interpolation range");
            print!(" {dark_current:.4} |");
        }
        println!();
    }

    // Now plot the dark current curves
    plot_dark_current_curves()?;

    Ok(())
}

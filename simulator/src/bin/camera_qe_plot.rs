//! Plot QE curves for all known camera sensors
//!
//! This example generates a PNG plot of quantum efficiency curves
//! for all predefined camera models.
//!
//! Usage:
//! ```
//! cargo run --example camera_qe_plot
//! ```

use once_cell::sync::Lazy;
use plotters::prelude::*;
use simulator::hardware::sensor::{models, SensorConfig};

const OUTPUT_PATH: &str = "plots/camera_qe_curves.png";
const WAVELENGTH_MIN: f32 = 150.0;
const WAVELENGTH_MAX: f32 = 1150.0;
const TITLE: &str = "Camera Quantum Efficiency Curves";
const SAMPLE_POINTS: usize = 100;

// A struct to hold sensor information including display name and color
struct SensorInfo<'a> {
    sensor: &'a Lazy<SensorConfig>,
    name: &'a str,
    color: RGBColor,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating QE curves plot...");

    // Create a list of all camera sensors with display info
    let sensor_list = vec![
        SensorInfo {
            sensor: &models::GSENSE4040BSI,
            name: "GSENSE4040BSI",
            color: RED,
        },
        SensorInfo {
            sensor: &models::GSENSE6510BSI,
            name: "GSENSE6510BSI",
            color: MAGENTA,
        },
        SensorInfo {
            sensor: &models::HWK4123,
            name: "HWK4123",
            color: GREEN,
        },
        SensorInfo {
            sensor: &models::IMX455,
            name: "IMX455",
            color: BLUE,
        },
    ];

    // Create plots directory if it doesn't exist
    std::fs::create_dir_all("plots")?;

    // Create a new drawing area
    let root = BitMapBackend::new(OUTPUT_PATH, (1024, 768)).into_drawing_area();

    // Fill the background with white
    root.fill(&WHITE)?;

    // Create chart
    let mut chart = ChartBuilder::on(&root)
        .caption(TITLE, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(WAVELENGTH_MIN..WAVELENGTH_MAX, 0.0..1.0)?;

    // Configure the mesh with tick marks every 50nm
    chart
        .configure_mesh()
        .x_labels(21) // 150, 200, 250, 300, ... 1050, 1100, 1150
        .x_label_formatter(&|x| format!("{:.0}", x)) // Remove decimal places
        .y_labels(11) // 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        .y_label_formatter(&|y| format!("{:.1}", y)) // One decimal place for y-axis
        .x_desc("Wavelength (nm)")
        .y_desc("Quantum Efficiency")
        .axis_desc_style(("sans-serif", 18))
        .draw()?;

    // Sample points for the curves
    let wavelengths: Vec<f32> = (0..SAMPLE_POINTS)
        .map(|i| {
            WAVELENGTH_MIN
                + (WAVELENGTH_MAX - WAVELENGTH_MIN) * (i as f32) / (SAMPLE_POINTS as f32 - 1.0)
        })
        .collect();

    // Draw each sensor's QE curve
    for sensor_info in &sensor_list {
        // Generate QE points for this sensor
        let qe_points: Vec<(f32, f64)> = wavelengths
            .iter()
            .map(|&wavelength| {
                (
                    wavelength,
                    sensor_info.sensor.qe_at_wavelength(wavelength as u32),
                )
            })
            .collect();

        // Draw the curve
        chart
            .draw_series(LineSeries::new(qe_points, &sensor_info.color))?
            .label(sensor_info.name)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], sensor_info.color));
    }

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // Save the plot
    root.present()?;

    println!("Plot saved to: {}", OUTPUT_PATH);
    Ok(())
}

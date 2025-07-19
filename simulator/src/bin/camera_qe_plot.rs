//! Plot QE curves for all known camera sensors
//!
//! This example generates a PNG plot of quantum efficiency curves
//! for all predefined camera models.
//!
//! Usage:
//! ```
//! cargo run --example camera_qe_plot
//! ```

use plotters::prelude::*;
use simulator::hardware::sensor::models::ALL_SENSORS;

const OUTPUT_PATH: &str = "plots/camera_qe_curves.png";
const WAVELENGTH_MIN: f32 = 150.0;
const WAVELENGTH_MAX: f32 = 1150.0;
const TITLE: &str = "Camera Quantum Efficiency Curves";
const SAMPLE_POINTS: usize = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging from environment variables
    env_logger::init();

    println!("Generating QE curves plot...");

    // Define colors for each sensor
    let colors = [RED, MAGENTA, GREEN, BLUE];

    // Zip sensors with colors for plotting
    let sensor_color_pairs: Vec<_> = ALL_SENSORS.iter().zip(colors.iter()).collect();

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
        .x_label_formatter(&|x| format!("{x:.0}")) // Remove decimal places
        .y_labels(11) // 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        .y_label_formatter(&|y| format!("{y:.1}")) // One decimal place for y-axis
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
    for (sensor, color) in &sensor_color_pairs {
        // Generate QE points for this sensor
        let qe_points: Vec<(f32, f64)> = wavelengths
            .iter()
            .map(|&wavelength| (wavelength, sensor.qe_at_wavelength(wavelength as u32)))
            .collect();

        // Capture color for legend closure
        let legend_color = **color;

        // Draw the curve
        chart
            .draw_series(LineSeries::new(qe_points, color))?
            .label(&sensor.name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], legend_color));
    }

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // Save the plot
    root.present()?;

    println!("Plot saved to: {OUTPUT_PATH}");
    Ok(())
}

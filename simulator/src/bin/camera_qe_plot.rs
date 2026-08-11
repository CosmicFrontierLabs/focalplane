//! Plot QE curves for all known camera sensors.
//!
//! Generates a PNG plot of quantum efficiency curves for all predefined
//! camera models. Output is saved to `plots/camera_qe_curves.png`.
//!
//! Usage:
//! ```
//! cargo run --release --bin camera_qe_plot
//! ```

use plotters::prelude::*;
use simulator::hardware::sensor::models::ALL_SENSORS;
use simulator::plotting::save_plot_png;

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

    save_plot_png(OUTPUT_PATH, (1024, 768), |root| {
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(root)
            .caption(TITLE, ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(WAVELENGTH_MIN..WAVELENGTH_MAX, 0.0..1.0)?;

        chart
            .configure_mesh()
            .x_labels(21)
            .x_label_formatter(&|x| format!("{x:.0}"))
            .y_labels(11)
            .y_label_formatter(&|y| format!("{y:.1}"))
            .x_desc("Wavelength (nm)")
            .y_desc("Quantum Efficiency")
            .axis_desc_style(("sans-serif", 18))
            .draw()?;

        let wavelengths: Vec<f32> = (0..SAMPLE_POINTS)
            .map(|i| {
                WAVELENGTH_MIN
                    + (WAVELENGTH_MAX - WAVELENGTH_MIN) * (i as f32) / (SAMPLE_POINTS as f32 - 1.0)
            })
            .collect();

        for (sensor, color) in &sensor_color_pairs {
            let qe_points: Vec<(f32, f64)> = wavelengths
                .iter()
                .map(|&wavelength| (wavelength, sensor.qe_at_wavelength(wavelength as u32)))
                .collect();

            let legend_color = **color;

            chart
                .draw_series(LineSeries::new(qe_points, color))?
                .label(&sensor.name)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], legend_color));
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        Ok(())
    })?;

    println!("Plot saved to: {OUTPUT_PATH}");
    Ok(())
}

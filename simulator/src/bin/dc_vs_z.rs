//! Dark current vs zodiacal background analysis tool.
//!
//! Compares sensor dark current against zodiacal background flux across a range
//! of sky positions (from solar exclusion angle to opposition). Helps determine
//! whether dark current or sky background is the dominant noise source for a
//! given telescope/sensor combination.
//!
//! Outputs a table of zodiacal brightness values and optionally generates a
//! comparison plot.
//!
//! Usage:
//! ```
//! cargo run --release --bin dc_vs_z -- -t demo50cm -s imx455 -e 45.0
//! cargo run --release --bin dc_vs_z -- -t demo50cm -s imx455 -e 45.0 --plot
//! ```

use clap::Parser;
use plotters::prelude::*;
use simulator::photometry::zodiacal::{
    SolarAngularCoordinates, ZodiacalLight, ELONG_OF_MIN, LAT_OF_MIN,
};
use simulator::photometry::{photoconversion, QuantumEfficiency};
use simulator::shared_args::{SensorModel, TelescopeModel};
use simulator::units::{AreaExt, LengthExt, TemperatureExt};
use std::time::Duration;

#[derive(Parser)]
#[command(name = "dc_vs_z")]
#[command(about = "Dark current vs zodiacal background analysis tool")]
#[command(version)]
struct Args {
    /// Telescope configuration to use
    #[arg(short, long)]
    telescope: TelescopeModel,

    /// Sensor configuration to use  
    #[arg(short, long)]
    sensor: SensorModel,

    /// Solar exclusion angle in degrees
    #[arg(short = 'e', long, value_name = "DEGREES")]
    solar_exclusion: f64,

    /// Generate comparison plot vs dark current
    #[arg(long)]
    plot: bool,

    /// Output file for plot (default: dc_vs_zodiacal.png)
    #[arg(long, default_value = "dc_vs_zodiacal.png")]
    plot_output: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Dark Current vs Zodiacal Analysis");
    println!("=================================");
    println!("Telescope: {}", args.telescope);
    println!("Sensor: {}", args.sensor);
    println!("Solar exclusion: {}°", args.solar_exclusion);
    println!();

    let telescope_config = args.telescope.to_config();
    let sensor_config = args.sensor.to_config();

    // Calculate max zodiacal brightness at solar exclusion angle (closest to sun)
    let max_coords = SolarAngularCoordinates::new(args.solar_exclusion, 0.0)?;
    let zodiacal_light = ZodiacalLight::new();
    let max_brightness = zodiacal_light.get_brightness_mag_per_square_arcsec(&max_coords)?;

    // Calculate min zodiacal brightness at opposition (minimum zodiacal light position)
    let min_coords = SolarAngularCoordinates::zodiacal_minimum();
    let min_brightness = zodiacal_light.get_brightness_mag_per_square_arcsec(&min_coords)?;

    // Get zodiacal spectra for both positions
    let max_spectrum = zodiacal_light.get_zodiacal_spectrum(&max_coords)?;
    let min_spectrum = zodiacal_light.get_zodiacal_spectrum(&min_coords)?;

    // Calculate telescope and sensor properties
    let aperture_area_m2 =
        std::f64::consts::PI * (telescope_config.aperture.as_meters() / 2.0).powi(2);
    let aperture_area_cm2 = aperture_area_m2 * 10000.0; // Convert to cm²

    // Calculate pixel angular size in arcseconds
    let pixel_size_m = sensor_config.pixel_size().as_meters();
    let focal_length_m = telescope_config.focal_length.as_meters();
    let pixel_angular_size_rad = pixel_size_m / focal_length_m;
    let pixel_angular_size_arcsec =
        pixel_angular_size_rad * (180.0 / std::f64::consts::PI) * 3600.0;
    let pixel_area_arcsec2 = pixel_angular_size_arcsec.powi(2);

    // Create combined telescope + sensor quantum efficiency
    // Combine QE by creating product QE curve over overlapping wavelength range
    let tele_band = telescope_config.quantum_efficiency.band();
    let sensor_band = sensor_config.quantum_efficiency.band();

    // Find overlapping wavelength range
    let min_wavelength_nm = tele_band.lower_nm.max(sensor_band.lower_nm);
    let max_wavelength_nm = tele_band.upper_nm.min(sensor_band.upper_nm);

    // Sample wavelengths in overlapping range
    let num_samples = ((max_wavelength_nm - min_wavelength_nm) / 1.0) as usize + 1;
    let wavelengths: Vec<f64> = (0..num_samples)
        .map(|i| min_wavelength_nm + i as f64)
        .collect();

    // Calculate combined QE = telescope_qe * sensor_qe
    let mut combined_efficiencies: Vec<f64> = wavelengths
        .iter()
        .map(|&w| {
            let wl = simulator::units::Wavelength::from_nanometers(w);
            telescope_config.quantum_efficiency.at(wl) * sensor_config.quantum_efficiency.at(wl)
        })
        .collect();

    // Ensure boundaries are zero for proper QE curve
    if !combined_efficiencies.is_empty() {
        let last_idx = combined_efficiencies.len() - 1;
        combined_efficiencies[0] = 0.0;
        combined_efficiencies[last_idx] = 0.0;
    }

    let combined_qe = QuantumEfficiency::from_table(wavelengths, combined_efficiencies)?;
    let exposure = Duration::from_secs(1); // 1 second reference

    // Calculate photoelectrons per second per cm² using proper spectral integration
    let one_cm2 = simulator::units::Area::from_square_centimeters(1.0);
    let max_photoelectrons_per_s_per_cm2 =
        photoconversion::photo_electrons(&max_spectrum, &combined_qe, one_cm2, &exposure);
    let min_photoelectrons_per_s_per_cm2 =
        photoconversion::photo_electrons(&min_spectrum, &combined_qe, one_cm2, &exposure);

    // Scale by telescope aperture and pixel solid angle to get photoelectrons per pixel per second
    let max_photoelectrons_per_s_per_pixel =
        max_photoelectrons_per_s_per_cm2 * aperture_area_cm2 * pixel_area_arcsec2;
    let min_photoelectrons_per_s_per_pixel =
        min_photoelectrons_per_s_per_cm2 * aperture_area_cm2 * pixel_area_arcsec2;

    println!("Zodiacal Light Analysis:");
    println!("------------------------");
    println!(
        "Maximum brightness ({}°, 0°): {:.2} mag/arcsec²",
        args.solar_exclusion, max_brightness
    );
    println!("  → {max_photoelectrons_per_s_per_pixel:.2e} photoelectrons/s/pixel");
    println!();
    println!(
        "Minimum brightness ({ELONG_OF_MIN}°, {LAT_OF_MIN}°): {min_brightness:.2} mag/arcsec²"
    );
    println!("  → {min_photoelectrons_per_s_per_pixel:.2e} photoelectrons/s/pixel");
    println!();
    println!(
        "Brightness ratio (max/min): {:.1}x",
        10_f64.powf((min_brightness - max_brightness) / 2.5)
    );

    // Generate comparison plot if requested
    if args.plot {
        println!();
        println!("Generating dark current vs zodiacal comparison plot...");

        create_comparison_plot(
            telescope_config,
            sensor_config,
            max_photoelectrons_per_s_per_pixel,
            min_photoelectrons_per_s_per_pixel,
            args.solar_exclusion,
            &args.plot_output,
        )?;

        println!("Plot saved to: {}", args.plot_output);
    }

    Ok(())
}

fn create_comparison_plot(
    telescope_config: &simulator::hardware::TelescopeConfig,
    sensor_config: &simulator::hardware::SensorConfig,
    max_zodiacal_e_per_s: f64,
    min_zodiacal_e_per_s: f64,
    solar_exclusion: f64,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Temperature range from -20°C to +20°C
    let temperatures: Vec<f64> = (-20..=20).step_by(2).map(|t| t as f64).collect();

    // Calculate dark current rates for each temperature
    let mut dark_current_rates = Vec::new();
    let mut max_ratios = Vec::new();
    let mut min_ratios = Vec::new();

    for &temp_c in &temperatures {
        // Get dark current in electrons per second per pixel
        let temp = simulator::units::Temperature::from_celsius(temp_c);
        let dark_current_e_per_s = sensor_config
            .dark_current_estimator
            .estimate_at_temperature(temp)
            .unwrap_or(0.001);

        // Calculate ratios: dark_current / zodiacal (inverted)
        let max_ratio = dark_current_e_per_s / max_zodiacal_e_per_s;
        let min_ratio = dark_current_e_per_s / min_zodiacal_e_per_s;

        dark_current_rates.push(dark_current_e_per_s);
        max_ratios.push(max_ratio);
        min_ratios.push(min_ratio);
    }

    // Create the plot
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let title = format!(
        "Dark Current vs Zodiacal Light Ratio - {} + {} ({}° solar exclusion)",
        telescope_config.name, sensor_config.name, solar_exclusion
    );

    // Find plot bounds
    let min_temp = temperatures.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_temp = temperatures
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_ratio = max_ratios
        .iter()
        .chain(min_ratios.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_ratio = max_ratios
        .iter()
        .chain(min_ratios.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(
            min_temp..max_temp,
            ((min_ratio * 0.5).max(0.001)..(max_ratio * 2.0)).log_scale(),
        )?;

    chart
        .configure_mesh()
        .x_labels(11)
        .x_label_formatter(&|x| format!("{x:.0}°C"))
        .y_labels(8)
        .y_label_formatter(&|y| {
            if *y >= 1.0 {
                format!("{y:.1}")
            } else {
                format!("{y:.2}")
            }
        })
        .x_desc("Temperature (°C)")
        .y_desc("Dark Current / Zodiacal Light Ratio (log scale)")
        .axis_desc_style(("sans-serif", 18))
        .draw()?;

    // Add horizontal line at ratio = 1 (crossover point)
    chart.draw_series(std::iter::once(Rectangle::new(
        [(min_temp, 1.0), (max_temp, 1.0)],
        BLACK.stroke_width(2),
    )))?;

    // Plot max zodiacal ratio
    let max_data: Vec<(f64, f64)> = temperatures
        .iter()
        .zip(max_ratios.iter())
        .map(|(&t, &r)| (t, r))
        .collect();
    chart.draw_series(LineSeries::new(max_data, RED.stroke_width(3)))?;

    // Plot min zodiacal ratio
    let min_data: Vec<(f64, f64)> = temperatures
        .iter()
        .zip(min_ratios.iter())
        .map(|(&t, &r)| (t, r))
        .collect();
    chart.draw_series(LineSeries::new(min_data, BLUE.stroke_width(3)))?;

    // Add text annotations for key insights
    chart.draw_series(std::iter::once(Text::new(
        "Dark current dominates above line",
        (min_temp + 2.0, 1.2),
        ("sans-serif", 14).into_font(),
    )))?;

    chart.draw_series(std::iter::once(Text::new(
        "Zodiacal dominates below line",
        (max_temp - 15.0, 0.8),
        ("sans-serif", 14).into_font(),
    )))?;

    // Draw legend in top-left corner manually
    let legend_x = min_temp + 2.0;
    let legend_y_base = max_ratio * 0.7;

    // Draw legend background
    chart.draw_series(std::iter::once(Rectangle::new(
        [
            (legend_x - 1.0, legend_y_base * 1.5),
            (legend_x + 12.0, legend_y_base * 0.3),
        ],
        WHITE.mix(0.9).filled().stroke_width(1),
    )))?;

    // Draw legend lines and text
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(legend_x, legend_y_base), (legend_x + 3.0, legend_y_base)],
        RED.stroke_width(3),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        format!("vs Max Zodiacal ({solar_exclusion}° from Sun)"),
        (legend_x + 3.5, legend_y_base),
        ("sans-serif", 12).into_font(),
    )))?;

    chart.draw_series(std::iter::once(PathElement::new(
        vec![
            (legend_x, legend_y_base * 0.7),
            (legend_x + 3.0, legend_y_base * 0.7),
        ],
        BLUE.stroke_width(3),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        "vs Min Zodiacal (165°, 75°)",
        (legend_x + 3.5, legend_y_base * 0.7),
        ("sans-serif", 12).into_font(),
    )))?;

    root.present()?;
    Ok(())
}

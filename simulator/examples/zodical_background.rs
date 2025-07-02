//! Zodiacal background computation example
//!
//! This example computes the zodiacal light background in both DN/second
//! and electrons/second for different telescope/sensor combinations across
//! a matrix of solar elongation and ecliptic latitude coordinates.
//!
//! Usage:
//! ```
//! cargo run --example zodical_background -- [OPTIONS]
//! ```

use clap::Parser;
use ndarray::{Array1, Array2};
use plotters::prelude::*;
use simulator::algo::MinMaxScan;
use simulator::hardware::sensor::models::ALL_SENSORS;
use simulator::hardware::telescope::models::DEMO_50CM;
use simulator::hardware::SatelliteConfig;
use simulator::image_proc::render::quantize_image;
use simulator::photometry::{spectrum::Spectrum, zodical::SolarAngularCoordinates, ZodicalLight};
use simulator::shared_args::DurationArg;

/// Command line arguments for zodiacal background computation
#[derive(Parser, Debug)]
#[command(
    name = "Zodiacal Background Calculator",
    about = "Computes zodiacal light background DN/s and electrons/s across elongation/latitude matrix",
    long_about = None
)]
struct Args {
    /// Image domain size (pixels)
    #[arg(long, default_value_t = 256)]
    domain: usize,

    /// Elongation range: start,end,step (degrees)
    #[arg(long, default_value = "60.0,120.0,5.0")]
    elongation_range: String,

    /// Latitude range: start,end,step (degrees)
    #[arg(long, default_value = "0.0,30.0,5.0")]
    latitude_range: String,

    /// Exposure time for averaging (e.g., "1s", "500ms", "0.1s", "10m")
    #[arg(long, default_value = "1000s")]
    exposure: DurationArg,
}

/// Parse range string in format "start,end,step"
fn parse_range(range_str: &str) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let parts: Vec<&str> = range_str.split(',').collect();
    if parts.len() != 3 {
        return Err(format!(
            "Invalid range format. Expected 'start,end,step', got '{}'",
            range_str
        )
        .into());
    }

    let start = parts[0].trim().parse::<f64>()?;
    let end = parts[1].trim().parse::<f64>()?;
    let step = parts[2].trim().parse::<f64>()?;

    if step <= 0.0 {
        return Err("Step must be positive".into());
    }

    let num_points = ((end - start) / step + 1.0).round() as usize;
    let range = Array1::from_iter((0..num_points).map(|i| start + i as f64 * step));

    Ok(range)
}

/// Create PNG plot of zodiacal light spectrum using plotters
fn create_spectrum_plot(
    z_light: &ZodicalLight,
    coords: &SolarAngularCoordinates,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let spectrum = z_light.get_zodical_spectrum(coords)?;

    // Sample wavelengths from 100nm to 1100nm (full STIS spectrum range)
    let wavelengths: Vec<f64> = (100..=1100).step_by(5).map(|w| w as f64).collect();
    let mut irradiances = Vec::new();

    for &wavelength in &wavelengths {
        let irradiance = spectrum.spectral_irradiance(wavelength);
        // Convert from erg s⁻¹ cm⁻² Hz⁻¹ to erg s⁻¹ cm⁻² arcsec⁻² Hz⁻¹
        // Since we have spectrum per unit solid angle, multiply by 1 arcsec²
        let irradiance_per_arcsec2 = irradiance;
        irradiances.push(irradiance_per_arcsec2);
    }

    // Find data bounds for scaling
    let irradiance_scan = MinMaxScan::new(&irradiances);
    let (min_irradiance, max_irradiance) = irradiance_scan.min_max().unwrap_or((0.0, 1.0));
    let min_wavelength = *wavelengths.first().unwrap();
    let max_wavelength = *wavelengths.last().unwrap();

    // Create a new drawing area
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();

    // Fill the background with white
    root.fill(&WHITE)?;

    // Create chart with title including coordinates
    let title = format!(
        "Zodiacal Light Spectrum at {:.1}° elongation, {:.1}° latitude",
        coords.elongation(),
        coords.latitude()
    );

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(
            min_wavelength..max_wavelength,
            min_irradiance..max_irradiance,
        )?;

    // Configure the mesh
    chart
        .configure_mesh()
        .x_labels(11)
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_labels(6)
        .y_label_formatter(&|y| format!("{:.2e}", y))
        .x_desc("Wavelength (nm)")
        .y_desc("Spectral Irradiance (erg/(cm²·arcsec²·Hz))")
        .axis_desc_style(("sans-serif", 18))
        .draw()?;

    // Create data points for plotting
    let data_points: Vec<(f64, f64)> = wavelengths
        .iter()
        .zip(irradiances.iter())
        .filter_map(|(&wavelength, &irradiance)| {
            if irradiance > 0.0 && irradiance.is_finite() {
                Some((wavelength, irradiance))
            } else {
                None
            }
        })
        .collect();

    // Draw the spectrum curve
    chart
        .draw_series(LineSeries::new(data_points, &BLUE))?
        .label("Zodiacal Light Spectrum")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // Save the plot
    root.present()?;

    println!("Zodiacal spectrum plot saved to: {}", output_path);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging from environment variables
    env_logger::init();

    let args = Args::parse();

    // Parse ranges
    let elongations = parse_range(&args.elongation_range)?;
    let latitudes = parse_range(&args.latitude_range)?;

    // Define sensors to test
    let sensors: Vec<_> = ALL_SENSORS
        .iter()
        .map(|sensor| sensor.with_dimensions(args.domain, args.domain))
        .collect();

    let telescope = DEMO_50CM.clone();
    let exposure = args.exposure.0;

    println!("Zodiacal Background Analysis (DN/second & electrons/second)");
    println!("===========================================================");
    println!(
        "Telescope: {} (aperture: {:.2}m)",
        telescope.name, telescope.aperture_m
    );
    println!("Domain Size: {}x{} pixels", args.domain, args.domain);
    println!();

    let z_light = ZodicalLight::new();

    // Create plots directory if it doesn't exist
    std::fs::create_dir_all("plots")?;

    // Create spectrum plot for reference coordinates (120° elongation, 15° latitude)
    if let Ok(ref_coords) = SolarAngularCoordinates::new(120.0, 15.0) {
        let plot_path = "plots/zodiacal_spectrum.png";
        if let Err(e) = create_spectrum_plot(&z_light, &ref_coords, plot_path) {
            eprintln!("Warning: Failed to create spectrum plot: {}", e);
        }
    }

    for sensor in &sensors {
        println!("Sensor: {}", sensor.name);

        // Create result matrices for both DN and electrons
        let mut dn_matrix = Array2::<f64>::zeros((elongations.len(), latitudes.len()));
        let mut electron_matrix = Array2::<f64>::zeros((elongations.len(), latitudes.len()));

        // Create SatelliteConfig for zodiacal background calculation
        let satellite = SatelliteConfig::new(
            telescope.clone(),
            sensor.clone(),
            -10.0, // Default temperature for example
            550.0, // Default wavelength for example
        );

        // Compute DN/s and electrons/s for each coordinate combination
        for (elong_idx, &elongation) in elongations.iter().enumerate() {
            for (lat_idx, &latitude) in latitudes.iter().enumerate() {
                // Create coordinates
                let coords = SolarAngularCoordinates::new(elongation, latitude)
                    .expect("Invalid solar angular coordinates from parsed ranges");

                // Generate zodiacal background in electrons
                let zodical_e = z_light.generate_zodical_background(&satellite, &exposure, &coords);

                // Calculate mean electrons/s (divide by exposure time to get per second)
                let mean_electrons_total = zodical_e.mean().unwrap();
                let mean_electrons_per_s = mean_electrons_total / exposure.as_secs_f64();
                electron_matrix[[elong_idx, lat_idx]] = mean_electrons_per_s;

                // Convert to DN
                let zodical_dn = quantize_image(&zodical_e, &satellite.sensor);

                // Calculate mean DN/s (divide by exposure time to get per second)
                let mean_dn_total = zodical_dn.map(|&x| x as f64).mean().unwrap();
                let mean_dn_per_s = mean_dn_total / exposure.as_secs_f64();
                dn_matrix[[elong_idx, lat_idx]] = mean_dn_per_s;
            }
        }

        // Print DN/s matrix with headers
        println!("  DN/s Table:");
        print!("  Elong\\Lat |");
        for &lat in &latitudes {
            print!(" {:8.1}° |", lat);
        }
        println!();

        print!("  ----------|");
        for _ in &latitudes {
            print!("----------|");
        }
        println!();

        for (elong_idx, &elong) in elongations.iter().enumerate() {
            print!("  {:6.1}°   |", elong);
            for lat_idx in 0..latitudes.len() {
                let dn_per_s = dn_matrix[[elong_idx, lat_idx]];
                if dn_per_s.is_nan() {
                    print!("   ---    |");
                } else {
                    print!(" {:8.7} |", dn_per_s);
                }
            }
            println!();
        }
        println!();

        // Print electrons/s matrix with headers
        println!("  Electrons/s Table:");
        print!("  Elong\\Lat |");
        for &lat in &latitudes {
            print!(" {:8.1}° |", lat);
        }
        println!();

        print!("  ----------|");
        for _ in &latitudes {
            print!("----------|");
        }
        println!();

        for (elong_idx, &elong) in elongations.iter().enumerate() {
            print!("  {:6.1}°   |", elong);
            for lat_idx in 0..latitudes.len() {
                let electrons_per_s = electron_matrix[[elong_idx, lat_idx]];
                if electrons_per_s.is_nan() {
                    print!("   ---    |");
                } else {
                    print!(" {:8.2e} |", electrons_per_s);
                }
            }
            println!();
        }
        println!();
    }

    Ok(())
}

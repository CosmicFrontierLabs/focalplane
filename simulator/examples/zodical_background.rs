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
use simulator::hardware::SatelliteConfig;
use simulator::image_proc::render::quantize_image;
use simulator::photometry::{spectrum::Spectrum, zodical::SolarAngularCoordinates, ZodicalLight};
use simulator::shared_args::{DurationArg, TelescopeModel};
use simulator::units::{LengthExt, Temperature, TemperatureExt, Wavelength};

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
    #[arg(long, default_value = "0.0,180.0,10.0")]
    elongation_range: String,

    /// Latitude range: start,end,step (degrees)
    #[arg(long, default_value = "0.0,90.0,10.0")]
    latitude_range: String,

    /// Exposure time for averaging (e.g., "1s", "500ms", "0.1s", "10m")
    #[arg(long, default_value = "1000s")]
    exposure: DurationArg,

    /// Telescope model to use for simulations
    #[arg(long, default_value_t = TelescopeModel::Demo50cm)]
    telescope: TelescopeModel,
}

/// Parses a range specification string into an array of evenly spaced values.
///
/// This function takes a comma-separated string specification and generates
/// a 1D array of floating-point values with the specified start, end, and step.
///
/// # Format
///
/// The input string must follow the exact format: `"start,end,step"`
/// where all three components are valid floating-point numbers.
///
/// # Arguments
///
/// * `range_str` - Range specification in format "start,end,step" (e.g., "0.0,10.0,0.5")
///
/// # Returns
///
/// Returns `Ok(Array1<f64>)` containing the generated range values, or an error if:
/// - The format is invalid (not exactly 3 comma-separated values)
/// - Any value cannot be parsed as f64
/// - The step is zero or negative
///
/// # Examples
///
/// ```rust
/// let angles = parse_range("0.0,90.0,15.0")?; // [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
/// let coords = parse_range("-10.0,10.0,5.0")?; // [-10.0, -5.0, 0.0, 5.0, 10.0]
/// ```
///
/// # Error Cases
///
/// ```rust
/// parse_range("1,2");        // Error: wrong number of components
/// parse_range("1,2,0");      // Error: step must be positive
/// parse_range("a,b,c");      // Error: invalid number format
/// ```
fn parse_range(range_str: &str) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let parts: Vec<&str> = range_str.split(',').collect();
    if parts.len() != 3 {
        return Err(
            format!("Invalid range format. Expected 'start,end,step', got '{range_str}'").into(),
        );
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

/// Creates a detailed PNG plot of the zodiacal light spectrum at specified coordinates.
///
/// This function generates a publication-quality plot showing the spectral irradiance
/// of zodiacal light as a function of wavelength. The plot includes proper axis labels,
/// title with coordinate information, and scientific notation for irradiance values.
///
/// # Spectrum Coverage
///
/// The plot covers wavelengths from 100nm to 1100nm with 5nm sampling, spanning:
/// - UV range: 100-400nm (important for young stars)
/// - Visible range: 400-700nm (human-visible spectrum)
/// - Near-IR range: 700-1100nm (stellar photosphere peaks)
///
/// # Arguments
///
/// * `z_light` - ZodicalLight instance for spectrum calculation
/// * `coords` - Solar angular coordinates (elongation and ecliptic latitude)
/// * `output_path` - File path for the output PNG image
///
/// # Returns
///
/// Returns `Ok(())` on successful plot creation, or an error if:
/// - Spectrum calculation fails for the given coordinates
/// - File I/O fails during image saving
/// - Plotting library encounters rendering errors
///
/// # Plot Features
///
/// - **Title**: Shows elongation and latitude coordinates
/// - **X-axis**: Wavelength in nanometers (100-1100nm)
/// - **Y-axis**: Spectral irradiance in erg/(cm²·arcsec²·Hz) with scientific notation
/// - **Legend**: Identifies the zodiacal light spectrum curve
/// - **Resolution**: 1024×768 pixels for clear display
///
/// # Example Usage
///
/// ```rust
/// let coords = SolarAngularCoordinates::zodiacal_minimum();
/// let z_light = ZodicalLight::new();
/// create_spectrum_plot(&z_light, &coords, "spectrum.png")?;
/// ```
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
        let irradiance = spectrum.spectral_irradiance(Wavelength::from_nanometers(wavelength));
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
        .x_label_formatter(&|x| format!("{x:.0}"))
        .y_labels(6)
        .y_label_formatter(&|y| format!("{y:.2e}"))
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

    println!("Zodiacal spectrum plot saved to: {output_path}");

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

    let telescope = args.telescope.to_config().clone();
    let exposure = args.exposure.0;

    println!("Zodiacal Background Analysis (DN/second & electrons/second)");
    println!("===========================================================");
    println!();
    println!("SETTINGS:");
    println!("---------");
    println!("Telescope Configuration:");
    println!("  Model Selected: {}", args.telescope);
    println!("  Name: {}", telescope.name);
    println!("  Aperture: {:.2} m", telescope.aperture.as_meters());
    println!(
        "  Focal Length: {:.2} m",
        telescope.focal_length.as_meters()
    );
    println!("  F-number: f/{:.1}", telescope.f_number());
    println!(
        "  Plate Scale: {:.2} arcsec/mm",
        telescope.plate_scale_arcsec_per_mm()
    );
    println!();
    println!("Analysis Parameters:");
    println!("  Domain Size: {}x{} pixels", args.domain, args.domain);
    println!(
        "  Exposure Time: {} ({:.2} seconds)",
        args.exposure,
        exposure.as_secs_f64()
    );
    println!(
        "  Elongation Range: {} ({} values)",
        args.elongation_range,
        elongations.len()
    );
    println!(
        "  Latitude Range: {} ({} values)",
        args.latitude_range,
        latitudes.len()
    );
    println!(
        "  Total Grid Points: {}",
        elongations.len() * latitudes.len()
    );
    println!();
    println!("Sensors to Analyze:");
    for sensor in &sensors {
        println!(
            "  - {} ({:.1}μm pitch, {}x{} pixels)",
            sensor.name,
            sensor.pixel_size.as_micrometers(),
            sensor.width_px,
            sensor.height_px
        );
    }
    println!();
    println!("Output Information:");
    println!("  Spectrum Plot: plots/zodiacal_spectrum.png");
    println!("  Reference Coordinates: 120° elongation, 15° latitude");
    println!("  Units: DN/second and electrons/second");
    println!();
    println!("===========================================================");
    println!();

    // Output zodiacal minimum coordinates
    let min_coords = SolarAngularCoordinates::zodiacal_minimum();
    println!("Zodiacal Light Minimum Coordinates:");
    println!("  Elongation: {:.1}°", min_coords.elongation());
    println!("  Latitude: {:.1}°", min_coords.latitude());
    println!();

    let z_light = ZodicalLight::new();

    // Create plots directory if it doesn't exist
    std::fs::create_dir_all("plots")?;

    // Create spectrum plot for reference coordinates (120° elongation, 15° latitude)
    if let Ok(ref_coords) = SolarAngularCoordinates::new(120.0, 15.0) {
        let plot_path = "plots/zodiacal_spectrum.png";
        if let Err(e) = create_spectrum_plot(&z_light, &ref_coords, plot_path) {
            eprintln!("Warning: Failed to create spectrum plot: {e}");
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
            Temperature::from_celsius(-10.0), // Default temperature for example
            Wavelength::from_nanometers(550.0), // Default wavelength for example
        );

        // Calculate zodical minimum e-/s for this sensor
        let min_zodical_e = z_light.generate_zodical_background(&satellite, &exposure, &min_coords);
        let min_electrons_total = min_zodical_e.mean().unwrap();
        let min_electrons_per_s = min_electrons_total / exposure.as_secs_f64();
        println!("  Zodical minimum: {:.2e} e-/s", min_electrons_per_s);

        // Compute DN/s and electrons/s for each coordinate combination
        for (elong_idx, &elongation) in elongations.iter().enumerate() {
            for (lat_idx, &latitude) in latitudes.iter().enumerate() {
                // Create coordinates
                let coords = SolarAngularCoordinates::new(elongation, latitude)
                    .expect("Invalid solar angular coordinates from parsed ranges");

                // Try to generate zodiacal background, handle interpolation errors
                match z_light.get_zodical_spectrum(&coords) {
                    Ok(_) => {
                        // Generate zodiacal background in electrons
                        let zodical_e =
                            z_light.generate_zodical_background(&satellite, &exposure, &coords);

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
                    Err(_) => {
                        // Set NaN for invalid interpolation points
                        electron_matrix[[elong_idx, lat_idx]] = f64::NAN;
                        dn_matrix[[elong_idx, lat_idx]] = f64::NAN;
                    }
                }
            }
        }

        // Print DN/s matrix with headers
        println!("  DN/s Table:");
        print!("  Elong\\Lat |");
        for &lat in &latitudes {
            print!(" {lat:8.1}° |");
        }
        println!();

        print!("  ----------|");
        for _ in &latitudes {
            print!("----------|");
        }
        println!();

        for (elong_idx, &elong) in elongations.iter().enumerate() {
            print!("  {elong:6.1}°   |");
            for lat_idx in 0..latitudes.len() {
                let dn_per_s = dn_matrix[[elong_idx, lat_idx]];
                if dn_per_s.is_nan() {
                    print!("     -    |");
                } else {
                    print!(" {dn_per_s:8.7} |");
                }
            }
            println!();
        }
        println!();

        // Print electrons/s matrix with headers
        println!("  Electrons/s Table:");
        print!("  Elong\\Lat |");
        for &lat in &latitudes {
            print!(" {lat:8.1}° |");
        }
        println!();

        print!("  ----------|");
        for _ in &latitudes {
            print!("----------|");
        }
        println!();

        for (elong_idx, &elong) in elongations.iter().enumerate() {
            print!("  {elong:6.1}°   |");
            for lat_idx in 0..latitudes.len() {
                let electrons_per_s = electron_matrix[[elong_idx, lat_idx]];
                if electrons_per_s.is_nan() {
                    print!("     -    |");
                } else {
                    print!(" {electrons_per_s:8.2e} |");
                }
            }
            println!();
        }
        println!();
    }

    Ok(())
}

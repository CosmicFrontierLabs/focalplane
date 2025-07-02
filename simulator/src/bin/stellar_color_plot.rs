//! Plot stellar spectra with human-perceived colors
//!
//! This tool generates plots of stellar spectra at different temperatures,
//! using human vision models to determine the perceived color of each spectrum.
//!
//! Usage:
//! ```
//! cargo run --bin stellar_color_plot -- [OPTIONS]
//! ```
//!
//! See --help for detailed options.

use clap::Parser;
use plotters::prelude::*;
use rayon::prelude::*;
use simulator::algo::MinMaxScan;
use simulator::photometry::color::SpectralClass;
use simulator::photometry::spectrum::Spectrum;
use simulator::photometry::stellar::BlackbodyStellarSpectrum;
use simulator::photometry::{
    generate_temperature_sequence, spectrum_to_rgb_values, temperature_to_spectral_class,
};
use std::error::Error;

/// Command line arguments for stellar spectrum plotting
#[derive(Parser, Debug)]
#[command(
    name = "Stellar Color Plotter",
    about = "Plots stellar spectra at different temperatures with human-perceived colors",
    long_about = None
)]
struct Args {
    /// Number of spectra to plot
    #[arg(short, long, default_value_t = 5)]
    n_spectra: usize,

    /// Minimum stellar temperature in Kelvin
    #[arg(long, default_value_t = 4000.0)]
    min_temp: f64,

    /// Maximum stellar temperature in Kelvin
    #[arg(long, default_value_t = 10000.0)]
    max_temp: f64,

    /// Output file path
    #[arg(short, long, default_value = "plots/stellar_spectrum.png")]
    output: String,

    /// Wavelength range minimum (nm)
    #[arg(long, default_value_t = 200.0)]
    wavelength_min: f64,

    /// Wavelength range maximum (nm)
    #[arg(long, default_value_t = 2000.0)]
    wavelength_max: f64,

    /// Number of sample points for spectra
    #[arg(long, default_value_t = 100)]
    sample_points: usize,
}

/// Information about a stellar spectrum and its properties
struct StellarInfo {
    /// Temperature in Kelvin
    temperature: f64,
    /// Spectral class (O, B, A, F, G, K, M)
    spectral_class: SpectralClass,
    /// Blackbody stellar spectrum model
    spectrum: BlackbodyStellarSpectrum,
    /// RGB color based on human vision
    color: RGBColor,
}

/// Convert RGB floating point values to RGBColor for plotting
///
/// # Arguments
///
/// * `r` - Red value (0.0-1.0)
/// * `g` - Green value (0.0-1.0)
/// * `b` - Blue value (0.0-1.0)
///
/// # Returns
///
/// RGBColor value for plotters
fn rgb_values_to_color(r: f64, g: f64, b: f64) -> RGBColor {
    RGBColor(
        (r * 255.0).min(255.0) as u8,
        (g * 255.0).min(255.0) as u8,
        (b * 255.0).min(255.0) as u8,
    )
}

/// Create stellar spectra for plotting
///
/// # Arguments
///
/// * `args` - Command-line arguments including temperature range and count
///
/// # Returns
///
/// Vector of StellarInfo containing spectrum and color information
fn create_stellar_spectra(args: &Args) -> Vec<StellarInfo> {
    // Generate logarithmically spaced temperatures
    let temperatures = generate_temperature_sequence(args.min_temp, args.max_temp, args.n_spectra);

    // Create spectra with calculated colors
    temperatures
        .into_par_iter()
        .map(|temp| {
            // Create blackbody spectrum for this temperature
            let spectrum = BlackbodyStellarSpectrum::new(temp, 1.0);

            // Calculate visual color based on human vision
            let (r, g, b) = spectrum_to_rgb_values(&spectrum);
            let color = rgb_values_to_color(r, g, b);

            // Determine spectral class from temperature
            let spectral_class = temperature_to_spectral_class(temp);

            StellarInfo {
                temperature: temp,
                spectral_class,
                spectrum,
                color,
            }
        })
        .collect()
}

/// Main function to generate the plot
fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging from environment variables
    env_logger::init();

    // Parse command line arguments
    let args = Args::parse();

    println!(
        "Generating stellar spectra plot with temperatures from {}K to {}K...",
        args.min_temp, args.max_temp
    );

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("plots")?;

    // Generate the plot
    generate_stellar_plot(&args)?;

    println!("Plot saved to: {}", args.output);
    Ok(())
}

/// Generate the stellar spectra plot
fn generate_stellar_plot(args: &Args) -> Result<(), Box<dyn Error>> {
    // Generate spectral info including human-perceived colors
    let stellar_info = create_stellar_spectra(args);

    // Create drawing area for the plot
    let root = BitMapBackend::new(&args.output, (1024, 768)).into_drawing_area();

    // Fill the background with black
    root.fill(&BLACK)?;

    // Create chart with specified wavelength range
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Stellar Spectra by Temperature",
            ("sans-serif", 30).into_font().color(&WHITE),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(args.wavelength_min..args.wavelength_max, 0.0f64..1.0f64)?;

    // Configure grid and labels
    // Calculate grid spacing based on input parameters - much sparser
    let x_grid_spacing = 500.0; // nm - much wider spacing
    let y_grid_spacing = 0.25; // normalized irradiance - 4 grid lines total

    // Use explicit function to resolve ambiguous float type
    fn ceil_div(a: f64, b: f64) -> usize {
        (a / b).ceil() as usize
    }

    let x_label_count = ceil_div(args.wavelength_max - args.wavelength_min, x_grid_spacing) + 1;
    let y_label_count = ceil_div(1.0, y_grid_spacing) + 1;

    chart
        .configure_mesh()
        .x_labels(x_label_count) // Compute from wavelength range and desired spacing
        .y_labels(y_label_count) // Will show 0.0, 0.1, 0.2, ... 1.0
        .x_label_formatter(&|x| format!("{}", *x as i32)) // Integer labels
        .y_label_formatter(&|y| format!("{:.1}", y)) // One decimal point for y-axis labels
        .axis_desc_style(("sans-serif", 18).into_font().color(&WHITE))
        .label_style(("sans-serif", 14).into_font().color(&WHITE))
        .x_desc("Wavelength (nm)")
        .y_desc("Normalized Spectral Irradiance")
        .light_line_style(WHITE.mix(0.4)) // More visible grey grid lines
        .draw()?;

    // Sample wavelengths for the plot
    let wavelengths: Vec<f64> = (0..args.sample_points)
        .map(|i| {
            args.wavelength_min
                + (args.wavelength_max - args.wavelength_min) * (i as f64)
                    / (args.sample_points as f64 - 1.0)
        })
        .collect();

    // Get the maximum irradiance value across all spectra for normalization
    let all_irradiances: Vec<f64> = stellar_info
        .iter()
        .flat_map(|info| {
            wavelengths
                .iter()
                .map(|&wl| info.spectrum.spectral_irradiance(wl))
        })
        .collect();
    let irr_scan = MinMaxScan::new(&all_irradiances);
    let max_irr = irr_scan.max().unwrap_or(1.0);

    // Draw each spectrum
    for star in &stellar_info {
        // Generate normalized data points
        let data_points: Vec<(f64, f64)> = wavelengths
            .iter()
            .map(|&wavelength| {
                let irradiance = star.spectrum.spectral_irradiance(wavelength);
                (wavelength, irradiance / max_irr)
            })
            .collect();

        // Create label with temperature and spectral class
        let label = format!(
            "{}K (Class {})",
            star.temperature as u32, star.spectral_class
        );

        // Draw the spectrum with its perceived color
        chart
            .draw_series(LineSeries::new(data_points, &star.color))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], star.color));
    }

    // Draw the legend with black background for visibility
    chart
        .configure_series_labels()
        .background_style(BLACK.mix(0.7))
        .border_style(WHITE)
        .label_font(("sans-serif", 14).into_font().color(&WHITE))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    Ok(())
}

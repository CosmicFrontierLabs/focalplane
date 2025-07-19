//! Demonstration of chromatic Airy disk effects in telescope PSF modeling.
//!
//! This example shows how wavelength-dependent diffraction affects the
//! point spread function of a telescope, comparing monochromatic and
//! chromatic PSF models for different stellar spectra.

use clap::Parser;
use ndarray::Array2;
use plotters::prelude::*;
use simulator::algo::misc::normalize;
use simulator::hardware::sensor::models::{HWK4123, IMX455};
use simulator::image_proc::airy::PixelScaledAiryDisk;
use simulator::photometry::{
    color::{rgb_values_to_color, spectrum_to_rgb_values},
    photon_electron_fluxes,
    stellar::{temperature_from_bv, BlackbodyStellarSpectrum},
    QuantumEfficiency,
};
use simulator::star_math::DEFAULT_BV;
use std::error::Error;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "chromatic_airy_demo")]
#[command(about = "Demonstration of chromatic Airy disk effects")]
struct Args {}

/// Analyzes chromatic PSF broadening for different stellar spectra on a specific sensor.
///
/// This function computes the effective PSF size when imaging various stellar types
/// through a telescope+sensor system, accounting for chromatic effects where the PSF
/// size varies with wavelength and the sensor's quantum efficiency curve.
///
/// # Arguments
///
/// * `sensor_name` - Name of the sensor for display purposes
/// * `detector_qe` - Quantum efficiency curve of the detector
/// * `airy_disk` - Reference PSF model with nominal FWHM and wavelength
/// * `stars` - Array of stellar spectra with their type labels and temperatures
/// * `aperture_cm2` - Telescope aperture area in cm²
/// * `integration_time` - Integration time for the observation
///
/// # Returns
///
/// Vector of tuples containing:
/// - Star type label (String)
/// - Temperature in Kelvin (f64)
/// - Effective PSF FWHM in pixels (f64)
/// - Percent broadening relative to monochromatic PSF (f64)
/// - Effective wavelength in nm (f64)
///
/// # Example Output
///
/// For a cool star on an IR-sensitive detector, the function might return:
/// ```
/// ("Cool star", 3500.0, 3.948, 22.1, 724.0)
/// ```
/// This indicates the PSF is 22.1% broader than monochromatic, with an effective
/// wavelength of 724nm due to the star's red spectrum and detector sensitivity.
fn analyze_sensor_psf(
    sensor_name: &str,
    detector_qe: &QuantumEfficiency,
    airy_disk: &PixelScaledAiryDisk,
    stars: &[(&BlackbodyStellarSpectrum, &str, f64)],
) -> Vec<(String, f64, f64, f64, f64)> {
    println!("\n{sensor_name} Sensor Analysis");
    println!("{}", "=".repeat(sensor_name.len() + 16));
    println!(
        "\nNominal PSF wavelength: {:.0} nm",
        airy_disk.reference_wavelength
    );
    println!(
        "Monochromatic FWHM at {:.0}nm: {:.3}",
        airy_disk.reference_wavelength,
        airy_disk.fwhm()
    );

    let mut results = Vec::new();

    for (star, star_type, temp) in stars {
        let pe = photon_electron_fluxes(airy_disk, *star, detector_qe).electrons;

        let broadening = (pe.disk.fwhm() / airy_disk.fwhm() - 1.0) * 100.0;
        let effective_wavelength = pe.disk.reference_wavelength;

        results.push((
            star_type.to_string(),
            *temp,
            pe.disk.fwhm(),
            broadening,
            effective_wavelength,
        ));
    }

    // Print table
    println!(
        "\n{:<20} {:>8} {:>10} {:>12} {:>12}",
        "Star Type", "Temp (K)", "FWHM", "Broadening %", "Eff. λ (nm)"
    );
    println!("{}", "-".repeat(65));

    for (star_type, temp, fwhm, broadening, eff_lambda) in &results {
        println!("{star_type:<20} {temp:>8.0} {fwhm:>10.3} {broadening:>12.1} {eff_lambda:>12.0}");
    }

    results
}

/// Creates radial profile plots comparing chromatic and monochromatic PSFs for a sensor.
///
/// This function generates high-resolution plots showing how different stellar spectra
/// produce different PSF sizes due to chromatic aberration effects. The plots include:
/// - Monochromatic reference PSF at 550nm
/// - Monochromatic Gaussian approximation
/// - Chromatic PSFs for various stellar types
///
/// # Arguments
///
/// * `sensor_name` - Name of the sensor for plot title and filename
/// * `sensor_qe` - Quantum efficiency curve of the detector
/// * `airy_disk` - Reference PSF model
/// * `stars` - Array of stellar spectra to plot
/// * `aperture_cm2` - Telescope aperture area in cm²
/// * `integration_time` - Integration time
/// * `max_radius` - Maximum radius for plot in normalized units
/// * `n_points` - Number of points for radial profile
///
/// # Returns
///
/// Result indicating success or plotting error
fn create_sensor_plot(
    sensor_name: &str,
    sensor_qe: &QuantumEfficiency,
    airy_disk: &PixelScaledAiryDisk,
    stars: &[(&BlackbodyStellarSpectrum, &str, f64)],
    aperture_cm2: f64,
    integration_time: &Duration,
    max_radius: f64,
    n_points: usize,
) -> Result<(), Box<dyn Error>> {
    // Generate radius values
    let radii: Vec<f64> = (0..n_points)
        .map(|i| i as f64 * max_radius / (n_points - 1) as f64)
        .collect();

    // Generate monochromatic profiles from the airy disk
    let mono_profile: Vec<_> = radii.iter().map(|&r| airy_disk.intensity(r)).collect();
    let mono_gauss_profile: Vec<_> = radii
        .iter()
        .map(|&r| airy_disk.gaussian_approximation(r))
        .collect();

    // Recompute PSFs for this sensor
    let mut star_psfs = Vec::new();
    for (star, _, _) in stars {
        let pe = photon_electron_fluxes(airy_disk, *star, sensor_qe).electrons;
        star_psfs.push(pe);
    }

    // Generate radial profiles
    let mut star_profiles = vec![Vec::with_capacity(n_points); stars.len()];

    for &r in radii.iter() {
        // Star profiles
        for (j, psf) in star_psfs.iter().enumerate() {
            star_profiles[j].push(
                psf.disk.gaussian_approximation(r)
                    * psf.integrated_over(integration_time, aperture_cm2),
            );
        }
    }

    // Normalize profiles
    let star_profiles: Vec<_> = star_profiles.into_iter().map(normalize).collect();

    // Get stellar colors
    let star_colors: Vec<_> = stars
        .iter()
        .map(|(star, _, _)| {
            let (r, g, b) = spectrum_to_rgb_values(*star);
            rgb_values_to_color(r, g, b)
        })
        .collect();

    let mono_color = RGBColor(0, 255, 0); // Laser green for monochromatic

    // Create plot
    let filename = format!("plots/chromatic_airy_{}.png", sensor_name.to_lowercase());
    let root = BitMapBackend::new(&filename, (2048, 1536)).into_drawing_area();
    root.fill(&BLACK)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{sensor_name}: Chromatic vs Monochromatic Airy Disk Profiles"),
            ("sans-serif", 50).into_font().color(&WHITE),
        )
        .margin(40)
        .x_label_area_size(80)
        .y_label_area_size(100)
        .build_cartesian_2d(0.0..max_radius, 0.0001f64..1.1f64)?;

    chart
        .configure_mesh()
        .x_desc("Radius (normalized units)")
        .y_desc("Normalized Intensity")
        .y_label_formatter(&|y| {
            if *y >= 0.1 {
                format!("{y:.1}")
            } else {
                format!("{y:.0e}")
            }
        })
        .y_labels(11)
        .label_style(("sans-serif", 24).into_font().color(&WHITE))
        .axis_style(WHITE)
        .draw()?;

    // Plot monochromatic profiles
    chart
        .draw_series(LineSeries::new(
            radii.iter().cloned().zip(mono_profile.iter().cloned()),
            ShapeStyle::from(&mono_color).stroke_width(2),
        ))?
        .label("Monochromatic (550nm)")
        .legend(move |(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&mono_color).stroke_width(2),
            )
        });

    chart
        .draw_series(
            LineSeries::new(
                radii
                    .iter()
                    .cloned()
                    .zip(mono_gauss_profile.iter().cloned()),
                ShapeStyle::from(&mono_color).stroke_width(1),
            )
            .point_size(0),
        )?
        .label("Mono Gaussian approx")
        .legend(move |(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&mono_color).stroke_width(1),
            )
        });

    // Plot star profiles
    for (i, ((_, star_type, temp), profile)) in stars.iter().zip(star_profiles.iter()).enumerate() {
        let color = &star_colors[i];
        let eff_wavelength = star_psfs[i].disk.reference_wavelength;
        let label = if star_type.contains("B-V") {
            format!("{star_type} ({temp:.0}K, {eff_wavelength:.0}nm)")
        } else {
            format!("{star_type} ({temp:.0}K, {eff_wavelength:.0}nm)")
        };

        chart
            .draw_series(LineSeries::new(
                radii.iter().cloned().zip(profile.iter().cloned()),
                ShapeStyle::from(color).stroke_width(1),
            ))?
            .label(&label)
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(color).stroke_width(1),
                )
            });
    }

    chart
        .configure_series_labels()
        .background_style(BLACK.mix(0.8))
        .border_style(WHITE)
        .label_font(("sans-serif", 20).into_font().color(&WHITE))
        .draw()?;

    root.present()?;
    Ok(())
}

/// Creates 2D PSF comparison images showing monochromatic vs chromatic PSFs.
///
/// This function generates side-by-side 2D intensity maps comparing:
/// - Left: Monochromatic PSF at reference wavelength
/// - Right: Chromatic PSF for a sun-like star
///
/// # Arguments
///
/// * `airy_disk` - Reference PSF model
/// * `sun_like` - Sun-like stellar spectrum for chromatic PSF
/// * `sensor_qe` - Quantum efficiency of the first sensor
/// * `sensor_name` - Name of the sensor for display
/// * `aperture_cm2` - Telescope aperture area in cm²
/// * `integration_time` - Integration time
/// * `max_radius` - Maximum radius in normalized units (should match radial plots)
///
/// # Returns
///
/// Result indicating success or plotting error
fn create_2d_psf_comparison(
    airy_disk: &PixelScaledAiryDisk,
    sun_like: &BlackbodyStellarSpectrum,
    sensor_qe: &QuantumEfficiency,
    sensor_name: &str,
    aperture_cm2: f64,
    integration_time: &Duration,
    max_radius: f64,
) -> Result<(), Box<dyn Error>> {
    let image_size = 64;
    // Calculate pixel scale to match max_radius
    let pixel_scale = (2.0 * max_radius) / image_size as f64; // Full diameter / pixels

    // Function to create 2D PSF image
    let create_psf_image = |intensity_fn: &dyn Fn(f64) -> f64| -> Array2<f64> {
        let mut image = Array2::zeros((image_size, image_size));
        let center = image_size as f64 / 2.0;

        for i in 0..image_size {
            for j in 0..image_size {
                let dx = (j as f64 - center + 0.5) * pixel_scale;
                let dy = (i as f64 - center + 0.5) * pixel_scale;
                let r = (dx * dx + dy * dy).sqrt();
                image[[i, j]] = intensity_fn(r);
            }
        }

        image
    };

    // Create monochromatic PSF image
    let mono_image = create_psf_image(&|r| airy_disk.intensity(r));

    // Create chromatic PSF image for sun-like star
    let sun_pe = photon_electron_fluxes(airy_disk, sun_like, sensor_qe).electrons;
    let sun_image = create_psf_image(&|r| {
        sun_pe.disk.intensity(r) * sun_pe.integrated_over(integration_time, aperture_cm2)
    });

    // Log scale for visualization
    let log_scale = |x: f64| -> f64 {
        if x > 0.0 {
            (x.log10() + 4.0) / 4.0 // Map 1e-4 to 1 -> 0 to 1
        } else {
            0.0
        }
    };

    // Save 2D PSF comparison
    let root2d = BitMapBackend::new("plots/chromatic_psf_2d.png", (768, 384)).into_drawing_area();
    root2d.fill(&WHITE)?;

    let (left, right) = root2d.split_horizontally(384);

    // Plot monochromatic PSF
    let mut mono_chart = ChartBuilder::on(&left)
        .caption("Monochromatic PSF", ("sans-serif", 20))
        .margin(10)
        .build_cartesian_2d(0..image_size, 0..image_size)?;

    mono_chart.draw_series(mono_image.indexed_iter().map(|((y, x), &v)| {
        Rectangle::new(
            [(x, y), (x + 1, y + 1)],
            HSLColor(0.0, 0.0, 1.0 - log_scale(v)).filled(),
        )
    }))?;

    // Plot chromatic PSF
    let mut chrom_chart = ChartBuilder::on(&right)
        .caption(
            format!("Chromatic PSF (Sun-like star, {sensor_name})"),
            ("sans-serif", 20),
        )
        .margin(10)
        .build_cartesian_2d(0..image_size, 0..image_size)?;

    chrom_chart.draw_series(sun_image.indexed_iter().map(|((y, x), &v)| {
        Rectangle::new(
            [(x, y), (x + 1, y + 1)],
            HSLColor(0.0, 0.0, 1.0 - log_scale(v)).filled(),
        )
    }))?;

    root2d.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let _args = Args::parse();

    println!("Chromatic Airy Disk Demonstration");
    println!("=================================");

    // Telescope parameters - 50cm diameter telescope
    let aperture = 0.5; // 0.5 meter diameter (50cm)
    let reference_wavelength = 550.0; // 550nm (green)

    // Calculate aperture area in cm² for photon calculations
    let aperture_cm2 = std::f64::consts::PI * (aperture * 100.0 / 2.0_f64).powi(2); // π × (25cm)²
    let integration_time = Duration::from_secs(1); // 1 second integration

    // Create PSF model
    let airy_disk = PixelScaledAiryDisk::with_fwhm(1.0, reference_wavelength); // Default PSF

    // Use real sensor QE curves
    let sensors = vec![
        ("HWK4123", &HWK4123.quantum_efficiency),
        ("IMX455", &IMX455.quantum_efficiency),
    ];

    // Different stellar temperatures with same flux scaling
    let hot_star = BlackbodyStellarSpectrum::new(10000.0, 1e-10); // Hot blue star
    let sun_like = BlackbodyStellarSpectrum::new(5780.0, 1e-10); // Sun-like star
    let cool_star = BlackbodyStellarSpectrum::new(3500.0, 1e-10); // Cool red star

    // Default catalog star with DEFAULT_BV
    let default_temp = temperature_from_bv(DEFAULT_BV);
    let default_star = BlackbodyStellarSpectrum::new(default_temp, 1e-10);
    let default_label = format!("B-V={DEFAULT_BV:.1}");

    // Prepare star data - need references to avoid moves
    let stars = vec![
        (&hot_star, "Hot star", 10000.0),
        (&sun_like, "Sun-like", 5780.0),
        (&default_star, default_label.as_str(), default_temp),
        (&cool_star, "Cool star", 3500.0),
    ];

    // Analyze all sensors
    let mut all_results = Vec::new();
    for (sensor_name, sensor_qe) in &sensors {
        let results = analyze_sensor_psf(sensor_name, sensor_qe, &airy_disk, &stars);
        all_results.push((sensor_name, results));
    }

    // Print comparison table
    println!("\n\nPSF Photoelectron Broadening Comparison");
    println!("=======================================");
    if all_results.len() == 2 {
        println!(
            "\n{:<20} {:>15} {:>15} {:>15}",
            "Star Type", "HWK4123 FWHM", "IMX455 FWHM", "Difference"
        );
        println!("{}", "-".repeat(70));

        for i in 0..stars.len() {
            let hwk_fwhm = all_results[0].1[i].2;
            let imx_fwhm = all_results[1].1[i].2;
            let diff = (hwk_fwhm - imx_fwhm).abs();
            println!(
                "{:<20} {:>15.3} {:>15.3} {:>15.3}",
                stars[i].1, hwk_fwhm, imx_fwhm, diff
            );
        }
    }

    // Generate radial profiles for plotting with high resolution
    let n_points = 800; // 4x more interpolation points
    let max_radius = 2.0;

    println!("\nTelescope aperture area: {aperture_cm2:.1} cm²");

    // Create plots directory if it doesn't exist
    std::fs::create_dir_all("plots")?;

    // Generate plots for all sensors
    for (sensor_name, sensor_qe) in &sensors {
        create_sensor_plot(
            sensor_name,
            sensor_qe,
            &airy_disk,
            &stars,
            aperture_cm2,
            &integration_time,
            max_radius,
            n_points,
        )?;
    }

    // Generate 2D PSF images
    println!("\nGenerating 2D PSF images...");

    create_2d_psf_comparison(
        &airy_disk,
        &sun_like,
        sensors[0].1,
        sensors[0].0,
        aperture_cm2,
        &integration_time,
        max_radius,
    )?;

    println!("\nPlots saved:");
    println!("- plots/chromatic_airy_hwk4123.png: HWK4123 radial profile comparison");
    println!("- plots/chromatic_airy_imx455.png: IMX455 radial profile comparison");
    println!("- plots/chromatic_psf_2d.png: 2D PSF visualization");

    Ok(())
}

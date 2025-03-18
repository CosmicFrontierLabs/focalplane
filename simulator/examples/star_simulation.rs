//! Star field simulation example
//!
//! This example demonstrates:
//! 1. Computing the diameter of a telescope/sensor pair field of view
//! 2. Collecting stars within the field from a catalog
//! 3. Computing the flux from stars collected by the telescope
//! 4. Generating a histogram of star magnitudes in the field

use simulator::hardware::sensor::models as sensor_models;
use simulator::hardware::telescope::models as telescope_models;
use simulator::{field_diameter, filter_stars_in_field, magnitude_to_photon_flux};
use starfield::catalogs::{BinaryCatalog, StarCatalog};
use std::path::PathBuf;
use viz::histogram::{Histogram, HistogramConfig, Scale};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Default values
    let mut catalog_path = None;
    let mut ra_deg = 100.0;
    let mut dec_deg = 45.0;
    let mut telescope_name = "1m Final";
    let mut sensor_name = "GSENSE4040BSI";

    // Parse arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--catalog" => {
                if i + 1 < args.len() {
                    catalog_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    return Err("Missing value for --catalog".into());
                }
            }
            "--ra" => {
                if i + 1 < args.len() {
                    ra_deg = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --ra".into());
                }
            }
            "--dec" => {
                if i + 1 < args.len() {
                    dec_deg = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --dec".into());
                }
            }
            "--telescope" => {
                if i + 1 < args.len() {
                    telescope_name = &args[i + 1];
                    i += 2;
                } else {
                    return Err("Missing value for --telescope".into());
                }
            }
            "--sensor" => {
                if i + 1 < args.len() {
                    sensor_name = &args[i + 1];
                    i += 2;
                } else {
                    return Err("Missing value for --sensor".into());
                }
            }
            "--help" => {
                println!("Star Simulation Example");
                println!("======================");
                println!("Usage: cargo run --example star_simulation -- [OPTIONS]");
                println!("Options:");
                println!("  --catalog PATH     Path to binary star catalog");
                println!("  --ra DEGREES       Right ascension of field center (default: 100.0)");
                println!("  --dec DEGREES      Declination of field center (default: 45.0)");
                println!("  --telescope NAME   Telescope model (default: '1m Final')");
                println!("  --sensor NAME      Sensor model (default: 'GSENSE4040BSI')");
                println!("  --help             Display this help message");
                return Ok(());
            }
            _ => {
                println!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    // Select telescope and sensor models
    let telescope = match telescope_name {
        "50cm Demo" => telescope_models::DEMO_50CM.clone(),
        "1m Final" => telescope_models::FINAL_1M.clone(),
        _ => return Err(format!("Unknown telescope model: {}", telescope_name).into()),
    };

    let sensor = match sensor_name {
        "GSENSE4040BSI" => sensor_models::GSENSE4040BSI.clone(),
        "HWK4123" => sensor_models::HWK4123.clone(),
        "IMX455" => sensor_models::IMX455.clone(),
        _ => return Err(format!("Unknown sensor model: {}", sensor_name).into()),
    };

    println!("Star Field Simulation");
    println!("====================");
    println!(
        "Telescope: {} (aperture: {}m, focal length: {}m)",
        telescope.name, telescope.aperture_m, telescope.focal_length_m
    );
    println!(
        "Sensor: {} ({}×{} pixels, {}μm/px)",
        sensor.name, sensor.width_px, sensor.height_px, sensor.pixel_size_um
    );
    println!("Field center: RA={:.4}°, Dec={:.4}°", ra_deg, dec_deg);

    // Step 1: Calculate the field diameter
    let fov_diameter = field_diameter(&telescope, &sensor);
    println!("Field diameter: {:.4}°", fov_diameter);
    println!(
        "We'll search for stars within {:.4}° (2× field diameter)",
        fov_diameter * 2.0
    );

    // Check if we have a catalog path
    if let Some(path) = catalog_path {
        // Load the catalog
        println!("Loading catalog from: {}", path.display());
        let catalog = BinaryCatalog::load(path)?;
        println!("Loaded catalog: {}", catalog.description());
        println!("Total stars in catalog: {}", catalog.len());

        // Step 2: Get stars as StarData objects and filter those in field
        let search_diameter = fov_diameter * 2.0;

        // Convert catalog stars to StarData (which already implements StarPosition)
        let star_data: Vec<_> = catalog.star_data().collect();

        // Filter stars in field
        let visible_stars = filter_stars_in_field(&star_data, ra_deg, dec_deg, search_diameter);
        println!("Found {} stars within search area", visible_stars.len());

        // Step 3: Calculate flux for each star
        // Simulate a 1-second exposure
        let exposure_time = 1.0;
        let wavelength_nm = 550.0; // Green light

        // Calculate flux for each star
        let mut stars_with_flux: Vec<_> = visible_stars
            .iter()
            .map(|&star| {
                let flux = magnitude_to_photon_flux(
                    star.magnitude,
                    exposure_time,
                    &telescope,
                    &sensor,
                    wavelength_nm,
                );
                (star, flux)
            })
            .collect();

        // Sort by descending flux (brightest first)
        stars_with_flux.sort_by(|(_, flux_a), (_, flux_b)| flux_b.partial_cmp(flux_a).unwrap());

        // Display the brightest stars
        println!("\nBrightest stars in field:");
        println!(
            "{:<5} {:<10} {:<10} {:<10} {:<15}",
            "ID", "RA (°)", "Dec (°)", "Mag", "Flux (photons)"
        );
        println!("{:-<55}", "");

        for (i, (star, flux)) in stars_with_flux.iter().take(10).enumerate() {
            println!(
                "{:<5} {:<10.4} {:<10.4} {:<10.2} {:<15.1}",
                i + 1,
                star.ra,
                star.dec,
                star.magnitude,
                flux
            );
        }

        // Calculate total flux from all stars
        let total_flux: f64 = stars_with_flux.iter().map(|(_, flux)| flux).sum();
        println!(
            "\nTotal photon flux from all stars: {:.1} photons",
            total_flux
        );

        // Summary statistics
        if !stars_with_flux.is_empty() {
            let min_mag = stars_with_flux
                .iter()
                .map(|(star, _)| star.magnitude)
                .fold(f64::INFINITY, f64::min);

            let max_mag = stars_with_flux
                .iter()
                .map(|(star, _)| star.magnitude)
                .fold(f64::NEG_INFINITY, f64::max);

            println!("Magnitude range: {:.2} to {:.2}", min_mag, max_mag);

            // Create magnitude histogram
            println!("\nMagnitude Histogram:");

            // Extract magnitudes for histogram
            let magnitudes: Vec<f64> = stars_with_flux
                .iter()
                .map(|(star, _)| star.magnitude)
                .collect();

            // Round min/max to create nice histogram boundaries
            let hist_min = (min_mag.floor() - 1.0).max(-1.0);
            let hist_max = (max_mag.ceil() + 1.0).min(25.0);
            let bin_size = 0.5; // 0.5 magnitude bins
            let num_bins = ((hist_max - hist_min) / bin_size).ceil() as usize;

            // Create the histogram
            let mut hist = Histogram::new_equal_bins(hist_min..hist_max, num_bins)?;

            // Configure histogram display
            let mut config = HistogramConfig::default();
            config.title = Some(format!(
                "Star Magnitude Distribution (Field center: RA={:.2}°, Dec={:.2}°)",
                ra_deg, dec_deg
            ));
            config.max_bar_width = 50;
            config.show_empty_bins = true;
            hist = hist.with_config(config);

            // Add values to histogram
            hist.add_all(magnitudes);

            // Display standard histogram
            println!("{}", hist.format()?);

            // Also show log-scaled version for better visibility of distribution
            let mut log_config = HistogramConfig::default();
            log_config.title = Some(format!(
                "Star Magnitude Distribution - Log Scale (Field center: RA={:.2}°, Dec={:.2}°)",
                ra_deg, dec_deg
            ));
            log_config.scale = Scale::Log10;
            log_config.max_bar_width = 50;
            log_config.show_empty_bins = true;

            let log_hist = hist.with_config(log_config);
            println!("\n{}", log_hist.format()?);
        }
    } else {
        println!("No catalog provided. Use --catalog to specify a binary catalog file.");
    }

    Ok(())
}

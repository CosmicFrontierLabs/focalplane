//! Tool to download and analyze a star catalog
//!
//! This binary downloads the Hipparcos star catalog and prints statistics about it.

// No need for HashMap as we're using vectors
use std::io::{self, Write};

use starfield::catalogs::StarCatalog;
use starfield::Loader;

/// Print a simple progress bar
fn print_progress(progress: f64, width: usize) {
    let filled_width = (progress * width as f64).round() as usize;
    let empty_width = width - filled_width;

    print!("\r[");
    for _ in 0..filled_width {
        print!("#");
    }
    for _ in 0..empty_width {
        print!(" ");
    }
    print!("] {:.1}%", progress * 100.0);
    io::stdout().flush().unwrap();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Star Catalog Statistics Tool");
    println!("===========================");

    // Create a loader
    let loader = Loader::new();

    // Load the catalog with a high magnitude limit to get many stars
    println!("Loading Hipparcos catalog (this may take a moment)...");
    let catalog = loader.load_hipparcos_catalog(10.0)?;

    println!("\nCatalog loaded successfully!");
    println!("Total stars: {}", catalog.len());

    // Calculate magnitude distribution
    println!("\nCalculating magnitude distribution...");
    let mag_ranges = [
        (-2.0, 0.0, "Very bright stars (mag -2 to 0)"),
        (0.0, 2.0, "Bright stars (mag 0 to 2)"),
        (2.0, 4.0, "Medium bright stars (mag 2 to 4)"),
        (4.0, 6.0, "Naked eye visible stars (mag 4 to 6)"),
        (6.0, 8.0, "Binocular visible stars (mag 6 to 8)"),
        (8.0, 10.0, "Telescope visible stars (mag 8 to 10)"),
    ];

    let mut magnitude_counts = vec![0; mag_ranges.len()];

    for (i, (min, max, _)) in mag_ranges.iter().enumerate() {
        print_progress(i as f64 / mag_ranges.len() as f64, 50);
        let stars_in_range = catalog
            .filter(|star| star.mag >= *min && star.mag < *max)
            .len();
        magnitude_counts[i] = stars_in_range;
    }
    print_progress(1.0, 50);
    println!();

    // Print magnitude distribution
    println!("\nMagnitude Distribution:");
    for (i, (_, _, desc)) in mag_ranges.iter().enumerate() {
        let count = magnitude_counts[i];
        let percentage = count as f64 / catalog.len() as f64 * 100.0;
        println!("  {}: {} stars ({:.1}%)", desc, count, percentage);
    }

    // Calculate spatial distribution
    println!("\nCalculating spatial distribution by declination bands...");
    let dec_bands = [
        (-90.0, -60.0, "South polar region (dec -90° to -60°)"),
        (-60.0, -30.0, "South temperate (dec -60° to -30°)"),
        (-30.0, 0.0, "South tropical (dec -30° to 0°)"),
        (0.0, 30.0, "North tropical (dec 0° to 30°)"),
        (30.0, 60.0, "North temperate (dec 30° to 60°)"),
        (60.0, 90.0, "North polar region (dec 60° to 90°)"),
    ];

    let mut dec_counts = vec![0; dec_bands.len()];
    for (i, (min, max, _)) in dec_bands.iter().enumerate() {
        print_progress(i as f64 / dec_bands.len() as f64, 50);
        let stars_in_band = catalog
            .filter(|star| star.dec >= *min && star.dec < *max)
            .len();
        dec_counts[i] = stars_in_band;
    }
    print_progress(1.0, 50);
    println!();

    // Print declination distribution
    println!("\nSpatial Distribution (by declination):");
    for (i, (_, _, desc)) in dec_bands.iter().enumerate() {
        let count = dec_counts[i];
        let percentage = count as f64 / catalog.len() as f64 * 100.0;
        println!("  {}: {} stars ({:.1}%)", desc, count, percentage);
    }

    // Find and list some notable stars
    println!("\nNotable Stars:");

    // Sirius (Alpha Canis Majoris) - brightest star
    if let Some(sirius) = catalog.get_star(32349) {
        println!("  Sirius (Alpha Canis Majoris):");
        println!("    HIP: {}", sirius.hip);
        println!("    Magnitude: {:.2}", sirius.mag);
        println!("    Position: RA {:.4}°, Dec {:.4}°", sirius.ra, sirius.dec);
        if let Some(parallax) = sirius.parallax {
            let distance_pc = 1000.0 / parallax;
            println!(
                "    Distance: {:.2} parsecs ({:.2} light years)",
                distance_pc,
                distance_pc * 3.26156
            );
        }
    } else {
        println!("  Sirius not found in catalog");
    }

    // Betelgeuse (Alpha Orionis) - red supergiant
    if let Some(betelgeuse) = catalog.get_star(27989) {
        println!("\n  Betelgeuse (Alpha Orionis):");
        println!("    HIP: {}", betelgeuse.hip);
        println!("    Magnitude: {:.2}", betelgeuse.mag);
        println!(
            "    Position: RA {:.4}°, Dec {:.4}°",
            betelgeuse.ra, betelgeuse.dec
        );
        if let Some(b_v) = betelgeuse.b_v {
            println!("    B-V Color Index: {:.2} (reddish)", b_v);
        }
    } else {
        println!("  Betelgeuse not found in catalog");
    }

    // Vega (Alpha Lyrae) - blue-white star
    if let Some(vega) = catalog.get_star(91262) {
        println!("\n  Vega (Alpha Lyrae):");
        println!("    HIP: {}", vega.hip);
        println!("    Magnitude: {:.2}", vega.mag);
        println!("    Position: RA {:.4}°, Dec {:.4}°", vega.ra, vega.dec);
        if let Some(b_v) = vega.b_v {
            println!("    B-V Color Index: {:.2} (blue-white)", b_v);
        }
    } else {
        println!("  Vega not found in catalog");
    }

    // Find fastest moving stars (highest proper motion)
    println!("\nStars with Highest Proper Motion:");
    let mut stars_vec: Vec<_> = catalog.stars().collect();
    stars_vec.sort_by(|a, b| {
        let a_pm = a
            .pm_ra
            .unwrap_or(0.0)
            .abs()
            .hypot(a.pm_dec.unwrap_or(0.0).abs());
        let b_pm = b
            .pm_ra
            .unwrap_or(0.0)
            .abs()
            .hypot(b.pm_dec.unwrap_or(0.0).abs());
        b_pm.partial_cmp(&a_pm).unwrap()
    });

    for (i, star) in stars_vec.iter().take(5).enumerate() {
        let pm = star.pm_ra.unwrap_or(0.0).hypot(star.pm_dec.unwrap_or(0.0));
        println!(
            "  {}. HIP {} - Proper Motion: {:.2} mas/year",
            i + 1,
            star.hip,
            pm
        );
    }

    // Create a simple ASCII sky map
    println!("\nSimple Sky Map (RA vs Dec, brightest stars only):");
    println!("  Legend: * (mag < 0), + (mag < 1), . (mag < 2)");

    // Get terminal dimensions and set grid size
    let height: usize = 20; // Fixed height

    // Use a fixed width for the grid
    let width: usize = 100;

    // Create a grid for the map
    let mut grid = vec![vec![' '; width]; height];

    // Place stars on the grid (only bright ones)
    let bright_stars = catalog.brighter_than(2.0);
    for star in bright_stars {
        // Map RA (0-360) to x (0-width)
        let x = ((star.ra / 360.0) * width as f64) as usize % width;

        // Map Dec (-90 to +90) to y (height-1 to 0)
        let y = ((90.0 - star.dec) / 180.0 * height as f64) as usize;
        let y = y.min(height - 1);

        // Choose a character based on magnitude
        let char = if star.mag < 0.0 {
            '*' // Very bright stars
        } else if star.mag < 1.0 {
            '+' // Bright stars
        } else {
            '.' // Visible stars
        };

        // Place the star on the grid if in bounds
        if y < height && x < width {
            grid[y][x] = char;
        }
    }

    // Draw the grid with borders
    println!("  {}", "-".repeat(width + 2));
    for row in grid {
        print!("  |");
        for cell in row {
            print!("{}", cell);
        }
        println!("|");
    }
    println!("  {}", "-".repeat(width + 2));
    println!("  South Pole");
    println!("  (RA increases from left to right, Dec increases from bottom to top)");
    println!(
        "  Map width: {} characters (auto-adjusted to terminal size)",
        width
    );

    Ok(())
}

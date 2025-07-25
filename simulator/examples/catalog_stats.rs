//! Catalog comparison tool
//!
//! Compares bright stars from Hipparcos catalog against a binary catalog
//! and reports which bright stars are missing from the binary catalog.

use clap::Parser;
use simulator::algo::MinMaxScan;
use starfield::catalogs::binary_catalog::{BinaryCatalog, MinimalStar};
use starfield::catalogs::hipparcos::HipparcosCatalog;
use starfield::catalogs::{StarCatalog, StarPosition};
use starfield::Equatorial;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use viz::histogram::create_magnitude_histogram;

/// Converts equatorial coordinates from one epoch to another using approximate precession.
///
/// This function applies precession corrections to account for the slow wobble of Earth's
/// rotational axis. It uses simplified IAU 2000-based constants for general precession.
///
/// # Limitations
///
/// This is a simplified calculation that **ignores**:
/// - Proper motion of stars
/// - Parallax effects
/// - Stellar aberration
/// - Nutation (short-term oscillations)
/// - Relativistic effects
///
/// For high-precision astrometric work, use a full-featured astronomical library.
///
/// # Arguments
///
/// * `coords` - Original equatorial coordinates (RA/Dec)
/// * `from_epoch` - Source epoch in Julian years (e.g., 1991.25 for J1991.25)
/// * `to_epoch` - Target epoch in Julian years (e.g., 2015.0 for J2015.0)
///
/// # Returns
///
/// New equatorial coordinates adjusted for precession between epochs.
/// RA is normalized to 0-360° range, Dec is clamped to ±90°.
///
/// # Examples
///
/// ```rust
/// use starfield::Equatorial;
/// use catalog_stats::convert_epoch;
///
/// // Convert Hipparcos J1991.25 coordinates to Gaia J2015.0
/// let hip_coords = Equatorial::from_degrees(123.45, 67.89);
/// let gaia_coords = convert_epoch(hip_coords, 1991.25, 2015.0);
/// ```
pub fn convert_epoch(coords: Equatorial, from_epoch: f64, to_epoch: f64) -> Equatorial {
    let dt = to_epoch - from_epoch; // Time difference in years

    // Get coordinates in degrees
    let ra_deg = coords.ra_degrees();
    let dec_deg = coords.dec_degrees();

    // Convert to radians for calculations
    let ra_rad = ra_deg * PI / 180.0;
    let dec_rad = dec_deg * PI / 180.0;

    // Precession constants (approximate, based on IAU 2000)
    // These are in arcseconds per year
    let m = 3.075; // General precession in RA
    let n = 1.336; // Precession coefficient

    // Calculate precession corrections in arcseconds
    let delta_ra_arcsec = (m + n * ra_rad.sin() * dec_rad.tan()) * dt;
    let delta_dec_arcsec = n * ra_rad.cos() * dt;

    // Convert arcseconds to degrees
    let delta_ra_deg = delta_ra_arcsec / 3600.0;
    let delta_dec_deg = delta_dec_arcsec / 3600.0;

    // Apply corrections
    let new_ra = ra_deg + delta_ra_deg;
    let new_dec = dec_deg + delta_dec_deg;

    // Normalize RA to 0-360 range
    let normalized_ra = if new_ra < 0.0 {
        new_ra + 360.0
    } else if new_ra >= 360.0 {
        new_ra - 360.0
    } else {
        new_ra
    };

    // Clamp Dec to -90 to +90 range
    let clamped_dec = new_dec.max(-90.0).min(90.0);

    Equatorial::from_degrees(normalized_ra, clamped_dec)
}

/// Efficient collector for the brightest stars with automatic brightness-based filtering.
///
/// This structure maintains a sorted list of at most `max_n` stars, automatically
/// dropping dimmer stars when the limit is exceeded. Stars are kept sorted by
/// magnitude (brightest first), enabling efficient collection of the N brightest
/// stars from a large catalog without needing to sort the entire dataset.
///
/// # Performance
///
/// - Time complexity: O(n log k) where n is total stars processed, k is max_n
/// - Space complexity: O(k) where k is max_n
/// - Uses binary search for efficient insertion
///
/// # Examples
///
/// ```rust
/// use catalog_stats::MaxMagnitudeCollector;
/// use starfield::catalogs::binary_catalog::MinimalStar;
///
/// let mut collector = MaxMagnitudeCollector::new(100);
/// // Add stars from catalog - only brightest 100 will be kept
/// for star in catalog.stars() {
///     collector.add(*star);
/// }
/// let brightest = collector.stars(); // Already sorted brightest to dimmest
/// ```
#[derive(Debug)]
pub struct MaxMagnitudeCollector {
    stars: Vec<MinimalStar>,
    max_n: usize,
}

impl MaxMagnitudeCollector {
    /// Creates a new collector that will retain at most `max_n` brightest stars.
    ///
    /// # Arguments
    ///
    /// * `max_n` - Maximum number of stars to keep in the collection
    ///
    /// # Examples
    ///
    /// ```rust
    /// use catalog_stats::MaxMagnitudeCollector;
    /// let collector = MaxMagnitudeCollector::new(1000); // Keep 1000 brightest
    /// ```
    pub fn new(max_n: usize) -> Self {
        Self {
            stars: Vec::with_capacity(max_n),
            max_n,
        }
    }

    /// Adds a star to the collection, maintaining brightness order.
    ///
    /// If the collection is not yet full, the star is inserted in the correct
    /// sorted position. If full, the star is only added if it's brighter than
    /// the dimmest star currently in the collection.
    ///
    /// # Arguments
    ///
    /// * `star` - Star to potentially add to the collection
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use catalog_stats::MaxMagnitudeCollector;
    /// # use starfield::catalogs::binary_catalog::MinimalStar;
    /// let mut collector = MaxMagnitudeCollector::new(3);
    /// collector.add(MinimalStar { magnitude: 5.0, ..Default::default() });
    /// collector.add(MinimalStar { magnitude: 3.0, ..Default::default() });
    /// collector.add(MinimalStar { magnitude: 4.0, ..Default::default() });
    /// // Collection now has stars with magnitudes [3.0, 4.0, 5.0]
    /// ```
    pub fn add(&mut self, star: MinimalStar) {
        // If list not full, just insert in sorted position
        if self.stars.len() < self.max_n {
            let insert_pos = self
                .stars
                .binary_search_by(|s| s.magnitude.partial_cmp(&star.magnitude).unwrap())
                .unwrap_or_else(|pos| pos);
            self.stars.insert(insert_pos, star);
            return;
        }

        // List is full - only add if brighter than dimmest star
        if let Some(dimmest) = self.stars.last() {
            if star.magnitude < dimmest.magnitude {
                // Remove the dimmest star
                self.stars.pop();

                // Insert new star in sorted position
                let insert_pos = self
                    .stars
                    .binary_search_by(|s| s.magnitude.partial_cmp(&star.magnitude).unwrap())
                    .unwrap_or_else(|pos| pos);
                self.stars.insert(insert_pos, star);
            }
        }
    }

    /// Returns the collected stars, sorted from brightest to dimmest.
    ///
    /// The returned slice is guaranteed to be sorted by magnitude in ascending
    /// order (brightest stars have the lowest magnitude values).
    ///
    /// # Returns
    ///
    /// Slice of stars sorted by magnitude (brightest first)
    pub fn stars(&self) -> &[MinimalStar] {
        &self.stars
    }

    /// Returns the number of stars currently in the collection.
    ///
    /// This will be at most the `max_n` value specified during construction.
    pub fn len(&self) -> usize {
        self.stars.len()
    }

    /// Returns `true` if the collection contains no stars.
    pub fn is_empty(&self) -> bool {
        self.stars.is_empty()
    }
}

/// Command-line arguments for the catalog comparison tool.
///
/// This structure defines all the parameters needed to compare bright stars
/// between the Hipparcos catalog and a binary star catalog, finding stars
/// that are present in Hipparcos but missing from the binary catalog.
#[derive(Parser, Debug)]
#[command(name = "catalog_stats")]
#[command(about = "Find bright stars missing from binary catalog compared to Hipparcos")]
struct Args {
    /// Path to the binary catalog file
    #[arg(short, long)]
    catalog: PathBuf,

    /// Brightness threshold (magnitude limit - stars brighter than this)
    #[arg(short, long, default_value_t = 6.0)]
    threshold: f64,

    /// Output CSV file for missing stars
    #[arg(short, long, default_value = "missing_stars.csv")]
    output: PathBuf,

    /// Position tolerance in degrees for matching stars
    #[arg(long, default_value_t = 0.01)]
    position_tolerance: f64,

    /// Magnitude tolerance for matching stars
    #[arg(long, default_value_t = 1.5)]
    magnitude_tolerance: f64,

    /// Hipparcos catalog epoch (default: J1991.25)
    #[arg(long, default_value_t = 1991.25)]
    hipparcos_epoch: f64,

    /// Binary catalog epoch (default: J2015.0 for Gaia)
    #[arg(long, default_value_t = 2015.0)]
    binary_epoch: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load Hipparcos catalog from cache
    println!("Loading Hipparcos catalog...");
    let hipparcos_path = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache/starfield/hip_main.dat");
    let hipparcos = HipparcosCatalog::from_dat_file(&hipparcos_path, args.threshold)?;
    println!("Loaded {} stars from Hipparcos", hipparcos.len());

    // Load binary catalog
    println!("Loading binary catalog from: {}", args.catalog.display());
    let binary_catalog = BinaryCatalog::load(&args.catalog)?;
    println!("Loaded {} stars from binary catalog", binary_catalog.len());

    // Efficiently collect 2000 brightest stars from binary catalog
    let mut collector = MaxMagnitudeCollector::new(2000);

    // Process all stars in the catalog, keeping only the brightest
    for star in binary_catalog.stars().iter() {
        collector.add(*star);
    }

    let brightest_2000 = collector.stars();
    let num_brightest = brightest_2000.len();

    println!("\n=== BINARY CATALOG ANALYSIS: {num_brightest} BRIGHTEST STARS ===");

    // Detailed statistics
    let magnitudes: Vec<f64> = brightest_2000.iter().map(|s| s.magnitude).collect();
    let mag_scan = MinMaxScan::new(&magnitudes);
    let min_mag = mag_scan.min().unwrap_or(f64::INFINITY);
    let max_mag = mag_scan.max().unwrap_or(f64::NEG_INFINITY);
    let mean_mag = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;

    // Calculate median
    let mut sorted_mags = magnitudes.clone();
    sorted_mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_mag = if sorted_mags.len() % 2 == 0 {
        (sorted_mags[sorted_mags.len() / 2 - 1] + sorted_mags[sorted_mags.len() / 2]) / 2.0
    } else {
        sorted_mags[sorted_mags.len() / 2]
    };

    // Calculate standard deviation
    let variance = magnitudes
        .iter()
        .map(|&mag| (mag - mean_mag).powi(2))
        .sum::<f64>()
        / magnitudes.len() as f64;
    let std_dev = variance.sqrt();

    println!("Magnitude Statistics:");
    println!("  Brightest (min):     {min_mag:.3}");
    println!("  Faintest (max):      {max_mag:.3}");
    println!("  Range:               {:.3}", max_mag - min_mag);
    println!("  Mean:                {mean_mag:.3}");
    println!("  Median:              {median_mag:.3}");
    println!("  Standard Deviation:  {std_dev:.3}");
    println!("  Total stars analyzed: {num_brightest}");

    // Create and display magnitude histogram
    println!("\nMagnitude Distribution Histogram:");
    match create_magnitude_histogram(
        &magnitudes,
        Some(format!(
            "Binary Catalog - {num_brightest} Brightest Stars Magnitude Distribution"
        )),
        false, // Linear scale
    ) {
        Ok(hist) => match hist.format() {
            Ok(formatted) => println!("{formatted}"),
            Err(e) => println!("Error formatting histogram: {e}"),
        },
        Err(e) => println!("Error creating histogram: {e}"),
    }

    println!();

    // Filter Hipparcos stars by brightness threshold and sort brightest first
    let mut bright_hipparcos: Vec<_> = hipparcos
        .stars()
        .filter(|star| star.mag <= args.threshold)
        .collect();

    bright_hipparcos.sort_by(|a, b| {
        a.mag
            .partial_cmp(&b.mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "Found {} bright stars (mag <= {:.1}) in Hipparcos",
        bright_hipparcos.len(),
        args.threshold
    );

    // Filter binary catalog stars by brightness threshold
    let bright_binary: Vec<_> = binary_catalog
        .stars()
        .iter()
        .filter(|star| star.magnitude <= args.threshold)
        .collect();

    println!(
        "Found {} bright stars (mag <= {:.1}) in binary catalog",
        bright_binary.len(),
        args.threshold
    );

    // Print magnitude statistics for binary catalog
    let all_mags: Vec<f64> = binary_catalog.stars().iter().map(|s| s.magnitude).collect();
    if !all_mags.is_empty() {
        let all_mag_scan = MinMaxScan::new(&all_mags);
        if let Ok((min_mag, max_mag)) = all_mag_scan.min_max() {
            println!("Binary catalog magnitude range: {min_mag:.2} to {max_mag:.2}");
        }
    }

    // Print matching tolerances and epochs
    println!(
        "Using matching tolerances: {:.3}° position, {:.1} magnitude",
        args.position_tolerance, args.magnitude_tolerance
    );
    println!(
        "Epoch conversion: Hipparcos J{:.2} -> Binary catalog J{:.1}",
        args.hipparcos_epoch, args.binary_epoch
    );

    // Find missing stars by comparing positions and magnitudes using angular distance
    let mut missing_stars = Vec::new();
    let position_tolerance = args.position_tolerance;
    let magnitude_tolerance = args.magnitude_tolerance;
    const CLOSEST_MATCH_MAG_TOLERANCE: f64 = 5.0; // magnitude difference for closest match reporting

    for hip_star in &bright_hipparcos {
        // Convert Hipparcos coordinates from its epoch to the binary catalog epoch
        let hip_coords = Equatorial::from_degrees(hip_star.ra(), hip_star.dec());
        let hip_equatorial = convert_epoch(hip_coords, args.hipparcos_epoch, args.binary_epoch);
        let mut found = false;
        let mut closest_distance = f64::INFINITY;
        let mut closest_star = None;

        for bin_star in &bright_binary {
            let bin_equatorial = Equatorial::from_degrees(bin_star.ra(), bin_star.dec());
            let angular_distance = hip_equatorial
                .angular_distance(&bin_equatorial)
                .to_degrees();
            let mag_diff = (hip_star.mag - bin_star.magnitude).abs();

            // Check if this is a match within tolerances
            if angular_distance < position_tolerance && mag_diff < magnitude_tolerance {
                found = true;
                break;
            }

            // Track closest star within magnitude range for reporting (only if reasonably close)
            if mag_diff < CLOSEST_MATCH_MAG_TOLERANCE
                && angular_distance < closest_distance
                && angular_distance < position_tolerance
            {
                closest_distance = angular_distance;
                closest_star = Some(bin_star);
            }
        }

        if !found {
            missing_stars.push((hip_star, closest_star, closest_distance));
        }
    }

    // Already sorted by magnitude (brightest first)
    println!(
        "\n{} bright stars missing from binary catalog:",
        missing_stars.len()
    );

    // Display results with closest match information
    println!(
        "{:>4} {:>12} {:>12} {:>8} {:>12} {:>12} {:>8} {:>12}",
        "Rank",
        "RA (deg)",
        "Dec (deg)",
        "Mag",
        "Closest_RA",
        "Closest_Dec",
        "Closest_Mag",
        "Distance(°)"
    );
    println!("{}", "-".repeat(100));

    for (i, (star, closest_star, closest_distance)) in missing_stars.iter().enumerate() {
        let closest_info = if let Some(closest) = closest_star {
            format!(
                "{:>12.6} {:>12.6} {:>8.2} {:>12.6}",
                closest.ra(),
                closest.dec(),
                closest.magnitude,
                closest_distance
            )
        } else {
            format!("{:>12} {:>12} {:>8} {:>12}", "N/A", "N/A", "N/A", "N/A")
        };

        println!(
            "{:>4} {:>12.6} {:>12.6} {:>8.2} {}",
            i + 1,
            star.ra(),
            star.dec(),
            star.mag,
            closest_info
        );
    }

    // Save to CSV file
    let mut csv_file = File::create(&args.output)?;
    writeln!(csv_file, "Rank,RA_deg,Dec_deg,Magnitude,Closest_RA_deg,Closest_Dec_deg,Closest_Magnitude,Angular_Distance_deg")?;

    for (i, (star, closest_star, closest_distance)) in missing_stars.iter().enumerate() {
        let closest_info = if let Some(closest) = closest_star {
            format!(
                "{:.6},{:.6},{:.2},{:.6}",
                closest.ra(),
                closest.dec(),
                closest.magnitude,
                closest_distance
            )
        } else {
            ",,,,".to_string()
        };

        writeln!(
            csv_file,
            "{},{:.6},{:.6},{:.2},{}",
            i + 1,
            star.ra(),
            star.dec(),
            star.mag,
            closest_info
        )?;
    }

    println!("\nMissing stars saved to: {}", args.output.display());

    Ok(())
}

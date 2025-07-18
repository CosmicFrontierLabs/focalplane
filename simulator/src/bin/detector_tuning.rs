//! Unified detector parameter tuning tool
//!
//! This tool helps optimize star detection parameters for different algorithms (DAO, IRAF, and Naive).
//!
//! # Usage
//!
//! ```bash
//! # Build the tool
//! cargo build --release --bin detector_tuning
//!
//! # Quick test at a single position
//! cargo run --release --bin detector_tuning -- quick -d dao
//! cargo run --release --bin detector_tuning -- quick -d iraf
//! cargo run --release --bin detector_tuning -- quick -d naive
//!
//! # Quick test with custom FWHM
//! cargo run --release --bin detector_tuning -- quick -d dao --fwhm 4.0
//!
//! # Grid test for overall performance
//! cargo run --release --bin detector_tuning -- grid -d dao -g 20 -i 256
//! cargo run --release --bin detector_tuning -- grid -d iraf -g 20 -i 256
//!
//! # Parameter sweep to find optimal values
//! cargo run --release --bin detector_tuning -- sweep -d dao -p fwhm
//! cargo run --release --bin detector_tuning -- sweep -d dao -p sigma_radius
//! cargo run --release --bin detector_tuning -- sweep -d iraf -p fwhm
//! ```
//!
//! # Commands
//!
//! ## `quick` - Quick single-position test
//! - Tests a detector at position (25.3, 25.7)
//! - Shows centroid error
//! - Options:
//!   - `-d, --detector`: dao, iraf, or naive
//!   - `-f, --fwhm`: Optional FWHM override
//!
//! ## `grid` - Grid test for accuracy statistics
//! - Tests detector on a grid of sub-pixel positions
//! - Reports average error, max error, and success rate
//! - Options:
//!   - `-d, --detector`: dao, iraf, or naive
//!   - `-g, --grid-size`: Grid size (default: 20)
//!   - `-i, --image-size`: Image size in pixels (default: 256)
//!
//! ## `sweep` - Parameter sweep
//! - Tests a range of values for a specific parameter
//! - Shows which values give best accuracy
//! - Options:
//!   - `-d, --detector`: dao or iraf
//!   - `-p, --parameter`: fwhm, sigma_radius, or threshold
//!
//! # Key Findings from Optimization
//!
//! - **DAO**: Needs FWHM ≥ 3.5 for excellent accuracy (< 0.002 pixels)
//! - **IRAF**: Needs FWHM ≈ 2.5 for best accuracy (< 0.02 pixels)
//! - Both detectors benefit from larger FWHM values than traditionally used
//! - The optimized configs are already implemented in `src/image_proc/detection/config.rs`

use clap::{Parser, Subcommand};
use ndarray::Array2;
use simulator::algo::min_max_scan::MinMaxScan;
use simulator::image_proc::airy::PixelScaledAiryDisk;
use starfield::image::starfinders::{
    DAOStarFinder, DAOStarFinderConfig, IRAFStarFinder, IRAFStarFinderConfig, StellarSource,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Quick test of a single detector at one position
    Quick {
        /// Detector type: dao, iraf, or naive
        #[arg(short, long)]
        detector: String,

        /// FWHM multiplier (default based on detector)
        #[arg(short, long)]
        fwhm: Option<f64>,
    },

    /// Grid test for parameter optimization
    Grid {
        /// Detector type: dao, iraf, or naive
        #[arg(short, long)]
        detector: String,

        /// Grid size for sub-pixel testing (default: 20)
        #[arg(short, long, default_value = "20")]
        grid_size: usize,

        /// Image size in pixels (default: 256)
        #[arg(short, long, default_value = "256")]
        image_size: usize,
    },

    /// Sweep through parameter ranges
    Sweep {
        /// Detector type: dao or iraf
        #[arg(short, long)]
        detector: String,

        /// Parameter to sweep: fwhm, sigma_radius, threshold, sharpness, roundness
        #[arg(short, long)]
        parameter: String,
    },
}

/// Generate star image using PixelScaledAiryDisk
fn generate_star_image(
    image_size: usize,
    position_x: f64,
    position_y: f64,
    scaled_airy_disk: &PixelScaledAiryDisk,
    flux: f64,
) -> Array2<f64> {
    let mut image = Array2::<f64>::zeros((image_size, image_size));

    // Create star using PixelScaledAiryDisk's Simpson integration
    for y in 0..image_size {
        for x in 0..image_size {
            let x_pixel = x as f64 - position_x;
            let y_pixel = y as f64 - position_y;
            image[[y, x]] = scaled_airy_disk.pixel_flux_simpson(x_pixel, y_pixel, flux);
        }
    }

    image
}

/// Test DAO detector at a single position
fn test_dao_single(
    config: DAOStarFinderConfig,
    position_x: f64,
    position_y: f64,
    image: &Array2<f64>,
) -> Option<f64> {
    match DAOStarFinder::new(config) {
        Ok(starfinder) => {
            let stars = starfinder.find_stars(image, None);

            if !stars.is_empty() {
                let (star_x, star_y) = stars[0].get_centroid();
                let error = ((star_x - position_x).powi(2) + (star_y - position_y).powi(2)).sqrt();
                Some(error)
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

/// Test IRAF detector at a single position
fn test_iraf_single(
    config: IRAFStarFinderConfig,
    position_x: f64,
    position_y: f64,
    image: &Array2<f64>,
) -> Option<f64> {
    // Scale image like in the original test
    let max_scan = MinMaxScan::new(image.as_slice().unwrap());
    let max_val = max_scan
        .max()
        .expect("Failed to compute max value: image contains NaN or is empty");
    let scale = if max_val > 0.0 {
        65535.0 / max_val
    } else {
        1.0
    };
    let scaled_image = image.mapv(|v| v * scale);

    match IRAFStarFinder::new(config) {
        Ok(starfinder) => {
            let stars = starfinder.find_stars(&scaled_image, None);

            if !stars.is_empty() {
                let (star_x, star_y) = stars[0].get_centroid();
                let error = ((star_x - position_x).powi(2) + (star_y - position_y).powi(2)).sqrt();
                Some(error)
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

/// Test naive detector at a single position
fn test_naive_single(position_x: f64, position_y: f64, image: &Array2<f64>) -> Option<f64> {
    let stars = simulator::image_proc::detection::detect_stars(&image.view(), Some(0.1));

    if !stars.is_empty() {
        let (star_x, star_y) = stars[0].get_centroid();
        let error = ((star_x - position_x).powi(2) + (star_y - position_y).powi(2)).sqrt();
        Some(error)
    } else {
        None
    }
}

/// Run grid test for any detector
fn run_grid_test(detector: &str, grid_size: usize, image_size: usize) {
    let fwhm = 2.355; // FWHM in pixels
    let scaled_airy_disk = PixelScaledAiryDisk::with_fwhm(fwhm, 550.0);
    let background_rms = 10.0; // Reasonable noise level for test stars
    let detection_sigma = 3.0;
    let flux = 1000.0; // Total integrated flux

    let center_x = image_size as f64 / 2.0;
    let center_y = image_size as f64 / 2.0;

    let mut total_error = 0.0;
    let mut max_error: f64 = 0.0;
    let mut total_tests = 0;
    let mut failed_detections = 0;

    println!(
        "Running {}x{} grid test for {} detector...",
        grid_size, grid_size, detector
    );

    for i in 0..grid_size {
        for j in 0..grid_size {
            let sub_x = i as f64 / grid_size as f64;
            let sub_y = j as f64 / grid_size as f64;
            let position_x = center_x + sub_x;
            let position_y = center_y + sub_y;

            // Generate the image once for all detectors
            let image =
                generate_star_image(image_size, position_x, position_y, &scaled_airy_disk, flux);

            let error_opt = match detector {
                "dao" => {
                    let config = simulator::image_proc::detection::config::dao_autoconfig(
                        &scaled_airy_disk,
                        background_rms,
                        detection_sigma,
                    );
                    test_dao_single(config, position_x, position_y, &image)
                }
                "iraf" => {
                    let config = simulator::image_proc::detection::config::iraf_autoconfig(
                        &scaled_airy_disk,
                        background_rms,
                        detection_sigma,
                    );
                    test_iraf_single(config, position_x, position_y, &image)
                }
                "naive" => test_naive_single(position_x, position_y, &image),
                _ => panic!("Unknown detector: {}", detector),
            };

            if let Some(error) = error_opt {
                total_error += error;
                max_error = max_error.max(error);
                total_tests += 1;
            } else {
                failed_detections += 1;
            }
        }
    }

    let avg_error = if total_tests > 0 {
        total_error / total_tests as f64
    } else {
        f64::INFINITY
    };

    println!("\nResults:");
    println!("Average Error: {:.6} pixels", avg_error);
    println!("Maximum Error: {:.6} pixels", max_error);
    println!(
        "Success Rate: {:.2}% ({}/{})",
        100.0 * total_tests as f64 / (grid_size * grid_size) as f64,
        total_tests,
        grid_size * grid_size
    );
    if failed_detections > 0 {
        println!("Failed Detections: {}", failed_detections);
    }
}

/// Run parameter sweep for DAO
fn sweep_dao_parameter(param: &str) {
    let fwhm = 2.355; // FWHM in pixels
    let scaled_airy_disk = PixelScaledAiryDisk::with_fwhm(fwhm, 550.0);
    let test_x = 25.3;
    let test_y = 25.7;
    let image_size = 51;
    let flux = 1000.0;

    println!("Sweeping {} parameter for DAO detector\n", param);
    println!("Value\tError (pixels)\tNotes");
    println!("-----\t--------------\t-----");

    let base_config = DAOStarFinderConfig {
        threshold: 30.0,
        fwhm: 2.0 * fwhm,
        ratio: 1.0,
        theta: 0.0,
        sigma_radius: 1.5,
        sharpness: 0.2..=5.0,
        roundness: -0.5..=0.5,
        exclude_border: false,
        brightest: None,
        peakmax: None,
        min_separation: 0.8 * scaled_airy_disk.first_zero(),
    };

    let values: Vec<f64> = match param {
        "fwhm" => vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
        "sigma_radius" => vec![0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
        "threshold" => vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
        _ => panic!("Unknown parameter: {}", param),
    };

    // Generate the image once outside the loop
    let image = generate_star_image(image_size, test_x, test_y, &scaled_airy_disk, flux);

    for value in values {
        let mut config = base_config.clone();
        match param {
            "fwhm" => config.fwhm = value,
            "sigma_radius" => config.sigma_radius = value,
            "threshold" => config.threshold = value,
            _ => {}
        }

        if let Some(error) = test_dao_single(config, test_x, test_y, &image) {
            let note = if error < 0.002 {
                "★ Excellent!"
            } else if error < 0.02 {
                "Good"
            } else {
                ""
            };
            println!("{:.2}\t{:.6}\t{}", value, error, note);
        } else {
            println!("{:.2}\tNo detection", value);
        }
    }
}

/// Run parameter sweep for IRAF
fn sweep_iraf_parameter(param: &str) {
    let fwhm = 2.355; // FWHM in pixels
    let scaled_airy_disk = PixelScaledAiryDisk::with_fwhm(fwhm, 550.0);
    let test_x = 25.3;
    let test_y = 25.7;
    let image_size = 51;
    let flux = 1000.0;

    println!("Sweeping {} parameter for IRAF detector\n", param);
    println!("Value\tError (pixels)\tNotes");
    println!("-----\t--------------\t-----");

    let base_config = IRAFStarFinderConfig {
        threshold: 30.0,
        fwhm: 1.25 * fwhm,
        sigma_radius: 1.5,
        minsep_fwhm: 1.5,
        sharpness: 0.2..=5.0,
        roundness: -0.3..=0.3,
        exclude_border: false,
        brightest: None,
        peakmax: None,
        min_separation: None,
    };

    let values: Vec<f64> = match param {
        "fwhm" => vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "sigma_radius" => vec![0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
        "threshold" => vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
        _ => panic!("Unknown parameter: {}", param),
    };

    // Generate the image once outside the loop
    let image = generate_star_image(image_size, test_x, test_y, &scaled_airy_disk, flux);

    for value in values {
        let mut config = base_config.clone();
        match param {
            "fwhm" => config.fwhm = value,
            "sigma_radius" => config.sigma_radius = value,
            "threshold" => config.threshold = value,
            _ => {}
        }

        if let Some(error) = test_iraf_single(config, test_x, test_y, &image) {
            let note = if error < 0.02 {
                "★ Excellent!"
            } else if error < 0.05 {
                "Good"
            } else {
                ""
            };
            println!("{:.2}\t{:.6}\t{}", value, error, note);
        } else {
            println!("{:.2}\tNo detection", value);
        }
    }
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Quick { detector, fwhm } => {
            let sigma = 1.0;
            let test_x = 25.3;
            let test_y = 25.7;
            let image_size = 51;

            println!(
                "Quick test of {} detector at position ({:.1}, {:.1})",
                detector, test_x, test_y
            );

            let base_fwhm = sigma * 2.355;
            let scaled_airy_disk = PixelScaledAiryDisk::with_fwhm(base_fwhm, 550.0);
            let flux = 1000.0;

            // Generate the image once
            let image = generate_star_image(image_size, test_x, test_y, &scaled_airy_disk, flux);

            let error_opt = match detector.as_str() {
                "dao" => {
                    let mut config = simulator::image_proc::detection::config::dao_autoconfig(
                        &scaled_airy_disk,
                        10.0,
                        3.0,
                    );
                    if let Some(f) = fwhm {
                        config.fwhm = f;
                    }
                    println!("Using FWHM: {}", config.fwhm);
                    test_dao_single(config, test_x, test_y, &image)
                }
                "iraf" => {
                    let mut config = simulator::image_proc::detection::config::iraf_autoconfig(
                        &scaled_airy_disk,
                        10.0,
                        3.0,
                    );
                    if let Some(f) = fwhm {
                        config.fwhm = f;
                    }
                    println!("Using FWHM: {}", config.fwhm);
                    test_iraf_single(config, test_x, test_y, &image)
                }
                "naive" => test_naive_single(test_x, test_y, &image),
                _ => panic!("Unknown detector: {}", detector),
            };

            if let Some(error) = error_opt {
                println!("\nError: {:.6} pixels", error);
                if error < 0.002 {
                    println!("★ Excellent centroid accuracy!");
                } else if error < 0.02 {
                    println!("Good centroid accuracy");
                }
            } else {
                println!("\nNo star detected!");
            }
        }

        Commands::Grid {
            detector,
            grid_size,
            image_size,
        } => {
            run_grid_test(&detector, grid_size, image_size);
        }

        Commands::Sweep {
            detector,
            parameter,
        } => match detector.as_str() {
            "dao" => sweep_dao_parameter(&parameter),
            "iraf" => sweep_iraf_parameter(&parameter),
            _ => panic!("Sweep only supports dao and iraf detectors"),
        },
    }
}

//! Detection Algorithm Comparison Tool
//!
//! A focused comparison of star detection algorithms measuring:
//! - True positive rate (correct detections)
//! - False positive rate (spurious + multiple detections)
//! - Detection accuracy for each algorithm

use clap::Parser;
use rayon::prelude::*;
use shared::image_proc::airy::PixelScaledAiryDisk;
use shared::image_proc::detection::{detect_stars_unified, StarDetection, StarFinder};
use shared::units::{Temperature, TemperatureExt};
use simulator::hardware::sensor::models::ALL_SENSORS;
use simulator::hardware::sensor_noise::generate_sensor_noise;
use simulator::hardware::SatelliteConfig;
use simulator::image_proc::render::{add_stars_to_image, quantize_image, StarInFrame};
use simulator::photometry::ZodiacalLight;
use simulator::shared_args::SharedSimulationArgs;
use simulator::star_data_to_fluxes;
use starfield::catalogs::StarData;
use starfield::Equatorial;

/// Command line arguments for detection comparison
#[derive(Parser, Debug)]
#[command(
    name = "Detection Algorithm Comparison",
    about = "Compares false/true detection rates across star finding algorithms"
)]
struct Args {
    #[command(flatten)]
    shared: SharedSimulationArgs,

    /// Number of experiments per algorithm per configuration
    #[arg(long, default_value_t = 100)]
    experiments: u32,

    /// Domain size for test images (width and height in pixels)
    #[arg(long, default_value_t = 64)]
    domain: usize,

    /// Star magnitude for testing
    #[arg(long, default_value_t = 14.0)]
    magnitude: f64,

    /// PSF disk size in Airy disk FWHM units
    #[arg(long, default_value_t = 2.0)]
    disk_size: f64,
}

/// Results for a single algorithm test
#[derive(Debug)]
struct AlgorithmResults {
    algorithm: StarFinder,
    sensor_name: String,
    experiments: u32,
    true_positives: u32,
    false_positives: u32,
    spurious_detections: u32,
    multiple_detections: u32,
    #[allow(dead_code)]
    no_detections: u32,
    avg_error_pixels: f64,
}

impl AlgorithmResults {
    fn true_positive_rate(&self) -> f64 {
        self.true_positives as f64 / self.experiments as f64
    }

    fn false_positive_rate(&self) -> f64 {
        self.false_positives as f64 / self.experiments as f64
    }

    #[allow(dead_code)]
    fn no_detection_rate(&self) -> f64 {
        self.no_detections as f64 / self.experiments as f64
    }
}

/// Run detection comparison for one algorithm
fn test_algorithm(
    algorithm: StarFinder,
    satellite: &SatelliteConfig,
    args: &Args,
) -> AlgorithmResults {
    let half_d = args.domain as f64 / 2.0;
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut spurious_detections = 0;
    let mut multiple_detections = 0;
    let mut no_detections = 0;
    let mut total_error = 0.0;

    for _ in 0..args.experiments {
        // Generate random position in center area
        let xpos = rand::random::<f64>() * half_d + half_d;
        let ypos = rand::random::<f64>() * half_d + half_d;

        // Create test star
        let star_data = StarData {
            id: 0,
            position: Equatorial::from_degrees(0.0, 0.0),
            magnitude: args.magnitude,
            b_v: None,
        };

        let star = StarInFrame {
            x: xpos,
            y: ypos,
            spot: star_data_to_fluxes(&star_data, satellite),
            star: star_data,
        };

        // Create electron image and add star
        let e_image = add_stars_to_image(
            args.domain,
            args.domain,
            &vec![star],
            &args.shared.exposure.0,
            satellite.telescope.clear_aperture_area(),
        );

        // Generate and add noise
        let sensor_noise = generate_sensor_noise(
            &satellite.sensor,
            &args.shared.exposure.0,
            satellite.temperature,
            None,
        );

        let z_light = ZodiacalLight::new();
        let zodiacal = z_light.generate_zodiacal_background(
            satellite,
            &args.shared.exposure.0,
            &args.shared.coordinates,
        );

        let noise = &sensor_noise + &zodiacal;
        let total_e_image = &e_image + &noise;
        let quantized = quantize_image(&total_e_image, &satellite.sensor);

        // Calculate detection parameters
        let quantized_noise = quantize_image(&noise, &satellite.sensor);
        let background_rms = quantized_noise.map(|&x| x as f64).mean().unwrap();
        let airy_disk_pixels = satellite.airy_disk_fwhm_sampled().fwhm();
        let detection_sigma = args.shared.noise_multiple;

        // Run detection algorithm
        let scaled_airy_disk =
            PixelScaledAiryDisk::with_fwhm(airy_disk_pixels, satellite.telescope.corrected_to);
        let detected_stars: Vec<StarDetection> = match detect_stars_unified(
            quantized.view(),
            algorithm,
            &scaled_airy_disk,
            background_rms,
            detection_sigma,
        ) {
            Ok(stars) => stars
                .into_iter()
                .map(|star| {
                    let (x, y) = star.get_centroid();
                    StarDetection {
                        id: 0,
                        x,
                        y,
                        flux: star.flux(),
                        m_xx: 1.0,
                        m_yy: 1.0,
                        m_xy: 0.0,
                        aspect_ratio: 1.0,
                        diameter: 2.0,
                    }
                })
                .collect(),
            Err(_) => {
                no_detections += 1;
                continue;
            }
        };

        // Analyze detections
        if detected_stars.is_empty() {
            no_detections += 1;
        } else if detected_stars.len() > 1 {
            multiple_detections += 1;
            false_positives += 1;
        } else {
            // Single detection - check if it's correct
            let detection = &detected_stars[0];
            let x_diff = detection.x - xpos;
            let y_diff = detection.y - ypos;
            let err = (x_diff * x_diff + y_diff * y_diff).sqrt();

            let airy_disk = satellite.airy_disk_fwhm_sampled();
            if err > airy_disk.first_zero().max(1.0) * 3.0 {
                spurious_detections += 1;
                false_positives += 1;
            } else {
                true_positives += 1;
                total_error += err;
            }
        }
    }

    let avg_error = if true_positives > 0 {
        total_error / true_positives as f64
    } else {
        f64::NAN
    };

    AlgorithmResults {
        algorithm,
        sensor_name: satellite.sensor.name.clone(),
        experiments: args.experiments,
        true_positives,
        false_positives,
        spurious_detections,
        multiple_detections,
        no_detections,
        avg_error_pixels: avg_error,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    println!("=== Star Detection Algorithm Comparison ===");
    println!("Experiments per algorithm: {}", args.experiments);
    println!("Domain size: {}x{} pixels", args.domain, args.domain);
    println!("Test magnitude: {}", args.magnitude);
    println!("PSF disk size: {} FWHM", args.disk_size);
    println!("Detection sigma: {}", args.shared.noise_multiple);
    println!();

    // Create test satellite configurations
    let telescope_config = args.shared.telescope.to_config().clone();
    let test_satellites: Vec<SatelliteConfig> = ALL_SENSORS
        .iter()
        .map(|sensor| {
            let sized_sensor = sensor.with_dimensions(args.domain, args.domain);
            let mut satellite = SatelliteConfig::new(
                telescope_config.clone(),
                sized_sensor,
                Temperature::from_celsius(args.shared.temperature),
            );
            satellite = satellite.with_fwhm_sampling(args.disk_size);
            satellite
        })
        .collect();

    let algorithms = [StarFinder::Naive, StarFinder::Dao, StarFinder::Iraf];

    // Run all combinations in parallel
    let mut all_tests = Vec::new();
    for satellite in &test_satellites {
        for &algorithm in &algorithms {
            all_tests.push((algorithm, satellite.clone()));
        }
    }

    println!("Running {} tests in parallel...", all_tests.len());
    let results: Vec<AlgorithmResults> = all_tests
        .par_iter()
        .map(|(algorithm, satellite)| test_algorithm(*algorithm, satellite, &args))
        .collect();

    // Print results
    println!("\n=== RESULTS ===");
    println!(
        "{:>8} {:>15} {:>6} {:>6} {:>8} {:>6} {:>8} {:>6} {:>8}",
        "Algo",
        "Sensor",
        "True+",
        "False+",
        "TP Rate",
        "FP Rate",
        "Spurious",
        "Multiple",
        "Avg Err"
    );
    println!("{}", "-".repeat(85));

    for result in &results {
        println!(
            "{:>8} {:>15} {:>6} {:>6} {:>7.1}% {:>7.1}% {:>8} {:>8} {:>7.2}",
            format!("{:?}", result.algorithm),
            result.sensor_name,
            result.true_positives,
            result.false_positives,
            result.true_positive_rate() * 100.0,
            result.false_positive_rate() * 100.0,
            result.spurious_detections,
            result.multiple_detections,
            result.avg_error_pixels
        );
    }

    // Summary by algorithm
    println!("\n=== ALGORITHM SUMMARY ===");
    for &algorithm in &algorithms {
        let algo_results: Vec<_> = results
            .iter()
            .filter(|r| r.algorithm == algorithm)
            .collect();
        let total_tp: u32 = algo_results.iter().map(|r| r.true_positives).sum();
        let total_fp: u32 = algo_results.iter().map(|r| r.false_positives).sum();
        let total_experiments: u32 = algo_results.iter().map(|r| r.experiments).sum();

        println!(
            "{:?}: TP Rate: {:.1}%, FP Rate: {:.1}%",
            algorithm,
            total_tp as f64 / total_experiments as f64 * 100.0,
            total_fp as f64 / total_experiments as f64 * 100.0
        );
    }

    Ok(())
}

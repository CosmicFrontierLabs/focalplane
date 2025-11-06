//! Dark frame analysis tool for sensor characterization.
//!
//! Captures multiple dark frames (zero-light exposures) and analyzes them to characterize
//! sensor noise properties including read noise, hot pixels, and variance outliers.
//!
//! Supports both real PlayerOne cameras and mock cameras for testing.

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array2;
use rand::{Rng, SeedableRng};
use shared::{
    bad_pixel_map::BadPixelMap,
    camera_interface::{mock::MockCameraInterface, CameraInterface},
    config_storage::ConfigStorage,
    dark_frame::DarkFrameAnalysis,
};
use simulator::io::fits::{write_typed_fits, FitsDataType};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
use test_bench::poa::camera::PlayerOneCamera;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Dark frame analysis tool for sensor characterization"
)]
struct Args {
    #[arg(short = 'i', long, help = "Camera ID (for PlayerOne cameras)")]
    camera_id: Option<i32>,

    #[arg(
        long,
        help = "Use mock camera with random noise (for testing)",
        conflicts_with = "camera_id"
    )]
    mock: bool,

    #[arg(
        short = 'n',
        long,
        default_value = "50",
        help = "Number of dark frames to capture"
    )]
    num_frames: usize,

    #[arg(
        short = 'o',
        long,
        default_value = "./dark_analysis",
        help = "Output directory for analysis results"
    )]
    output_dir: PathBuf,

    #[arg(
        long,
        default_value = "5.0",
        help = "Sigma threshold for hot pixel detection"
    )]
    hot_pixel_threshold: f64,

    #[arg(
        long,
        default_value = "0.1",
        help = "Variance threshold below which pixels are considered dead"
    )]
    dead_pixel_threshold: f64,

    #[arg(
        long,
        help = "Override camera exposure time (milliseconds)",
        value_name = "MS"
    )]
    exposure_ms: Option<f64>,

    #[arg(
        long,
        help = "Save mean and variance maps as FITS files (requires fitsio)"
    )]
    save_maps: bool,

    #[arg(
        long,
        help = "Save bad pixel map to config store (~/.cf_config/bad_pixel_maps/)"
    )]
    save_to_config_store: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Dark Frame Analysis Tool");
    info!("========================");

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", args.output_dir))?;

    if args.mock {
        info!("Using mock camera for testing");
        let camera = create_mock_camera(args.exposure_ms);
        run_analysis(camera, &args)?;
    } else {
        let camera_id = args
            .camera_id
            .context("Must specify --camera-id or --mock")?;
        info!("Initializing PlayerOne camera with ID {}", camera_id);
        let mut camera = PlayerOneCamera::new(camera_id)
            .map_err(|e| anyhow::anyhow!("Failed to initialize camera: {e}"))?;

        if let Some(exposure_ms) = args.exposure_ms {
            let exposure = Duration::from_micros((exposure_ms * 1000.0) as u64);
            camera
                .set_exposure(exposure)
                .map_err(|e| anyhow::anyhow!("Failed to set exposure: {e}"))?;
            info!("Set exposure to {:.2} ms", exposure_ms);
        }

        run_analysis(camera, &args)?;
    }

    Ok(())
}

fn create_mock_camera(exposure_ms: Option<f64>) -> MockCameraInterface {
    use shared::image_proc::noise::generate::simple_normal_array;

    let width = 1280;
    let height = 960;
    let bit_depth = 12;

    let bias = 100.0;
    let read_noise = 5.0;
    let seed = 42;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let hot_pixel_data: Vec<((usize, usize), u16)> = (0..10)
        .map(|_| {
            let location = (rng.gen_range(0..width), rng.gen_range(0..height));
            let stuck_value = (bias + rng.gen_range(50.0..100.0)) as u16;
            (location, stuck_value)
        })
        .collect();

    let dead_pixel_locations: Vec<(usize, usize)> = (0..5)
        .map(|_| (rng.gen_range(0..width), rng.gen_range(0..height)))
        .collect();

    let mut camera =
        MockCameraInterface::with_generator((width, height), bit_depth, move |params| {
            let noise = simple_normal_array(
                (params.height, params.width),
                0.0,
                read_noise,
                params.frame_number,
            );

            let mut frame_u16 = Array2::from_shape_fn((params.height, params.width), |(y, x)| {
                let value = bias + noise[[y, x]];
                value.clamp(0.0, 4095.0) as u16
            });

            for &((x, y), stuck_value) in &hot_pixel_data {
                frame_u16[[y, x]] = stuck_value;
            }

            for &(x, y) in &dead_pixel_locations {
                frame_u16[[y, x]] = 0;
            }

            frame_u16
        })
        .with_saturation(4095.0);

    if let Some(ms) = exposure_ms {
        let exposure = Duration::from_micros((ms * 1000.0) as u64);
        camera.set_exposure(exposure).unwrap();
    }

    camera
}

fn run_analysis<C: CameraInterface>(mut camera: C, args: &Args) -> Result<()> {
    let geometry = camera.geometry();
    let exposure = camera.get_exposure();
    let bit_depth = camera.get_bit_depth();
    info!("Sensor: {}x{} pixels", geometry.width, geometry.height);
    info!("Exposure: {:?}", exposure);
    info!("Bit depth: {} bits", bit_depth);
    info!("Saturation value: {:.0}", camera.saturation_value());

    let mut analysis = DarkFrameAnalysis::new(geometry.width, geometry.height);

    info!("\nCapturing {} dark frames...", args.num_frames);
    info!("(Ensure lens cap is on or camera is in complete darkness)");

    for i in 0..args.num_frames {
        if (i + 1) % 10 == 0 || i == 0 {
            info!("  Frame {}/{}", i + 1, args.num_frames);
        }

        let (frame, metadata) = camera
            .capture_frame()
            .map_err(|e| anyhow::anyhow!("Frame capture failed: {e}"))?;

        analysis.add_frame(&frame);
        analysis.add_temperature_readings(metadata.temperatures);

        if i < 3 {
            let sum: u64 = frame.iter().map(|&x| x as u64).sum();
            let mean = sum as f64 / frame.len() as f64;
            let min = *frame.iter().min().unwrap();
            let max = *frame.iter().max().unwrap();
            info!(
                "    Frame {} stats: mean={:.1}, min={}, max={}",
                i + 1,
                mean,
                min,
                max
            );
        }
    }

    info!("\nAnalyzing {} frames...", args.num_frames);
    analysis.finalize();

    let report = analysis.generate_report(args.hot_pixel_threshold, args.dead_pixel_threshold);

    print_report(&report);

    let json_path = args.output_dir.join("dark_frame_report.json");
    let json =
        serde_json::to_string_pretty(&report).context("Failed to serialize report to JSON")?;
    fs::write(&json_path, json)
        .with_context(|| format!("Failed to write JSON to {json_path:?}"))?;
    info!("\nReport saved to: {:?}", json_path);

    let viz_path = args.output_dir.join("dark_frame_anomalies.png");
    let viz_img = analysis.visualize_anomalies(&report);
    viz_img
        .save(&viz_path)
        .with_context(|| format!("Failed to save visualization to {viz_path:?}"))?;
    info!("Anomaly visualization saved to: {:?}", viz_path);
    info!("  Red=Stuck, Blue=Dead, Orange=Hot");

    let human_report_path = args.output_dir.join("SENSOR_REPORT.md");
    let human_report = report.generate_human_report();
    fs::write(&human_report_path, human_report)
        .with_context(|| format!("Failed to write human report to {human_report_path:?}"))?;
    info!("Human-readable report saved to: {:?}", human_report_path);

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut bad_pixel_map =
        BadPixelMap::new(camera.name().to_string(), camera.get_serial(), timestamp);

    for ((x, y), _anomaly) in &report.anomalies {
        bad_pixel_map.add_pixel(*x, *y);
    }

    let bad_pixel_path = args
        .output_dir
        .join(format!("bad-pixels-{}.json", camera.get_serial()));
    bad_pixel_map
        .save_to_file(&bad_pixel_path)
        .with_context(|| format!("Failed to save bad pixel map to {bad_pixel_path:?}"))?;
    info!(
        "Bad pixel map saved to: {:?} ({} pixels)",
        bad_pixel_path,
        bad_pixel_map.num_bad_pixels()
    );

    if args.save_to_config_store {
        let config_store =
            ConfigStorage::new().with_context(|| "Failed to initialize config storage")?;
        let saved_path = config_store
            .save_bad_pixel_map(&bad_pixel_map)
            .with_context(|| "Failed to save bad pixel map to config store")?;
        info!("Bad pixel map also saved to config store: {:?}", saved_path);
    }

    if args.save_maps {
        save_fits_maps(&analysis, &args.output_dir)?;
    }

    Ok(())
}

fn print_report(report: &shared::dark_frame::DarkFrameReport) {
    use shared::dark_frame::PixelAnomaly;

    info!("\n{}", "=".repeat(60));
    info!("DARK FRAME ANALYSIS REPORT");
    info!("{}", "=".repeat(60));

    info!("\nSensor Information:");
    info!(
        "  Dimensions: {}x{}",
        report.sensor_width, report.sensor_height
    );
    info!(
        "  Total pixels: {}",
        report.sensor_width * report.sensor_height
    );
    info!("  Frames analyzed: {}", report.num_frames);

    info!("\nGlobal Statistics:");
    info!("  Global mean (bias): {:.2} ADU", report.global_mean);
    info!(
        "  Global std of means: {:.2} ADU",
        report.global_std_of_means
    );

    info!("\nRead Noise Estimates:");
    info!(
        "  Median read noise: {:.3} ADU ({:.3} e⁻ at gain=1)",
        report.median_read_noise, report.median_read_noise
    );
    info!(
        "  Mean read noise: {:.3} ADU ({:.3} e⁻ at gain=1)",
        report.mean_read_noise, report.mean_read_noise
    );

    // Classify anomalies by type
    let mut hot_pixels = Vec::new();
    let mut stuck_pixels = Vec::new();
    let mut dead_pixels = Vec::new();

    for ((x, y), anomaly) in &report.anomalies {
        match anomaly {
            PixelAnomaly::Hot {
                mean,
                std_dev,
                sigma_from_population,
            } => hot_pixels.push((*x, *y, *mean, *std_dev, *sigma_from_population)),
            PixelAnomaly::Stuck {
                mean,
                std_dev,
                sigma_from_population,
            } => stuck_pixels.push((*x, *y, *mean, *std_dev, *sigma_from_population)),
            PixelAnomaly::Dead => dead_pixels.push((*x, *y)),
        }
    }

    // Sort by sigma_from_population (descending)
    hot_pixels.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
    stuck_pixels.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

    info!("\nPixel Anomalies:");
    info!("  Hot pixels: {}", hot_pixels.len());
    if !hot_pixels.is_empty() {
        info!("    Top 5 hottest:");
        for (i, &(x, y, mean, _std_dev, sigma)) in hot_pixels.iter().take(5).enumerate() {
            info!(
                "      {}. ({}, {}) - mean={:.1} ADU ({:.1}σ above population)",
                i + 1,
                x,
                y,
                mean,
                sigma
            );
        }
    }

    info!("  Stuck pixels: {}", stuck_pixels.len());
    if !stuck_pixels.is_empty() {
        info!("    Top 5 (high mean, zero variance):");
        for (i, &(x, y, mean, std_dev, _sigma)) in stuck_pixels.iter().take(5).enumerate() {
            info!(
                "      {}. ({}, {}) - mean={:.1} ADU, std_dev={:.3} ADU",
                i + 1,
                x,
                y,
                mean,
                std_dev
            );
        }
    }

    info!("  Dead pixels: {}", dead_pixels.len());
    if !dead_pixels.is_empty() && dead_pixels.len() <= 10 {
        for (x, y) in &dead_pixels {
            info!("    ({}, {})", x, y);
        }
    }

    if !report.temperature_readings.is_empty() {
        info!("\nTemperature Readings:");
        for (sensor, readings) in &report.temperature_readings {
            if !readings.is_empty() {
                let mean = readings.iter().sum::<f64>() / readings.len() as f64;
                let min = readings.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = readings.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                info!(
                    "  {}: mean={:.1}°C, range=[{:.1}, {:.1}]°C",
                    sensor, mean, min, max
                );
            }
        }
    }

    info!("{}", "=".repeat(60));
}

fn save_fits_maps(analysis: &DarkFrameAnalysis, output_dir: &Path) -> Result<()> {
    info!("\nSaving FITS maps...");

    let fits_path = output_dir.join("dark_frame_maps.fits");

    let mut data = HashMap::new();
    data.insert("MEAN".to_string(), FitsDataType::Float64(analysis.mean()));
    data.insert(
        "VARIANCE".to_string(),
        FitsDataType::Float64(analysis.variance()),
    );

    match write_typed_fits(&data, &fits_path) {
        Ok(_) => info!("  Maps saved to: {:?}", fits_path),
        Err(e) => warn!("  Failed to save FITS maps: {}", e),
    }

    Ok(())
}

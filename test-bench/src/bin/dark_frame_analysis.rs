//! Dark frame analysis tool for sensor characterization.
//!
//! Captures multiple dark frames (zero-light exposures) and analyzes them to characterize
//! sensor noise properties including read noise, hot pixels, and variance outliers.
//!
//! Supports multiple camera types via feature flags:
//! - Mock camera: Always available (for testing)
//! - PlayerOne cameras: Requires "playerone" feature
//! - NSV455 cameras: Requires "nsv455" feature

use anyhow::{Context, Result};
use clap::Parser;
use shared::{
    bad_pixel_map::BadPixelMap, camera_interface::CameraInterface, config_storage::ConfigStorage,
    dark_frame::DarkFrameAnalysis,
};
use simulator::io::fits::{write_typed_fits, FitsDataType};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
use test_bench::camera_init::{initialize_camera, CameraArgs, OptionalExposureArgs};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Analyze dark frames to characterize sensor noise and detect bad pixels",
    long_about = "Dark frame analysis tool for camera sensor characterization.\n\n\
        This tool captures multiple dark frames (exposures with no light) and performs \
        statistical analysis to identify:\n\n  \
        - Hot pixels: Pixels with abnormally high dark current\n  \
        - Dead pixels: Pixels that show no response (zero variance)\n  \
        - Stuck pixels: Pixels locked at a fixed value\n\n\
        The analysis outputs a bad pixel map that can be used by other tools to mask \
        defective pixels during image processing. Results are saved to the config store \
        (~/.cf_config/bad_pixel_maps/) by default for use by other applications."
)]
struct Args {
    #[command(flatten)]
    camera: CameraArgs,

    #[arg(
        short = 'n',
        long,
        default_value = "50",
        help = "Number of dark frames to capture for analysis",
        long_help = "Number of dark frames to capture for statistical analysis. More frames \
            provide better noise characterization but take longer to acquire. A minimum of \
            10 frames is recommended for reliable statistics; 50+ frames are ideal for \
            production calibration."
    )]
    num_frames: usize,

    #[arg(
        short = 'o',
        long,
        default_value = "./dark_analysis",
        help = "Directory for analysis output files",
        long_help = "Output directory for analysis results including:\n  \
            - SENSOR_REPORT.md: Human-readable analysis summary\n  \
            - anomaly_visualization.png: Color-coded bad pixel map image\n  \
            - Mean/variance FITS files (if --save-maps is enabled)\n\n\
            The directory will be created if it does not exist."
    )]
    output_dir: PathBuf,

    #[arg(
        long,
        default_value = "5.0",
        help = "Sigma threshold for hot pixel detection",
        long_help = "Statistical threshold (in standard deviations) above which a pixel's \
            dark current is considered anomalously high. Lower values detect more hot pixels \
            but may produce false positives. Typical range: 3.0-7.0 sigma."
    )]
    hot_pixel_threshold: f64,

    #[arg(
        long,
        default_value = "0.1",
        help = "Variance threshold for dead pixel detection",
        long_help = "Pixels with temporal variance below this threshold are flagged as dead \
            or stuck. Dead pixels show no response to photons; stuck pixels are locked at a \
            constant ADU value. The threshold should be set based on expected read noise."
    )]
    dead_pixel_threshold: f64,

    #[command(flatten)]
    exposure: OptionalExposureArgs,

    #[arg(
        short = 'g',
        long,
        help = "Camera gain/ISO setting",
        long_help = "Analog gain setting for the camera sensor. Higher gain amplifies both \
            signal and noise, potentially revealing more hot pixels. Use the same gain \
            setting you plan to use for actual observations."
    )]
    gain: Option<f64>,

    #[arg(
        long,
        help = "Save mean and variance maps as FITS files",
        long_help = "Export the computed mean and variance maps as FITS files in the output \
            directory. These files can be used for detailed inspection of sensor behavior \
            or imported into other analysis tools. Requires the fitsio feature."
    )]
    save_maps: bool,

    #[arg(
        long,
        default_value = "true",
        help = "Save bad pixel map to system config store",
        long_help = "Save the bad pixel map to the system configuration directory at \
            ~/.cf_config/bad_pixel_maps/. This allows other applications to automatically \
            load and apply the bad pixel map for this specific camera. The map is stored \
            using the camera model and serial number as identifiers. Use --no-save-to-config \
            to disable this behavior."
    )]
    save_to_config: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Dark Frame Analysis Tool");
    info!("========================");

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", args.output_dir))?;

    let mut camera = initialize_camera(&args.camera)?;

    // Apply exposure override if specified
    if let Some(exposure) = args.exposure.as_duration() {
        camera
            .set_exposure(exposure)
            .map_err(|e| anyhow::anyhow!("Failed to set exposure: {e}"))?;
        info!(
            "Set exposure to {:.2} ms",
            args.exposure.exposure_ms.unwrap()
        );
    }

    // Apply gain override if specified
    if let Some(gain) = args.gain {
        camera
            .set_gain(gain)
            .map_err(|e| anyhow::anyhow!("Failed to set gain: {e}"))?;
        info!("Set gain to {}", gain);
    }

    run_analysis(camera.as_mut(), &args)?;

    Ok(())
}

fn run_analysis(camera: &mut dyn CameraInterface, args: &Args) -> Result<()> {
    let geometry = camera.geometry();
    let exposure = camera.get_exposure();
    let bit_depth = camera.get_bit_depth();
    let camera_name = camera.name().to_string();
    let camera_serial = camera.get_serial();

    info!("Camera: {} (serial: {})", camera_name, camera_serial);
    info!("Sensor: {}x{} pixels", geometry.width(), geometry.height());
    info!("Exposure: {:?}", exposure);
    info!("Bit depth: {} bits", bit_depth);
    info!("Saturation value: {:.0}", camera.saturation_value());

    let mut analysis = DarkFrameAnalysis::new(geometry.width(), geometry.height());

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
    let mut bad_pixel_map = BadPixelMap::new(camera_name.clone(), camera_serial.clone(), timestamp);

    for ((x, y), _anomaly) in &report.anomalies {
        bad_pixel_map.add_pixel(*x, *y);
    }

    if args.save_to_config {
        let config_store =
            ConfigStorage::new().with_context(|| "Failed to initialize config storage")?;
        let saved_path = config_store
            .save_bad_pixel_map(&bad_pixel_map)
            .with_context(|| "Failed to save bad pixel map to config store")?;
        info!(
            "Bad pixel map saved to config store: {:?} ({} pixels)",
            saved_path,
            bad_pixel_map.num_bad_pixels()
        );
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
        report.sensor_size.width, report.sensor_size.height
    );
    info!("  Total pixels: {}", report.sensor_size.pixel_count());
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

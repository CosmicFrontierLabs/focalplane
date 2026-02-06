//! Convert stellar magnitudes to expected sensor flux in DN

use clap::Parser;
use plotters::prelude::*;
use shared::image_proc::detection::{detect_stars_unified, StarFinder};
use shared::image_proc::{save_u8_image, stretch_histogram, u16_to_u8_scaled};
use shared::range_arg::RangeArg;
use shared::units::{Temperature, TemperatureExt};
use simulator::hardware::SatelliteConfig;
use simulator::image_proc::render::StarInFrame;
use simulator::photometry::zodiacal::SolarAngularCoordinates;
use simulator::shared_args::{SensorModel, TelescopeModel};
use simulator::star_data_to_fluxes;
use simulator::Scene;
use starfield::catalogs::StarData;
use starfield::Equatorial;
use std::path::Path;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about = "Convert stellar magnitudes to flux")]
struct Args {
    /// Magnitude range (start:stop:step)
    #[arg(long, default_value = "-2:25:0.5")]
    magnitude_range: RangeArg,

    /// Sensor temperature in degrees Celsius
    #[arg(long, default_value_t = -10.0)]
    temperature: f64,

    /// Telescope model
    #[arg(long, default_value_t = TelescopeModel::CosmicFrontierJbt50cm)]
    telescope: TelescopeModel,

    /// Sensor model
    #[arg(long, default_value_t = SensorModel::Hwk4123)]
    sensor: SensorModel,

    /// Domain size for test images (width and height in pixels)
    #[arg(long, default_value_t = 128)]
    domain: usize,

    /// Override automatic linear fit range calculation (optional)
    #[arg(long)]
    fit_mag_start: Option<f64>,

    /// Override automatic linear fit range calculation (optional)
    #[arg(long)]
    fit_mag_end: Option<f64>,

    /// Output plot filename
    #[arg(long, default_value = "plots/magnitude_flux_plot.png")]
    output: String,

    /// Save rendered images to subdirectory
    #[arg(long, default_value_t = false)]
    save_images: bool,

    /// B-V color index for the star (typical range: -0.3 to 2.0)
    /// Examples: -0.3 (blue O-type), 0.0 (white A-type), 0.65 (G-type like Sun), 1.5 (red M-type)
    #[arg(long, default_value_t = 0.0)]
    b_v: f64,
}

/// Results from a single star finder algorithm
struct DetectorResults {
    detector_name: String,
    data_points: Vec<(f64, f64)>,
    fit_range: (f64, f64),
}

/// Create and save the magnitude vs flux plot for all detectors
fn create_plot(
    all_results: &[DetectorResults],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create plots directory if it doesn't exist
    std::fs::create_dir_all("plots")?;

    // Create plot with 2x resolution
    let root = BitMapBackend::new(output_path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find min/max for scaling across all detectors
    let min_mag = all_results
        .iter()
        .flat_map(|r| r.data_points.iter())
        .map(|&(m, _)| m)
        .fold(f64::INFINITY, f64::min);
    let max_mag = all_results
        .iter()
        .flat_map(|r| r.data_points.iter())
        .map(|&(m, _)| m)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_flux = all_results
        .iter()
        .flat_map(|r| r.data_points.iter())
        .map(|&(_, f)| f)
        .fold(f64::INFINITY, f64::min)
        .max(1.0);
    let max_flux = all_results
        .iter()
        .flat_map(|r| r.data_points.iter())
        .map(|&(_, f)| f)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Magnitude vs Flux (DN)", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min_mag..max_mag, (min_flux..max_flux).log_scale())?;

    chart
        .configure_mesh()
        .x_desc("Magnitude")
        .y_desc("Flux (DN) - Log Scale")
        .draw()?;

    // Define colors for each detector
    let detector_colors = [("Dao", &BLUE), ("Iraf", &GREEN), ("Naive", &RED)];

    // Plot data and fits for each detector
    for (idx, results) in all_results.iter().enumerate() {
        let color = detector_colors[idx].1;

        // Plot measured data points
        chart
            .draw_series(PointSeries::of_element(
                results.data_points.iter().map(|&(m, f)| (m, f)),
                5,
                color,
                &|c, s, st| Circle::new(c, s, st.filled()),
            ))?
            .label(&results.detector_name)
            .legend(move |(x, y)| Circle::new((x + 5, y), 3, color.filled()));

        // Calculate linear fit for this detector
        let fit_points: Vec<(f64, f64)> = results
            .data_points
            .iter()
            .filter(|&&(m, f)| m >= results.fit_range.0 && m <= results.fit_range.1 && f > 0.0)
            .copied()
            .collect();

        if !fit_points.is_empty() {
            // Calculate linear regression in log space
            let n = fit_points.len() as f64;
            let sum_x: f64 = fit_points.iter().map(|(m, _)| m).sum();
            let sum_y: f64 = fit_points.iter().map(|(_, f)| f.log10()).sum();
            let sum_xy: f64 = fit_points.iter().map(|(m, f)| m * f.log10()).sum();
            let sum_x2: f64 = fit_points.iter().map(|(m, _)| m * m).sum();

            // Linear regression formulas
            let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n;

            // Create fitted line endpoints
            let fit_min_mag = fit_points
                .iter()
                .map(|(m, _)| *m)
                .fold(f64::INFINITY, f64::min);
            let fit_max_mag = fit_points
                .iter()
                .map(|(m, _)| *m)
                .fold(f64::NEG_INFINITY, f64::max);

            let pogson_points = vec![
                (fit_min_mag, 10_f64.powf(slope * fit_min_mag + intercept)),
                (fit_max_mag, 10_f64.powf(slope * fit_max_mag + intercept)),
            ];

            chart
                .draw_series(LineSeries::new(pogson_points, &color.mix(0.7)))?
                .label(format!(
                    "{} fit (slope={:.3})",
                    results.detector_name, slope
                ))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color.mix(0.7)));
        }
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("\nPlot saved to {output_path}");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // List of star detection algorithms to test
    let detectors = vec![StarFinder::Dao, StarFinder::Iraf, StarFinder::Naive];

    let mut all_results = Vec::new();

    // Create satellite configuration (same for all detectors)
    let telescope = args.telescope.to_config().clone();
    let sensor = args
        .sensor
        .to_config()
        .with_dimensions(args.domain, args.domain);
    let temperature = Temperature::from_celsius(args.temperature);
    let satellite_config = SatelliteConfig::new(telescope, sensor, temperature);

    // Run analysis for each detector
    for detector in detectors {
        println!("\n=== Testing detector: {detector:?} ===");

        let (start, stop, step) = args.magnitude_range.as_tuple();
        let mut magnitude = start;
        let mut data_points: Vec<(f64, f64)> = Vec::new();

        while if step > 0.0 {
            magnitude <= stop
        } else {
            magnitude >= stop
        } {
            // Create star at center of domain with given magnitude and B-V color
            let star_data = StarData {
                id: 0,
                position: Equatorial::from_degrees(0.0, 0.0),
                magnitude,
                b_v: Some(args.b_v),
            };

            // Position star at center of domain
            let xpos = args.domain as f64 / 2.0;
            let ypos = args.domain as f64 / 2.0;

            let star_in_frame = StarInFrame {
                star: star_data,
                x: xpos,
                y: ypos,
                spot: star_data_to_fluxes(&star_data, &satellite_config),
            };

            // Create scene with single star
            let scene = Scene::from_stars(
                satellite_config.clone(),
                vec![star_in_frame],
                Equatorial::from_degrees(0.0, 0.0), // Dummy pointing
                SolarAngularCoordinates::zodiacal_minimum(),
            );

            // Render the scene
            let exposure = Duration::from_secs(1);
            let render_result = scene.render_with_seed(&exposure, Some(42));

            // Save image if requested
            if args.save_images {
                save_rendered_image(
                    &render_result.quantized_image,
                    magnitude,
                    &args.output,
                    satellite_config.sensor.bit_depth,
                )?;
            }

            // Run star detection
            let background_rms = render_result.background_rms();
            let detected_stars = detect_stars_unified(
                render_result.quantized_image.view(),
                detector,
                &satellite_config.airy_disk_fwhm_sampled(),
                background_rms,
                5.0, // noise multiplier
            )?;

            // Find the star closest to center (should be our star)
            let detected_flux = if !detected_stars.is_empty() {
                let center_x = args.domain as f64 / 2.0;
                let center_y = args.domain as f64 / 2.0;

                let mut closest_star = None;
                let mut min_distance = f64::INFINITY;

                for star in &detected_stars {
                    let (x, y) = star.get_centroid();
                    let distance = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();
                    if distance < min_distance {
                        min_distance = distance;
                        closest_star = Some(star);
                    }
                }

                if let Some(star) = closest_star {
                    star.flux()
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let star_flux = render_result.star_image.sum();
            let ratio = if star_flux > 0.0 {
                detected_flux / star_flux
            } else {
                0.0
            };
            println!(
            "Magnitude: {magnitude:.2}, Rendered flux: {star_flux:.2}, Detected flux: {detected_flux:.2}, Ratio: {ratio:.4}, Detected stars: {}",
            detected_stars.len()
        );

            if detected_flux > 0.0 {
                data_points.push((magnitude, detected_flux));
            }

            magnitude += step;
        }

        // Calculate the linear fit range for this detector
        // Use faintest detected magnitude and go brighter by bit_depth/3 magnitudes
        let faintest_detected = data_points
            .iter()
            .map(|(mag, _)| *mag)
            .fold(f64::NEG_INFINITY, f64::max);

        let magnitude_range = satellite_config.sensor.bit_depth as f64 / 3.0;

        // Use command line args if provided, otherwise calculate automatically
        let fit_start = args
            .fit_mag_start
            .unwrap_or(faintest_detected - magnitude_range);
        let fit_end = args.fit_mag_end.unwrap_or(faintest_detected);

        println!(
            "Using linear fit range: {:.1} to {:.1} (bit depth: {})",
            fit_start, fit_end, satellite_config.sensor.bit_depth
        );

        // Store results for this detector
        all_results.push(DetectorResults {
            detector_name: format!("{detector:?}"),
            data_points,
            fit_range: (fit_start, fit_end),
        });
    }

    // Create the combined plot
    create_plot(&all_results, &args.output)?;

    Ok(())
}

/// Save rendered star image to disk
fn save_rendered_image(
    image: &ndarray::Array2<u16>,
    magnitude: f64,
    output_base: &str,
    bit_depth: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create subdirectory based on output path
    let output_path = Path::new(output_base);
    let parent_dir = output_path.parent().unwrap_or(Path::new("."));
    let images_dir = parent_dir.join("magnitude_images");
    std::fs::create_dir_all(&images_dir)?;

    // Calculate max value from sensor bit depth
    let max_value = (1u32 << bit_depth) - 1;

    // Convert to u8 for PNG output
    let u8_image = u16_to_u8_scaled(image, max_value);

    // Save regular image
    let filename = format!("mag_{magnitude:.1}.png");
    let image_path = images_dir.join(&filename);
    save_u8_image(&u8_image, &image_path)?;

    // Also save stretched version for better visibility
    let stretched_u16 = stretch_histogram(image.view(), 0.02, 0.98);
    let stretched_u8 = u16_to_u8_scaled(&stretched_u16, max_value);
    let stretched_filename = format!("mag_{magnitude:.1}_stretched.png");
    let stretched_path = images_dir.join(&stretched_filename);
    save_u8_image(&stretched_u8, &stretched_path)?;

    println!("  Saved images: {filename} and {stretched_filename}");

    Ok(())
}

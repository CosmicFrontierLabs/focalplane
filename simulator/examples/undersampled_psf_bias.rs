//! Demonstrates systematic centroid bias for undersampled PSFs
//!
//! This example shows how centroiding accuracy degrades when the PSF is undersampled,
//! leading to systematic biases that depend on the sub-pixel position of the star.

use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::Array2;
use plotters::prelude::*;
use rayon::prelude::*;
use simulator::hardware::sensor::models::IMX455;
use simulator::hardware::telescope::models::DEMO_50CM;
use simulator::hardware::SatelliteConfig;
use simulator::image_proc::detection::{detect_stars_unified, StarFinder};
use simulator::image_proc::render::StarInFrame;
use simulator::photometry::zodical::SolarAngularCoordinates;
use simulator::scene::Scene;
use simulator::star_data_to_electrons;
use starfield::catalogs::StarData;
use starfield::Equatorial;
use std::error::Error;
use std::time::Duration;

/// Command line arguments for undersampled PSF bias demo
#[derive(Parser, Debug)]
#[command(
    name = "Undersampled PSF Bias Demo",
    about = "Demonstrates systematic centroid bias for undersampled PSFs"
)]
struct Args {
    /// Grid size for sub-pixel positions (e.g., 32 means 32x32 grid within each pixel)
    #[arg(long, default_value_t = 32)]
    grid_size: usize,

    /// Number of trials per position for statistics
    #[arg(long, default_value_t = 100)]
    trials: u32,
}

/// Parameters for a single bias experiment
#[derive(Clone)]
struct BiasExperimentParams {
    /// Center position of image
    center: f64,
    /// Sub-pixel offsets
    x_offset: f64,
    y_offset: f64,
    /// Satellite configuration
    satellite: SatelliteConfig,
    /// Star magnitude
    magnitude: f64,
    /// Exposure duration
    exposure: Duration,
    /// Number of trials
    num_trials: u32,
    /// Solar coordinates
    coordinates: SolarAngularCoordinates,
    /// Noise floor multiplier
    noise_floor_multiplier: f64,
    /// Random seed
    seed: u64,
}

/// Results from a bias experiment
struct BiasExperimentResults {
    /// Mean centroid error
    mean_error: f64,
    /// Mean X bias
    mean_x_bias: f64,
    /// Mean Y bias
    mean_y_bias: f64,
    /// Number of successful detections
    detections: usize,
}

/// Run a single bias experiment
fn run_bias_experiment(params: &BiasExperimentParams) -> BiasExperimentResults {
    let mut errors = Vec::new();
    let mut x_biases = Vec::new();
    let mut y_biases = Vec::new();

    let airy_disk = params.satellite.airy_disk_fwhm_sampled();

    for trial in 0..params.num_trials {
        // Place star at center pixel + subpixel offset
        let true_x = params.center + params.x_offset - 0.5;
        let true_y = params.center + params.y_offset - 0.5;

        let star = create_star_at_position(
            true_x,
            true_y,
            params.magnitude,
            &params.exposure,
            &params.satellite,
        );

        // Create scene and render
        let scene = Scene::from_stars(
            params.satellite.clone(),
            vec![star],
            Equatorial::from_degrees(0.0, 0.0), // Dummy pointing
            params.exposure,
            params.coordinates,
        );
        let render_result = scene.render_with_seed(Some(params.seed + trial as u64));

        // Calculate background RMS
        let background_rms = render_result.background_rms();

        // Find star using naive detection
        match detect_stars_unified(
            render_result.quantized_image.view(),
            StarFinder::Naive,
            &airy_disk,
            background_rms,
            params.noise_floor_multiplier,
        ) {
            Ok(detections) => {
                if detections.len() == 1 {
                    // Get centroid from detection
                    let (measured_x, measured_y) = detections[0].get_centroid();

                    // Calculate error
                    let x_error = measured_x - true_x;
                    let y_error = measured_y - true_y;
                    let total_error = (x_error * x_error + y_error * y_error).sqrt();

                    errors.push(total_error);
                    x_biases.push(x_error);
                    y_biases.push(y_error);
                }
            }
            Err(_) => {
                // Detection failed, skip this trial
            }
        }
    }

    // Calculate statistics
    if errors.is_empty() {
        BiasExperimentResults {
            mean_error: 0.0,
            mean_x_bias: 0.0,
            mean_y_bias: 0.0,
            detections: 0,
        }
    } else {
        BiasExperimentResults {
            mean_error: errors.iter().sum::<f64>() / errors.len() as f64,
            mean_x_bias: x_biases.iter().sum::<f64>() / x_biases.len() as f64,
            mean_y_bias: y_biases.iter().sum::<f64>() / y_biases.len() as f64,
            detections: errors.len(),
        }
    }
}

/// Generate a grid of star positions across a pixel
fn generate_subpixel_grid(grid_size: usize) -> Vec<(f64, f64)> {
    let mut positions = Vec::new();
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x_offset = i as f64 / (grid_size - 1) as f64;
            let y_offset = j as f64 / (grid_size - 1) as f64;
            positions.push((x_offset, y_offset));
        }
    }
    positions
}

/// Create a StarInFrame at given position with specified magnitude
fn create_star_at_position(
    x: f64,
    y: f64,
    magnitude: f64,
    exposure: &Duration,
    satellite: &SatelliteConfig,
) -> StarInFrame {
    let star_data = StarData {
        id: 0,
        position: Equatorial::from_degrees(0.0, 0.0),
        magnitude,
        b_v: None,
    };

    let flux = star_data_to_electrons(&star_data, exposure, satellite);

    StarInFrame {
        star: star_data,
        x,
        y,
        flux,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args = Args::parse();

    println!("Undersampled PSF Centroid Bias Demo");
    println!("===================================\n");

    // Image parameters
    let image_size = 16;
    let center = image_size as f64 / 2.0;

    // Telescope and sensor setup - use DEMO_50CM from models
    let telescope = DEMO_50CM.clone();
    let sensor = IMX455.with_dimensions(image_size, image_size);
    let wavelength = 550.0; // nm (green light)
    let temperature = 0.0; // 0°C in sensor's expected range

    // Test parameters
    let psf_sampling_values = vec![0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]; // In units of FWHM/pixel, steps of 0.25
    let grid_size = args.grid_size; // Grid size from CLI args
    let magnitude = 12.0; // Brighter star for better SNR
    let exposure = Duration::from_millis(1000); // Longer exposure
    let num_trials = args.trials; // Number of trials from CLI args
    let noise_floor_multiplier = 3.0; // More sensitive detection

    // Solar coordinates (minimum zodiacal background)
    let coordinates = SolarAngularCoordinates::zodiacal_minimum();

    // Storage for results
    let mut all_results = Vec::new();

    println!("Testing PSF sampling values: {:?}", psf_sampling_values);
    println!("Grid size: {}x{} positions per pixel", grid_size, grid_size);
    println!("Trials per position: {}\n", num_trials);

    // Generate all experiment parameters
    let mut all_experiments = Vec::new();
    let positions = generate_subpixel_grid(grid_size);

    for &psf_sampling in &psf_sampling_values {
        // Create satellite config with specified PSF sampling
        let base_satellite =
            SatelliteConfig::new(telescope.clone(), sensor.clone(), temperature, wavelength);
        let satellite = base_satellite.with_fwhm_sampling(psf_sampling);

        // Create experiment for each position
        for (idx, &(x_offset, y_offset)) in positions.iter().enumerate() {
            let params = BiasExperimentParams {
                center,
                x_offset,
                y_offset,
                satellite: satellite.clone(),
                magnitude,
                exposure,
                num_trials,
                coordinates,
                noise_floor_multiplier,
                seed: 42 + (idx as u64 * 1000), // Unique seed per position
            };
            all_experiments.push((psf_sampling, idx, params));
        }
    }

    println!(
        "Running {} experiments in parallel...",
        all_experiments.len()
    );

    // Setup progress tracking
    let multi_progress = MultiProgress::new();
    let progress_style = ProgressStyle::default_bar()
        .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ ");

    let pb = multi_progress.add(ProgressBar::new(all_experiments.len() as u64));
    pb.set_style(progress_style);
    pb.set_message("Running bias experiments");

    // Run experiments in parallel
    let results: Vec<_> = all_experiments
        .par_iter()
        .map(|(psf_sampling, idx, params)| {
            let result = run_bias_experiment(params);
            pb.inc(1);
            (*psf_sampling, *idx, result)
        })
        .collect();

    pb.finish_with_message("Experiments complete!");

    // Process results into arrays for each PSF sampling
    for &psf_sampling in &psf_sampling_values {
        println!("\nTesting PSF sampling = {:.2} FWHM/pixel", psf_sampling);

        // Create satellite just to get FWHM info
        let base_satellite =
            SatelliteConfig::new(telescope.clone(), sensor.clone(), temperature, wavelength);
        let satellite = base_satellite.with_fwhm_sampling(psf_sampling);
        let airy_disk = satellite.airy_disk_fwhm_sampled();
        let psf_fwhm_pixels = airy_disk.fwhm();
        println!("  Actual PSF FWHM: {:.3} pixels", psf_fwhm_pixels);

        // Storage for this sampling value
        let mut sampling_results = Array2::<f64>::zeros((grid_size, grid_size));
        let mut sampling_x_bias = Array2::<f64>::zeros((grid_size, grid_size));
        let mut sampling_y_bias = Array2::<f64>::zeros((grid_size, grid_size));

        // Fill arrays from results
        for &(result_psf, idx, ref result) in &results {
            if (result_psf - psf_sampling).abs() < 1e-6 {
                let grid_x = idx % grid_size;
                let grid_y = idx / grid_size;

                sampling_results[[grid_y, grid_x]] = result.mean_error;
                sampling_x_bias[[grid_y, grid_x]] = result.mean_x_bias;
                sampling_y_bias[[grid_y, grid_x]] = result.mean_y_bias;

                // Debug output for first position
                if idx == 0 && psf_sampling == psf_sampling_values[0] {
                    println!("    Debug - First position: detections={}, mean_error={:.6}, x_bias={:.6}, y_bias={:.6}", 
                             result.detections, result.mean_error, result.mean_x_bias, result.mean_y_bias);
                }
            }
        }

        // Calculate statistics
        let max_error = sampling_results.iter().cloned().fold(0.0f64, f64::max);
        let min_error = sampling_results
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let mean_error = sampling_results.mean().unwrap_or(0.0);

        println!("  Error statistics (pixels):");
        println!(
            "    Min: {:.4}, Max: {:.4}, Mean: {:.4}",
            min_error, max_error, mean_error
        );
        println!("    Range: {:.4} pixels\n", max_error - min_error);

        all_results.push((
            psf_sampling,
            sampling_results,
            sampling_x_bias,
            sampling_y_bias,
        ));
    }

    // Create visualization
    println!("Creating visualization...");

    let root =
        BitMapBackend::new("plots/undersampled_psf_bias.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find maximum error across all results for Y-axis scaling
    let max_error = all_results
        .iter()
        .flat_map(|(_, results, _, _)| results.iter())
        .cloned()
        .fold(0.0f64, f64::max);

    let y_max = max_error + 0.01; // Pad by 0.01 above maximum

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Centroid Error vs Sub-pixel Position for Different PSF Sampling",
            ("sans-serif", 30),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..1.0, 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Sub-pixel X Position")
        .y_desc("Mean Centroid Error (pixels)")
        .draw()?;

    // Plot results for different sampling values
    // Create color interpolation from red (undersampled) to green (well-sampled)
    let mut colors = Vec::new();
    for i in 0..psf_sampling_values.len() {
        let t = i as f64 / (psf_sampling_values.len() - 1) as f64;
        // Interpolate from red to green
        let r = (255.0 * (1.0 - t)) as u8;
        let g = (255.0 * t) as u8;
        let b = 0u8;
        colors.push(RGBColor(r, g, b));
    }

    for (idx, &(psf_sampling, ref results, _, _)) in all_results.iter().enumerate() {
        let color = &colors[idx];

        // Extract center row for plotting
        let center_row = grid_size / 2;
        let mut plot_data = Vec::new();

        for x in 0..grid_size {
            let x_pos = x as f64 / (grid_size - 1) as f64;
            let error = results[[center_row, x]];
            plot_data.push((x_pos, error));
        }

        let color_clone = color.clone();
        chart
            .draw_series(LineSeries::new(plot_data.clone(), color.clone()))?
            .label(format!("PSF = {:.2} FWHM/px", psf_sampling))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color_clone));

        // Add points
        chart.draw_series(PointSeries::of_element(
            plot_data,
            3,
            color.filled(),
            &|c, s, st| {
                return EmptyElement::at(c) + Circle::new((0, 0), s, st);
            },
        ))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    println!("\nVisualization saved to 'plots/undersampled_psf_bias.png'");

    // Create heatmaps for all PSF sampling values
    println!("\nCreating bias heatmaps for all PSF sampling values...");

    for (idx, (psf_sampling, _, x_bias, y_bias)) in all_results.iter().enumerate() {
        let filename = format!(
            "plots/psf_bias_heatmap_{:.2}_fwhm_per_pixel.png",
            psf_sampling
        );

        let root_heatmap = BitMapBackend::new(&filename, (1000, 400)).into_drawing_area();
        root_heatmap.fill(&WHITE)?;

        let areas = root_heatmap.split_evenly((1, 2));
        let areas_x = &areas[0];
        let areas_y = &areas[1];

        // X bias heatmap
        let mut chart_x = ChartBuilder::on(&areas_x)
            .caption(
                format!("X Centroid Bias (PSF = {:.2} FWHM/px)", psf_sampling),
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0..grid_size, 0..grid_size)?;

        chart_x
            .configure_mesh()
            .x_desc("Sub-pixel X (grid)")
            .y_desc("Sub-pixel Y (grid)")
            .draw()?;

        // Find max bias for adaptive scaling, but keep minimum scale for visibility
        let max_abs_bias = x_bias.iter().cloned().map(f64::abs).fold(0.0f64, f64::max);
        let scale_limit = max_abs_bias.max(0.05); // Minimum scale of 0.05 for visibility

        chart_x.draw_series(x_bias.indexed_iter().map(|((y, x), &value)| {
            let clamped_value = value.clamp(-scale_limit, scale_limit);

            // Blue for negative, red for positive, white at zero
            let color = if value < 0.0 {
                let intensity = (value.abs() / scale_limit).min(1.0);
                let blue_val = (intensity * 200.0 + 55.0) as u8; // Range 55-255 for better visibility
                RGBColor(255 - blue_val, 255 - blue_val, 255)
            } else if value > 0.0 {
                let intensity = (value.abs() / scale_limit).min(1.0);
                let red_val = (intensity * 200.0 + 55.0) as u8; // Range 55-255 for better visibility
                RGBColor(255, 255 - red_val, 255 - red_val)
            } else {
                RGBColor(255, 255, 255) // White for zero
            };
            Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
        }))?;

        // Y bias heatmap
        let mut chart_y = ChartBuilder::on(&areas_y)
            .caption(
                format!("Y Centroid Bias (PSF = {:.2} FWHM/px)", psf_sampling),
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0..grid_size, 0..grid_size)?;

        chart_y
            .configure_mesh()
            .x_desc("Sub-pixel X (grid)")
            .y_desc("Sub-pixel Y (grid)")
            .draw()?;

        // Find max bias for Y adaptive scaling
        let max_abs_bias_y = y_bias.iter().cloned().map(f64::abs).fold(0.0f64, f64::max);
        let scale_limit_y = max_abs_bias_y.max(0.05); // Minimum scale of 0.05 for visibility

        chart_y.draw_series(y_bias.indexed_iter().map(|((y, x), &value)| {
            let clamped_value = value.clamp(-scale_limit_y, scale_limit_y);

            // Blue for negative, red for positive, white at zero
            let color = if value < 0.0 {
                let intensity = (value.abs() / scale_limit_y).min(1.0);
                let blue_val = (intensity * 200.0 + 55.0) as u8; // Range 55-255 for better visibility
                RGBColor(255 - blue_val, 255 - blue_val, 255)
            } else if value > 0.0 {
                let intensity = (value.abs() / scale_limit_y).min(1.0);
                let red_val = (intensity * 200.0 + 55.0) as u8; // Range 55-255 for better visibility
                RGBColor(255, 255 - red_val, 255 - red_val)
            } else {
                RGBColor(255, 255, 255) // White for zero
            };
            Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
        }))?;

        root_heatmap.present()?;

        println!("  Saved heatmap: {}", filename);
    }

    println!("\nAll bias heatmaps saved to 'plots/' directory");

    println!("\nDemo complete!");
    println!("\nKey findings:");
    println!("- Undersampled PSFs (< 2 FWHM/pixel) show systematic centroid bias");
    println!("- Bias depends on sub-pixel position of the star");
    println!("- Well-sampled PSFs (> 2 FWHM/pixel) have minimal position-dependent bias");

    Ok(())
}

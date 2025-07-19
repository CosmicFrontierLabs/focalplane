//! Airy Disk approximation demonstration and comparison example.
//!
//! This example demonstrates the theoretical Airy disk pattern produced by a perfect circular aperture
//! and compares it with two practical approximations: Gaussian and Triangle functions.
//!
//! The Airy disk is the diffraction pattern of light from a point source as it passes through
//! a circular aperture. In stellar imaging, this represents the fundamental limit of angular
//! resolution for a telescope.
//!
//! # Outputs
//!
//! This example generates:
//! - Console output with numerical comparison of approximation quality
//! - 1D comparison plot showing all three functions overlaid
//! - 2D grayscale images of each approximation with marked FWHM and first zero rings
//!
//! All plots are saved to the `plots/` directory.

use plotters::prelude::*;
use simulator::image_proc::AiryDisk;
use std::path::Path;

/// Type of Airy disk approximation to render in 2D images.
#[derive(Clone, Copy)]
enum ImageType {
    /// Exact theoretical Airy disk pattern using Bessel functions
    Exact,
    /// Gaussian approximation - simpler computation, good near center
    Gaussian,
    /// Triangle approximation - simplest computation, adequate for many uses
    Triangle,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Airy Disk Approximation Demonstration");
    println!("====================================");
    println!();

    let airy_disk = AiryDisk::new();

    println!("Parameters:");
    println!(
        "  First zero location (r₀): {:.4} radians",
        airy_disk.first_zero
    );
    println!(
        "  Full-width-half-maximum (FWHM): {:.4} radians",
        airy_disk.fwhm
    );

    // Debug: Check intensity at FWHM/2
    let fwhm_half = airy_disk.fwhm / 2.0;
    let intensity_at_fwhm_half = airy_disk.intensity(fwhm_half);
    println!("  Debug: Intensity at FWHM/2 ({fwhm_half:.4}): {intensity_at_fwhm_half:.6}");
    println!();

    // Generate comparison samples with high resolution
    let num_points = 1000;
    let (radii, exact, gaussian, triangle) = airy_disk.generate_comparison_samples(num_points);

    // Calculate approximation errors
    let gaussian_mse = AiryDisk::calculate_mse(&exact, &gaussian);
    let triangle_mse = AiryDisk::calculate_mse(&exact, &triangle);

    // Calculate total summed absolute errors
    let gaussian_total_error: f64 = exact
        .iter()
        .zip(gaussian.iter())
        .map(|(e, g)| (e - g).abs())
        .sum();
    let triangle_total_error: f64 = exact
        .iter()
        .zip(triangle.iter())
        .map(|(e, t)| (e - t).abs())
        .sum();
    let approximations_total_error: f64 = gaussian
        .iter()
        .zip(triangle.iter())
        .map(|(g, t)| (g - t).abs())
        .sum();

    println!("Approximation Quality:");
    println!("  Gaussian MSE: {gaussian_mse:.6}");
    println!("  Triangle MSE: {triangle_mse:.6}");
    println!("  Gaussian Total Summed Error: {gaussian_total_error:.6}");
    println!("  Triangle Total Summed Error: {triangle_total_error:.6}");
    println!("  Gaussian vs Triangle Total Summed Error: {approximations_total_error:.6}");
    println!();

    // Find maximum errors for each approximation
    let (gauss_max_err, gauss_max_r, gauss_5pct_r) =
        AiryDisk::find_max_error(&radii, &exact, &gaussian);
    let (tri_max_err, tri_max_r, tri_5pct_r) = AiryDisk::find_max_error(&radii, &exact, &triangle);
    let (approx_max_err, approx_max_r, approx_5pct_r) =
        AiryDisk::find_max_error(&radii, &gaussian, &triangle);

    println!("Gaussian Approximation Error Analysis:");
    println!("  Maximum error: {gauss_max_err:.4} at r = {gauss_max_r:.4}");
    if let Some(r) = gauss_5pct_r {
        println!(
            "  First 5% error at: r = {:.4} ({:.1}% of r₀)",
            r,
            100.0 * r / airy_disk.first_zero
        );
    }
    println!();

    println!("Triangle Approximation Error Analysis:");
    println!("  Maximum error: {tri_max_err:.4} at r = {tri_max_r:.4}");
    if let Some(r) = tri_5pct_r {
        println!(
            "  First 5% error at: r = {:.4} ({:.1}% of r₀)",
            r,
            100.0 * r / airy_disk.first_zero
        );
    }
    println!();

    println!("Gaussian vs Triangle Approximation Difference:");
    println!("  Maximum difference: {approx_max_err:.4} at r = {approx_max_r:.4}");
    if let Some(r) = approx_5pct_r {
        println!(
            "  First 5% difference at: r = {:.4} ({:.1}% of r₀)",
            r,
            100.0 * r / airy_disk.first_zero
        );
    }
    println!();

    // Display some key values at important points
    println!("Function Values at Key Points:");
    println!("  At r = 0 (center):");
    println!("    Exact: {:.6}", airy_disk.intensity(0.0));
    println!("    Gaussian: {:.6}", airy_disk.gaussian_approximation(0.0));
    println!("    Triangle: {:.6}", airy_disk.triangle_approximation(0.0));

    let r_half = airy_disk.first_zero * 0.5;
    println!("  At r = r₀/2 ({r_half:.4}):");
    println!("    Exact: {:.6}", airy_disk.intensity(r_half));
    println!(
        "    Gaussian: {:.6}",
        airy_disk.gaussian_approximation(r_half)
    );
    println!(
        "    Triangle: {:.6}",
        airy_disk.triangle_approximation(r_half)
    );

    println!("  At r = r₀ ({:.4}) - first zero:", airy_disk.first_zero);
    println!(
        "    Exact: {:.6}",
        airy_disk.intensity(airy_disk.first_zero)
    );
    println!(
        "    Gaussian: {:.6}",
        airy_disk.gaussian_approximation(airy_disk.first_zero)
    );
    println!(
        "    Triangle: {:.6}",
        airy_disk.triangle_approximation(airy_disk.first_zero)
    );

    let r_1_5 = airy_disk.first_zero * 1.5;
    println!("  At r = 1.5×r₀ ({r_1_5:.4}):");
    println!("    Exact: {:.6}", airy_disk.intensity(r_1_5));
    println!(
        "    Gaussian: {:.6}",
        airy_disk.gaussian_approximation(r_1_5)
    );
    println!(
        "    Triangle: {:.6}",
        airy_disk.triangle_approximation(r_1_5)
    );
    println!();

    // Create plots directory if it doesn't exist
    if !Path::new("plots").exists() {
        std::fs::create_dir("plots")?;
    }

    // Create a proper plot using plotters
    let plot_path = "plots/airy_disk_comparison.png";
    create_airy_comparison_plot(&radii, &exact, &gaussian, &triangle, &airy_disk, plot_path)?;

    println!("Plot saved to: {plot_path}");

    // Create 2D Airy disk comparison images
    let exact_path = "plots/airy_disk_exact.png";
    let gaussian_path = "plots/airy_disk_gaussian.png";
    let triangle_path = "plots/airy_disk_triangle.png";

    create_airy_disk_image(&airy_disk, exact_path, ImageType::Exact)?;
    create_airy_disk_image(&airy_disk, gaussian_path, ImageType::Gaussian)?;
    create_airy_disk_image(&airy_disk, triangle_path, ImageType::Triangle)?;

    println!("2D Airy disk images saved:");
    println!("  Exact: {exact_path}");
    println!("  Gaussian: {gaussian_path}");
    println!("  Triangle: {triangle_path}");
    println!();

    println!("Summary:");
    println!(
        "  • The first zero (dark ring) occurs at r₀ = {:.4} radians",
        airy_disk.first_zero
    );
    println!("  • Gaussian approximation works best near the center (MSE: {gaussian_mse:.6})");
    println!("  • Triangle approximation is simpler but less accurate (MSE: {triangle_mse:.6})");
    println!(
        "  • Total summed error: Gaussian = {gaussian_total_error:.3}, Triangle = {triangle_total_error:.3}, Difference = {approximations_total_error:.3}"
    );
    println!("  • Both approximations break down significantly beyond r₀");

    Ok(())
}

/// Creates a detailed 1D comparison plot of the Airy disk and its approximations.
///
/// This function generates a symmetric plot showing the intensity profiles of the exact
/// Airy disk pattern alongside Gaussian and Triangle approximations. The plot includes:
///
/// - Exact Airy disk (blue line) - theoretical pattern using Bessel functions
/// - Gaussian approximation (red line) - computationally efficient, good near center
/// - Triangle approximation (green line) - simplest approximation
/// - Vertical markers at first zero (r₀) and FWHM positions
///
/// # Arguments
///
/// * `radii` - Array of radial distances for evaluation
/// * `exact` - Exact Airy disk intensity values
/// * `gaussian` - Gaussian approximation intensity values
/// * `triangle` - Triangle approximation intensity values
/// * `airy_disk` - AiryDisk instance containing parameters
/// * `save_path` - File path where the plot image should be saved
///
/// # Returns
///
/// Returns `Ok(())` on successful plot generation, or an error if file I/O fails.
///
/// # Examples
///
/// ```rust,no_run
/// use simulator::image_proc::AiryDisk;
///
/// let airy = AiryDisk::new();
/// let (radii, exact, gauss, tri) = airy.generate_comparison_samples(1000);
/// create_airy_comparison_plot(&radii, &exact, &gauss, &tri, &airy, "plot.png")?;
/// ```
fn create_airy_comparison_plot(
    radii: &[f64],
    exact: &[f64],
    gaussian: &[f64],
    triangle: &[f64],
    airy_disk: &AiryDisk,
    save_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Calculate normalized radii (symmetric around 0)
    let max_r_normalized = 2.0;
    let normalized_radii: Vec<f64> = radii.iter().map(|&r| r / airy_disk.first_zero).collect();

    // Set up the plot with 4:3 aspect ratio
    let root = BitMapBackend::new(save_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(20, 20, 20, 20);

    // Single panel: Function comparison
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Airy Disk vs Approximations",
            ("sans-serif", 32).into_font().color(&BLACK),
        )
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(-max_r_normalized..max_r_normalized, 0.0..1.0)?;

    chart
        .configure_mesh()
        .x_desc("Normalized Radius (r/r₀)")
        .y_desc("Normalized Intensity")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 16))
        .y_max_light_lines(10)
        .draw()?;

    // Create symmetric data points (positive and negative r)
    let symmetric_points = |data: &[f64]| {
        let mut points = Vec::new();
        // Add negative side (reversed order for proper line drawing)
        for (i, &r) in normalized_radii.iter().enumerate().rev() {
            if r > 0.0 {
                points.push((-r, data[i]));
            }
        }
        // Add positive side
        for (i, &r) in normalized_radii.iter().enumerate() {
            points.push((r, data[i]));
        }
        points
    };

    // Plot the exact Airy disk function
    chart
        .draw_series(LineSeries::new(symmetric_points(exact), BLUE))?
        .label("Exact Airy Disk")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));

    // Plot the Gaussian approximation
    chart
        .draw_series(LineSeries::new(symmetric_points(gaussian), RED))?
        .label("Gaussian Approximation")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));

    // Plot the Triangle approximation
    chart
        .draw_series(LineSeries::new(symmetric_points(triangle), GREEN))?
        .label("Triangle Approximation")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(2)));

    // Add dotted vertical lines at first zero (r₀) - light blue to match 2D rings
    let first_zero_color = RGBColor(173, 216, 230); // Light blue
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(1.0, 0.0), (1.0, 1.0)],
        first_zero_color.stroke_width(2),
    )))?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(-1.0, 0.0), (-1.0, 1.0)],
        first_zero_color.stroke_width(2),
    )))?;

    // Add FWHM annotation lines and axvlines
    let fwhm_half_normalized = airy_disk.fwhm / 2.0 / airy_disk.first_zero;
    let fwhm_intensity = 0.5; // Half maximum by definition
    let fwhm_color = RGBColor(255, 182, 193); // Light pink/red to match 2D rings

    // Keep the grey spanning bar (horizontal line across full FWHM)
    chart.draw_series(std::iter::once(PathElement::new(
        vec![
            (-fwhm_half_normalized, fwhm_intensity),
            (fwhm_half_normalized, fwhm_intensity),
        ],
        RGBColor(128, 128, 128).stroke_width(2),
    )))?;

    // Add dotted vertical axvlines at FWHM points - light red to match 2D rings
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(fwhm_half_normalized, 0.0), (fwhm_half_normalized, 1.0)],
        fwhm_color.stroke_width(2),
    )))?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(-fwhm_half_normalized, 0.0), (-fwhm_half_normalized, 1.0)],
        fwhm_color.stroke_width(2),
    )))?;

    // Add text annotation for first zero
    chart.draw_series(std::iter::once(Text::new(
        format!("r₀ = {:.3}", airy_disk.first_zero),
        (1.05, 0.9),
        ("sans-serif", 18).into_font().color(&BLACK),
    )))?;

    // Add text annotation for FWHM
    chart.draw_series(std::iter::once(Text::new(
        "FWHM",
        (0.05, 0.55),
        ("sans-serif", 18)
            .into_font()
            .color(&RGBColor(128, 128, 128)),
    )))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 18))
        .draw()?;

    root.present()?;

    Ok(())
}

/// Creates a 512×512 grayscale 2D image of an Airy disk pattern with colored rings.
///
/// This function renders a 2D representation of the specified Airy disk approximation
/// as a grayscale intensity map. Key features are highlighted with colored rings:
///
/// - Light blue ring marks the first zero (dark ring) at radius r₀
/// - Light pink ring marks the Full-Width-Half-Maximum (FWHM) radius
/// - Grayscale intensity represents the theoretical light distribution
///
/// The image spans ±2 normalized radii (±2×r₀) from center to edge.
///
/// # Arguments
///
/// * `airy_disk` - AiryDisk instance containing the physical parameters
/// * `save_path` - File path where the image should be saved (PNG format)
/// * `image_type` - Which approximation type to render (Exact, Gaussian, or Triangle)
///
/// # Returns
///
/// Returns `Ok(())` on successful image generation, or an error if file I/O fails.
///
/// # Examples
///
/// ```rust,no_run
/// use simulator::image_proc::AiryDisk;
///
/// let airy = AiryDisk::new();
/// create_airy_disk_image(&airy, "exact.png", ImageType::Exact)?;
/// create_airy_disk_image(&airy, "gauss.png", ImageType::Gaussian)?;
/// ```
fn create_airy_disk_image(
    airy_disk: &AiryDisk,
    save_path: &str,
    image_type: ImageType,
) -> Result<(), Box<dyn std::error::Error>> {
    let size = 512;
    let center = size as f64 / 2.0;
    let max_radius = center; // Use full half-size (edges = ±2 normalized)

    // Scale factor: edges at ±2 normalized radii
    let scale_factor = airy_disk.first_zero * 2.0 / max_radius;

    // Calculate ring radii in pixels
    let first_zero_radius = airy_disk.first_zero / scale_factor;
    let fwhm_radius = (airy_disk.fwhm / 2.0) / scale_factor;

    let root = BitMapBackend::new(save_path, (size as u32, size as u32)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create pixel buffer
    for y in 0..size {
        for x in 0..size {
            let dx = x as f64 - center;
            let dy = y as f64 - center;
            let pixel_radius = (dx * dx + dy * dy).sqrt();

            // Convert pixel radius to angular radius
            let angular_radius = pixel_radius * scale_factor;

            // Calculate intensity based on image type (0 to 1)
            let intensity = match image_type {
                ImageType::Exact => airy_disk.intensity(angular_radius),
                ImageType::Gaussian => airy_disk.gaussian_approximation(angular_radius),
                ImageType::Triangle => airy_disk.triangle_approximation(angular_radius),
            };

            // Convert to greyscale (0-255)
            let grey_value = (intensity * 255.0) as u8;

            // Check if we're near the ring positions
            let is_first_zero_ring = (pixel_radius - first_zero_radius).abs() < 0.8;
            let is_fwhm_ring = (pixel_radius - fwhm_radius).abs() < 0.8;

            let color = if is_first_zero_ring {
                // Light blue ring at first zero
                RGBColor(173, 216, 230) // Light blue
            } else if is_fwhm_ring {
                // Light red ring at FWHM
                RGBColor(255, 182, 193) // Light pink/red
            } else {
                // Greyscale based on intensity
                RGBColor(grey_value, grey_value, grey_value)
            };

            root.draw_pixel((x, y), &color)?;
        }
    }

    root.present()?;
    Ok(())
}

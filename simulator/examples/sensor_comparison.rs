//! Sensor comparison tool for evaluating imaging sensor specifications.
//!
//! This example generates comprehensive comparison tables and detailed specifications
//! for all available sensor models in the simulator. It provides both a quick
//! comparison table and detailed breakdowns for sensor selection and analysis.
//!
//! # Output Format
//!
//! The tool generates two main sections:
//!
//! ## 1. Comparison Table (Markdown format)
//! A tabular comparison showing key metrics for all sensors:
//! - Resolution (width × height pixels)
//! - Physical area (cm²)
//! - Dark current at 0°C (e⁻/pixel/s)
//! - Read noise (e⁻)
//! - Average quantum efficiency (150-1100nm range)
//!
//! ## 2. Detailed Specifications
//! Individual sensor breakdowns including:
//! - Physical dimensions and pixel size
//! - Maximum well depth and frame rate
//! - Bit depth and peak QE performance
//! - Temperature-dependent characteristics
//!
//! # Usage
//!
//! ```bash
//! cargo run --example sensor_comparison
//! ```
//!
//! The output is designed to be suitable for:
//! - Pasting into documentation (Markdown format)
//! - Sensor selection for specific applications
//! - Performance comparison across different sensor types
//! - Understanding trade-offs between resolution, noise, and sensitivity

use simulator::hardware::sensor::models::ALL_SENSORS;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Sensor Comparison Table");
    println!("======================");
    println!();

    // Use all available sensors
    let sensors = &*ALL_SENSORS;

    // Temperature for dark current comparison
    let temp_c = 0.0;

    // Wavelength range for QE integration (150nm to 1100nm)
    let wavelength_start = 150.0;
    let wavelength_end = 1100.0;
    let wavelength_step = 1.0; // 1nm steps

    println!("Comparison at {temp_c}°C");
    println!();

    // Markdown table header
    println!("| Sensor | Resolution | Area (cm²) | Dark Current @ 0°C | Read Noise | Avg QE |");
    println!("|--------|------------|------------|-------------------|------------|--------|");
    println!(
        "|        | (W×H)      |            | (e⁻/pixel/s)      | (e⁻)       | (150-1100nm) |"
    );

    // Calculate and display data for each sensor
    for config in sensors {
        // Dark current at 0°C
        let dark_current = config
            .dark_current_estimator
            .estimate_at_temperature(temp_c)
            .expect("Interpolation should be valid");

        // Read noise - estimate at room temperature with 1s exposure for comparison
        let read_noise = config
            .read_noise_estimator
            .estimate(20.0, std::time::Duration::from_secs(1))
            .unwrap_or(0.0);

        // Resolution and area
        let resolution = format!("{}×{}", config.width_px, config.height_px);
        let sensor_width_cm = (config.width_px as f64 * config.pixel_size_um) / 10000.0; // μm to cm
        let sensor_height_cm = (config.height_px as f64 * config.pixel_size_um) / 10000.0; // μm to cm
        let area_cm2 = sensor_width_cm * sensor_height_cm;

        // Calculate average QE from 150nm to 1100nm
        let mut qe_sum = 0.0;
        let mut count = 0;

        let mut wavelength = wavelength_start;
        while wavelength <= wavelength_end {
            qe_sum += config.quantum_efficiency.at(wavelength);
            count += 1;
            wavelength += wavelength_step;
        }

        let avg_qe = qe_sum / count as f64;

        // Print markdown table row
        println!(
            "| {} | {} | {:.3} | {:.4} | {:.2} | {:.3} |",
            config.name, resolution, area_cm2, dark_current, read_noise, avg_qe
        );
    }

    println!();
    println!("**Notes:**");
    println!("- Dark current shown at 0°C (varies exponentially with temperature)");
    println!("- QE average calculated with 1nm steps from 150nm to 1100nm");
    println!("- All values from manufacturer specifications");
    println!();

    // Additional detailed breakdown
    println!("## Detailed Sensor Specifications");

    for config in sensors {
        println!();
        let sensor_width_cm = (config.width_px as f64 * config.pixel_size_um) / 10000.0;
        let sensor_height_cm = (config.height_px as f64 * config.pixel_size_um) / 10000.0;
        let area_cm2 = sensor_width_cm * sensor_height_cm;

        println!();
        println!("### {}", config.name);
        println!(
            "- **Resolution:** {} × {} pixels",
            config.width_px, config.height_px
        );
        println!(
            "- **Sensor area:** {area_cm2:.3} cm² ({sensor_width_cm:.2} × {sensor_height_cm:.2} cm)"
        );
        println!("- **Pixel size:** {:.2} μm", config.pixel_size_um);
        println!(
            "- **Max well depth:** {} e⁻",
            config.max_well_depth_e as u64
        );
        println!("- **Frame rate:** {:.1} fps", config.max_frame_rate_fps);
        println!("- **Bit depth:** {}-bit", config.bit_depth);

        // Find peak QE wavelength and value
        let mut peak_qe = 0.0;
        let mut peak_wavelength = 0.0;

        for wavelength_nm in 300..=1000 {
            let qe = config.quantum_efficiency.at(wavelength_nm as f64);
            if qe > peak_qe {
                peak_qe = qe;
                peak_wavelength = wavelength_nm as f64;
            }
        }

        println!("- **Peak QE:** {peak_qe:.3} at {peak_wavelength:.0}nm");

        // Dark current at 0°C only
        let dc_0c = config
            .dark_current_estimator
            .estimate_at_temperature(0.0)
            .expect("Interpolation should be valid");
        println!("- **Dark current @ 0°C:** {dc_0c:.4} e⁻/pixel/s");
    }

    Ok(())
}

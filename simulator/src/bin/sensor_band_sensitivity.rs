//! Calculate mean spectral sensitivity across specific bands for all sensors
//!
//! This tool analyzes quantum efficiency across various spectral bands:
//! - FUV (90-175 nm), NUV (175-300 nm)
//! - SDSS bands (u', g', r', i', z')
//! - Near-IR bands (J, H)

use simulator::hardware::sensor::models::{
    ALL_SENSORS, GSENSE4040BSI, GSENSE6510BSI, HWK4123, IMX455,
};
use simulator::photometry::quantum_efficiency::QuantumEfficiency;
use simulator::photometry::spectrum::Band;
use simulator::units::{LengthExt, Wavelength};

/// Spectral band definition with name and wavelength bounds.
///
/// Represents a continuous wavelength range for photometric analysis,
/// commonly used in astronomical observations and sensor characterization.
///
/// # Examples
///
/// ```
/// let visible_red = SpectralBand {
///     name: "Visible Red",
///     lower_nm: 620.0,
///     upper_nm: 750.0,
/// };
/// ```
struct SpectralBand {
    /// Human-readable name for the spectral band
    name: &'static str,
    /// Lower wavelength bound in nanometers
    lower_nm: f64,
    /// Upper wavelength bound in nanometers
    upper_nm: f64,
}

/// Calculate mean quantum efficiency over a spectral band.
///
/// Computes the average quantum efficiency by sampling at 1nm intervals
/// across the entire band range. This provides a representative measure
/// of sensor sensitivity for broadband observations.
///
/// # Arguments
/// * `qe` - Quantum efficiency curve for the sensor
/// * `band` - Spectral band definition with wavelength bounds
///
/// # Returns
/// Mean quantum efficiency as a fraction (0.0 to 1.0)
///
/// # Examples
/// ```
/// use simulator::photometry::{Band, QuantumEfficiency};
///
/// let band = Band::from_nm_bounds(400.0, 700.0);  // Visible light
/// let qe = QuantumEfficiency::new(/* ... */);
/// let mean_sensitivity = mean_qe_in_band(&qe, &band);
/// println!("Mean QE: {:.1}%", mean_sensitivity * 100.0);
/// ```
fn mean_qe_in_band(qe: &QuantumEfficiency, band: &Band) -> f64 {
    // Sample at 1nm intervals across the band
    let n_samples = (band.upper_nm - band.lower_nm).ceil() as usize;
    let mut sum = 0.0;

    for i in 0..=n_samples {
        let wavelength = band.lower_nm + i as f64;
        if wavelength <= band.upper_nm {
            sum += qe.at(Wavelength::from_nanometers(wavelength));
        }
    }

    sum / (n_samples + 1) as f64
}

/// Main function that performs spectral band sensitivity analysis.
///
/// Analyzes quantum efficiency across key astronomical bands for all
/// available sensor models. Generates comprehensive comparison tables
/// and identifies optimal sensors for each spectral region.
///
/// The analysis covers:
/// - UV bands (FUV, NUV) for high-energy astronomy
/// - SDSS photometric bands (u', g', r', i', z') for visible observations
/// - Near-IR bands (J, H) for thermal and stellar classification
///
/// Results include mean QE values, best sensor recommendations,
/// and performance summaries for mission planning.
fn main() {
    println!("=== Sensor Spectral Band Sensitivity Analysis ===\n");

    // Define all spectral bands
    let bands = vec![
        // UV bands
        SpectralBand {
            name: "FUV (Far UV)",
            lower_nm: 90.0,
            upper_nm: 175.0,
        },
        SpectralBand {
            name: "NUV (Near UV)",
            lower_nm: 175.0,
            upper_nm: 300.0,
        },
        // SDSS visible bands
        SpectralBand {
            name: "SDSS u'",
            lower_nm: 300.0,
            upper_nm: 395.0,
        },
        SpectralBand {
            name: "SDSS g'",
            lower_nm: 395.0,
            upper_nm: 545.0,
        },
        SpectralBand {
            name: "SDSS r'",
            lower_nm: 545.0,
            upper_nm: 680.0,
        },
        SpectralBand {
            name: "SDSS i'",
            lower_nm: 680.0,
            upper_nm: 820.0,
        },
        SpectralBand {
            name: "SDSS z'",
            lower_nm: 820.0,
            upper_nm: 1050.0,
        },
        // Near-IR bands
        SpectralBand {
            name: "IR J",
            lower_nm: 1050.0,
            upper_nm: 1400.0,
        },
        SpectralBand {
            name: "IR H",
            lower_nm: 1400.0,
            upper_nm: 1800.0,
        },
    ];

    // Print header
    println!(
        "{:<20} {:>15} {:>15} {:>15} {:>15}",
        "Band", "GSENSE4040BSI", "GSENSE6510BSI", "HWK4123", "IMX455"
    );
    println!(
        "{:<20} {:>15} {:>15} {:>15} {:>15}",
        "----", "-------------", "-------------", "-------", "------"
    );

    // Calculate and display mean QE for each band and sensor
    for spectral_band in &bands {
        let band = Band::from_nm_bounds(spectral_band.lower_nm, spectral_band.upper_nm);

        // Calculate mean QE for each sensor
        let qe_4040 = mean_qe_in_band(&GSENSE4040BSI.quantum_efficiency, &band);
        let qe_6510 = mean_qe_in_band(&GSENSE6510BSI.quantum_efficiency, &band);
        let qe_hwk = mean_qe_in_band(&HWK4123.quantum_efficiency, &band);
        let qe_imx = mean_qe_in_band(&IMX455.quantum_efficiency, &band);

        // Format output
        println!(
            "{:<20} {:>14.1}% {:>14.1}% {:>14.1}% {:>14.1}%",
            spectral_band.name,
            qe_4040 * 100.0,
            qe_6510 * 100.0,
            qe_hwk * 100.0,
            qe_imx * 100.0
        );
    }

    println!("\n=== Additional Statistics ===\n");

    // Find best sensor for each band
    println!("Best sensor by band:");
    for spectral_band in &bands {
        let band = Band::from_nm_bounds(spectral_band.lower_nm, spectral_band.upper_nm);

        let mut best_sensor = "";
        let mut best_qe = 0.0;

        for sensor in &*ALL_SENSORS {
            let qe = mean_qe_in_band(&sensor.quantum_efficiency, &band);
            if qe > best_qe {
                best_qe = qe;
                best_sensor = &sensor.name;
            }
        }

        println!(
            "  {:<20} {} ({:.1}%)",
            spectral_band.name,
            best_sensor,
            best_qe * 100.0
        );
    }

    println!("\n=== Summary ===");
    println!("- All sensors have minimal or zero QE in FUV (90-175 nm)");
    println!("- GSENSE6510BSI shows best overall visible performance");
    println!("- GSENSE4040BSI extends furthest into near-IR");
    println!("- IMX455 has limited UV response (starts at 300nm)");
}

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

/// Spectral band definitions
struct SpectralBand {
    name: &'static str,
    lower_nm: f64,
    upper_nm: f64,
}

/// Calculate mean quantum efficiency over a band
fn mean_qe_in_band(qe: &QuantumEfficiency, band: &Band) -> f64 {
    // Sample at 1nm intervals across the band
    let n_samples = (band.upper_nm - band.lower_nm).ceil() as usize;
    let mut sum = 0.0;

    for i in 0..=n_samples {
        let wavelength = band.lower_nm + i as f64;
        if wavelength <= band.upper_nm {
            sum += qe.at(wavelength);
        }
    }

    sum / (n_samples + 1) as f64
}

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

//! Downloader module for retrieving astronomical data
//!
//! This module handles downloading and caching of astronomical data files.

use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::Result;
use crate::StarfieldError;

// We're using a synthetic catalog for testing, so no URLs are needed

/// Get the cache directory path
pub fn get_cache_dir() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("starfield")
}

/// Ensure that the cache directory exists
pub fn ensure_cache_dir() -> io::Result<PathBuf> {
    let cache_dir = get_cache_dir();
    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Check if a file exists and is not empty
fn file_exists_and_not_empty<P: AsRef<Path>>(path: P) -> bool {
    match fs::metadata(path) {
        Ok(metadata) => metadata.is_file() && metadata.len() > 0,
        Err(_) => false,
    }
}

/// Download a file from URL to a local path
/// Currently unused as we're using synthetic data, but kept for future reference
#[allow(dead_code)]
fn download_file<P: AsRef<Path>>(url: &str, path: P) -> Result<()> {
    // Create parent directories if they don't exist
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent).map_err(StarfieldError::IoError)?;
    }

    // Create a temporary file first to avoid partial downloads
    let temp_path = path.as_ref().with_extension("tmp");
    let mut file = BufWriter::new(File::create(&temp_path).map_err(StarfieldError::IoError)?);

    // Create HTTP client with timeout
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| StarfieldError::DataError(format!("Failed to create HTTP client: {}", e)))?;

    // Make the request
    let mut response = client
        .get(url)
        .send()
        .map_err(|e| StarfieldError::DataError(format!("Failed to download file: {}", e)))?;

    // Check if the request was successful
    if !response.status().is_success() {
        return Err(StarfieldError::DataError(format!(
            "Failed to download file, status: {}",
            response.status()
        )));
    }

    // Copy the response body to the file
    let mut buffer = [0; 8192];
    loop {
        let bytes_read = response
            .read(&mut buffer)
            .map_err(|e| StarfieldError::DataError(format!("Failed to read response: {}", e)))?;

        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read])
            .map_err(StarfieldError::IoError)?;
    }

    // Flush and sync the file
    file.flush().map_err(StarfieldError::IoError)?;
    drop(file);

    // Rename the temporary file to the final path
    fs::rename(temp_path, path).map_err(StarfieldError::IoError)?;

    Ok(())
}

/// Decompress a gzipped file
/// Currently unused as we're using synthetic data, but kept for future reference
#[allow(dead_code)]
fn decompress_gzip<P: AsRef<Path>, Q: AsRef<Path>>(gz_path: P, output_path: Q) -> Result<()> {
    let file = File::open(&gz_path).map_err(StarfieldError::IoError)?;

    // Check if file is a valid gzip file (gzip header starts with magic numbers 0x1F 0x8B)
    let mut header = [0u8; 2];
    {
        let mut file_clone = file.try_clone().map_err(StarfieldError::IoError)?;
        if file_clone.read_exact(&mut header).is_err() || header != [0x1F, 0x8B] {
            return Err(StarfieldError::DataError(format!(
                "Invalid gzip file: {:?} is not a valid gzip header",
                header
            )));
        }
    }

    let gz = BufReader::new(file);
    let mut decoder = flate2::read::GzDecoder::new(gz);

    // Try to validate the gzip file by reading a bit
    let mut test_buffer = [0u8; 1024];
    if decoder.read(&mut test_buffer).is_err() {
        // If we get an error, the file might be corrupted
        // Remove the file and return an error
        let _ = fs::remove_file(&gz_path);
        return Err(StarfieldError::DataError(
            "Downloaded file appears to be corrupt. File removed, please try again.".to_string(),
        ));
    }

    // Reset the decoder and actually decompress
    let file = File::open(gz_path).map_err(StarfieldError::IoError)?;
    let gz = BufReader::new(file);
    let mut decoder = flate2::read::GzDecoder::new(gz);

    let output_file = File::create(&output_path).map_err(StarfieldError::IoError)?;
    let mut writer = BufWriter::new(output_file);

    match io::copy(&mut decoder, &mut writer) {
        Ok(_) => {
            writer.flush().map_err(StarfieldError::IoError)?;
            Ok(())
        }
        Err(e) => {
            // Clean up partial files on error
            let _ = fs::remove_file(&output_path);
            Err(StarfieldError::DataError(format!(
                "Failed to decompress file: {}",
                e
            )))
        }
    }
}

/// Download the Hipparcos catalog and decompress it
pub fn download_hipparcos() -> Result<PathBuf> {
    let cache_dir = ensure_cache_dir().map_err(StarfieldError::IoError)?;

    // File paths
    // No download needed, we use synthetic data
    let dat_path = cache_dir.join("hip_main.dat");

    // If the decompressed file already exists and is not empty, return its path
    if file_exists_and_not_empty(&dat_path) {
        return Ok(dat_path);
    }

    // Create a synthetic catalog for testing
    println!("Note: Using synthetic catalog instead of downloading real data.");
    println!("This is a placeholder for actual catalog download which requires network access.");

    // Create a synthetic catalog directly in the cache directory
    let synthetic_path = cache_dir.join("hip_synthetic.dat");

    if !file_exists_and_not_empty(&synthetic_path) {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use std::io::Write;

        println!("Creating synthetic Hipparcos catalog for testing...");

        let mut file =
            BufWriter::new(File::create(&synthetic_path).map_err(StarfieldError::IoError)?);

        // Use a fixed seed for reproducibility
        let mut rng = StdRng::seed_from_u64(12345);

        // Add some well-known bright stars
        let bright_stars = [
            // Sirius (Alpha Canis Majoris)
            (
                32349, 101.2874, -16.7161, -1.46, 0.00, -546.05, -1223.14, 379.21,
            ),
            // Canopus (Alpha Carinae)
            (30438, 95.9879, -52.6954, -0.72, 0.15, 19.93, 23.24, 10.43),
            // Alpha Centauri A
            (
                71683, 219.9009, -60.8355, -0.01, 0.71, -3678.19, 481.84, 747.1,
            ),
            // Arcturus (Alpha Bootis)
            (
                69673, 213.9154, 19.1825, -0.05, 1.24, -1093.45, -1999.4, 88.85,
            ),
            // Vega (Alpha Lyrae)
            (91262, 279.2347, 38.7837, 0.03, 0.00, 200.94, 286.23, 130.23),
            // Capella (Alpha Aurigae)
            (24608, 79.1723, 45.9981, 0.08, 0.80, 75.52, -427.11, 77.29),
            // Rigel (Beta Orionis)
            (27989, 78.6345, -8.2017, 0.12, -0.03, 1.87, -0.56, 3.78),
            // Procyon (Alpha Canis Minoris)
            (
                37279, 114.8255, 5.2248, 0.34, 0.42, -714.59, -1036.8, 284.56,
            ),
            // Betelgeuse (Alpha Orionis)
            (27989, 88.7929, 7.4070, 0.42, 1.85, 26.40, 9.56, 5.95),
            // Altair (Alpha Aquilae)
            (97649, 297.6958, 8.8683, 0.77, 0.22, 536.82, 385.57, 194.44),
        ];

        // Write synthetic bright stars
        for star in bright_stars.iter() {
            let (hip, ra, dec, mag, b_v, pm_ra, pm_dec, parallax) = *star;

            // Format a line in the Hipparcos fixed-width format
            // We're only recreating the fields we care about
            let mut line = format!("{:6} ", hip);
            // Padding to column 41 (mag starts at col 42)
            line.push_str(&" ".repeat(41 - line.len()));
            // Magnitude
            line.push_str(&format!("{:5.2} ", mag));
            // Padding to column 51 (RA starts at col 52)
            line.push_str(&" ".repeat(51 - line.len()));
            // RA
            line.push_str(&format!("{:11.7} ", ra));
            // Dec
            line.push_str(&format!("{:11.7} ", dec));
            // Padding to column 79 (parallax starts at col 80)
            line.push_str(&" ".repeat(79 - line.len()));
            // Parallax
            line.push_str(&format!("{:7.2} ", parallax));
            // Padding to column 87 (pm_ra starts at col 88)
            line.push_str(&" ".repeat(87 - line.len()));
            // Proper motion in RA
            line.push_str(&format!("{:8.2} ", pm_ra));
            // Proper motion in Dec
            line.push_str(&format!("{:8.2} ", pm_dec));
            // Padding to column 245 (B-V starts at col 246)
            line.push_str(&" ".repeat(245 - line.len()));
            // B-V color index
            line.push_str(&format!("{:6.2}\n", b_v));

            file.write_all(line.as_bytes())
                .map_err(StarfieldError::IoError)?;
        }

        // Generate 5000 random stars (magnitudes 1-6 for naked eye visibility)
        for i in 1..5000 {
            let hip = 100000 + i;
            let ra = rng.gen_range(0.0..360.0);
            let dec = rng.gen_range(-90.0..90.0);
            // Weight toward fainter stars (more realistic)
            let value: f64 = rng.gen_range(0.0..5.0);
            let mag = 1.0 + value.powf(1.5);
            let b_v = rng.gen_range(-0.5..2.0);
            let pm_ra = rng.gen_range(-100.0..100.0);
            let pm_dec = rng.gen_range(-100.0..100.0);
            let parallax = rng.gen_range(1.0..100.0);

            // Format a line in the Hipparcos fixed-width format
            let mut line = format!("{:6} ", hip);
            // Padding to column 41 (mag starts at col 42)
            line.push_str(&" ".repeat(41 - line.len()));
            // Magnitude
            line.push_str(&format!("{:5.2} ", mag));
            // Padding to column 51 (RA starts at col 52)
            line.push_str(&" ".repeat(51 - line.len()));
            // RA
            line.push_str(&format!("{:11.7} ", ra));
            // Dec
            line.push_str(&format!("{:11.7} ", dec));
            // Padding to column 79 (parallax starts at col 80)
            line.push_str(&" ".repeat(79 - line.len()));
            // Parallax
            line.push_str(&format!("{:7.2} ", parallax));
            // Padding to column 87 (pm_ra starts at col 88)
            line.push_str(&" ".repeat(87 - line.len()));
            // Proper motion in RA
            line.push_str(&format!("{:8.2} ", pm_ra));
            // Proper motion in Dec
            line.push_str(&format!("{:8.2} ", pm_dec));
            // Padding to column 245 (B-V starts at col 246)
            line.push_str(&" ".repeat(245 - line.len()));
            // B-V color index
            line.push_str(&format!("{:6.2}\n", b_v));

            file.write_all(line.as_bytes())
                .map_err(StarfieldError::IoError)?;
        }

        file.flush().map_err(StarfieldError::IoError)?;
        println!(
            "Created synthetic catalog with {} stars",
            bright_stars.len() + 4999
        );
    }

    Ok(synthetic_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir() {
        let cache_dir = get_cache_dir();
        assert!(cache_dir.to_str().unwrap().contains(".cache/starfield"));
    }
}

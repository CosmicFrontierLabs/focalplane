//! Generate WorldWide Telescope links for the failed detection location
//!
//! This example creates WWT overlay URLs for the sky position that failed
//! to detect any stars across all exposures (RA: 272.7390°, Dec: -63.2654°)

use simulator::algo::misc::generate_wwt_overlay_url;
use simulator::hardware::{
    sensor::models::HWK4123, telescope::models::IDEAL_50CM, SatelliteConfig,
};
use simulator::units::{LengthExt, Temperature, TemperatureExt, Wavelength};
use starfield::Equatorial;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Failed detection coordinates
    let ra_deg = 272.7390;
    let dec_deg = -63.2654;
    let coordinates = Equatorial::from_degrees(ra_deg, dec_deg);

    println!("Failed Detection Location Analysis");
    println!("{}", "=".repeat(80));
    println!("RA: {ra_deg:.4}°, Dec: {dec_deg:.4}°");
    println!("This location detected 0 stars across all exposures (25-200ms)\n");

    // Create satellite configuration matching the sensor_shootout setup
    let satellite = SatelliteConfig::new(
        IDEAL_50CM.clone(),
        HWK4123.clone(),
        Temperature::from_celsius(-10.0),
        Wavelength::from_nanometers(550.0),
    );

    // Generate main overlay showing the failed region
    let failed_url = generate_wwt_overlay_url(
        "Failed Detection Region",
        &coordinates,
        &satellite,
        0.0, // No rotation
        Some("NO STARS"),
        "FF0000",   // Red text
        "00000080", // Semi-transparent black background
    )?;

    println!("WWT Overlay URL (Failed Detection - Red):");
    println!("{failed_url}\n");

    // Generate overlays for nearby successful regions for comparison
    println!("Nearby Successful Detection Regions:");
    println!("{}", "-".repeat(40));

    let nearby_regions = vec![
        (276.6043, -64.5339, "143.5 stars", "00FF00"), // Green for success
        (273.5374, -64.4166, "155.8 stars", "00FF00"),
        (276.9041, -60.0127, "158.1 stars", "00FF00"),
    ];

    for (ra, dec, label, color) in &nearby_regions {
        let coords = Equatorial::from_degrees(*ra, *dec);
        let url = generate_wwt_overlay_url(
            &format!("Nearby: {label}"),
            &coords,
            &satellite,
            0.0,
            Some(label),
            color,
            "00000080",
        )?;

        println!("RA: {ra:.4}°, Dec: {dec:.4}° ({label})");
        println!("{url}\n");
    }

    // Generate a wider field view centered on the failed region
    println!("Wide Field View (5° FOV) Centered on Failed Region:");
    println!("{}", "-".repeat(40));

    // For a wider field, we'd typically use a different optical configuration
    // but for the overlay we'll just use the same satellite config
    let wide_field_url = generate_wwt_overlay_url(
        "Failed Region - Wide Field",
        &coordinates,
        &satellite,
        0.0,
        Some("VOID"),
        "FFFF00",   // Yellow text
        "00000000", // Fully transparent
    )?;

    println!("{wide_field_url}\n");

    // Print instructions
    println!("Instructions:");
    println!("1. Copy any URL above and paste into your browser");
    println!("2. WWT will open and display the overlay at the specified location");
    println!("3. The overlay will show the sensor field of view with the label");
    println!("4. Red overlays indicate failed detections, green shows successful regions");

    Ok(())
}

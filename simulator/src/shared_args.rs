use clap::Parser;

/// Parse coordinates string in format "elongation,latitude"
fn parse_coordinates(s: &str) -> Result<(f64, f64), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err("Coordinates must be in format 'elongation,latitude'".to_string());
    }

    let elongation = parts[0]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid elongation value".to_string())?;
    let latitude = parts[1]
        .trim()
        .parse::<f64>()
        .map_err(|_| "Invalid latitude value".to_string())?;

    Ok((elongation, latitude))
}

/// Common arguments shared across multiple simulation binaries
#[derive(Parser, Debug, Clone)]
pub struct SharedSimulationArgs {
    /// Exposure time in seconds
    #[arg(long, default_value_t = 1.0)]
    pub exposure: f64,

    /// Wavelength in nanometers
    #[arg(long, default_value_t = 550.0)]
    pub wavelength: f64,

    /// Sensor temperature in degrees Celsius for dark current calculation
    #[arg(long, default_value_t = 20.0)]
    pub temperature: f64,

    /// Enable debug output
    #[arg(long, default_value_t = false)]
    pub debug: bool,

    /// Solar elongation and coordinates for zodiacal background (format: "elongation,latitude")
    #[arg(long, default_value = "165.0,75.0", value_parser = parse_coordinates)]
    pub coordinates: (f64, f64),
}

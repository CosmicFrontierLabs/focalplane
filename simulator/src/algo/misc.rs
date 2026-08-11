//! WorldWide Telescope (WWT) overlay utilities.
//!
//! This module provides functions for generating WorldWide Telescope overlay URLs
//! for visualizing telescope field of view on the sky.

use crate::hardware::SatelliteConfig;
use crate::star_math::field_diameter;
use shared::units::AngleExt;
use starfield::Equatorial;
use url::Url;

// Re-export shared misc functions
pub use shared::algo::misc::{dec_dms_to_deg, interp, normalize, ra_hms_to_deg, InterpError};

/// Generates a WorldWide Telescope (WWT) URL for overlaying an observation field on the sky.
///
/// Creates a URL that will display a rectangular overlay in WorldWide Telescope representing
/// the field of view of the telescope/sensor combination at a specific sky position.
///
/// # Arguments
/// * `object_name` - A descriptive name for the target.
/// * `coordinates` - Equatorial coordinates (RA/Dec) of the center of the box.
/// * `satellite` - Satellite configuration containing telescope and sensor parameters.
/// * `rotation_deg` - Rotation of the image on the sky in degrees (0 = North up, East left).
/// * `text` - Optional text to display in the center of the overlay.
/// * `text_color` - Hex color code for the text (e.g., "FFFFFF" for white).
/// * `background_color` - Hex color code for the background (e.g., "00000000" for transparent).
///
/// # Returns
/// A `Result` containing the generated URL as a `String` or an `url::ParseError`.
pub fn generate_wwt_overlay_url(
    object_name: &str,
    coordinates: &Equatorial,
    satellite: &SatelliteConfig,
    rotation_deg: f64,
    text: Option<&str>,
    text_color: &str,
    background_color: &str,
) -> Result<String, url::ParseError> {
    // Calculate field of view from satellite configuration
    let fov = field_diameter(&satellite.telescope, &satellite.sensor);
    let fov_deg = fov.as_degrees();

    // Calculate center coordinates (convert from radians to degrees)
    // Round to reasonable precision to avoid floating point issues
    let ra_deg = (coordinates.ra.to_degrees() * 1e10).round() / 1e10;
    let dec_deg = (coordinates.dec.to_degrees() * 1e10).round() / 1e10;

    // Build the WWT URL
    let mut wwt_url = Url::parse("http://www.worldwidetelescope.org/wwtweb/ShowImage.aspx")?;
    wwt_url
        .query_pairs_mut()
        .append_pair("name", object_name)
        .append_pair("ra", &ra_deg.to_string())
        .append_pair("dec", &dec_deg.to_string())
        .append_pair("x", &fov_deg.to_string())
        .append_pair("y", &fov_deg.to_string())
        .append_pair("rotation", &rotation_deg.to_string());

    // Handle optional text overlay
    if let Some(display_text) = text {
        wwt_url
            .query_pairs_mut()
            .append_pair("wtml", "true")
            .append_pair("imageset", "box")
            .append_pair("text", display_text)
            .append_pair("textcolor", text_color)
            .append_pair("bgcolor", background_color);
    }

    Ok(wwt_url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{sensor::models::GSENSE6510BSI, telescope::models::IDEAL_50CM};
    use shared::units::{Temperature, TemperatureExt};

    #[test]
    fn test_generate_wwt_overlay_url() {
        let telescope = IDEAL_50CM.clone();
        let sensor = GSENSE6510BSI.clone();
        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(-10.0));

        let coord = Equatorial::from_degrees(150.0, 30.0);

        let url = generate_wwt_overlay_url(
            "Test Object",
            &coord,
            &satellite,
            45.0,
            None,
            "FFFFFF",
            "00000000",
        )
        .unwrap();

        // Check that URL contains expected parameters
        assert!(url.contains("name=Test+Object"));
        // Float values might have decimal places
        assert!(url.contains("ra=150"));
        assert!(url.contains("dec=30"));
        assert!(url.contains("rotation=45"));
    }

    #[test]
    fn test_generate_wwt_overlay_url_with_text() {
        let telescope = IDEAL_50CM.clone();
        let sensor = GSENSE6510BSI.clone();
        let satellite = SatelliteConfig::new(telescope, sensor, Temperature::from_celsius(-10.0));

        let coord = Equatorial::from_degrees(150.0, 30.0);

        let url = generate_wwt_overlay_url(
            "Test Object",
            &coord,
            &satellite,
            0.0,
            Some("Test Label"),
            "FF0000",
            "000000FF",
        )
        .unwrap();

        // Check that URL contains text parameters
        assert!(url.contains("text=Test+Label"));
        assert!(url.contains("textcolor=FF0000"));
        assert!(url.contains("bgcolor=000000FF"));
        assert!(url.contains("wtml=true"));
        assert!(url.contains("imageset=box"));
    }
}

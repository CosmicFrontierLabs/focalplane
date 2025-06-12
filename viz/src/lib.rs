//! Visualization tools for meter-sim
//!
//! This crate provides visualization tools for the meter-sim project,
//! including text-based histograms, data plotting, and other visualization utilities.

use std::fmt;
use thiserror::Error;

/// Error types for visualization operations
#[derive(Debug, Error)]
pub enum VizError {
    #[error("Histogram error: {0}")]
    HistogramError(String),

    #[error("Formatting error: {0}")]
    FmtError(#[from] fmt::Error),
}

/// Result type for visualization operations
pub type Result<T> = std::result::Result<T, VizError>;

pub mod density_map;
pub mod histogram;

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};
    use simulator::image_proc::overlay::overlay_to_image;

    #[test]
    fn test_text_rendering() {
        // Create a test image with various text rendering options
        let width = 800;
        let height = 600;
        let text_color = "#000000"; // Black text
        let bg_color = "#ffffff"; // White background
        let highlight_color = "#eeeeee"; // Light gray for text backgrounds

        // Create basic SVG with various text rendering approaches
        let svg_data = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
                <!-- Background rectangle -->
                <rect x="0" y="0" width="{}" height="{}" fill="{}" />
                
                <!-- Standard text elements with different fonts -->
                <rect x="20" y="20" width="760" height="30" fill="{}" />
                <text x="30" y="40" font-family="sans-serif" font-size="20" fill="{}">Text Test 1: sans-serif font</text>
                
                <rect x="20" y="60" width="760" height="30" fill="{}" />
                <text x="30" y="80" font-family="serif" font-size="20" fill="{}">Text Test 2: serif font</text>
                
                <rect x="20" y="100" width="760" height="30" fill="{}" />
                <text x="30" y="120" font-family="monospace" font-size="20" fill="{}">Text Test 3: monospace font</text>
                
                <!-- Text with explicit fallback fonts -->
                <rect x="20" y="140" width="760" height="30" fill="{}" />
                <text x="30" y="160" font-family="Arial, Helvetica, sans-serif" font-size="20" fill="{}">
                    Text Test 4: Multiple font fallbacks
                </text>
                
                <!-- Text with path-based approach (should always work) -->
                <rect x="20" y="180" width="760" height="30" fill="{}" />
                <path d="M30 200 L50 200 L50 220 L30 220 Z" fill="{}" />
                <path d="M60 200 L80 200 L80 220 L60 220 Z" fill="{}" />
                <text x="100" y="215" font-family="sans-serif" font-size="20" fill="{}">
                    Text Test 5: With path objects nearby
                </text>
                
                <!-- Styled text with different attributes -->
                <rect x="20" y="240" width="760" height="30" fill="{}" />
                <text x="30" y="260" font-family="sans-serif" font-size="20" font-weight="bold" fill="{}">
                    Text Test 6: Bold text
                </text>
                
                <rect x="20" y="280" width="760" height="30" fill="{}" />
                <text x="30" y="300" font-family="sans-serif" font-size="20" font-style="italic" fill="{}">
                    Text Test 7: Italic text
                </text>
                
                <!-- Text with explicit attributes -->
                <rect x="20" y="320" width="760" height="30" fill="{}" />
                <text x="30" y="340" font-family="sans-serif" font-size="20" font-weight="bold" 
                      text-rendering="geometricPrecision" fill="{}">
                    Text Test 8: With text-rendering attribute
                </text>
                
                <!-- SVG doesn't have direct ttf embedding, but we can try system fonts -->
                <rect x="20" y="360" width="760" height="30" fill="{}" />
                <text x="30" y="380" font-family="Liberation Sans, Ubuntu, DejaVu Sans" font-size="20" fill="{}">
                    Text Test 9: Common Linux system fonts
                </text>
                
                <!-- Simple shapes for verification -->
                <rect x="30" y="420" width="200" height="50" stroke="black" stroke-width="2" fill="blue" />
                <circle cx="400" cy="445" r="25" fill="red" />
                <line x1="500" y1="420" x2="700" y2="470" stroke="green" stroke-width="5" />
            </svg>"##,
            width,
            height,
            width,
            height,
            bg_color,
            highlight_color,
            text_color,
            highlight_color,
            text_color,
            highlight_color,
            text_color,
            highlight_color,
            text_color,
            highlight_color,
            text_color,
            text_color,
            text_color,
            highlight_color,
            text_color,
            highlight_color,
            text_color,
            highlight_color,
            text_color,
            highlight_color,
            text_color
        );

        // Create a blank white image
        let mut image = RgbImage::new(width, height);
        for pixel in image.pixels_mut() {
            *pixel = Rgb([255, 255, 255]);
        }
        let base_image = DynamicImage::ImageRgb8(image);

        // Create output directory
        let output_dir = test_helpers::get_output_dir();

        // Process the SVG
        let result_image = overlay_to_image(&base_image, &svg_data);

        // Save the result
        let output_path = output_dir.join("text_rendering_test.png");
        result_image
            .save(&output_path)
            .expect("Failed to save text test image");

        // Verify the image size is correct
        assert_eq!(result_image.width(), { width });
        assert_eq!(result_image.height(), { height });

        // This test mainly checks if the overlay function works without errors
        // Visual verification needs to be done manually
    }
}

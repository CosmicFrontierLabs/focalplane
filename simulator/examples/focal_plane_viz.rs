//! Focal plane visualization tool
//!
//! Renders the back focal area of a satellite with sensor footprints overlaid.
//! Shows a circle representing the full focal plane area and rectangles for each
//! sensor, colored by sensor model with labels.

use simulator::hardware::{
    sensor::models::*,
    sensor_array::{PositionedSensor, SensorArray, SensorPosition},
};
use simulator::units::{Length, LengthExt};
use std::collections::HashMap;
use std::fs;

/// Generate a color for a sensor model using a hash of its name
fn get_sensor_color(sensor_name: &str) -> String {
    // Define a palette of distinct colors
    let colors = vec![
        "#FF6B6B", // Red
        "#4ECDC4", // Teal
        "#45B7D1", // Blue
        "#FFA07A", // Light Salmon
        "#98D8C8", // Mint
        "#FFD93D", // Yellow
        "#6C5CE7", // Purple
        "#A8E6CF", // Light Green
        "#FF8B94", // Pink
        "#C7CEEA", // Lavender
    ];

    // Hash the sensor name to get a consistent color
    let hash: usize = sensor_name.bytes().map(|b| b as usize).sum();
    let color_index = hash % colors.len();
    colors[color_index].to_string()
}

/// Create an SVG visualization of the focal plane with sensor footprints
fn create_focal_plane_svg(
    array: &SensorArray,
    focal_length_mm: f64,
    field_of_view_deg: f64,
    output_size_px: u32,
) -> String {
    // Calculate the focal plane radius in mm
    let fov_rad = field_of_view_deg.to_radians() / 2.0;
    let focal_plane_radius_mm = focal_length_mm * fov_rad.tan();

    // Add margin around the focal plane circle
    let margin_factor = 1.2;
    let view_radius_mm = focal_plane_radius_mm * margin_factor;

    // SVG canvas setup
    let svg_size = output_size_px;
    let center = svg_size as f64 / 2.0;
    let scale = center / view_radius_mm; // pixels per mm

    let mut svg = String::new();

    // Start SVG
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{svg_size}\" height=\"{svg_size}\" viewBox=\"0 0 {svg_size} {svg_size}\">\n"
    ));

    // Background
    svg.push_str(&format!(
        "  <rect width=\"{svg_size}\" height=\"{svg_size}\" fill=\"#1a1a2e\"/>\n"
    ));

    // Grid pattern
    svg.push_str("  <defs>\n");
    svg.push_str(
        "    <pattern id=\"grid\" width=\"50\" height=\"50\" patternUnits=\"userSpaceOnUse\">\n",
    );
    svg.push_str("      <path d=\"M 50 0 L 0 0 0 50\" fill=\"none\" stroke=\"#2a2a3e\" stroke-width=\"0.5\"/>\n");
    svg.push_str("    </pattern>\n");
    svg.push_str("  </defs>\n");
    svg.push_str(&format!(
        "  <rect width=\"{svg_size}\" height=\"{svg_size}\" fill=\"url(#grid)\"/>\n"
    ));

    // Focal plane circle
    svg.push_str(&format!(
        "  <circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"none\" stroke=\"#666\" stroke-width=\"2\" stroke-dasharray=\"5,5\" opacity=\"0.7\"/>\n",
        center, center, focal_plane_radius_mm * scale
    ));

    // Title
    svg.push_str(&format!(
        "  <text x=\"{center}\" y=\"30\" font-family=\"monospace\" font-size=\"18\" fill=\"white\" text-anchor=\"middle\" font-weight=\"bold\">Focal Plane Layout</text>\n"
    ));

    // Subtitle
    svg.push_str(&format!(
        "  <text x=\"{center}\" y=\"50\" font-family=\"monospace\" font-size=\"14\" fill=\"#aaaaaa\" text-anchor=\"middle\">FOV: {field_of_view_deg:.1}° | Focal Length: {focal_length_mm:.1}mm | Radius: {focal_plane_radius_mm:.1}mm</text>\n"
    ));

    // Group sensors by model for legend
    let mut sensor_models: HashMap<String, String> = HashMap::new();

    // Draw sensor footprints
    for (idx, positioned_sensor) in array.sensors.iter().enumerate() {
        let sensor = &positioned_sensor.sensor;
        let pos = &positioned_sensor.position;

        // Get sensor dimensions in mm
        let (width_um, height_um) = sensor.dimensions_um();
        let width_mm = Length::from_micrometers(width_um).as_millimeters();
        let height_mm = Length::from_micrometers(height_um).as_millimeters();

        // Convert position to SVG coordinates
        let svg_x = center + pos.x_mm * scale;
        let svg_y = center - pos.y_mm * scale; // Flip Y axis for SVG

        // Calculate rectangle corners
        let rect_x = svg_x - (width_mm * scale / 2.0);
        let rect_y = svg_y - (height_mm * scale / 2.0);
        let rect_width = width_mm * scale;
        let rect_height = height_mm * scale;

        // Get color for this sensor model
        let sensor_name = sensor.name.clone();
        let color = get_sensor_color(&sensor_name);
        sensor_models.insert(sensor_name.clone(), color.clone());

        // Draw sensor rectangle
        svg.push_str(&format!("  <!-- Sensor {idx}: {sensor_name} -->\n"));
        svg.push_str(&format!(
            "  <rect x=\"{rect_x}\" y=\"{rect_y}\" width=\"{rect_width}\" height=\"{rect_height}\" fill=\"{color}\" fill-opacity=\"0.3\" stroke=\"{color}\" stroke-width=\"2\"/>\n"
        ));

        // Add sensor label in center
        let label_font_size = (rect_width.min(rect_height) / 8.0).max(10.0).min(16.0);
        svg.push_str(&format!(
            "  <text x=\"{svg_x}\" y=\"{svg_y}\" font-family=\"monospace\" font-size=\"{label_font_size}\" fill=\"white\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-weight=\"bold\">{sensor_name}</text>\n"
        ));

        // Add position info below the name
        let info_font_size = label_font_size * 0.7;
        svg.push_str(&format!(
            "  <text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"{}\" fill=\"#cccccc\" text-anchor=\"middle\" dominant-baseline=\"middle\">({:.1}, {:.1})mm</text>\n",
            svg_x, svg_y + label_font_size, info_font_size, pos.x_mm, pos.y_mm
        ));
    }

    // Add legend background
    let legend_y_start = svg_size as f64 - 30.0 - (sensor_models.len() as f64 * 25.0);
    let legend_height = sensor_models.len() as f64 * 25.0 + 40.0;

    svg.push_str(&format!(
        "  <rect x=\"20\" y=\"{}\" width=\"200\" height=\"{}\" fill=\"#1a1a2e\" fill-opacity=\"0.9\" stroke=\"#666666\" stroke-width=\"1\"/>\n",
        legend_y_start - 10.0, legend_height
    ));

    svg.push_str(&format!(
        "  <text x=\"30\" y=\"{}\" font-family=\"monospace\" font-size=\"14\" fill=\"white\" font-weight=\"bold\">Sensor Models:</text>\n",
        legend_y_start + 10.0
    ));

    // Add legend entries
    for (i, (model_name, color)) in sensor_models.iter().enumerate() {
        let y_pos = legend_y_start + 35.0 + (i as f64 * 25.0);

        svg.push_str(&format!(
            "  <rect x=\"30\" y=\"{}\" width=\"15\" height=\"15\" fill=\"{}\" fill-opacity=\"0.3\" stroke=\"{}\" stroke-width=\"2\"/>\n",
            y_pos - 7.5, color, color
        ));

        svg.push_str(&format!(
            "  <text x=\"50\" y=\"{y_pos}\" font-family=\"monospace\" font-size=\"12\" fill=\"white\" dominant-baseline=\"middle\">{model_name}</text>\n"
        ));
    }

    // Add coordinate axes
    svg.push_str(&format!(
        "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#666666\" stroke-width=\"1\" opacity=\"0.5\"/>\n",
        center, center - view_radius_mm * scale, center, center + view_radius_mm * scale
    ));
    svg.push_str(&format!(
        "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#666666\" stroke-width=\"1\" opacity=\"0.5\"/>\n",
        center - view_radius_mm * scale, center, center + view_radius_mm * scale, center
    ));

    // Axis labels
    svg.push_str(&format!(
        "  <text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"12\" fill=\"#666666\" text-anchor=\"start\">+X</text>\n",
        center + view_radius_mm * scale - 20.0, center
    ));
    svg.push_str(&format!(
        "  <text x=\"{}\" y=\"{}\" font-family=\"monospace\" font-size=\"12\" fill=\"#666666\" text-anchor=\"middle\">+Y</text>\n",
        center, center - view_radius_mm * scale + 20.0
    ));

    svg.push_str("</svg>");
    svg
}

fn main() {
    println!("Focal Plane Visualization Tool");
    println!("==============================\n");

    // Example 1: Single sensor at origin
    let single_array = SensorArray::single(HWK4123.clone());

    // Example 2: Side-by-side sensors
    let dual_array = SensorArray::new(vec![
        PositionedSensor {
            sensor: IMX455.clone(),
            position: SensorPosition {
                x_mm: -15.0,
                y_mm: 0.0,
            },
        },
        PositionedSensor {
            sensor: IMX455.clone(),
            position: SensorPosition {
                x_mm: 15.0,
                y_mm: 0.0,
            },
        },
    ]);

    // Example 3: Quad array with mixed sensors
    let quad_array = SensorArray::new(vec![
        PositionedSensor {
            sensor: HWK4123.clone(),
            position: SensorPosition {
                x_mm: -12.0,
                y_mm: 12.0,
            },
        },
        PositionedSensor {
            sensor: HWK4123.clone(),
            position: SensorPosition {
                x_mm: 12.0,
                y_mm: 12.0,
            },
        },
        PositionedSensor {
            sensor: IMX455.clone(),
            position: SensorPosition {
                x_mm: -12.0,
                y_mm: -12.0,
            },
        },
        PositionedSensor {
            sensor: IMX455.clone(),
            position: SensorPosition {
                x_mm: 12.0,
                y_mm: -12.0,
            },
        },
    ]);

    // Example 4: Large mosaic with GSENSE sensors
    let mosaic_array = SensorArray::new(vec![
        // Center
        PositionedSensor {
            sensor: GSENSE6510BSI.clone(),
            position: SensorPosition {
                x_mm: 0.0,
                y_mm: 0.0,
            },
        },
        // Surrounding smaller sensors
        PositionedSensor {
            sensor: GSENSE4040BSI.clone(),
            position: SensorPosition {
                x_mm: -30.0,
                y_mm: 0.0,
            },
        },
        PositionedSensor {
            sensor: GSENSE4040BSI.clone(),
            position: SensorPosition {
                x_mm: 30.0,
                y_mm: 0.0,
            },
        },
        PositionedSensor {
            sensor: GSENSE4040BSI.clone(),
            position: SensorPosition {
                x_mm: 0.0,
                y_mm: -30.0,
            },
        },
        PositionedSensor {
            sensor: GSENSE4040BSI.clone(),
            position: SensorPosition {
                x_mm: 0.0,
                y_mm: 30.0,
            },
        },
    ]);

    // Use typical telescope parameters
    let focal_length_mm = 3000.0; // 3 meter focal length
    let field_of_view_deg = 2.0; // 2 degree FOV

    // Generate SVGs for each configuration
    let configs = vec![
        ("single_sensor", single_array),
        ("dual_sensors", dual_array),
        ("quad_array", quad_array),
        ("mosaic_array", mosaic_array),
    ];

    for (name, array) in configs {
        println!("Generating {name} configuration...");
        println!("  Sensors: {}", array.sensor_count());
        println!("  Total pixels: {}", array.total_pixel_count());

        if let Some(aabb) = array.total_aabb_mm() {
            println!(
                "  Total area: {:.1}mm x {:.1}mm",
                aabb.2 - aabb.0,
                aabb.3 - aabb.1
            );
        }

        let svg = create_focal_plane_svg(
            &array,
            focal_length_mm,
            field_of_view_deg,
            800, // Output size in pixels
        );

        let filename = format!("focal_plane_{name}.svg");
        fs::write(&filename, svg).expect("Failed to write SVG file");
        println!("  Saved to: {filename}\n");
    }

    println!(
        "Done! Open the SVG files in a browser or image viewer to see the focal plane layouts."
    );
}

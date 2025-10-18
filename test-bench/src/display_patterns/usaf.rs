use crate::display_assets::Assets;
use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb};
use std::sync::Arc;
use tiny_skia::{Pixmap, Transform};
use usvg::{fontdb, Options, Tree};

pub fn generate(width: u32, height: u32) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let asset_data =
        Assets::get("usaf-1951.svg").context("Failed to load embedded USAF-1951 SVG")?;
    let svg_data = std::str::from_utf8(&asset_data.data).context("SVG file is not valid UTF-8")?;

    let mut fontdb = fontdb::Database::new();
    fontdb.load_system_fonts();

    let options = Options {
        fontdb: Arc::new(fontdb),
        font_family: "DejaVu Sans".to_string(),
        text_rendering: usvg::TextRendering::GeometricPrecision,
        shape_rendering: usvg::ShapeRendering::GeometricPrecision,
        image_rendering: usvg::ImageRendering::OptimizeQuality,
        ..Default::default()
    };

    let svg_tree = Tree::from_str(svg_data, &options).context("Failed to parse SVG")?;

    let svg_size = svg_tree.size();
    let scale_x = width as f32 / svg_size.width();
    let scale_y = height as f32 / svg_size.height();
    let scale = scale_x.min(scale_y);

    let transform = Transform::from_scale(scale, scale);

    let mut pixmap = Pixmap::new(width, height).context("Failed to create pixmap")?;

    pixmap.fill(tiny_skia::Color::WHITE);

    resvg::render(&svg_tree, transform, &mut pixmap.as_mut());

    let mut img = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            if let Some(pixel) = pixmap.pixel(x, y) {
                img.put_pixel(
                    x,
                    y,
                    Rgb([255 - pixel.red(), 255 - pixel.green(), 255 - pixel.blue()]),
                );
            }
        }
    }

    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usaf_pattern_generation() {
        let result = generate(1024, 768);
        if let Err(e) = &result {
            panic!("USAF pattern generation failed: {e}");
        }

        let img = result.unwrap();
        assert_eq!(img.width(), 1024);
        assert_eq!(img.height(), 768);
    }
}

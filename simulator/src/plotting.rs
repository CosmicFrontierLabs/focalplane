//! Plot-to-PNG helper that pairs `plotters::BitMapBackend` with the `image`
//! crate for encoding.
//!
//! Plotters' built-in PNG saving (`BitMapBackend::new`) requires its
//! `bitmap_encoder` feature, which hard-pulls `image 0.24` via
//! `plotters-bitmap 0.3.7`. To keep the dependency tree on a single `image`
//! version, we disable that feature and drive the buffer + encoder path here.

use std::error::Error;
use std::path::Path;

use plotters::coord::Shift;
use plotters::prelude::*;

/// Draw a plot into an RGB8 buffer and save it to `path` as PNG.
pub fn save_plot_png<P, F>(path: P, size: (u32, u32), draw: F) -> Result<(), Box<dyn Error>>
where
    P: AsRef<Path>,
    F: FnOnce(&DrawingArea<BitMapBackend<'_>, Shift>) -> Result<(), Box<dyn Error>>,
{
    let (w, h) = size;
    let mut buffer = vec![0u8; (w as usize) * (h as usize) * 3];
    {
        let root = BitMapBackend::with_buffer(&mut buffer, size).into_drawing_area();
        draw(&root)?;
        root.present()?;
    }
    image::save_buffer(path, &buffer, w, h, image::ColorType::Rgb8)?;
    Ok(())
}

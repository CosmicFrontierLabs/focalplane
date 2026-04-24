//! Context-view previsualization renderer.
//!
//! Produces an RGB8 PNG per trajectory frame showing the star field, the
//! focal-plane sensor outlines, and a reticle at the boresight. The view
//! is rendered in body-frame millimeters: the boresight sits at image
//! center, sensors are drawn at their fixed `(x_mm, y_mm)` positions, and
//! stars are projected through the orientation so they appear to swirl
//! around the instrument as the telescope rotates.
//!
//! Star sizes are inflated well above their physical PSF so they remain
//! visible at the coarse pixel pitch of the 4K context image. Brighter
//! stars render as larger, more saturated Gaussian blobs.

use std::path::Path;

use ab_glyph::{FontRef, PxScale};
use image::{ImageBuffer, Rgb};
use imageproc::drawing::{draw_text_mut, text_size};
use nalgebra::UnitQuaternion;
use ndarray::Array2;
use once_cell::sync::Lazy;
use shared::units::LengthExt;
use starfield::catalogs::StarData;

use crate::hardware::satellite::FocalPlaneConfig;
use crate::sims::orientation::boresight_of;
use crate::sims::roi_render::RoiAnchor;
use crate::sims::trajectory::TrajectoryError;

type RgbImage = ImageBuffer<Rgb<u8>, Vec<u8>>;

/// Embedded DejaVu Sans Mono Bold — used for all context-view labels.
/// Bold weight keeps thin strokes readable at small px scales.
static FONT_DATA: &[u8] = include_bytes!("../../assets/fonts/DejaVuSansMono-Bold.ttf");
static FONT: Lazy<FontRef<'static>> =
    Lazy::new(|| FontRef::try_from_slice(FONT_DATA).expect("bundled font parses"));

/// Configuration for the context-view renderer.
#[derive(Debug, Clone)]
pub struct ContextRenderConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Fraction of each edge reserved as empty margin. A value of `0.05`
    /// keeps the focal-plane AABB in the inner 90% of the canvas.
    pub padding_fraction: f64,
    /// Star Gaussian radius (pixels) at the dim end.
    pub star_radius_min_px: f64,
    /// Star Gaussian radius (pixels) at the bright end.
    pub star_radius_max_px: f64,
    /// Magnitude mapped to `star_radius_max_px`.
    pub mag_bright: f64,
    /// Magnitude mapped to `star_radius_min_px`; stars dimmer than this
    /// are clamped, and stars with an unknown magnitude get the midpoint.
    pub mag_dim: f64,
    /// Stars at or brighter than this magnitude get a deep-blue ring,
    /// marking them as usable for tracking.
    pub tracking_mag_limit: f64,
    /// Stars at or brighter than this magnitude (but dimmer than
    /// `tracking_mag_limit`) get a light-blue ring, marking them as
    /// usable for astrometry only.
    pub astrometry_mag_limit: f64,
    /// Ring radius in pixels for tracking stars.
    pub tracking_ring_radius_px: i32,
    /// Ring radius in pixels for astrometry stars.
    pub astrometry_ring_radius_px: i32,
}

/// Per-call inputs for rendering a zoom-panel "region of interest"
/// onto the context view. The panel is physically rendered via
/// [`crate::sims::roi_render::render_roi_patch`] and composited as a
/// nearest-neighbor upsample framed in red, with a matching red box
/// drawn on the parent sensor's outline to show the source location.
pub struct RoiOverlay {
    /// Sensor-pixel anchor selected once per run (see
    /// [`crate::sims::roi_render::pick_roi_anchor`]).
    pub anchor: RoiAnchor,
    /// Pre-rendered u16 patch (shape `(size_px, size_px)`), typically
    /// the output of [`crate::sims::roi_render::render_roi_patch`] for
    /// this frame.
    pub patch: Array2<u16>,
    /// Nearest-neighbor zoom factor: the panel is drawn at
    /// `anchor.size_px * zoom` pixels on each side.
    pub zoom: u32,
    /// Fixed `(black, white)` stretch for the u16 patch. If `None`,
    /// the overlay autoscales per-frame; pin the value (typically
    /// taken from the first frame) to keep intensity changes between
    /// frames visible rather than re-normalized away.
    pub display_range: Option<(u16, u16)>,
}

impl Default for ContextRenderConfig {
    fn default() -> Self {
        Self {
            width: 3840,
            height: 2160,
            padding_fraction: 0.05,
            star_radius_min_px: 0.6,
            star_radius_max_px: 7.0,
            mag_bright: 2.0,
            mag_dim: 12.0,
            tracking_mag_limit: 11.0,
            astrometry_mag_limit: 14.0,
            tracking_ring_radius_px: 4,
            astrometry_ring_radius_px: 3,
        }
    }
}

/// Render a single context-view frame to `output_path`.
///
/// `roi_overlay`, if `Some`, also renders a physically correct zoom
/// panel for the chosen sensor anchor and composites it onto the
/// output, along with a matching red source-region box on the
/// sensor's outline.
pub fn render_context_frame(
    orientation: &UnitQuaternion<f64>,
    stars: &[StarData],
    fp: &FocalPlaneConfig,
    config: &ContextRenderConfig,
    roi_overlay: Option<&RoiOverlay>,
    output_path: &Path,
) -> Result<(), TrajectoryError> {
    let (width, height) = (config.width, config.height);
    if width == 0 || height == 0 {
        return Err(TrajectoryError::ImageWrite(
            "context width/height must be > 0".into(),
        ));
    }
    let mut img: RgbImage = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));

    let (aabb_min_x, aabb_min_y, aabb_max_x, aabb_max_y) =
        fp.total_aabb_mm().unwrap_or((-10.0, -10.0, 10.0, 10.0));
    // Pick one mm_per_px that keeps both AABB dimensions inside the padded
    // interior of the frame. Taking the larger ratio is what guarantees the
    // whole focal plane fits — the other axis ends up with extra breathing
    // room on both sides.
    let pad = config.padding_fraction.clamp(0.0, 0.45);
    let usable_w_px = (width as f64) * (1.0 - 2.0 * pad);
    let usable_h_px = (height as f64) * (1.0 - 2.0 * pad);
    let aabb_w_mm = (aabb_max_x - aabb_min_x).max(f64::EPSILON);
    let aabb_h_mm = (aabb_max_y - aabb_min_y).max(f64::EPSILON);
    let mm_per_px = (aabb_w_mm / usable_w_px).max(aabb_h_mm / usable_h_px);
    let center_x_px = width as f64 / 2.0;
    let center_y_px = height as f64 / 2.0;

    // `sky_to_mm` returns coordinates anchored so that the boresight lands
    // at the AABB center. Subtract it so image center == boresight.
    let (abc_x, abc_y) = (
        (aabb_min_x + aabb_max_x) / 2.0,
        (aabb_min_y + aabb_max_y) / 2.0,
    );

    let mm_to_px = |x_mm: f64, y_mm: f64| -> (f64, f64) {
        let dx = x_mm - abc_x;
        let dy = y_mm - abc_y;
        // Image Y axis points down; focal-plane Y points up.
        let px = center_x_px + dx / mm_per_px;
        let py = center_y_px - dy / mm_per_px;
        (px, py)
    };

    for star in stars {
        let Some((x_mm, y_mm)) = fp.sky_to_mm(&star.position, orientation) else {
            continue;
        };
        let (px, py) = mm_to_px(x_mm, y_mm);
        // Skip stars comfortably outside the canvas. The 20 px slack covers
        // the Gaussian tail so a star just off-canvas still paints its halo.
        if px < -20.0 || px > width as f64 + 20.0 || py < -20.0 || py > height as f64 + 20.0 {
            continue;
        }
        let radius = mag_to_radius(Some(star.magnitude), config);
        let intensity = mag_to_intensity(Some(star.magnitude), config);
        let (r_mul, g_mul, b_mul) = bv_to_rgb(star.b_v);
        let to_byte = |c: f64| (255.0 * intensity * c).round().clamp(0.0, 255.0) as u8;
        splat_gaussian(
            &mut img,
            (px, py),
            radius,
            Rgb([to_byte(r_mul), to_byte(g_mul), to_byte(b_mul)]),
        );
        // Only ring stars that actually land on a sensor — the ring
        // indicates usability for tracking/astrometry, which requires
        // the star to hit a detector.
        if fp.array.mm_to_pixel(x_mm, y_mm).is_some() {
            draw_availability_ring(&mut img, (px, py), star.magnitude, config);
        }
    }

    // Sensor outlines in bright green. `x_mm`/`y_mm` are sensor centers,
    // so walk the four corners using the sensor's own width/height.
    let green = Rgb([40, 255, 120]);
    for ps in &fp.array.sensors {
        let (w_len, h_len) = ps.sensor.dimensions.get_width_height();
        let half_w = w_len.as_millimeters() / 2.0;
        let half_h = h_len.as_millimeters() / 2.0;
        let (cx, cy) = (ps.position.x_mm, ps.position.y_mm);
        let corners = [
            mm_to_px(cx - half_w, cy + half_h),
            mm_to_px(cx + half_w, cy + half_h),
            mm_to_px(cx + half_w, cy - half_h),
            mm_to_px(cx - half_w, cy - half_h),
        ];
        for i in 0..4 {
            let (x0, y0) = corners[i];
            let (x1, y1) = corners[(i + 1) % 4];
            draw_line(
                &mut img,
                (x0 as i32, y0 as i32),
                (x1 as i32, y1 as i32),
                green,
            );
        }
    }

    // Reticle at image center. Since we render in body frame, the reticle
    // arms naturally align with the instrument axes and rotate with the
    // star field as the telescope rolls. Base the arm length on the
    // shorter image dimension so aspect-ratio changes don't inflate it.
    let red = Rgb([255, 90, 90]);
    let short_side = width.min(height) as i32;
    let arm_len = short_side / 40;
    let gap = short_side / 200;
    let cx_i = center_x_px as i32;
    let cy_i = center_y_px as i32;
    draw_reticle(&mut img, (cx_i, cy_i), arm_len.max(8), gap.max(2), red);

    // Reticle labels — stacked above the right-shooting horizontal arm,
    // both starting where the arm starts (just past the center gap).
    // Top line: "NON-ANALYTIC". Bottom line: current boresight RA/Dec.
    let label_scale = PxScale::from((short_side as f32 / 80.0).max(14.0));
    let tag_text = "NON-ANALYTIC";
    let (_tag_w, tag_h) = text_size(label_scale, &*FONT, tag_text);
    let bore = boresight_of(orientation);
    let ra_text = format!("RA:  {:10.2}", bore.ra_degrees());
    let dec_text = format!("DEC: {:+10.2}", bore.dec_degrees());
    let label_x = cx_i + gap;
    // Stack, top-to-bottom:
    //   NON-ANALYTIC
    //   RA:  <value>
    //   DEC: <value>
    // each line separated from the next by 2 px, with the bottom of
    // the DEC line sitting a few px above the right reticle arm.
    let dec_y = cy_i - tag_h as i32 - 8;
    let ra_y = dec_y - tag_h as i32 - 2;
    let tag_y = ra_y - tag_h as i32 - 2;
    draw_text_mut(&mut img, red, label_x, tag_y, label_scale, &*FONT, tag_text);
    draw_text_mut(&mut img, red, label_x, ra_y, label_scale, &*FONT, &ra_text);
    draw_text_mut(
        &mut img,
        red,
        label_x,
        dec_y,
        label_scale,
        &*FONT,
        &dec_text,
    );

    // Optional ROI zoom panel: draws a physically correct oversampled
    // patch of a chosen sensor region in an auto-picked empty corner,
    // along with a red rectangle on the parent sensor outline marking
    // where the ROI was sampled from.
    if let Some(overlay) = roi_overlay {
        composite_roi_overlay(&mut img, fp, overlay, &mm_to_px)?;
    }

    img.save(output_path)
        .map_err(|e| TrajectoryError::ImageWrite(e.to_string()))?;
    Ok(())
}

/// Paint the ROI zoom panel and its source-location marker onto the
/// context image.
fn composite_roi_overlay(
    img: &mut RgbImage,
    fp: &FocalPlaneConfig,
    overlay: &RoiOverlay,
    mm_to_px: &dyn Fn(f64, f64) -> (f64, f64),
) -> Result<(), TrajectoryError> {
    let red = Rgb([255, 80, 80]);

    // Draw the source-region box on the sensor outline. We convert the
    // ROI's sensor-pixel bbox into sensor-local mm (pixel_size × px),
    // add the sensor's mm-space offset, and map to context image pixels.
    let sensor_idx = overlay.anchor.sensor_idx;
    if let Some(ps) = fp.array.sensors.get(sensor_idx) {
        let pixel_size_mm = ps.sensor.dimensions.pixel_size().as_millimeters();
        let (sensor_w_len, sensor_h_len) = ps.sensor.dimensions.get_width_height();
        let half_w = sensor_w_len.as_millimeters() / 2.0;
        let half_h = sensor_h_len.as_millimeters() / 2.0;
        let size_px = overlay.anchor.size_px as f64;
        let (cx_px, cy_px) = overlay.anchor.center_px;
        let (w_px, h_px) = ps.sensor.dimensions.get_pixel_width_height();
        // Sensor-pixel (0,0) is top-left; focal-plane mm +y is up. Invert y.
        let mm_from_pixel = |px: f64, py: f64| -> (f64, f64) {
            let x_mm = ps.position.x_mm - half_w + px * pixel_size_mm;
            let y_mm = ps.position.y_mm + half_h - py * pixel_size_mm;
            (x_mm, y_mm)
        };
        let (lo_x_mm, lo_y_mm) = mm_from_pixel(cx_px - size_px / 2.0, cy_px + size_px / 2.0);
        let (hi_x_mm, hi_y_mm) = mm_from_pixel(cx_px + size_px / 2.0, cy_px - size_px / 2.0);
        let (x0_px, y0_px) = mm_to_px(lo_x_mm, lo_y_mm);
        let (x1_px, y1_px) = mm_to_px(hi_x_mm, hi_y_mm);
        // Axis-aligned rectangle in image pixels.
        draw_rect_outline(
            img,
            (x0_px.min(x1_px) as i32, y0_px.min(y1_px) as i32),
            (x0_px.max(x1_px) as i32, y0_px.max(y1_px) as i32),
            red,
        );
        let _ = (w_px, h_px);
    }

    // Upsample the patch nearest-neighbor to zoom × size on each side.
    let panel_side = overlay.anchor.size_px as u32 * overlay.zoom.max(1);
    let (width_f, height_f) = (img.width() as f64, img.height() as f64);
    // Auto-pick: prefer the corner whose 5%-margin rectangle fits the
    // panel entirely. Try bottom-right, bottom-left, top-right,
    // top-left in order.
    let margin = ((img.width().min(img.height()) as f64) * 0.02).round() as i32;
    let candidates: &[(i32, i32)] = &[
        (
            img.width() as i32 - panel_side as i32 - margin,
            img.height() as i32 - panel_side as i32 - margin,
        ),
        (margin, img.height() as i32 - panel_side as i32 - margin),
        (img.width() as i32 - panel_side as i32 - margin, margin),
        (margin, margin),
    ];
    let (panel_x, panel_y) = *candidates.first().unwrap_or(&(0, 0));
    let (width_i, height_i) = (img.width() as i32, img.height() as i32);
    // Clamp so even on tiny test canvases we don't try to draw off-image.
    let panel_x = panel_x.max(0).min(width_i - panel_side as i32 - 1);
    let panel_y = panel_y.max(0).min(height_i - panel_side as i32 - 1);

    // Normalize the u16 patch to 0..=255 for display.
    let (lo, hi) = overlay
        .display_range
        .unwrap_or_else(|| autoscale_range(&overlay.patch));
    let span = (hi as f64 - lo as f64).max(1.0);
    let (patch_w, patch_h) = overlay.patch.dim();
    let zoom = overlay.zoom.max(1) as i32;
    for py in 0..(patch_h as i32) {
        for px in 0..(patch_w as i32) {
            let v = overlay.patch[[py as usize, px as usize]] as f64;
            let gray = (((v - lo as f64).clamp(0.0, span)) / span * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let color = Rgb([gray, gray, gray]);
            for dy in 0..zoom {
                for dx in 0..zoom {
                    let x = panel_x + px * zoom + dx;
                    let y = panel_y + py * zoom + dy;
                    if x >= 0 && x < width_i && y >= 0 && y < height_i {
                        img.put_pixel(x as u32, y as u32, color);
                    }
                }
            }
        }
    }

    // Red frame around the panel.
    draw_rect_outline(
        img,
        (panel_x, panel_y),
        (
            panel_x + panel_side as i32 - 1,
            panel_y + panel_side as i32 - 1,
        ),
        red,
    );

    let _ = (width_f, height_f);
    Ok(())
}

/// Draw a 1-pixel-thick rectangle outline between two opposite corners.
fn draw_rect_outline(img: &mut RgbImage, a: (i32, i32), b: (i32, i32), color: Rgb<u8>) {
    let (x0, y0) = (a.0.min(b.0), a.1.min(b.1));
    let (x1, y1) = (a.0.max(b.0), a.1.max(b.1));
    draw_line(img, (x0, y0), (x1, y0), color);
    draw_line(img, (x1, y0), (x1, y1), color);
    draw_line(img, (x1, y1), (x0, y1), color);
    draw_line(img, (x0, y1), (x0, y0), color);
}

/// Cheap 1st-percentile / 99th-percentile autoscale for the ROI patch
/// when the caller hasn't pinned a display range.
fn autoscale_range(patch: &Array2<u16>) -> (u16, u16) {
    let mut vals: Vec<u16> = patch.iter().copied().collect();
    if vals.is_empty() {
        return (0, u16::MAX);
    }
    vals.sort_unstable();
    let lo_idx = vals.len() / 100;
    let hi_idx = vals.len() - 1 - vals.len() / 100;
    (vals[lo_idx], vals[hi_idx])
}

/// Linear position between `mag_bright` (t = 1.0) and `mag_dim` (t = 0.0),
/// clamped to the range. Magnitudes are technically log-flux, but linear
/// visual scaling keeps the full magnitude range readable — a true flux
/// scale blows out bright stars and drops everything else below visibility.
fn mag_to_t(mag: Option<f64>, config: &ContextRenderConfig) -> f64 {
    let m = mag.unwrap_or((config.mag_bright + config.mag_dim) / 2.0);
    let span = config.mag_dim - config.mag_bright;
    if span.abs() < f64::EPSILON {
        return 0.5;
    }
    ((config.mag_dim - m) / span).clamp(0.0, 1.0)
}

fn mag_to_radius(mag: Option<f64>, config: &ContextRenderConfig) -> f64 {
    let t = mag_to_t(mag, config);
    config.star_radius_min_px + t * (config.star_radius_max_px - config.star_radius_min_px)
}

fn mag_to_intensity(mag: Option<f64>, config: &ContextRenderConfig) -> f64 {
    // Floor so dim stars still paint a clearly visible pixel; bright stars
    // saturate. Linear in magnitude-space.
    0.35 + 0.65 * mag_to_t(mag, config)
}

/// Map a stellar B-V color index to an RGB tint multiplier in `[0, 1]`.
/// Missing B-V renders as neutral white. Keyframes approximate stellar
/// colors from hot blue-white (B-V ≈ -0.4) through yellow (sun, ≈ 0.6)
/// to deep red (≈ 2.0).
fn bv_to_rgb(bv: Option<f64>) -> (f64, f64, f64) {
    let Some(bv) = bv else {
        return (1.0, 1.0, 1.0);
    };
    const KEYFRAMES: &[(f64, f64, f64, f64)] = &[
        (-0.40, 0.61, 0.69, 1.00),
        (0.00, 1.00, 1.00, 1.00),
        (0.30, 1.00, 0.98, 0.92),
        (0.60, 1.00, 0.95, 0.82),
        (1.00, 1.00, 0.82, 0.62),
        (1.50, 1.00, 0.68, 0.50),
        (2.00, 1.00, 0.40, 0.30),
    ];
    let bv = bv.clamp(KEYFRAMES[0].0, KEYFRAMES[KEYFRAMES.len() - 1].0);
    for pair in KEYFRAMES.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if bv <= b.0 {
            let t = if (b.0 - a.0).abs() < f64::EPSILON {
                0.0
            } else {
                (bv - a.0) / (b.0 - a.0)
            };
            return (
                a.1 + t * (b.1 - a.1),
                a.2 + t * (b.2 - a.2),
                a.3 + t * (b.3 - a.3),
            );
        }
    }
    (1.0, 1.0, 1.0)
}

fn splat_gaussian(img: &mut RgbImage, center: (f64, f64), radius: f64, color: Rgb<u8>) {
    if radius <= 0.0 {
        return;
    }
    let (cx, cy) = center;
    let sigma = radius / 2.0;
    let max = (radius.ceil() as i32) + 1;
    let (w, h) = (img.width() as i32, img.height() as i32);
    let xi = cx.round() as i32;
    let yi = cy.round() as i32;
    let two_sigma2 = 2.0 * sigma * sigma;
    for dy in -max..=max {
        for dx in -max..=max {
            let x = xi + dx;
            let y = yi + dy;
            if x < 0 || x >= w || y < 0 || y >= h {
                continue;
            }
            let rx = x as f64 - cx;
            let ry = y as f64 - cy;
            let r2 = rx * rx + ry * ry;
            let alpha = (-r2 / two_sigma2).exp();
            if alpha < 0.01 {
                continue;
            }
            let p = img.get_pixel_mut(x as u32, y as u32);
            p[0] = (p[0] as f64 + alpha * color[0] as f64).min(255.0) as u8;
            p[1] = (p[1] as f64 + alpha * color[1] as f64).min(255.0) as u8;
            p[2] = (p[2] as f64 + alpha * color[2] as f64).min(255.0) as u8;
        }
    }
}

/// Bresenham line drawing. Clips to the image bounds.
fn draw_line(img: &mut RgbImage, a: (i32, i32), b: (i32, i32), color: Rgb<u8>) {
    let (x0, y0) = a;
    let (x1, y1) = b;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);
    let (w, h) = (img.width() as i32, img.height() as i32);
    loop {
        if x >= 0 && x < w && y >= 0 && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn draw_reticle(img: &mut RgbImage, center: (i32, i32), arm_len: i32, gap: i32, color: Rgb<u8>) {
    let (cx, cy) = center;
    let w = img.width() as i32;
    let h = img.height() as i32;
    let plot = |img: &mut RgbImage, x: i32, y: i32| {
        if x >= 0 && x < w && y >= 0 && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
    };
    for x in (cx + gap)..=(cx + arm_len) {
        plot(img, x, cy);
    }
    for x in (cx - arm_len)..=(cx - gap) {
        plot(img, x, cy);
    }
    for y in (cy + gap)..=(cy + arm_len) {
        plot(img, cx, y);
    }
    for y in (cy - arm_len)..=(cy - gap) {
        plot(img, cx, y);
    }
}

/// Draw an anti-aliased 1-pixel-thick circle outline. For each pixel in
/// the radius-`r` bounding box, `alpha = clamp(1 - |distance - r|, 0, 1)`
/// — fully opaque at exactly `r`, fading over ±1 px — and the result is
/// alpha-blended over whatever is already in the image. No new deps.
fn draw_circle(img: &mut RgbImage, center: (i32, i32), r: i32, color: Rgb<u8>) {
    if r <= 0 {
        return;
    }
    let (cx, cy) = center;
    let (w, h) = (img.width() as i32, img.height() as i32);
    let r_f = r as f64;
    let reach = r + 2;
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let d = ((dx * dx + dy * dy) as f64).sqrt();
            let alpha = (1.0 - (d - r_f).abs()).clamp(0.0, 1.0);
            if alpha <= 0.0 {
                continue;
            }
            let px = cx + dx;
            let py = cy + dy;
            if px < 0 || px >= w || py < 0 || py >= h {
                continue;
            }
            let p = img.get_pixel_mut(px as u32, py as u32);
            let inv = 1.0 - alpha;
            p[0] = (p[0] as f64 * inv + color[0] as f64 * alpha)
                .round()
                .clamp(0.0, 255.0) as u8;
            p[1] = (p[1] as f64 * inv + color[1] as f64 * alpha)
                .round()
                .clamp(0.0, 255.0) as u8;
            p[2] = (p[2] as f64 * inv + color[2] as f64 * alpha)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
}

/// Mark a star with a colored ring based on how bright it is: deep blue for
/// tracking-grade (≤ `tracking_mag_limit`), light blue for astrometry-only
/// (between tracking and `astrometry_mag_limit`), nothing for dimmer.
fn draw_availability_ring(
    img: &mut RgbImage,
    center: (f64, f64),
    magnitude: f64,
    config: &ContextRenderConfig,
) {
    let (px, py) = center;
    let cx = px.round() as i32;
    let cy = py.round() as i32;
    if magnitude <= config.tracking_mag_limit {
        let deep_blue = Rgb([40, 100, 255]);
        draw_circle(img, (cx, cy), config.tracking_ring_radius_px, deep_blue);
    } else if magnitude <= config.astrometry_mag_limit {
        let light_blue = Rgb([140, 200, 255]);
        draw_circle(img, (cx, cy), config.astrometry_ring_radius_px, light_blue);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::sensor::models::GSENSE4040BSI;
    use crate::hardware::sensor_array::SensorArray;
    use crate::hardware::telescope::TelescopeConfig;
    use crate::sims::orientation::orientation_from_pointing;
    use shared::units::{Length, LengthExt, TemperatureExt};
    use starfield::Equatorial;

    fn tiny_fp() -> FocalPlaneConfig {
        let telescope = TelescopeConfig::new(
            "Test",
            Length::from_meters(0.5),
            Length::from_meters(2.5),
            0.8,
        );
        FocalPlaneConfig::new(
            telescope,
            SensorArray::single(GSENSE4040BSI.clone().with_dimensions(64, 64)),
            simulator_temp(),
        )
    }

    fn simulator_temp() -> shared::units::Temperature {
        shared::units::Temperature::from_celsius(-10.0)
    }

    #[test]
    fn test_mag_to_radius_monotone_and_clamped() {
        let cfg = ContextRenderConfig::default();
        let bright = mag_to_radius(Some(cfg.mag_bright), &cfg);
        let dim = mag_to_radius(Some(cfg.mag_dim), &cfg);
        assert!(bright > dim, "brighter stars must render larger");
        // Bright-end saturation: stars at or beyond mag_bright hit the max radius.
        let too_bright = mag_to_radius(Some(-5.0), &cfg);
        assert!((too_bright - cfg.star_radius_max_px).abs() < 1e-9);
        // Dim-end: beyond-mag_dim stars never render smaller than the floor
        // and never larger than the mag_dim star.
        let too_dim = mag_to_radius(Some(30.0), &cfg);
        assert!(too_dim >= cfg.star_radius_min_px);
        assert!(too_dim <= dim + 1e-9);
    }

    #[test]
    fn test_render_writes_a_png() {
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(10.0, 20.0);
        let orient = orientation_from_pointing(&pointing, 0.0);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ctx.png");
        let cfg = ContextRenderConfig {
            width: 256,
            height: 256,
            ..Default::default()
        };
        render_context_frame(&orient, &[], &fp, &cfg, None, &path).unwrap();
        assert!(path.is_file(), "context PNG must exist");
        let meta = std::fs::metadata(&path).unwrap();
        assert!(meta.len() > 100, "context PNG is suspiciously small");
    }

    #[test]
    fn test_render_draws_sensor_outline() {
        // With a single sensor centered at the array origin on a 256x256
        // canvas, at least a handful of non-black pixels must appear along
        // the outline.
        let fp = tiny_fp();
        let pointing = Equatorial::from_degrees(45.0, 30.0);
        let orient = orientation_from_pointing(&pointing, 0.0);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ctx.png");
        let cfg = ContextRenderConfig {
            width: 256,
            height: 256,
            ..Default::default()
        };
        render_context_frame(&orient, &[], &fp, &cfg, None, &path).unwrap();
        let img = image::open(&path).unwrap().into_rgb8();
        let non_black = img.pixels().filter(|p| p.0 != [0, 0, 0]).count();
        assert!(
            non_black > 50,
            "expected sensor outline + reticle to paint some pixels, got {non_black}"
        );
    }
}

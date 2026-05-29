//! Context-view previsualization renderer.
//!
//! Produces an RGB8 PNG per trajectory frame showing the star field, the
//! focal-plane sensor outlines, and a reticle at the boresight. The view
//! is rendered in body-frame millimeters: the focal-plane AABB is centred
//! in the canvas (so the silicon fills the frame regardless of asymmetric
//! offsets), sensors are drawn at their fixed `(x_mm, y_mm)` positions,
//! and the red reticle marks the boresight projection at the focal-plane
//! origin — which sits off canvas-centre when the array is offset from
//! the optical axis. Stars are projected through the orientation so they
//! appear to swirl around the instrument as the telescope rotates.
//!
//! Star sizes are inflated well above their physical PSF so they remain
//! visible at the coarse pixel pitch of the 4K context image. Brighter
//! stars render as larger, more saturated Gaussian blobs.

use std::path::Path;
use std::sync::LazyLock;

use ab_glyph::{FontRef, PxScale};
use image::{ImageBuffer, Rgb};
use imageproc::drawing::{draw_text_mut, text_size};
use nalgebra::UnitQuaternion;
use shared::units::LengthExt;
use starfield::catalogs::StarData;

use crate::hardware::satellite::FocalPlaneConfig;
use crate::sims::nsa_galaxies::GalaxyInField;
use crate::sims::orientation::boresight_of;
use crate::sims::trajectory::TrajectoryError;

const RAD_TO_ARCSEC: f64 = 206_264.806_247_096_4;

type RgbImage = ImageBuffer<Rgb<u8>, Vec<u8>>;

/// Embedded DejaVu Sans Mono Bold — used for all context-view labels.
/// Bold weight keeps thin strokes readable at small px scales.
static FONT_DATA: &[u8] = include_bytes!("../../assets/fonts/DejaVuSansMono-Bold.ttf");
static FONT: LazyLock<FontRef<'static>> =
    LazyLock::new(|| FontRef::try_from_slice(FONT_DATA).expect("bundled font parses"));

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
/// `galaxies` is a sky-position+ellipse-only list of NSA galaxies in
/// the focal-plane envelope. Each entry is marked with a yellow dotted
/// ellipse sized at the galaxy's Sérsic half-light radius so the
/// preview shows where extended sources will land relative to the
/// silicon. Pass an empty slice to skip the overlay.
pub fn render_context_frame(
    orientation: &UnitQuaternion<f64>,
    stars: &[StarData],
    galaxies: &[GalaxyInField],
    fp: &FocalPlaneConfig,
    config: &ContextRenderConfig,
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

    // Anchor the projection on the AABB center so the silicon fills the
    // canvas regardless of where the boresight (mm origin) sits relative
    // to it. The reticle is drawn at `mm_to_px(0.0, 0.0)` below, so it
    // lands at the true boresight projection — which need not coincide
    // with the image center when the array is offset from the optical
    // axis (e.g. the Spencer mosaic).
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

    // Galaxy overlays: dotted ellipses at the Sérsic half-light radius.
    // Drawn before the sensor outlines so the green box stays on top
    // (helps eyeballing which galaxies actually clip a detector).
    if !galaxies.is_empty() {
        let arcsec_per_mm = fp.plate_scale_rad_per_mm() * RAD_TO_ARCSEC;
        let yellow = Rgb([255, 200, 60]);
        for g in galaxies {
            let Some((x_mm, y_mm)) = fp.sky_to_mm(&g.position, orientation) else {
                continue;
            };
            let (cx, cy) = mm_to_px(x_mm, y_mm);
            // Convert θ_eff (arcsec) → mm at the focal plane → context px.
            let semi_major_mm = g.theta_half_arcsec / arcsec_per_mm;
            let a_px = semi_major_mm / mm_per_px;
            let b_px = a_px * g.axis_ratio;
            // Wider canvas slack than stars: an off-centre ellipse may
            // still poke partway into the canvas.
            let slack = (a_px.max(b_px) + 4.0).ceil();
            if cx + slack < 0.0
                || cx - slack > width as f64
                || cy + slack < 0.0
                || cy - slack > height as f64
            {
                continue;
            }
            // NSA position angle is east-of-north on sky. The focal
            // plane has +y mm = north (up in the canvas after the
            // y-flip in `mm_to_px`), so rotating the ellipse's parametric
            // axes by `pa` gives the correct on-canvas orientation.
            draw_dotted_ellipse(
                &mut img,
                (cx, cy),
                a_px,
                b_px,
                g.position_angle_deg.to_radians(),
                yellow,
            );
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

    // Reticle at the boresight projection. Since we render in body frame,
    // the reticle arms naturally align with the instrument axes and rotate
    // with the star field as the telescope rolls. Base the arm length on
    // the shorter image dimension so aspect-ratio changes don't inflate it.
    let red = Rgb([255, 90, 90]);
    let short_side = width.min(height) as i32;
    let arm_len = short_side / 40;
    let gap = short_side / 200;
    let (bore_px, bore_py) = mm_to_px(0.0, 0.0);
    let cx_i = bore_px.round() as i32;
    let cy_i = bore_py.round() as i32;
    draw_reticle(&mut img, (cx_i, cy_i), arm_len.max(8), gap.max(2), red);

    // Reticle labels — stacked above the right-shooting horizontal arm,
    // both starting where the arm starts (just past the center gap).
    // Top line: "NON-ANALYTIC". Bottom line: current boresight RA/Dec.
    let label_scale = PxScale::from((short_side as f32 / 80.0).max(14.0));
    let tag_text = "NON-ANALYTIC";
    let (_tag_w, tag_h) = text_size(label_scale, &*FONT, tag_text);
    let bore = boresight_of(orientation);
    let radec_text = format!(
        "RA {:7.3}  DEC {:+7.3}",
        bore.ra_degrees(),
        bore.dec_degrees()
    );
    let label_x = cx_i + gap;
    let radec_y = cy_i - tag_h as i32 - 8;
    let tag_y = radec_y - tag_h as i32 - 2;
    draw_text_mut(&mut img, red, label_x, tag_y, label_scale, &*FONT, tag_text);
    draw_text_mut(
        &mut img,
        red,
        label_x,
        radec_y,
        label_scale,
        &*FONT,
        &radec_text,
    );

    img.save(output_path)
        .map_err(|e| TrajectoryError::ImageWrite(e.to_string()))?;
    Ok(())
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

/// Draw a dashed (dotted) outline of an ellipse with semi-axes
/// `a_px` (along the rotated x axis) and `b_px` (along y), rotated by
/// `pa_rad`. The arc is sampled in `n` parametric segments, each
/// segment plotted as a 2x2 block so the dashes stay visible at 4K
/// canvas density. Every other segment is left empty for the dash.
///
/// Falls through silently if either semi-axis is below 1 px — there's
/// nothing to draw at sub-pixel scale.
fn draw_dotted_ellipse(
    img: &mut RgbImage,
    center: (f64, f64),
    a_px: f64,
    b_px: f64,
    pa_rad: f64,
    color: Rgb<u8>,
) {
    let max_axis = a_px.max(b_px);
    if max_axis < 1.0 {
        return;
    }
    let (cx, cy) = center;
    let (sin_pa, cos_pa) = pa_rad.sin_cos();
    // ~one parametric step per outline pixel keeps the dashes from
    // bunching at the apex of an elongated ellipse.
    let n = (std::f64::consts::TAU * max_axis).ceil().max(48.0) as usize;
    let (w, h) = (img.width() as i32, img.height() as i32);
    for i in 0..n {
        // Dotted: two-on, two-off pattern.
        if (i / 2) % 2 == 1 {
            continue;
        }
        let t = (i as f64) * std::f64::consts::TAU / n as f64;
        let (st, ct) = t.sin_cos();
        let lx = a_px * ct;
        let ly = b_px * st;
        // Rotate by `pa`, then flip the vertical contribution because
        // image y points down while focal-plane y points up.
        let dx = lx * cos_pa - ly * sin_pa;
        let dy_canvas = -(lx * sin_pa + ly * cos_pa);
        let x = (cx + dx).round() as i32;
        let y = (cy + dy_canvas).round() as i32;
        for oy in 0..=1 {
            for ox in 0..=1 {
                let px = x + ox;
                let py = y + oy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    img.put_pixel(px as u32, py as u32, color);
                }
            }
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
        render_context_frame(&orient, &[], &[], &fp, &cfg, &path).unwrap();
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
        render_context_frame(&orient, &[], &[], &fp, &cfg, &path).unwrap();
        let img = image::open(&path).unwrap().into_rgb8();
        let non_black = img.pixels().filter(|p| p.0 != [0, 0, 0]).count();
        assert!(
            non_black > 50,
            "expected sensor outline + reticle to paint some pixels, got {non_black}"
        );
    }
}

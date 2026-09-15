//! Informational overlay: an SVG that sits 1:1 on a rendered frame.
//!
//! The overlay is rendered from the same scene parameters as the image
//! (same window, same projector, same epoch) and drawn in the image's
//! pixel coordinates, so `<svg viewBox="0 0 W H">` laid over a `W × H`
//! frame lines up exactly. It carries two things for two audiences:
//!
//! - **For eyes**: a footprint per source (disk, ellipse or point
//!   marker), a label placed by [`layout`] so it never covers a source
//!   or another label, and a leader line when the label had to move
//!   away. Colours and weights come from CSS classes per
//!   [`AnnotationKind`], so a consumer can restyle without touching
//!   geometry.
//! - **For programs**: a `<metadata>` block holding the whole
//!   [`OverlayDocument`] as JSON (every annotation with its pixel
//!   anchor, footprint, placed label box and a structured payload such
//!   as the frame-metadata record it came from), and `id` / `data-*`
//!   attributes on every drawn element linking back to that JSON by
//!   annotation id. The JSON is the contract; the drawing is a view of it.
//!
//! # Pixel convention
//!
//! Image pixel `(i, j)` covers `[i, i+1) × [j, j+1)` in SVG user units,
//! so a source whose centre the renderer reports at pixel-index
//! coordinates `(x, y)` ("pixel index = pixel centre") is drawn at
//! `(x + 0.5, y + 0.5)`. [`Overlay::push`] takes renderer coordinates
//! and applies the shift; the JSON records both.
//!
//! # Text metrics
//!
//! Label widths are estimated as `0.6 × font size` per character, and
//! every `<text>` carries `textLength` with `lengthAdjust`, so a viewer
//! renders each label at exactly the width the layout assumed whatever
//! font it substitutes.

pub mod layout;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use layout::{KeepOut, LabelRequest, LayoutConfig, PlacedLabel};

/// Version tag written into the JSON and the root element.
pub const FORMAT: &str = "focalplane-overlay/1";

/// Namespace for the metadata payload element.
pub const METADATA_NS: &str = "https://github.com/CosmicFrontierLabs/focalplane/overlay/v1";

/// Half-pixel shift from renderer pixel-index coordinates to SVG user
/// units.
pub const PIXEL_CENTRE_OFFSET: f64 = 0.5;

/// What kind of thing an annotation marks; selects the CSS class and the
/// default priority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationKind {
    Sun,
    Planet,
    Moon,
    MinorPlanet,
    Star,
    Galaxy,
    /// A named point on a body's surface.
    Site,
    /// Any other point of interest (centroid, aimpoint, fiducial).
    Marker,
}

impl AnnotationKind {
    /// CSS class suffix.
    pub fn class(self) -> &'static str {
        match self {
            AnnotationKind::Sun => "sun",
            AnnotationKind::Planet => "planet",
            AnnotationKind::Moon => "moon",
            AnnotationKind::MinorPlanet => "minor-planet",
            AnnotationKind::Star => "star",
            AnnotationKind::Galaxy => "galaxy",
            AnnotationKind::Site => "site",
            AnnotationKind::Marker => "marker",
        }
    }

    /// The JSON name (`snake_case`), also used in `data-kind`.
    pub fn name(self) -> &'static str {
        match self {
            AnnotationKind::Sun => "sun",
            AnnotationKind::Planet => "planet",
            AnnotationKind::Moon => "moon",
            AnnotationKind::MinorPlanet => "minor_planet",
            AnnotationKind::Star => "star",
            AnnotationKind::Galaxy => "galaxy",
            AnnotationKind::Site => "site",
            AnnotationKind::Marker => "marker",
        }
    }

    /// Default label priority: resolved bodies first, faint field
    /// sources last.
    pub fn default_priority(self) -> f64 {
        match self {
            AnnotationKind::Sun => 100.0,
            AnnotationKind::Planet => 90.0,
            AnnotationKind::Moon => 80.0,
            AnnotationKind::Site => 70.0,
            AnnotationKind::Marker => 60.0,
            AnnotationKind::MinorPlanet => 40.0,
            AnnotationKind::Galaxy => 30.0,
            AnnotationKind::Star => 20.0,
        }
    }
}

/// The on-sky extent of a source in pixels, centred on the anchor.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum Footprint {
    /// Unresolved: drawn as a small ring of `marker_radius_px`.
    Point { marker_radius_px: f64 },
    /// A disk of `radius_px`.
    Circle { radius_px: f64 },
    /// An ellipse; `position_angle_deg` is the sky position angle of the
    /// major axis, north through east, with the frame north-up east-left.
    Ellipse {
        semi_major_px: f64,
        semi_minor_px: f64,
        position_angle_deg: f64,
    },
}

impl Footprint {
    /// Radius of the disk that contains the footprint (the layout
    /// keep-out).
    pub fn bounding_radius_px(&self) -> f64 {
        match *self {
            Footprint::Point { marker_radius_px } => marker_radius_px,
            Footprint::Circle { radius_px } => radius_px,
            Footprint::Ellipse { semi_major_px, .. } => semi_major_px,
        }
    }
}

/// One annotated thing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    /// Unique within the overlay; becomes the SVG element id suffix.
    pub id: String,
    pub kind: AnnotationKind,
    /// Text drawn next to the source. Empty draws the footprint only.
    pub label: String,
    /// Source centre in renderer pixel-index coordinates.
    pub anchor_px: (f64, f64),
    pub footprint: Footprint,
    /// Layout order; larger is placed first. `None` uses the kind's
    /// default.
    pub priority: Option<f64>,
    /// Structured description of the source for consumers: typically a
    /// serialised frame-metadata record. Free-form JSON.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub payload: serde_json::Value,
}

impl Annotation {
    /// A point or resolved source with no payload.
    pub fn new(
        id: impl Into<String>,
        kind: AnnotationKind,
        label: impl Into<String>,
        anchor_px: (f64, f64),
        footprint: Footprint,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            label: label.into(),
            anchor_px,
            footprint,
            priority: None,
            payload: serde_json::Value::Null,
        }
    }

    /// Attach a serialisable payload.
    pub fn with_payload<T: Serialize>(mut self, payload: &T) -> Self {
        self.payload = serde_json::to_value(payload).unwrap_or(serde_json::Value::Null);
        self
    }

    /// Override the layout priority.
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = Some(priority);
        self
    }

    fn effective_priority(&self) -> f64 {
        self.priority.unwrap_or(self.kind.default_priority())
    }

    /// Anchor in SVG user units.
    pub fn anchor_svg(&self) -> (f64, f64) {
        (
            self.anchor_px.0 + PIXEL_CENTRE_OFFSET,
            self.anchor_px.1 + PIXEL_CENTRE_OFFSET,
        )
    }
}

/// Drawing parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OverlayStyle {
    /// Label font size in pixels.
    pub font_px: f64,
    /// Estimated advance per character as a fraction of `font_px`.
    pub char_width_em: f64,
    /// Padding inside the label box, pixels.
    pub label_pad_px: f64,
    /// Clearance between labels and keep-outs, pixels.
    pub margin_px: f64,
    /// Keep-out radius added around every footprint, pixels.
    pub footprint_clearance_px: f64,
    /// Draw a leader line when the label sits further than this from
    /// its footprint, pixels.
    pub leader_threshold_px: f64,
    /// Stroke width of footprints, pixels.
    pub stroke_px: f64,
    /// Colour per kind, any CSS colour.
    pub colours: BTreeMap<AnnotationKind, String>,
}

impl OverlayStyle {
    /// Style scaled for a frame `height` pixels tall: font 1/40 of the
    /// height, clamped to 9–18 px.
    pub fn for_frame_height(height: f64) -> Self {
        let font_px = (height / 40.0).clamp(9.0, 18.0);
        let colours = [
            (AnnotationKind::Sun, "#ffd166"),
            (AnnotationKind::Planet, "#7dcfff"),
            (AnnotationKind::Moon, "#c0caf5"),
            (AnnotationKind::MinorPlanet, "#ffd600"),
            (AnnotationKind::Star, "#9ece6a"),
            (AnnotationKind::Galaxy, "#bb9af7"),
            (AnnotationKind::Site, "#00dcff"),
            (AnnotationKind::Marker, "#f7768e"),
        ]
        .into_iter()
        .map(|(k, c)| (k, c.to_string()))
        .collect();
        Self {
            font_px,
            char_width_em: 0.6,
            label_pad_px: 2.0,
            margin_px: 3.0,
            footprint_clearance_px: 2.0,
            leader_threshold_px: 6.0,
            stroke_px: 1.0,
            colours,
        }
    }

    /// Label box for `text`.
    fn label_box(&self, text: &str) -> (f64, f64) {
        let chars = text.chars().count().max(1) as f64;
        (
            chars * self.font_px * self.char_width_em + 2.0 * self.label_pad_px,
            self.font_px * 1.25 + 2.0 * self.label_pad_px,
        )
    }
}

/// Where a label was drawn, in SVG user units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    /// False when the layout could not keep this label off every
    /// source, label and frame edge; drawn faded.
    pub clean: bool,
    /// A leader line was drawn from the box to the footprint.
    pub leader: bool,
}

/// The machine-readable content of an overlay: what the `<metadata>`
/// block holds and what a consumer should parse.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OverlayDocument {
    pub format: String,
    pub width_px: usize,
    pub height_px: usize,
    pub pixel_convention: String,
    /// Frame-level provenance supplied by the caller (epoch, instrument,
    /// commit, …), if any.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub frame: serde_json::Value,
    pub style: OverlayStyle,
    pub annotations: Vec<Annotation>,
    /// One entry per annotation with a non-empty label, keyed by id.
    pub labels: BTreeMap<String, LabelBox>,
}

/// An overlay under construction.
#[derive(Clone, Debug)]
pub struct Overlay {
    width_px: usize,
    height_px: usize,
    style: OverlayStyle,
    frame: serde_json::Value,
    annotations: Vec<Annotation>,
}

impl Overlay {
    /// Empty overlay for a `width × height` frame with the default style.
    pub fn new(width_px: usize, height_px: usize) -> Self {
        Self {
            width_px,
            height_px,
            style: OverlayStyle::for_frame_height(height_px as f64),
            frame: serde_json::Value::Null,
            annotations: Vec::new(),
        }
    }

    /// Replace the style.
    pub fn with_style(mut self, style: OverlayStyle) -> Self {
        self.style = style;
        self
    }

    /// Attach frame-level provenance (anything serialisable).
    pub fn with_frame<T: Serialize>(mut self, frame: &T) -> Self {
        self.frame = serde_json::to_value(frame).unwrap_or(serde_json::Value::Null);
        self
    }

    /// Add an annotation. Anchors outside the frame are kept (their
    /// footprint may still intrude) but never labelled.
    pub fn push(&mut self, annotation: Annotation) -> &mut Self {
        self.annotations.push(annotation);
        self
    }

    pub fn annotations(&self) -> &[Annotation] {
        &self.annotations
    }

    pub fn style(&self) -> &OverlayStyle {
        &self.style
    }

    fn in_frame(&self, (x, y): (f64, f64)) -> bool {
        x >= 0.0 && y >= 0.0 && x <= self.width_px as f64 && y <= self.height_px as f64
    }

    /// Run the layout and return the document (JSON side).
    pub fn document(&self) -> OverlayDocument {
        let (labels, _) = self.layout();
        OverlayDocument {
            format: FORMAT.to_string(),
            width_px: self.width_px,
            height_px: self.height_px,
            pixel_convention: format!(
                "image pixel (i, j) spans [i, i+1) x [j, j+1) in SVG user units; anchor_px \
                 are renderer pixel-index coordinates (pixel index = pixel centre), drawn at \
                 anchor_px + {PIXEL_CENTRE_OFFSET}"
            ),
            frame: self.frame.clone(),
            style: self.style.clone(),
            annotations: self.annotations.clone(),
            labels,
        }
    }

    /// Place every label; returns the boxes keyed by annotation id and
    /// the raw placements.
    fn layout(&self) -> (BTreeMap<String, LabelBox>, Vec<(usize, PlacedLabel)>) {
        let keepouts: Vec<KeepOut> = self
            .annotations
            .iter()
            .map(|a| {
                let (cx, cy) = a.anchor_svg();
                KeepOut {
                    cx,
                    cy,
                    radius: a.footprint.bounding_radius_px() + self.style.footprint_clearance_px,
                }
            })
            .collect();
        let labelled: Vec<usize> = self
            .annotations
            .iter()
            .enumerate()
            .filter(|(_, a)| !a.label.is_empty() && self.in_frame(a.anchor_svg()))
            .map(|(i, _)| i)
            .collect();
        let requests: Vec<LabelRequest> = labelled
            .iter()
            .map(|&i| {
                let a = &self.annotations[i];
                let (w, h) = self.style.label_box(&a.label);
                LabelRequest {
                    anchor: a.anchor_svg(),
                    anchor_radius: a.footprint.bounding_radius_px()
                        + self.style.footprint_clearance_px,
                    w,
                    h,
                    priority: a.effective_priority(),
                }
            })
            .collect();
        let mut config = LayoutConfig::for_frame(self.width_px as f64, self.height_px as f64);
        config.margin = self.style.margin_px;
        let placed = layout::place_labels(&requests, &keepouts, &config);

        let mut labels = BTreeMap::new();
        let mut raw = Vec::with_capacity(placed.len());
        for p in placed {
            let idx = labelled[p.request];
            let a = &self.annotations[idx];
            labels.insert(
                a.id.clone(),
                LabelBox {
                    x: p.rect.x,
                    y: p.rect.y,
                    w: p.rect.w,
                    h: p.rect.h,
                    clean: p.clean,
                    leader: p.gap > self.style.leader_threshold_px,
                },
            );
            raw.push((idx, p));
        }
        (labels, raw)
    }

    /// The SVG text.
    pub fn to_svg(&self) -> String {
        let doc = self.document();
        let (w, h) = (self.width_px, self.height_px);
        let s = &self.style;
        let mut out = String::with_capacity(4096 + 512 * self.annotations.len());

        let _ = writeln!(out, r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        let _ = writeln!(
            out,
            r#"<svg xmlns="http://www.w3.org/2000/svg" xmlns:fp="{METADATA_NS}" width="{w}" height="{h}" viewBox="0 0 {w} {h}" data-format="{FORMAT}">"#
        );
        let _ = writeln!(out, "  <title>focalplane overlay, {w}x{h} px</title>");
        let _ = writeln!(out, "  <desc>{}</desc>", escape(&doc.pixel_convention));
        // JSON payload for consumers. The element name is namespaced so
        // generic SVG tools ignore it; the JSON is XML-escaped text.
        let json = serde_json::to_string(&doc).unwrap_or_default();
        let _ = writeln!(
            out,
            r#"  <metadata><fp:overlay type="application/json">{}</fp:overlay></metadata>"#,
            escape(&json)
        );

        // Styles: one class per kind, plus shared geometry classes.
        let _ = writeln!(out, "  <style>");
        let _ = writeln!(
            out,
            "    .footprint {{ fill: none; stroke-width: {}px; vector-effect: non-scaling-stroke; }}",
            s.stroke_px
        );
        let _ = writeln!(
            out,
            "    .label {{ font-family: 'DejaVu Sans Mono', 'Menlo', 'Consolas', monospace; font-size: {}px; paint-order: stroke fill; stroke: rgba(0,0,0,0.85); stroke-width: 3px; stroke-linejoin: round; }}",
            s.font_px
        );
        let _ = writeln!(out, "    .label.unclean {{ opacity: 0.55; }}");
        let _ = writeln!(
            out,
            "    .leader {{ fill: none; stroke-width: 1px; stroke-opacity: 0.7; }}"
        );
        for (kind, colour) in &s.colours {
            let _ = writeln!(
                out,
                "    .kind-{c} {{ stroke: {colour}; fill: {colour}; }} .footprint.kind-{c} {{ fill: none; }}",
                c = kind.class()
            );
        }
        let _ = writeln!(out, "  </style>");

        // Footprints.
        let _ = writeln!(out, r#"  <g id="footprints">"#);
        for a in &self.annotations {
            let (cx, cy) = a.anchor_svg();
            let attrs = format!(
                r#"id="fp-{id}" class="footprint kind-{c}" data-id="{id}" data-kind="{k}" data-label="{l}""#,
                id = escape(&a.id),
                c = a.kind.class(),
                k = a.kind.name(),
                l = escape(&a.label)
            );
            let title = format!("<title>{}</title>", escape(&a.label));
            match a.footprint {
                Footprint::Point { marker_radius_px } => {
                    let _ = writeln!(
                        out,
                        r#"    <circle {attrs} cx="{cx:.2}" cy="{cy:.2}" r="{marker_radius_px:.2}">{title}</circle>"#
                    );
                }
                Footprint::Circle { radius_px } => {
                    let _ = writeln!(
                        out,
                        r#"    <circle {attrs} cx="{cx:.2}" cy="{cy:.2}" r="{radius_px:.2}">{title}</circle>"#
                    );
                }
                Footprint::Ellipse {
                    semi_major_px,
                    semi_minor_px,
                    position_angle_deg,
                } => {
                    // Sky PA north-through-east; frame north up, east left,
                    // so the major axis direction in image coordinates is
                    // (−sin PA, −cos PA): an SVG rotation of −(90° − PA)
                    // applied to an ellipse whose rx is the major axis.
                    let rotate = -(90.0 - position_angle_deg);
                    let _ = writeln!(
                        out,
                        r#"    <ellipse {attrs} cx="{cx:.2}" cy="{cy:.2}" rx="{semi_major_px:.2}" ry="{semi_minor_px:.2}" transform="rotate({rotate:.2} {cx:.2} {cy:.2})">{title}</ellipse>"#
                    );
                }
            }
        }
        let _ = writeln!(out, "  </g>");

        // Leaders and labels.
        let _ = writeln!(out, r#"  <g id="leaders">"#);
        for (id, label) in &doc.labels {
            if !label.leader {
                continue;
            }
            let a = self
                .annotations
                .iter()
                .find(|a| &a.id == id)
                .expect("label id comes from annotations");
            let (ax, ay) = a.anchor_svg();
            let r = a.footprint.bounding_radius_px();
            // From the label box edge nearest the anchor to the footprint
            // edge along the same line.
            let (bx, by) = nearest_point_on_rect(label, ax, ay);
            let (dx, dy) = (ax - bx, ay - by);
            let d = (dx * dx + dy * dy).sqrt().max(1e-9);
            let (ex, ey) = (ax - dx / d * r, ay - dy / d * r);
            let _ = writeln!(
                out,
                r#"    <line id="leader-{id}" class="leader kind-{c}" data-id="{id}" x1="{bx:.2}" y1="{by:.2}" x2="{ex:.2}" y2="{ey:.2}"/>"#,
                id = escape(id),
                c = a.kind.class()
            );
        }
        let _ = writeln!(out, "  </g>");

        let _ = writeln!(out, r#"  <g id="labels">"#);
        for (id, label) in &doc.labels {
            let a = self
                .annotations
                .iter()
                .find(|a| &a.id == id)
                .expect("label id comes from annotations");
            let text_w = label.w - 2.0 * s.label_pad_px;
            let x = label.x + s.label_pad_px;
            // Baseline roughly 0.8 em below the top of the text line.
            let y = label.y + s.label_pad_px + s.font_px * 0.95;
            let _ = writeln!(
                out,
                r#"    <text id="label-{id}" class="label kind-{c}{u}" data-id="{id}" x="{x:.2}" y="{y:.2}" textLength="{text_w:.2}" lengthAdjust="spacingAndGlyphs">{t}</text>"#,
                id = escape(id),
                c = a.kind.class(),
                u = if label.clean { "" } else { " unclean" },
                t = escape(&a.label)
            );
        }
        let _ = writeln!(out, "  </g>");
        let _ = writeln!(out, "</svg>");
        out
    }
}

/// Point on the label box boundary nearest to `(px, py)`.
fn nearest_point_on_rect(label: &LabelBox, px: f64, py: f64) -> (f64, f64) {
    let x = px.clamp(label.x, label.x + label.w);
    let y = py.clamp(label.y, label.y + label.h);
    (x, y)
}

/// XML text/attribute escaping.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

/// Parse the JSON document back out of an overlay SVG.
pub fn document_from_svg(svg: &str) -> Option<OverlayDocument> {
    let start = svg.find(r#"<fp:overlay type="application/json">"#)?
        + r#"<fp:overlay type="application/json">"#.len();
    let end = start + svg[start..].find("</fp:overlay>")?;
    let json = unescape(&svg[start..end]);
    serde_json::from_str(&json).ok()
}

fn unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::layout::Rect;
    use super::*;

    fn sample() -> Overlay {
        let mut ov = Overlay::new(512, 512);
        ov.push(
            Annotation::new(
                "earth",
                AnnotationKind::Planet,
                "Earth",
                (255.5, 255.5),
                Footprint::Circle { radius_px: 71.0 },
            )
            .with_payload(&serde_json::json!({"range_au": 0.9486, "phase_angle_deg": 98.0})),
        );
        ov.push(Annotation::new(
            "palomar",
            AnnotationKind::Site,
            "Palomar",
            (300.7, 235.9),
            Footprint::Point {
                marker_radius_px: 4.0,
            },
        ));
        ov.push(Annotation::new(
            "mp-03390",
            AnnotationKind::MinorPlanet,
            "(3390) Demanet V18.9",
            (60.0, 400.0),
            Footprint::Point {
                marker_radius_px: 5.0,
            },
        ));
        ov.push(Annotation::new(
            "ngc1",
            AnnotationKind::Galaxy,
            "NGC 1 <test> & \"quotes\"",
            (420.0, 100.0),
            Footprint::Ellipse {
                semi_major_px: 20.0,
                semi_minor_px: 8.0,
                position_angle_deg: 30.0,
            },
        ));
        ov.push(Annotation::new(
            "offscreen",
            AnnotationKind::Star,
            "never labelled",
            (-40.0, 10.0),
            Footprint::Point {
                marker_radius_px: 3.0,
            },
        ));
        ov
    }

    #[test]
    fn labels_avoid_every_footprint_and_each_other() {
        let ov = sample();
        let doc = ov.document();
        assert_eq!(doc.labels.len(), 4, "off-frame anchor gets no label");
        let boxes: Vec<(&String, &LabelBox)> = doc.labels.iter().collect();
        for (i, (id, l)) in boxes.iter().enumerate() {
            assert!(l.clean, "{id} not clean: {l:?}");
            let r = Rect {
                x: l.x,
                y: l.y,
                w: l.w,
                h: l.h,
            };
            for a in ov.annotations() {
                let (cx, cy) = a.anchor_svg();
                assert_eq!(
                    r.overlap_area_with_disk(cx, cy, a.footprint.bounding_radius_px()),
                    0.0,
                    "{id} covers {}",
                    a.id
                );
            }
            for (jd, m) in boxes.iter().skip(i + 1) {
                let o = Rect {
                    x: m.x,
                    y: m.y,
                    w: m.w,
                    h: m.h,
                };
                assert_eq!(r.overlap_area(&o), 0.0, "{id} overlaps {jd}");
            }
        }
        // The planet label is placed first and lands outside its disk.
        let earth = &doc.labels["earth"];
        assert!(
            earth.x > 256.0 + 71.0
                || earth.x + earth.w < 256.0 - 71.0
                || earth.y + earth.h < 256.0 - 71.0
                || earth.y > 256.0 + 71.0
        );
    }

    #[test]
    fn svg_is_well_formed_and_round_trips_the_document() {
        let ov = sample();
        let svg = ov.to_svg();
        assert!(svg.starts_with("<?xml"));
        assert!(svg.contains(r#"viewBox="0 0 512 512""#));
        assert!(svg.contains(r#"id="fp-earth""#));
        assert!(svg.contains(r#"data-kind="planet""#));
        assert!(svg.contains("textLength="));
        // Escaping: raw angle brackets from the galaxy label never appear
        // outside tags.
        assert!(!svg.contains("NGC 1 <test>"));
        assert!(svg.contains("NGC 1 &lt;test&gt; &amp; &quot;quotes&quot;"));
        // Every open tag closes (crude XML balance check).
        for tag in ["svg", "g", "style", "metadata", "title", "desc"] {
            let opens = svg.matches(&format!("<{tag}")).count();
            let closes = svg.matches(&format!("</{tag}>")).count();
            assert_eq!(opens, closes, "<{tag}> balance");
        }
        let back = document_from_svg(&svg).expect("metadata JSON parses");
        assert_eq!(back, ov.document());
        assert_eq!(back.annotations[0].payload["range_au"], 0.9486);
    }

    #[test]
    fn anchor_shift_is_half_a_pixel() {
        let a = Annotation::new(
            "x",
            AnnotationKind::Marker,
            "",
            (10.0, 20.0),
            Footprint::Point {
                marker_radius_px: 1.0,
            },
        );
        assert_eq!(a.anchor_svg(), (10.5, 20.5));
    }

    #[test]
    fn unlabelled_annotation_still_draws_a_footprint() {
        let mut ov = Overlay::new(64, 64);
        ov.push(Annotation::new(
            "dot",
            AnnotationKind::Star,
            "",
            (32.0, 32.0),
            Footprint::Point {
                marker_radius_px: 2.0,
            },
        ));
        let svg = ov.to_svg();
        assert!(svg.contains(r#"id="fp-dot""#));
        assert!(!svg.contains(r#"id="label-dot""#));
        assert!(ov.document().labels.is_empty());
    }
}

//! Label placement that keeps text off the sources.
//!
//! Every annotated source is a keep-out region (its footprint grown by a
//! margin); the frame edge is a hard wall; labels already placed are
//! obstacles for the ones that follow. Labels are placed greedily in
//! priority order. For each label a ring of candidate boxes around the
//! anchor is scored, nearest ring first, and the best candidate wins:
//!
//! ```text
//! score = 1000 · (out-of-frame area)
//!       +  100 · (area overlapping a placed label)
//!       +   10 · (area overlapping a keep-out)
//!       +    1 · (distance from the anchor, in font heights)
//!       +  0.5 · (direction preference, right → below → above → left)
//! ```
//!
//! Area terms are in units of the label's own area, so a fully covered
//! label costs 10 for a keep-out and 100 for a collision. A label whose
//! best candidate still overlaps something is placed anyway and flagged
//! `clean = false`; the caller decides whether to draw it faded or drop
//! it. Nothing here knows about SVG; it is pure rectangle geometry.

use std::f64::consts::TAU;

/// Overlap fractions below this are rounding noise from subtracting
/// areas, not contact.
const AREA_EPS: f64 = 1e-9;

/// An axis-aligned box in pixel coordinates (`x`, `y` are the top-left
/// corner; pixel index `i` spans `[i, i + 1)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    /// Box of `w × h` centred on `(cx, cy)`.
    pub fn centred(cx: f64, cy: f64, w: f64, h: f64) -> Self {
        Self {
            x: cx - w / 2.0,
            y: cy - h / 2.0,
            w,
            h,
        }
    }

    pub fn area(&self) -> f64 {
        self.w * self.h
    }

    pub fn centre(&self) -> (f64, f64) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    pub fn right(&self) -> f64 {
        self.x + self.w
    }

    pub fn bottom(&self) -> f64 {
        self.y + self.h
    }

    /// Area shared with `other`.
    pub fn overlap_area(&self, other: &Rect) -> f64 {
        let w = (self.right().min(other.right()) - self.x.max(other.x)).max(0.0);
        let h = (self.bottom().min(other.bottom()) - self.y.max(other.y)).max(0.0);
        w * h
    }

    /// Area of this box outside `frame`.
    pub fn area_outside(&self, frame: &Rect) -> f64 {
        self.area() - self.overlap_area(frame)
    }

    /// Area shared with a disk, by a 6×6 sub-sample of the box.
    pub fn overlap_area_with_disk(&self, cx: f64, cy: f64, r: f64) -> f64 {
        // Cheap rejection: disk bounding box.
        let bbox = Rect::centred(cx, cy, 2.0 * r, 2.0 * r);
        if self.overlap_area(&bbox) == 0.0 {
            return 0.0;
        }
        const N: usize = 6;
        let mut inside = 0usize;
        for i in 0..N {
            for j in 0..N {
                let px = self.x + (i as f64 + 0.5) / N as f64 * self.w;
                let py = self.y + (j as f64 + 0.5) / N as f64 * self.h;
                if (px - cx).powi(2) + (py - cy).powi(2) <= r * r {
                    inside += 1;
                }
            }
        }
        self.area() * inside as f64 / (N * N) as f64
    }

    /// Shortest distance from the box to a point, zero if inside.
    pub fn distance_to_point(&self, px: f64, py: f64) -> f64 {
        let dx = (self.x - px).max(0.0).max(px - self.right());
        let dy = (self.y - py).max(0.0).max(py - self.bottom());
        (dx * dx + dy * dy).sqrt()
    }
}

/// A region labels must not cover: a disk around a source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeepOut {
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
}

/// One label to place.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelRequest {
    /// Point the label refers to.
    pub anchor: (f64, f64),
    /// Radius of the anchor's own keep-out; candidates start outside it.
    pub anchor_radius: f64,
    /// Label box size.
    pub w: f64,
    pub h: f64,
    /// Larger is placed first.
    pub priority: f64,
}

/// Where a label ended up.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedLabel {
    /// Index into the request list.
    pub request: usize,
    pub rect: Rect,
    /// True when the box touches no keep-out, no other label and no
    /// frame edge.
    pub clean: bool,
    /// Distance from the box to the anchor's keep-out edge; a leader
    /// line is worth drawing when this exceeds a few pixels.
    pub gap: f64,
}

/// Layout parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutConfig {
    /// Frame the labels must stay inside.
    pub frame: Rect,
    /// Extra clearance between a label and any keep-out, pixels.
    pub margin: f64,
    /// How far out from the anchor to search, in multiples of the label
    /// height.
    pub max_reach_label_heights: f64,
    /// Candidate directions per ring.
    pub directions: usize,
}

impl LayoutConfig {
    /// Defaults for a `width × height` frame.
    pub fn for_frame(width: f64, height: f64) -> Self {
        Self {
            frame: Rect {
                x: 0.0,
                y: 0.0,
                w: width,
                h: height,
            },
            margin: 3.0,
            max_reach_label_heights: 6.0,
            directions: 16,
        }
    }
}

/// Direction preference: 0 for due right, rising through below and
/// above to 1 for due left. Angles are image angles (y down), radians.
fn direction_penalty(angle: f64) -> f64 {
    // cos = +1 right, −1 left; below (sin > 0) slightly preferred to above.
    let right = angle.cos();
    let below = angle.sin();
    0.5 * (1.0 - right) + if below < 0.0 { 0.1 } else { 0.0 }
}

/// Place every label. Returns one entry per request, in placement order
/// (highest priority first).
pub fn place_labels(
    requests: &[LabelRequest],
    keepouts: &[KeepOut],
    config: &LayoutConfig,
) -> Vec<PlacedLabel> {
    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by(|&a, &b| {
        requests[b]
            .priority
            .total_cmp(&requests[a].priority)
            .then(a.cmp(&b))
    });

    let mut placed: Vec<PlacedLabel> = Vec::with_capacity(requests.len());
    for idx in order {
        let req = &requests[idx];
        let (ax, ay) = req.anchor;
        let label_area = (req.w * req.h).max(1.0);
        let start = req.anchor_radius + config.margin;
        let reach = req.h * config.max_reach_label_heights;
        let ring_step = (req.h * 0.75).max(2.0);

        // Overall best (used when no ring has a clean spot) and the best
        // clean candidate on the current ring (the nearest ring with one
        // wins, since every further ring only adds distance).
        let mut best: Option<(f64, Rect)> = None;
        let mut chosen: Option<Rect> = None;
        let mut radius = start;
        while radius <= start + reach && chosen.is_none() {
            let mut best_clean: Option<(f64, Rect)> = None;
            for k in 0..config.directions {
                let angle = TAU * k as f64 / config.directions as f64;
                // Put the box so that its nearest edge/corner sits on the
                // ring: offset the centre by half the box extent along the
                // direction.
                let (dx, dy) = (angle.cos(), angle.sin());
                let cx = ax + dx * (radius + req.w / 2.0 * dx.abs());
                let cy = ay + dy * (radius + req.h / 2.0 * dy.abs());
                let rect = Rect::centred(cx, cy, req.w, req.h);

                let outside = rect.area_outside(&config.frame) / label_area;
                let collide: f64 = placed
                    .iter()
                    .map(|p| rect.overlap_area(&p.rect))
                    .sum::<f64>()
                    / label_area;
                let keepout: f64 = keepouts
                    .iter()
                    .map(|k| rect.overlap_area_with_disk(k.cx, k.cy, k.radius + config.margin))
                    .sum::<f64>()
                    / label_area;
                let distance = rect.distance_to_point(ax, ay) / req.h.max(1.0);
                let score = 1000.0 * outside
                    + 100.0 * collide
                    + 10.0 * keepout
                    + distance
                    + 0.5 * direction_penalty(angle);
                if best.is_none_or(|(s, _)| score < s) {
                    best = Some((score, rect));
                }
                if outside <= AREA_EPS
                    && collide <= AREA_EPS
                    && keepout <= AREA_EPS
                    && best_clean.is_none_or(|(s, _)| score < s)
                {
                    best_clean = Some((score, rect));
                }
            }
            chosen = best_clean.map(|(_, r)| r);
            radius += ring_step;
        }

        let clean = chosen.is_some();
        let rect = chosen.unwrap_or_else(|| best.expect("at least one candidate direction").1);
        let gap = (rect.distance_to_point(ax, ay) - req.anchor_radius).max(0.0);
        placed.push(PlacedLabel {
            request: idx,
            rect,
            clean,
            gap,
        });
    }
    placed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> LayoutConfig {
        LayoutConfig::for_frame(512.0, 512.0)
    }

    #[test]
    fn rect_geometry() {
        let a = Rect::centred(10.0, 10.0, 4.0, 2.0);
        let b = Rect::centred(11.0, 10.0, 4.0, 2.0);
        assert_eq!(a.overlap_area(&b), 3.0 * 2.0);
        assert_eq!(a.distance_to_point(10.0, 10.0), 0.0);
        assert_eq!(a.distance_to_point(20.0, 10.0), 8.0);
        let disk_cover = a.overlap_area_with_disk(10.0, 10.0, 100.0);
        assert_eq!(disk_cover, a.area());
        assert_eq!(a.overlap_area_with_disk(100.0, 100.0, 1.0), 0.0);
    }

    #[test]
    fn single_label_sits_to_the_right_and_off_the_disk() {
        let req = LabelRequest {
            anchor: (256.0, 256.0),
            anchor_radius: 20.0,
            w: 60.0,
            h: 12.0,
            priority: 1.0,
        };
        let keep = [KeepOut {
            cx: 256.0,
            cy: 256.0,
            radius: 20.0,
        }];
        let placed = place_labels(&[req], &keep, &frame());
        assert_eq!(placed.len(), 1);
        let p = &placed[0];
        assert!(p.clean);
        assert!(
            p.rect.x > 256.0 + 20.0,
            "label {:?} should be right of the disk",
            p.rect
        );
        assert!((p.rect.centre().1 - 256.0).abs() < 1.0);
        assert!(p.rect.overlap_area_with_disk(256.0, 256.0, 20.0) == 0.0);
    }

    #[test]
    fn labels_do_not_overlap_each_other_or_sources() {
        // A tight cluster of six sources with long labels.
        let mut requests = Vec::new();
        let mut keepouts = Vec::new();
        for k in 0..6 {
            let angle = TAU * k as f64 / 6.0;
            let (x, y) = (256.0 + 18.0 * angle.cos(), 256.0 + 18.0 * angle.sin());
            requests.push(LabelRequest {
                anchor: (x, y),
                anchor_radius: 4.0,
                w: 90.0,
                h: 14.0,
                priority: 6.0 - k as f64,
            });
            keepouts.push(KeepOut {
                cx: x,
                cy: y,
                radius: 4.0,
            });
        }
        let cfg = frame();
        let placed = place_labels(&requests, &keepouts, &cfg);
        assert_eq!(placed.len(), 6);
        for (i, a) in placed.iter().enumerate() {
            assert!(a.clean, "label {i} not clean: {:?}", a.rect);
            assert_eq!(a.rect.area_outside(&cfg.frame), 0.0);
            for b in placed.iter().skip(i + 1) {
                assert_eq!(
                    a.rect.overlap_area(&b.rect),
                    0.0,
                    "{:?} vs {:?}",
                    a.rect,
                    b.rect
                );
            }
            for k in &keepouts {
                assert_eq!(
                    a.rect
                        .overlap_area_with_disk(k.cx, k.cy, k.radius + cfg.margin),
                    0.0
                );
            }
        }
        // Highest priority was placed first.
        assert_eq!(placed[0].request, 0);
    }

    #[test]
    fn label_near_the_frame_edge_stays_inside() {
        let req = LabelRequest {
            anchor: (505.0, 6.0),
            anchor_radius: 3.0,
            w: 80.0,
            h: 12.0,
            priority: 1.0,
        };
        let cfg = frame();
        let placed = place_labels(&[req], &[], &cfg);
        assert!(placed[0].clean);
        assert_eq!(placed[0].rect.area_outside(&cfg.frame), 0.0);
    }

    #[test]
    fn hopeless_label_is_placed_but_flagged() {
        // A disk that fills the frame: nowhere clean to go.
        let req = LabelRequest {
            anchor: (256.0, 256.0),
            anchor_radius: 400.0,
            w: 40.0,
            h: 12.0,
            priority: 1.0,
        };
        let keep = [KeepOut {
            cx: 256.0,
            cy: 256.0,
            radius: 400.0,
        }];
        let placed = place_labels(&[req], &keep, &frame());
        assert!(!placed[0].clean);
    }
}

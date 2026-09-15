# Informational overlay SVG (`focalplane-overlay/1`)

Every `planet_view` frame is written with a sibling `<out>.svg` produced by
`simulator::overlay`. The SVG has the same width and height as the 16-bit
PNG and the preview, so it can be laid directly over either with no scaling
or offset. This document is the contract for anything that consumes it.

## Coordinates

- `<svg width="W" height="H" viewBox="0 0 W H">`: one SVG user unit is one
  image pixel.
- Image pixel `(i, j)` occupies `[i, i+1) × [j, j+1)`. The renderer reports
  source centres in pixel-index coordinates where the index is the pixel
  centre, so a source at renderer `(x, y)` is drawn at `(x + 0.5, y + 0.5)`.
  The JSON carries the renderer coordinates (`anchor_px`); the drawn
  elements carry the shifted ones.
- Image parity is the frame's: north up, east left.

## Machine-readable content

The `<metadata>` element holds one namespaced child,

```xml
<metadata><fp:overlay type="application/json">{ …XML-escaped JSON… }</fp:overlay></metadata>
```

with `xmlns:fp="https://github.com/CosmicFrontierLabs/focalplane/overlay/v1"`.
Unescape `&lt; &gt; &quot; &apos; &amp;` and parse. The JSON is an
`OverlayDocument` (`simulator::overlay::OverlayDocument`, serde, so a Rust
consumer can deserialise it directly; `overlay::document_from_svg` does the
extraction):

| Key | Content |
|---|---|
| `format` | `"focalplane-overlay/1"` |
| `width_px`, `height_px` | frame size |
| `pixel_convention` | the sentence above, for humans |
| `frame` | frame-level provenance: for `planet_view` the whole `FrameMetadata` sidecar (generator, epoch, instrument, models, radiometry, bodies, sites, minor planets, outputs) |
| `style` | font size, character width used for layout, margins, colours per kind |
| `annotations[]` | one per annotated source, see below |
| `labels{}` | placed label boxes keyed by annotation `id`: `x, y, w, h` in SVG units, `clean` (true when the box touches no source, label or frame edge), `leader` (a leader line was drawn) |

Each annotation:

| Key | Content |
|---|---|
| `id` | unique in the overlay; the SVG element ids are `fp-<id>`, `label-<id>`, `leader-<id>` |
| `kind` | `sun`, `planet`, `moon`, `minor_planet`, `star`, `galaxy`, `site`, `marker` |
| `label` | text drawn, possibly empty (footprint only) |
| `anchor_px` | `[x, y]` renderer pixel-index coordinates |
| `footprint` | `{"shape":"point","marker_radius_px":r}`, `{"shape":"circle","radius_px":r}` or `{"shape":"ellipse","semi_major_px":a,"semi_minor_px":b,"position_angle_deg":pa}` (sky PA of the major axis, north through east) |
| `priority` | layout order override, or `null` for the kind's default |
| `payload` | free-form JSON describing the source; `planet_view` puts the matching sidecar record here (`BodyRecord`, `SiteRecord`, `MinorPlanetRecord`, or a Gaia star record `{gaia_source_id, ra_deg, dec_deg, magnitude_g, b_v}`) |

Every drawn element also carries `data-id`, and footprints carry
`data-kind` and `data-label`, so DOM-side code can join back to the JSON
without parsing it.

## Drawing

Three groups in draw order: `#footprints`, `#leaders`, `#labels`. Styling
is entirely in the `<style>` block by class (`.footprint`, `.label`,
`.leader`, `.kind-<kind>`, `.label.unclean`), so a consumer can restyle or
hide kinds with CSS and nothing else changes.

Labels are `<text>` with `textLength` and `lengthAdjust="spacingAndGlyphs"`,
so any viewer renders each label at exactly the width the layout assumed
(0.6 em per character) regardless of the font it substitutes. Text is
drawn with a dark stroke under the fill (`paint-order: stroke fill`) for
legibility on any background.

## Layout

`simulator::overlay::layout` places labels greedily in priority order
(Sun, planets, Moon, sites, markers, minor planets, galaxies, stars). Every
footprint plus a clearance is a keep-out disk; placed labels and the frame
edge are obstacles. Candidates are searched on rings of increasing radius
around the anchor in sixteen directions, preferring right, then below,
then above, then left; the nearest ring with a collision-free candidate
wins. A label with no clean spot within six label heights is placed at the
least-bad candidate and marked `unclean` (drawn faded). A leader line is
drawn when the label ended more than a few pixels from its footprint.

## Reproducing the composite

```bash
inkscape frame.svg --export-type=png --export-filename=overlay.png --export-background-opacity=0
# then alpha-composite overlay.png onto frame_preview.png at (0, 0), no scaling
```

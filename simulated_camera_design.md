# Simulated Camera — CameraInterface Trait Mapping

Mi got da picture. Existing simulator pieces line up well — let mi map dem to da trait.

## Trait → simulator mapping

| Trait method | Existing simulator piece | Effort |
|---|---|---|
| `geometry()` | `SensorConfig::dimensions: SensorGeometry` | trivial — direct return |
| `name()`, `get_serial()` | `SensorConfig::name` + synthetic serial | trivial |
| `saturation_value()` | `max_well_depth_e × dn_per_electron` (already computed at `sensor.rs:202`) | trivial |
| `get_bit_depth()` / `set_bit_depth()` | `SensorConfig::bit_depth: u8` | trivial; set just stores override |
| `get/set_exposure()` | passed straight into `Renderer::render(exposure, ...)` | trivial |
| `get/set_gain()` | scale `dn_per_electron`; new override field | easy |
| `set_roi()` / `get_roi()` / `clear_roi()` | new `Option<AABB>` field in impl | easy |
| `check_roi_size()` | compare against `dimensions.image_size()` | easy |
| `capture_frame()` | calls `Renderer::render()` + crops to ROI + returns `quantized_image` | see ROI question |
| `stream(callback)` | loop on `capture_frame()` with `tokio::time::interval` or `std::thread::sleep_until` based on exposure | easy, ~50 lines |
| `is_ready()` | `true` once `Renderer` constructed | trivial |

## Da big ROI question

Existing `Renderer` is full-frame only — `base_star_image: Array2<f64>` is the whole sensor (~488 MB for IMX455 in f64). Two paths:

### Path A — render full + crop (~1 day)

```rust
let result = self.renderer.render(&exposure, &zodi);
let img = match self.roi {
    Some(roi) => result.quantized_image.slice(s![roi.y.., roi.x..]).to_owned(),
    None => result.quantized_image,
};
```

- ✅ Drops in immediately, all sensors work
- ❌ Full IMX455 render is the cost regardless of ROI size. Look at `noise_throughput.rs` example to see current single-frame cost on da Orin — probably 100s of ms per IMX455 frame, misses 40 Hz target.
- Fine for GSENSE4040 / HWK4123. Probably not for IMX455.

### Path B — push ROI into render pipeline (~3–5 days)

- `Renderer::render_roi(roi, exposure, zodi)` — only stars whose splat overlaps ROI go through `add_stars_to_image` into a ROI-sized buffer
- Noise sampling on ROI-sized buffer (zodi/dark/read all are pointwise so they scale O(roi_pixels))
- `quantize_image` already O(pixels), parallelized per recent commit `5a94743`
- For 256² on IMX455: ROI is 0.001× full-frame area → render cost drops ~1000×
- ✅ Hits 40 Hz comfortable on IMX455
- ❌ Touches `add_stars_to_image`, `quantize_image`, all the noise samplers — careful work to keep tests passing

**Honest take**: start with Path A, ship the trait, measure on da Orin. If GSENSE4040 hits 40 Hz at full-render-then-crop, you might not need Path B at all for da sensors you actually use. Path B is the right answer for IMX455 specifically.

## What's missing from the trait for sim

The trait is shaped for real hardware where the scene is the sky. For sim, two things have nowhere to live:

### 1. Scene config

Sources, trajectory, zodi coords. Goes in constructor, not the trait:

```rust
impl SimulatedCamera {
    pub fn new(sensor: SensorConfig, scene: SceneConfig) -> Self { ... }
    pub fn update_scene(&mut self, scene: SceneConfig) { ... }
}
```

### 2. Time model

`stream()` cadence is wall clock for real cameras. For sim, you may want sim-time advance per frame (so trajectory at `t = frame_idx × exposure`). One option: take a `TimeSource` enum in constructor — `WallClock`, `SimTime { step: Duration }`, `Manual`.

Stays out of the trait — `SimulatedCamera` impls `CameraInterface` plus has these extras as inherent methods. Consumer code only sees the trait.

## Total estimate

| Phase | Effort | Outcome |
|---|---|---|
| Trait impl, Path A, wall-clock streaming, all sensors | 1–2 days | Drops into existing control-system test bench, GSENSE4040/HWK4123 likely hit 40 Hz |
| Benchmark on `orin-005`, decide if Path B is needed | 0.5 day | data |
| Path B (ROI-native render pipeline) if IMX455 target | 3–5 days | IMX455 hits 40 Hz |
| Tests + integration | 1 day | confidence |

**~1 week for the full thing, ~2 days for a usable v1.** The trait shape is genuinely good for this — separation is clean, real cameras and the sim look identical to da consumer. Sasa ke?

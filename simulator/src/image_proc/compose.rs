//! Second compositing pass for opaque extended bodies.
//!
//! Stars and galaxies are *additive*: each deposits mean electrons into
//! the sky image and the renderer relies on that linearity throughout.
//! Solar-system bodies are *opaque*: a planet hides the stars, galaxies
//! and zodiacal light behind it and replaces them with its own radiance.
//! That multiplicative step is kept out of the deposit machinery and
//! expressed once, here, as a second pass over the pass-1 mean image.
//!
//! # Contract
//!
//! A [`SecondPass`] implementation receives a [`PassContext`] describing
//! the sensor, the region of interest, the exposure and the spacecraft
//! orientation samples across the exposure, and returns a [`Composite`]:
//! an ordered list of far-to-near [`BodyComposite`] layers, each a
//! bounded box of per-pixel `coverage` in `[0, 1]` and per-pixel mean
//! `electrons`.
//!
//! The renderer applies each layer in order as
//!
//! ```text
//! sky = sky × (1 − coverage) + electrons
//! ```
//!
//! on the **mean** image, before dark current is added and before the
//! single Poisson draw. INVARIANTS §1 (one Poisson over the combined
//! mean) therefore holds unchanged: the pass only reshapes the mean.
//! Depth ordering falls out of applying layers sequentially, since a
//! nearer body's coverage also masks a farther body's electrons.
//!
//! A pass that returns an empty composite leaves the image bit-identical
//! to a render with no pass at all; this is locked by tests in the
//! renderers.
//!
//! Coverage is the PSF-convolved geometric disk, so occulted background
//! fades across the limb with the same blur the limb itself has. The
//! renderers apply one composite per exposure using the exposure's
//! orientation samples; a body and the background stars move together
//! under pointing jitter, so masking the stamp-integrated star image
//! with the stamp-integrated coverage is exact up to the body's own
//! motion relative to the stars within the exposure (sub-milliarcsecond
//! for any solar-system body over a sub-second exposure).

use std::fmt;
use std::time::Duration;

use nalgebra::UnitQuaternion;
use ndarray::Array2;
use thiserror::Error;

use crate::epoch::Epoch;
use crate::hardware::satellite::{FocalPlaneConfig, SatelliteConfig};

/// Spacecraft orientation at one instant within an exposure.
#[derive(Clone, Debug, PartialEq)]
pub struct OrientationSample {
    /// Offset from the exposure start.
    pub offset: Duration,
    /// Body-frame orientation (`+Z` boresight, `+X` east, `+Y` north).
    pub orientation: UnitQuaternion<f64>,
}

/// Everything a [`SecondPass`] needs to know about the tile it is
/// compositing onto.
#[derive(Clone, Debug)]
pub struct PassContext<'a> {
    /// Telescope, sensor array and temperature.
    pub focal_plane: &'a FocalPlaneConfig,
    /// The per-sensor view (QE, plate scale, aperture) for `sensor_idx`.
    pub satellite: &'a SatelliteConfig,
    /// Which sensor on the focal-plane array the buffer belongs to.
    pub sensor_idx: usize,
    /// Sensor pixel `(col, row)` at buffer index `[0, 0]`.
    pub roi_origin: (usize, usize),
    /// Buffer size `(width, height)` in pixels.
    pub roi_size: (usize, usize),
    /// Integration time of the exposure being composited.
    pub exposure: Duration,
    /// Absolute time at exposure start. `None` when the scene has no
    /// epoch; passes that need one should return
    /// [`ComposeError::MissingEpoch`].
    pub epoch: Option<Epoch>,
    /// Orientation samples across the exposure, in time order. A static
    /// render supplies exactly one sample at offset zero.
    pub samples: &'a [OrientationSample],
}

impl PassContext<'_> {
    /// Absolute time of one orientation sample, if the context has an
    /// epoch.
    pub fn epoch_at(&self, sample: &OrientationSample) -> Option<Epoch> {
        self.epoch.as_ref().map(|e| e.offset(sample.offset))
    }
}

/// Errors a [`SecondPass`] can raise.
#[derive(Debug, Error)]
pub enum ComposeError {
    /// The pass needs an absolute epoch and the scene did not supply one.
    #[error("second pass requires an epoch but the scene has none")]
    MissingEpoch,
    /// Any other failure, with a human-readable reason.
    #[error("second pass failed: {0}")]
    Failed(String),
}

/// One opaque body's contribution to a buffer: a bounded box of
/// coverage and mean electrons.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyComposite {
    /// Buffer coordinates `(col, row)` of the box's `[0, 0]` element.
    /// May be negative or exceed the buffer; out-of-range elements are
    /// clipped when applied.
    pub origin: (i64, i64),
    /// Fraction of each pixel hidden by the body, in `[0, 1]`, shaped
    /// `(rows, cols)`.
    pub coverage: Array2<f64>,
    /// Mean electrons the body contributes to each pixel over the
    /// exposure, same shape as `coverage`.
    pub electrons: Array2<f64>,
}

impl BodyComposite {
    /// Build a layer, validating that the two arrays agree in shape and
    /// that coverage lies in `[0, 1]`.
    pub fn new(origin: (i64, i64), coverage: Array2<f64>, electrons: Array2<f64>) -> Self {
        assert_eq!(
            coverage.dim(),
            electrons.dim(),
            "coverage and electrons must share a shape"
        );
        debug_assert!(
            coverage.iter().all(|&c| (0.0..=1.0).contains(&c)),
            "coverage must lie in [0, 1]"
        );
        Self {
            origin,
            coverage,
            electrons,
        }
    }

    /// Visit every element of this layer that lands inside `buf`,
    /// yielding the buffer element together with the layer's coverage
    /// and electrons at that pixel.
    fn for_each_overlap(&self, buf: &mut Array2<f64>, mut f: impl FnMut(&mut f64, f64, f64)) {
        let (buf_rows, buf_cols) = buf.dim();
        let (rows, cols) = self.coverage.dim();
        for r in 0..rows {
            let br = self.origin.1 + r as i64;
            if br < 0 || br >= buf_rows as i64 {
                continue;
            }
            for c in 0..cols {
                let bc = self.origin.0 + c as i64;
                if bc < 0 || bc >= buf_cols as i64 {
                    continue;
                }
                f(
                    &mut buf[[br as usize, bc as usize]],
                    self.coverage[[r, c]],
                    self.electrons[[r, c]],
                );
            }
        }
    }
}

/// Ordered (far-to-near) layers produced by one [`SecondPass`] call.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Composite {
    /// Layers in application order: the last layer is nearest the
    /// observer and masks everything before it.
    pub layers: Vec<BodyComposite>,
}

impl Composite {
    /// A composite with no layers; applying it is a no-op.
    pub fn empty() -> Self {
        Self::default()
    }

    /// True when applying this composite would not touch any pixel.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Mask and add every layer in order: `buf = buf × (1 − c) + e`.
    pub fn apply(&self, buf: &mut Array2<f64>) {
        for layer in &self.layers {
            layer.for_each_overlap(buf, |px, c, e| *px = *px * (1.0 - c) + e);
        }
    }

    /// Mask every layer in order without adding electrons:
    /// `buf = buf × (1 − c)`. Used for buffers that hold only background
    /// (zodiacal light) which the body hides but does not replace.
    pub fn apply_mask(&self, buf: &mut Array2<f64>) {
        for layer in &self.layers {
            layer.for_each_overlap(buf, |px, c, _| *px *= 1.0 - c);
        }
    }
}

/// A producer of opaque-body composites for one tile.
///
/// Implementations must be deterministic and free of RNG: the composite
/// is a mean, and the renderer's single Poisson stage supplies all shot
/// noise.
pub trait SecondPass: Send + Sync + fmt::Debug {
    /// Build the far-to-near composite for the tile described by `ctx`.
    /// Return [`Composite::empty`] when no body touches the tile.
    fn composite(&self, ctx: &PassContext<'_>) -> Result<Composite, ComposeError>;
}

/// A pass that never composites anything. Exists so renderers can lock
/// the "empty composite is bit-identical to no pass" contract in tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopPass;

impl SecondPass for NoopPass {
    fn composite(&self, _ctx: &PassContext<'_>) -> Result<Composite, ComposeError> {
        Ok(Composite::empty())
    }
}

/// A pass that returns the same composite for every tile, ignoring the
/// context. Useful for synthetic scenes and renderer tests.
#[derive(Clone, Debug)]
pub struct ConstantPass {
    composite: Composite,
}

impl ConstantPass {
    /// Wrap a fixed composite.
    pub fn new(composite: Composite) -> Self {
        Self { composite }
    }
}

impl SecondPass for ConstantPass {
    fn composite(&self, _ctx: &PassContext<'_>) -> Result<Composite, ComposeError> {
        Ok(self.composite.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn layer(origin: (i64, i64), coverage: f64, electrons: f64) -> BodyComposite {
        BodyComposite::new(
            origin,
            Array2::from_elem((2, 2), coverage),
            Array2::from_elem((2, 2), electrons),
        )
    }

    #[test]
    fn empty_composite_is_a_no_op() {
        let mut buf = Array2::from_shape_fn((3, 3), |(r, c)| (r * 3 + c) as f64);
        let before = buf.clone();
        Composite::empty().apply(&mut buf);
        Composite::empty().apply_mask(&mut buf);
        assert_eq!(buf, before);
    }

    #[test]
    fn full_coverage_replaces_background_with_body_electrons() {
        let mut buf = Array2::from_elem((3, 3), 10.0);
        let composite = Composite {
            layers: vec![layer((1, 1), 1.0, 5.0)],
        };
        composite.apply(&mut buf);
        let expected = array![[10.0, 10.0, 10.0], [10.0, 5.0, 5.0], [10.0, 5.0, 5.0]];
        assert_eq!(buf, expected);
    }

    #[test]
    fn partial_coverage_blends_linearly() {
        let mut buf = Array2::from_elem((2, 2), 8.0);
        let composite = Composite {
            layers: vec![layer((0, 0), 0.25, 2.0)],
        };
        composite.apply(&mut buf);
        for &px in buf.iter() {
            assert_abs_diff_eq!(px, 8.0 * 0.75 + 2.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn mask_only_hides_without_adding() {
        let mut buf = Array2::from_elem((2, 2), 8.0);
        let composite = Composite {
            layers: vec![layer((0, 0), 0.5, 999.0)],
        };
        composite.apply_mask(&mut buf);
        for &px in buf.iter() {
            assert_abs_diff_eq!(px, 4.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn nearer_layer_masks_farther_layer() {
        let mut buf = Array2::zeros((2, 2));
        let composite = Composite {
            layers: vec![layer((0, 0), 1.0, 100.0), layer((0, 0), 1.0, 7.0)],
        };
        composite.apply(&mut buf);
        for &px in buf.iter() {
            assert_abs_diff_eq!(px, 7.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn layers_clip_at_buffer_edges() {
        let mut buf = Array2::from_elem((3, 3), 1.0);
        let composite = Composite {
            layers: vec![layer((-1, -1), 1.0, 3.0), layer((2, 2), 1.0, 5.0)],
        };
        composite.apply(&mut buf);
        let expected = array![[3.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 5.0]];
        assert_eq!(buf, expected);
    }

    #[test]
    fn noop_and_constant_passes_return_what_they_promise() {
        use shared::units::TemperatureExt;
        let sat = crate::hardware::satellite::SatelliteConfig::new(
            crate::hardware::telescope::models::SMALL_50MM.clone(),
            crate::hardware::sensor::models::IMX455.clone(),
            shared::units::Temperature::from_celsius(-10.0),
        );
        let fp = crate::hardware::satellite::FocalPlaneConfig::from_satellite(&sat);
        let samples = [OrientationSample {
            offset: Duration::ZERO,
            orientation: UnitQuaternion::identity(),
        }];
        let ctx = PassContext {
            focal_plane: &fp,
            satellite: &sat,
            sensor_idx: 0,
            roi_origin: (0, 0),
            roi_size: (4, 4),
            exposure: Duration::from_millis(10),
            epoch: None,
            samples: &samples,
        };
        assert!(NoopPass.composite(&ctx).unwrap().is_empty());
        let fixed = Composite {
            layers: vec![layer((0, 0), 1.0, 1.0)],
        };
        assert_eq!(
            ConstantPass::new(fixed.clone()).composite(&ctx).unwrap(),
            fixed
        );
        assert!(ctx.epoch_at(&samples[0]).is_none());
    }
}

//! Parallel motion-blur renderer for focal-plane trajectories.
//!
//! Each frame's exposure window is integrated as a single flat
//! stratified-Monte-Carlo sequence of PSF stamps. The total stamp
//! count is chosen adaptively from a per-stamp drift budget so the
//! trajectory's angular path between consecutive stamps stays below
//! a fraction of a pixel; for a static trajectory the schedule
//! collapses to a single stamp.
//!
//! # Noise model
//!
//! A single unified Poisson draw is taken per `(frame, sensor)` over the
//! mean-electron image comprising:
//!
//! - Star contributions accumulated across all stamps (each stamp
//!   contributes `flux × dt` electrons where `dt = exposure / stamps_per_exposure`).
//! - A uniform zodiacal mean evaluated at the frame's central boresight.
//! - A uniform dark-current mean from the per-sensor dark current rate
//!   times the full exposure duration.
//!
//! Gaussian read noise is added separately after the Poisson draw because
//! it is an electronic readout effect, not a photon shot-noise contribution.
//!
//! # Parallelism
//!
//! The top-level entry point parallelizes over `(frame_idx, sensor_idx)`
//! pairs via rayon. Each tile independently consumes its pre-filtered
//! star slice, materializes a per-sensor mean-electron accumulator,
//! draws noise, quantizes, and writes its PNG. Nothing is reduced
//! across tiles.
//!
//! # Layout
//!
//! - [`schedule`] — [`SubsampleSchedule`] and the per-stamp time grid.
//! - [`accumulator`] — [`SensorAccumulator`] electron accumulation.
//! - [`render`] — the render pipeline ([`render_motion_trajectory`],
//!   [`render_one_frame`], [`render_one_frame_roi`]) and its tile/frame
//!   orchestration.
//! - [`metadata`] — assembly of the per-render `metadata.json`.

mod accumulator;
mod metadata;
mod render;
mod schedule;

pub use accumulator::SensorAccumulator;
pub use render::{
    render_motion_trajectory, render_one_frame, render_one_frame_roi, FluxCache, LightSources,
    MotionBlurConfig,
};
pub use schedule::{SubsampleSchedule, DEFAULT_MAX_DRIFT_PER_STAMP_PX};

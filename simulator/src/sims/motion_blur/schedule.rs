//! Time-domain stamp scheduling for the motion-blur renderer.
//!
//! A [`SubsampleSchedule`] divides an exposure window into a flat
//! stratified-Monte-Carlo sequence of PSF-stamp times; the stamp count is
//! chosen adaptively from a per-stamp drift budget.

use std::time::Duration;

/// Default per-stamp drift budget (in pixels). The total stamp count is
/// derived as `ceil(total_drift_rad / (DEFAULT_MAX_DRIFT_PER_STAMP_PX * pixel_scale_rad))`.
pub const DEFAULT_MAX_DRIFT_PER_STAMP_PX: f64 = 0.1;

/// Time-domain stamp schedule inside an exposure window.
///
/// A `SubsampleSchedule` describes a single flat sequence of
/// PSF-stamp times across the exposure: the window is divided into
/// `stamps_per_exposure` equal-width sub-bins and each stamp lands at
/// a uniform-random offset within its sub-bin (stratified Monte
/// Carlo). Stratification preserves smooth-integrand convergence
/// while removing the systematic bias a regular grid produces against
/// trajectory tones near `k / Δt_stamp`.
///
/// Each stamp queries the trajectory at its own time, projects every
/// in-field star, and deposits `flux / stamps_per_exposure` electrons
/// at the per-time projected pixel position.
///
/// `envelope_padding_rad` is the safety padding the *envelope
/// prefilter* uses to inflate the focal-plane AABB so that any star
/// whose excursion brings it onto the sensor at any point during the
/// exposure is retained. It is the only thing the renderer needs from
/// "scene-state cadence" — there is no separate subsample-level loop
/// for scene state.
#[derive(Debug, Clone, Copy)]
pub struct SubsampleSchedule {
    /// Absolute trajectory time at which the frame's exposure starts.
    pub frame_start: Duration,
    /// Total exposure duration for this frame.
    pub exposure: Duration,
    /// Total number of stratified-MC PSF stamps across the exposure (>= 1).
    pub stamps_per_exposure: usize,
    /// Peak excursion of the trajectory from the frame's mid-time
    /// orientation, in radians. The envelope prefilter inflates the
    /// focal-plane AABB by this amount (converted to mm) on top of
    /// its own PSF-extent padding.
    pub envelope_padding_rad: f64,
}

impl SubsampleSchedule {
    /// Per-stamp duration: `exposure / stamps_per_exposure`. Each
    /// stamp's electron contribution is integrated over this `dt`.
    pub fn dt(&self) -> Duration {
        Duration::from_secs_f64(
            self.exposure.as_secs_f64() / self.stamps_per_exposure.max(1) as f64,
        )
    }

    /// Quasi-random stamp times across the entire exposure window,
    /// derived from the 1D golden-ratio low-discrepancy sequence (see
    /// [`crate::sims::quasi_random`]). Returns a vector of length
    /// `stamps_per_exposure.max(1)`. Stamp `j` lands at
    /// `frame_start + offset_j · exposure`, where `offset_j` is the
    /// `j`-th element of the golden-ratio sequence with the given
    /// `phase` shift.
    ///
    /// The golden-ratio sequence is **deterministic** — no RNG is
    /// consumed. `phase` is the only source of per-tile variation;
    /// the renderer derives it from the tile seed via
    /// [`crate::sims::quasi_random::phase_from_seed`].
    pub fn stamp_times(&self, phase: f64) -> Vec<Duration> {
        let total = self.stamps_per_exposure.max(1);
        let exposure_s = self.exposure.as_secs_f64();
        let t0 = self.frame_start.as_secs_f64();
        crate::sims::quasi_random::golden_offsets(total, phase)
            .into_iter()
            .map(|u| Duration::from_secs_f64(t0 + u * exposure_s))
            .collect()
    }

    /// Frame mid-time, used as the reference orientation for envelope
    /// prefiltering and zodiacal evaluation.
    pub fn mid_time(&self) -> Duration {
        self.frame_start + self.exposure / 2
    }

    /// Construct a schedule from a per-stamp drift budget plus a peak
    /// excursion (both in radians). Stamp count is
    /// `ceil(total_drift_rad / (max_drift_per_stamp_px * pixel_scale_rad))`,
    /// minimum 1. Envelope padding stores `peak_excursion_rad` directly
    /// for later mm conversion in `envelope_prefilter`.
    pub fn from_drift_budget(
        frame_start: Duration,
        exposure: Duration,
        total_drift_rad_over_exposure: f64,
        peak_excursion_rad: f64,
        pixel_scale_rad: f64,
        max_drift_per_stamp_px: f64,
    ) -> Self {
        let stamps_per_exposure = if total_drift_rad_over_exposure <= 0.0
            || pixel_scale_rad <= 0.0
            || max_drift_per_stamp_px <= 0.0
        {
            1
        } else {
            let budget_rad = max_drift_per_stamp_px * pixel_scale_rad;
            ((total_drift_rad_over_exposure / budget_rad).ceil() as usize).max(1)
        };
        Self {
            frame_start,
            exposure,
            stamps_per_exposure,
            envelope_padding_rad: peak_excursion_rad.max(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_from_drift_budget_static_gives_one_stamp() {
        let sched = SubsampleSchedule::from_drift_budget(
            Duration::ZERO,
            Duration::from_secs(1),
            0.0, // zero drift
            0.0, // zero excursion
            1e-6,
            0.1,
        );
        assert_eq!(sched.stamps_per_exposure, 1);
        assert_eq!(sched.envelope_padding_rad, 0.0);
    }

    #[test]
    fn test_from_drift_budget_picks_total_stamps_from_path_length() {
        // 10 px of total path length at pixel_scale=1e-5 rad/px with a
        // 0.1 px per-stamp budget should yield 100 total stamps.
        let pixel_scale = 1e-5_f64;
        let drift_px = 10.0;
        let budget_px = 0.1;
        let excursion_rad = 0.5_f64 * pixel_scale; // arbitrary positive
        let sched = SubsampleSchedule::from_drift_budget(
            Duration::ZERO,
            Duration::from_secs(1),
            drift_px * pixel_scale,
            excursion_rad,
            pixel_scale,
            budget_px,
        );
        let expected = (drift_px / budget_px).ceil() as usize;
        assert_eq!(sched.stamps_per_exposure, expected);
        assert_abs_diff_eq!(sched.envelope_padding_rad, excursion_rad, epsilon = 1e-15);
    }

    #[test]
    fn test_stamp_times_span_the_exposure_window() {
        // Stamps must all fall inside [frame_start, frame_start + exposure).
        let sched = SubsampleSchedule {
            frame_start: Duration::from_secs(10),
            exposure: Duration::from_secs(4),
            stamps_per_exposure: 16,
            envelope_padding_rad: 0.0,
        };
        let stamps = sched.stamp_times(0.0);
        assert_eq!(stamps.len(), 16);
        let lo = 10.0;
        let hi = 14.0;
        for t in &stamps {
            let v = t.as_secs_f64();
            assert!(v >= lo && v < hi, "stamp {v} outside [{lo}, {hi})");
        }
    }

    #[test]
    fn test_stamp_times_are_reproducible_for_same_phase() {
        // The R2 stamp times are a pure function of (schedule, phase).
        let sched = SubsampleSchedule {
            frame_start: Duration::ZERO,
            exposure: Duration::from_secs(1),
            stamps_per_exposure: 128,
            envelope_padding_rad: 0.0,
        };
        let a = sched.stamp_times(0.123);
        let b = sched.stamp_times(0.123);
        assert_eq!(a, b);
        let c = sched.stamp_times(0.124);
        assert_ne!(a, c);
    }

    #[test]
    fn test_mid_time_is_exposure_midpoint() {
        let sched = SubsampleSchedule {
            frame_start: Duration::from_secs(10),
            exposure: Duration::from_secs(4),
            stamps_per_exposure: 4,
            envelope_padding_rad: 0.0,
        };
        assert_eq!(sched.mid_time(), Duration::from_secs(12));
    }
}

//! Pink-noise pointing-jitter synthesis.
//!
//! Generates pink (1/f) noise residuals suitable for shaping a quasi-static
//! pointing into a jittered trajectory. The core sample generator is Paul
//! Kellett's 7-pole IIR approximation, which yields a roughly `-10 dB/decade`
//! slope across about three decades of frequency at no FFT cost.
//!
//! The high-level entry point is [`build_pink_trajectory`], which composes a
//! 2-D pink-noise residual ([`pink_noise_2d_arcsec`]) onto a fixed sky
//! pointing and returns a [`Trajectory`] ready for the renderer.

use std::time::Duration;

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use starfield::Equatorial;

use crate::sims::trajectory::{Trajectory, TrajectoryError, Waypoint};

/// Generate `n` samples of pink (1/f) noise via Paul Kellett's 7-pole
/// filter. Good approximation across about three decades of frequency.
pub fn pink_noise_series(n: usize, rng: &mut impl Rng) -> Vec<f64> {
    let mut b = [0.0_f64; 7];
    let mut out = Vec::with_capacity(n);
    // Warm the filter so the first samples aren't transient-heavy.
    for _ in 0..256 {
        let white: f64 = rng.sample(StandardNormal);
        b[0] = 0.99886 * b[0] + white * 0.0555179;
        b[1] = 0.99332 * b[1] + white * 0.0750759;
        b[2] = 0.96900 * b[2] + white * 0.1538520;
        b[3] = 0.86650 * b[3] + white * 0.3104856;
        b[4] = 0.55000 * b[4] + white * 0.5329522;
        b[5] = -0.7616 * b[5] - white * 0.0168980;
        let _ = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362;
        b[6] = white * 0.115926;
    }
    for _ in 0..n {
        let white: f64 = rng.sample(StandardNormal);
        b[0] = 0.99886 * b[0] + white * 0.0555179;
        b[1] = 0.99332 * b[1] + white * 0.0750759;
        b[2] = 0.96900 * b[2] + white * 0.1538520;
        b[3] = 0.86650 * b[3] + white * 0.3104856;
        b[4] = 0.55000 * b[4] + white * 0.5329522;
        b[5] = -0.7616 * b[5] - white * 0.0168980;
        let pink = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362;
        b[6] = white * 0.115926;
        out.push(pink);
    }
    out
}

/// Generate a 2-D pink-noise offset series (RA, Dec) in arcseconds,
/// scaled so the mean magnitude `mean(|offset|)` equals
/// `target_mean_arcsec`. Centered at zero after generation.
pub fn pink_noise_2d_arcsec(n: usize, target_mean_arcsec: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ra = pink_noise_series(n, &mut rng);
    let dec = pink_noise_series(n, &mut rng);
    let ra_mean = ra.iter().sum::<f64>() / n as f64;
    let dec_mean = dec.iter().sum::<f64>() / n as f64;
    let ra: Vec<f64> = ra.iter().map(|v| v - ra_mean).collect();
    let dec: Vec<f64> = dec.iter().map(|v| v - dec_mean).collect();
    let raw_mean_mag: f64 = ra
        .iter()
        .zip(&dec)
        .map(|(a, b)| (a * a + b * b).sqrt())
        .sum::<f64>()
        / n as f64;
    let scale = if raw_mean_mag > 1e-12 {
        target_mean_arcsec / raw_mean_mag
    } else {
        0.0
    };
    (
        ra.into_iter().map(|v| v * scale).collect(),
        dec.into_iter().map(|v| v * scale).collect(),
    )
}

/// Build a pink-spectrum residual trajectory centered on `pointing`,
/// scaled so the mean pointing offset magnitude equals
/// `mean_arcsec`. The residual is spread over `segments` waypoints
/// across `duration`, with roll interpolated linearly from
/// `start_roll_deg` to `start_roll_deg + total_roll_deg`.
pub fn build_pink_trajectory(
    pointing: Equatorial,
    mean_arcsec: f64,
    segments: usize,
    seed: u64,
    duration: Duration,
    start_roll_deg: f64,
    total_roll_deg: f64,
) -> Result<Trajectory, TrajectoryError> {
    let segments = segments.max(8);
    let (ra_offset_arcsec, dec_offset_arcsec) =
        pink_noise_2d_arcsec(segments + 1, mean_arcsec, seed);
    let ra0 = pointing.ra_degrees();
    let dec0 = pointing.dec_degrees();
    let cos_dec0 = dec0.to_radians().cos().max(1e-6);
    let duration_s = duration.as_secs_f64();
    let waypoints: Vec<Waypoint> = (0..=segments)
        .map(|i| {
            let frac = i as f64 / segments as f64;
            let t = Duration::from_secs_f64(duration_s * frac);
            let ra = ra0 + (ra_offset_arcsec[i] / 3600.0) / cos_dec0;
            let dec = dec0 + dec_offset_arcsec[i] / 3600.0;
            let eq = Equatorial::from_degrees(ra, dec);
            let roll_deg = start_roll_deg + frac * total_roll_deg;
            Waypoint::from_pointing_and_roll(t, eq, roll_deg.to_radians())
        })
        .collect();
    Trajectory::new(waypoints)
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rustfft::{num_complex::Complex, FftPlanner};

    /// `pink_noise_series` must produce a power spectrum that falls at
    /// roughly `-10 dB/decade` (i.e. PSD ∝ 1/f). We fit a linear slope to
    /// `log10(PSD)` vs `log10(f)` over a band that excludes the lowest few
    /// FFT bins (dominated by finite-sample DC drift) and the very top of
    /// the band (where Kellett's filter rolls off and aliases bite).
    #[test]
    fn pink_noise_series_psd_slope_is_about_minus_one_per_decade() {
        let n: usize = 1 << 16; // 65536 samples → plenty of bins to fit
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let samples = pink_noise_series(n, &mut rng);
        assert_eq!(samples.len(), n);

        // FFT the time series.
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n);
        let mut buf: Vec<Complex<f64>> = samples
            .iter()
            .map(|&v| Complex { re: v, im: 0.0 })
            .collect();
        fft.process(&mut buf);

        // PSD per bin (one-sided magnitude squared / N). Normalisation
        // cancels out in the slope, so any consistent scaling is fine.
        let psd: Vec<f64> = buf[..n / 2]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) / n as f64)
            .collect();

        // Fit slope over k = [n/256 .. n/16) i.e. 1.5 decades well inside
        // the validated band — skips the DC-dominated low end and the
        // filter rolloff above ~F_s/16.
        let k_lo = (n / 256).max(2);
        let k_hi = n / 16;

        // Frequency in cycles/sample is k/n; the slope vs log10(f) is the
        // same as the slope vs log10(k) (offset only), so we can fit in k.
        let mut sx = 0.0_f64;
        let mut sy = 0.0_f64;
        let mut sxx = 0.0_f64;
        let mut sxy = 0.0_f64;
        let mut count = 0.0_f64;
        for (k, &p) in psd.iter().enumerate().take(k_hi).skip(k_lo) {
            let x = (k as f64).log10();
            let y = p.log10();
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
            count += 1.0;
        }
        let slope = (count * sxy - sx * sy) / (count * sxx - sx * sx);

        // Expect roughly -1.0 (i.e. -10 dB/decade). Kellett's filter has
        // some ripple, so allow ±0.2 around the ideal.
        assert!(
            (slope - (-1.0)).abs() < 0.2,
            "PSD slope {slope:.3} not within 0.2 of -1.0 (target -10 dB/decade)"
        );
    }

    #[test]
    fn pink_noise_2d_hits_requested_mean_magnitude() {
        let (ra, dec) = pink_noise_2d_arcsec(4096, 6.0, 7);
        let mean_mag: f64 = ra
            .iter()
            .zip(&dec)
            .map(|(a, b)| (a * a + b * b).sqrt())
            .sum::<f64>()
            / ra.len() as f64;
        assert_relative_eq!(mean_mag, 6.0, epsilon = 1e-6);
    }

    #[test]
    fn build_pink_trajectory_produces_requested_waypoint_count() {
        let pointing = Equatorial::from_degrees(213.39, -55.86);
        let traj = build_pink_trajectory(pointing, 5.0, 128, 42, Duration::from_secs(10), 0.0, 0.0)
            .expect("pink trajectory builds");
        // segments + 1 waypoints
        assert_eq!(traj.waypoints().len(), 129);
        assert_relative_eq!(traj.end_time().as_secs_f64(), 10.0, epsilon = 1e-9);
    }
}

//! Pointing-jitter trajectory synthesis.
//!
//! Two synthesis routes:
//!
//! 1. **Pink-noise IIR** — Paul Kellett's 7-pole 1/f filter (see
//!    [`pink_noise_series`]). Cheap, no FFT, yields a roughly
//!    `-10 dB/decade` slope across about three decades of frequency.
//!    Composed onto a fixed pointing by [`build_pink_trajectory`].
//!
//! 2. **PSD-driven FFT synthesis** — fixed-magnitude/random-phase synthesis
//!    of a real time series whose one-sided PSD matches a tabulated
//!    rad²/Hz spectrum ([`synthesize_from_psd`]). The high-level
//!    [`build_trajectory_from_los_psd`] applies independent X- and Y-axis
//!    body-frame rotation perturbations from a two-column LOS PSD onto a
//!    nominal RA/Dec/roll pointing and returns a per-sample [`Trajectory`].
//!    The CSV loader [`load_los_psd_csv`] parses the same file format
//!    used by `scripts/los_psd_to_trajectory_csv.py`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use serde::{Deserialize, Serialize};
use starfield::Equatorial;
use thiserror::Error;

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

/// Generate a 2-D pink-noise offset series (east, north) in arcseconds,
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
///
/// The east/north offsets are applied as small rotations about the body
/// X/Y axes composed onto the zero-roll boresight quaternion (see
/// [`nominal_body_to_world`]), so the synthesis has no RA `1/cos(dec)`
/// pole singularity and is valid at any declination.
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
    let (east_offset_arcsec, north_offset_arcsec) =
        pink_noise_2d_arcsec(segments + 1, mean_arcsec, seed);
    let base = nominal_body_to_world(pointing.ra_degrees(), pointing.dec_degrees(), 0.0);
    let duration_s = duration.as_secs_f64();
    let waypoints: Vec<Waypoint> = (0..=segments)
        .map(|i| {
            let frac = i as f64 / segments as f64;
            let t = Duration::from_secs_f64(duration_s * frac);
            let east = (east_offset_arcsec[i] / 3600.0).to_radians();
            let north = (north_offset_arcsec[i] / 3600.0).to_radians();
            // A rotation about body +Y tips the boresight (+Z) toward
            // body +X (sky east); a rotation about body -X tips it
            // toward body +Y (sky north).
            let perturb = UnitQuaternion::from_scaled_axis(Vector3::new(-north, east, 0.0));
            let roll_rad = (start_roll_deg + frac * total_roll_deg).to_radians();
            let twist = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, roll_rad));
            Waypoint::new(t, base * perturb * twist)
        })
        .collect();
    Trajectory::new(waypoints)
}

/// Two-axis line-of-sight pointing-jitter PSD loaded from a CSV file.
///
/// The two PSD columns describe the recovered rotation degrees-of-freedom
/// about the FEM/optical local X and Y axes, expressed in rad²/Hz on a
/// strictly-ascending frequency grid. They are *not* displacement PSDs
/// along sky east / north — see [`load_los_psd_csv`] for the full
/// convention.
#[derive(Debug, Clone)]
pub struct LosPsd {
    /// Frequency grid in Hz, strictly ascending.
    pub freq_hz: Vec<f64>,
    /// PSD of the rotation about the FEM/optical local X axis, in rad²/Hz.
    pub psd_x_rad2_per_hz: Vec<f64>,
    /// PSD of the rotation about the FEM/optical local Y axis, in rad²/Hz.
    pub psd_y_rad2_per_hz: Vec<f64>,
}

impl LosPsd {
    /// Highest frequency in the PSD's frequency grid. Useful for picking a
    /// synthesis sample rate ≥ 2× this value.
    pub fn max_freq_hz(&self) -> f64 {
        *self
            .freq_hz
            .last()
            .expect("freq_hz is non-empty by construction")
    }
}

/// Error type for [`load_los_psd_csv`].
#[derive(Debug, Error)]
pub enum LosPsdLoadError {
    #[error("opening {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path} is empty (no header row)")]
    EmptyFile { path: PathBuf },
    #[error("{path}: missing header column `{column}`")]
    MissingColumn { path: PathBuf, column: String },
    #[error("{path}: line {line}: missing `{column}` cell")]
    MissingCell {
        path: PathBuf,
        line: usize,
        column: String,
    },
    #[error("{path}: line {line}: bad `{column}` value: {source}")]
    BadCell {
        path: PathBuf,
        line: usize,
        column: String,
        #[source]
        source: std::num::ParseFloatError,
    },
    #[error("{path}: CSV record at line {line}: {source}")]
    Csv {
        path: PathBuf,
        line: u64,
        #[source]
        source: csv::Error,
    },
    #[error("{path}: needs at least 2 frequency rows, found {found}")]
    TooFewRows { path: PathBuf, found: usize },
    #[error("{path}: line {line}: `freq_hz` is not strictly ascending")]
    FrequencyNotAscending { path: PathBuf, line: usize },
}

/// Load a two-axis LOS PSD from a CSV file.
///
/// # File format
///
/// The first non-empty, non-`#`-comment line is treated as a header. The
/// required columns are:
///
/// - `freq_hz` — frequency in Hz, strictly ascending
/// - `psd_los_x_rad2_per_hz` — PSD of rotation about the FEM optical local
///   X axis, in rad²/Hz
/// - `psd_los_y_rad2_per_hz` — PSD of rotation about the FEM optical local
///   Y axis, in rad²/Hz
///
/// Column order doesn't matter — the parser dispatches by header name.
/// Lines whose first non-whitespace character is `#` are skipped (both
/// before and after the header). Extra columns are ignored.
///
/// The format matches the input expected by
/// `scripts/los_psd_to_trajectory_csv.py`.
pub fn load_los_psd_csv(path: impl AsRef<Path>) -> Result<LosPsd, LosPsdLoadError> {
    let path_ref = path.as_ref();
    let file = std::fs::File::open(path_ref).map_err(|source| LosPsdLoadError::Io {
        path: path_ref.to_path_buf(),
        source,
    })?;
    let mut reader = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .trim(csv::Trim::All)
        .from_reader(file);
    let headers = reader
        .headers()
        .map_err(|source| LosPsdLoadError::Csv {
            path: path_ref.to_path_buf(),
            line: source.position().map_or(1, |p| p.line()),
            source,
        })?
        .clone();
    if headers.is_empty() {
        return Err(LosPsdLoadError::EmptyFile {
            path: path_ref.to_path_buf(),
        });
    }
    for column in ["freq_hz", "psd_los_x_rad2_per_hz", "psd_los_y_rad2_per_hz"] {
        if !headers.iter().any(|header| header == column) {
            return Err(LosPsdLoadError::MissingColumn {
                path: path_ref.to_path_buf(),
                column: column.to_owned(),
            });
        }
    }
    let mut freq_hz: Vec<f64> = Vec::new();
    let mut psd_x: Vec<f64> = Vec::new();
    let mut psd_y: Vec<f64> = Vec::new();
    while let Some(row) = reader.deserialize::<LosPsdCsvRow>().next() {
        let row = row.map_err(|source| LosPsdLoadError::Csv {
            path: path_ref.to_path_buf(),
            line: source.position().map_or(0, |p| p.line()),
            source,
        })?;
        let line_no = reader.position().line() as usize;
        let f = row.freq_hz;
        if let Some(&prev) = freq_hz.last() {
            if f <= prev {
                return Err(LosPsdLoadError::FrequencyNotAscending {
                    path: path_ref.to_path_buf(),
                    line: line_no,
                });
            }
        }
        freq_hz.push(f);
        psd_x.push(row.psd_los_x_rad2_per_hz);
        psd_y.push(row.psd_los_y_rad2_per_hz);
    }
    if freq_hz.len() < 2 {
        return Err(LosPsdLoadError::TooFewRows {
            path: path_ref.to_path_buf(),
            found: freq_hz.len(),
        });
    }

    Ok(LosPsd {
        freq_hz,
        psd_x_rad2_per_hz: psd_x,
        psd_y_rad2_per_hz: psd_y,
    })
}

#[derive(Debug, Deserialize, Serialize)]
struct LosPsdCsvRow {
    freq_hz: f64,
    psd_los_x_rad2_per_hz: f64,
    psd_los_y_rad2_per_hz: f64,
}

/// Body→world rotation defining the simulator body frame at the nominal
/// pointing.
///
/// Conventions:
///
/// - body **+Z** = boresight (radial on the sky)
/// - body **+X** = sky east at the boresight at `roll_deg = 0` (twisted by
///   `roll_deg` about +Z)
/// - body **+Y** = sky north at the boresight at `roll_deg = 0` (twisted
///   by `roll_deg` about +Z)
///
/// The FEM/optical local X and Y axes are mapped onto body +X and +Y
/// respectively, so a positive rotation about the FEM optical X axis
/// appears as a rotation about body +X in the perturbation step of
/// [`build_trajectory_from_los_psd`].
pub fn nominal_body_to_world(ra_deg: f64, dec_deg: f64, roll_deg: f64) -> UnitQuaternion<f64> {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let (sin_ra, cos_ra) = ra.sin_cos();
    let (sin_dec, cos_dec) = dec.sin_cos();
    let east = Vector3::new(-sin_ra, cos_ra, 0.0);
    let north = Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);
    let bore = Vector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec);
    let base = UnitQuaternion::from_matrix(&Matrix3::from_columns(&[east, north, bore]));
    let twist = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, roll_deg.to_radians()));
    base * twist
}

/// Linear interpolation of `(xs, ys)` at `x`, with zero extrapolation
/// outside the table (matches `numpy.interp(..., left=0.0, right=0.0)`).
///
/// `xs` must be strictly ascending and non-empty.
fn interp_linear_or_zero(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert!(!xs.is_empty());
    if x <= xs[0] || x >= *xs.last().unwrap() {
        return 0.0;
    }
    let pos = xs.partition_point(|&xi| xi < x);
    let x0 = xs[pos - 1];
    let x1 = xs[pos];
    let y0 = ys[pos - 1];
    let y1 = ys[pos];
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Fixed-magnitude / random-phase synthesis of a real time series whose
/// one-sided PSD matches `psd(freq_hz)` linearly interpolated onto the
/// rfft frequency grid.
///
/// Returns `n_samples` samples at sample rate `fs` (Hz).
///
/// # Derivation
///
/// A tone `A cos(2π f_k t + φ)` contributes `A²/2` to the signal
/// variance, and we want each bin to contribute `S(f_k) * df`, so
/// `A_k = sqrt(2 * S(f_k) * df)`. With numpy's irfft normalisation
/// (`X[k] = sum_n x[n] * exp(-2πi k n / N)`), that bin maps to
/// `|X[k]| = (N/2) * A_k = sqrt(N * fs * S(f_k) / 2)`. DC and Nyquist
/// bins are zeroed — no mean, no aliased tone at the band edge.
///
/// `realfft`'s inverse pass is unnormalised; the result is rescaled by
/// `1/N` here to match the numpy convention so the synthesised variance
/// equals the PSD integral.
pub fn synthesize_from_psd(
    freq_hz: &[f64],
    psd: &[f64],
    fs: f64,
    n_samples: usize,
    rng: &mut impl Rng,
) -> Vec<f64> {
    assert!(n_samples >= 4, "n_samples must be >= 4 for IRFFT");
    assert_eq!(
        freq_hz.len(),
        psd.len(),
        "freq_hz and psd must be the same length"
    );
    assert!(fs > 0.0, "fs must be positive");

    let n_one_sided = n_samples / 2 + 1;
    let mut planner = RealFftPlanner::<f64>::new();
    let inv = planner.plan_fft_inverse(n_samples);

    let mut spectrum = inv.make_input_vec();
    debug_assert_eq!(spectrum.len(), n_one_sided);

    let nyquist_bin = if n_samples % 2 == 0 {
        Some(n_one_sided - 1)
    } else {
        None
    };
    let bin_hz = fs / n_samples as f64;
    for (k, slot) in spectrum.iter_mut().enumerate() {
        if k == 0 || Some(k) == nyquist_bin {
            *slot = Complex::new(0.0, 0.0);
            continue;
        }
        let f = k as f64 * bin_hz;
        let s = interp_linear_or_zero(freq_hz, psd, f);
        let magnitude = (n_samples as f64 * fs * s / 2.0).max(0.0).sqrt();
        let phase = rng.random::<f64>() * std::f64::consts::TAU;
        *slot = Complex::from_polar(magnitude, phase);
    }

    let mut output = inv.make_output_vec();
    inv.process(&mut spectrum, &mut output)
        .expect("realfft inverse with matching shapes cannot fail");

    let scale = 1.0 / n_samples as f64;
    for s in output.iter_mut() {
        *s *= scale;
    }
    output
}

/// Build a per-sample [`Trajectory`] by synthesizing two independent
/// body-axis rotation time series from a [`LosPsd`] and composing them
/// onto a nominal RA/Dec/roll pointing.
///
/// Output trajectory has `floor(duration * fs) + 1` waypoints uniformly
/// spaced at `dt = 1/fs`, spanning `[0, n_samples_minus_one / fs]`.
///
/// # Arguments
///
/// - `psd` — two-axis LOS PSD (e.g. from [`load_los_psd_csv`]).
/// - `ra_deg`, `dec_deg`, `roll_deg` — nominal boresight pointing and
///   roll. See [`nominal_body_to_world`] for the body-frame convention.
/// - `duration` — trajectory duration. The synthesis emits
///   `floor(duration * fs) + 1` samples so the last waypoint lands at
///   `floor(duration * fs) / fs` (≤ `duration`).
/// - `fs` — synthesis sample rate in Hz. Should be ≥ `2 *
///   psd.max_freq_hz()`; PSD content above `fs / 2` is discarded.
/// - `seed` — RNG seed for the random-phase synthesis. Two adjacent
///   sub-streams (`seed` and `seed + 1`) drive the X and Y axes
///   independently so the two axes remain uncorrelated.
///
/// Body-Z (roll) is held at zero throughout — the input PSD describes
/// pitch/yaw only.
pub fn build_trajectory_from_los_psd(
    psd: &LosPsd,
    ra_deg: f64,
    dec_deg: f64,
    roll_deg: f64,
    duration: Duration,
    fs: f64,
    seed: u64,
) -> Result<Trajectory, TrajectoryError> {
    assert!(fs > 0.0, "fs must be positive");
    let n_samples = (duration.as_secs_f64() * fs).floor() as usize + 1;
    let n_samples = n_samples.max(4);

    let mut rng_x = StdRng::seed_from_u64(seed);
    let mut rng_y = StdRng::seed_from_u64(seed.wrapping_add(1));
    let body_x = synthesize_from_psd(
        &psd.freq_hz,
        &psd.psd_x_rad2_per_hz,
        fs,
        n_samples,
        &mut rng_x,
    );
    let body_y = synthesize_from_psd(
        &psd.freq_hz,
        &psd.psd_y_rad2_per_hz,
        fs,
        n_samples,
        &mut rng_y,
    );

    let nominal = nominal_body_to_world(ra_deg, dec_deg, roll_deg);

    let waypoints: Vec<Waypoint> = (0..n_samples)
        .map(|i| {
            let t = Duration::from_secs_f64(i as f64 / fs);
            let perturb = UnitQuaternion::from_scaled_axis(Vector3::new(body_x[i], body_y[i], 0.0));
            Waypoint::new(t, nominal * perturb)
        })
        .collect();

    Trajectory::new(waypoints)
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::{assert_abs_diff_eq, assert_relative_eq};
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

    /// Regression for the pole singularity (issue #134): a pink
    /// trajectory 0.36 arcsec from the celestial pole must stay within
    /// jitter scale of the nominal orientation, with no giant
    /// inter-waypoint field rotations.
    #[test]
    fn build_pink_trajectory_is_pole_safe() {
        let pointing = Equatorial::from_degrees(0.0, 89.9999);
        let traj = build_pink_trajectory(pointing, 4.0, 1024, 7, Duration::from_secs(60), 0.0, 0.0)
            .expect("pink trajectory builds at the pole");

        let nominal = nominal_body_to_world(0.0, 89.9999, 0.0);
        let arcsec = (1.0_f64 / 3600.0).to_radians();
        let mut max_excursion = 0.0_f64;
        let mut max_step = 0.0_f64;
        for pair in traj.waypoints().windows(2) {
            max_excursion = max_excursion.max(nominal.angle_to(&pair[0].orientation));
            max_step = max_step.max(pair[0].orientation.angle_to(&pair[1].orientation));
        }
        // Pink noise with a 4-arcsec mean magnitude peaks at a few tens
        // of arcsec; anything near a degree means the pole singularity
        // is back.
        assert!(
            max_excursion < 120.0 * arcsec,
            "peak excursion {:.1} arcsec exceeds 120 arcsec",
            max_excursion / arcsec
        );
        assert!(
            max_step < 120.0 * arcsec,
            "peak inter-waypoint rotation {:.1} arcsec exceeds 120 arcsec",
            max_step / arcsec
        );
    }

    /// The on-sky offset pattern must be independent of declination:
    /// the same seed yields identical per-waypoint angular offsets at
    /// the equator and at the pole, with mean magnitude `mean_arcsec`.
    #[test]
    fn build_pink_trajectory_offsets_match_at_all_declinations() {
        let mean_arcsec = 4.0;
        let build = |dec_deg: f64| {
            build_pink_trajectory(
                Equatorial::from_degrees(10.0, dec_deg),
                mean_arcsec,
                256,
                11,
                Duration::from_secs(30),
                0.0,
                0.0,
            )
            .expect("pink trajectory builds")
        };
        let offsets = |dec_deg: f64| -> Vec<f64> {
            let nominal = nominal_body_to_world(10.0, dec_deg, 0.0);
            build(dec_deg)
                .waypoints()
                .iter()
                .map(|wp| nominal.angle_to(&wp.orientation))
                .collect()
        };

        let at_equator = offsets(0.0);
        let at_pole = offsets(89.9999);
        for (a, b) in at_equator.iter().zip(&at_pole) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }

        let arcsec = (1.0_f64 / 3600.0).to_radians();
        let mean = at_equator.iter().sum::<f64>() / at_equator.len() as f64;
        assert_relative_eq!(mean / arcsec, mean_arcsec, epsilon = 1e-6);
    }

    /// Roll interpolation must survive the body-axis composition: away
    /// from the pole the recovered roll of each waypoint tracks the
    /// linear start→end ramp.
    #[test]
    fn build_pink_trajectory_interpolates_roll() {
        use crate::sims::orientation::roll_of;

        let pointing = Equatorial::from_degrees(80.0, 35.0);
        let traj = build_pink_trajectory(pointing, 2.0, 64, 3, Duration::from_secs(20), 10.0, 30.0)
            .expect("pink trajectory builds");
        let n = traj.waypoints().len();
        for (i, wp) in traj.waypoints().iter().enumerate() {
            let frac = i as f64 / (n - 1) as f64;
            let expected = (10.0 + frac * 30.0).to_radians();
            assert_abs_diff_eq!(roll_of(&wp.orientation), expected, epsilon = 1e-4);
        }
    }

    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Write a synthetic two-axis LOS PSD CSV to a tempfile and return
    /// the handle. Format matches `scripts/los_psd_to_trajectory_csv.py`.
    fn write_psd_csv(rows: &[(f64, f64, f64)]) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("tempfile");
        writeln!(f, "# synthetic LOS PSD for test").unwrap();
        writeln!(f, "# generated by jitter::tests::write_psd_csv").unwrap();
        {
            let mut writer = csv::Writer::from_writer(&mut f);
            for &(freq_hz, psd_los_x_rad2_per_hz, psd_los_y_rad2_per_hz) in rows {
                writer
                    .serialize(LosPsdCsvRow {
                        freq_hz,
                        psd_los_x_rad2_per_hz,
                        psd_los_y_rad2_per_hz,
                    })
                    .unwrap();
            }
            writer.flush().unwrap();
        }
        f.flush().unwrap();
        f
    }

    #[test]
    fn load_los_psd_csv_parses_comments_and_headers() {
        let rows = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0e-12, 2.0e-12),
            (2.0, 1.5e-12, 2.5e-12),
            (10.0, 1.0e-13, 2.0e-13),
        ];
        let tmp = write_psd_csv(&rows);
        let psd = load_los_psd_csv(tmp.path()).expect("loads");
        assert_eq!(psd.freq_hz, vec![0.0, 1.0, 2.0, 10.0]);
        assert_eq!(psd.psd_x_rad2_per_hz, vec![0.0, 1.0e-12, 1.5e-12, 1.0e-13]);
        assert_eq!(psd.psd_y_rad2_per_hz, vec![0.0, 2.0e-12, 2.5e-12, 2.0e-13]);
        assert_relative_eq!(psd.max_freq_hz(), 10.0);
    }

    #[test]
    fn load_los_psd_csv_rejects_non_ascending_frequency() {
        let rows = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0e-12, 2.0e-12),
            (0.5, 1.5e-12, 2.5e-12),
        ];
        let tmp = write_psd_csv(&rows);
        let err = load_los_psd_csv(tmp.path()).expect_err("must reject non-ascending freq");
        assert!(matches!(err, LosPsdLoadError::FrequencyNotAscending { .. }));
    }

    #[test]
    fn load_los_psd_csv_rejects_missing_column() {
        let mut f = NamedTempFile::new().unwrap();
        // Header missing `psd_los_y_rad2_per_hz`.
        writeln!(f, "freq_hz,psd_los_x_rad2_per_hz").unwrap();
        writeln!(f, "1.0,1.0e-12").unwrap();
        writeln!(f, "2.0,1.5e-12").unwrap();
        f.flush().unwrap();
        let err = load_los_psd_csv(f.path()).expect_err("must reject missing column");
        match err {
            LosPsdLoadError::MissingColumn { column, .. } => {
                assert_eq!(column, "psd_los_y_rad2_per_hz")
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    /// Variance recovery: a flat single-axis PSD `S₀` over a band of width
    /// `B` Hz integrates to a variance of `S₀ · B`. After synthesizing,
    /// the empirical variance of the output should match to within
    /// ~few percent on a 65 k-sample series.
    #[test]
    fn synthesize_from_psd_recovers_variance() {
        let fs = 1024.0_f64;
        let n_samples = 1 << 16; // 65536
                                 // Flat PSD of value S0 over [1, 100] Hz, zero elsewhere.
        let s0 = 1.0e-10_f64;
        let bandwidth_hz = 99.0_f64;
        let expected_var = s0 * bandwidth_hz;

        // Three points are enough — linear interp in the band == constant.
        let freq = vec![1.0, 100.0];
        let psd = vec![s0, s0];

        let mut rng = StdRng::seed_from_u64(0xBADCAFE);
        let series = synthesize_from_psd(&freq, &psd, fs, n_samples, &mut rng);
        assert_eq!(series.len(), n_samples);

        let mean = series.iter().sum::<f64>() / n_samples as f64;
        let var = series.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n_samples as f64;

        // Mean should be ~0 (DC bin zeroed).
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1.0e-7);
        // Variance should be within 10% of S0 * B (statistical fluctuation).
        let rel_err = (var - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.10,
            "variance {var:e} vs expected {expected_var:e}, rel_err={rel_err}"
        );
    }

    #[test]
    fn build_trajectory_from_los_psd_anchors_pose_at_nominal_pointing() {
        // Zero PSD everywhere → perturbation should be the identity for
        // every waypoint, so each pose equals the nominal body-to-world.
        let rows = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (100.0, 0.0, 0.0)];
        let tmp = write_psd_csv(&rows);
        let psd = load_los_psd_csv(tmp.path()).unwrap();

        let (ra, dec, roll) = (213.39_f64, -55.86_f64, 17.0_f64);
        let fs = 100.0_f64;
        let duration = Duration::from_secs(2);
        let traj = build_trajectory_from_los_psd(&psd, ra, dec, roll, duration, fs, 42)
            .expect("trajectory builds");

        // floor(2 * 100) + 1 = 201 waypoints at dt = 10 ms.
        assert_eq!(traj.waypoints().len(), 201);

        let nominal = nominal_body_to_world(ra, dec, roll);
        let q0 = traj.orientation_at(Duration::ZERO).unwrap();
        let q_end = traj.orientation_at(duration).unwrap();
        // Zero perturbation: every sample is the nominal pose.
        assert_relative_eq!(q0.angle_to(&nominal), 0.0, epsilon = 1e-9);
        assert_relative_eq!(q_end.angle_to(&nominal), 0.0, epsilon = 1e-9);
    }
}

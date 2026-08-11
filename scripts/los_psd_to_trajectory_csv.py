"""
Synthesize a quaternion-waypoint trajectory CSV from a 2-axis FEM-derived
optical-frame rotation PSD, suitable for
`motion_simulator --mode csv --trajectory-csv ...`.

Input PSD format (header columns, `#`-prefixed comments allowed):
    freq_hz, psd_los_x_rad2_per_hz, psd_los_y_rad2_per_hz

What those columns mean
-----------------------
The two PSD columns are the recovered rotation degrees-of-freedom
about the FEM/optical local axes — *not* sky tangent-plane LOS X/Y
displacements:

  - `psd_los_x_rad2_per_hz`: PSD of the recovered rotation **about**
    the FEM/optical local X axis (in rad²/Hz).
  - `psd_los_y_rad2_per_hz`: PSD of the recovered rotation **about**
    the FEM/optical local Y axis (in rad²/Hz).

These should NOT be interpreted as displacement PSDs along sky east
(RA·cos(Dec)) and north (Dec). They are mechanical optical-axis
rotation spectra; the script maps them onto the simulator's body
frame, where:

  - simulator body **+Z** is the boresight,
  - simulator body **+X** is the FEM optical local X axis,
  - simulator body **+Y** is the FEM optical local Y axis.

`--roll` chooses how the FEM optical X/Y axes are oriented relative
to sky east/north for the nominal RA/Dec. With `--roll 0.0`, body +X
aligns with sky east and body +Y aligns with sky north (right-handed
J2000 tangent frame). Pass non-zero `--roll` (in degrees) to rotate
the optical-axis pair about the boresight before applying the
perturbations.

Roll-convention sanity reference (with --roll 0.0)
--------------------------------------------------
For a rotation by +θ about a body axis, the boresight (body +Z)
moves on the sky as follows (right-hand rule, small angle):

  - Positive rotation about body +X (FEM optical X) →
    boresight tilts **south** by |θ| (body +Z gains a -body-Y component).
  - Positive rotation about body +Y (FEM optical Y) →
    boresight tilts **east** by |θ| (body +Z gains a +body-X component).

Inverting either input column flips the corresponding sky direction;
adding 90° to `--roll` swaps the X and Y deflection directions.

Synthesis
---------
Each axis is synthesized independently via fixed-magnitude /
random-phase per-bin spectrum + irfft. DC and Nyquist bins are
zeroed. Body-Z (roll) perturbation is held at zero — the input PSD
describes pitch/yaw only.

Sample timing
-------------
The output CSV contains `N + 1` waypoints with `dt = 1/fs`, spanning
times `[0, duration]` inclusive. Setting `--duration 100 --fs 2000`
writes 200 001 rows, with the first at `t=0` and the last at
`t=100.0` exactly.

Usage
-----
    uv run --with numpy --with pandas --with scipy \\
      scripts/los_psd_to_trajectory_csv.py \\
      --input los_psd_at_2000rpm.csv \\
      --output trajectory_los_2000rpm_100s.csv \\
      --duration 100.0 --fs 2000.0 \\
      --ra 213.39 --dec -55.86 --roll 0.0 \\
      --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def nominal_body_to_world(ra_deg: float, dec_deg: float, roll_deg: float) -> Rotation:
    """Body->world rotation defining the simulator body frame at the nominal
    pointing (RA, Dec, roll). Conventions:

      - body **+Z** = boresight (radial on the sky)
      - body **+X** = sky east at the boresight at roll=0 (twisted by `roll_deg` about +Z)
      - body **+Y** = sky north at the boresight at roll=0 (twisted by `roll_deg` about +Z)

    The FEM/optical local X and Y axes are mapped onto body +X and +Y
    respectively, so a positive rotation about the FEM optical X axis
    appears as a rotation about body +X in the perturbation step."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    sin_ra, cos_ra = np.sin(ra), np.cos(ra)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    east = np.array([-sin_ra, cos_ra, 0.0])
    north = np.array([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec])
    bore = np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])
    base = Rotation.from_matrix(np.column_stack([east, north, bore]))
    twist = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(roll_deg)])
    return base * twist


def synthesize_from_psd(
    freq_hz: np.ndarray,
    psd: np.ndarray,
    fs: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Fixed-magnitude, random-phase synthesis of a real time series whose
    one-sided PSD matches `psd(freq_hz)` (interpolated linearly onto the
    rfft frequency grid). Returns `n_samples` samples at sample rate `fs`.

    Derivation: a tone A cos(2π f_k t + φ) contributes A^2/2 to the
    signal variance, and we want each bin to contribute S(f_k)*df, so
    A_k = sqrt(2 * S(f_k) * df). With numpy's rfft convention
    (X[k] = sum_n x[n] exp(-2πi k n/N)), that bin maps to
    |X[k]| = (N/2)*A_k = sqrt(N * fs * S(f_k) / 2). DC and Nyquist are
    zeroed (no mean, no aliased tone at the band edge).
    """
    n_one_sided = n_samples // 2 + 1
    rfft_freq = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    psd_interp = np.interp(rfft_freq, freq_hz, psd, left=0.0, right=0.0)
    magnitude = np.sqrt(n_samples * fs * psd_interp / 2.0)
    phase = 2.0 * np.pi * rng.random(n_one_sided)
    X = magnitude * np.exp(1j * phase)
    X[0] = 0.0
    if n_samples % 2 == 0:
        X[-1] = 0.0
    return np.fft.irfft(X, n=n_samples)


def load_psd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a PSD CSV with `#`-prefixed comment lines and three numeric columns."""
    df = pd.read_csv(path, comment="#")
    return (
        df["freq_hz"].to_numpy(dtype=float),
        df["psd_los_x_rad2_per_hz"].to_numpy(dtype=float),
        df["psd_los_y_rad2_per_hz"].to_numpy(dtype=float),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", type=Path, required=True, help="LOS PSD CSV")
    p.add_argument("--output", type=Path, required=True, help="Destination trajectory CSV")
    p.add_argument("--duration", type=float, default=10.0, help="Trajectory duration in seconds")
    p.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Synthesis sample rate in Hz. Default = 2 * max(freq) from the PSD.",
    )
    p.add_argument("--ra", type=float, default=0.0, help="Nominal boresight RA in degrees")
    p.add_argument("--dec", type=float, default=0.0, help="Nominal boresight Dec in degrees")
    p.add_argument(
        "--roll",
        type=float,
        default=0.0,
        help="Roll about the boresight (degrees) that orients the FEM optical "
        "X/Y axes relative to sky east/north. roll=0 aligns body +X with sky "
        "east and body +Y with sky north (so a +X-axis input rotation tilts "
        "the boresight south, a +Y-axis input rotation tilts it east). "
        "See module docstring for the full sky-direction reference.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the random-phase synthesis")
    args = p.parse_args()

    freq, psd_x, psd_y = load_psd(args.input)
    fs = args.fs if args.fs is not None else 2.0 * float(freq.max())
    if fs <= 2.0 * float(freq.max()) - 1e-9:
        print(
            f"warning: fs={fs:.1f} Hz is below 2*max(freq)={2*float(freq.max()):.1f} Hz; "
            "PSD content above Nyquist will be discarded"
        )
    # N + 1 samples spanning [0, duration] inclusive at dt = 1/fs.
    # Setting --duration 100 --fs 2000 writes 200 001 rows, with the
    # first at t=0 and the last at t=100.0 exactly.
    n_samples = int(round(args.duration * fs)) + 1
    if n_samples < 4:
        raise SystemExit(f"duration*fs + 1 = {n_samples} samples is too short")

    rng = np.random.default_rng(args.seed)
    bdyx = synthesize_from_psd(freq, psd_x, fs, n_samples, rng)
    bdyy = synthesize_from_psd(freq, psd_y, fs, n_samples, rng)
    bdyz = np.zeros_like(bdyx)

    nominal = nominal_body_to_world(args.ra, args.dec, args.roll)
    perturb = Rotation.from_rotvec(np.stack([bdyx, bdyy, bdyz], axis=1))
    total = nominal * perturb

    quat_xyzw = total.as_quat()
    quat_wxyz = np.column_stack(
        [quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]]
    )

    t = np.arange(n_samples) / fs
    out = pd.DataFrame(
        {
            "time_s": t,
            "qw": quat_wxyz[:, 0],
            "qx": quat_wxyz[:, 1],
            "qy": quat_wxyz[:, 2],
            "qz": quat_wxyz[:, 3],
        }
    )
    out.to_csv(args.output, index=False)

    var_x = float(np.var(bdyx))
    var_y = float(np.var(bdyy))
    target_var_x = float(np.trapezoid(psd_x, freq))
    target_var_y = float(np.trapezoid(psd_y, freq))
    print(
        f"wrote {len(out)} rows to {args.output}\n"
        f"  duration={args.duration:.3f}s  fs={fs:.1f}Hz  dt={1/fs:.6f}s  "
        f"(samples at t=0 ... t={t[-1]:.6f})\n"
        f"  axis X: synth std={np.sqrt(var_x)*206265:.3f}\" "
        f"(target from PSD integral {np.sqrt(target_var_x)*206265:.3f}\")\n"
        f"  axis Y: synth std={np.sqrt(var_y)*206265:.3f}\" "
        f"(target from PSD integral {np.sqrt(target_var_y)*206265:.3f}\")"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

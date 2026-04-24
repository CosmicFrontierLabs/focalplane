"""
Convert an ODY Att Error CSV into the quaternion-waypoint format consumed
by `motion_simulator --mode csv --trajectory-csv …`.

The ODY file records *errors* in body axes (BDYX, BDYY, BDYZ in radians)
around a commanded attitude. This script composes those errors onto a
user-supplied nominal pointing (RA/Dec/roll, all optional) to produce a
series of absolute orientation quaternions, and writes a simple CSV with
columns `time_s, qw, qx, qy, qz`.

Usage:
  uv run --with pandas --with numpy --with scipy \
    scripts/ody_to_trajectory_csv.py \
    --input Run0001_20260224205907_2999.csv \
    --output trajectory.csv \
    --ra 213.39 --dec -55.86 --roll 0.0 \
    --start-time 1000 --end-time 2000

Small-angle approximation is fine for ODY errors (<< 1 rad); the
perturbation quaternion is built directly from the axis-angle form with
a non-normalized input vector, then normalized.

Keeps the Rust side completely agnostic about CSV layout or conventions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


DEFAULT_COLUMNS = {
    "time": "SCET",
    "bdyx": "OER_TOTATTERR_BDYX",
    "bdyy": "OER_TOTATTERR_BDYY",
    "bdyz": "OER_TOTATTERR_BDYZ",
}


def nominal_body_to_world(ra_deg: float, dec_deg: float, roll_deg: float) -> Rotation:
    """Return the nominal body->world rotation: body +Z = boresight, body +X = east, body +Y = north."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    sin_ra, cos_ra = np.sin(ra), np.cos(ra)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    east = np.array([-sin_ra, cos_ra, 0.0])
    north = np.array([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec])
    bore = np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])
    base_mat = np.column_stack([east, north, bore])
    base = Rotation.from_matrix(base_mat)
    twist = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(roll_deg)])
    return base * twist


def body_error_to_perturbation(bdyx: np.ndarray, bdyy: np.ndarray, bdyz: np.ndarray) -> Rotation:
    """Stack per-sample small-angle body-axis errors into a Rotation sequence."""
    vecs = np.stack([bdyx, bdyy, bdyz], axis=1)  # (N, 3)
    return Rotation.from_rotvec(vecs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", type=Path, required=True, help="ODY telemetry CSV")
    p.add_argument("--output", type=Path, required=True, help="Destination trajectory CSV")
    p.add_argument("--ra", type=float, default=0.0, help="Nominal boresight RA in degrees")
    p.add_argument("--dec", type=float, default=0.0, help="Nominal boresight Dec in degrees")
    p.add_argument("--roll", type=float, default=0.0, help="Nominal roll in degrees")
    p.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Lower bound on SCET-relative time (seconds)",
    )
    p.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Upper bound on SCET-relative time (seconds)",
    )
    p.add_argument(
        "--time-col",
        default=DEFAULT_COLUMNS["time"],
        help=f"Time column name (default: {DEFAULT_COLUMNS['time']})",
    )
    p.add_argument("--bdyx-col", default=DEFAULT_COLUMNS["bdyx"])
    p.add_argument("--bdyy-col", default=DEFAULT_COLUMNS["bdyy"])
    p.add_argument("--bdyz-col", default=DEFAULT_COLUMNS["bdyz"])
    args = p.parse_args()

    # Row 1 of the reference telemetry CSVs holds the short OER_* names.
    df = pd.read_csv(args.input, header=1, low_memory=False)
    t_raw = df[args.time_col].to_numpy(dtype=float)
    t = t_raw - t_raw[0]

    mask = np.ones_like(t, dtype=bool)
    if args.start_time is not None:
        mask &= t >= args.start_time
    if args.end_time is not None:
        mask &= t <= args.end_time
    if not mask.any():
        raise SystemExit(
            f"No rows in window [{args.start_time}, {args.end_time}] (spans 0 to {t[-1]:.2f}s)"
        )

    df = df.loc[mask].reset_index(drop=True)
    t = t[mask]
    t_rel = t - t[0]  # re-zero so the output starts at t=0

    bdyx = df[args.bdyx_col].to_numpy(dtype=float)
    bdyy = df[args.bdyy_col].to_numpy(dtype=float)
    bdyz = df[args.bdyz_col].to_numpy(dtype=float)

    nominal = nominal_body_to_world(args.ra, args.dec, args.roll)
    perturb = body_error_to_perturbation(bdyx, bdyy, bdyz)
    # Compose in world frame: absolute = nominal ∘ perturbation_in_body
    total = nominal * perturb

    # scipy returns (x, y, z, w); reorder to (w, x, y, z) to match nalgebra.
    quat_xyzw = total.as_quat()
    quat_wxyz = np.column_stack(
        [quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]]
    )

    out = pd.DataFrame(
        {
            "time_s": t_rel,
            "qw": quat_wxyz[:, 0],
            "qx": quat_wxyz[:, 1],
            "qy": quat_wxyz[:, 2],
            "qz": quat_wxyz[:, 3],
        }
    )
    out.to_csv(args.output, index=False)

    print(
        f"wrote {len(out)} rows to {args.output}  "
        f"(window {t_rel[0]:.3f}–{t_rel[-1]:.3f} s, dt={np.mean(np.diff(t_rel)):.4f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

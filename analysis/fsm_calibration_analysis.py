#!/usr/bin/env python3
"""
FSM Calibration Data Analysis

Analyzes calibration sweep data from fsm_calibration_controller and generates
plots:
1. 2D centroid trajectories for each axis sweep (one file per axis)
2. Linear response plots (X and Y vs FSM command) with regression fits

Usage:
    python3 fsm_calibration_analysis.py calibration_sweep.csv

Output:
    calibration_2d_axis1.png   - 2D centroid positions for axis 1 sweep
    calibration_2d_axis2.png   - 2D centroid positions for axis 2 sweep
    calibration_linearity.png  - Linear response plots with R² values
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_calibration_data(csv_path: str) -> pd.DataFrame:
    """Load calibration CSV, skipping comment lines."""
    return pd.read_csv(csv_path, comment="#")


def plot_2d_sweep_single_axis(
    df: pd.DataFrame, axis: int, output_path: str, xlim=None, ylim=None
) -> dict:
    """
    Plot 2D centroid positions for a single axis sweep.

    Returns motion summary dict for this axis.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    cmap = plt.cm.viridis

    data = df[df["axis"] == axis]

    # Get unique positions for coloring
    positions = sorted(data["fsm_command_urad"].unique())
    colors = cmap(np.linspace(0, 1, len(positions)))

    # Plot each position's samples
    for pos, color in zip(positions, colors):
        pos_data = data[data["fsm_command_urad"] == pos]
        ax.scatter(
            pos_data["centroid_x"],
            pos_data["centroid_y"],
            c=[color],
            alpha=0.3,
            s=2,
        )

    # Plot mean positions with larger markers
    means = (
        data.groupby("fsm_command_urad")
        .agg({"centroid_x": "mean", "centroid_y": "mean"})
        .reset_index()
    )

    ax.plot(means["centroid_x"], means["centroid_y"], "k-", linewidth=2, zorder=10)
    ax.scatter(
        means["centroid_x"],
        means["centroid_y"],
        c="red",
        alpha=0.6,
        s=30,
        marker="x",
        linewidths=1.5,
        zorder=11,
    )

    # Annotate start and end
    ax.annotate(
        f"Start\n{positions[0]:.0f}urad",
        (means["centroid_x"].iloc[0], means["centroid_y"].iloc[0]),
        textcoords="offset points",
        xytext=(-30, 10),
        fontsize=9,
    )
    ax.annotate(
        f"End\n{positions[-1]:.0f}urad",
        (means["centroid_x"].iloc[-1], means["centroid_y"].iloc[-1]),
        textcoords="offset points",
        xytext=(10, -20),
        fontsize=9,
    )

    if axis == 1:
        ax.set_title("Axis 1 Sweep (Axis 1 varied, Axis 2 at center)", fontsize=12)
    else:
        ax.set_title("Axis 2 Sweep (Axis 2 varied, Axis 1 at center)", fontsize=12)

    ax.set_xlabel("Centroid X (pixels)")
    ax.set_ylabel("Centroid Y (pixels)")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=positions[0], vmax=positions[-1])
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("FSM Command (urad)")

    # Apply axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Equal aspect ratio so pixels are square
    ax.set_aspect("equal")

    # Add grid lines at every pixel, but only label every 10th
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(True, which="major", alpha=0.7, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.3, linewidth=0.3)

    # Calculate motion summary
    dx = means["centroid_x"].iloc[-1] - means["centroid_x"].iloc[0]
    dy = means["centroid_y"].iloc[-1] - means["centroid_y"].iloc[0]
    total_motion = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dy, dx))

    summary = {
        "delta_x": dx,
        "delta_y": dy,
        "total_motion": total_motion,
        "angle": angle,
    }

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return summary


def plot_linearity(df: pd.DataFrame, output_path: str) -> dict:
    """
    Plot linear response (X and Y vs FSM command) with regression fits.

    Returns regression summary dict.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    summary = {}

    for axis in [1, 2]:
        data = df[df["axis"] == axis]
        positions = (
            data.groupby("fsm_command_urad")
            .agg(
                {
                    "centroid_x": ["mean", "std"],
                    "centroid_y": ["mean", "std"],
                }
            )
            .reset_index()
        )
        positions.columns = ["cmd", "x_mean", "x_std", "y_mean", "y_std"]

        summary[axis] = {}

        # Plot X response
        ax = axes[axis - 1, 0]
        ax.errorbar(
            positions["cmd"],
            positions["x_mean"],
            yerr=positions["x_std"],
            fmt="o-",
            capsize=3,
            color="C0",
        )
        slope, intercept, r, _, _ = stats.linregress(
            positions["cmd"], positions["x_mean"]
        )
        fit_line = slope * positions["cmd"] + intercept
        ax.plot(
            positions["cmd"],
            fit_line,
            "r--",
            label=f"slope={slope:.6f} px/urad\nR²={r**2:.4f}",
        )
        ax.set_xlabel("FSM Command (urad)")
        ax.set_ylabel("Centroid X (pixels)")
        ax.set_title(f"Axis {axis} - X Response")
        ax.legend()
        ax.grid(True, alpha=0.3)

        summary[axis]["x_slope"] = slope
        summary[axis]["x_r2"] = r**2

        # Plot Y response
        ax = axes[axis - 1, 1]
        ax.errorbar(
            positions["cmd"],
            positions["y_mean"],
            yerr=positions["y_std"],
            fmt="o-",
            capsize=3,
            color="C1",
        )
        slope, intercept, r, _, _ = stats.linregress(
            positions["cmd"], positions["y_mean"]
        )
        fit_line = slope * positions["cmd"] + intercept
        ax.plot(
            positions["cmd"],
            fit_line,
            "r--",
            label=f"slope={slope:.6f} px/urad\nR²={r**2:.4f}",
        )
        ax.set_xlabel("FSM Command (urad)")
        ax.set_ylabel("Centroid Y (pixels)")
        ax.set_title(f"Axis {axis} - Y Response")
        ax.legend()
        ax.grid(True, alpha=0.3)

        summary[axis]["y_slope"] = slope
        summary[axis]["y_r2"] = r**2

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FSM calibration sweep data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to calibration CSV file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for plots (default: current directory)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {csv_path}...")
    df = load_calibration_data(str(csv_path))
    print(f"Loaded {len(df)} samples")

    # Compute shared axis limits from both axes combined
    all_x = df["centroid_x"]
    all_y = df["centroid_y"]
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.1
    xlim = (all_x.min() - x_margin, all_x.max() + x_margin)
    ylim = (all_y.min() - y_margin, all_y.max() + y_margin)

    # Generate 2D sweep plots (one per axis) with same axis limits
    motion_summary = {}
    for axis in [1, 2]:
        sweep_path = output_dir / f"calibration_2d_axis{axis}.png"
        print(f"Generating 2D sweep plot for axis {axis}: {sweep_path}")
        motion_summary[axis] = plot_2d_sweep_single_axis(
            df, axis, str(sweep_path), xlim=xlim, ylim=ylim
        )

    # Generate linearity plot
    linearity_path = output_dir / "calibration_linearity.png"
    print(f"Generating linearity plot: {linearity_path}")
    regression_summary = plot_linearity(df, str(linearity_path))

    # Print summary
    print("\n=== Motion Summary ===")
    for axis in [1, 2]:
        m = motion_summary[axis]
        print(f"\nAxis {axis}:")
        print(f"  Delta X: {m['delta_x']:.2f} pixels")
        print(f"  Delta Y: {m['delta_y']:.2f} pixels")
        print(f"  Total motion: {m['total_motion']:.2f} pixels")
        print(f"  Motion angle: {m['angle']:.1f} deg")

    print("\n=== Linearity Summary ===")
    for axis in [1, 2]:
        r = regression_summary[axis]
        print(f"\nAxis {axis}:")
        print(f"  X: slope={r['x_slope']:.6f} px/urad, R²={r['x_r2']:.4f}")
        print(f"  Y: slope={r['y_slope']:.6f} px/urad, R²={r['y_r2']:.4f}")

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()

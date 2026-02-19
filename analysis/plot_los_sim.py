#!/usr/bin/env python3
"""Plot LOS jitter simulation results.

Usage:
    python analysis/plot_los_sim.py [output_dir]

Reads los_sim_timeseries.csv and los_sim_psd.csv from the output directory
and generates time series and PSD plots.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_timeseries(df: pd.DataFrame, output_path: Path):
    """Plot time series of jitter, error, and FSM commands."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Limit to first 2 seconds for readability
    mask = df["time"] <= 2.0
    t = df["time"][mask]

    # Input jitter
    ax = axes[0]
    ax.plot(t, df["jitter_x"][mask], "b-", alpha=0.7, label="Jitter X")
    ax.plot(t, df["jitter_y"][mask], "r-", alpha=0.7, label="Jitter Y")
    ax.set_ylabel("Jitter (px)")
    ax.legend(loc="upper right")
    ax.set_title("Input Jitter")
    ax.grid(True, alpha=0.3)

    # Centroid error after correction
    ax = axes[1]
    ax.plot(t, df["error_x"][mask], "b-", alpha=0.7, label="Error X")
    ax.plot(t, df["error_y"][mask], "r-", alpha=0.7, label="Error Y")
    ax.set_ylabel("Error (px)")
    ax.legend(loc="upper right")
    ax.set_title("Centroid Error (after FSM correction)")
    ax.grid(True, alpha=0.3)

    # FSM commands
    ax = axes[2]
    ax.plot(t, df["fsm_x"][mask], "b-", alpha=0.7, label="FSM X")
    ax.plot(t, df["fsm_y"][mask], "r-", alpha=0.7, label="FSM Y")
    ax.set_ylabel("FSM (µrad)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.set_title("FSM Commands")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved time series plot: {output_path}")
    plt.close()


def plot_psd(df: pd.DataFrame, output_path: Path):
    """Plot power spectral density comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    freq = df["frequency"]

    # X axis
    ax = axes[0]
    ax.semilogy(freq, df["psd_jitter_x"], "b-", alpha=0.7, label="Input Jitter")
    ax.semilogy(freq, df["psd_error_x"], "g-", alpha=0.7, label="Residual Error")
    ax.set_ylabel("PSD (px²/Hz)")
    ax.legend(loc="upper right")
    ax.set_title("X Axis Power Spectral Density")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, freq.max())

    # Y axis
    ax = axes[1]
    ax.semilogy(freq, df["psd_jitter_y"], "r-", alpha=0.7, label="Input Jitter")
    ax.semilogy(freq, df["psd_error_y"], "g-", alpha=0.7, label="Residual Error")
    ax.set_ylabel("PSD (px²/Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.legend(loc="upper right")
    ax.set_title("Y Axis Power Spectral Density")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved PSD plot: {output_path}")
    plt.close()


def plot_rejection_ratio(psd_df: pd.DataFrame, output_path: Path):
    """Plot rejection ratio (input/residual) vs frequency."""
    fig, ax = plt.subplots(figsize=(10, 5))

    freq = psd_df["frequency"]
    # Avoid division by zero
    eps = 1e-20
    rejection_x = psd_df["psd_jitter_x"] / (psd_df["psd_error_x"] + eps)
    rejection_y = psd_df["psd_jitter_y"] / (psd_df["psd_error_y"] + eps)

    ax.semilogy(freq, rejection_x, "b-", alpha=0.7, label="X axis")
    ax.semilogy(freq, rejection_y, "r-", alpha=0.7, label="Y axis")
    ax.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="Unity (no rejection)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Rejection Ratio (PSD_in / PSD_out)")
    ax.set_title("Closed-Loop Rejection Ratio vs Frequency")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, freq.max())

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved rejection ratio plot: {output_path}")
    plt.close()


def compute_stats(ts_df: pd.DataFrame):
    """Compute and print RMS statistics."""
    jitter_rms_x = np.sqrt(np.mean(ts_df["jitter_x"] ** 2))
    jitter_rms_y = np.sqrt(np.mean(ts_df["jitter_y"] ** 2))
    error_rms_x = np.sqrt(np.mean(ts_df["error_x"] ** 2))
    error_rms_y = np.sqrt(np.mean(ts_df["error_y"] ** 2))
    fsm_rms_x = np.sqrt(np.mean(ts_df["fsm_x"] ** 2))
    fsm_rms_y = np.sqrt(np.mean(ts_df["fsm_y"] ** 2))

    rejection_x = jitter_rms_x / error_rms_x if error_rms_x > 0 else float("inf")
    rejection_y = jitter_rms_y / error_rms_y if error_rms_y > 0 else float("inf")

    print("\n=== RMS Statistics ===")
    print(f"Input Jitter:  X={jitter_rms_x:.4f} px, Y={jitter_rms_y:.4f} px")
    print(f"Residual Error: X={error_rms_x:.4f} px, Y={error_rms_y:.4f} px")
    print(f"FSM Commands:  X={fsm_rms_x:.2f} µrad, Y={fsm_rms_y:.2f} µrad")
    print(f"Rejection:     X={rejection_x:.1f}x ({20*np.log10(rejection_x):.1f} dB)")
    print(f"               Y={rejection_y:.1f}x ({20*np.log10(rejection_y):.1f} dB)")


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    timeseries_path = output_dir / "los_sim_timeseries.csv"
    psd_path = output_dir / "los_sim_psd.csv"

    if not timeseries_path.exists():
        print(f"Error: {timeseries_path} not found")
        print("Run los_jitter_sim first to generate data")
        sys.exit(1)

    print(f"Loading data from {output_dir}")

    # Load data
    ts_df = pd.read_csv(timeseries_path)
    psd_df = pd.read_csv(psd_path) if psd_path.exists() else None

    # Print statistics
    compute_stats(ts_df)

    # Generate plots
    plot_timeseries(ts_df, output_dir / "los_sim_timeseries.png")

    if psd_df is not None:
        plot_psd(psd_df, output_dir / "los_sim_psd.png")
        plot_rejection_ratio(psd_df, output_dir / "los_sim_rejection.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

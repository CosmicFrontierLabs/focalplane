#!/usr/bin/env python3
"""Plot gyro data from parsed CSV."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_full_csv(csv_path: str) -> pd.DataFrame:
    """Load full CSV including skipped rows."""
    return pd.read_csv(csv_path, low_memory=False)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and filter to data rows only."""
    df = pd.read_csv(csv_path, low_memory=False)
    return df[df["type"] == "data"].copy()


def compute_timestamp_diffs(timestamps: np.ndarray, max_val: int = 2**32) -> np.ndarray:
    """Compute timestamp differences, correcting for rollover."""
    diffs = np.diff(timestamps.astype(np.int64))
    # Correct for rollover - if diff is negative, add max_val
    diffs = np.where(diffs < 0, diffs + max_val, diffs)
    return diffs


def find_skip_indices(df_full: pd.DataFrame) -> list[int]:
    """Find sample indices where skipped data occurred."""
    is_data = (df_full["type"] == "data").values
    # Cumsum of data rows gives us the data index at each position
    data_idx_at_pos = np.cumsum(is_data)
    # Where we have skipped rows, get the data index (which is the index of next data row)
    skip_mask = ~is_data
    return data_idx_at_pos[skip_mask].tolist()


def plot_timestamp_sawtooth(df: pd.DataFrame, skip_indices: list[int], output_path: str = None):
    """Plot timestamps showing sawtooth pattern with markers for missing data."""
    timestamps = df["gyro_time"].values
    sample_idx = np.arange(len(timestamps))

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(sample_idx, timestamps, linewidth=0.5, alpha=0.8)

    # Add vertical lines where data was skipped
    for idx in skip_indices:
        ax.axvline(idx, color="red", linestyle="--", alpha=0.7, linewidth=1)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Timestamp (counts)")
    ax.set_title(f"Gyro Timestamps (sawtooth from rollover)\nRed lines = missing data ({len(skip_indices)} gaps)")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")


def plot_timestamp_histogram(df: pd.DataFrame, output_path: str = None):
    """Plot histogram of timestamp differences."""
    timestamps = df["gyro_time"].values
    diffs = compute_timestamp_diffs(timestamps)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Use log scale for y-axis since most values cluster tightly
    ax.hist(diffs, bins=100, edgecolor="black", alpha=0.7)
    ax.set_yscale("log")

    ax.set_xlabel("Timestamp Difference (counts)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title(f"Gyro Timestamp Differences\n(n={len(diffs):,} samples)")

    # Add stats
    median = np.median(diffs)
    mean = np.mean(diffs)
    std = np.std(diffs)
    ax.axvline(median, color="red", linestyle="--", label=f"Median: {median:.1f}")
    ax.axvline(mean, color="green", linestyle="--", label=f"Mean: {mean:.1f}")
    ax.legend()

    # Add text with stats
    stats_text = f"Std: {std:.1f}\nMin: {diffs.min()}\nMax: {diffs.max()}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")


def plot_temperatures(df: pd.DataFrame, output_path: str = None):
    """Plot all temperature channels over time."""
    sample_idx = np.arange(len(df))

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    temp_fields = [
        ("board_temp", "Board Temperature"),
        ("sia_fil_temp", "SIA Filter Temperature"),
        ("org_fil_temp", "Organizer Temperature"),
        ("inter_temp", "Interface Temperature"),
    ]

    for ax, (field, label) in zip(axes, temp_fields):
        values = df[field].values
        ax.plot(sample_idx, values, linewidth=0.3, alpha=0.8)
        ax.set_ylabel(f"{label}\n(raw counts)")
        ax.grid(True, alpha=0.3)

        # Add stats
        mean = np.mean(values)
        std = np.std(values)
        ax.axhline(mean, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(0.02, 0.95, f"Mean: {mean:.1f}, Std: {std:.2f}",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[-1].set_xlabel("Sample Index")
    axes[0].set_title(f"Gyro Temperature Channels\n(n={len(df):,} samples)")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot gyro data from parsed CSV")
    parser.add_argument("csv_file", help="Input CSV file from parse_dump")
    parser.add_argument("-o", "--output", help="Output PNG file")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    df_full = load_full_csv(args.csv_file)
    df = df_full[df_full["type"] == "data"].copy()
    print(f"Loaded {len(df):,} data records")

    skip_indices = find_skip_indices(df_full)
    print(f"Found {len(skip_indices)} skip events")

    plot_timestamp_sawtooth(df, skip_indices, args.output)
    plot_temperatures(df)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

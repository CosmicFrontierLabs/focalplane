#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='Plot histograms of tracking error standard deviations')
parser.add_argument('--input', '-i', type=str, help='Path to results.csv file (default: most recent experiment)')
parser.add_argument('--output', '-o', type=str, default='plots/error_histograms.png', help='Output plot path')
parser.add_argument('--cutoff', '-c', type=float, default=0.5, help='Maximum error value to include in pixels (default: 0.5)')
parser.add_argument('--no-show', action='store_true', help='Do not display plot interactively (default: show)')
args = parser.parse_args()

# Ensure output directory exists
Path(args.output).parent.mkdir(parents=True, exist_ok=True)

if args.input:
    results_file = Path(args.input)
else:
    fgs_output_dir = Path("fgs_output")
    experiment_dirs = sorted(fgs_output_dir.glob("experiment_*"))
    if not experiment_dirs:
        raise FileNotFoundError("No experiment directories found in fgs_output/")
    latest_experiment = experiment_dirs[-1]
    results_file = latest_experiment / "results.csv"

print(f"Loading data from: {results_file}")
df = pd.read_csv(results_file)

# Filter data based on cutoff
df_filtered = df[
    (df['std_x_error_pixels'] <= args.cutoff) &
    (df['std_y_error_pixels'] <= args.cutoff)
]
print(f"Filtered from {len(df)} to {len(df_filtered)} rows (cutoff: {args.cutoff} pixels)")

# Get unique exposure times
exposure_times = sorted(df_filtered['exposure_ms'].unique())
num_exposures = len(exposure_times)

print(f"Found {num_exposures} exposure times: {exposure_times}")

# Calculate global x-axis limits
x_min, x_max = 0, args.cutoff
y_min, y_max = 0, args.cutoff

# Create figure with 2 rows and N columns
fig, axes = plt.subplots(2, num_exposures, figsize=(4 * num_exposures, 8))

# If only one exposure, make axes 2D
if num_exposures == 1:
    axes = axes.reshape(2, 1)

# Plot histograms for each exposure time
for col_idx, exposure in enumerate(exposure_times):
    # Filter data for this exposure time
    exposure_data = df_filtered[df_filtered['exposure_ms'] == exposure]

    # Top row: std_x_error_pixels
    ax_x = axes[0, col_idx]
    color_x = 'tab:blue'
    n_x, bins_x, patches_x = ax_x.hist(
        exposure_data['std_x_error_pixels'],
        bins=20,
        range=(x_min, x_max),
        edgecolor='black',
        alpha=0.7,
        color=color_x
    )
    mean_x = exposure_data['std_x_error_pixels'].mean()
    ax_x.axvline(mean_x, color=color_x, linestyle='--', linewidth=2, alpha=0.5)
    ax_x.text(mean_x, ax_x.get_ylim()[1] * 0.95, f'μ={mean_x:.3f} (n={len(exposure_data)})',
              ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax_x.set_title(f'Exposure: {exposure} ms')
    ax_x.set_xlabel('std_x_error_pixels')
    ax_x.set_ylabel('Frequency')
    ax_x.set_xlim(x_min, x_max)
    ax_x.grid(True, alpha=0.3)

    # Bottom row: std_y_error_pixels
    ax_y = axes[1, col_idx]
    color_y = 'tab:orange'
    n_y, bins_y, patches_y = ax_y.hist(
        exposure_data['std_y_error_pixels'],
        bins=20,
        range=(y_min, y_max),
        edgecolor='black',
        alpha=0.7,
        color=color_y
    )
    mean_y = exposure_data['std_y_error_pixels'].mean()
    ax_y.axvline(mean_y, color=color_y, linestyle='--', linewidth=2, alpha=0.5)
    ax_y.text(mean_y, ax_y.get_ylim()[1] * 0.95, f'μ={mean_y:.3f}',
              ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax_y.set_xlabel('std_y_error_pixels')
    ax_y.set_ylabel('Frequency')
    ax_y.set_xlim(y_min, y_max)
    ax_y.grid(True, alpha=0.3)

# Add row labels
fig.text(0.02, 0.75, 'X Error', rotation=90, va='center', fontsize=14, fontweight='bold')
fig.text(0.02, 0.25, 'Y Error', rotation=90, va='center', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {args.output}")

# Second figure: Magnitude vs X Error for each exposure
fig2, axes2 = plt.subplots(1, num_exposures, figsize=(5 * num_exposures, 5))

# If only one exposure, make axes iterable
if num_exposures == 1:
    axes2 = [axes2]

# Calculate global magnitude axis limits
mag_min = df_filtered['guide_star_magnitude'].min()
mag_max = df_filtered['guide_star_magnitude'].max()

for col_idx, exposure in enumerate(exposure_times):
    exposure_data = df_filtered[df_filtered['exposure_ms'] == exposure]

    ax = axes2[col_idx]
    ax.scatter(
        exposure_data['guide_star_magnitude'],
        exposure_data['std_x_error_pixels'],
        alpha=0.5,
        s=20,
        edgecolors='black',
        linewidth=0.5
    )
    ax.set_title(f'Exposure: {exposure} ms')
    ax.set_xlabel('Guide Star Magnitude')
    ax.set_ylabel('std_x_error_pixels')
    ax.set_xlim(mag_min, mag_max)
    ax.set_ylim(0, args.cutoff)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
magnitude_output = Path(args.output).parent / 'magnitude_vs_error.png'
plt.savefig(magnitude_output, dpi=150, bbox_inches='tight')
print(f"Saved magnitude plot to: {magnitude_output}")

if not args.no_show:
    plt.show()

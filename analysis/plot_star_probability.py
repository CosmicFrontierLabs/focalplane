#!/usr/bin/env python3
"""
Analyze probability of finding stars of a given magnitude in a frame.

Uses sensor-view-stats CSV output to generate:
1. Magnitude distribution histogram across all pointings
2. Cumulative probability plot (P(star brighter than X in frame))
3. Per-magnitude probability distribution
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Plot star brightness probability distributions from sensor-view-stats output'
)
parser.add_argument(
    '--input', '-i', type=str,
    help='Path to sensor_view_stats.csv file',
    default='sensor_view_stats.csv'
)
parser.add_argument(
    '--output', '-o', type=str,
    default='plots/star_probability.png',
    help='Output plot path'
)
parser.add_argument(
    '--no-show', action='store_true',
    help='Do not display plot interactively'
)
args = parser.parse_args()

# Ensure output directory exists
Path(args.output).parent.mkdir(parents=True, exist_ok=True)

# Load CSV file
print(f"Loading data from: {args.input}")

# Read CSV, skipping header lines until we find "Detailed Results:"
with open(args.input, 'r') as f:
    lines = f.readlines()

# Find the line with "Detailed Results:"
data_start = None
for i, line in enumerate(lines):
    if 'Detailed Results:' in line:
        data_start = i + 1
        break

if data_start is None:
    raise ValueError("Could not find 'Detailed Results:' section in CSV")

# Read the actual data
df = pd.read_csv(args.input, skiprows=data_start)

print(f"Loaded {len(df)} pointings")

# Parse the star_magnitudes column (semicolon-separated values in quotes)
all_magnitudes = []
magnitudes_per_pointing = []

for idx, row in df.iterrows():
    mag_str = row['star_magnitudes']
    if pd.notna(mag_str) and mag_str.strip():
        mags = [float(m) for m in mag_str.split(';') if m.strip()]
        all_magnitudes.extend(mags)
        magnitudes_per_pointing.append(mags)
    else:
        magnitudes_per_pointing.append([])

print(f"Total stars across all pointings: {len(all_magnitudes)}")

# Create figure with 4 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Overall magnitude histogram
ax1.hist(all_magnitudes, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Star Magnitude')
ax1.set_ylabel('Count (across all pointings)')
ax1.set_title('Star Magnitude Distribution')
ax1.grid(True, alpha=0.3)
ax1.axvline(np.median(all_magnitudes), color='red', linestyle='--',
            label=f'Median: {np.median(all_magnitudes):.2f}')
ax1.legend()

# Plot 2: Cumulative probability - P(at least one star brighter than X)
mag_bins = np.linspace(min(all_magnitudes), max(all_magnitudes), 100)
prob_brighter = []

for mag_threshold in mag_bins:
    # Count how many pointings have at least one star brighter than threshold
    pointings_with_bright_star = sum(
        1 for mags in magnitudes_per_pointing
        if any(m <= mag_threshold for m in mags)
    )
    prob = pointings_with_bright_star / len(magnitudes_per_pointing)
    prob_brighter.append(prob)

ax2.plot(mag_bins, prob_brighter, linewidth=2, color='darkgreen')
ax2.set_xlabel('Magnitude Threshold')
ax2.set_ylabel('P(at least one star ≤ threshold)')
ax2.set_title('Probability of Finding Star Brighter Than Threshold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.05])

# Add horizontal lines for common probability levels
for p in [0.5, 0.9, 0.95, 0.99]:
    # Find magnitude where probability crosses this level
    idx = np.argmin(np.abs(np.array(prob_brighter) - p))
    mag_at_p = mag_bins[idx]
    ax2.axhline(p, color='gray', linestyle=':', alpha=0.5)
    ax2.text(mag_bins[-1], p, f'  {p:.0%}', va='center', fontsize=9)
    ax2.axvline(mag_at_p, color='gray', linestyle=':', alpha=0.5)
    ax2.text(mag_at_p, 0.02, f'{mag_at_p:.1f}', ha='center', fontsize=9)

# Plot 3: Average number of stars per magnitude bin per pointing
mag_bin_edges = np.arange(int(min(all_magnitudes)), int(max(all_magnitudes)) + 2, 0.5)
stars_per_bin = []

for i in range(len(mag_bin_edges) - 1):
    bin_min, bin_max = mag_bin_edges[i], mag_bin_edges[i + 1]
    total_stars_in_bin = sum(
        sum(1 for m in mags if bin_min <= m < bin_max)
        for mags in magnitudes_per_pointing
    )
    avg_per_pointing = total_stars_in_bin / len(magnitudes_per_pointing)
    stars_per_bin.append(avg_per_pointing)

bin_centers = (mag_bin_edges[:-1] + mag_bin_edges[1:]) / 2
ax3.bar(bin_centers, stars_per_bin, width=0.4, edgecolor='black',
        alpha=0.7, color='coral')
ax3.set_xlabel('Magnitude Bin')
ax3.set_ylabel('Average Stars per Pointing')
ax3.set_title('Average Star Count per Magnitude Bin')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Histogram of brightest star in each pointing
brightest_per_pointing = [min(mags) if mags else np.nan for mags in magnitudes_per_pointing]
brightest_per_pointing = [b for b in brightest_per_pointing if not np.isnan(b)]

ax4.hist(brightest_per_pointing, bins=30, edgecolor='black', alpha=0.7, color='darkviolet')
ax4.set_xlabel('Magnitude of Brightest Star')
ax4.set_ylabel('Number of Pointings')
ax4.set_title('Distribution of Brightest Star per Pointing')
ax4.grid(True, alpha=0.3)

# Annotate mean
mean_brightest = np.mean(brightest_per_pointing)
ax4.axvline(mean_brightest, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {mean_brightest:.2f}')
ax4.legend()

plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {args.output}")

# Print summary statistics
print("\nSummary Statistics:")
print(f"  Total pointings: {len(magnitudes_per_pointing)}")
print(f"  Total stars: {len(all_magnitudes)}")
print(f"  Mean stars per pointing: {len(all_magnitudes) / len(magnitudes_per_pointing):.1f}")
print(f"  Magnitude range: {min(all_magnitudes):.2f} to {max(all_magnitudes):.2f}")
print(f"  Median magnitude: {np.median(all_magnitudes):.2f}")
print(f"\nBrightest Star per Pointing:")
print(f"  Mean brightest: {mean_brightest:.2f}")
print(f"  Brightest overall: {min(brightest_per_pointing):.2f}")
print(f"  Dimmest 'brightest': {max(brightest_per_pointing):.2f}")

# Print probability of finding bright stars
print("\nProbability of finding at least one star brighter than:")
for mag_threshold in [6, 8, 10, 12, 14]:
    pointings_with = sum(
        1 for mags in magnitudes_per_pointing
        if any(m <= mag_threshold for m in mags)
    )
    prob = pointings_with / len(magnitudes_per_pointing)
    print(f"  Mag {mag_threshold:2d}: {prob:6.1%}")

if not args.no_show:
    plt.show()

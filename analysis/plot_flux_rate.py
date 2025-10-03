#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='Plot flux rate vs magnitude from tracking data')
parser.add_argument('--input', '-i', type=str, required=True, help='Path to tracking.csv file')
parser.add_argument('--output', '-o', type=str, default='plots/flux_rate_vs_magnitude.png', help='Output plot path')
parser.add_argument('--no-show', action='store_true', help='Do not display plot interactively')
args = parser.parse_args()

# Ensure output directory exists
Path(args.output).parent.mkdir(parents=True, exist_ok=True)

print(f"Loading tracking data from: {args.input}")
df = pd.read_csv(args.input)

print(f"Loaded {len(df)} tracking data points")

# Calculate flux rate (flux per unit time)
df['flux_rate'] = df['flux'] / df['exposure_ms']

# Get unique exposure times
exposure_times = sorted(df['exposure_ms'].unique())
print(f"Found {len(exposure_times)} exposure times: {exposure_times}")

# HWK4123 sensor saturation parameters (from SensorConfig)
bit_depth = 12
dn_per_electron = 7.42
max_well_depth_e = 7500.0

# Calculate saturating_reading the same way as SensorConfig
well_saturation_dn = max_well_depth_e * dn_per_electron
adc_max_dn = (2**bit_depth - 1)
saturating_reading_dn = min(well_saturation_dn, adc_max_dn)

# Convert back to electrons (flux is in electrons)
saturating_flux_e = saturating_reading_dn / dn_per_electron

print(f"\nHWK4123 Saturation:")
print(f"  ADC max: {adc_max_dn} DN")
print(f"  Well saturation: {well_saturation_dn:.1f} DN")
print(f"  Saturating reading: {saturating_reading_dn} DN")
print(f"  Saturating flux: {saturating_flux_e:.1f} electrons\n")

# Create figure with 2 rows: scatter plots on top, histograms on bottom
num_exposures = len(exposure_times)
fig, axes = plt.subplots(2, num_exposures, figsize=(5 * num_exposures, 10))

# If only one exposure, make axes 2D
if num_exposures == 1:
    axes = axes.reshape(2, 1)

# Calculate global axis limits
mag_min = df['magnitude'].min()
mag_max = df['magnitude'].max()
flux_min = df['flux'].min()
flux_max = df['flux'].max()

# Plot each exposure time separately
for idx, exp_time in enumerate(exposure_times):
    ax_scatter = axes[0, idx]
    ax_hist = axes[1, idx]
    exposure_data = df[df['exposure_ms'] == exp_time]

    # Calculate mean magnitude for this exposure
    mean_mag = exposure_data['magnitude'].mean()

    # Top row: Plot flux vs magnitude
    ax_scatter.scatter(
        exposure_data['magnitude'],
        exposure_data['flux'],
        alpha=0.2,
        s=20,
        color='blue',
        edgecolors='none'
    )

    # Add saturation line in DN
    # Constants are a ballpark of the upper/lower multiples we expect on the
    # Disk spread in centered and cornered configurations
    lower, upper = 3.75, 5.78
    ax_scatter.axhline(saturating_reading_dn*lower, color='red', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Saturation ({saturating_reading_dn:.0f} DN) × {lower:.2f}')

    ax_scatter.axhline(saturating_reading_dn*upper, color='green', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Saturation ({saturating_reading_dn:.0f} DN) × {upper:.2f}')

    # Add mean magnitude vertical line
    ax_scatter.axvline(mean_mag, color='orange', linestyle='--', linewidth=2, alpha=0.8)
    ax_scatter.text(mean_mag, ax_scatter.get_ylim()[1] * 0.5, f'μ={mean_mag:.2f}',
                    rotation=90, va='bottom', ha='right', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax_scatter.set_ylabel('Counts (DN)')
    ax_scatter.set_title(f'Exposure: {exp_time} ms')
    ax_scatter.legend(loc='best')
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_yscale('log')
    ax_scatter.set_xlim(mag_min, mag_max)
    ax_scatter.set_ylim(flux_min, flux_max)

    # Bottom row: Histogram of magnitudes (as percentages)
    weights = np.ones_like(exposure_data['magnitude']) / len(exposure_data['magnitude']) * 100
    ax_hist.hist(
        exposure_data['magnitude'],
        bins=50,
        range=(mag_min, mag_max),
        edgecolor='black',
        alpha=0.7,
        color='blue',
        weights=weights
    )

    # Add mean magnitude vertical line
    ax_hist.axvline(mean_mag, color='orange', linestyle='--', linewidth=2, alpha=0.8)
    ax_hist.text(mean_mag, ax_hist.get_ylim()[1] * 0.95, f'μ={mean_mag:.2f}',
                 rotation=90, va='top', ha='right', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax_hist.set_xlabel('Stellar Magnitude (Gaia)')
    ax_hist.set_ylabel('Percentage (%)')
    ax_hist.set_xlim(mag_min, mag_max)
    ax_hist.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"Saved flux plot to: {args.output}")

if not args.no_show:
    plt.show()

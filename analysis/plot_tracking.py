#!/usr/bin/env python3
"""Plot tracking data from cam_track CSV output."""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot tracking data from cam_track CSV output')
parser.add_argument('csv_file', help='Path to tracking CSV file')
parser.add_argument('--output', '-o', default='tracking_analysis.png', help='Output plot filename (default: tracking_analysis.png)')
args = parser.parse_args()

# Read CSV data
with open(args.csv_file, 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Extract data
timestamps = np.array([float(row['timestamp']) for row in data])
x_positions = np.array([float(row['x']) for row in data])
y_positions = np.array([float(row['y']) for row in data])

# Normalize timestamps to start at 0
timestamps_rel = timestamps - timestamps[0]

# Calculate timestamp deltas (jitter)
dt = np.diff(timestamps)

# Fit sinusoids to x and y data
def sinusoid(t, amplitude, frequency, phase, offset):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

# Initial guesses based on data
x_amp_guess = (x_positions.max() - x_positions.min()) / 2
x_offset_guess = x_positions.mean()
y_amp_guess = (y_positions.max() - y_positions.min()) / 2
y_offset_guess = y_positions.mean()
freq_guess = 1.0 / 10.0  # 10 second period from calibrate.rs

# Fit x position
x_params, _ = curve_fit(
    sinusoid,
    timestamps_rel,
    x_positions,
    p0=[x_amp_guess, freq_guess, 0, x_offset_guess]
)
x_fit = sinusoid(timestamps_rel, *x_params)
x_residuals = x_positions - x_fit

# Fit y position
y_params, _ = curve_fit(
    sinusoid,
    timestamps_rel,
    y_positions,
    p0=[y_amp_guess, freq_guess, 0, y_offset_guess]
)
y_fit = sinusoid(timestamps_rel, *y_params)
y_residuals = y_positions - y_fit

print(f"\nSinusoidal Fit Parameters:")
print(f"X: amplitude={x_params[0]:.3f} px, freq={x_params[1]:.4f} Hz, phase={x_params[2]:.3f} rad, offset={x_params[3]:.3f} px")
print(f"Y: amplitude={y_params[0]:.3f} px, freq={y_params[1]:.4f} Hz, phase={y_params[2]:.3f} rad, offset={y_params[3]:.3f} px")
print(f"X residual RMS: {np.sqrt(np.mean(x_residuals**2)):.4f} px")
print(f"Y residual RMS: {np.sqrt(np.mean(y_residuals**2)):.4f} px")

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 14))

# Plot 1: X position over time with fit
axes[0, 0].plot(timestamps_rel, x_positions, 'b.', markersize=4, label='Data')
axes[0, 0].plot(timestamps_rel, x_fit, 'r-', linewidth=0.5, alpha=0.9, label=f'Fit (A={x_params[0]:.2f}px, f={x_params[1]:.4f}Hz)')
axes[0, 0].axhline(y=x_params[3], color='k', linestyle=':', linewidth=1, alpha=0.7, label=f'Mean: {x_params[3]:.2f}')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('X Position (pixels)')
axes[0, 0].set_title(f'X Position vs Time (Δ={x_positions.max()-x_positions.min():.2f} px)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(loc='best', fontsize=8)

# Plot 2: Y position over time with fit
axes[0, 1].plot(timestamps_rel, y_positions, 'r.', markersize=4, label='Data')
axes[0, 1].plot(timestamps_rel, y_fit, 'b-', linewidth=0.5, alpha=0.9, label=f'Fit (A={y_params[0]:.2f}px, f={y_params[1]:.4f}Hz)')
axes[0, 1].axhline(y=y_params[3], color='k', linestyle=':', linewidth=1, alpha=0.7, label=f'Mean: {y_params[3]:.2f}')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Y Position (pixels)')
axes[0, 1].set_title(f'Y Position vs Time (Δ={y_positions.max()-y_positions.min():.2f} px)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(loc='best', fontsize=8)

# Plot 3: X residuals (red to match X fit line)
axes[1, 0].plot(timestamps_rel, x_residuals, 'r.', markersize=2, alpha=0.5)
axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=1)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('X Residual (pixels)')
axes[1, 0].set_title(f'X Residuals (RMS={np.sqrt(np.mean(x_residuals**2)):.4f} px)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Y residuals (blue to match Y fit line)
axes[1, 1].plot(timestamps_rel, y_residuals, 'b.', markersize=2, alpha=0.5)
axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=1)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Y Residual (pixels)')
axes[1, 1].set_title(f'Y Residuals (RMS={np.sqrt(np.mean(y_residuals**2)):.4f} px)')
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: X vs Y (trajectory) with nominal circle
axes[2, 0].scatter(x_positions, y_positions, c=timestamps_rel, cmap='viridis', s=2, alpha=0.4)
# Draw nominal circle from fit parameters
theta = np.linspace(0, 2*np.pi, 100)
x_circle = x_params[0] * np.cos(theta) + x_params[3]
y_circle = y_params[0] * np.sin(theta) + y_params[3]
axes[2, 0].plot(x_circle, y_circle, 'r--', linewidth=0.5, alpha=0.9, label='Nominal circle')
axes[2, 0].plot(x_params[3], y_params[3], 'r+', markersize=15, markeredgewidth=3, label='Center')
axes[2, 0].set_xlabel('X Position (pixels)')
axes[2, 0].set_ylabel('Y Position (pixels)')
axes[2, 0].set_title('Tracking Trajectory (colored by time)')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_aspect('equal', adjustable='box')
axes[2, 0].legend(loc='best', fontsize=8)
cbar = plt.colorbar(axes[2, 0].collections[0], ax=axes[2, 0])
cbar.set_label('Time (s)')

# Plot 6: Timestamp jitter histogram
axes[2, 1].hist(dt * 1000, bins=50, alpha=0.7, edgecolor='black')
axes[2, 1].set_xlabel('Time between updates (ms)')
axes[2, 1].set_ylabel('Count')
axes[2, 1].set_title(f'Timestamp Jitter\nMean: {np.mean(dt)*1000:.2f}ms, Std: {np.std(dt)*1000:.2f}ms')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(args.output, dpi=150)
print(f"Saved {args.output}")
plt.show()

# Print statistics
print(f"\nTracking Statistics:")
print(f"  Duration: {timestamps_rel[-1]:.2f} seconds")
print(f"  Updates: {len(data)}")
print(f"  Update rate: {len(data)/timestamps_rel[-1]:.2f} Hz")
print(f"\nPosition Statistics:")
print(f"  X range: {x_positions.min():.4f} to {x_positions.max():.4f} (Δ={x_positions.max()-x_positions.min():.4f})")
print(f"  Y range: {y_positions.min():.4f} to {y_positions.max():.4f} (Δ={y_positions.max()-y_positions.min():.4f})")
print(f"  X std dev: {np.std(x_positions):.4f} pixels")
print(f"  Y std dev: {np.std(y_positions):.4f} pixels")
print(f"\nTiming Statistics:")
print(f"  Mean update interval: {np.mean(dt)*1000:.2f} ms")
print(f"  Std dev: {np.std(dt)*1000:.2f} ms")
print(f"  Min interval: {np.min(dt)*1000:.2f} ms")
print(f"  Max interval: {np.max(dt)*1000:.2f} ms")

#!/usr/bin/env python3
"""
Plot focal plane trajectory data (t, x, y) in radians
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

if len(sys.argv) < 2:
    print("Usage: python3 plot_trajectory.py <trajectory_file.txt> [--output output.png]")
    sys.exit(1)

input_file = sys.argv[1]
output_file = None
if '--output' in sys.argv:
    output_idx = sys.argv.index('--output')
    if output_idx + 1 < len(sys.argv):
        output_file = sys.argv[output_idx + 1]

data = np.loadtxt(input_file)
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]

x_mas = x * 206265000
y_mas = y * 206265000

FOCAL_LENGTH_MM = 5987.0
MM_TO_MAS = 206265000 / FOCAL_LENGTH_MM

SENSOR_WIDTH_MM = 9568 * 3.76 / 1000
SENSOR_HEIGHT_MM = 6380 * 3.76 / 1000

sensor_positions = [
    (20.0, 77.75),
    (-20.0, 77.75),
    (58.75, 73.75),
    (-58.75, 73.75),
]

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1],
                       hspace=0.05, wspace=0.05)

ax_main = fig.add_subplot(gs[0, 0])
ax_main.scatter(x_mas, y_mas, c='blue', s=1, alpha=0.4, edgecolors='none')

circle = plt.Circle((0, 0), 18000, color='red', fill=False, linewidth=2, linestyle='--', label='18 arcsec')
ax_main.add_patch(circle)

for i, (x_mm, y_mm) in enumerate(sensor_positions):
    x_center_mas = x_mm * MM_TO_MAS
    y_center_mas = y_mm * MM_TO_MAS
    width_mas = SENSOR_WIDTH_MM * MM_TO_MAS
    height_mas = SENSOR_HEIGHT_MM * MM_TO_MAS

    rect = Rectangle(
        (x_center_mas - width_mas/2, y_center_mas - height_mas/2),
        width_mas, height_mas,
        linewidth=1.5, edgecolor='orange', facecolor='none',
        label='IMX455 sensors' if i == 0 else ''
    )
    ax_main.add_patch(rect)

ax_main.set_ylabel('Y (milliarcseconds)', fontsize=12)
ax_main.set_title('Focal Plane Trajectory - SPENCER Array Plan (JBT .5m)', fontsize=14, fontweight='bold')
ax_main.grid(True, alpha=0.3)
ax_main.set_aspect('equal', adjustable='box')
ax_main.legend()
ax_main.tick_params(labelbottom=False)

ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
ax_right.scatter(t, y_mas, c='green', s=0.5, alpha=0.4, edgecolors='none')
ax_right.set_xlabel('Time (s)', fontsize=10)
ax_right.set_title('Y vs Time', fontsize=12)
ax_right.grid(True, alpha=0.3)
ax_right.tick_params(labelleft=False)
ax_right.invert_xaxis()

ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)
ax_bottom.scatter(x_mas, t, c='red', s=0.5, alpha=0.4, edgecolors='none')
ax_bottom.set_xlabel('X (milliarcseconds)', fontsize=12)
ax_bottom.set_ylabel('Time (s)', fontsize=10)
ax_bottom.set_title('X vs Time', fontsize=12)
ax_bottom.grid(True, alpha=0.3)
ax_bottom.invert_yaxis()

r = np.sqrt(x**2 + y**2) * 206265000
ax_corner = fig.add_subplot(gs[1, 1])
ax_corner.scatter(t, r, c='purple', s=0.5, alpha=0.4, edgecolors='none')
ax_corner.set_xlabel('Time (s)', fontsize=10)
ax_corner.set_ylabel('Radius (mas)', fontsize=10)
ax_corner.set_title('Radial Distance', fontsize=12)
ax_corner.grid(True, alpha=0.3)

if output_file:
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
else:
    plt.show()

print(f"Data points: {len(t)}")
print(f"Time range: {t[0]:.3f} to {t[-1]:.3f} seconds")
print(f"X range: {x_mas.min():.3f} to {x_mas.max():.3f} milliarcseconds")
print(f"Y range: {y_mas.min():.3f} to {y_mas.max():.3f} milliarcseconds")
print(f"Max radial distance: {r.max():.3f} milliarcseconds")

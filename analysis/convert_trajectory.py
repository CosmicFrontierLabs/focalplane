#!/usr/bin/env python3
"""Convert focal plane trajectory from radians to pixels."""

import sys

# IMX455 pixel size: 3.76 micrometers
PIXEL_SIZE_M = 3.76e-6

# Focal length: 6 meters
FOCAL_LENGTH_M = 6.0

# Plate scale: radians per pixel
PLATE_SCALE_RAD_PER_PIX = PIXEL_SIZE_M / FOCAL_LENGTH_M

print(f"Plate scale: {PLATE_SCALE_RAD_PER_PIX:.6e} radians/pixel")
print(f"Inverse: {1/PLATE_SCALE_RAD_PER_PIX:.2f} pixels/radian")

input_file = "focal_plane_trajectory.txt"
output_file = "trajectory_from_joel.csv"

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line_num, line in enumerate(f_in, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) < 3:
            print(f"Warning: Line {line_num} has {len(parts)} fields, expected 3")
            continue

        time_s = float(parts[0])
        x_rad = float(parts[1])
        y_rad = float(parts[2])

        x_pix = x_rad / PLATE_SCALE_RAD_PER_PIX
        y_pix = y_rad / PLATE_SCALE_RAD_PER_PIX

        f_out.write(f"{time_s:.6f},{x_pix:.6f},{y_pix:.6f}\n")

print(f"Converted trajectory written to {output_file}")

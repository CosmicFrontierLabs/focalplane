#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Dark current temperature scaling: doubles every 8°C
def scale_dark_current(base_dc, base_temp, new_temp):
    # Using 8°C doubling temperature
    return base_dc * (2 ** ((new_temp - base_temp) / 8.0))

# Sensor data
sensors = {
    "GSENSE4040BSI": {
        "read_noise": 2.30,  # e-
        "dark_current": 1.2800,  # e-/px/s
    },
    "GSENSE6510BSI": {
        "read_noise": 0.70,  # e-
        "dark_current": 0.4757,  # e-/px/s
    },
    "HWK4123": {
        "read_noise": 0.27,  # e-
        "dark_current": 0.1000,  # e-/px/s
    },
    "IMX455": {
        "read_noise": 2.67,  # e-
        "dark_current": 0.0113,  # e-/px/s
    }
}

# Integration times (seconds)
t = np.logspace(-2, 3, 1000)  # 0.01s to 1000s

# Temperatures to plot
temperatures = [-20, 0, 20]

# Create 3 vertical subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)

for idx, temp in enumerate(temperatures):
    ax = axes[idx]
    
    # Plot total noise for each sensor
    for name, params in sensors.items():
        read_noise = params["read_noise"]
        base_dark_current = params["dark_current"]
        
        # Scale dark current for temperature
        dark_current = scale_dark_current(base_dark_current, 0.0, temp)
        
        # Total noise = sqrt(read_noise^2 + dark_current * t)
        total_noise = np.sqrt(read_noise**2 + dark_current * t)
        
        ax.loglog(t, total_noise, linewidth=2, 
                  label=f"{name} (DC={dark_current:.4f})")
    
    # Add grid and labels
    ax.grid(True, which="both", alpha=0.3)
    if idx == 2:  # Bottom plot
        ax.set_xlabel("Integration Time (seconds)", fontsize=12)
    ax.set_ylabel("Total Noise (e⁻)", fontsize=12)
    ax.set_title(f"Temperature: {temp}°C", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    
    # Add reference lines with labels
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=60, color='gray', linestyle='--', alpha=0.5)
    
    # Add reference line labels on rightmost edge
    if idx == 0:  # Only on top plot
        ax.text(1000, 1.1, '1 e⁻', ha='right', va='bottom', fontsize=8, color='gray')
        ax.text(1000, 11, '10 e⁻', ha='right', va='bottom', fontsize=8, color='gray')
        ax.text(1.1, 0.1, '1s', ha='left', va='bottom', fontsize=8, color='gray')
        ax.text(66, 0.1, '1min', ha='left', va='bottom', fontsize=8, color='gray')
    
    # Show crossover points with labels
    # Define colors for consistency
    sensor_colors = {"GSENSE4040BSI": "C0", "GSENSE6510BSI": "C1", 
                     "HWK4123": "C2", "IMX455": "C3"}
    
    for name, params in sensors.items():
        read_noise = params["read_noise"]
        base_dark_current = params["dark_current"]
        dark_current = scale_dark_current(base_dark_current, 0.0, temp)
        t_cross = read_noise**2 / dark_current
        if 0.01 < t_cross < 1000:
            color = sensor_colors[name]
            ax.axvline(x=t_cross, linestyle=':', alpha=0.5, color=color)
            # Add label at bottom of plot
            ax.text(t_cross, ax.get_ylim()[0]*2, name[:6], 
                    rotation=45, ha='left', va='bottom', fontsize=8, color=color)
    
    # Add explanatory text for first subplot only
    if idx == 0:
        ax.text(0.02, 0.95, 
                'Dotted lines: Read/Dark crossover\n' +
                'Left of line: Read noise limited\n' + 
                'Right of line: Dark current limited',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')

# Overall title
fig.suptitle('Total Integrated Noise vs Integration Time\n' + 
             r'$\sigma_{total} = \sqrt{\sigma_{read}^2 + \sigma_{dark}^2 \cdot t}$', 
             fontsize=16)

plt.tight_layout()
output_file = "noise_lines_comparison.png"
plt.savefig(output_file, dpi=150)
print(f"Plot saved to {output_file}")
plt.show()
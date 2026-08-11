#!/usr/bin/env python3
"""
Plot centroid accuracy vs magnitude for all sensors from single detection matrix results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

def parse_sensor_data(csv_file_path):
    """Parse the CSV file and extract centroid accuracy data for each sensor."""
    
    sensors = {}
    current_sensor = None
    reading_error_data = False
    reading_pixel_data = False
    reading_detection_data = False
    
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:  # Skip empty rows
                continue
                
            # Check for sensor name
            if row[0].startswith('SENSOR:'):
                current_sensor = row[0].split(':')[1].strip()
                sensors[current_sensor] = {'magnitudes': [], 'errors': [], 'pixel_errors': [], 'detection_rates': []}
                reading_error_data = False
                reading_pixel_data = False
                reading_detection_data = False
                continue
            
            # Check for different data sections
            if row[0] == 'Detection Rate Matrix (%)':
                reading_detection_data = True
                reading_error_data = False
                reading_pixel_data = False
                continue
            
            if row[0] == 'Mean Position Error Matrix (milliarcseconds)':
                reading_error_data = True
                reading_pixel_data = False
                reading_detection_data = False
                continue
            
            if row[0] == 'RMS Position Error Matrix (pixels)':
                reading_error_data = False
                reading_pixel_data = True
                reading_detection_data = False
                continue
            
            # Reset reading flags for other sections
            if row[0].startswith('Spurious'):
                reading_error_data = False
                reading_pixel_data = False
                reading_detection_data = False
                continue
            
            # Skip header row for all data sections
            if (reading_error_data or reading_pixel_data or reading_detection_data) and row[0] == 'Q_Value':
                # Extract magnitude values from header (only once)
                if not sensors[current_sensor]['magnitudes']:
                    mags = [float(x) for x in row[2:] if x]  # Skip Q_Value and Exposure_ms columns
                    sensors[current_sensor]['magnitudes'] = mags
                continue
            
            # Read actual data
            if (reading_error_data or reading_pixel_data or reading_detection_data) and current_sensor and row[0] == '1.68':
                values = []
                # Only process up to the number of magnitudes we have
                num_mags = len(sensors[current_sensor]['magnitudes'])
                for i, val in enumerate(row[2:]):  # Skip Q_Value and Exposure_ms columns
                    if i >= num_mags:  # Don't go beyond the number of magnitudes
                        break
                    if val and val.strip():  # Not empty and not just whitespace
                        try:
                            values.append(float(val))
                        except ValueError:
                            values.append(np.nan)
                    else:
                        values.append(np.nan)
                
                if reading_error_data:
                    sensors[current_sensor]['errors'] = values
                elif reading_pixel_data:
                    sensors[current_sensor]['pixel_errors'] = values
                elif reading_detection_data:
                    # Convert detection rates from percentage to fraction
                    sensors[current_sensor]['detection_rates'] = [v/100.0 for v in values]
                
                reading_error_data = False
                reading_pixel_data = False
                reading_detection_data = False
                continue
    
    return sensors

def find_90_percent_threshold(magnitudes, detection_rates):
    """Find the first magnitude where detection drops below 90%."""
    mags = np.array(magnitudes)
    rates = np.array(detection_rates)
    
    # Remove NaN values
    valid_mask = ~np.isnan(rates)
    valid_mags = mags[valid_mask]
    valid_rates = rates[valid_mask]
    
    # Find first point where detection drops below 90%
    below_90_mask = valid_rates < 0.9
    if np.any(below_90_mask):
        first_below_90_idx = np.where(below_90_mask)[0][0]
        return valid_mags[first_below_90_idx]
    else:
        return None  # Never drops below 90%

def plot_centroid_accuracy(csv_file_path, output_path=None):
    """Create a plot showing centroid accuracy vs magnitude for all sensors."""
    
    # Parse the data
    sensors = parse_sensor_data(csv_file_path)
    
    # Create the plot with higher DPI for publication quality
    plt.figure(figsize=(14, 10))
    
    # Color scheme for sensors - more distinct colors
    colors = {
        'GSENSE4040BSI': '#1f77b4',
        'GSENSE6510BSI': '#ff7f0e', 
        'HWK4123': '#2ca02c',
        'IMX455': '#d62728'
    }
    
    # Line styles for better distinction
    line_styles = {
        'GSENSE4040BSI': '-',
        'GSENSE6510BSI': '--', 
        'HWK4123': '-.',
        'IMX455': ':'
    }
    
    # Plot each sensor
    for sensor_name, data in sensors.items():
        mags = np.array(data['magnitudes'])
        errors = np.array(data['errors'])
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(errors)
        valid_mags = mags[valid_mask]
        valid_errors = errors[valid_mask]
        
        if len(valid_mags) > 0:
            # Plot with solid lines and no marker skipping
            plt.plot(valid_mags, valid_errors, 
                    color=colors.get(sensor_name, 'black'),
                    label=sensor_name, 
                    linewidth=2.5, 
                    alpha=0.85)
    
    # Formatting
    plt.xlabel('Star Magnitude', fontsize=16)
    plt.ylabel('Centroid Accuracy (milliarcseconds)', fontsize=16)
    plt.title('Centroid Accuracy vs Star Magnitude\n(500ms exposure, 1.68 FWHM sampling, 2000 experiments)', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, loc='upper left')
    
    # Set y-axis to log scale since errors vary widely
    plt.yscale('log')
    
    # Add vertical lines for 90% detection thresholds
    for sensor_name, data in sensors.items():
        threshold_mag = find_90_percent_threshold(data['magnitudes'], data['detection_rates'])
        if threshold_mag is not None:
            plt.axvline(x=threshold_mag, 
                       color=colors.get(sensor_name, 'black'), 
                       linestyle='--', 
                       alpha=0.4, 
                       linewidth=1.5)
    
    # Improve layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        mas_output_path = output_path.replace('.png', '_mas.png')
        plt.savefig(mas_output_path, dpi=300, bbox_inches='tight')
        print(f"Milliarcsecond plot saved to: {mas_output_path}")
    else:
        plt.show()
    
    # Create second plot for pixel errors
    plt.figure(figsize=(14, 10))
    
    # Plot each sensor (pixel errors)
    for sensor_name, data in sensors.items():
        mags = np.array(data['magnitudes'])
        errors = np.array(data['pixel_errors'])
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(errors)
        valid_mags = mags[valid_mask]
        valid_errors = errors[valid_mask]
        
        if len(valid_mags) > 0:
            # Plot with solid lines and small markers to show all data points
            plt.plot(valid_mags, valid_errors, '-o',
                    color=colors.get(sensor_name, 'black'),
                    label=sensor_name, 
                    linewidth=2.5,
                    markersize=1,  # Very small markers to show density
                    alpha=0.85)
    
    # Formatting for pixel plot
    plt.xlabel('Star Magnitude', fontsize=16)
    plt.ylabel('Centroid Accuracy (pixels)', fontsize=16)
    plt.title('Centroid Accuracy vs Star Magnitude\n(500ms exposure, 1.68 FWHM sampling, 2000 experiments)', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, loc='upper left')
    
    # Set y-axis to log scale for pixels too
    plt.yscale('log')
    
    # Add vertical lines for 90% detection thresholds on pixel plot too
    for sensor_name, data in sensors.items():
        threshold_mag = find_90_percent_threshold(data['magnitudes'], data['detection_rates'])
        if threshold_mag is not None:
            plt.axvline(x=threshold_mag, 
                       color=colors.get(sensor_name, 'black'), 
                       linestyle='--', 
                       alpha=0.4, 
                       linewidth=1.5)
    
    # Improve layout
    plt.tight_layout()
    
    # Save pixel plot
    if output_path:
        pixel_output_path = output_path.replace('.png', '_pixels.png')
        plt.savefig(pixel_output_path, dpi=300, bbox_inches='tight')
        print(f"Pixel plot saved to: {pixel_output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Input and output paths
    csv_path = "../sensor_floor_results.csv"
    output_path = "../plots/centroid_accuracy_vs_magnitude.png"
    
    # Create the plot
    plot_centroid_accuracy(csv_path, output_path)
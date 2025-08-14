#!/usr/bin/env python3
"""
Unified sensor comparison analysis tool
Combines mean pointing error plots and win percentage analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.lines import Line2D


# ============================================================================
# Shared Functions
# ============================================================================

def get_f_number_color(f_number, cmap):
    """Get consistent color for f-number across all plots.
    
    Maps f-numbers to colors based on a fixed range of f/8 to f/18,
    ensuring consistent colors across different datasets.
    """
    # Fixed range for consistent mapping
    f_min = 8.0
    f_max = 18.0
    
    # Clamp f_number to range
    f_clamped = max(f_min, min(f_max, f_number))
    
    # Map to 0-1 range
    normalized = (f_clamped - f_min) / (f_max - f_min)
    
    return cmap(normalized)


def load_experiment_data(csv_path, aperture_m=0.485, mask_outliers=True, pixel_error_cutoff=2.0):
    """Load experiment CSV data with preprocessing and optional outlier masking"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sensors found: {df['sensor'].unique()}")
    
    # Calculate f-number and round to nearest integer
    df['f_number'] = (df['focal_length_m'] / aperture_m).round()
    print(f"F-numbers found: {sorted(df['f_number'].unique())}")
    
    # Get sensor pixel sizes
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Calculate plate scale for each row based on sensor
    for sensor_name, pixel_size_um in sensor_pixel_sizes.items():
        sensor_mask = df['sensor'] == sensor_name
        focal_length_m = df.loc[sensor_mask, 'f_number'] * aperture_m
        plate_scale_arcsec_per_mm = 206265.0 / (focal_length_m * 1000.0)
        plate_scale_arcsec_per_pixel = plate_scale_arcsec_per_mm * (pixel_size_um / 1000.0)
        df.loc[sensor_mask, 'plate_scale_mas_per_pixel'] = plate_scale_arcsec_per_pixel * 1000.0
    
    # Calculate pointing error in milliarcseconds
    df['error_mas'] = df['pixel_error'] * df['plate_scale_mas_per_pixel']
    
    if mask_outliers:
        # Mask out experiments with pixel error > cutoff AND 2 or fewer stars
        # These are likely mismatched detections with insufficient stars for good matching
        mask = (df['pixel_error'] > pixel_error_cutoff) & (df['star_count'] <= 2)
        num_masked = mask.sum()
        if num_masked > 0:
            print(f"Masking {num_masked} rows with pixel_error > {pixel_error_cutoff} pixels and star_count <= 2 (likely mismatched detections)")
            # Set star_count to 0 and other values to NaN for these mismatches
            df.loc[mask, 'star_count'] = 0
            df.loc[mask, ['brightest_mag', 'faintest_mag', 'pixel_error', 'error_mas']] = np.nan
    
    return df


def get_sensor_pixel_sizes():
    """Return sensor pixel sizes in micrometers"""
    return {
        'IMX455': 3.76,
        'HWK4123': 4.6
    }


def calculate_plate_scale(f_number, aperture_m, pixel_size_um):
    """Calculate plate scale in milliarcseconds per pixel"""
    focal_length_m = f_number * aperture_m
    plate_scale_arcsec_per_mm = 206265.0 / (focal_length_m * 1000.0)  # 206265 arcsec per radian
    plate_scale_arcsec_per_pixel = plate_scale_arcsec_per_mm * (pixel_size_um / 1000.0)
    # TODO(meawoppl) pickup here with simplified version of the above:
    # pixel_size_um * 1e6 * 206265.0 * 1000.0 / focal_length_m
    return plate_scale_arcsec_per_pixel * 1000.0  # Convert to milliarcseconds

# TODO(meawoppl): next analyses:
# faster focal lengths, where does precision turn around
# best star vs all stars in scene analysis

# ============================================================================
# Mean Star Count Analysis
# ============================================================================

def calculate_star_count_statistics(df, sensor_name, f_number):
    """Calculate mean and std dev of star count for given sensor/f-number"""
    # Filter for specific sensor and f-number
    mask = (df['sensor'] == sensor_name) & (df['f_number'] == f_number)
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return pd.DataFrame()
    
    # Group by exposure time and calculate statistics, ignoring NaN values
    grouped = subset.groupby('exposure_ms')['star_count'].agg([
        ('mean_star_count', lambda x: x.mean()),
        ('std_star_count', lambda x: x.std()),
        ('count', lambda x: x.notna().sum())  # Count non-NaN values
    ]).reset_index()
    
    # Calculate standard error of the mean
    grouped['sem_star_count'] = grouped['std_star_count'] / np.sqrt(grouped['count'])
    
    return grouped


# ============================================================================
# Brightest Star Magnitude Analysis
# ============================================================================

def calculate_brightest_magnitude_statistics(df, sensor_name, f_number):
    """Calculate mean and std dev of brightest star magnitude for given sensor/f-number"""
    # Filter for specific sensor and f-number
    mask = (df['sensor'] == sensor_name) & (df['f_number'] == f_number)
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return pd.DataFrame()
    
    # Group by exposure time and calculate statistics, ignoring NaN values
    grouped = subset.groupby('exposure_ms')['brightest_mag'].agg([
        ('mean_brightest_mag', lambda x: x.mean()),
        ('std_brightest_mag', lambda x: x.std()),
        ('count', lambda x: x.notna().sum())  # Count non-NaN values
    ]).reset_index()
    
    # Calculate standard error of the mean
    grouped['sem_brightest_mag'] = grouped['std_brightest_mag'] / np.sqrt(grouped['count'])
    
    return grouped


# ============================================================================
# High Accuracy Percentage Analysis
# ============================================================================

def calculate_high_accuracy_statistics(df, sensor_name, f_number):
    """Calculate percentage of experiments with pixel error < 0.1"""
    # Filter for specific sensor and f-number
    mask = (df['sensor'] == sensor_name) & (df['f_number'] == f_number)
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return pd.DataFrame()
    
    # Group by exposure time and calculate high accuracy percentage
    grouped = subset.groupby('exposure_ms').agg({
        'pixel_error': [
            ('high_accuracy_pct', lambda x: ((x < 0.1) & x.notna()).sum() / len(x) * 100),
            ('total_experiments', 'count'),
            ('experiments_high_accuracy', lambda x: ((x < 0.1) & x.notna()).sum())
        ]
    })
    
    # Flatten column names
    grouped.columns = ['high_accuracy_pct', 'total_experiments', 'experiments_high_accuracy']
    grouped = grouped.reset_index()
    
    return grouped


# ============================================================================
# Field Closure Percentage Analysis
# ============================================================================

def calculate_field_closure_statistics(df, sensor_name, f_number):
    """Calculate percentage of experiments that detected at least 1 star"""
    # Filter for specific sensor and f-number
    mask = (df['sensor'] == sensor_name) & (df['f_number'] == f_number)
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return pd.DataFrame()
    
    # Group by exposure time and calculate field closure percentage
    grouped = subset.groupby('exposure_ms').agg({
        'star_count': [
            ('field_closure_pct', lambda x: (x > 0).sum() / len(x) * 100),
            ('total_experiments', 'count'),
            ('experiments_with_stars', lambda x: (x > 0).sum())
        ]
    })
    
    # Flatten column names
    grouped.columns = ['field_closure_pct', 'total_experiments', 'experiments_with_stars']
    grouped = grouped.reset_index()
    
    return grouped


# ============================================================================
# Faintest Star Magnitude Analysis
# ============================================================================

def calculate_faintest_magnitude_statistics(df, sensor_name, f_number):
    """Calculate mean and std dev of faintest star magnitude for given sensor/f-number"""
    # Filter for specific sensor and f-number
    mask = (df['sensor'] == sensor_name) & (df['f_number'] == f_number)
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return pd.DataFrame()
    
    # Group by exposure time and calculate statistics, ignoring NaN values
    grouped = subset.groupby('exposure_ms')['faintest_mag'].agg([
        ('mean_faintest_mag', lambda x: x.mean()),
        ('std_faintest_mag', lambda x: x.std()),
        ('count', lambda x: x.notna().sum())  # Count non-NaN values
    ]).reset_index()
    
    # Calculate standard error of the mean
    grouped['sem_faintest_mag'] = grouped['std_faintest_mag'] / np.sqrt(grouped['count'])
    
    return grouped


# ============================================================================
# Mean Pointing Error Analysis
# ============================================================================

def calculate_pointing_statistics(df, sensor_name, f_number):
    """Calculate mean and std dev of pointing error for given sensor/f-number"""
    # Filter for specific sensor and f-number
    mask = (df['sensor'] == sensor_name) & (df['f_number'] == f_number)
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return pd.DataFrame()
    
    # Group by exposure time and calculate statistics, ignoring NaN values
    grouped = subset.groupby('exposure_ms')['error_mas'].agg([
        ('mean_error_mas', lambda x: x.mean()),
        ('std_error_mas', lambda x: x.std()),
        ('count', lambda x: x.notna().sum())  # Count non-NaN values
    ]).reset_index()
    
    # Calculate standard error of the mean
    grouped['sem_error_mas'] = grouped['std_error_mas'] / np.sqrt(grouped['count'])
    
    return grouped


def plot_mean_pointing_error(df, output_path=None, show_error_bars=False):
    """Create separate analysis plots for each sensor showing mean pointing error"""
    # Get sensor pixel sizes
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use a consistent colormap for both sensors
    cmap = cm.viridis  # Natural color gradient
    
    # First pass: collect all data to find global y-axis limits
    max_y_value = 0
    all_plot_data = {}
    
    for sensor_name in sensors:
        pixel_size_um = sensor_pixel_sizes.get(sensor_name)
        if pixel_size_um is None:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        
        sensor_data = []
        for f_num in f_numbers:
            stats_data = calculate_pointing_statistics(df, sensor_name, f_num)
            if len(stats_data) > 0:
                sensor_data.append((f_num, stats_data))
                # Update max y value
                if show_error_bars:
                    max_val = (stats_data['mean_error_mas'] + stats_data['std_error_mas']).max()
                else:
                    max_val = stats_data['mean_error_mas'].max()
                max_y_value = max(max_y_value, max_val)
        
        all_plot_data[sensor_name] = sensor_data
    
    # Add 10% padding to max value
    y_limit = max_y_value * 1.1
    
    # Second pass: plot with consistent y-axis
    for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
        # Get pixel size for this sensor
        pixel_size_um = sensor_pixel_sizes.get(sensor_name)
        
        # Store handles for legend
        legend_elements = []
        
        # Plot for each f-number using pre-calculated data
        for i, (f_num, stats_data) in enumerate(all_plot_data[sensor_name]):
            color = get_f_number_color(f_num, cmap)
            
            # Convert exposure from ms to seconds for display
            exposure_s = stats_data['exposure_ms'] / 1000.0
            
            # Plot mean with optional error bars
            if show_error_bars:
                ax.errorbar(exposure_s, stats_data['mean_error_mas'], 
                           yerr=stats_data['std_error_mas'],
                           color=color, linewidth=2, marker='o', markersize=6,
                           capsize=5, capthick=1.5, elinewidth=1.5,
                           label=f"f/{f_num}")
            else:
                ax.plot(exposure_s, stats_data['mean_error_mas'], 
                       color=color, linewidth=2, marker='o', markersize=6,
                       label=f"f/{f_num}")
            
            legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                        label=f"f/{f_num}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Mean Pointing Error (milliarcseconds)', fontsize=12)
        ax.set_title(f'{sensor_name} Pointing Precision Analysis\n' + 
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # Set consistent Y-axis limits
        ax.set_ylim(0, y_limit)
        
        # X-axis log scale if exposure range is large
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    # Overall title
    title = 'Sensor Mean Pointing Error Comparison'
    if show_error_bars:
        title += ' (with ±1σ error bars)'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


def plot_mean_star_count(df, output_path=None, show_error_bars=False):
    """Create separate analysis plots for each sensor showing mean star count"""
    # Get sensor pixel sizes (not used here but keeping for consistency)
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use a consistent colormap for both sensors
    cmap = cm.viridis  # Natural color gradient
    
    # First pass: collect all data to find global y-axis limits
    max_y_value = 0
    all_plot_data = {}
    
    for sensor_name in sensors:
        sensor_data = []
        for f_num in f_numbers:
            stats_data = calculate_star_count_statistics(df, sensor_name, f_num)
            if len(stats_data) > 0:
                sensor_data.append((f_num, stats_data))
                # Update max y value
                if show_error_bars:
                    max_val = (stats_data['mean_star_count'] + stats_data['std_star_count']).max()
                else:
                    max_val = stats_data['mean_star_count'].max()
                max_y_value = max(max_y_value, max_val)
        
        all_plot_data[sensor_name] = sensor_data
    
    # Add 10% padding to max value
    y_limit = max_y_value * 1.1
    
    # Second pass: plot with consistent y-axis
    for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
        # Get pixel size for this sensor
        pixel_size_um = sensor_pixel_sizes.get(sensor_name)
        
        # Store handles for legend
        legend_elements = []
        
        # Plot for each f-number using pre-calculated data
        for i, (f_num, stats_data) in enumerate(all_plot_data[sensor_name]):
            color = get_f_number_color(f_num, cmap)
            
            # Convert exposure from ms to seconds for display
            exposure_s = stats_data['exposure_ms'] / 1000.0
            
            # Plot mean with optional error bars
            if show_error_bars:
                ax.errorbar(exposure_s, stats_data['mean_star_count'], 
                           yerr=stats_data['std_star_count'],
                           color=color, linewidth=2, marker='o', markersize=6,
                           capsize=5, capthick=1.5, elinewidth=1.5,
                           label=f"f/{f_num}")
            else:
                ax.plot(exposure_s, stats_data['mean_star_count'], 
                       color=color, linewidth=2, marker='o', markersize=6,
                       label=f"f/{f_num}")
            
            legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                        label=f"f/{f_num}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Mean Star Count', fontsize=12)
        ax.set_title(f'{sensor_name} Star Detection Performance\n' + 
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # Set consistent Y-axis limits
        ax.set_ylim(0, y_limit)
        
        # X-axis log scale if exposure range is large
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    # Overall title
    title = 'Sensor Mean Star Count Comparison'
    if show_error_bars:
        title += ' (with ±1σ error bars)'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


def plot_faintest_magnitude(df, output_path=None, show_error_bars=False):
    """Create separate analysis plots for each sensor showing faintest star magnitude"""
    # Get sensor pixel sizes (not used here but keeping for consistency)
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use a consistent colormap for both sensors
    cmap = cm.viridis  # Natural color gradient
    
    # First pass: collect all data to find global y-axis limits
    min_y_value = float('inf')
    max_y_value = 0
    all_plot_data = {}
    
    for sensor_name in sensors:
        sensor_data = []
        for f_num in f_numbers:
            stats_data = calculate_faintest_magnitude_statistics(df, sensor_name, f_num)
            if len(stats_data) > 0:
                sensor_data.append((f_num, stats_data))
                # Update y value range (note: fainter stars have higher magnitude)
                if show_error_bars:
                    min_val = (stats_data['mean_faintest_mag'] - stats_data['std_faintest_mag']).min()
                    max_val = (stats_data['mean_faintest_mag'] + stats_data['std_faintest_mag']).max()
                else:
                    min_val = stats_data['mean_faintest_mag'].min()
                    max_val = stats_data['mean_faintest_mag'].max()
                min_y_value = min(min_y_value, min_val)
                max_y_value = max(max_y_value, max_val)
        
        all_plot_data[sensor_name] = sensor_data
    
    # Add padding
    y_range = max_y_value - min_y_value
    y_min = min_y_value - 0.1 * y_range
    y_max = max_y_value + 0.1 * y_range
    
    # Second pass: plot with consistent y-axis
    for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
        # Get pixel size for this sensor
        pixel_size_um = sensor_pixel_sizes.get(sensor_name)
        
        # Store handles for legend
        legend_elements = []
        
        # Plot for each f-number using pre-calculated data
        for i, (f_num, stats_data) in enumerate(all_plot_data[sensor_name]):
            color = get_f_number_color(f_num, cmap)
            
            # Convert exposure from ms to seconds for display
            exposure_s = stats_data['exposure_ms'] / 1000.0
            
            # Plot mean with optional error bars
            if show_error_bars:
                ax.errorbar(exposure_s, stats_data['mean_faintest_mag'], 
                           yerr=stats_data['std_faintest_mag'],
                           color=color, linewidth=2, marker='o', markersize=6,
                           capsize=5, capthick=1.5, elinewidth=1.5,
                           label=f"f/{f_num}")
            else:
                ax.plot(exposure_s, stats_data['mean_faintest_mag'], 
                       color=color, linewidth=2, marker='o', markersize=6,
                       label=f"f/{f_num}")
            
            legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                        label=f"f/{f_num}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Faintest Star Magnitude', fontsize=12)
        ax.set_title(f'{sensor_name} Detection Depth\n' + 
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # Set consistent Y-axis limits
        ax.set_ylim(y_min, y_max)
        
        # Invert y-axis (fainter stars have higher magnitude)
        ax.invert_yaxis()
        
        # X-axis log scale if exposure range is large
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    # Overall title
    title = 'Sensor Faintest Star Detection Comparison'
    if show_error_bars:
        title += ' (with ±1σ error bars)'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


def plot_high_accuracy_percentage(df, output_path=None):
    """Create separate analysis plots for each sensor showing high accuracy percentage"""
    # Get sensor pixel sizes (not used here but keeping for consistency)
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use a consistent colormap for both sensors
    cmap = cm.viridis  # Natural color gradient
    
    # Plot data for each sensor
    for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
        # Get pixel size for this sensor
        pixel_size_um = sensor_pixel_sizes.get(sensor_name)
        
        # Store handles for legend
        legend_elements = []
        
        # Plot for each f-number
        for i, f_num in enumerate(f_numbers):
            color = get_f_number_color(f_num, cmap)
            
            stats_data = calculate_high_accuracy_statistics(df, sensor_name, f_num)
            
            if len(stats_data) > 0:
                # Convert exposure from ms to seconds for display
                exposure_s = stats_data['exposure_ms'] / 1000.0
                
                # Plot high accuracy percentage
                ax.plot(exposure_s, stats_data['high_accuracy_pct'], 
                       color=color, linewidth=2, marker='o', markersize=6,
                       label=f"f/{f_num}")
                
                # Add text annotation for number of experiments at first point
                if len(stats_data) > 0:
                    first_row = stats_data.iloc[0]
                    ax.annotate(f'n={first_row["total_experiments"]}', 
                               xy=(exposure_s.iloc[0], first_row['high_accuracy_pct']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
                
                legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                            label=f"f/{f_num}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('High Accuracy Percentage (%)', fontsize=12)
        ax.set_title(f'{sensor_name} High Accuracy Rate\n' + 
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # Set Y-axis limits 0-100%
        ax.set_ylim(0, 105)
        
        # X-axis log scale if exposure range is large
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    # Overall title
    fig.suptitle('High Accuracy Percentage\n(Percentage of experiments with pixel error < 0.1)', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


def plot_field_closure_percentage(df, output_path=None):
    """Create separate analysis plots for each sensor showing field closure percentage"""
    # Get sensor pixel sizes (not used here but keeping for consistency)
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use a consistent colormap for both sensors
    cmap = cm.viridis  # Natural color gradient
    
    # Plot data for each sensor
    for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
        # Get pixel size for this sensor
        pixel_size_um = sensor_pixel_sizes.get(sensor_name)
        
        # Store handles for legend
        legend_elements = []
        
        # Plot for each f-number
        for i, f_num in enumerate(f_numbers):
            color = get_f_number_color(f_num, cmap)
            
            stats_data = calculate_field_closure_statistics(df, sensor_name, f_num)
            
            if len(stats_data) > 0:
                # Convert exposure from ms to seconds for display
                exposure_s = stats_data['exposure_ms'] / 1000.0
                
                # Plot field closure percentage
                ax.plot(exposure_s, stats_data['field_closure_pct'], 
                       color=color, linewidth=2, marker='o', markersize=6,
                       label=f"f/{f_num}")
                
                # Add text annotation for number of experiments at first point
                if len(stats_data) > 0:
                    first_row = stats_data.iloc[0]
                    ax.annotate(f'n={first_row["total_experiments"]}', 
                               xy=(exposure_s.iloc[0], first_row['field_closure_pct']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
                
                legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                            label=f"f/{f_num}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Field Closure Percentage (%)', fontsize=12)
        ax.set_title(f'{sensor_name} Field Closure Rate\n' + 
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # Set Y-axis limits 0-100%
        ax.set_ylim(0, 105)
        
        # X-axis log scale if exposure range is large
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    # Overall title
    fig.suptitle('Minimal Detection Percentage\n(Percentage of experiments detecting ≥1 star)', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


def plot_brightest_magnitude(df, output_path=None, show_error_bars=False):
    """Create separate analysis plots for each sensor showing brightest star magnitude"""
    # Get sensor pixel sizes (not used here but keeping for consistency)
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use a consistent colormap for both sensors
    cmap = cm.viridis  # Natural color gradient
    
    # First pass: collect all data to find global y-axis limits
    min_y_value = float('inf')
    max_y_value = 0
    all_plot_data = {}
    
    for sensor_name in sensors:
        sensor_data = []
        for f_num in f_numbers:
            stats_data = calculate_brightest_magnitude_statistics(df, sensor_name, f_num)
            if len(stats_data) > 0:
                sensor_data.append((f_num, stats_data))
                # Update y value range
                if show_error_bars:
                    min_val = (stats_data['mean_brightest_mag'] - stats_data['std_brightest_mag']).min()
                    max_val = (stats_data['mean_brightest_mag'] + stats_data['std_brightest_mag']).max()
                else:
                    min_val = stats_data['mean_brightest_mag'].min()
                    max_val = stats_data['mean_brightest_mag'].max()
                min_y_value = min(min_y_value, min_val)
                max_y_value = max(max_y_value, max_val)
        
        all_plot_data[sensor_name] = sensor_data
    
    # Add padding
    y_range = max_y_value - min_y_value
    y_min = min_y_value - 0.1 * y_range
    y_max = max_y_value + 0.1 * y_range
    
    # Second pass: plot with consistent y-axis
    for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
        # Get pixel size for this sensor
        pixel_size_um = sensor_pixel_sizes.get(sensor_name)
        
        # Store handles for legend
        legend_elements = []
        
        # Plot for each f-number using pre-calculated data
        for i, (f_num, stats_data) in enumerate(all_plot_data[sensor_name]):
            color = get_f_number_color(f_num, cmap)
            
            # Convert exposure from ms to seconds for display
            exposure_s = stats_data['exposure_ms'] / 1000.0
            
            # Plot mean with optional error bars
            if show_error_bars:
                ax.errorbar(exposure_s, stats_data['mean_brightest_mag'], 
                           yerr=stats_data['std_brightest_mag'],
                           color=color, linewidth=2, marker='o', markersize=6,
                           capsize=5, capthick=1.5, elinewidth=1.5,
                           label=f"f/{f_num}")
            else:
                ax.plot(exposure_s, stats_data['mean_brightest_mag'], 
                       color=color, linewidth=2, marker='o', markersize=6,
                       label=f"f/{f_num}")
            
            legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                        label=f"f/{f_num}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Brightest Star Magnitude', fontsize=12)
        ax.set_title(f'{sensor_name} Brightest Star Detection\n' + 
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # Set consistent Y-axis limits
        ax.set_ylim(y_min, y_max)
        
        # X-axis log scale if exposure range is large
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    # Overall title
    title = 'Sensor Brightest Star Detection Comparison'
    if show_error_bars:
        title += ' (with ±1σ error bars)'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


# ============================================================================
# Win Percentage Analysis (Pointing Error Based)
# ============================================================================

def calculate_win_percentage(df, f_number):
    """Calculate HWK win percentage for each exposure time at given f-number (based on pointing error)"""
    # Filter for specific f-number
    mask = df['f_number'] == f_number
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return pd.DataFrame()
    
    # Separate by sensor (pointing errors already calculated)
    hwk_data = subset[subset['sensor'] == 'HWK4123'].copy()
    imx_data = subset[subset['sensor'] == 'IMX455'].copy()
    
    # Merge on experiment_num and exposure_ms to compare same experiments
    merged = pd.merge(
        hwk_data[['experiment_num', 'exposure_ms', 'error_mas']],
        imx_data[['experiment_num', 'exposure_ms', 'error_mas']],
        on=['experiment_num', 'exposure_ms'],
        suffixes=('_hwk', '_imx')
    )
    
    # Group by exposure time and calculate win percentage
    results = []
    for exposure_ms in sorted(merged['exposure_ms'].unique()):
        exp_data = merged[merged['exposure_ms'] == exposure_ms]
        
        # Count wins: HWK wins if it has lower pointing error OR if IMX has NaN
        hwk_wins = 0
        total_comparisons = 0
        
        for _, row in exp_data.iterrows():
            hwk_error = row['error_mas_hwk']
            imx_error = row['error_mas_imx']
            
            # Skip if both are NaN (no winner)
            if pd.isna(hwk_error) and pd.isna(imx_error):
                continue
            
            total_comparisons += 1
            
            # HWK wins if IMX has NaN or HWK has lower error (and isn't NaN)
            if pd.isna(imx_error) and not pd.isna(hwk_error):
                hwk_wins += 1
            elif not pd.isna(hwk_error) and not pd.isna(imx_error) and hwk_error < imx_error:
                hwk_wins += 1
        
        if total_comparisons > 0:
            win_percentage = (hwk_wins / total_comparisons) * 100
            results.append({
                'exposure_ms': exposure_ms,
                'win_percentage': win_percentage,
                'hwk_wins': hwk_wins,
                'total_comparisons': total_comparisons
            })
    
    return pd.DataFrame(results)


# ============================================================================
# Relative Performance Analysis (Quotient)
# ============================================================================

def calculate_relative_performance(df, f_number):
    """Calculate IMX/HWK performance quotient (how much better IMX is than HWK)"""
    # Filter for specific f-number
    mask = df['f_number'] == f_number
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return pd.DataFrame()
    
    # Separate by sensor (pointing errors already calculated)
    hwk_data = subset[subset['sensor'] == 'HWK4123'].copy()
    imx_data = subset[subset['sensor'] == 'IMX455'].copy()
    
    # Merge on experiment_num and exposure_ms to compare same experiments
    merged = pd.merge(
        hwk_data[['experiment_num', 'exposure_ms', 'error_mas']],
        imx_data[['experiment_num', 'exposure_ms', 'error_mas']],
        on=['experiment_num', 'exposure_ms'],
        suffixes=('_hwk', '_imx')
    )
    
    # Group by exposure time and calculate statistics
    results = []
    for exposure_ms in sorted(merged['exposure_ms'].unique()):
        exp_data = merged[merged['exposure_ms'] == exposure_ms]
        
        # Calculate quotients for valid pairs (neither is NaN)
        valid_mask = ~(exp_data['error_mas_hwk'].isna() | exp_data['error_mas_imx'].isna())
        valid_data = exp_data[valid_mask]
        
        if len(valid_data) > 0:
            # Calculate HWK/IMX quotient (>1 means HWK has worse performance)
            quotients = valid_data['error_mas_hwk'] / valid_data['error_mas_imx']
            
            results.append({
                'exposure_ms': exposure_ms,
                'mean_quotient': quotients.mean(),
                'median_quotient': quotients.median(),
                'std_quotient': quotients.std(),
                'min_quotient': quotients.min(),
                'max_quotient': quotients.max(),
                'count': len(quotients)
            })
    
    return pd.DataFrame(results)


# ============================================================================
# Star Count Analysis Functions
# ============================================================================

def plot_ra_dec_coverage(df, output_path=None, show=False, failed_exposure_ms=None):
    """Create sky coverage plots with Mollweide and rectangular projections
    
    Args:
        df: DataFrame with experiment data
        output_path: Path to save the plot
        show: Whether to display the plot
        failed_exposure_ms: If specified, mark failed detections at this exposure time with red X
    """
    # Get unique experiments (each experiment tests all sensors at same position)
    unique_experiments = df.drop_duplicates(subset=['experiment_num'])[['ra', 'dec', 'star_count']]
    
    # If exposure time specified, find failed detections
    failed_detections = None
    if failed_exposure_ms is not None:
        # Get experiments at specific exposure time with no stars detected
        exposure_mask = (df['exposure_ms'] == failed_exposure_ms) & (df['star_count'] == 0)
        failed_detections = df[exposure_mask][['ra', 'dec']].drop_duplicates()
    
    # Convert RA/Dec to radians for Mollweide projection
    ra_rad = np.radians(unique_experiments['ra'].values - 180)  # Center at 180 degrees
    dec_rad = np.radians(unique_experiments['dec'].values)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Left plot: Mollweide projection
    ax1 = plt.subplot(121, projection='mollweide')
    
    # Create 2D histogram for density in Mollweide
    # Note: Mollweide expects longitude from -pi to pi, latitude from -pi/2 to pi/2
    ra_bins = np.linspace(-np.pi, np.pi, 100)
    dec_bins = np.linspace(-np.pi/2, np.pi/2, 50)
    
    # Calculate mean star count per bin
    H, ra_edges, dec_edges = np.histogram2d(ra_rad, dec_rad, bins=[ra_bins, dec_bins])
    
    # For star counts, also bin the detected counts
    star_counts = np.zeros_like(H)
    for i in range(len(ra_bins)-1):
        for j in range(len(dec_bins)-1):
            mask = ((ra_rad >= ra_edges[i]) & (ra_rad < ra_edges[i+1]) &
                    (dec_rad >= dec_edges[j]) & (dec_rad < dec_edges[j+1]))
            if np.any(mask):
                star_counts[i, j] = unique_experiments.loc[unique_experiments.index[mask], 'star_count'].mean()
    
    # Plot density map on Mollweide
    ra_centers = (ra_edges[:-1] + ra_edges[1:]) / 2
    dec_centers = (dec_edges[:-1] + dec_edges[1:]) / 2
    RA, DEC = np.meshgrid(ra_centers, dec_centers)
    
    # Plot the density
    im1 = ax1.pcolormesh(RA, DEC, star_counts.T, cmap='viridis', vmin=0, vmax=50)
    
    # Overlay scatter points
    ax1.scatter(ra_rad, dec_rad, c='white', s=1, alpha=0.3, marker='.')
    
    # Add failed detection markers if specified
    if failed_detections is not None and len(failed_detections) > 0:
        failed_ra_rad = np.radians(failed_detections['ra'].values - 180)
        failed_dec_rad = np.radians(failed_detections['dec'].values)
        ax1.scatter(failed_ra_rad, failed_dec_rad, c='red', s=15, marker='x', 
                   linewidth=1, alpha=0.8, label=f'Failed @ {failed_exposure_ms}ms')
    
    # Add grid and labels for Mollweide
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    ax1.set_title('Sky Coverage - Mollweide Projection', fontsize=14, pad=20)
    
    # Add RA labels
    ax1.set_xticks(np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]))
    ax1.set_xticklabels([f'{int((x+180)%360)}°' for x in [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]])
    
    # Add Dec labels
    ax1.set_yticks(np.radians([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]))
    ax1.set_yticklabels([f'{x}°' for x in [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]])
    
    ax1.set_xlabel('Right Ascension', fontsize=12)
    ax1.set_ylabel('Declination', fontsize=12)
    
    # Right plot: Rectangular projection
    ax2 = plt.subplot(122)
    
    # Create 2D histogram for rectangular projection
    ra_bins_rect = np.linspace(0, 360, 120)
    dec_bins_rect = np.linspace(-90, 90, 60)
    
    H_rect, ra_edges_rect, dec_edges_rect = np.histogram2d(
        unique_experiments['ra'].values, 
        unique_experiments['dec'].values, 
        bins=[ra_bins_rect, dec_bins_rect]
    )
    
    # Calculate mean star counts for rectangular projection
    star_counts_rect = np.zeros_like(H_rect)
    for i in range(len(ra_bins_rect)-1):
        for j in range(len(dec_bins_rect)-1):
            mask = ((unique_experiments['ra'].values >= ra_edges_rect[i]) & 
                   (unique_experiments['ra'].values < ra_edges_rect[i+1]) &
                   (unique_experiments['dec'].values >= dec_edges_rect[j]) & 
                   (unique_experiments['dec'].values < dec_edges_rect[j+1]))
            if np.any(mask):
                star_counts_rect[i, j] = unique_experiments.loc[unique_experiments.index[mask], 'star_count'].mean()
    
    # Plot density map
    im2 = ax2.pcolormesh(ra_edges_rect, dec_edges_rect, star_counts_rect.T, cmap='viridis', vmin=0, vmax=50)
    
    # Overlay scatter points
    ax2.scatter(unique_experiments['ra'], unique_experiments['dec'], 
               c='white', s=1, alpha=0.3, marker='.')
    
    # Add failed detection markers if specified
    if failed_detections is not None and len(failed_detections) > 0:
        ax2.scatter(failed_detections['ra'], failed_detections['dec'], 
                   c='red', s=15, marker='x', linewidth=1, alpha=0.8,
                   label=f'Failed @ {failed_exposure_ms}ms')
        ax2.legend(loc='upper right', fontsize=10)
    
    ax2.set_xlabel('Right Ascension (degrees)', fontsize=12)
    ax2.set_ylabel('Declination (degrees)', fontsize=12)
    ax2.set_title('Sky Coverage - Rectangular Projection', fontsize=14, pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-90, 90)
    
    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar1.set_label('Mean Star Count', fontsize=10)
    
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical', pad=0.02)
    cbar2.set_label('Number of Pointings', fontsize=10, rotation=270, labelpad=20)
    
    # Add overall title and statistics
    n_experiments = len(unique_experiments)
    n_observations = len(df)
    mean_stars = unique_experiments['star_count'].mean()
    max_stars = unique_experiments['star_count'].max()
    no_star_pointings = (unique_experiments['star_count'] == 0).sum()
    
    fig.suptitle(f'Sky Coverage Map - {n_experiments} Unique Pointings', fontsize=16, y=0.98)
    
    # Add statistics box at bottom
    stats_text = (f'Total unique pointings: {n_experiments}\n'
                 f'Total observations: {n_observations}\n'
                 f'Mean stars per pointing: {mean_stars:.1f}\n'
                 f'Max stars: {max_stars}\n'
                 f'Pointings with no stars: {no_star_pointings}')
    
    if failed_detections is not None and len(failed_detections) > 0:
        stats_text += f'\nFailed detections @ {failed_exposure_ms}ms: {len(failed_detections)}'
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"Saved RA/Dec coverage plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def print_star_count_extremes(df, n=5):
    """Print the top and bottom N ra/dec locations by star count"""
    # Group by ra/dec and calculate mean star count across all experiments
    location_stats = df.groupby(['ra', 'dec'])['star_count'].agg(['mean', 'count']).reset_index()
    location_stats = location_stats[location_stats['count'] > 0]  # Filter out NaN-only groups
    
    # Sort by mean star count
    location_stats_sorted = location_stats.sort_values('mean', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"Top {n} sky locations with HIGHEST star counts:")
    print(f"{'='*60}")
    print(f"{'RA':>10} {'Dec':>10} {'Mean Stars':>12} {'N Obs':>8}")
    print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*8}")
    
    for _, row in location_stats_sorted.head(n).iterrows():
        print(f"{row['ra']:10.4f} {row['dec']:10.4f} {row['mean']:12.2f} {int(row['count']):8d}")
    
    print(f"\n{'='*60}")
    print(f"Top {n} sky locations with LOWEST star counts:")
    print(f"{'='*60}")
    print(f"{'RA':>10} {'Dec':>10} {'Mean Stars':>12} {'N Obs':>8}")
    print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*8}")
    
    for _, row in location_stats_sorted.tail(n).iterrows():
        print(f"{row['ra']:10.4f} {row['dec']:10.4f} {row['mean']:12.2f} {int(row['count']):8d}")
    
    print(f"{'='*60}\n")


def plot_win_percentage(df, output_path=None):
    """Create win percentage plot for HWK4123 vs IMX455"""
    # Get unique f-numbers from the data
    f_numbers = sorted(df['f_number'].unique())
    print(f"F-numbers in data: {f_numbers}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use viridis colormap for consistency
    cmap = cm.viridis
    
    # Plot for each f-number
    for i, f_num in enumerate(f_numbers):
        color = get_f_number_color(f_num, cmap)
        
        # Calculate win percentage data
        win_data = calculate_win_percentage(df, f_num)
        
        if len(win_data) > 0:
            # Convert exposure from ms to seconds for display
            exposure_s = win_data['exposure_ms'] / 1000.0
            
            # Plot line
            ax.plot(exposure_s, win_data['win_percentage'], 
                   color=color, linewidth=2, marker='o', markersize=6,
                   label=f"f/{f_num}")
            
            # Add text annotation for number of comparisons at first point
            if len(win_data) > 0:
                first_row = win_data.iloc[0]
                ax.annotate(f'n={first_row["total_comparisons"]}', 
                           xy=(exposure_s.iloc[0], first_row['win_percentage']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    # Add 50% reference line
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (Equal performance)')
    
    # Formatting
    ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
    ax.set_ylabel('HWK4123 Win Percentage (%)', fontsize=12)
    ax.set_title('HWK4123 vs IMX455 Pointing Precision Comparison\n' + 
                '(Lower pointing error wins; NaN always loses)', fontsize=14)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', framealpha=0.9)
    
    # Y-axis range
    ax.set_ylim(0, 100)
    
    # X-axis log scale if exposure range is large
    if len(win_data) > 0:
        max_exp = df['exposure_ms'].max() / 1000.0
        min_exp = df['exposure_ms'].min() / 1000.0
        if max_exp / min_exp > 10:
            ax.set_xscale('log')
            ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax


def plot_relative_performance(df, output_path=None):
    """Create relative performance plot showing HWK/IMX quotient"""
    # Get unique f-numbers from the data
    f_numbers = sorted(df['f_number'].unique())
    print(f"F-numbers in data: {f_numbers}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use viridis colormap for consistency
    cmap = cm.viridis
    
    # Plot for each f-number
    for i, f_num in enumerate(f_numbers):
        color = get_f_number_color(f_num, cmap)
        
        # Calculate relative performance data
        perf_data = calculate_relative_performance(df, f_num)
        
        if len(perf_data) > 0:
            # Convert exposure from ms to seconds for display
            exposure_s = perf_data['exposure_ms'] / 1000.0
            
            # Plot mean quotient without error bars
            ax.plot(exposure_s, perf_data['mean_quotient'], 
                   color=color, linewidth=2, marker='o', markersize=6,
                   label=f"f/{f_num}")
            
            # Add text annotation for number of comparisons at first point
            if len(perf_data) > 0:
                first_row = perf_data.iloc[0]
                ax.annotate(f'n={first_row["count"]}', 
                           xy=(exposure_s.iloc[0], first_row['mean_quotient']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    # Add reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1.0 (Equal performance)')
    ax.axhline(y=1.22, color='red', linestyle=':', alpha=0.3, label='1.22 (Pixel size ratio)')
    
    # Formatting
    ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
    ax.set_ylabel('HWK/IMX Pointing Error Quotient', fontsize=12)
    ax.set_title('Relative Sensor Performance: HWK4123 vs IMX455\n' + 
                '(Values > 1 indicate IMX455 performs better)', fontsize=14)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', framealpha=0.9)
    
    # Y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    # X-axis log scale if exposure range is large
    if len(perf_data) > 0:
        max_exp = df['exposure_ms'].max() / 1000.0
        min_exp = df['exposure_ms'].min() / 1000.0
        if max_exp / min_exp > 10:
            ax.set_xscale('log')
            ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax


def plot_error_variance(df, output_path=None):
    """Plot error variance as a function of exposure time"""
    # Get unique sensors
    sensors = df['sensor'].unique()
    
    # Setup plot
    fig, axes = plt.subplots(1, len(sensors), figsize=(7*len(sensors), 6))
    if len(sensors) == 1:
        axes = [axes]
    
    # Use consistent colormap for f-numbers
    cmap = cm.get_cmap('viridis')
    
    for idx, sensor in enumerate(sensors):
        ax = axes[idx]
        sensor_data = df[df['sensor'] == sensor]
        
        # Get unique f-numbers for this sensor
        f_numbers = sorted(sensor_data['f_number'].unique())
        
        for f_number in f_numbers:
            mask = (sensor_data['f_number'] == f_number)
            subset = sensor_data[mask].copy()
            
            if len(subset) == 0:
                continue
            
            # Group by exposure time and calculate variance
            grouped = subset.groupby('exposure_ms')['pixel_error'].agg(['var', 'std', 'count'])
            grouped = grouped[grouped['count'] >= 5]  # Only show if we have enough samples
            
            if len(grouped) == 0:
                continue
            
            # Plot variance
            color = get_f_number_color(f_number, cmap)
            ax.plot(grouped.index, grouped['var'], 
                   marker='o', linewidth=2, markersize=8,
                   label=f'f/{f_number:.0f}', color=color)
        
        ax.set_title(f'{sensor} - Error Variance vs Exposure Time')
        ax.set_xlabel('Exposure Time (ms)')
        ax.set_ylabel('Pixel Error Variance')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle('Pointing Error Variance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


def plot_error_histograms_separate(df, output_path=None):
    """Plot error histograms as separate files for each exposure time"""
    # Get unique sensors and exposure times
    sensors = df['sensor'].unique()
    exposure_times = sorted(df['exposure_ms'].unique())
    
    # Use consistent colormap for f-numbers
    cmap = cm.get_cmap('viridis')
    
    # Generate base path for outputs
    if output_path:
        output_path = Path(output_path)
        base_name = output_path.stem.replace('_histograms', '')
        ext = output_path.suffix
        parent_dir = output_path.parent
    
    saved_files = []
    
    for exposure_ms in exposure_times:
        # Create figure for this exposure time
        n_sensors = len(sensors)
        fig, axes = plt.subplots(1, n_sensors, figsize=(7*n_sensors, 5))
        
        if n_sensors == 1:
            axes = [axes]
        
        for sen_idx, sensor in enumerate(sensors):
            ax = axes[sen_idx]
            
            # Filter data for this sensor and exposure
            mask = (df['sensor'] == sensor) & (df['exposure_ms'] == exposure_ms)
            subset = df[mask]
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{sensor} - {exposure_ms}ms')
                continue
            
            # Get f-numbers present in this subset
            f_numbers = sorted(subset['f_number'].unique())
            
            # Plot histogram for each f-number
            nan_counts = {}
            outlier_counts = {}
            
            for f_number in f_numbers:
                f_mask = subset['f_number'] == f_number
                errors_all = subset[f_mask]['pixel_error']
                
                # Count NaN values (non-converged/no detection cases)
                nan_count = errors_all.isna().sum()
                total_count = len(errors_all)
                
                # Get only valid (non-NaN) errors
                errors = errors_all.dropna()
                
                if len(errors) > 0:
                    color = get_f_number_color(f_number, cmap)
                    
                    # Count outliers outside 0-1 range
                    outliers_below = (errors < 0).sum()
                    outliers_above = (errors > 1.0).sum()
                    total_outliers = outliers_below + outliers_above
                    
                    if total_outliers > 0:
                        outlier_counts[f_number] = (outliers_below, outliers_above)
                    
                    # Plot histogram with actual data but fixed x-axis range
                    # This will show the real distribution but only display 0-1 range
                    counts, bins, _ = ax.hist(errors, bins=30, 
                                             range=(0, 1.0), alpha=0.6, 
                                             label=f'f/{f_number:.0f} (n={len(errors)})',
                                             color=color, edgecolor='black',
                                             linewidth=0.5)
                    
                    print("Histogram Dump for Exposure: {exposure_ms}ms")
                    for i, count in enumerate(counts):
                        print(f"Bin: {bins[i]:.3f} to {bins[i+1]:.3f}, Count: {int(count)}")

                    # Add mean line if within range
                    mean_err = errors[errors < 1.0].mean()
                    if 0 <= mean_err <= 1.0:
                        ax.axvline(mean_err, color=color, linestyle='--', 
                                  linewidth=1.5, alpha=0.8)
                    
                    # Store NaN count for this f-number
                    if nan_count > 0:
                        nan_counts[f_number] = nan_count
            
            ax.set_title(f'{sensor}')
            ax.set_xlabel('Pixel Error')
            ax.set_ylabel('Count')
            ax.set_xlim(0, 1.0)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Calculate overall statistics
            valid_errors = subset['pixel_error'].dropna()
            nan_total = subset['pixel_error'].isna().sum()
            
            # Add statistics box
            stats_text = f'Detected: {len(valid_errors)}\nTotal: {len(subset)}'
            if len(valid_errors) > 0:
                stats_text += f'\n\nμ={valid_errors.mean():.3f}\nσ={valid_errors.std():.3f}'
            
            # Add outlier counts if any
            if outlier_counts:
                stats_text += '\n\nOutliers:'
                for f_num, (below, above) in outlier_counts.items():
                    outlier_str = f'\nf/{f_num:.0f}: '
                    parts = []
                    if below > 0:
                        parts.append(f'{below} < 0')
                    if above > 0:
                        parts.append(f'{above} > 1')
                    stats_text += outlier_str + ', '.join(parts)
            
            # Add NaN breakdown by f-number if needed
            if nan_counts:
                stats_text += '\n\nNo detection:'
                for f_num, count in nan_counts.items():
                    stats_text += f'\nf/{f_num:.0f}: {count}'
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Error Distribution - {exposure_ms}ms Exposure', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save with exposure time in filename
        if output_path:
            specific_output = parent_dir / f"{base_name}_hist_{int(exposure_ms)}ms{ext}"
            plt.savefig(specific_output, dpi=300, bbox_inches='tight')
            saved_files.append(str(specific_output))
            print(f"Plot saved to: {specific_output}")
            plt.close(fig)
        else:
            plt.show()
    
    if saved_files:
        print(f"Created {len(saved_files)} histogram plots")
    
    return saved_files


def plot_error_histograms(df, output_path=None):
    """Plot error histograms for each exposure time"""
    # Get unique sensors and exposure times
    sensors = df['sensor'].unique()
    exposure_times = sorted(df['exposure_ms'].unique())
    
    # Create subplots grid
    n_exposures = len(exposure_times)
    n_sensors = len(sensors)
    
    fig, axes = plt.subplots(n_exposures, n_sensors, 
                             figsize=(6*n_sensors, 4*n_exposures))
    
    # Handle single sensor or single exposure case
    if n_exposures == 1:
        axes = axes.reshape(1, -1)
    if n_sensors == 1:
        axes = axes.reshape(-1, 1)
    
    # Use consistent colormap for f-numbers
    cmap = cm.get_cmap('viridis')
    
    for exp_idx, exposure_ms in enumerate(exposure_times):
        for sen_idx, sensor in enumerate(sensors):
            ax = axes[exp_idx, sen_idx]
            
            # Filter data for this sensor and exposure
            mask = (df['sensor'] == sensor) & (df['exposure_ms'] == exposure_ms)
            subset = df[mask]
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{sensor} - {exposure_ms}ms')
                continue
            
            # Get f-numbers present in this subset
            f_numbers = sorted(subset['f_number'].unique())
            
            # Plot histogram for each f-number
            outliers_text_lines = []
            for f_number in f_numbers:
                f_mask = subset['f_number'] == f_number
                errors = subset[f_mask]['pixel_error']
                
                if len(errors) > 0:
                    color = get_f_number_color(f_number, cmap)
                    
                    # Count outliers
                    outliers_below = (errors < 0).sum()
                    outliers_above = (errors > 1.0).sum()
                    
                    # Clamp errors to 0-1 range for histogram
                    errors_clamped = errors.clip(0, 1.0)
                    
                    # Calculate histogram with fixed range
                    counts, bins, _ = ax.hist(errors_clamped, bins=30, 
                                             range=(0, 1.0), alpha=0.6, 
                                             label=f'f/{f_number:.0f}',
                                             color=color, edgecolor='black',
                                             linewidth=0.5)
                    
                    # Add mean line (if within range)
                    mean_err = errors.mean()
                    if 0 <= mean_err <= 1.0:
                        ax.axvline(mean_err, color=color, linestyle='--', 
                                  linewidth=1.5, alpha=0.8)
                    
                    # Collect outlier info
                    if outliers_below > 0 or outliers_above > 0:
                        outlier_str = f'f/{f_number:.0f}: '
                        if outliers_below > 0:
                            outlier_str += f'{outliers_below} < 0'
                        if outliers_above > 0:
                            if outliers_below > 0:
                                outlier_str += ', '
                            outlier_str += f'{outliers_above} > 1'
                        outliers_text_lines.append(outlier_str)
            
            ax.set_title(f'{sensor} - {exposure_ms}ms')
            ax.set_xlabel('Pixel Error')
            ax.set_ylabel('Count')
            ax.set_xlim(0, 1.0)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add statistics box
            stats_text = f'n={len(subset)}\nμ={subset["pixel_error"].mean():.3f}\nσ={subset["pixel_error"].std():.3f}'
            
            # Add outlier counts if any
            if outliers_text_lines:
                stats_text += '\n\nOutliers:\n' + '\n'.join(outliers_text_lines)
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Error Distribution by Exposure Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified sensor comparison analysis')
    parser.add_argument('csv_file', type=str, help='Path to experiment CSV file')
    parser.add_argument('--mode', type=str, choices=['mean', 'win', 'quotient', 'stars', 'brightest', 'faintest', 'closure', 'accuracy', 'variance', 'histograms', 'coverage', 'all'], default='mean',
                       help='Analysis mode: mean pointing error, win percentage, quotient, star count, brightest/faintest magnitude, field closure, high accuracy, error variance, error histograms, sky coverage, or all')
    parser.add_argument('--aperture', type=float, default=0.485,
                       help='Telescope aperture in meters (default: 0.485m)')
    parser.add_argument('--output', type=str, help='Output plot filename (PNG/PDF/etc)')
    parser.add_argument('--error-bars', action='store_true',
                       help='Show error bars on mean pointing error and star count plots (default: off)')
    parser.add_argument('--pixel-error-cutoff', type=float, default=2.0,
                       help='Pixel error threshold for filtering mismatched detections (default: 2.0)')
    parser.add_argument('--failed-exposure-ms', type=float, default=250,
                       help='Exposure time in ms to highlight failed star detections in coverage plot (default: 250ms)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        return 1
    
    # Load data once with preprocessing
    df = load_experiment_data(args.csv_file, args.aperture, pixel_error_cutoff=args.pixel_error_cutoff)
    
    # Set default output directory if no output specified
    if not args.output:
        # Create plots directory if it doesn't exist
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Use CSV filename as base for output names
        csv_base = Path(args.csv_file).stem
        default_output = plots_dir / f"{csv_base}.png"
        args.output = str(default_output)
    
    # Generate output filenames
    output_path = Path(args.output)
    base_name = output_path.stem
    ext = output_path.suffix
    parent_dir = output_path.parent
    
    if args.mode == 'all':
        # For 'all' mode, append suffix to each plot type
        mean_output = str(parent_dir / f"{base_name}_mean{ext}")
        win_output = str(parent_dir / f"{base_name}_win{ext}")
        quotient_output = str(parent_dir / f"{base_name}_quotient{ext}")
        stars_output = str(parent_dir / f"{base_name}_stars{ext}")
        brightest_output = str(parent_dir / f"{base_name}_brightest{ext}")
        faintest_output = str(parent_dir / f"{base_name}_faintest{ext}")
        closure_output = str(parent_dir / f"{base_name}_closure{ext}")
        accuracy_output = str(parent_dir / f"{base_name}_accuracy{ext}")
        variance_output = str(parent_dir / f"{base_name}_variance{ext}")
        histograms_output = str(parent_dir / f"{base_name}_histograms{ext}")
        coverage_output = str(parent_dir / f"{base_name}_coverage{ext}")
    else:
        # For individual modes, append mode name if not already present
        if args.mode == 'variance' and '_variance' not in base_name:
            variance_output = str(parent_dir / f"{base_name}_variance{ext}")
        else:
            variance_output = args.output
            
        if args.mode == 'histograms' and '_histograms' not in base_name:
            histograms_output = str(parent_dir / f"{base_name}_histograms{ext}")
        else:
            histograms_output = args.output
            
        if args.mode == 'coverage' and '_coverage' not in base_name:
            coverage_output = str(parent_dir / f"{base_name}_coverage{ext}")
        else:
            coverage_output = args.output
            
        # Keep other outputs same as args.output for single mode
        mean_output = args.output if args.mode != 'mean' or '_mean' in base_name else str(parent_dir / f"{base_name}_mean{ext}")
        win_output = args.output if args.mode != 'win' or '_win' in base_name else str(parent_dir / f"{base_name}_win{ext}")
        quotient_output = args.output if args.mode != 'quotient' or '_quotient' in base_name else str(parent_dir / f"{base_name}_quotient{ext}")
        stars_output = args.output if args.mode != 'stars' or '_stars' in base_name else str(parent_dir / f"{base_name}_stars{ext}")
        brightest_output = args.output if args.mode != 'brightest' or '_brightest' in base_name else str(parent_dir / f"{base_name}_brightest{ext}")
        faintest_output = args.output if args.mode != 'faintest' or '_faintest' in base_name else str(parent_dir / f"{base_name}_faintest{ext}")
        closure_output = args.output if args.mode != 'closure' or '_closure' in base_name else str(parent_dir / f"{base_name}_closure{ext}")
        accuracy_output = args.output if args.mode != 'accuracy' or '_accuracy' in base_name else str(parent_dir / f"{base_name}_accuracy{ext}")
    
    # Create requested plots
    if args.mode in ['mean', 'all']:
        plot_mean_pointing_error(df, mean_output, args.error_bars)
    
    if args.mode in ['win', 'all']:
        plot_win_percentage(df, win_output)
    
    if args.mode in ['quotient', 'all']:
        plot_relative_performance(df, quotient_output)
    
    if args.mode in ['stars', 'all']:
        plot_mean_star_count(df, stars_output, args.error_bars)
    
    if args.mode in ['brightest', 'all']:
        plot_brightest_magnitude(df, brightest_output, args.error_bars)
    
    if args.mode in ['faintest', 'all']:
        plot_faintest_magnitude(df, faintest_output, args.error_bars)
    
    if args.mode in ['closure', 'all']:
        plot_field_closure_percentage(df, closure_output)
    
    if args.mode in ['accuracy', 'all']:
        plot_high_accuracy_percentage(df, accuracy_output)
    
    if args.mode in ['variance', 'all']:
        plot_error_variance(df, variance_output)
    
    if args.mode in ['histograms', 'all']:
        plot_error_histograms_separate(df, histograms_output)
    
    if args.mode in ['coverage', 'all']:
        plot_ra_dec_coverage(df, coverage_output, failed_exposure_ms=args.failed_exposure_ms)
    
    # Always print star count extremes at the end
    print_star_count_extremes(df, n=5)
    
    return 0


if __name__ == '__main__':
    exit(main())
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

def load_experiment_data(csv_path, aperture_m=0.485, mask_outliers=True):
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
        # Mask out experiments with pixel error > 2 pixels
        mask = df['pixel_error'] > 2.0
        num_masked = mask.sum()
        if num_masked > 0:
            print(f"Masking {num_masked} rows with pixel_error > 2 pixels")
            df.loc[mask, ['star_count', 'brightest_mag', 'faintest_mag', 'pixel_error', 'error_mas']] = np.nan
    
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
    n_f_numbers = len(f_numbers)
    color_indices = np.linspace(0, 1, n_f_numbers)
    
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
            color = cmap(color_indices[f_numbers.index(f_num)])
            
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
    n_f_numbers = len(f_numbers)
    color_indices = np.linspace(0, 1, n_f_numbers)
    
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
            color = cmap(color_indices[f_numbers.index(f_num)])
            
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
    n_f_numbers = len(f_numbers)
    color_indices = np.linspace(0, 1, n_f_numbers)
    
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
            color = cmap(color_indices[f_numbers.index(f_num)])
            
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
    n_f_numbers = len(f_numbers)
    color_indices = np.linspace(0, 1, n_f_numbers)
    
    # Plot data for each sensor
    for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
        # Get pixel size for this sensor
        pixel_size_um = sensor_pixel_sizes.get(sensor_name)
        
        # Store handles for legend
        legend_elements = []
        
        # Plot for each f-number
        for i, f_num in enumerate(f_numbers):
            color = cmap(color_indices[i])
            
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
    n_f_numbers = len(f_numbers)
    color_indices = np.linspace(0, 1, n_f_numbers)
    
    # Plot data for each sensor
    for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
        # Get pixel size for this sensor
        pixel_size_um = sensor_pixel_sizes.get(sensor_name)
        
        # Store handles for legend
        legend_elements = []
        
        # Plot for each f-number
        for i, f_num in enumerate(f_numbers):
            color = cmap(color_indices[i])
            
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
    n_f_numbers = len(f_numbers)
    color_indices = np.linspace(0, 1, n_f_numbers)
    
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
            color = cmap(color_indices[f_numbers.index(f_num)])
            
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
    n_f_numbers = len(f_numbers)
    color_indices = np.linspace(0, 1, n_f_numbers)
    
    # Plot for each f-number
    for i, f_num in enumerate(f_numbers):
        color = cmap(color_indices[i])
        
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
    n_f_numbers = len(f_numbers)
    color_indices = np.linspace(0, 1, n_f_numbers)
    
    # Plot for each f-number
    for i, f_num in enumerate(f_numbers):
        color = cmap(color_indices[i])
        
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


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified sensor comparison analysis')
    parser.add_argument('csv_file', type=str, help='Path to experiment CSV file')
    parser.add_argument('--mode', type=str, choices=['mean', 'win', 'quotient', 'stars', 'brightest', 'faintest', 'closure', 'accuracy', 'all'], default='mean',
                       help='Analysis mode: mean pointing error, win percentage, quotient, star count, brightest/faintest magnitude, field closure, high accuracy, or all')
    parser.add_argument('--aperture', type=float, default=0.485,
                       help='Telescope aperture in meters (default: 0.485m)')
    parser.add_argument('--output', type=str, help='Output plot filename (PNG/PDF/etc)')
    parser.add_argument('--error-bars', action='store_true',
                       help='Show error bars on mean pointing error and star count plots (default: off)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        return 1
    
    # Load data once with preprocessing
    df = load_experiment_data(args.csv_file, args.aperture)
    
    # Generate output filenames if multiple mode is selected
    if args.mode == 'all' and args.output:
        base_name = Path(args.output).stem
        ext = Path(args.output).suffix
        mean_output = f"{base_name}_mean{ext}"
        win_output = f"{base_name}_win{ext}"
        quotient_output = f"{base_name}_quotient{ext}"
        stars_output = f"{base_name}_stars{ext}"
        brightest_output = f"{base_name}_brightest{ext}"
        faintest_output = f"{base_name}_faintest{ext}"
        closure_output = f"{base_name}_closure{ext}"
        accuracy_output = f"{base_name}_accuracy{ext}"
    else:
        mean_output = args.output
        win_output = args.output
        quotient_output = args.output
        stars_output = args.output
        brightest_output = args.output
        faintest_output = args.output
        closure_output = args.output
        accuracy_output = args.output
    
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
    
    # Always print star count extremes at the end
    print_star_count_extremes(df, n=5)
    
    return 0


if __name__ == '__main__':
    exit(main())
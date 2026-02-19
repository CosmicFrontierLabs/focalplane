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

# Histogram bounds for standard deviation charts (in pixels)
HISTOGRAM_BOUND = 0.5


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
    
    # Calculate f-number and round to two decimal places
    df['f_number'] = (df['focal_length_m'] / aperture_m).round(2)
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

def calculate_mean_star_statistics(df, sensor, f_number):
    """Calculate mean, std, and median star count for each exposure time"""
    mask = (df['sensor'] == sensor) & (df['f_number'] == f_number)
    sensor_data = df[mask]
    
    if len(sensor_data) == 0:
        return pd.DataFrame()
    
    # Group by exposure and calculate statistics
    stats = sensor_data.groupby('exposure_ms')['star_count'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('count', 'size'),
    ]).reset_index()
    
    # Calculate standard error
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    
    # Sort by exposure time
    stats = stats.sort_values('exposure_ms')
    
    return stats


def plot_mean_star_count(df, output_path=None, show_error_bars=False):
    """Create separate analysis plots for each sensor showing mean star count"""
    # Get sensor pixel sizes (not used here but keeping for consistency)
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors, figsize=(8*n_sensors, 8))
    
    # Make axes iterable even for single sensor
    if n_sensors == 1:
        axes = [axes]
    
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
            
            stats_data = calculate_mean_star_statistics(df, sensor_name, f_num)
            
            if len(stats_data) > 0:
                # Convert exposure from ms to seconds for display
                exposure_s = stats_data['exposure_ms'] / 1000.0
                
                # Plot mean star count with optional error bars
                if show_error_bars:
                    ax.errorbar(exposure_s, stats_data['mean'], 
                               yerr=stats_data['sem'], 
                               color=color, linewidth=2, marker='o', markersize=6,
                               capsize=4, capthick=1.5, alpha=0.8,
                               label=f"f/{f_num:.2f}")
                else:
                    ax.plot(exposure_s, stats_data['mean'], 
                           color=color, linewidth=2, marker='o', markersize=6,
                           label=f"f/{f_num:.2f}")
                
                
                legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                            label=f"f/{f_num:.2f}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Mean Star Count', fontsize=12)
        ax.set_title(f'{sensor_name} Mean Star Count\n' + 
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # X-axis log scale if exposure range is large
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    # Overall title
    fig.suptitle('Star Detection Performance\n(Mean detected stars per exposure)', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


# ============================================================================
# Brightest Magnitude Analysis
# ============================================================================

def calculate_brightest_magnitude_statistics(df, sensor, f_number):
    """Calculate mean, std, and median brightest magnitude for each exposure time"""
    mask = (df['sensor'] == sensor) & (df['f_number'] == f_number)
    sensor_data = df[mask]
    
    if len(sensor_data) == 0:
        return pd.DataFrame()
    
    # Filter out NaN values for brightest magnitude
    valid_data = sensor_data.dropna(subset=['brightest_mag'])
    
    # Group by exposure and calculate statistics
    stats = valid_data.groupby('exposure_ms')['brightest_mag'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('count', 'size'),
    ]).reset_index()
    
    # Calculate standard error
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    
    # Sort by exposure time
    stats = stats.sort_values('exposure_ms')
    
    return stats


def plot_brightest_magnitude(df, output_path=None, show_error_bars=False):
    """Create separate analysis plots for each sensor showing brightest detected magnitude"""
    # Get sensor pixel sizes (not used here but keeping for consistency)
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors, figsize=(8*n_sensors, 8))
    
    # Make axes iterable even for single sensor
    if n_sensors == 1:
        axes = [axes]
    
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
            
            stats_data = calculate_brightest_magnitude_statistics(df, sensor_name, f_num)
            
            if len(stats_data) > 0:
                # Convert exposure from ms to seconds for display
                exposure_s = stats_data['exposure_ms'] / 1000.0
                
                # Plot mean brightest magnitude with optional error bars
                if show_error_bars:
                    ax.errorbar(exposure_s, stats_data['mean'], 
                               yerr=stats_data['sem'], 
                               color=color, linewidth=2, marker='o', markersize=6,
                               capsize=4, capthick=1.5, alpha=0.8,
                               label=f"f/{f_num:.2f}")
                else:
                    ax.plot(exposure_s, stats_data['mean'], 
                           color=color, linewidth=2, marker='o', markersize=6,
                           label=f"f/{f_num:.2f}")
                
                
                legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                            label=f"f/{f_num:.2f}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Brightest Star Magnitude', fontsize=12)
        ax.set_title(f'{sensor_name} Brightest Star Detection\n' + 
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # Invert Y-axis (lower magnitude = brighter star)
        ax.invert_yaxis()
        
        # X-axis log scale if exposure range is large
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    # Overall title
    fig.suptitle('Brightest Star Detection\n(Mean magnitude of brightest detected star)', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


# ============================================================================
# Faintest Magnitude Analysis
# ============================================================================

def calculate_faintest_magnitude_statistics(df, sensor, f_number):
    """Calculate mean, std, and median faintest magnitude for each exposure time"""
    mask = (df['sensor'] == sensor) & (df['f_number'] == f_number)
    sensor_data = df[mask]
    
    if len(sensor_data) == 0:
        return pd.DataFrame()
    
    # Filter out NaN values for faintest magnitude
    valid_data = sensor_data.dropna(subset=['faintest_mag'])
    
    # Group by exposure and calculate statistics
    stats = valid_data.groupby('exposure_ms')['faintest_mag'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('count', 'size'),
    ]).reset_index()
    
    # Calculate standard error
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    
    # Sort by exposure time
    stats = stats.sort_values('exposure_ms')
    
    return stats


def plot_faintest_magnitude(df, output_path=None, show_error_bars=False):
    """Create separate analysis plots for each sensor showing faintest detected magnitude"""
    # Get sensor pixel sizes (not used here but keeping for consistency)
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors, figsize=(8*n_sensors, 8))
    
    # Make axes iterable even for single sensor
    if n_sensors == 1:
        axes = [axes]
    
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
            
            stats_data = calculate_faintest_magnitude_statistics(df, sensor_name, f_num)
            
            if len(stats_data) > 0:
                # Convert exposure from ms to seconds for display
                exposure_s = stats_data['exposure_ms'] / 1000.0
                
                # Plot mean faintest magnitude with optional error bars
                if show_error_bars:
                    ax.errorbar(exposure_s, stats_data['mean'], 
                               yerr=stats_data['sem'], 
                               color=color, linewidth=2, marker='o', markersize=6,
                               capsize=4, capthick=1.5, alpha=0.8,
                               label=f"f/{f_num:.2f}")
                else:
                    ax.plot(exposure_s, stats_data['mean'], 
                           color=color, linewidth=2, marker='o', markersize=6,
                           label=f"f/{f_num:.2f}")
                
                
                legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                            label=f"f/{f_num:.2f}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Faintest Star Magnitude', fontsize=12)
        ax.set_title(f'{sensor_name} Faintest Star Detection\n' + 
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # X-axis log scale if exposure range is large
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    # Overall title
    fig.suptitle('Faintest Star Detection\n(Mean magnitude of faintest detected star)', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


# ============================================================================
# High Accuracy Percentage Analysis
# ============================================================================

def calculate_high_accuracy_statistics(df, sensor, f_number, threshold=0.1):
    """Calculate percentage of experiments with pixel error < threshold"""
    mask = (df['sensor'] == sensor) & (df['f_number'] == f_number)
    sensor_data = df[mask]
    
    if len(sensor_data) == 0:
        return pd.DataFrame()
    
    # Filter out NaN values for pixel_error
    valid_data = sensor_data.dropna(subset=['pixel_error'])
    
    # Group by exposure and calculate high accuracy percentage
    stats = []
    for exposure_ms in sorted(valid_data['exposure_ms'].unique()):
        exp_data = valid_data[valid_data['exposure_ms'] == exposure_ms]
        
        total_experiments = len(exp_data)
        high_accuracy_count = (exp_data['pixel_error'] < threshold).sum()
        high_accuracy_pct = (high_accuracy_count / total_experiments * 100) if total_experiments > 0 else 0
        
        stats.append({
            'exposure_ms': exposure_ms,
            'high_accuracy_pct': high_accuracy_pct,
            'high_accuracy_count': high_accuracy_count,
            'total_experiments': total_experiments
        })
    
    return pd.DataFrame(stats)


def plot_high_accuracy_percentage(df, output_path=None):
    """Create analysis plot showing high accuracy percentage"""
    # Get sensor info
    sensor_name = df['sensor'].iloc[0]  # Assume single sensor
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    pixel_size_um = sensor_pixel_sizes.get(sensor_name, 0)
    
    # Get f-numbers from the data
    f_numbers = sorted(df['f_number'].unique())
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a consistent colormap
    cmap = cm.viridis
    
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
                   label=f"f/{f_num:.2f}")
            
            # Add text annotation for number of experiments at first point
            if len(stats_data) > 0:
                first_row = stats_data.iloc[0]
                ax.annotate(f'n={first_row["total_experiments"]}', 
                           xy=(exposure_s.iloc[0], first_row['high_accuracy_pct']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                        label=f"f/{f_num:.2f}"))
    
    # Formatting
    ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
    ax.set_ylabel('High Accuracy Percentage (%)', fontsize=12)
    ax.set_title(f'{sensor_name} High Accuracy Rate\n' + 
                f'(Percentage of experiments with pixel error < 0.1)', fontsize=14)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
    
    # Set Y-axis limits 0-100%
    ax.set_ylim(0, 105)
    
    # X-axis log scale if exposure range is large
    if f_numbers and len(stats_data) > 0:
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


def plot_field_closure_percentage(df, output_path=None):
    """Create separate analysis plots for each sensor showing field closure percentage"""
    # Get sensor pixel sizes (not used here but keeping for consistency)
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors, figsize=(8*n_sensors, 8))
    
    # Make axes iterable even for single sensor
    if n_sensors == 1:
        axes = [axes]
    
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
            
            # Calculate field closure percentage for this sensor and f-number
            mask = (df['sensor'] == sensor_name) & (df['f_number'] == f_num)
            sensor_data = df[mask]
            
            if len(sensor_data) > 0:
                # Group by exposure and calculate closure percentage
                closure_data = []
                for exposure_ms in sorted(sensor_data['exposure_ms'].unique()):
                    exp_data = sensor_data[sensor_data['exposure_ms'] == exposure_ms]
                    
                    total_experiments = len(exp_data)
                    closed_experiments = (exp_data['star_count'] >= 1).sum()
                    closure_pct = (closed_experiments / total_experiments * 100) if total_experiments > 0 else 0
                    
                    closure_data.append({
                        'exposure_ms': exposure_ms,
                        'closure_pct': closure_pct,
                        'total': total_experiments
                    })
                
                closure_df = pd.DataFrame(closure_data)
                
                # Convert exposure from ms to seconds for display
                exposure_s = closure_df['exposure_ms'] / 1000.0
                
                # Plot closure percentage
                ax.plot(exposure_s, closure_df['closure_pct'], 
                       color=color, linewidth=2, marker='o', markersize=6,
                       label=f"f/{f_num:.2f}")
                
                # Add text annotation for number of experiments at first point
                if len(closure_df) > 0:
                    first_row = closure_df.iloc[0]
                    ax.annotate(f'n={first_row["total"]}', 
                               xy=(exposure_s.iloc[0], first_row['closure_pct']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
                
                legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                            label=f"f/{f_num:.2f}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Field Closure Percentage (%)', fontsize=12)
        ax.set_title(f'{sensor_name} Field Closure Rate\n' + 
                    f'(% detecting ≥1 star)', fontsize=14)
        
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
    fig.suptitle('Field Closure Performance\n(Percentage of experiments detecting at least one star)', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


# ============================================================================
# Mean Pointing Error Analysis
# ============================================================================

def calculate_pointing_statistics(df, sensor, f_number):
    """Calculate mean and std pointing error for each exposure time"""
    mask = (df['sensor'] == sensor) & (df['f_number'] == f_number)
    sensor_data = df[mask]
    
    if len(sensor_data) == 0:
        return pd.DataFrame()
    
    # Filter out NaN values for error_mas
    valid_data = sensor_data.dropna(subset=['error_mas'])
    
    # Group by exposure and calculate statistics
    stats = valid_data.groupby('exposure_ms')['error_mas'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('count', 'size'),
    ]).reset_index()
    
    # Calculate standard error
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    
    # Sort by exposure time
    stats = stats.sort_values('exposure_ms')
    
    return stats


def plot_mean_pointing_error(df, output_path=None, show_error_bars=False):
    """Create separate analysis plots for each sensor"""
    # Get sensor pixel sizes
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors, figsize=(8*n_sensors, 8))
    
    # Make axes iterable even for single sensor
    if n_sensors == 1:
        axes = [axes]
    
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
            
            stats_data = calculate_pointing_statistics(df, sensor_name, f_num)
            
            if len(stats_data) > 0:
                # Convert exposure from ms to seconds for display
                exposure_s = stats_data['exposure_ms'] / 1000.0
                
                # Plot mean pointing error with optional error bars
                if show_error_bars:
                    ax.errorbar(exposure_s, stats_data['mean'], 
                               yerr=stats_data['sem'], 
                               color=color, linewidth=2, marker='o', markersize=6,
                               capsize=4, capthick=1.5, alpha=0.8,
                               label=f"f/{f_num:.2f}")
                else:
                    ax.plot(exposure_s, stats_data['mean'], 
                           color=color, linewidth=2, marker='o', markersize=6,
                           label=f"f/{f_num:.2f}")
                
                
                legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                            label=f"f/{f_num:.2f}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Mean Pointing Error (milliarcsec)', fontsize=12)
        ax.set_title(f'{sensor_name} Pointing Performance\n' + 
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # X-axis log scale if exposure range is large  
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
    
    # Overall title
    fig.suptitle('Mean Pointing Error Comparison\n(Lower is better)', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


# ============================================================================
# Win Percentage Analysis
# ============================================================================

def plot_win_percentage(df, output_path=None):
    """Plot win percentage for IMX455 vs HWK4123 at each f-number"""
    
    # Get unique f-numbers
    f_numbers = sorted(df['f_number'].unique())
    print(f"F-numbers in data: {f_numbers}")
    
    # Filter for sensors
    imx455_data = df[df['sensor'] == 'IMX455']
    hwk4123_data = df[df['sensor'] == 'HWK4123']
    
    # Check if we have data for both sensors
    if len(imx455_data) == 0 or len(hwk4123_data) == 0:
        print("Warning: Need data for both IMX455 and HWK4123 sensors for win percentage analysis")
        return None, None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use consistent colormap
    cmap = cm.viridis
    
    # Store data for each f-number
    for f_num in f_numbers:
        color = get_f_number_color(f_num, cmap)
        
        # Get data for this f-number
        imx455_f = imx455_data[imx455_data['f_number'] == f_num]
        hwk4123_f = hwk4123_data[hwk4123_data['f_number'] == f_num]
        
        if len(imx455_f) == 0 or len(hwk4123_f) == 0:
            print(f"Skipping f/{f_num:.2f} - insufficient data")
            continue
        
        # Get unique exposures that both sensors have
        exposures = sorted(set(imx455_f['exposure_ms'].unique()) & 
                          set(hwk4123_f['exposure_ms'].unique()))
        
        if len(exposures) == 0:
            continue
        
        win_percentages = []
        exposure_times_s = []
        
        for exp_ms in exposures:
            # Get error data for each sensor at this exposure
            imx455_errors = imx455_f[imx455_f['exposure_ms'] == exp_ms]['error_mas'].dropna()
            hwk4123_errors = hwk4123_f[hwk4123_f['exposure_ms'] == exp_ms]['error_mas'].dropna()
            
            if len(imx455_errors) == 0 or len(hwk4123_errors) == 0:
                continue
            
            # Calculate win percentage (IMX455 wins when it has lower error)
            # We compare all combinations
            total_comparisons = 0
            imx455_wins = 0
            
            for imx_error in imx455_errors:
                for hwk_error in hwk4123_errors:
                    total_comparisons += 1
                    if imx_error < hwk_error:
                        imx455_wins += 1
            
            if total_comparisons > 0:
                win_pct = (imx455_wins / total_comparisons) * 100
                win_percentages.append(win_pct)
                exposure_times_s.append(exp_ms / 1000.0)
        
        if len(win_percentages) > 0:
            # Plot win percentage
            ax.plot(exposure_times_s, win_percentages,
                   color=color, linewidth=2, marker='o', markersize=6,
                   label=f"f/{f_num:.2f}")
    
    # Add 50% reference line
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
               label='Equal performance')
    
    # Add shaded regions
    ax.axhspan(50, 100, alpha=0.1, color='blue', label='IMX455 better')
    ax.axhspan(0, 50, alpha=0.1, color='red', label='HWK4123 better')
    
    # Formatting
    ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
    ax.set_ylabel('IMX455 Win Percentage (%)', fontsize=12)
    ax.set_title('Sensor Win Percentage\n(% of experiments where IMX455 has lower pointing error than HWK4123)',
                fontsize=14)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', framealpha=0.9)
    
    # Y-axis limits
    ax.set_ylim(0, 100)
    
    # X-axis log scale if exposure range is large
    if len(exposures) > 0:
        max_exp = max(exposures) / 1000.0
        min_exp = min(exposures) / 1000.0
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
# Performance Quotient Analysis  
# ============================================================================

def plot_relative_performance(df, output_path=None):
    """Plot relative performance quotient (HWK4123 / IMX455) for each f-number"""
    
    # Get unique f-numbers
    f_numbers = sorted(df['f_number'].unique())
    print(f"F-numbers in data: {f_numbers}")
    
    # Filter for sensors
    imx455_data = df[df['sensor'] == 'IMX455']
    hwk4123_data = df[df['sensor'] == 'HWK4123']
    
    # Check if we have data for both sensors
    if len(imx455_data) == 0 or len(hwk4123_data) == 0:
        print("Warning: Need data for both IMX455 and HWK4123 sensors for quotient analysis")
        return None, None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use consistent colormap
    cmap = cm.viridis
    
    # Store data for each f-number
    for f_num in f_numbers:
        color = get_f_number_color(f_num, cmap)
        
        # Get data for this f-number
        imx455_f = imx455_data[imx455_data['f_number'] == f_num]
        hwk4123_f = hwk4123_data[hwk4123_data['f_number'] == f_num]
        
        if len(imx455_f) == 0 or len(hwk4123_f) == 0:
            print(f"Skipping f/{f_num:.2f} - insufficient data")
            continue
        
        # Calculate statistics for each exposure
        imx455_stats = calculate_pointing_statistics(df, 'IMX455', f_num)
        hwk4123_stats = calculate_pointing_statistics(df, 'HWK4123', f_num)
        
        # Merge on exposure_ms
        merged = pd.merge(imx455_stats, hwk4123_stats, on='exposure_ms', 
                         suffixes=('_imx', '_hwk'))
        
        if len(merged) > 0:
            # Calculate quotient (HWK4123 / IMX455)
            # Values > 1 mean IMX455 is better (lower error)
            # Values < 1 mean HWK4123 is better
            quotient = merged['mean_hwk'] / merged['mean_imx']
            exposure_s = merged['exposure_ms'] / 1000.0
            
            # Plot quotient
            ax.plot(exposure_s, quotient,
                   color=color, linewidth=2, marker='o', markersize=6,
                   label=f"f/{f_num:.2f}")
    
    # Add reference line at 1 (equal performance)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Equal performance')
    
    # Add shaded regions with labels
    # Get y-axis limits for shading
    ylim = ax.get_ylim()
    ax.axhspan(1, ylim[1], alpha=0.1, color='blue')
    ax.axhspan(ylim[0], 1, alpha=0.1, color='red')
    
    # Add text annotations for regions
    ax.text(0.02, 0.98, 'IMX455 Better', transform=ax.transAxes,
           fontsize=10, va='top', color='blue', alpha=0.7, weight='bold')
    ax.text(0.02, 0.02, 'HWK4123 Better', transform=ax.transAxes,
           fontsize=10, va='bottom', color='red', alpha=0.7, weight='bold')
    
    # Formatting
    ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
    ax.set_ylabel('Performance Quotient (HWK4123 error / IMX455 error)', fontsize=12)
    ax.set_title('Relative Sensor Performance\n(Values > 1 indicate IMX455 superiority)',
                fontsize=14)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', framealpha=0.9)
    
    # Y-axis log scale for better visualization of quotients
    ax.set_yscale('log')
    ax.set_ylabel('Performance Quotient (log scale)\n(HWK4123 error / IMX455 error)', fontsize=12)
    
    # Set reasonable y-axis limits
    ax.set_ylim(0.5, 2.0)
    
    # X-axis log scale if exposure range is large
    if len(exposure_s) > 0:
        max_exp = exposure_s.max()
        min_exp = exposure_s.min()
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
# Error Variance Analysis
# ============================================================================

def plot_error_variance(df, output_path=None):
    """Plot variance of pointing error for each sensor at different exposures"""
    # Get sensor pixel sizes
    sensor_pixel_sizes = get_sensor_pixel_sizes()
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor
    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors, figsize=(8*n_sensors, 8))
    
    # Make axes iterable even for single sensor
    if n_sensors == 1:
        axes = [axes]
    
    # Use a consistent colormap
    cmap = cm.viridis
    
    # Plot data for each sensor
    for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
        # Get pixel size for this sensor
        pixel_size_um = sensor_pixel_sizes.get(sensor_name)
        
        # Store handles for legend
        legend_elements = []
        
        # Plot for each f-number
        for i, f_num in enumerate(f_numbers):
            color = get_f_number_color(f_num, cmap)
            
            # Get statistics
            stats_data = calculate_pointing_statistics(df, sensor_name, f_num)
            
            if len(stats_data) > 0 and 'std' in stats_data.columns:
                # Convert exposure from ms to seconds for display
                exposure_s = stats_data['exposure_ms'] / 1000.0
                
                # Plot variance (std squared)
                variance = stats_data['std'] ** 2
                
                ax.plot(exposure_s, variance,
                       color=color, linewidth=2, marker='o', markersize=6,
                       label=f"f/{f_num:.2f}")
                
                legend_elements.append(Line2D([0], [0], color=color, lw=2,
                                            label=f"f/{f_num:.2f}"))
        
        # Formatting for each subplot
        ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
        ax.set_ylabel('Error Variance (mas²)', fontsize=12)
        ax.set_title(f'{sensor_name} Pointing Error Variance\n' +
                    f'(pixel size: {pixel_size_um} μm)', fontsize=14)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        
        # Use log scale for both axes if range is large
        if 'exposure_s' in locals() and len(exposure_s) > 0:
            max_exp = df['exposure_ms'].max() / 1000.0
            min_exp = df['exposure_ms'].min() / 1000.0
            if max_exp / min_exp > 10:
                ax.set_xscale('log')
                ax.set_xlabel('Exposure Time (seconds, log scale)', fontsize=12)
            
            # Also use log scale for y-axis
            ax.set_yscale('log')
            ax.set_ylabel('Error Variance (mas², log scale)', fontsize=12)
    
    # Overall title
    fig.suptitle('Pointing Error Variance Analysis\n(Lower variance indicates more consistent performance)',
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


# ============================================================================
# Error Histogram Analysis
# ============================================================================

def plot_error_histograms_separate(df, output_base_path=None):
    """Create separate histogram plots for each exposure time"""
    # Get unique exposures and f-numbers
    exposures = sorted(df['exposure_ms'].unique())
    f_numbers = sorted(df['f_number'].unique())
    sensors = sorted(df['sensor'].unique())
    
    # Use consistent colormap
    cmap = cm.viridis
    
    # Track outlier information for each f-number
    outlier_summary = {}
    
    # Create a plot for each exposure
    plot_count = 0
    for exp_ms in exposures:
        exp_data = df[df['exposure_ms'] == exp_ms]
        
        # Skip if no data for this exposure
        if len(exp_data) == 0:
            continue
        
        # Create figure with subplots for each sensor
        n_sensors = len(sensors)
        fig, axes = plt.subplots(1, n_sensors, figsize=(8*n_sensors, 6))
        
        # Make axes iterable even for single sensor
        if n_sensors == 1:
            axes = [axes]
        
        for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
            sensor_data = exp_data[exp_data['sensor'] == sensor_name]
            
            # Track outliers for this exposure/sensor
            outlier_counts = {}
            
            # Plot histogram for each f-number
            for f_num in f_numbers:
                color = get_f_number_color(f_num, cmap)
                
                # Get errors for this f-number
                f_data = sensor_data[sensor_data['f_number'] == f_num]
                errors = f_data['pixel_error'].dropna()
                
                if len(errors) > 0:
                    # Count outliers (errors > 1.0 pixel)
                    outliers_above = (errors > 1.0).sum()
                    outliers_below = (errors < 0.0).sum()  # Should be 0, but checking anyway
                    
                    # Store outlier counts
                    if outliers_above > 0 or outliers_below > 0:
                        outlier_counts[f_num] = (outliers_below, outliers_above)
                    
                    # Plot histogram with actual data but fixed x-axis range
                    # This will show the real distribution but only display 0-1 range
                    counts, bins, _ = ax.hist(errors, bins=30, 
                                             range=(0, 1.0), alpha=0.6, 
                                             label=f"f/{f_num:.2f} (n={len(errors)})",
                                             color=color, edgecolor='black',
                                             linewidth=0.5)
                    
                    # Add mean line if within range
                    mean_err = errors[errors < 1.0].mean()
                    if 0 <= mean_err <= 1.0:
                        ax.axvline(mean_err, color=color, linestyle='--', 
                                  linewidth=1.5, alpha=0.8)
            
            # Formatting
            ax.set_xlabel('Pixel Error', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f'{sensor_name} @ {exp_ms}ms', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
            ax.set_xlim(0, 1.0)
            
            # Add outlier annotation if present
            if outlier_counts:
                outlier_text = "Outliers (>1px):\n"
                for f_num, (below, above) in outlier_counts.items():
                    if above > 0:
                        outlier_text += f"f/{f_num:.2f}: {above}\n"
                
                ax.text(0.98, 0.98, outlier_text.strip(), 
                       transform=ax.transAxes,
                       fontsize=8, va='top', ha='right',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Overall title for this exposure
        fig.suptitle(f'Pixel Error Distribution - {exp_ms}ms Exposure', fontsize=14)
        plt.tight_layout()
        
        # Save or show
        if output_base_path:
            # Generate unique filename for each exposure
            output_path = Path(output_base_path)
            parent_dir = output_path.parent
            base_name = output_path.stem.replace('_histograms', '')
            ext = output_path.suffix
            
            hist_output = str(parent_dir / f"{base_name}_hist_{int(exp_ms)}ms{ext}")
            plt.savefig(hist_output, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {hist_output}")
        else:
            plt.show()
        
        plt.close(fig)
        plot_count += 1
    
    print(f"Created {plot_count} histogram plots")
    return None


# ============================================================================
# Sky Coverage Analysis
# ============================================================================

def plot_sky_coverage_mollweide(df, output_path=None, exposure_ms=None, precision_threshold=0.1):
    """Plot sky coverage on Mollweide projection with color background"""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata
    
    # Filter for specific exposure if requested
    if exposure_ms is not None:
        df_filtered = df[df['exposure_ms'] == exposure_ms].copy()
        title_suffix = f" @ {exposure_ms}ms"
        if len(df_filtered) == 0:
            print(f"No data found for exposure {exposure_ms}ms")
            print(f"Available exposures: {sorted(df['exposure_ms'].unique())}")
            return None
        df = df_filtered
    else:
        title_suffix = " (all exposures)"
    
    # Get unique pointings with their mean star counts and precision
    grouped = df.groupby(['ra', 'dec']).agg({
        'star_count': ['mean', 'count'],
        'pixel_error': 'mean'
    }).reset_index()
    grouped.columns = ['ra', 'dec', 'star_mean', 'star_count', 'pixel_error_mean']
    
    if len(grouped) == 0:
        print("No grouped data available after filtering")
        return None
    
    # Create figure with Mollweide projection
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='mollweide')
    
    # Create background interpolation
    ra_grid = np.linspace(0, 360, 180)
    dec_grid = np.linspace(-90, 90, 90)
    ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
    
    # Interpolate star counts to create smooth background
    points = np.column_stack([grouped['ra'], grouped['dec']])
    star_values = grouped['star_mean'].values
    grid_stars = griddata(points, star_values, (ra_mesh, dec_mesh), method='linear', fill_value=0)
    
    # Define colormap parameters
    vmin = 0
    vmax = np.percentile(grouped['star_mean'], 95)
    
    # Convert grid to radians for Mollweide
    ra_mesh_rad = np.radians(ra_mesh - 180)
    dec_mesh_rad = np.radians(dec_mesh)
    
    # Plot background color mesh
    mesh = ax.pcolormesh(ra_mesh_rad, dec_mesh_rad, grid_stars,
                         cmap='viridis', shading='auto',
                         vmin=vmin, vmax=vmax, alpha=0.8)
    
    # Convert pointing positions to radians
    ra_rad = np.radians(grouped['ra'] - 180)
    dec_rad = np.radians(grouped['dec'])
    
    # Add white dots for all pointings
    ax.scatter(ra_rad, dec_rad, c='white', s=1, alpha=0.3, zorder=2)
    
    # Mark failed detections with red X
    failed = grouped[grouped['star_mean'] == 0]
    if len(failed) > 0:
        ax.scatter(np.radians(failed['ra'] - 180), 
                  np.radians(failed['dec']),
                  c='red', s=8, marker='x', alpha=0.8, zorder=3, label='No stars')
    
    # Mark low precision with orange X
    low_precision = grouped[(grouped['pixel_error_mean'] > precision_threshold) & (grouped['star_mean'] > 0)]
    if len(low_precision) > 0:
        ax.scatter(np.radians(low_precision['ra'] - 180),
                  np.radians(low_precision['dec']),
                  c='orange', s=8, marker='x', alpha=0.8, zorder=3, label=f'Error > {precision_threshold}px')
    
    ax.set_title(f'Sky Coverage - Mollweide Projection{title_suffix}\n{len(grouped)} Unique Pointings', fontsize=14)
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    
    # Add legend if there are marked points
    if len(failed) > 0 or len(low_precision) > 0:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved Mollweide sky coverage to {output_path}")
    else:
        plt.show()
    
    plt.close(fig)
    return fig


def plot_sky_coverage_rectangular(df, output_path=None, exposure_ms=None, precision_threshold=0.1):
    """Plot sky coverage on rectangular projection with color background"""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata
    
    # Filter for specific exposure if requested
    if exposure_ms is not None:
        df_filtered = df[df['exposure_ms'] == exposure_ms].copy()
        title_suffix = f" @ {exposure_ms}ms"
        if len(df_filtered) == 0:
            print(f"No data found for exposure {exposure_ms}ms")
            print(f"Available exposures: {sorted(df['exposure_ms'].unique())}")
            return None
        df = df_filtered
    else:
        title_suffix = " (all exposures)"
    
    # Get unique pointings with their mean star counts and precision
    grouped = df.groupby(['ra', 'dec']).agg({
        'star_count': ['mean', 'count'],
        'pixel_error': 'mean'
    }).reset_index()
    grouped.columns = ['ra', 'dec', 'star_mean', 'star_count', 'pixel_error_mean']
    
    if len(grouped) == 0:
        print("No grouped data available after filtering")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    
    # Create background interpolation
    ra_grid = np.linspace(0, 360, 180)
    dec_grid = np.linspace(-90, 90, 90)
    ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
    
    # Interpolate star counts to create smooth background
    points = np.column_stack([grouped['ra'], grouped['dec']])
    star_values = grouped['star_mean'].values
    grid_stars = griddata(points, star_values, (ra_mesh, dec_mesh), method='linear', fill_value=0)
    
    # Define colormap parameters
    vmin = 0
    vmax = np.percentile(grouped['star_mean'], 95)
    
    # Plot background color mesh
    im = ax.pcolormesh(ra_mesh, dec_mesh, grid_stars,
                       cmap='viridis', shading='auto',
                       vmin=vmin, vmax=vmax, alpha=0.8)
    
    # Add white dots for all pointings
    ax.scatter(grouped['ra'], grouped['dec'], c='white', s=1, alpha=0.3, zorder=2)
    
    # Mark failed detections with red X
    failed = grouped[grouped['star_mean'] == 0]
    if len(failed) > 0:
        ax.scatter(failed['ra'], failed['dec'], 
                  c='red', s=8, marker='x', alpha=0.8, zorder=3)
    
    # Mark low precision with orange X
    low_precision = grouped[(grouped['pixel_error_mean'] > precision_threshold) & (grouped['star_mean'] > 0)]
    if len(low_precision) > 0:
        ax.scatter(low_precision['ra'], low_precision['dec'],
                  c='orange', s=8, marker='x', alpha=0.8, zorder=3)
    
    ax.set_title(f'Sky Coverage - Rectangular Projection{title_suffix}\n{len(grouped)} Unique Pointings', fontsize=14)
    ax.set_xlabel('Right Ascension (degrees)')
    ax.set_ylabel('Declination (degrees)')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=6, label='Pointing', linestyle='None'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=8, label='No stars detected', linestyle='None'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='orange', markersize=8, label=f'Precision > {precision_threshold} px', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved rectangular sky coverage to {output_path}")
    else:
        plt.show()
    
    plt.close(fig)
    return fig


def plot_sky_coverage_heatmap(df, output_path=None, exposure_ms=None, precision_threshold=0.1):
    """Plot both sky coverage projections for each exposure value"""
    from pathlib import Path
    
    # If exposure_ms is specified, generate just for that exposure
    if exposure_ms is not None:
        # Generate separate output paths if one is provided
        if output_path:
            path = Path(output_path)
            base = path.stem
            suffix = path.suffix
            parent = path.parent
            
            moll_path = str(parent / f"{base}_mollweide{suffix}")
            rect_path = str(parent / f"{base}_rectangular{suffix}")
        else:
            moll_path = None
            rect_path = None
        
        # Generate both plots
        fig_moll = plot_sky_coverage_mollweide(df, moll_path, exposure_ms, precision_threshold)
        fig_rect = plot_sky_coverage_rectangular(df, rect_path, exposure_ms, precision_threshold)
        
        return fig_moll, fig_rect
    
    # Otherwise, generate for all exposures
    exposures = sorted(df['exposure_ms'].unique())
    print(f"Generating sky coverage plots for {len(exposures)} exposures: {exposures}")
    
    figures = []
    
    for exp_ms in exposures:
        # Generate separate output paths if one is provided
        if output_path:
            path = Path(output_path)
            base = path.stem.replace('_heatmap', '')
            suffix = path.suffix
            parent = path.parent
            
            moll_path = str(parent / f"{base}_{int(exp_ms)}ms_mollweide{suffix}")
            rect_path = str(parent / f"{base}_{int(exp_ms)}ms_rectangular{suffix}")
        else:
            moll_path = None
            rect_path = None
        
        # Generate both plots for this exposure
        print(f"Processing exposure: {exp_ms}ms")
        fig_moll = plot_sky_coverage_mollweide(df, moll_path, exp_ms, precision_threshold)
        fig_rect = plot_sky_coverage_rectangular(df, rect_path, exp_ms, precision_threshold)
        
        figures.append((exp_ms, fig_moll, fig_rect))
    
    print(f"Created {len(exposures) * 2} sky coverage plots ({len(exposures)} exposures × 2 projections)")
    
    return figures


def plot_ra_dec_coverage(df, output_path=None, failed_exposure_ms=250):
    """Plot RA/Dec coverage map with star detection success"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get data for specified exposure
    exp_data = df[df['exposure_ms'] == failed_exposure_ms].copy()
    
    if len(exp_data) == 0:
        print(f"No data found for exposure {failed_exposure_ms}ms")
        # Try to use all data instead
        exp_data = df.copy()
        title_suffix = "(All exposures)"
    else:
        title_suffix = f"({failed_exposure_ms}ms exposure)"
    
    # Categorize experiments
    exp_data['detection_category'] = 'Unknown'
    exp_data.loc[exp_data['star_count'] == 0, 'detection_category'] = 'No stars detected'
    exp_data.loc[exp_data['star_count'] >= 1, 'detection_category'] = 'Stars detected'
    
    # Plot different categories with different colors and markers
    categories = {
        'Stars detected': {'color': 'green', 'marker': 'o', 'size': 20, 'alpha': 0.6},
        'No stars detected': {'color': 'red', 'marker': 'x', 'size': 40, 'alpha': 0.8},
    }
    
    for category, style in categories.items():
        cat_data = exp_data[exp_data['detection_category'] == category]
        if len(cat_data) > 0:
            ax.scatter(cat_data['ra'], cat_data['dec'], 
                      c=style['color'], marker=style['marker'], 
                      s=style['size'], alpha=style['alpha'],
                      label=f'{category} (n={len(cat_data)})')
    
    # Add galactic plane approximation (simplified)
    # This is a very rough approximation
    ra_galactic = np.linspace(0, 360, 100)
    dec_galactic = 5 * np.sin(np.radians(ra_galactic - 266))  # Rough galactic plane
    ax.plot(ra_galactic, dec_galactic, 'b--', alpha=0.3, linewidth=1, label='Approx. Galactic Plane')
    
    # Formatting
    ax.set_xlabel('Right Ascension (degrees)', fontsize=12)
    ax.set_ylabel('Declination (degrees)', fontsize=12)
    ax.set_title(f'Sky Coverage and Detection Success {title_suffix}', fontsize=14)
    
    # Set axis limits
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)  # Celestial equator
    ax.axhline(y=23.44, color='orange', linestyle=':', linewidth=0.5, alpha=0.5, label='Ecliptic bounds')
    ax.axhline(y=-23.44, color='orange', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Add statistics text
    total_experiments = len(exp_data)
    failed_experiments = len(exp_data[exp_data['star_count'] == 0])
    success_rate = ((total_experiments - failed_experiments) / total_experiments * 100) if total_experiments > 0 else 0
    
    stats_text = f"Total pointings: {total_experiments}\n"
    stats_text += f"Failed detections: {failed_experiments}\n"
    stats_text += f"Success rate: {success_rate:.2f}%"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved RA/Dec coverage plot to {output_path}")
    else:
        plt.show()
    
    return fig, ax


# ============================================================================
# Star Count Extremes
# ============================================================================

def print_star_count_extremes(df, n=5):
    """Print the top N and bottom N sky locations by mean star count"""
    
    # Group by unique sky position (RA, Dec)
    # Round to avoid floating point comparison issues
    df['ra_rounded'] = df['ra'].round(4)
    df['dec_rounded'] = df['dec'].round(4)
    
    # Calculate mean star count for each position
    position_stats = df.groupby(['ra_rounded', 'dec_rounded'])['star_count'].agg([
        ('mean', 'mean'),
        ('count', 'size')
    ]).reset_index()
    
    # Sort by mean star count
    position_stats = position_stats.sort_values('mean', ascending=False)
    
    print("\n" + "="*60)
    print(f"Top {n} sky locations with HIGHEST star counts:")
    print("="*60)
    print(f"{'RA':>10} {'Dec':>10} {'Mean Stars':>12} {'N Obs':>8}")
    print("-"*40)
    
    for idx, row in position_stats.head(n).iterrows():
        print(f"{row['ra_rounded']:>10.4f} {row['dec_rounded']:>10.4f} {row['mean']:>12.2f} {row['count']:>8}")
    
    print("\n" + "="*60)
    print(f"Top {n} sky locations with LOWEST star counts:")
    print("="*60)
    print(f"{'RA':>10} {'Dec':>10} {'Mean Stars':>12} {'N Obs':>8}")
    print("-"*40)
    
    for idx, row in position_stats.tail(n).iterrows():
        print(f"{row['ra_rounded']:>10.4f} {row['dec_rounded']:>10.4f} {row['mean']:>12.2f} {row['count']:>8}")
    
    print("="*60)


# ============================================================================
# ICP Translation and Brightest Star Offset Analysis
# ============================================================================

def plot_icp_translations(df, output_path=None):
    """Plot histograms of ICP translation_x and translation_y for each exposure"""
    # Filter out NaN values
    valid_mask = ~(df['translation_x'].isna() | df['translation_y'].isna())
    valid_data = df[valid_mask]
    
    # Get unique exposure values
    exposures = sorted(valid_data['exposure_ms'].unique())
    
    # Generate output files for each exposure
    if output_path:
        base_path = Path(output_path)
        parent = base_path.parent
        stem = base_path.stem.replace('_icp', '')
        suffix = base_path.suffix
    
    for exposure_ms in exposures:
        exp_data = valid_data[valid_data['exposure_ms'] == exposure_ms]
        
        if len(exp_data) == 0:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Translation X histogram
        ax = axes[0]
        
        # Count outliers outside ±HISTOGRAM_BOUND pixel range
        outliers_x_below = (exp_data['translation_x'] < -HISTOGRAM_BOUND).sum()
        outliers_x_above = (exp_data['translation_x'] > HISTOGRAM_BOUND).sum()
        total_outliers_x = outliers_x_below + outliers_x_above
        
        n, bins, patches = ax.hist(exp_data['translation_x'], bins=50,
                                   range=(-HISTOGRAM_BOUND, HISTOGRAM_BOUND), color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
        ax.set_xlabel('Translation X (pixels)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('ICP Translation X Distribution', fontsize=14)
        ax.set_xlim(-HISTOGRAM_BOUND, HISTOGRAM_BOUND)
        ax.grid(True, alpha=0.3)
        
        # Calculate statistics excluding outliers (only data within ±HISTOGRAM_BOUND)
        x_filtered = exp_data['translation_x'][
            (exp_data['translation_x'] >= -HISTOGRAM_BOUND) & 
            (exp_data['translation_x'] <= HISTOGRAM_BOUND)
        ]
        mean_x = x_filtered.mean() if len(x_filtered) > 0 else np.nan
        std_x = x_filtered.std() if len(x_filtered) > 0 else np.nan
        median_x = x_filtered.median() if len(x_filtered) > 0 else np.nan
        
        # Only show lines if they're within the plot range
        if -HISTOGRAM_BOUND <= mean_x <= HISTOGRAM_BOUND:
            ax.axvline(mean_x, color='green', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_x:.4f}')
        if -HISTOGRAM_BOUND <= median_x <= HISTOGRAM_BOUND:
            ax.axvline(median_x, color='orange', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Median: {median_x:.4f}')
        
        # Stats text with outlier info (stats computed without outliers)
        n_filtered = len(x_filtered) if 'x_filtered' in locals() else 0
        stats_text = f'n={len(exp_data)} ({n_filtered} shown)\nμ={mean_x:.4f}\nσ={std_x:.4f}\nmedian={median_x:.4f}'
        if total_outliers_x > 0:
            stats_text += f'\n\nOutliers: {total_outliers_x}'
            if outliers_x_below > 0:
                stats_text += f'\n  < -{HISTOGRAM_BOUND}: {outliers_x_below}'
            if outliers_x_above > 0:
                stats_text += f'\n  > {HISTOGRAM_BOUND}: {outliers_x_above}'
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(loc='upper left', fontsize=10)
        
        # Translation Y histogram  
        ax = axes[1]
        
        # Count outliers outside ±HISTOGRAM_BOUND pixel range
        outliers_y_below = (exp_data['translation_y'] < -HISTOGRAM_BOUND).sum()
        outliers_y_above = (exp_data['translation_y'] > HISTOGRAM_BOUND).sum()
        total_outliers_y = outliers_y_below + outliers_y_above
        
        n, bins, patches = ax.hist(exp_data['translation_y'], bins=50,
                                   range=(-HISTOGRAM_BOUND, HISTOGRAM_BOUND), color='darkgreen', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
        ax.set_xlabel('Translation Y (pixels)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('ICP Translation Y Distribution', fontsize=14)
        ax.set_xlim(-HISTOGRAM_BOUND, HISTOGRAM_BOUND)
        ax.grid(True, alpha=0.3)
        
        # Calculate statistics excluding outliers (only data within ±HISTOGRAM_BOUND)
        y_filtered = exp_data['translation_y'][
            (exp_data['translation_y'] >= -HISTOGRAM_BOUND) & 
            (exp_data['translation_y'] <= HISTOGRAM_BOUND)
        ]
        mean_y = y_filtered.mean() if len(y_filtered) > 0 else np.nan
        std_y = y_filtered.std() if len(y_filtered) > 0 else np.nan
        median_y = y_filtered.median() if len(y_filtered) > 0 else np.nan
        
        # Only show lines if they're within the plot range
        if -HISTOGRAM_BOUND <= mean_y <= HISTOGRAM_BOUND:
            ax.axvline(mean_y, color='green', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_y:.4f}')
        if -HISTOGRAM_BOUND <= median_y <= HISTOGRAM_BOUND:
            ax.axvline(median_y, color='orange', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Median: {median_y:.4f}')
        
        # Stats text with outlier info (stats computed without outliers)
        n_filtered = len(y_filtered) if 'y_filtered' in locals() else 0
        stats_text = f'n={len(exp_data)} ({n_filtered} shown)\nμ={mean_y:.4f}\nσ={std_y:.4f}\nmedian={median_y:.4f}'
        if total_outliers_y > 0:
            stats_text += f'\n\nOutliers: {total_outliers_y}'
            if outliers_y_below > 0:
                stats_text += f'\n  < -{HISTOGRAM_BOUND}: {outliers_y_below}'
            if outliers_y_above > 0:
                stats_text += f'\n  > {HISTOGRAM_BOUND}: {outliers_y_above}'
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(loc='upper left', fontsize=10)
        
        plt.suptitle(f'ICP Translation Components - {exposure_ms}ms Exposure', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if output_path:
            exp_output = str(parent / f"{stem}_icp_{int(exposure_ms)}ms{suffix}")
            plt.savefig(exp_output, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {exp_output}")
        else:
            plt.show()
        
        plt.close(fig)
    
    print(f"Created {len(exposures)} ICP translation plots for exposures: {exposures}")
    return None


def plot_trial_error_stddev(df, output_path=None):
    """Plot standard deviation of pixel errors across trials vs brightest magnitude
    
    This aggregates multiple trials for the same pointing (same experiment_num) 
    and calculates the standard deviation of the pixel errors across trials.
    """
    # Check if we have trial_num column (new format)
    if 'trial_num' not in df.columns:
        print("Warning: No trial_num column found. Assuming single trial per experiment.")
        df['trial_num'] = 0
    
    # Get unique sensors and f-numbers from the data
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    
    # Create figure with subplots - one for each sensor  
    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors, figsize=(10*n_sensors, 8))
    
    # Make axes iterable even for single sensor
    if n_sensors == 1:
        axes = [axes]
    
    # Use consistent colormap
    cmap = cm.viridis
    
    for idx, (sensor_name, ax) in enumerate(zip(sensors, axes)):
        legend_elements = []
        
        for f_num in f_numbers:
            color = get_f_number_color(f_num, cmap)
            
            # Filter data for this sensor and f-number
            mask = (df['sensor'] == sensor_name) & (df['f_number'] == f_num)
            sensor_data = df[mask]
            
            if len(sensor_data) == 0:
                continue
            
            # Group by experiment_num and exposure_ms to aggregate trials
            grouped = sensor_data.groupby(['experiment_num', 'exposure_ms'])
            
            brightest_mags = []
            pixel_error_stds = []
            
            for (exp_num, exp_ms), group in grouped:
                # Only compute std if we have multiple trials
                if len(group) > 1:
                    # Get brightest magnitude (should be same for all trials in this group)
                    brightest_mag = group['brightest_mag'].iloc[0]
                    
                    # Calculate standard deviation of pixel errors across trials
                    pixel_error_std = group['pixel_error'].std()
                    
                    # Skip if NaN
                    if not np.isnan(brightest_mag) and not np.isnan(pixel_error_std):
                        brightest_mags.append(brightest_mag)
                        pixel_error_stds.append(pixel_error_std)
            
            # Plot if we have data
            if len(brightest_mags) > 0:
                # Sort by brightest magnitude for cleaner plot
                sorted_indices = np.argsort(brightest_mags)
                brightest_mags = np.array(brightest_mags)[sorted_indices]
                pixel_error_stds = np.array(pixel_error_stds)[sorted_indices]
                
                # Plot with low opacity to show overlapping points better
                ax.scatter(brightest_mags, pixel_error_stds, 
                          color=color, alpha=0.1, s=30,
                          label=f'f/{f_num:.2f}')
                
                # Optional: add trend line
                if len(brightest_mags) > 3:
                    z = np.polyfit(brightest_mags, pixel_error_stds, 2)
                    p = np.poly1d(z)
                    x_trend = np.linspace(brightest_mags.min(), brightest_mags.max(), 100)
                    ax.plot(x_trend, p(x_trend), color=color, alpha=0.3, linestyle='--', linewidth=1)
        
        # Formatting
        ax.set_xlabel('Brightest Star Magnitude', fontsize=12)
        ax.set_ylabel('Std Dev of Pixel Error Across Trials', fontsize=12)
        ax.set_title(f'{sensor_name} - Trial-to-Trial Error Consistency', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # Set reasonable y-axis limits
        ax.set_ylim(bottom=0)
        
        # Invert x-axis (brighter stars have lower magnitude)
        ax.invert_xaxis()
    
    plt.suptitle('Standard Deviation of Errors Across Trials vs Brightest Star', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Trial stddev plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


def plot_brightest_magnitude_distribution(df, output_path=None):
    """Plot distribution of brightest star in scene across all sky pointings
    
    Uses only the longest exposure time to show the distribution of the brightest
    star magnitude available in each scene/pointing across the full sky survey.
    """
    # Get longest exposure time
    longest_exposure = df['exposure_ms'].max()
    
    # Filter for longest exposure only
    longest_exp_data = df[df['exposure_ms'] == longest_exposure]
    
    # Get unique brightest magnitudes (one per experiment_num since all trials have same catalog stars)
    unique_pointings = longest_exp_data.groupby('experiment_num')['brightest_mag'].first().dropna()
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(unique_pointings, bins=50, 
                              color='steelblue', alpha=0.7, 
                              edgecolor='black', linewidth=0.5)
    
    # Add statistics
    mean_mag = unique_pointings.mean()
    median_mag = unique_pointings.median()
    std_mag = unique_pointings.std()
    min_mag = unique_pointings.min()
    max_mag = unique_pointings.max()
    
    # Add vertical lines for mean and median
    ax.axvline(mean_mag, color='red', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'Mean: {mean_mag:.2f}')
    ax.axvline(median_mag, color='green', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'Median: {median_mag:.2f}')
    
    # Add text with statistics
    stats_text = (f'n={len(unique_pointings)} pointings\n'
                 f'μ={mean_mag:.2f}\n'
                 f'σ={std_mag:.2f}\n'
                 f'Range: {min_mag:.2f} - {max_mag:.2f}')
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels and title
    ax.set_xlabel('Brightest Star Magnitude in Scene', fontsize=12)
    ax.set_ylabel('Number of Sky Pointings', fontsize=12)
    ax.set_title(f'Distribution of Brightest Star in Scene Across Sky Survey\n(Using {longest_exposure}ms exposures)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add annotation explaining what this shows
    annotation_text = ("Each pointing represents a unique sky location.\n"
                      "Shows the brightest star available in the star catalog\n"
                      "for each field of view across the survey.")
    ax.text(0.02, 0.02, annotation_text, transform=ax.transAxes,
           fontsize=9, va='bottom', ha='left', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Brightest star in scene distribution plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax


def plot_brightest_magnitude_distribution_by_fnumber(df, output_path=None):
    """Plot distribution of brightest star magnitude for each f-number separately"""
    # Get longest exposure time
    longest_exposure = df['exposure_ms'].max()
    
    # Filter for longest exposure only
    longest_exp_data = df[df['exposure_ms'] == longest_exposure]
    
    # Get unique f-numbers
    f_numbers = sorted(longest_exp_data['f_number'].unique())
    
    # Use colormap for f-numbers
    cmap = cm.viridis
    
    # Generate output files for each f-number
    if output_path:
        base_path = Path(output_path)
        parent = base_path.parent
        stem = base_path.stem.replace('_brightest_dist_f', '')
        suffix = base_path.suffix
    
    plot_files = []
    
    for f_num in f_numbers:
        # Filter data for this f-number
        f_data = longest_exp_data[longest_exp_data['f_number'] == f_num]
        
        if len(f_data) == 0:
            continue
            
        # Get unique brightest magnitudes (one per experiment_num)
        unique_pointings = f_data.groupby('experiment_num')['brightest_mag'].first().dropna()
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        # Get color for this f-number
        f_color = get_f_number_color(f_num, cmap)
        
        # Create histogram
        n, bins, patches = ax.hist(unique_pointings, bins=30, 
                                  color=f_color, alpha=0.7, 
                                  edgecolor='black', linewidth=0.5,
                                  range=(8, 16))
        
        # Add statistics
        mean_mag = unique_pointings.mean()
        median_mag = unique_pointings.median()
        std_mag = unique_pointings.std()
        min_mag = unique_pointings.min()
        max_mag = unique_pointings.max()
        
        # Add vertical lines for mean and median
        ax.axvline(mean_mag, color='red', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Mean: {mean_mag:.2f}')
        ax.axvline(median_mag, color='green', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Median: {median_mag:.2f}')
        
        # Add text with statistics
        stats_text = (f'f/{f_num:.2f}\n'
                     f'n={len(unique_pointings)} pointings\n'
                     f'μ={mean_mag:.2f}\n'
                     f'σ={std_mag:.2f}\n'
                     f'Range: {min_mag:.2f} - {max_mag:.2f}')
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=11, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Labels and title
        ax.set_xlabel('Brightest Star Magnitude in Scene', fontsize=12)
        ax.set_ylabel('Number of Sky Pointings', fontsize=12)
        ax.set_title(f'Distribution of Brightest Star - f/{f_num:.2f}\n({longest_exposure}ms exposure)', fontsize=14)
        ax.set_xlim(8, 16)
        ax.set_ylim(0, 350)  # Fixed y-axis for consistent animation
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        if output_path:
            f_output = str(parent / f"{stem}_brightest_dist_f{f_num:.2f}{suffix}")
            plt.savefig(f_output, dpi=300, bbox_inches='tight')
            print(f"Brightest star distribution for f/{f_num:.2f} saved to: {f_output}")
            plot_files.append(f_output)
        else:
            plt.show()
        
        plt.close(fig)
    
    print(f"Generated {len(plot_files)} distribution plots for f-numbers")
    
    # Create animation if plots were saved
    if output_path and len(plot_files) > 0:
        import subprocess
        animation_path = str(parent / f"{stem}_brightest_dist_animation.avi")
        
        # Build mencoder command
        pattern = str(parent / f"{stem}_brightest_dist_f*.png")
        cmd = [
            'mencoder',
            f'mf://{pattern}',
            '-mf', 'fps=2:type=png',
            '-ovc', 'lavc',
            '-lavcopts', 'vcodec=mpeg4:vbitrate=2000',
            '-oac', 'copy',
            '-o', animation_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(parent))
            if result.returncode == 0:
                print(f"Animation saved to: {animation_path}")
            else:
                print(f"mencoder error: {result.stderr}")
        except FileNotFoundError:
            print("mencoder not found - animation not created. Install with: sudo apt-get install mencoder")
    
    return plot_files


def plot_brightest_star_error_vs_magnitude(df, output_path=None):
    """Plot brightest star position error vs magnitude for longest exposure - one plot per f-number"""
    # Get longest exposure
    longest_exposure = df['exposure_ms'].max()
    
    # Filter to longest exposure only
    long_exp_data = df[df['exposure_ms'] == longest_exposure].copy()
    
    # Calculate total position error in pixels
    long_exp_data['brightest_star_error'] = np.sqrt(
        long_exp_data['brightest_star_dx']**2 + 
        long_exp_data['brightest_star_dy']**2
    )
    
    # Filter out NaN values
    valid_mask = ~(long_exp_data['brightest_star_error'].isna() | long_exp_data['brightest_mag'].isna())
    valid_data = long_exp_data[valid_mask]
    
    if len(valid_data) == 0:
        print(f"No valid data for error vs magnitude plot at {longest_exposure}ms exposure")
        return None, None
    
    # Get unique sensors and f-numbers
    sensors = sorted(valid_data['sensor'].unique())
    f_numbers = sorted(valid_data['f_number'].unique())
    
    # Use colormap for f-numbers
    cmap = cm.viridis
    sensor_colors = {'IMX455': 'blue', 'HWK4123': 'red'}
    
    # Generate output files for each f-number
    if output_path:
        base_path = Path(output_path)
        parent = base_path.parent
        stem = base_path.stem.replace('_brightest_error_mag', '')
        suffix = base_path.suffix
    
    # Create a plot for each f-number
    for f_num in f_numbers:
        f_data = valid_data[valid_data['f_number'] == f_num]
        
        if len(f_data) == 0:
            continue
        
        # Create figure with single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        # Get color for this f-number
        f_color = get_f_number_color(f_num, cmap)
        
        # Plot all data points for this f-number
        ax.scatter(f_data['brightest_mag'], f_data['brightest_star_error'],
                   alpha=0.3, s=4, color=f_color)
        
        ax.set_xlabel('Brightest Star Magnitude', fontsize=12)
        ax.set_ylabel('Position Error (pixels)', fontsize=12)
        ax.set_title(f'Brightest Star Error vs Magnitude - f/{f_num:.2f}\n{longest_exposure}ms exposure', fontsize=14)
        ax.set_xlim(8, 16)
        ax.set_ylim(0, 0.2)
        ax.grid(True, alpha=0.3)
        
        # Add statistics for this f-number
        mean_error_f = f_data['brightest_star_error'].mean()
        median_error_f = f_data['brightest_star_error'].median()
        std_error_f = f_data['brightest_star_error'].std()
        
        stats_text_f = (f'Statistics:\n'
                       f'Mean: {mean_error_f:.3f} px\n'
                       f'Median: {median_error_f:.3f} px\n'
                       f'Std: {std_error_f:.3f} px\n'
                       f'n={len(f_data)} detections')
        
        ax.text(0.98, 0.98, stats_text_f, transform=ax.transAxes,
                fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            f_output = str(parent / f"{stem}_f{f_num:.2f}{suffix}")
            plt.savefig(f_output, dpi=300, bbox_inches='tight')
            print(f"Brightest star error vs magnitude plot for f/{f_num:.2f} saved to: {f_output}")
        else:
            plt.show()
        
        plt.close(fig)
    
    print(f"Generated {len(f_numbers)} plots for f-numbers: {[f'f/{f:.2f}' for f in f_numbers]}")
    return None, None


def plot_brightest_star_offsets(df, output_path=None):
    """Plot histograms of brightest star dx and dy offsets for each exposure"""
    # Filter out NaN values
    valid_mask = ~(df['brightest_star_dx'].isna() | df['brightest_star_dy'].isna())
    valid_data = df[valid_mask]
    
    # Get unique exposure values
    exposures = sorted(valid_data['exposure_ms'].unique())
    
    # Generate output files for each exposure
    if output_path:
        base_path = Path(output_path)
        parent = base_path.parent
        stem = base_path.stem.replace('_brightest_star', '')
        suffix = base_path.suffix
    
    for exposure_ms in exposures:
        exp_data = valid_data[valid_data['exposure_ms'] == exposure_ms]
        
        if len(exp_data) == 0:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Brightest star dx histogram
        ax = axes[0]
        
        # Count outliers outside ±HISTOGRAM_BOUND pixel range
        outliers_dx_below = (exp_data['brightest_star_dx'] < -HISTOGRAM_BOUND).sum()
        outliers_dx_above = (exp_data['brightest_star_dx'] > HISTOGRAM_BOUND).sum()
        total_outliers_dx = outliers_dx_below + outliers_dx_above
        
        n, bins, patches = ax.hist(exp_data['brightest_star_dx'], bins=50,
                                   range=(-HISTOGRAM_BOUND, HISTOGRAM_BOUND), color='coral', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
        ax.set_xlabel('Brightest Star ΔX (pixels)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Brightest Star X Offset Distribution', fontsize=14)
        ax.set_xlim(-HISTOGRAM_BOUND, HISTOGRAM_BOUND)
        ax.grid(True, alpha=0.3)
        
        # Calculate statistics excluding outliers (only data within ±HISTOGRAM_BOUND)
        dx_filtered = exp_data['brightest_star_dx'][
            (exp_data['brightest_star_dx'] >= -HISTOGRAM_BOUND) & 
            (exp_data['brightest_star_dx'] <= HISTOGRAM_BOUND)
        ]
        mean_dx = dx_filtered.mean() if len(dx_filtered) > 0 else np.nan
        std_dx = dx_filtered.std() if len(dx_filtered) > 0 else np.nan
        median_dx = dx_filtered.median() if len(dx_filtered) > 0 else np.nan
        
        # Only show lines if they're within the plot range
        if -HISTOGRAM_BOUND <= mean_dx <= HISTOGRAM_BOUND:
            ax.axvline(mean_dx, color='green', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_dx:.4f}')
        if -HISTOGRAM_BOUND <= median_dx <= HISTOGRAM_BOUND:
            ax.axvline(median_dx, color='orange', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Median: {median_dx:.4f}')
        
        # Stats text with outlier info (stats computed without outliers)
        n_filtered = len(dx_filtered) if 'dx_filtered' in locals() else 0
        stats_text = f'n={len(exp_data)} ({n_filtered} shown)\nμ={mean_dx:.4f}\nσ={std_dx:.4f}\nmedian={median_dx:.4f}'
        if total_outliers_dx > 0:
            stats_text += f'\n\nOutliers: {total_outliers_dx}'
            if outliers_dx_below > 0:
                stats_text += f'\n  < -{HISTOGRAM_BOUND}: {outliers_dx_below}'
            if outliers_dx_above > 0:
                stats_text += f'\n  > {HISTOGRAM_BOUND}: {outliers_dx_above}'
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(loc='upper left', fontsize=10)
        
        # Brightest star dy histogram
        ax = axes[1]
        
        # Count outliers outside ±HISTOGRAM_BOUND pixel range
        outliers_dy_below = (exp_data['brightest_star_dy'] < -HISTOGRAM_BOUND).sum()
        outliers_dy_above = (exp_data['brightest_star_dy'] > HISTOGRAM_BOUND).sum()
        total_outliers_dy = outliers_dy_below + outliers_dy_above
        
        n, bins, patches = ax.hist(exp_data['brightest_star_dy'], bins=50,
                                   range=(-HISTOGRAM_BOUND, HISTOGRAM_BOUND), color='mediumpurple', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
        ax.set_xlabel('Brightest Star ΔY (pixels)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Brightest Star Y Offset Distribution', fontsize=14)
        ax.set_xlim(-HISTOGRAM_BOUND, HISTOGRAM_BOUND)
        ax.grid(True, alpha=0.3)
        
        # Calculate statistics excluding outliers (only data within ±HISTOGRAM_BOUND)
        dy_filtered = exp_data['brightest_star_dy'][
            (exp_data['brightest_star_dy'] >= -HISTOGRAM_BOUND) & 
            (exp_data['brightest_star_dy'] <= HISTOGRAM_BOUND)
        ]
        mean_dy = dy_filtered.mean() if len(dy_filtered) > 0 else np.nan
        std_dy = dy_filtered.std() if len(dy_filtered) > 0 else np.nan
        median_dy = dy_filtered.median() if len(dy_filtered) > 0 else np.nan
        
        # Only show lines if they're within the plot range
        if -HISTOGRAM_BOUND <= mean_dy <= HISTOGRAM_BOUND:
            ax.axvline(mean_dy, color='green', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_dy:.4f}')
        if -HISTOGRAM_BOUND <= median_dy <= HISTOGRAM_BOUND:
            ax.axvline(median_dy, color='orange', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Median: {median_dy:.4f}')
        
        # Stats text with outlier info (stats computed without outliers)
        n_filtered = len(dy_filtered) if 'dy_filtered' in locals() else 0
        stats_text = f'n={len(exp_data)} ({n_filtered} shown)\nμ={mean_dy:.4f}\nσ={std_dy:.4f}\nmedian={median_dy:.4f}'
        if total_outliers_dy > 0:
            stats_text += f'\n\nOutliers: {total_outliers_dy}'
            if outliers_dy_below > 0:
                stats_text += f'\n  < -{HISTOGRAM_BOUND}: {outliers_dy_below}'
            if outliers_dy_above > 0:
                stats_text += f'\n  > {HISTOGRAM_BOUND}: {outliers_dy_above}'
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(loc='upper left', fontsize=10)
        
        plt.suptitle(f'Brightest Star Offset Distribution - {exposure_ms}ms Exposure', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if output_path:
            exp_output = str(parent / f"{stem}_brightest_star_{int(exposure_ms)}ms{suffix}")
            plt.savefig(exp_output, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {exp_output}")
        else:
            plt.show()
        
        plt.close(fig)
    
    print(f"Created {len(exposures)} brightest star offset plots for exposures: {exposures}")
    return None


# ============================================================================
# Statistics Summary Table
# ============================================================================

def compute_statistics_table(df, output_path=None):
    """Compute statistics table for ICP and brightest star offsets, excluding outliers"""
    # Filter out NaN values
    valid_icp = df[~(df['translation_x'].isna() | df['translation_y'].isna())].copy()
    valid_bright = df[~(df['brightest_star_dx'].isna() | df['brightest_star_dy'].isna())].copy()
    
    # Get unique exposure values
    exposures = sorted(df['exposure_ms'].unique())
    
    # Prepare results
    results = []
    
    for exposure_ms in exposures:
        # Get data for this exposure
        icp_exp = valid_icp[valid_icp['exposure_ms'] == exposure_ms]
        bright_exp = valid_bright[valid_bright['exposure_ms'] == exposure_ms]
        
        # Filter outliers (keep only data within ±HISTOGRAM_BOUND)
        icp_x_filtered = icp_exp['translation_x'][
            (icp_exp['translation_x'] >= -HISTOGRAM_BOUND) & 
            (icp_exp['translation_x'] <= HISTOGRAM_BOUND)
        ]
        icp_y_filtered = icp_exp['translation_y'][
            (icp_exp['translation_y'] >= -HISTOGRAM_BOUND) & 
            (icp_exp['translation_y'] <= HISTOGRAM_BOUND)
        ]
        bright_x_filtered = bright_exp['brightest_star_dx'][
            (bright_exp['brightest_star_dx'] >= -HISTOGRAM_BOUND) & 
            (bright_exp['brightest_star_dx'] <= HISTOGRAM_BOUND)
        ]
        bright_y_filtered = bright_exp['brightest_star_dy'][
            (bright_exp['brightest_star_dy'] >= -HISTOGRAM_BOUND) & 
            (bright_exp['brightest_star_dy'] <= HISTOGRAM_BOUND)
        ]
        
        # Compute statistics
        row = {
            'exposure_ms': exposure_ms,
            'icp_x_mean': icp_x_filtered.mean() if len(icp_x_filtered) > 0 else np.nan,
            'icp_x_std': icp_x_filtered.std() if len(icp_x_filtered) > 0 else np.nan,
            'icp_x_n': len(icp_x_filtered),
            'icp_y_mean': icp_y_filtered.mean() if len(icp_y_filtered) > 0 else np.nan,
            'icp_y_std': icp_y_filtered.std() if len(icp_y_filtered) > 0 else np.nan,
            'icp_y_n': len(icp_y_filtered),
            'bright_x_mean': bright_x_filtered.mean() if len(bright_x_filtered) > 0 else np.nan,
            'bright_x_std': bright_x_filtered.std() if len(bright_x_filtered) > 0 else np.nan,
            'bright_x_n': len(bright_x_filtered),
            'bright_y_mean': bright_y_filtered.mean() if len(bright_y_filtered) > 0 else np.nan,
            'bright_y_std': bright_y_filtered.std() if len(bright_y_filtered) > 0 else np.nan,
            'bright_y_n': len(bright_y_filtered),
        }
        results.append(row)
    
    # Create DataFrame
    stats_df = pd.DataFrame(results)
    
    # Print formatted table
    print("\n" + "="*120)
    print(f"Statistics Summary (outliers beyond ±{HISTOGRAM_BOUND} pixels excluded)")
    print("="*120)
    print(f"{'Exposure':>8} | {'ICP Translation X':^25} | {'ICP Translation Y':^25} | {'Brightest Star X':^25} | {'Brightest Star Y':^25}")
    print(f"{'(ms)':>8} | {'Mean':>8} {'StdDev':>8} {'N':>7} | {'Mean':>8} {'StdDev':>8} {'N':>7} | {'Mean':>8} {'StdDev':>8} {'N':>7} | {'Mean':>8} {'StdDev':>8} {'N':>7}")
    print("-"*120)
    
    for _, row in stats_df.iterrows():
        print(f"{row['exposure_ms']:>8.0f} | "
              f"{row['icp_x_mean']:>8.4f} {row['icp_x_std']:>8.4f} {row['icp_x_n']:>7.0f} | "
              f"{row['icp_y_mean']:>8.4f} {row['icp_y_std']:>8.4f} {row['icp_y_n']:>7.0f} | "
              f"{row['bright_x_mean']:>8.4f} {row['bright_x_std']:>8.4f} {row['bright_x_n']:>7.0f} | "
              f"{row['bright_y_mean']:>8.4f} {row['bright_y_std']:>8.4f} {row['bright_y_n']:>7.0f}")
    
    print("="*120)
    
    # Save to CSV if output path provided
    if output_path:
        stats_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nStatistics table saved to: {output_path}")
    
    return stats_df


# ============================================================================
# Main Function
# ============================================================================

def print_magnitude_table(df, magnitude_type='brightest'):
    """Generate and print a table of magnitude values across exposures and f-numbers"""
    
    # Get unique values
    sensors = sorted(df['sensor'].unique())
    f_numbers = sorted(df['f_number'].unique())
    exposures = sorted(df['exposure_ms'].unique())
    
    for sensor in sensors:
        print(f"\n{'='*100}")
        print(f"{magnitude_type.capitalize()} Star Magnitude Table - {sensor}")
        print('='*100)
        
        # Create header
        header = "F-number | " + " | ".join([f"{exp:4d}ms" for exp in exposures])
        print(header)
        print('-' * len(header))
        
        # For each f-number
        for f_num in f_numbers:
            row_data = [f"f/{f_num:4.1f}  "]
            
            for exp in exposures:
                mask = (df['sensor'] == sensor) & \
                       (df['f_number'] == f_num) & \
                       (df['exposure_ms'] == exp)
                subset = df[mask]
                
                if len(subset) > 0:
                    if magnitude_type == 'brightest':
                        mean_mag = subset['brightest_mag'].mean()
                    else:
                        mean_mag = subset['faintest_mag'].mean()
                    row_data.append(f"{mean_mag:6.2f}")
                else:
                    row_data.append("   N/A")
            
            print(" | ".join(row_data))
        
        # Add statistics row
        print('-' * len(header))
        print("\nStatistics:")
        
        # Best (brightest is lowest mag, faintest is highest mag)
        if magnitude_type == 'brightest':
            best_val = df[df['sensor'] == sensor]['brightest_mag'].min()
            best_mask = df[(df['sensor'] == sensor) & 
                          (df['brightest_mag'] == best_val)]
        else:
            best_val = df[df['sensor'] == sensor]['faintest_mag'].max()
            best_mask = df[(df['sensor'] == sensor) & 
                          (df['faintest_mag'] == best_val)]
        
        if len(best_mask) > 0:
            best_row = best_mask.iloc[0]
            print(f"Best {magnitude_type}: {best_val:.2f} mag at f/{best_row['f_number']:.2f}, "
                  f"{best_row['exposure_ms']}ms exposure")


def main():
    parser = argparse.ArgumentParser(description='Unified sensor comparison analysis')
    parser.add_argument('csv_file', type=str, help='Path to experiment CSV file')
    parser.add_argument('--mode', type=str, choices=['mean', 'win', 'quotient', 'stars', 'brightest', 'faintest', 'closure', 'accuracy', 'variance', 'histograms', 'coverage', 'heatmap', 'icp', 'brightest_star', 'trial_stddev', 'brightest_distribution', 'brightest_error_mag', 'brightest_dist_f', 'stats', 'all'], default='mean',
                       help='Analysis mode: mean pointing error, win percentage, quotient, star count, brightest/faintest magnitude, field closure, high accuracy, error variance, error histograms, sky coverage, sky heatmap, ICP translations, brightest star offsets, trial std deviation, brightest star distribution, brightest error vs magnitude, brightest distribution by f-number, statistics table, or all')
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
        heatmap_output = str(parent_dir / f"{base_name}_heatmap{ext}")
        icp_output = str(parent_dir / f"{base_name}_icp{ext}")
        brightest_star_output = str(parent_dir / f"{base_name}_brightest_star{ext}")
        trial_stddev_output = str(parent_dir / f"{base_name}_trial_stddev{ext}")
        brightest_distribution_output = str(parent_dir / f"{base_name}_brightest_distribution{ext}")
        brightest_error_mag_output = str(parent_dir / f"{base_name}_brightest_error_mag{ext}")
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
            
        if args.mode == 'heatmap' and '_heatmap' not in base_name:
            heatmap_output = str(parent_dir / f"{base_name}_heatmap{ext}")
        else:
            heatmap_output = args.output
            
        if args.mode == 'icp' and '_icp' not in base_name:
            icp_output = str(parent_dir / f"{base_name}_icp{ext}")
        else:
            icp_output = args.output
            
        if args.mode == 'brightest_star' and '_brightest_star' not in base_name:
            brightest_star_output = str(parent_dir / f"{base_name}_brightest_star{ext}")
        else:
            brightest_star_output = args.output
            
        if args.mode == 'trial_stddev' and '_trial_stddev' not in base_name:
            trial_stddev_output = str(parent_dir / f"{base_name}_trial_stddev{ext}")
        else:
            trial_stddev_output = args.output
            
        if args.mode == 'brightest_distribution' and '_brightest_distribution' not in base_name:
            brightest_distribution_output = str(parent_dir / f"{base_name}_brightest_distribution{ext}")
        else:
            brightest_distribution_output = args.output
            
        # Keep other outputs same as args.output for single mode
        mean_output = args.output if args.mode != 'mean' or '_mean' in base_name else str(parent_dir / f"{base_name}_mean{ext}")
        win_output = args.output if args.mode != 'win' or '_win' in base_name else str(parent_dir / f"{base_name}_win{ext}")
        quotient_output = args.output if args.mode != 'quotient' or '_quotient' in base_name else str(parent_dir / f"{base_name}_quotient{ext}")
        stars_output = args.output if args.mode != 'stars' or '_stars' in base_name else str(parent_dir / f"{base_name}_stars{ext}")
        brightest_output = args.output if args.mode != 'brightest' or '_brightest' in base_name else str(parent_dir / f"{base_name}_brightest{ext}")
        faintest_output = args.output if args.mode != 'faintest' or '_faintest' in base_name else str(parent_dir / f"{base_name}_faintest{ext}")
        closure_output = args.output if args.mode != 'closure' or '_closure' in base_name else str(parent_dir / f"{base_name}_closure{ext}")
        accuracy_output = args.output if args.mode != 'accuracy' or '_accuracy' in base_name else str(parent_dir / f"{base_name}_accuracy{ext}")
        # heatmap_output is already set above
    
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
    
    if args.mode in ['heatmap', 'all']:
        # For heatmap, only pass exposure_ms if user wants a specific exposure (not the default 250)
        exposure_for_heatmap = None if args.failed_exposure_ms == 250 else args.failed_exposure_ms
        plot_sky_coverage_heatmap(df, heatmap_output, exposure_ms=exposure_for_heatmap)
    
    if args.mode in ['icp', 'all']:
        plot_icp_translations(df, icp_output)
    
    if args.mode in ['brightest_star', 'all']:
        plot_brightest_star_offsets(df, brightest_star_output)
    
    if args.mode in ['trial_stddev', 'all']:
        plot_trial_error_stddev(df, trial_stddev_output)
    
    if args.mode in ['brightest_distribution', 'all']:
        plot_brightest_magnitude_distribution(df, brightest_distribution_output)
    
    if args.mode in ['brightest_error_mag', 'all']:
        # Generate output path for brightest error vs magnitude plot
        if args.mode == 'all':
            brightest_error_mag_output = str(parent_dir / f"{base_name}_brightest_error_mag{ext}")
        else:
            brightest_error_mag_output = args.output
        plot_brightest_star_error_vs_magnitude(df, brightest_error_mag_output)
    
    if args.mode in ['brightest_dist_f', 'all']:
        # Generate output path for brightest distribution by f-number plots
        if args.mode == 'all':
            brightest_dist_f_output = str(parent_dir / f"{base_name}_brightest_dist_f{ext}")
        else:
            brightest_dist_f_output = args.output
        plot_brightest_magnitude_distribution_by_fnumber(df, brightest_dist_f_output)
    
    if args.mode in ['stats', 'all']:
        # Generate stats table output path
        if args.mode == 'all':
            stats_output = str(parent_dir / f"{base_name}_stats_table.csv")
        else:
            stats_output = str(parent_dir / f"{base_name}_stats_table.csv") if not args.output else args.output.replace('.png', '.csv').replace('.pdf', '.csv')
        compute_statistics_table(df, stats_output)
        
        # Also print magnitude tables
        print_magnitude_table(df, magnitude_type='brightest')
        print_magnitude_table(df, magnitude_type='faintest')
    
    # Always print star count extremes at the end
    print_star_count_extremes(df, n=5)
    
    return 0


if __name__ == '__main__':
    exit(main())
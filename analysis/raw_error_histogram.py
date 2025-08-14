#!/usr/bin/env python3
"""
Simple raw histogram of pixel errors without any filtering or preprocessing.
Shows the true distribution of errors for debugging purposes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_raw_pixel_errors(csv_path, exposure_ms=250):
    """
    Load CSV and plot raw pixel error histogram for specific exposure time.
    No filtering, no masking, just the raw data.
    """
    # Load raw data
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter to specific exposure time
    exposure_mask = df['exposure_ms'] == exposure_ms
    exposure_df = df[exposure_mask].copy()
    
    if len(exposure_df) == 0:
        print(f"No data found for exposure time {exposure_ms}ms")
        return
    
    print(f"Found {len(exposure_df)} rows for {exposure_ms}ms exposure")
    
    # Get the raw pixel errors (including NaN values)
    pixel_errors_all = exposure_df['pixel_error']
    
    # Count different categories
    nan_count = pixel_errors_all.isna().sum()
    valid_errors = pixel_errors_all.dropna()
    
    if len(valid_errors) == 0:
        print("No valid pixel errors found (all NaN)")
        return
    
    # Count outliers
    outliers_negative = (valid_errors < 0).sum()
    outliers_above_1 = (valid_errors > 1.0).sum()
    outliers_above_2 = (valid_errors > 2.0).sum()
    outliers_above_5 = (valid_errors > 5.0).sum()
    
    # Get errors in 0-1 range for histogram
    errors_in_range = valid_errors[(valid_errors >= 0) & (valid_errors <= 1.0)]
    
    # Print statistics
    print(f"\nStatistics for {exposure_ms}ms exposure:")
    print(f"  Total rows: {len(exposure_df)}")
    print(f"  NaN values: {nan_count} ({nan_count/len(exposure_df)*100:.1f}%)")
    print(f"  Valid errors: {len(valid_errors)}")
    print(f"  Errors in [0,1]: {len(errors_in_range)} ({len(errors_in_range)/len(valid_errors)*100:.1f}% of valid)")
    print(f"\nOutliers:")
    print(f"  Negative: {outliers_negative}")
    print(f"  > 1.0 pixel: {outliers_above_1}")
    print(f"  > 2.0 pixels: {outliers_above_2}")
    print(f"  > 5.0 pixels: {outliers_above_5}")
    
    if len(valid_errors) > 0:
        print(f"\nValid error statistics:")
        print(f"  Min: {valid_errors.min():.4f}")
        print(f"  Max: {valid_errors.max():.4f}")
        print(f"  Mean: {valid_errors.mean():.4f}")
        print(f"  Median: {valid_errors.median():.4f}")
        print(f"  Std: {valid_errors.std():.4f}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Histogram of errors in [0,1] range
    n_bins = 50
    counts, bins, patches = ax1.hist(errors_in_range, bins=n_bins, range=(0, 1.0), 
                                     edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # Print bin information for first few and last few bins
    print(f"\nHistogram bins (showing first 5 and last 5):")
    for i in range(min(5, len(counts))):
        print(f"  Bin [{bins[i]:.4f}, {bins[i+1]:.4f}]: {int(counts[i])} counts")
    if len(counts) > 10:
        print("  ...")
        for i in range(max(0, len(counts)-5), len(counts)):
            print(f"  Bin [{bins[i]:.4f}, {bins[i+1]:.4f}]: {int(counts[i])} counts")
    
    # Mark mean and median
    mean_in_range = errors_in_range.mean()
    median_in_range = errors_in_range.median()
    ax1.axvline(mean_in_range, color='red', linestyle='--', label=f'Mean: {mean_in_range:.3f}')
    ax1.axvline(median_in_range, color='green', linestyle='--', label=f'Median: {median_in_range:.3f}')
    
    ax1.set_xlabel('Pixel Error')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Raw Pixel Errors [{exposure_ms}ms exposure]\n(Showing [0,1] range only)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text box with outlier info
    outlier_text = f'Outside [0,1] range:\n'
    outlier_text += f'Negative: {outliers_negative}\n'
    outlier_text += f'>1.0: {outliers_above_1}\n'
    outlier_text += f'NaN: {nan_count}'
    ax1.text(0.98, 0.98, outlier_text, transform=ax1.transAxes,
             fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Full distribution including outliers (if any exist)
    if outliers_negative > 0 or outliers_above_1 > 0:
        # Use log scale if range is large
        error_range = valid_errors.max() - valid_errors.min()
        use_log = error_range > 10
        
        ax2.hist(valid_errors, bins=50, edgecolor='black', linewidth=0.5, alpha=0.7)
        ax2.set_xlabel('Pixel Error (full range)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Full Distribution (including outliers)')
        if use_log:
            ax2.set_yscale('log')
            ax2.set_ylabel('Count (log scale)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'n={len(valid_errors)}\n'
        stats_text += f'μ={valid_errors.mean():.3f}\n'
        stats_text += f'σ={valid_errors.std():.3f}\n'
        stats_text += f'min={valid_errors.min():.3f}\n'
        stats_text += f'max={valid_errors.max():.3f}'
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        # No outliers, just show a message
        ax2.text(0.5, 0.5, 'No outliers detected\nAll valid errors in [0,1] range', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        ax2.set_title('Full Distribution')
        ax2.set_xlabel('Pixel Error')
        ax2.set_ylabel('Count')
    
    plt.suptitle(f'Raw Pixel Error Analysis - {Path(csv_path).name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(csv_path).parent / f"raw_pixel_errors_{exposure_ms}ms.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # plt.show()  # Commented out to prevent blocking


def main():
    parser = argparse.ArgumentParser(description='Plot raw pixel error histogram')
    parser.add_argument('csv_file', help='Path to experiment CSV file')
    parser.add_argument('--exposure', type=float, default=250,
                       help='Exposure time in ms to analyze (default: 250)')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: File {args.csv_file} not found")
        return
    
    plot_raw_pixel_errors(args.csv_file, args.exposure)


if __name__ == '__main__':
    main()
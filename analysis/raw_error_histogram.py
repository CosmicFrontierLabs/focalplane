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
    If translation_x and translation_y columns exist, also plot them.
    """
    # Load raw data
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check if we have the new decomposed error columns
    has_components = 'translation_x' in df.columns and 'translation_y' in df.columns
    
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
    
    # Create figure - add more subplots if we have component data
    if has_components:
        fig = plt.figure(figsize=(20, 12))
        ax1 = plt.subplot(2, 3, 1)  # Top left - total error
        ax2 = plt.subplot(2, 3, 2)  # Top middle - full distribution
        ax5 = plt.subplot(2, 3, 3)  # Top right - X vs Y scatter
        ax3 = plt.subplot(2, 3, 4)  # Bottom left - X translation
        ax4 = plt.subplot(2, 3, 5)  # Bottom middle - Y translation
        ax6 = plt.subplot(2, 3, 6)  # Bottom right - empty or stats
    else:
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
    
    # Plot 3: X vs Y scatter plot and additional histograms if available
    if has_components:
        # Get translation components - need paired values for scatter
        valid_mask = exposure_df['translation_x'].notna() & exposure_df['translation_y'].notna()
        translation_x = exposure_df.loc[valid_mask, 'translation_x']
        translation_y = exposure_df.loc[valid_mask, 'translation_y']
        
        # Plot 3: X vs Y scatter plot
        if len(translation_x) > 0:
            # Filter to reasonable range for visualization
            scatter_mask = (translation_x.abs() <= 1.0) & (translation_y.abs() <= 1.0)
            x_scatter = translation_x[scatter_mask]
            y_scatter = translation_y[scatter_mask]
            
            ax5.scatter(x_scatter, y_scatter, alpha=0.5, s=10, c='purple')
            ax5.axhline(0, color='red', linestyle='--', alpha=0.3)
            ax5.axvline(0, color='red', linestyle='--', alpha=0.3)
            ax5.set_xlabel('X Translation (pixels)')
            ax5.set_ylabel('Y Translation (pixels)')
            ax5.set_title('X vs Y Frame Misalignment\n(Showing [-1,1] range)')
            ax5.grid(True, alpha=0.3)
            ax5.set_xlim(-1, 1)
            ax5.set_ylim(-1, 1)
            ax5.set_aspect('equal')
            
            # Add circle at different radii
            import numpy as np
            theta = np.linspace(0, 2*np.pi, 100)
            for r, alpha in [(0.1, 0.2), (0.5, 0.15), (1.0, 0.1)]:
                ax5.plot(r*np.cos(theta), r*np.sin(theta), 'k-', alpha=alpha, linewidth=0.5)
            
            # Stats for scatter
            outliers_scatter = len(translation_x) - len(x_scatter)
            stats_text = f'n={len(x_scatter)}'
            if outliers_scatter > 0:
                stats_text += f'\nOutside ±1: {outliers_scatter}'
            ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes,
                    fontsize=10, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
        
        # Plot X translation histogram
        if len(translation_x) > 0:
            # Clip to reasonable range for visualization
            tx_clipped = translation_x[(translation_x >= -1.0) & (translation_x <= 1.0)]
            ax3.hist(tx_clipped, bins=50, range=(-1.0, 1.0), 
                    edgecolor='black', linewidth=0.5, alpha=0.7, color='blue')
            ax3.axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero')
            ax3.set_xlabel('X Translation (pixels)')
            ax3.set_ylabel('Count')
            ax3.set_title(f'X Component of Frame Misalignment\n(Showing [-1,1] range only)')
            ax3.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'n={len(translation_x)}\n'
            stats_text += f'μ={translation_x.mean():.4f}\n'
            stats_text += f'σ={translation_x.std():.4f}\n'
            stats_text += f'median={translation_x.median():.4f}'
            outliers_x = ((translation_x < -1.0) | (translation_x > 1.0)).sum()
            if outliers_x > 0:
                stats_text += f'\nOutside ±1: {outliers_x}'
            ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
                    fontsize=10, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot Y translation histogram
        if len(translation_y) > 0:
            # Clip to reasonable range for visualization
            ty_clipped = translation_y[(translation_y >= -1.0) & (translation_y <= 1.0)]
            ax4.hist(ty_clipped, bins=50, range=(-1.0, 1.0), 
                    edgecolor='black', linewidth=0.5, alpha=0.7, color='green')
            ax4.axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero')
            ax4.set_xlabel('Y Translation (pixels)')
            ax4.set_ylabel('Count')
            ax4.set_title(f'Y Component of Frame Misalignment\n(Showing [-1,1] range only)')
            ax4.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'n={len(translation_y)}\n'
            stats_text += f'μ={translation_y.mean():.4f}\n'
            stats_text += f'σ={translation_y.std():.4f}\n'
            stats_text += f'median={translation_y.median():.4f}'
            outliers_y = ((translation_y < -1.0) | (translation_y > 1.0)).sum()
            if outliers_y > 0:
                stats_text += f'\nOutside ±1: {outliers_y}'
            ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes,
                    fontsize=10, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Plot 6: Summary statistics or correlation info
        ax6.axis('off')
        summary_text = "Frame Misalignment Summary\n" + "="*30 + "\n\n"
        summary_text += f"Total experiments: {len(exposure_df)}\n"
        summary_text += f"Valid measurements: {len(translation_x)}\n\n"
        
        if len(translation_x) > 0:
            # Calculate correlation
            correlation = np.corrcoef(translation_x, translation_y)[0, 1]
            summary_text += f"X-Y Correlation: {correlation:.3f}\n\n"
            
            # Combined magnitude stats
            magnitude = np.sqrt(translation_x**2 + translation_y**2)
            summary_text += f"Combined Magnitude:\n"
            summary_text += f"  Mean: {magnitude.mean():.4f} px\n"
            summary_text += f"  Median: {magnitude.median():.4f} px\n"
            summary_text += f"  Std: {magnitude.std():.4f} px\n\n"
            
            # Percentiles
            summary_text += f"Percentiles:\n"
            for p in [50, 75, 90, 95, 99]:
                summary_text += f"  {p}%: {magnitude.quantile(p/100):.4f} px\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=11, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Raw Pixel Error Analysis - {Path(csv_path).name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(csv_path).parent / f"raw_pixel_errors_{exposure_ms}ms.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()  # Show the plot


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
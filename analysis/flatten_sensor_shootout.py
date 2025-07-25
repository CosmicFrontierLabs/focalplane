#!/usr/bin/env python3
"""
Flatten sensor_shootout.rs CSV output into a format suitable for pandas/data analysis.
Converts wide format with nested columns into long format with one row per measurement.
"""

import csv
import sys
import re
import argparse
from pathlib import Path


def extract_f_number(filename):
    """Extract f-number from filename like experiment_log_10.csv -> 10."""
    match = re.search(r'experiment_log_(\d+)\.csv', str(filename))
    if match:
        return int(match.group(1))
    return None


def process_single_file(input_file, aperture_cm, rows_data):
    """Process a single sensor shootout CSV file and add rows to rows_data."""
    
    # Get f-number from filename
    f_number = extract_f_number(input_file)
    if f_number is None:
        print(f"Warning: Could not extract f-number from filename {input_file}")
        return
    
    # Calculate focal length from f-number and aperture
    focal_length_m = f_number * (aperture_cm / 100.0)
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Find where data starts
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Experiment,"):
            data_start_idx = i
            break
    
    if data_start_idx is None:
        print(f"Warning: Could not find data start in {input_file}")
        return
    
    # Parse the multi-level header
    header1 = lines[data_start_idx].strip().split(',')
    header2 = lines[data_start_idx + 1].strip().split(',')
    header3 = lines[data_start_idx + 2].strip().split(',')
    
    # Find sensor column ranges
    sensor_starts = {}
    current_sensor = None
    for i, col in enumerate(header1):
        if col and col not in ['Experiment', 'RA', 'Dec', 'Duration']:
            current_sensor = col
            sensor_starts[current_sensor] = i
    
    # Parse exposure times from header
    exposures = []
    for col in header3:
        if 'ms' in col:
            exposures.append(int(col.replace('ms', '')))
    exposures = sorted(list(set(exposures)))
    
    # Process data rows
    for line in lines[data_start_idx + 3:]:
        if not line.strip():
            continue
            
        cols = line.strip().split(',')
        if len(cols) < 4:
            continue
            
        # Extract base experiment info
        try:
            exp_num = int(cols[0])
            ra = float(cols[1])
            dec = float(cols[2])
        except (ValueError, IndexError):
            continue
        
        # Process each sensor
        for sensor_name, start_idx in sensor_starts.items():
            # Calculate column indices for this sensor
            num_exposures = len(exposures)
            star_count_start = start_idx
            bright_mag_start = star_count_start + num_exposures
            faint_mag_start = bright_mag_start + num_exposures
            error_start = faint_mag_start + num_exposures
            
            # Extract data for each exposure
            for i, exposure in enumerate(exposures):
                try:
                    star_count = cols[star_count_start + i]
                    bright_mag = cols[bright_mag_start + i]
                    faint_mag = cols[faint_mag_start + i]
                    error_px = cols[error_start + i]
                    
                    # Skip empty values
                    if not star_count or star_count == '':
                        continue
                        
                    rows_data.append([
                        exp_num, ra, dec, f"{focal_length_m:.6f}", sensor_name,
                        exposure, star_count, bright_mag, faint_mag, error_px
                    ])
                except IndexError:
                    continue
    
    print(f"Processed {input_file}: extracted {len(rows_data)} rows")


def main():
    parser = argparse.ArgumentParser(description='Flatten sensor shootout CSV files')
    parser.add_argument('aperture_cm', type=float, help='Telescope aperture in centimeters')
    parser.add_argument('output_file', help='Output CSV filename')
    parser.add_argument('input_files', nargs='+', help='Input CSV files to process')
    
    args = parser.parse_args()
    
    # Collect all data rows
    all_rows = []
    
    # Process each input file
    for input_file in args.input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: Input file '{input_file}' not found, skipping")
            continue
        
        process_single_file(input_path, args.aperture_cm, all_rows)
    
    # Write all collected data to output file
    with open(args.output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'experiment_num', 'ra', 'dec', 'focal_length_m', 'sensor',
            'exposure_ms', 'star_count', 'brightest_mag', 'faintest_mag', 'pixel_error'
        ])
        
        # Write all data rows
        writer.writerows(all_rows)
    
    print(f"\nTotal rows written to {args.output_file}: {len(all_rows)}")


if __name__ == "__main__":
    main()
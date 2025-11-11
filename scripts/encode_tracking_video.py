#!/usr/bin/env python3
"""
Encode tracking frames to scaled videos in multiple formats.

Reads PNG frames from tracking_export/frames/, scales them 8x with nearest neighbor,
and creates APNG, lossless FFV1, and lossy H.264 videos.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"[INFO] {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed:")
        print(result.stderr)
        sys.exit(1)
    print(f"[SUCCESS] {description} complete")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Encode tracking frames to videos")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("tracking_export/frames"),
        help="Directory containing frame PNG files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tracking_export"),
        help="Directory for output videos",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=8,
        help="Scale factor for output (default: 8x)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=10,
        help="Number of initial frames to skip (default: 10)",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=10,
        help="Output video framerate (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for scaling (default: 8)",
    )
    args = parser.parse_args()

    frames_dir = args.frames_dir
    output_dir = args.output_dir
    scaled_dir = output_dir / "frames_scaled"

    if not frames_dir.exists():
        print(f"[ERROR] Frames directory not found: {frames_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    scaled_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Processing frames from: {frames_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Scale factor: {args.scale_factor}x")
    print(f"[INFO] Skipping first {args.skip_frames} frames")

    scale_percent = args.scale_factor * 100

    run_command(
        f"ls {frames_dir}/frame_*.png | tail -n +{args.skip_frames + 1} | "
        f"parallel -j{args.workers} 'mogrify -path {scaled_dir} -filter point -resize {scale_percent}% {{}}'",
        f"Scaling frames {args.scale_factor}x with nearest neighbor ({args.workers} workers)",
    )

    apng_output = output_dir / "tracking_scaled.apng"
    run_command(
        f"ffmpeg -y -framerate {args.framerate} -pattern_type glob -i '{scaled_dir}/frame_*.png' -plays 0 {apng_output}",
        f"Creating APNG at {args.framerate}fps",
    )

    ffv1_output = output_dir / "tracking_scaled_lossless.mkv"
    run_command(
        f"ffmpeg -y -i {apng_output} -c:v ffv1 -level 3 {ffv1_output}",
        "Creating lossless FFV1 video",
    )

    h264_output = output_dir / "tracking_scaled.mp4"
    run_command(
        f"ffmpeg -y -i {apng_output} -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p {h264_output}",
        "Creating lossy H.264 video",
    )

    print("\n[INFO] All encoding complete!")
    print(f"  APNG:     {apng_output} ({apng_output.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  FFV1:     {ffv1_output} ({ffv1_output.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  H.264:    {h264_output} ({h264_output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

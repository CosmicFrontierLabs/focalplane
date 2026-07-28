#!/usr/bin/env python3
"""Regenerate a 10,000-case sky-image test set against the focalplane stack.

For each of `--count` random pointings (uniform on the sphere, seeded):
  1. Render a single JBT50 + IMX455 frame with Gaia DR3 stars and NSA galaxies.
  2. Extract sources with `zodiacal extract` straight off the 16-bit PNG.
  3. Drop the rendered PNG to bound disk use (unless --keep-pngs).

Output JSON layout matches the legacy `zodiacal::extraction::SourcesJson`
schema so the existing accuracy harness keeps working unchanged.

See focalplane#83 for the design notes and open decisions.

Usage:
    uv run scripts/regenerate_test_cases.py \\
        --count 10000 --output-dir test_cases \\
        --motion-simulator target/release/motion_simulator \\
        --zodiacal /home/meawoppl/repos/zodiacal/target/release/zodiacal \\
        --excerpt-dir ~/.cache/starfield/gaia-excerpts/dr3-mag20-bycell \\
        --galaxy-catalog ~/.cache/starfield/nsa/nsa_v0_1_2.fits
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RAD_TO_ARCSEC = 206_264.80624709636


def uniform_sky_pointings(n, seed):
    """Pick `n` random RA/Dec pointings uniformly distributed on the sphere."""
    rng = random.Random(seed)
    ra = []
    dec = []
    for _ in range(n):
        ra.append(rng.uniform(0.0, 360.0))
        # sin(dec) uniform in [-1, 1] gives equal-area sphere sampling.
        dec.append(math.degrees(math.asin(rng.uniform(-1.0, 1.0))))
    return ra, dec


def render_one(motion_simulator, ra, dec, excerpt_dir, galaxy_catalog,
               mag_limit, exposure, render_seed, rayon_threads, out_dir):
    """Render one pointing and return its PNG and derived plate scale."""
    cmd = [
        str(motion_simulator),
        "--mode", "line",
        "--start", f"{ra},{dec}",
        "--end", f"{ra},{dec}",
        "--duration", "1s",
        "--timestep", "1s",
        "--exposure", exposure,
        "--telescope", "cosmic-frontier-jbt50cm",
        "--array-format", "single",
        "--sensor", "imx455",
        "--gaia-dr3",
        "--gaia-excerpt-dir", str(excerpt_dir),
        "--gaia-mag-limit", str(mag_limit),
        "--galaxies",
        "--galaxy-catalog", str(galaxy_catalog),
        "--seed", str(render_seed),
        "--output-dir", str(out_dir),
        "--quiet",
    ]
    env = os.environ.copy()
    env.setdefault("RUST_LOG", "warn")
    env["RAYON_NUM_THREADS"] = str(rayon_threads)
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"motion_simulator failed for ra={ra:.4f}, dec={dec:.4f}:\n")
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"render failed: exit {res.returncode}")
    png_path = out_dir / "sensor_00" / "frame_000000.png"
    metadata_path = out_dir / "metadata.json"
    if not png_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("renderer did not produce the expected PNG and metadata")

    with metadata_path.open() as f:
        metadata = json.load(f)
    focal_plane = metadata["focal_plane"]
    focal_length_m = focal_plane["telescope"]["focal_length"]
    pixel_size_m = focal_plane["array"]["sensors"][0]["sensor"]["dimensions"]["pixel_size"]
    plate_scale_arcsec = RAD_TO_ARCSEC * pixel_size_m / focal_length_m
    return png_path, plate_scale_arcsec


def extract_sources(zodiacal, png_path, ra, dec, plate_scale, json_path,
                    threshold_sigma, max_sources):
    """Run `zodiacal extract` on the PNG, writing the legacy-schema JSON."""
    cmd = [
        str(zodiacal), "extract",
        str(png_path),
        "-o", str(json_path),
        "--threshold-sigma", str(threshold_sigma),
        "--max-sources", str(max_sources),
        "--ra", f"{ra}",
        "--dec", f"{dec}",
        "--plate-scale", f"{plate_scale}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"zodiacal extract failed for {png_path}:\n")
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"extract failed: exit {res.returncode}")


def validate_sources_json(json_path, ra, dec, plate_scale):
    """Validate the legacy SourcesJson shape and generation metadata."""
    with json_path.open() as f:
        data = json.load(f)
    expected_keys = {
        "image_width", "image_height", "ra_deg", "dec_deg",
        "plate_scale_arcsec", "sources",
    }
    if set(data) != expected_keys:
        raise RuntimeError(f"unexpected JSON fields in {json_path}: {sorted(data)}")
    if not isinstance(data["sources"], list):
        raise RuntimeError(f"sources is not a list in {json_path}")
    if not math.isclose(data["ra_deg"], ra) or not math.isclose(data["dec_deg"], dec):
        raise RuntimeError(f"pointing mismatch in {json_path}")
    if not math.isclose(data["plate_scale_arcsec"], plate_scale):
        raise RuntimeError(f"plate-scale mismatch in {json_path}")
    return len(data["sources"])


def require_file(path, description):
    if not path.is_file():
        raise SystemExit(f"{description} not found: {path}")


def require_dir(path, description):
    if not path.is_dir():
        raise SystemExit(f"{description} not found: {path}")


def generate_case(args, idx, ra, dec):
    """Generate and validate one case, returning its extracted source count."""
    json_path = args.output_dir / f"{idx:04d}.json"
    if json_path.exists():
        return None
    with tempfile.TemporaryDirectory(prefix=f"case_{idx:04d}_") as tmp:
        tmp = Path(tmp)
        try:
            png, plate_scale = render_one(
                args.motion_simulator, ra, dec, args.excerpt_dir,
                args.galaxy_catalog, args.mag_limit, args.exposure,
                args.seed + idx, args.rayon_threads, tmp,
            )
            extract_sources(args.zodiacal, png, ra, dec,
                            plate_scale, json_path,
                            args.threshold_sigma, args.max_sources)
            n_sources = validate_sources_json(json_path, ra, dec, plate_scale)
            if args.keep_pngs:
                shutil.copy2(png, args.output_dir / f"{idx:04d}.png")
            return n_sources
        except Exception:
            json_path.unlink(missing_ok=True)
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=10000,
                    help="Number of test cases to generate.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the pointing RNG (uniform-on-sphere).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where the <NNNN>.json files land.")
    ap.add_argument("--motion-simulator", type=Path, required=True,
                    help="Path to the focalplane motion_simulator binary.")
    ap.add_argument("--zodiacal", type=Path, required=True,
                    help="Path to the zodiacal CLI binary.")
    ap.add_argument("--excerpt-dir", type=Path, required=True,
                    help="HEALPix-sharded DR3 excerpt directory.")
    ap.add_argument("--galaxy-catalog", type=Path, required=True,
                    help="NSA FITS catalog used by --galaxies.")
    ap.add_argument("--mag-limit", type=float, default=19.0)
    ap.add_argument("--exposure", default="10s",
                    help="Exposure time string accepted by motion_simulator.")
    ap.add_argument("--threshold-sigma", type=float, default=5.0)
    ap.add_argument("--max-sources", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent render/extract jobs (default: 4).")
    ap.add_argument("--rayon-threads", type=int, default=1,
                    help="Rayon threads per renderer process (default: 1).")
    ap.add_argument("--keep-pngs", action="store_true",
                    help="Keep the rendered PNGs (default: delete after extraction).")
    ap.add_argument("--start-index", type=int, default=0,
                    help="Start numbering JSONs from this index (resume support).")
    args = ap.parse_args()

    if args.count <= 0:
        ap.error("--count must be positive")
    if args.start_index < 0:
        ap.error("--start-index must be non-negative")
    if args.start_index + args.count > 10000:
        ap.error("this set is limited to indices 0000 through 9999")
    if args.workers <= 0 or args.rayon_threads <= 0:
        ap.error("--workers and --rayon-threads must be positive")

    require_file(args.motion_simulator, "motion_simulator binary")
    require_file(args.zodiacal, "zodiacal binary")
    require_dir(args.excerpt_dir, "Gaia DR3 excerpt directory")
    require_file(args.galaxy_catalog, "NSA galaxy catalog")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    draw_count = args.start_index + args.count
    ra_all, dec_all = uniform_sky_pointings(draw_count, args.seed)
    ra_all = ra_all[args.start_index:]
    dec_all = dec_all[args.start_index:]

    t0 = time.time()
    failures = []
    completed = 0
    jobs = [
        (args.start_index + i, ra, dec)
        for i, (ra, dec) in enumerate(zip(ra_all, dec_all))
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_case, args, idx, ra, dec): (idx, ra, dec)
            for idx, ra, dec in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            idx, ra, dec = futures[future]
            try:
                n_sources = future.result()
            except Exception as e:
                sys.stderr.write(f"[{idx:04d}] FAILED: {e}\n")
                failures.append((idx, str(e)))
                continue
            completed += 1
            if n_sources is None:
                print(f"[{idx:04d}] skip (already present)")
                continue
            elapsed = time.time() - t0
            throughput = completed / elapsed
            eta = (args.count - completed) / throughput
            print(f"[{idx:04d}] ra={ra:7.3f} dec={dec:+7.3f} "
                  f"sources={n_sources:3d} ({completed}/{args.count}, "
                  f"{throughput:.2f} cases/s, ETA {eta/60:.1f}m)")

    if failures:
        sys.stderr.write(f"{len(failures)} case(s) failed:\n")
        for idx, message in failures:
            sys.stderr.write(f"  {idx:04d}: {message}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

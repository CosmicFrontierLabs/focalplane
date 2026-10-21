# Dither Sampling for Diffraction-Limited Composite Imaging

A reference for choosing sub-pixel dither positions within a fixed exposure when compositing diffraction-limited point sources whose underlying field has a power-law-shaped spectrum.

## Problem statement

Given:

- A field containing diffraction-limited point sources at known angular pointings.
- A fixed total exposure time, divided across *n* sub-pixel dithers about a single field.
- A spectral content that is dense but falls off as a power law (low-frequency-dominated, with a long tail to the diffraction cutoff at D/λ).
- A linear (additive) compositing step — shift-and-add, drizzle, or Fourier combination.

Goal: choose the *n* dither positions so the reconstructed composite is as close as possible to the true continuous field.

## Why this is a sampling-design problem

Each dither shifts the PSF on the detector grid by a sub-pixel offset. The composite reconstruction inherits two error sources:

1. **Aliasing** from the detector pixel pitch (if the optical system out-resolves the pixels).
2. **Reconstruction variance** from the finite, discrete set of dither positions used to estimate the continuous field.

For a power-law spectrum, most signal energy sits at low spatial frequencies. The reconstruction error at frequency *k* is roughly the product of the signal power *P(k)* and the *spectral signature* of the dither set — i.e. the power spectrum of the sampling pattern itself. Minimizing reconstruction RMS therefore means choosing a dither set whose own spectrum has energy pushed *away* from where the signal lives. Since the signal is low-frequency-heavy, the dither set should be **low-discrepancy at low frequencies**: uniform on the dither plane, with no clumps and no harmonic spikes.

## What to avoid

**Uniform random dithers.** Easy to generate but produce clumps and gaps. The spectral signature is flat (white noise), so reconstruction error decays only as 1/√n.

**Regular n×n lattices.** Excellent low-frequency uniformity but produce sharp aliasing spikes at the lattice harmonics. These spikes land inside the signal band for any spectrum with dense high-frequency content, contaminating the reconstruction.

## Recommended approaches

### R2 (low-discrepancy, deterministic)

The R2 sequence is the 2D generalization of the golden-ratio sequence, built from the plastic constant φ ≈ 1.32472 (real root of x³ = x + 1). It produces deterministic points that fill the unit square near-optimally for *any* stopping value of *n*.

Properties:

- Discrepancy decays as log(n)/n rather than 1/√n.
- The first *m* points of the n-point set are themselves a good m-point set, so convergence studies are trivial.
- No tuning parameters, no randomness, two lines of code.
- Reconstruction error within roughly 10–20% of true blue noise for typical power-law spectra.

```python
import numpy as np

def r2_sequence(n, seed=0.5):
    phi = 1.32471795724474602596
    alpha = np.array([1.0/phi, 1.0/phi**2])
    i = np.arange(n)
    return (seed + np.outer(i, alpha)) % 1.0

def dithers_r2(n, extent=1.0):
    return (r2_sequence(n) - 0.5) * extent
```

This is the recommended default. Use it unless you have a specific reason to reach for something else.

### Poisson-disk / blue noise (best spectral properties)

Blue-noise sampling enforces a minimum separation between points, producing a pattern whose own power spectrum is suppressed at low frequencies and approximately flat at high frequencies — the inverse of a power-law signal spectrum. This is the optimal match.

Use Bridson's algorithm (O(n)) for generation:

```python
import numpy as np

def poisson_disk_2d(n, extent=1.0, k=30, seed=None):
    rng = np.random.default_rng(seed)
    r = 0.75 * extent / np.sqrt(n)
    cell = r / np.sqrt(2)
    grid_size = int(np.ceil(extent / cell))
    grid = -np.ones((grid_size, grid_size), dtype=int)

    def grid_coords(p):
        return int(p[0] / cell), int(p[1] / cell)

    def fits(p, points):
        gx, gy = grid_coords(p)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = gx + dx, gy + dy
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    idx = grid[x, y]
                    if idx >= 0 and np.linalg.norm(points[idx] - p) < r:
                        return False
        return True

    p0 = rng.uniform(0, extent, 2)
    points = [p0]
    grid[grid_coords(p0)] = 0
    active = [0]

    while active and len(points) < n:
        i = rng.integers(len(active))
        idx = active[i]
        found = False
        for _ in range(k):
            theta = rng.uniform(0, 2*np.pi)
            rr = rng.uniform(r, 2*r)
            cand = points[idx] + rr * np.array([np.cos(theta), np.sin(theta)])
            if 0 <= cand[0] < extent and 0 <= cand[1] < extent and fits(cand, points):
                points.append(cand)
                grid[grid_coords(cand)] = len(points) - 1
                active.append(len(points) - 1)
                found = True
                break
        if not found:
            active.pop(i)

    pts = np.array(points[:n])
    return pts - extent/2
```

Bridson produces *approximately* n points. If exactly n is required, either generate ~1.2n and trim, or use Mitchell's best-candidate algorithm (O(n²) but exact).

Use Poisson-disk when reconstruction precision is the dominant concern, when the spectral slope is steep (steeper than roughly −2.5), or when many independent realizations are needed.

### Small-n special case: half-integer dithers

For very small *n* (n = 2, 4) and the specific goal of breaking detector pixel aliasing — rather than densely sampling the PSF — the classical HST-style half-integer offsets are provably optimal:

- n = 2: (0, 0), (0.5, 0.5)
- n = 4: (0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)

This is the right choice when the optical system is critically sampled but the detector under-samples. For n ≥ 8 or for over-sampled optics, switch to R2 or Poisson-disk.

## Practical setup

**Dither extent.** Set to ±0.5 pixels (one full pixel range) to break detector sampling, or to ~λ/D to densely sample the PSF itself. Choose based on which is the limiting factor.

**Linearity prerequisites.** Additive compositing is valid only when:

- Detector response is below saturation everywhere.
- Sources are mutually incoherent (true for thermal and astronomical sources; not for coherent illumination).
- The reconstruction grid is consistent across pointings so co-addition does not require nonlinear resampling.

**Sanity checks after generation.**

1. Plot the point set. It should look uniform — no clumps, no visible rows or columns.
2. Compute the radial power spectrum of the point set: |Σⱼ exp(−2πi **k**·**x**ⱼ)|² on a 2D *k*-grid, then azimuthally average. R2 shows a sharp dip near k=0 with gentle ripples; blue noise shows a clean low-frequency void rising to a flat plateau. Either pattern strongly outperforms the flat spectrum of uniform random sampling.

## Quick decision guide

| Situation | Recommended choice |
|---|---|
| General default, any *n* | R2 sequence |
| Need exact reproducibility / nested subsets | R2 sequence |
| Maximum precision, steep power law (≲ −2.5) | Poisson-disk |
| n ∈ {2, 4}, breaking detector aliasing | Half-integer offsets |
| Predictable, slightly forgiving of small *n* | Jittered rotated grid (irrational rotation angle) |

## References

- Roberts, M. (2018). *The Unreasonable Effectiveness of Quasirandom Sequences.* — R2 sequence and the plastic constant.
- Bridson, R. (2007). *Fast Poisson Disk Sampling in Arbitrary Dimensions.* SIGGRAPH sketches.
- Mitchell, D. (1991). *Spectrally Optimal Sampling for Distribution Ray Tracing.* SIGGRAPH.
- Fruchter, A. & Hook, R. (2002). *Drizzle: A Method for the Linear Reconstruction of Undersampled Images.* PASP — for the reconstruction-side considerations.

//! Microbenchmark: how fast can we generate per-pixel Gaussian and Poisson
//! noise into a 61 MP `Array2<f64>`?
//!
//! Tests several axes to bound where the production `apply_*_noise`
//! functions sit relative to the achievable ceiling:
//!
//! 1. Baseline — single-threaded `Normal::sample` via `StdRng` (the original
//!    `mapv` path before parallelisation).
//! 2. Single-threaded `StandardNormal` (ziggurat) via `StdRng`.
//! 3. Single-threaded `StandardNormal` via `SmallRng` (PCG-class non-crypto).
//! 4. Production parallel path — `apply_gaussian_read_noise` from
//!    cfl-foundations.
//! 5. Same array, parallel, but using `SmallRng` + `StandardNormal` in
//!    rayon chunks (DIY upper bound at "fast RNG + fast distribution").
//! 6. Production parallel Poisson path — `apply_poisson_photon_noise`,
//!    once with a low-flux pedestal (real Poisson sampling), once with a
//!    high-flux pedestal (which short-circuits to the normal approximation).
//!
//! Run with: `cargo run --release --example noise_throughput`

use ndarray::{Array2, Axis};
use rand::rngs::{SmallRng, StdRng};
use rand::{Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal, StandardNormal};
use rayon::prelude::*;
use shared::image_proc::noise::{apply_gaussian_read_noise, apply_poisson_photon_noise};
use std::time::Instant;

const HEIGHT: usize = 6388;
const WIDTH: usize = 9576;
const RMS_E: f64 = 2.5; // representative IMX455 read noise (e-)

fn make_image(pedestal: f64) -> Array2<f64> {
    Array2::from_elem((HEIGHT, WIDTH), pedestal)
}

fn report(label: &str, elapsed_secs: f64) {
    let pixels = (HEIGHT * WIDTH) as f64;
    let mp_per_s = pixels / elapsed_secs / 1.0e6;
    let ns_per_pixel = elapsed_secs * 1.0e9 / pixels;
    println!(
        "{:<55} {:>8.1} ms   {:>7.1} M px/s   {:>5.1} ns/px",
        label,
        elapsed_secs * 1.0e3,
        mp_per_s,
        ns_per_pixel,
    );
}

fn bench_sequential_normal_stdrng() {
    let mut image = make_image(500.0);
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let normal = Normal::new(0.0, RMS_E).expect("rms ok");
    let t = Instant::now();
    image.mapv_inplace(|p| (p + normal.sample(&mut rng)).max(0.0));
    report(
        "[1] StdRng + Normal::sample, single-thread, mapv_inplace",
        t.elapsed().as_secs_f64(),
    );
}

fn bench_sequential_standard_normal_stdrng() {
    let mut image = make_image(500.0);
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let t = Instant::now();
    image.mapv_inplace(|p| {
        let z: f64 = rng.sample(StandardNormal);
        (p + z * RMS_E).max(0.0)
    });
    report(
        "[2] StdRng + StandardNormal (ziggurat), single-thread",
        t.elapsed().as_secs_f64(),
    );
}

fn bench_sequential_standard_normal_smallrng() {
    let mut image = make_image(500.0);
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let t = Instant::now();
    image.mapv_inplace(|p| {
        let z: f64 = rng.sample(StandardNormal);
        (p + z * RMS_E).max(0.0)
    });
    report(
        "[3a] SmallRng + StandardNormal (ziggurat), single-thread",
        t.elapsed().as_secs_f64(),
    );
}

fn bench_sequential_normal_smallrng() {
    let mut image = make_image(500.0);
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let normal = Normal::new(0.0, RMS_E).expect("rms ok");
    let t = Instant::now();
    image.mapv_inplace(|p| (p + normal.sample(&mut rng)).max(0.0));
    report(
        "[3b] SmallRng + Normal::sample (polar), single-thread",
        t.elapsed().as_secs_f64(),
    );
}

fn bench_production_gaussian_parallel() {
    let image = make_image(500.0);
    let t = Instant::now();
    let out = apply_gaussian_read_noise(image, RMS_E, Some(0xC0FFEE));
    let elapsed = t.elapsed().as_secs_f64();
    std::hint::black_box(&out);
    report(
        "[4] apply_gaussian_read_noise (by-value, no clone)",
        elapsed,
    );
}

fn bench_diy_parallel_smallrng_standard_normal() {
    let image = make_image(500.0);
    let mut out = image.clone();
    let t = Instant::now();
    out.axis_chunks_iter_mut(Axis(0), 64)
        .into_par_iter()
        .enumerate()
        .for_each(|(idx, mut chunk)| {
            let mut rng = SmallRng::seed_from_u64(0xC0FFEE_u64.wrapping_add(idx as u64));
            chunk.iter_mut().for_each(|p| {
                let z: f64 = rng.sample(StandardNormal);
                *p = (*p + z * RMS_E).max(0.0);
            });
        });
    let elapsed = t.elapsed().as_secs_f64();
    std::hint::black_box(&out);
    report(
        "[5a] DIY rayon + SmallRng + StandardNormal (ziggurat)",
        elapsed,
    );
}

fn bench_diy_parallel_smallrng_normal() {
    let image = make_image(500.0);
    let mut out = image.clone();
    let normal = Normal::new(0.0, RMS_E).expect("rms ok");
    let t = Instant::now();
    out.axis_chunks_iter_mut(Axis(0), 64)
        .into_par_iter()
        .enumerate()
        .for_each(|(idx, mut chunk)| {
            let mut rng = SmallRng::seed_from_u64(0xC0FFEE_u64.wrapping_add(idx as u64));
            chunk.iter_mut().for_each(|p| {
                *p = (*p + normal.sample(&mut rng)).max(0.0);
            });
        });
    let elapsed = t.elapsed().as_secs_f64();
    std::hint::black_box(&out);
    report(
        "[5b] DIY rayon + SmallRng + Normal::sample (polar)",
        elapsed,
    );
}

fn bench_production_poisson_low_mean() {
    // Mean below the 20-electron threshold inside the helper, so it uses
    // the real Poisson sampler rather than the normal approximation.
    let image = make_image(5.0);
    let t = Instant::now();
    let out = apply_poisson_photon_noise(image, Some(0xC0FFEE));
    let elapsed = t.elapsed().as_secs_f64();
    std::hint::black_box(&out);
    report(
        "[6a] apply_poisson_photon_noise (real Poisson, mean=5)",
        elapsed,
    );
}

fn bench_production_poisson_high_mean() {
    // Mean well above the threshold, so it short-circuits to Normal.
    let image = make_image(500.0);
    let t = Instant::now();
    let out = apply_poisson_photon_noise(image, Some(0xC0FFEE));
    let elapsed = t.elapsed().as_secs_f64();
    std::hint::black_box(&out);
    report(
        "[6b] apply_poisson_photon_noise (Normal approx, mean=500)",
        elapsed,
    );
}

fn bench_just_rng_throughput() {
    // No distribution math, no array writes — purely how fast can the
    // RNGs hand us u64 words. This is the absolute ceiling for any
    // sample-per-pixel approach.
    let n = HEIGHT * WIDTH;
    let mut rng_std = StdRng::seed_from_u64(0xC0FFEE);
    let mut rng_small = SmallRng::seed_from_u64(0xC0FFEE);

    let t = Instant::now();
    let mut sink: u64 = 0;
    for _ in 0..n {
        sink ^= rng_std.next_u64();
    }
    report(
        "[0a] raw StdRng::next_u64 throughput, single-thread",
        t.elapsed().as_secs_f64(),
    );
    std::hint::black_box(sink);

    let t = Instant::now();
    let mut sink: u64 = 0;
    for _ in 0..n {
        sink ^= rng_small.next_u64();
    }
    report(
        "[0b] raw SmallRng::next_u64 throughput, single-thread",
        t.elapsed().as_secs_f64(),
    );
    std::hint::black_box(sink);
}

fn main() {
    println!(
        "Noise throughput benchmark — {} × {} = {:.1} MP per array, RMS={} e-\n",
        WIDTH,
        HEIGHT,
        (HEIGHT * WIDTH) as f64 / 1.0e6,
        RMS_E,
    );
    println!(
        "{:<55} {:>8}    {:>11}   {:>7}",
        "case", "elapsed", "throughput", "per-px"
    );
    println!("{}", "-".repeat(95));

    bench_just_rng_throughput();
    bench_sequential_normal_stdrng();
    bench_sequential_standard_normal_stdrng();
    bench_sequential_standard_normal_smallrng();
    bench_sequential_normal_smallrng();
    bench_production_gaussian_parallel();
    bench_diy_parallel_smallrng_standard_normal();
    bench_diy_parallel_smallrng_normal();
    bench_production_poisson_low_mean();
    bench_production_poisson_high_mean();
}

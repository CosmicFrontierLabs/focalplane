use std::arch::asm;
use std::time::Instant;

#[inline(always)]
fn read_tsc() -> u64 {
    let val: u64;
    unsafe {
        asm!("mrs {}, cntvct_el0", out(reg) val, options(nomem, nostack));
    }
    val
}

#[inline(always)]
fn read_tsc_freq() -> u64 {
    let freq: u64;
    unsafe {
        asm!("mrs {}, cntfrq_el0", out(reg) freq, options(nomem, nostack));
    }
    freq
}

fn tsc_to_ns(tsc_ticks: u64, tsc_freq: u64) -> u64 {
    (tsc_ticks as u128 * 1_000_000_000u128 / tsc_freq as u128) as u64
}

fn get_monotonic_time_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn warmup() {
    println!("Warmup...");
    for _ in 0..1000 {
        let _ = read_tsc();
        let _ = Instant::now();
        let _ = get_monotonic_time_ns();
    }
}

fn clock_calibration(tsc_freq: u64) {
    println!("\n=== Clock Calibration (Instant-based estimation) ===");

    let calibration_samples = 100;
    let mut tsc_instant_pairs = Vec::with_capacity(calibration_samples);

    let calibration_start = Instant::now();
    for _ in 0..calibration_samples {
        let tsc = read_tsc();
        let instant = Instant::now();
        tsc_instant_pairs.push((tsc, instant));
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let elapsed_ns = calibration_start.elapsed().as_nanos() as f64;
    let tsc_delta =
        (tsc_instant_pairs.last().unwrap().0 - tsc_instant_pairs.first().unwrap().0) as f64;
    let estimated_freq = tsc_delta / elapsed_ns * 1e9;

    let mut sum_instant_ns = 0.0;
    let mut sum_tsc = 0.0;
    let mut sum_instant_tsc = 0.0;
    let mut sum_instant_sq = 0.0;
    let n = tsc_instant_pairs.len() as f64;

    for (tsc, instant) in &tsc_instant_pairs {
        let instant_ns = instant.duration_since(tsc_instant_pairs[0].1).as_nanos() as f64;
        sum_instant_ns += instant_ns;
        sum_tsc += *tsc as f64;
        sum_instant_tsc += instant_ns * (*tsc as f64);
        sum_instant_sq += instant_ns * instant_ns;
    }

    let slope = (n * sum_instant_tsc - sum_instant_ns * sum_tsc)
        / (n * sum_instant_sq - sum_instant_ns * sum_instant_ns);
    let intercept = (sum_tsc - slope * sum_instant_ns) / n;

    let estimated_freq_regression = slope * 1e9;

    println!(
        "Estimated TSC frequency (simple): {:.2} Hz ({:.2} MHz)",
        estimated_freq,
        estimated_freq / 1e6
    );
    println!(
        "Estimated TSC frequency (regression): {:.2} Hz ({:.2} MHz)",
        estimated_freq_regression,
        estimated_freq_regression / 1e6
    );
    println!(
        "Actual TSC frequency: {} Hz ({:.2} MHz)",
        tsc_freq,
        tsc_freq as f64 / 1e6
    );
    println!(
        "Frequency error (simple): {:.2} ppm",
        (estimated_freq - tsc_freq as f64).abs() / tsc_freq as f64 * 1e6
    );
    println!(
        "Frequency error (regression): {:.2} ppm",
        (estimated_freq_regression - tsc_freq as f64).abs() / tsc_freq as f64 * 1e6
    );
    println!("Estimated offset: {intercept:.2} TSC cycles");

    let offset_seconds = intercept / estimated_freq_regression;
    let tsc_now = read_tsc();
    let mono_now_ns = get_monotonic_time_ns();
    let mono_now_seconds = mono_now_ns as f64 / 1e9;
    let tsc_now_seconds = tsc_now as f64 / estimated_freq_regression;

    println!(
        "TSC time (from offset): {:.6} seconds",
        tsc_now_seconds - offset_seconds
    );
    println!("Monotonic time: {mono_now_seconds:.6} seconds");
    println!(
        "Difference: {:.6} seconds",
        (tsc_now_seconds - offset_seconds - mono_now_seconds).abs()
    );
}

fn overhead_comparison() {
    println!("\n=== Overhead Comparison ===");

    let iterations = 10000;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = read_tsc();
    }
    let tsc_overhead = start.elapsed();
    println!(
        "TSC read overhead: {:?} ({:.2} ns/call)",
        tsc_overhead,
        tsc_overhead.as_nanos() as f64 / iterations as f64
    );

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = Instant::now();
    }
    let instant_overhead = start.elapsed();
    println!(
        "Instant::now() overhead: {:?} ({:.2} ns/call)",
        instant_overhead,
        instant_overhead.as_nanos() as f64 / iterations as f64
    );

    println!(
        "Speedup: {:.1}x faster",
        instant_overhead.as_nanos() as f64 / tsc_overhead.as_nanos() as f64
    );
}

fn timestamp_comparison(tsc_freq: u64) {
    println!("\n=== Timestamp Comparison ===");

    let tsc_start = read_tsc();
    let instant_start = Instant::now();

    std::thread::sleep(std::time::Duration::from_millis(100));

    let tsc_end = read_tsc();
    let instant_end = Instant::now();

    let tsc_elapsed_ns = tsc_to_ns(tsc_end - tsc_start, tsc_freq);
    let instant_elapsed_ns = instant_end.duration_since(instant_start).as_nanos() as u64;

    println!(
        "TSC elapsed: {} ns ({:.3} ms)",
        tsc_elapsed_ns,
        tsc_elapsed_ns as f64 / 1e6
    );
    println!(
        "Instant elapsed: {} ns ({:.3} ms)",
        instant_elapsed_ns,
        instant_elapsed_ns as f64 / 1e6
    );
    println!(
        "Difference: {} ns",
        (tsc_elapsed_ns as i64 - instant_elapsed_ns as i64).abs()
    );
}

fn high_frequency_sampling(tsc_freq: u64) {
    println!("\n=== High-Frequency Sampling ===");

    let samples = 1000;
    let mut tsc_samples = Vec::with_capacity(samples);

    for _ in 0..samples {
        tsc_samples.push(read_tsc());
    }

    let intervals: Vec<u64> = tsc_samples.windows(2).map(|w| w[1] - w[0]).collect();

    let min_interval = *intervals.iter().min().unwrap();
    let max_interval = *intervals.iter().max().unwrap();
    let avg_interval = intervals.iter().sum::<u64>() / intervals.len() as u64;

    println!("Captured {samples} samples back-to-back:");
    println!(
        "  Min interval: {} cycles ({:.2} ns)",
        min_interval,
        tsc_to_ns(min_interval, tsc_freq) as f64
    );
    println!(
        "  Max interval: {} cycles ({:.2} ns)",
        max_interval,
        tsc_to_ns(max_interval, tsc_freq) as f64
    );
    println!(
        "  Avg interval: {} cycles ({:.2} ns)",
        avg_interval,
        tsc_to_ns(avg_interval, tsc_freq) as f64
    );
}

fn main() {
    let tsc_freq = read_tsc_freq();
    println!(
        "TSC frequency: {} Hz ({:.2} MHz)",
        tsc_freq,
        tsc_freq as f64 / 1e6
    );

    warmup();
    clock_calibration(tsc_freq);
    overhead_comparison();
    timestamp_comparison(tsc_freq);
    high_frequency_sampling(tsc_freq);
}

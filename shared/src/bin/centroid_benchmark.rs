use ndarray::Array2;
use shared::image_proc::centroid::compute_centroid_from_mask;
use std::time::Instant;

fn main() {
    // Typical tracking parameters from monocle FGS
    const ROI_SIZE: usize = 32;
    const FWHM: f64 = 4.0;
    const CENTROID_RADIUS_MULTIPLIER: f64 = 3.0;
    const ITERATIONS: usize = 10000;

    // Calculate typical mask radius
    let mask_radius = FWHM * CENTROID_RADIUS_MULTIPLIER;

    // Create realistic synthetic star data for ROI
    let mut image = Array2::from_elem((ROI_SIZE, ROI_SIZE), 100.0); // Background
    let mut mask = Array2::from_elem((ROI_SIZE, ROI_SIZE), false);

    // Generate Gaussian star profile centered in ROI
    let center_x = ROI_SIZE as f64 / 2.0;
    let center_y = ROI_SIZE as f64 / 2.0;
    let peak_intensity = 8000.0;
    let sigma = FWHM / 2.355; // Convert FWHM to Gaussian sigma

    for row in 0..ROI_SIZE {
        for col in 0..ROI_SIZE {
            let dx = col as f64 - center_x;
            let dy = row as f64 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();

            // Add Gaussian intensity
            let gauss_intensity =
                peak_intensity * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            image[[row, col]] += gauss_intensity;

            // Create circular mask based on centroid radius
            if dist <= mask_radius {
                mask[[row, col]] = true;
            }
        }
    }

    // Warmup iterations
    println!("Warming up...");
    for _ in 0..100 {
        let _ = compute_centroid_from_mask(&image.view(), &mask.view());
    }

    // Benchmark iterations with detailed timing
    println!("Running {ITERATIONS} iterations...");
    let mut timings = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _result = compute_centroid_from_mask(&image.view(), &mask.view());
        let duration = start.elapsed();
        timings.push(duration);
    }

    // Calculate statistics
    timings.sort();
    let total_nanos: u128 = timings.iter().map(|d| d.as_nanos()).sum();
    let mean_nanos = total_nanos / ITERATIONS as u128;
    let median_nanos = timings[ITERATIONS / 2].as_nanos();
    let p95_nanos = timings[(ITERATIONS as f64 * 0.95) as usize].as_nanos();
    let p99_nanos = timings[(ITERATIONS as f64 * 0.99) as usize].as_nanos();
    let min_nanos = timings[0].as_nanos();
    let max_nanos = timings[ITERATIONS - 1].as_nanos();

    // Print results
    println!("\n========== CENTROID TIMING BENCHMARK ==========");
    println!("Configuration:");
    println!("  ROI Size: {ROI_SIZE}x{ROI_SIZE} pixels");
    println!("  FWHM: {FWHM:.1} pixels");
    println!("  Centroid Radius Multiplier: {CENTROID_RADIUS_MULTIPLIER:.1}x");
    println!("  Mask Radius: {mask_radius:.1} pixels");
    println!("  Iterations: {ITERATIONS}");
    println!("\nTiming Results:");
    let mean_us = mean_nanos as f64 / 1000.0;
    let median_us = median_nanos as f64 / 1000.0;
    let min_us = min_nanos as f64 / 1000.0;
    let max_us = max_nanos as f64 / 1000.0;
    let p95_us = p95_nanos as f64 / 1000.0;
    let p99_us = p99_nanos as f64 / 1000.0;
    println!("  Mean:   {mean_us:>8.2} µs");
    println!("  Median: {median_us:>8.2} µs");
    println!("  Min:    {min_us:>8.2} µs");
    println!("  Max:    {max_us:>8.2} µs");
    println!("  P95:    {p95_us:>8.2} µs");
    println!("  P99:    {p99_us:>8.2} µs");
    println!("===============================================\n");

    // Verify result is reasonable
    let final_result = compute_centroid_from_mask(&image.view(), &mask.view());
    println!("Verification:");
    let x = final_result.x;
    let y = final_result.y;
    let flux = final_result.flux;
    let aspect_ratio = final_result.aspect_ratio;
    println!("  Centroid X: {x:.3} (expected ~{center_x:.1})");
    println!("  Centroid Y: {y:.3} (expected ~{center_y:.1})");
    println!("  Flux: {flux:.1} DN");
    println!("  Aspect Ratio: {aspect_ratio:.3}");

    assert!(
        (final_result.x - center_x).abs() < 0.1,
        "Centroid x should be near center"
    );
    assert!(
        (final_result.y - center_y).abs() < 0.1,
        "Centroid y should be near center"
    );
    assert!(
        final_result.flux > peak_intensity,
        "Flux should be significant"
    );

    println!("\n✓ All verification checks passed!");
}

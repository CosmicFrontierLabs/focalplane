use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use shared::image_proc::airy::PixelScaledAiryDisk;
use shared::image_proc::detection::AABB;
use shared::units::{LengthExt, Temperature, TemperatureExt, Wavelength};
use simulator::hardware::satellite::FocalPlaneConfig;
use simulator::hardware::sensor::models::IMX455;
use simulator::hardware::SatelliteConfig;
use simulator::image_proc::render::{add_stars_to_image, quantize_image, StarInFrame};
use simulator::photometry::photoconversion::{SourceFlux, SpotFlux};
use simulator::photometry::zodiacal::SolarAngularCoordinates;
use simulator::sims::motion_blur::{render_one_frame, render_one_frame_roi, MotionBlurConfig};
use simulator::sims::orientation::orientation_from_pointing;
use simulator::sims::trajectory::{Trajectory, Waypoint};
use starfield::catalogs::StarData;
use starfield::Equatorial;
use std::time::Duration;

fn make_satellite() -> SatelliteConfig {
    let telescope = simulator::hardware::telescope::TelescopeConfig::new(
        "Bench Telescope".to_string(),
        shared::units::Length::from_meters(0.5),
        shared::units::Length::from_meters(5.0),
        0.9,
    );
    SatelliteConfig::new(telescope, IMX455.clone(), Temperature::from_celsius(-10.0))
}

fn make_test_stars(count: usize, satellite: &SatelliteConfig) -> Vec<StarInFrame> {
    let airy = satellite.airy_disk_pixel_space();
    let ref_wavelength = Wavelength::from_nanometers(550.0);
    let disk = PixelScaledAiryDisk::with_first_zero(airy.first_zero(), ref_wavelength);

    (0..count)
        .map(|i| {
            let x = 100.0 + (i as f64 * 50.0) % 900.0;
            let y = 100.0 + (i as f64 * 37.0) % 900.0;
            let flux = 1000.0 / (i as f64 + 1.0);
            StarInFrame {
                x,
                y,
                spot: SourceFlux {
                    photons: SpotFlux {
                        disk: disk.clone(),
                        flux,
                    },
                    electrons: SpotFlux {
                        disk: disk.clone(),
                        flux: flux * 0.8,
                    },
                },
                star: StarData::new(i as u64, 56.75, 24.12, 5.0 + i as f64, Some(0.5)),
            }
        })
        .collect()
}

fn bench_add_stars_to_image(c: &mut Criterion) {
    let satellite = make_satellite();
    let stars_10 = make_test_stars(10, &satellite);
    let stars_100 = make_test_stars(100, &satellite);
    let exposure = Duration::from_millis(1000);
    let aperture = satellite.telescope.clear_aperture_area();

    let mut group = c.benchmark_group("add_stars_to_image");
    group.bench_function("10_stars_1024x1024", |b| {
        b.iter(|| {
            add_stars_to_image(
                black_box(1024),
                black_box(1024),
                black_box(&stars_10),
                black_box(&exposure),
                black_box(aperture),
            )
        })
    });
    group.bench_function("100_stars_1024x1024", |b| {
        b.iter(|| {
            add_stars_to_image(
                black_box(1024),
                black_box(1024),
                black_box(&stars_100),
                black_box(&exposure),
                black_box(aperture),
            )
        })
    });
    group.finish();
}

fn bench_quantize_image(c: &mut Criterion) {
    let sensor = IMX455.clone();
    let image_small = Array2::from_shape_fn((512, 512), |(r, c)| (r * c) as f64 * 0.1);
    let image_large = Array2::from_shape_fn((2048, 2048), |(r, c)| (r * c) as f64 * 0.01);

    let mut group = c.benchmark_group("quantize_image");
    group.bench_function("512x512", |b| {
        b.iter(|| quantize_image(black_box(&image_small), black_box(&sensor)))
    });
    group.bench_function("2048x2048", |b| {
        b.iter(|| quantize_image(black_box(&image_large), black_box(&sensor)))
    });
    group.finish();
}

fn bench_renderer_render(c: &mut Criterion) {
    let satellite = make_satellite();
    let center = Equatorial::from_degrees(56.75, 24.12);

    let star_data: Vec<StarData> = (0..50)
        .map(|i| {
            StarData::new(
                i,
                56.75 + (i as f64 * 0.01),
                24.12 + (i as f64 * 0.005),
                4.0 + (i as f64 * 0.1),
                Some(0.5),
            )
        })
        .collect();
    let star_refs: Vec<&StarData> = star_data.iter().collect();

    let renderer =
        simulator::image_proc::render::Renderer::from_catalog(&star_refs, &center, satellite);
    let coords = simulator::photometry::zodiacal::SolarAngularCoordinates::new(90.0, 30.0).unwrap();

    c.bench_function("renderer_render_100ms", |b| {
        b.iter(|| {
            renderer.render_with_seed(
                black_box(&Duration::from_millis(100)),
                black_box(&coords),
                Some(42),
            )
        })
    });
}

/// Build a single-sensor focal plane on a representative IMX455-class
/// detector for the ROI vs full-frame benchmark. Resolution is the
/// full 9568x6380 native frame.
fn bench_focal_plane() -> FocalPlaneConfig {
    let telescope = simulator::hardware::telescope::TelescopeConfig::new(
        "Bench Telescope".to_string(),
        shared::units::Length::from_meters(0.5),
        shared::units::Length::from_meters(5.0),
        0.9,
    );
    let sat = SatelliteConfig::new(telescope, IMX455.clone(), Temperature::from_celsius(-10.0));
    FocalPlaneConfig::from_satellite(&sat)
}

fn bench_catalog(center: Equatorial, count: usize) -> Vec<StarData> {
    (0..count)
        .map(|i| {
            let ra = center.ra_degrees() + ((i as f64 * 0.013) % 1.0 - 0.5) * 0.5;
            let dec = center.dec_degrees() + ((i as f64 * 0.017) % 1.0 - 0.5) * 0.3;
            StarData {
                id: i as u64,
                magnitude: 8.0 + (i as f64 % 4.0),
                position: Equatorial::from_degrees(ra, dec),
                b_v: Some(0.6),
            }
        })
        .collect()
}

/// Compare `render_one_frame` (full 9568x6380) against
/// `render_one_frame_roi` for a 1024x1024 ROI on the same sensor.
/// Reports times so callers can read off the speedup ratio. Per
/// project convention this bench does **not** assert any timing —
/// CI is variable; the bench is for visibility.
fn bench_render_one_frame_full_vs_roi(c: &mut Criterion) {
    let fp = bench_focal_plane();
    let center = Equatorial::from_degrees(56.75, 24.12);
    let stars = bench_catalog(center, 200);
    let traj = Trajectory::new(vec![
        Waypoint::new(Duration::ZERO, orientation_from_pointing(&center, 0.0)),
        Waypoint::new(
            Duration::from_secs(10),
            orientation_from_pointing(&center, 0.0),
        ),
    ])
    .expect("static trajectory");
    let cfg = MotionBlurConfig {
        timestep: Duration::from_secs(1),
        exposure: Duration::from_secs(1),
        max_drift_per_stamp_px: 0.1,
        base_seed: Some(0xCAFE_F00D),
        force_static: true,
        quiet: true,
        ..Default::default()
    };
    let zodi = SolarAngularCoordinates::zodiacal_minimum();
    let sensor_idx = 0_usize;

    // Center the ROI in the middle of the 9568x6380 sensor.
    let sat = fp.satellite_for_sensor(sensor_idx).unwrap();
    let (sensor_w, sensor_h) = sat.sensor.dimensions.get_pixel_width_height();
    let roi_side = 1024_usize;
    let min_row = (sensor_h - roi_side) / 2;
    let min_col = (sensor_w - roi_side) / 2;
    let roi = AABB::from_coords(
        min_row,
        min_col,
        min_row + roi_side - 1,
        min_col + roi_side - 1,
    );

    let mut group = c.benchmark_group("render_one_frame_roi");
    group.sample_size(10);
    group.bench_function("full_9568x6380", |b| {
        b.iter(|| {
            render_one_frame(
                black_box(&traj),
                black_box(&stars),
                black_box(&[]),
                black_box(&fp),
                black_box(zodi),
                black_box(Duration::ZERO),
                black_box(0),
                black_box(&cfg),
                None,
            )
            .expect("render_one_frame")
        })
    });
    group.bench_function("roi_1024x1024", |b| {
        b.iter(|| {
            render_one_frame_roi(
                black_box(&traj),
                black_box(&stars),
                black_box(&[]),
                black_box(&fp),
                black_box(zodi),
                black_box(Duration::ZERO),
                black_box(0),
                black_box(&cfg),
                None,
                black_box(roi),
                black_box(sensor_idx),
            )
            .expect("render_one_frame_roi")
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_add_stars_to_image,
    bench_quantize_image,
    bench_renderer_render,
    bench_render_one_frame_full_vs_roi,
);
criterion_main!(benches);

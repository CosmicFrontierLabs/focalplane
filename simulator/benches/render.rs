use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use shared::image_proc::airy::PixelScaledAiryDisk;
use shared::units::{LengthExt, Temperature, TemperatureExt, Wavelength};
use simulator::hardware::sensor::models::IMX455;
use simulator::hardware::SatelliteConfig;
use simulator::image_proc::render::{add_stars_to_image, quantize_image, StarInFrame};
use simulator::photometry::photoconversion::{SourceFlux, SpotFlux};
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

criterion_group!(
    benches,
    bench_add_stars_to_image,
    bench_quantize_image,
    bench_renderer_render,
);
criterion_main!(benches);

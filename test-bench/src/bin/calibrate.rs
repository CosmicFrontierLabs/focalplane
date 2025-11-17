use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use std::path::PathBuf;
use std::time::SystemTime;
use test_bench::display_patterns as patterns;
use test_bench::display_utils::{get_display_resolution, list_displays, SdlResultExt};

#[derive(Debug, Clone, ValueEnum)]
enum PatternType {
    Check,
    Usaf,
    Static,
    Pixel,
    April,
    CirclingPixel,
    Uniform,
    WigglingGaussian,
    PixelGrid,
    SiemensStar,
    MotionProfile,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Display calibration utility", long_about = None)]
struct Args {
    #[arg(short, long, help = "Display index to use (0-based)")]
    display: Option<u32>,

    #[arg(short, long, help = "List available displays and exit")]
    list: bool,

    #[arg(
        short,
        long,
        help = "Pattern type to display",
        value_enum,
        default_value = "april"
    )]
    pattern: PatternType,

    #[arg(long, help = "Target width in pixels (defaults to display width)")]
    width: Option<u32>,

    #[arg(long, help = "Target height in pixels (defaults to display height)")]
    height: Option<u32>,

    #[arg(
        long,
        help = "Size of each checker square in pixels",
        default_value = "100"
    )]
    checker_size: u32,

    #[arg(
        long,
        help = "Size of each static pixel block in pixels",
        default_value = "1"
    )]
    static_pixel_size: u32,

    #[arg(
        long,
        help = "Number of orbiting pixels in circling pattern",
        default_value = "1"
    )]
    orbit_count: u32,

    #[arg(
        long,
        help = "Orbit radius as percentage of FOV (0-100)",
        default_value = "50"
    )]
    orbit_radius_percent: u32,

    #[arg(long, help = "Uniform brightness level (0-255)", default_value = "0")]
    uniform_level: u8,

    #[arg(
        long,
        help = "Gaussian FWHM (Full Width Half Maximum in pixels)",
        default_value = "47"
    )]
    gaussian_fwhm: f64,

    #[arg(long, help = "Wiggle radius in pixels", default_value = "3")]
    wiggle_radius_pixels: f64,

    #[arg(
        long,
        help = "Maximum intensity for wiggling gaussian (0-255)",
        default_value = "255"
    )]
    gaussian_intensity: f64,

    #[arg(long, help = "Pixel grid spacing in pixels", default_value = "50")]
    grid_spacing: u32,

    #[arg(long, help = "Number of spokes in Siemens star", default_value = "24")]
    siemens_spokes: u32,

    #[arg(long, help = "Path to PNG image for motion profile pattern")]
    image_path: Option<PathBuf>,

    #[arg(
        long,
        help = "Path to CSV file with motion profile (t, x, y in seconds, pixels, pixels)"
    )]
    motion_csv: Option<PathBuf>,

    #[arg(
        long,
        help = "Motion scaling percentage (100 = normal, 50 = half motion, 200 = double motion)",
        default_value = "100.0"
    )]
    motion_scale_percent: f64,

    #[arg(short, long, help = "Invert pattern colors (black <-> white)")]
    invert: bool,

    #[arg(short, long, help = "Save pattern to PNG file instead of displaying")]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let sdl_context = sdl2::init().map_err(|e| anyhow::anyhow!("SDL init failed: {e}"))?;
    let video_subsystem = sdl_context
        .video()
        .map_err(|e| anyhow::anyhow!("Video subsystem init failed: {e}"))?;

    if args.list {
        return list_displays(&video_subsystem);
    }

    let display_index = args.display.unwrap_or(0);

    let bounds = video_subsystem
        .display_bounds(display_index as i32)
        .sdl_context("Failed to get display bounds")?;
    let mode = video_subsystem
        .desktop_display_mode(display_index as i32)
        .sdl_context("Failed to get display mode")?;

    let (display_width, display_height) = get_display_resolution(&video_subsystem, display_index)?;

    let pattern_width = args.width.unwrap_or(display_width);
    let pattern_height = args.height.unwrap_or(display_height);

    let (mut img, window_title) = match args.pattern {
        PatternType::Check => {
            println!("Generating checkerboard pattern");
            println!("  Checker size: {}px", args.checker_size);
            println!("  Pattern size: {pattern_width}x{pattern_height}");
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::checkerboard::generate(pattern_width, pattern_height, args.checker_size),
                "Checkerboard Pattern",
            )
        }
        PatternType::Usaf => {
            println!("Rendering USAF-1951 test target from SVG");
            println!("  Target size: {pattern_width}x{pattern_height}");
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::usaf::generate(pattern_width, pattern_height)?,
                "USAF-1951 Test Target",
            )
        }
        PatternType::Static => {
            println!("Generating static pattern");
            println!("  Pattern size: {pattern_width}x{pattern_height}");
            println!("  Block size: {}px", args.static_pixel_size);
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::static_noise::generate(
                    pattern_width,
                    pattern_height,
                    args.static_pixel_size,
                ),
                "Digital Static",
            )
        }
        PatternType::Pixel => {
            println!("Generating center pixel pattern");
            println!("  Pattern size: {pattern_width}x{pattern_height}");
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::pixel::generate(pattern_width, pattern_height),
                "Center Pixel",
            )
        }
        PatternType::April => {
            println!("Generating AprilTag array pattern");
            println!("  Pattern size: {pattern_width}x{pattern_height}");
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::apriltag::generate(pattern_width, pattern_height)?,
                "AprilTag Array",
            )
        }
        PatternType::CirclingPixel => {
            println!("Generating circling pixel pattern");
            println!("  Pattern size: {pattern_width}x{pattern_height}");
            println!("  Orbit count: {}", args.orbit_count);
            println!("  Orbit radius: {}% FOV", args.orbit_radius_percent);
            println!("  Rotation period: 60 seconds");
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::circling_pixel::generate(
                    pattern_width,
                    pattern_height,
                    args.orbit_count,
                    args.orbit_radius_percent,
                ),
                "Circling Pixel",
            )
        }
        PatternType::Uniform => {
            println!("Generating uniform screen");
            println!("  Pattern size: {pattern_width}x{pattern_height}");
            println!("  Brightness level: {}", args.uniform_level);
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::uniform::generate(pattern_width, pattern_height, args.uniform_level),
                "Uniform Screen",
            )
        }
        PatternType::WigglingGaussian => {
            println!("Generating wiggling gaussian pattern");
            println!("  Pattern size: {pattern_width}x{pattern_height}");
            println!("  Gaussian FWHM: {} pixels", args.gaussian_fwhm);
            println!("  Wiggle radius: {} pixels", args.wiggle_radius_pixels);
            println!("  Maximum intensity: {}", args.gaussian_intensity);
            println!("  Rotation period: 10 seconds");
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::wiggling_gaussian::generate(
                    pattern_width,
                    pattern_height,
                    args.gaussian_fwhm,
                    args.wiggle_radius_pixels,
                    args.gaussian_intensity,
                ),
                "Wiggling Gaussian",
            )
        }
        PatternType::PixelGrid => {
            println!("Generating pixel grid pattern");
            println!("  Pattern size: {pattern_width}x{pattern_height}");
            println!("  Grid spacing: {} pixels", args.grid_spacing);
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::pixel_grid::generate(pattern_width, pattern_height, args.grid_spacing),
                "Pixel Grid",
            )
        }
        PatternType::SiemensStar => {
            println!("Generating Siemens star pattern");
            println!("  Pattern size: {pattern_width}x{pattern_height}");
            println!("  Number of spokes: {}", args.siemens_spokes);
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::siemens_star::generate(
                    pattern_width,
                    pattern_height,
                    args.siemens_spokes,
                ),
                "Siemens Star",
            )
        }
        PatternType::MotionProfile => {
            let image_path = args
                .image_path
                .as_ref()
                .context("--image-path required for motion-profile pattern")?;
            let motion_csv = args
                .motion_csv
                .as_ref()
                .context("--motion-csv required for motion-profile pattern")?;

            println!("Loading motion profile pattern");
            println!("  Image: {}", image_path.display());
            println!("  Motion CSV: {}", motion_csv.display());

            let base_img = patterns::motion_profile::load_and_downsample_image(
                image_path,
                pattern_width,
                pattern_height,
            )?;

            println!(
                "  Downsampled image size: {}x{}",
                base_img.width(),
                base_img.height()
            );
            println!("  Pattern size: {pattern_width}x{pattern_height}");
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );

            let mut img = image::RgbImage::new(pattern_width, pattern_height);
            for pixel in img.pixels_mut() {
                *pixel = image::Rgb([0, 0, 0]);
            }

            (img, "Motion Profile")
        }
    };

    if args.invert {
        for pixel in img.pixels_mut() {
            pixel[0] = 255 - pixel[0];
            pixel[1] = 255 - pixel[1];
            pixel[2] = 255 - pixel[2];
        }
    }

    if let Some(output_path) = args.output {
        img.save(&output_path)
            .with_context(|| format!("Failed to save image to {}", output_path.display()))?;
        println!("Pattern saved to {}", output_path.display());
        return Ok(());
    }

    let window = video_subsystem
        .window(window_title, mode.w as u32, mode.h as u32)
        .position(bounds.x(), bounds.y())
        .fullscreen_desktop()
        .build()
        .context("Failed to create window")?;

    let mut canvas = window
        .into_canvas()
        .build()
        .context("Failed to create canvas")?;
    let texture_creator = canvas.texture_creator();

    let mut texture = texture_creator
        .create_texture_streaming(
            sdl2::pixels::PixelFormatEnum::RGB24,
            pattern_width,
            pattern_height,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create texture: {e:?}"))?;

    texture
        .update(None, img.as_raw(), (pattern_width * 3) as usize)
        .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;

    let window_width = mode.w as u32;
    let window_height = mode.h as u32;

    let scale_x = window_width as f32 / pattern_width as f32;
    let scale_y = window_height as f32 / pattern_height as f32;
    let scale = scale_x.min(scale_y);

    let scaled_width = (pattern_width as f32 * scale) as u32;
    let scaled_height = (pattern_height as f32 * scale) as u32;

    let x = (window_width - scaled_width) / 2;
    let y = (window_height - scaled_height) / 2;

    let dst_rect = Rect::new(x as i32, y as i32, scaled_width, scaled_height);

    let mut event_pump = sdl_context
        .event_pump()
        .map_err(|e| anyhow::anyhow!("Failed to get event pump: {e}"))?;

    let is_animated = matches!(
        args.pattern,
        PatternType::Static
            | PatternType::CirclingPixel
            | PatternType::WigglingGaussian
            | PatternType::MotionProfile
    );
    let mut static_buffer = if is_animated {
        Some(vec![0u8; (pattern_width * pattern_height * 3) as usize])
    } else {
        None
    };

    let motion_data = if matches!(args.pattern, PatternType::MotionProfile) {
        let image_path = args.image_path.as_ref().unwrap();
        let motion_csv = args.motion_csv.as_ref().unwrap();

        let base_img = image::open(image_path)
            .with_context(|| format!("Failed to load image: {}", image_path.display()))?
            .into_rgb8();
        let motion_profile = patterns::motion_profile::load_motion_profile(motion_csv)?;

        Some((base_img, motion_profile))
    } else {
        None
    };

    // TODO: Clean up and clock rendering rate to SDL advertised display refresh rate
    // Currently runs unconstrained which can cause excessive CPU usage and inconsistent timing.
    // Should use mode.refresh_rate to set target frame time and add frame pacing logic.

    let mut frame_count = 0u64;
    let mut last_fps_report = std::time::Instant::now();

    'running: loop {
        frame_count += 1;
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape | Keycode::Q),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        if let Some(ref mut buffer) = static_buffer {
            match args.pattern {
                PatternType::Static => {
                    patterns::static_noise::generate_into_buffer(
                        buffer,
                        pattern_width,
                        pattern_height,
                        args.static_pixel_size,
                    );
                }
                PatternType::CirclingPixel => {
                    patterns::circling_pixel::generate_into_buffer(
                        buffer,
                        pattern_width,
                        pattern_height,
                        args.orbit_count,
                        args.orbit_radius_percent,
                    );
                }
                PatternType::WigglingGaussian => {
                    patterns::wiggling_gaussian::generate_into_buffer(
                        buffer,
                        pattern_width,
                        pattern_height,
                        args.gaussian_fwhm,
                        args.wiggle_radius_pixels,
                        args.gaussian_intensity,
                    );
                }
                PatternType::MotionProfile => {
                    if let Some((ref base_img, ref motion_profile)) = motion_data {
                        let elapsed = SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap();
                        patterns::motion_profile::generate_into_buffer(
                            buffer,
                            pattern_width,
                            pattern_height,
                            base_img,
                            motion_profile,
                            elapsed,
                            args.motion_scale_percent / 100.0,
                        );
                    }
                }
                _ => {}
            }
            texture
                .update(None, buffer, (pattern_width * 3) as usize)
                .map_err(|e| anyhow::anyhow!("Failed to update texture: {e}"))?;
        }

        canvas.set_draw_color(sdl2::pixels::Color::RGB(0, 0, 0));
        canvas.clear();
        canvas
            .copy(&texture, None, Some(dst_rect))
            .map_err(|e| anyhow::anyhow!("Failed to copy texture: {e}"))?;
        canvas.present();

        let elapsed = last_fps_report.elapsed();
        if elapsed.as_secs() >= 1 {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            println!("FPS: {fps:.1}");
            frame_count = 0;
            last_fps_report = std::time::Instant::now();
        }
    }

    Ok(())
}

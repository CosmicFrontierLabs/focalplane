use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use std::path::PathBuf;
use test_bench::display_patterns as patterns;

trait SdlResultExt<T> {
    fn sdl_context(self, msg: &str) -> Result<T>;
}

impl<T> SdlResultExt<T> for std::result::Result<T, String> {
    fn sdl_context(self, msg: &str) -> Result<T> {
        self.map_err(|e| anyhow::anyhow!("{msg}: {e}"))
    }
}

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
        help = "Gaussian sigma (standard deviation in pixels)",
        default_value = "20"
    )]
    gaussian_sigma: f64,

    #[arg(long, help = "Wiggle radius in pixels", default_value = "3")]
    wiggle_radius_pixels: f64,

    #[arg(long, help = "Pixel grid spacing in pixels", default_value = "50")]
    grid_spacing: u32,

    #[arg(short, long, help = "Invert pattern colors (black <-> white)")]
    invert: bool,

    #[arg(short, long, help = "Save pattern to PNG file instead of displaying")]
    output: Option<PathBuf>,
}

fn list_displays(video_subsystem: &sdl2::VideoSubsystem) -> Result<()> {
    let num_displays = video_subsystem
        .num_video_displays()
        .sdl_context("Failed to get display count")?;
    println!("Found {num_displays} display(s):");

    for i in 0..num_displays {
        let name = video_subsystem
            .display_name(i)
            .sdl_context("Failed to get display name")?;
        let bounds = video_subsystem
            .display_bounds(i)
            .sdl_context("Failed to get display bounds")?;
        let mode = video_subsystem
            .desktop_display_mode(i)
            .sdl_context("Failed to get display mode")?;

        println!("\nDisplay {i}:");
        println!("  Name: {name}");
        println!("  Position: ({}, {})", bounds.x(), bounds.y());
        println!("  Size: {}x{}", bounds.width(), bounds.height());
        println!("  Mode: {}x{} @ {}Hz", mode.w, mode.h, mode.refresh_rate);
    }

    Ok(())
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

    let num_displays = video_subsystem
        .num_video_displays()
        .sdl_context("Failed to get display count")?;
    if display_index >= num_displays as u32 {
        anyhow::bail!("Display {display_index} not found (have {num_displays} displays)");
    }

    let bounds = video_subsystem
        .display_bounds(display_index as i32)
        .sdl_context("Failed to get display bounds")?;
    let mode = video_subsystem
        .desktop_display_mode(display_index as i32)
        .sdl_context("Failed to get display mode")?;

    let pattern_width = args.width.unwrap_or(mode.w as u32);
    let pattern_height = args.height.unwrap_or(mode.h as u32);

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
            println!("  Gaussian sigma: {}", args.gaussian_sigma);
            println!("  Wiggle radius: {} pixels", args.wiggle_radius_pixels);
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
                    args.gaussian_sigma,
                    args.wiggle_radius_pixels,
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
        PatternType::Static | PatternType::CirclingPixel | PatternType::WigglingGaussian
    );
    let mut static_buffer = if is_animated {
        Some(vec![0u8; (pattern_width * pattern_height * 3) as usize])
    } else {
        None
    };

    'running: loop {
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
                        args.gaussian_sigma,
                        args.wiggle_radius_pixels,
                    );
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

        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    Ok(())
}

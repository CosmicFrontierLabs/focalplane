mod patterns;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use std::path::PathBuf;

trait SdlResultExt<T> {
    fn sdl_context(self, msg: &str) -> Result<T>;
}

impl<T> SdlResultExt<T> for std::result::Result<T, String> {
    fn sdl_context(self, msg: &str) -> Result<T> {
        self.map_err(|e| anyhow::anyhow!("{}: {}", msg, e))
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum PatternType {
    Check,
    Usaf,
    Static,
    Pixel,
    April,
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

    #[arg(
        short,
        long,
        help = "Path to SVG file for USAF pattern",
        default_value = "display_assets/usaf-1951.svg"
    )]
    svg: PathBuf,

    #[arg(long, help = "Target width in pixels", default_value = "2560")]
    width: u32,

    #[arg(long, help = "Target height in pixels", default_value = "2560")]
    height: u32,

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

    let sdl_context = sdl2::init().map_err(|e| anyhow::anyhow!("SDL init failed: {}", e))?;
    let video_subsystem = sdl_context
        .video()
        .map_err(|e| anyhow::anyhow!("Video subsystem init failed: {}", e))?;

    if args.list {
        return list_displays(&video_subsystem);
    }

    let display_index = args.display.unwrap_or(0);

    let num_displays = video_subsystem
        .num_video_displays()
        .sdl_context("Failed to get display count")?;
    if display_index >= num_displays as u32 {
        anyhow::bail!(
            "Display {} not found (have {} displays)",
            display_index,
            num_displays
        );
    }

    let bounds = video_subsystem
        .display_bounds(display_index as i32)
        .sdl_context("Failed to get display bounds")?;
    let mode = video_subsystem
        .desktop_display_mode(display_index as i32)
        .sdl_context("Failed to get display mode")?;

    let (mut img, window_title) = match args.pattern {
        PatternType::Check => {
            println!("Generating checkerboard pattern");
            println!("  Checker size: {}px", args.checker_size);
            println!("  Pattern size: {}x{}", args.width, args.height);
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::checkerboard::generate(args.width, args.height, args.checker_size),
                "Checkerboard Pattern",
            )
        }
        PatternType::Usaf => {
            println!("Rendering USAF-1951 test target from SVG");
            println!("  SVG: {}", args.svg.display());
            println!("  Target size: {}x{}", args.width, args.height);
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::usaf::generate(&args.svg, args.width, args.height)?,
                "USAF-1951 Test Target",
            )
        }
        PatternType::Static => {
            println!("Generating static pattern");
            println!("  Pattern size: {}x{}", args.width, args.height);
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
                patterns::static_noise::generate(args.width, args.height, args.static_pixel_size),
                "Digital Static",
            )
        }
        PatternType::Pixel => {
            println!("Generating center pixel pattern");
            println!("  Pattern size: {}x{}", args.width, args.height);
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::pixel::generate(args.width, args.height),
                "Center Pixel",
            )
        }
        PatternType::April => {
            println!("Generating AprilTag array pattern");
            println!("  Pattern size: {}x{}", args.width, args.height);
            println!(
                "  Display {}: {}x{} at ({}, {})",
                display_index,
                mode.w,
                mode.h,
                bounds.x(),
                bounds.y()
            );
            (
                patterns::apriltag::generate(args.width, args.height)?,
                "AprilTag Array",
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
            args.width,
            args.height,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create texture: {:?}", e))?;

    texture
        .update(None, img.as_raw(), (args.width * 3) as usize)
        .map_err(|e| anyhow::anyhow!("Failed to update texture: {:?}", e))?;

    let window_width = mode.w as u32;
    let window_height = mode.h as u32;

    let scale_x = window_width as f32 / args.width as f32;
    let scale_y = window_height as f32 / args.height as f32;
    let scale = scale_x.min(scale_y);

    let scaled_width = (args.width as f32 * scale) as u32;
    let scaled_height = (args.height as f32 * scale) as u32;

    let x = (window_width - scaled_width) / 2;
    let y = (window_height - scaled_height) / 2;

    let dst_rect = Rect::new(x as i32, y as i32, scaled_width, scaled_height);

    let mut event_pump = sdl_context
        .event_pump()
        .map_err(|e| anyhow::anyhow!("Failed to get event pump: {}", e))?;

    let is_animated = matches!(args.pattern, PatternType::Static);
    let mut static_buffer = if is_animated {
        Some(vec![0u8; (args.width * args.height * 3) as usize])
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
            patterns::static_noise::generate_into_buffer(
                buffer,
                args.width,
                args.height,
                args.static_pixel_size,
            );
            texture
                .update(None, buffer, (args.width * 3) as usize)
                .map_err(|e| anyhow::anyhow!("Failed to update texture: {}", e))?;
        }

        canvas.set_draw_color(sdl2::pixels::Color::RGB(0, 0, 0));
        canvas.clear();
        canvas
            .copy(&texture, None, Some(dst_rect))
            .map_err(|e| anyhow::anyhow!("Failed to copy texture: {}", e))?;
        canvas.present();

        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    Ok(())
}

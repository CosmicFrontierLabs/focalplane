use anyhow::Result;
use std::time::Duration;

/// OLED display resolution for autodetection
pub const OLED_WIDTH: u32 = 2560;
pub const OLED_HEIGHT: u32 = 2560;

pub trait SdlResultExt<T> {
    fn sdl_context(self, msg: &str) -> Result<T>;
}

impl<T> SdlResultExt<T> for std::result::Result<T, String> {
    fn sdl_context(self, msg: &str) -> Result<T> {
        self.map_err(|e| anyhow::anyhow!("{msg}: {e}"))
    }
}

/// Find the OLED display by looking for 2560x2560 resolution.
/// Returns the display index if found, None otherwise.
pub fn find_oled_display(video_subsystem: &sdl2::VideoSubsystem) -> Result<Option<u32>> {
    let num_displays = video_subsystem
        .num_video_displays()
        .sdl_context("Failed to get display count")?;

    for i in 0..num_displays {
        let mode = video_subsystem
            .desktop_display_mode(i)
            .sdl_context("Failed to get display mode")?;

        if mode.w as u32 == OLED_WIDTH && mode.h as u32 == OLED_HEIGHT {
            return Ok(Some(i as u32));
        }
    }

    Ok(None)
}

/// Get display index to use: explicit override > OLED autodetect > display 0
pub fn resolve_display_index(
    video_subsystem: &sdl2::VideoSubsystem,
    explicit_display: Option<u32>,
) -> Result<u32> {
    // If user specified a display, use it
    if let Some(idx) = explicit_display {
        return Ok(idx);
    }

    // Try to find OLED by resolution
    if let Some(oled_idx) = find_oled_display(video_subsystem)? {
        println!(
            "Autodetected OLED display at index {oled_idx} ({}x{})",
            OLED_WIDTH, OLED_HEIGHT
        );
        return Ok(oled_idx);
    }

    // Fall back to display 0
    println!("No OLED display found, using display 0");
    Ok(0)
}

pub fn list_displays(video_subsystem: &sdl2::VideoSubsystem) -> Result<()> {
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

pub fn get_display_resolution(
    video_subsystem: &sdl2::VideoSubsystem,
    display_index: u32,
) -> Result<(u32, u32)> {
    let num_displays = video_subsystem
        .num_video_displays()
        .sdl_context("Failed to get display count")?;
    if display_index >= num_displays as u32 {
        anyhow::bail!("Display {display_index} not found (have {num_displays} displays)");
    }

    let mode = video_subsystem
        .desktop_display_mode(display_index as i32)
        .sdl_context("Failed to get display mode")?;

    Ok((mode.w as u32, mode.h as u32))
}

/// Wait for the 2560x2560 OLED display to become available.
/// Re-initializes SDL on each poll to detect newly connected displays.
/// Returns the SDL context and video subsystem once the OLED is found.
pub fn wait_for_oled_display(
    poll_interval: Duration,
) -> Result<(sdl2::Sdl, sdl2::VideoSubsystem, u32)> {
    println!(
        "Waiting for OLED display ({}x{}) to become available...",
        OLED_WIDTH, OLED_HEIGHT
    );

    loop {
        // Initialize SDL fresh each time to detect new displays
        let sdl_context = sdl2::init().sdl_context("SDL init failed")?;
        let video_subsystem = sdl_context
            .video()
            .sdl_context("Video subsystem init failed")?;

        match find_oled_display(&video_subsystem)? {
            Some(oled_idx) => {
                println!(
                    "OLED display detected at index {oled_idx} ({}x{})",
                    OLED_WIDTH, OLED_HEIGHT
                );
                return Ok((sdl_context, video_subsystem, oled_idx));
            }
            None => {
                let num_displays = video_subsystem
                    .num_video_displays()
                    .sdl_context("Failed to get display count")?;
                println!(
                    "OLED not found ({} display(s) available), retrying in {}s...",
                    num_displays,
                    poll_interval.as_secs()
                );
                // Drop SDL context before sleeping to release resources
                drop(video_subsystem);
                drop(sdl_context);
                std::thread::sleep(poll_interval);
            }
        }
    }
}

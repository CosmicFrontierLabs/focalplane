use anyhow::Result;

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

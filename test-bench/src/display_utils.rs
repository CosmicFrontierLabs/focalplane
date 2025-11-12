use anyhow::Result;

pub trait SdlResultExt<T> {
    fn sdl_context(self, msg: &str) -> Result<T>;
}

impl<T> SdlResultExt<T> for std::result::Result<T, String> {
    fn sdl_context(self, msg: &str) -> Result<T> {
        self.map_err(|e| anyhow::anyhow!("{msg}: {e}"))
    }
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

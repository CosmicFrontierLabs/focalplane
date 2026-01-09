use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::RwLock as AsyncRwLock;

use super::pattern::PatternConfig;

/// OLED burn-in protection watchdog.
///
/// Tracks activity and triggers screen blanking after idle timeout.
/// Thread-safe: can be used from both async handlers and sync threads.
#[derive(Clone)]
pub struct OledSafetyWatchdog {
    last_activity: Arc<RwLock<Instant>>,
    timeout: Duration,
}

impl OledSafetyWatchdog {
    pub fn new(timeout: Duration) -> Self {
        Self {
            last_activity: Arc::new(RwLock::new(Instant::now())),
            timeout,
        }
    }

    /// Reset the watchdog timer (call on any display activity)
    pub fn reset(&self) {
        *self.last_activity.write().unwrap() = Instant::now();
    }

    /// Check if display should be blanked due to inactivity
    pub fn is_timed_out(&self) -> bool {
        self.last_activity.read().unwrap().elapsed() > self.timeout
    }
}

/// Trait for providing patterns to the display runner.
pub trait PatternSource: Send {
    /// Get the current pattern and invert state.
    fn current(&self) -> (PatternConfig, bool);

    /// Check if the display should be blanked due to timeout.
    fn should_blank(&self) -> bool {
        false
    }

    /// Get a receiver for update notifications.
    fn update_receiver(&mut self) -> Option<&mut Receiver<()>> {
        None
    }

    /// Called when pattern rendering completes (for stats/logging).
    fn on_rendered(&self) {}
}

/// Dynamic pattern source for web server - pattern can change via API.
pub struct DynamicPattern {
    pattern: Arc<AsyncRwLock<PatternConfig>>,
    invert: Arc<AsyncRwLock<bool>>,
    watchdog: OledSafetyWatchdog,
    update_rx: Receiver<()>,
}

impl DynamicPattern {
    pub fn new(
        pattern: Arc<AsyncRwLock<PatternConfig>>,
        invert: Arc<AsyncRwLock<bool>>,
        watchdog: OledSafetyWatchdog,
        update_rx: Receiver<()>,
    ) -> Self {
        Self {
            pattern,
            invert,
            watchdog,
            update_rx,
        }
    }
}

impl PatternSource for DynamicPattern {
    fn current(&self) -> (PatternConfig, bool) {
        let pattern = self.pattern.blocking_read().clone();
        let invert = *self.invert.blocking_read();
        (pattern, invert)
    }

    fn should_blank(&self) -> bool {
        self.watchdog.is_timed_out()
    }

    fn update_receiver(&mut self) -> Option<&mut Receiver<()>> {
        Some(&mut self.update_rx)
    }
}

/// Configuration for the display runner.
pub struct DisplayConfig {
    pub width: u32,
    pub height: u32,
    pub display_index: u32,
    pub show_fps: bool,
}

/// Run the SDL display loop with the given pattern source.
pub fn run_display<P: PatternSource>(
    sdl_context: sdl2::Sdl,
    config: DisplayConfig,
    mut source: P,
) -> Result<()> {
    let video_subsystem = sdl_context
        .video()
        .map_err(|e| anyhow::anyhow!("Video subsystem init failed: {e}"))?;

    // Hide the mouse cursor so it doesn't appear over the calibration pattern
    sdl_context.mouse().show_cursor(false);

    let display_bounds = video_subsystem
        .display_bounds(config.display_index as i32)
        .map_err(|e| anyhow::anyhow!("Failed to get display bounds: {e}"))?;

    let window = video_subsystem
        .window("Calibration Pattern", config.width, config.height)
        .position(display_bounds.x(), display_bounds.y())
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
            config.width,
            config.height,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create texture: {e:?}"))?;

    let mut event_pump = sdl_context
        .event_pump()
        .map_err(|e| anyhow::anyhow!("Failed to get event pump: {e}"))?;

    let render_pattern =
        |pattern: &PatternConfig, invert: bool| -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
            let mut img = pattern.generate(config.width, config.height)?;
            if invert {
                PatternConfig::apply_invert(&mut img);
            }
            Ok(img)
        };

    // Initial render
    let (mut current_pattern, mut current_invert) = source.current();
    let mut img = render_pattern(&current_pattern, current_invert)?;
    let mut buffer: Vec<u8> = vec![0; (config.width * config.height * 3) as usize];

    if !current_pattern.is_animated() {
        texture
            .update(None, img.as_raw(), (config.width * 3) as usize)
            .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;
    }

    canvas.clear();
    canvas
        .copy(&texture, None, None)
        .map_err(|e| anyhow::anyhow!("Failed to copy texture: {e}"))?;
    canvas.present();

    // State for timeout tracking
    let mut last_timeout_check = Instant::now();
    let mut was_timed_out = false;

    // FPS tracking
    let mut frame_count = 0u64;
    let mut last_fps_report = Instant::now();

    loop {
        // Handle SDL events
        for event in event_pump.poll_iter() {
            use sdl2::event::Event;
            use sdl2::keyboard::Keycode;
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape | Keycode::Q),
                    ..
                } => {
                    return Ok(());
                }
                _ => {}
            }
        }

        let mut pattern_changed = false;

        // Check for updates from external source (web API)
        if let Some(rx) = source.update_receiver() {
            if rx.try_recv().is_ok() {
                (current_pattern, current_invert) = source.current();
                pattern_changed = true;
                was_timed_out = false;

                if !current_pattern.is_animated() {
                    img = render_pattern(&current_pattern, current_invert)?;
                    texture
                        .update(None, img.as_raw(), (config.width * 3) as usize)
                        .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;
                }
            }
        }

        // Periodically check idle timeout (every second)
        if last_timeout_check.elapsed() > Duration::from_secs(1) {
            last_timeout_check = Instant::now();

            if source.should_blank() && !was_timed_out {
                tracing::info!("Idle timeout reached, blanking display");
                current_pattern = PatternConfig::Uniform { level: 0 };
                img = render_pattern(&current_pattern, current_invert)?;
                texture
                    .update(None, img.as_raw(), (config.width * 3) as usize)
                    .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;
                pattern_changed = true;
                was_timed_out = true;
            }
        }

        // Handle animated patterns
        if current_pattern.is_animated() || pattern_changed {
            current_pattern.generate_into_buffer(&mut buffer, config.width, config.height);

            if current_pattern.is_animated() {
                let buffer_to_use: Vec<u8> = if current_invert {
                    buffer.iter().map(|&b| 255 - b).collect()
                } else {
                    buffer.clone()
                };

                texture
                    .update(None, &buffer_to_use, (config.width * 3) as usize)
                    .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;
            }
        }

        // Render
        canvas.clear();
        canvas
            .copy(&texture, None, None)
            .map_err(|e| anyhow::anyhow!("Failed to copy texture: {e}"))?;
        canvas.present();

        source.on_rendered();
        frame_count += 1;

        // FPS reporting
        if config.show_fps {
            let elapsed = last_fps_report.elapsed();
            if elapsed.as_secs() >= 1 {
                let fps = frame_count as f64 / elapsed.as_secs_f64();
                println!("FPS: {fps:.1}");
                frame_count = 0;
                last_fps_report = Instant::now();
            }
        }

        std::thread::sleep(Duration::from_millis(16));
    }
}

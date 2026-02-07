use anyhow::{Context, Result};
use bytes::Bytes;
use image::{ImageBuffer, Rgb};
use shared::image_size::PixelShape;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock as AsyncRwLock;

use super::pattern::PatternConfig;
use crate::ws_stream::{WsBroadcaster, WsFrame};

/// Internal state for the watchdog, protected by mutex.
struct WatchdogInner {
    /// Time of last activity (pattern change, command, etc.)
    last_activity: Instant,
    /// Timeout duration before blanking
    timeout: Duration,
    /// Pattern saved when we blanked (to restore on wake)
    saved_pattern: Option<PatternConfig>,
    /// Whether display is currently blanked
    is_blanked: bool,
}

/// OLED burn-in protection watchdog.
///
/// Monitors activity and automatically blanks the display after idle timeout.
/// When activity resumes, restores the previous pattern.
///
/// This watchdog OWNS the timeout state - it directly modifies the shared
/// pattern when blanking/waking, making the logic obviously correct.
///
/// Cloning creates a handle to the same watchdog - only one monitoring thread
/// exists regardless of how many clones are made.
#[derive(Clone)]
pub struct OledSafetyWatchdog {
    inner: Arc<Mutex<WatchdogInner>>,
    /// Channel to notify display of updates
    update_tx: Sender<()>,
}

impl OledSafetyWatchdog {
    /// Create a new watchdog and spawn its monitoring thread.
    ///
    /// The watchdog will:
    /// - Blank the display after `timeout` of inactivity
    /// - Restore the previous pattern when `reset()` is called
    pub fn new(
        timeout: Duration,
        pattern: Arc<AsyncRwLock<PatternConfig>>,
        update_tx: Sender<()>,
    ) -> Self {
        let inner = Arc::new(Mutex::new(WatchdogInner {
            last_activity: Instant::now(),
            timeout,
            saved_pattern: None,
            is_blanked: false,
        }));

        // Spawn watchdog thread
        let inner_clone = inner.clone();
        let pattern_clone = pattern.clone();
        let update_tx_clone = update_tx.clone();

        std::thread::spawn(move || {
            watchdog_thread(inner_clone, pattern_clone, update_tx_clone);
        });

        Self { inner, update_tx }
    }

    /// Reset the watchdog timer (call on any display activity).
    ///
    /// If the display is currently blanked, this clears the blanked state.
    /// The caller is responsible for setting the new pattern - this method
    /// does NOT restore the saved pattern (since the caller likely just set
    /// a new pattern that should take precedence).
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.last_activity = Instant::now();

        if inner.is_blanked {
            tracing::info!("Watchdog: waking display, clearing blanked state");
            // Clear saved pattern - caller has already set the new pattern
            inner.saved_pattern = None;
            inner.is_blanked = false;
            // Notify display to update (caller's new pattern will be used)
            let _ = self.update_tx.send(());
        }
    }

    /// Check if the display is currently blanked due to timeout.
    pub fn is_blanked(&self) -> bool {
        self.inner.lock().unwrap().is_blanked
    }
}

/// Watchdog monitoring thread.
///
/// Checks every second if we've exceeded the idle timeout.
/// When timeout occurs, saves current pattern and blanks display.
fn watchdog_thread(
    inner: Arc<Mutex<WatchdogInner>>,
    pattern: Arc<AsyncRwLock<PatternConfig>>,
    update_tx: Sender<()>,
) {
    loop {
        std::thread::sleep(Duration::from_secs(1));

        let should_blank = {
            let inner = inner.lock().unwrap();
            !inner.is_blanked && inner.last_activity.elapsed() > inner.timeout
        };

        if should_blank {
            // Need to blank - save pattern and set to black
            let mut inner = inner.lock().unwrap();

            // Double-check after re-acquiring lock
            if inner.is_blanked || inner.last_activity.elapsed() <= inner.timeout {
                continue;
            }

            tracing::info!(
                "Watchdog: idle timeout ({:.0}s), blanking display",
                inner.timeout.as_secs_f64()
            );

            // Save current pattern
            let current = pattern.blocking_read().clone();
            inner.saved_pattern = Some(current);

            // Set to black
            let mut pattern_write = pattern.blocking_write();
            *pattern_write = PatternConfig::Uniform { level: 0 };

            inner.is_blanked = true;

            // Notify display
            let _ = update_tx.send(());
        }
    }
}

/// Trait for providing patterns to the display runner.
pub trait PatternSource: Send {
    /// Get the current pattern and invert state.
    fn current(&self) -> (PatternConfig, bool);

    /// Get a receiver for update notifications.
    fn update_receiver(&mut self) -> Option<&mut Receiver<()>> {
        None
    }

    /// Called when pattern rendering completes with the rendered image.
    fn on_rendered(&mut self, _img: &ImageBuffer<Rgb<u8>, Vec<u8>>) {}
}

/// Dynamic pattern source for web server - pattern can change via API.
pub struct DynamicPattern {
    pattern: Arc<AsyncRwLock<PatternConfig>>,
    invert: Arc<AsyncRwLock<bool>>,
    update_rx: Receiver<()>,
    /// Optional WebSocket broadcaster for streaming pattern preview
    ws_stream: Option<Arc<WsBroadcaster>>,
    /// Frame counter for streaming
    frame_number: u64,
    /// Last time we published a stream frame (for rate limiting)
    last_stream_publish: Instant,
}

/// Minimum interval between stream frame publishes (30 fps max).
const STREAM_MIN_INTERVAL: Duration = Duration::from_millis(33);

impl DynamicPattern {
    pub fn new(
        pattern: Arc<AsyncRwLock<PatternConfig>>,
        invert: Arc<AsyncRwLock<bool>>,
        update_rx: Receiver<()>,
    ) -> Self {
        Self {
            pattern,
            invert,
            update_rx,
            ws_stream: None,
            frame_number: 0,
            last_stream_publish: Instant::now() - STREAM_MIN_INTERVAL,
        }
    }

    /// Set the WebSocket broadcaster for streaming rendered frames.
    pub fn with_ws_stream(mut self, ws_stream: Arc<WsBroadcaster>) -> Self {
        self.ws_stream = Some(ws_stream);
        self
    }

    /// Encode and publish image to WebSocket stream (rate limited).
    fn maybe_publish_stream(&mut self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) {
        let has_subs = self
            .ws_stream
            .as_ref()
            .map(|w| w.subscriber_count() > 0)
            .unwrap_or(false);

        if !has_subs {
            return;
        }

        if self.last_stream_publish.elapsed() < STREAM_MIN_INTERVAL {
            return;
        }

        // Encode as JPEG
        let mut jpeg_bytes = Vec::new();
        if img
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg_bytes),
                image::ImageFormat::Jpeg,
            )
            .is_err()
        {
            return;
        }

        self.frame_number += 1;

        if let Some(ref ws_stream) = self.ws_stream {
            ws_stream.publish(WsFrame {
                jpeg_data: Bytes::from(jpeg_bytes),
                frame_number: self.frame_number,
                width: img.width(),
                height: img.height(),
            });
        }

        self.last_stream_publish = Instant::now();
    }
}

impl PatternSource for DynamicPattern {
    fn current(&self) -> (PatternConfig, bool) {
        let pattern = self.pattern.blocking_read().clone();
        let invert = *self.invert.blocking_read();
        (pattern, invert)
    }

    fn update_receiver(&mut self) -> Option<&mut Receiver<()>> {
        Some(&mut self.update_rx)
    }

    fn on_rendered(&mut self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) {
        self.maybe_publish_stream(img);
    }
}

/// Configuration for the display runner.
pub struct DisplayConfig {
    pub size: PixelShape,
    pub display_index: u32,
    pub show_fps: bool,
    /// When true, ignore keyboard quit shortcuts and window close events.
    /// The only way to stop the display is via SIGTERM (Ctrl+C).
    pub lock_display: bool,
}

impl DisplayConfig {
    /// Get width in pixels.
    pub fn width(&self) -> u32 {
        self.size.width as u32
    }

    /// Get height in pixels.
    pub fn height(&self) -> u32 {
        self.size.height as u32
    }
}

/// Run the SDL display loop with the given pattern source.
///
/// This loop is intentionally simple - it just renders whatever
/// `source.current()` returns. All timeout logic is handled by
/// the watchdog thread, which modifies the shared pattern state.
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
        .window("Calibration Pattern", config.width(), config.height())
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
            config.width(),
            config.height(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to create texture: {e:?}"))?;

    let mut event_pump = sdl_context
        .event_pump()
        .map_err(|e| anyhow::anyhow!("Failed to get event pump: {e}"))?;

    let display_size = config.size;
    let render_pattern =
        |pattern: &PatternConfig, invert: bool| -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
            let mut img = pattern.generate(display_size)?;
            if invert {
                PatternConfig::apply_invert(&mut img);
            }
            Ok(img)
        };

    // Initial render
    let (mut current_pattern, mut current_invert) = source.current();
    let mut img = render_pattern(&current_pattern, current_invert)?;
    let mut buffer: Vec<u8> = vec![0; (config.width() * config.height() * 3) as usize];

    if !current_pattern.is_animated() {
        texture
            .update(None, img.as_raw(), (config.width() * 3) as usize)
            .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;
        // Stream the rendered image
        source.on_rendered(&img);
    }

    canvas.clear();
    canvas
        .copy(&texture, None, None)
        .map_err(|e| anyhow::anyhow!("Failed to copy texture: {e}"))?;
    canvas.present();

    // FPS tracking
    let mut frame_count = 0u64;
    let mut last_fps_report = Instant::now();

    loop {
        // Handle SDL events
        for event in event_pump.poll_iter() {
            use sdl2::event::Event;
            use sdl2::event::WindowEvent;
            use sdl2::keyboard::Keycode;
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape | Keycode::Q),
                    ..
                } => {
                    if !config.lock_display {
                        return Ok(());
                    }
                    tracing::debug!("Quit event ignored (--lock-display active)");
                }
                // Re-assert fullscreen when focus is lost (e.g., system update UI pops up).
                //
                // Alternative approaches if this isn't sufficient:
                // 1. Periodic raise(): Call canvas.window_mut().raise() every N frames
                //    regardless of focus events (brute force, may cause flicker)
                // 2. True exclusive fullscreen: Use .fullscreen() instead of .fullscreen_desktop()
                //    at window creation - takes exclusive display control, but riskier if app crashes
                // 3. API endpoint: Add /refocus HTTP endpoint to trigger raise() on demand
                // 4. set_fullscreen(): Re-apply fullscreen_desktop mode after focus loss:
                //    canvas.window_mut().set_fullscreen(sdl2::video::FullscreenType::Desktop)
                //
                // Note: On Wayland, compositors may ignore raise() for security reasons.
                Event::Window {
                    win_event: WindowEvent::FocusLost,
                    ..
                } => {
                    tracing::warn!("Window focus lost, attempting to reclaim");
                    canvas.window_mut().raise();
                }
                _ => {}
            }
        }

        // Check for updates from external source (web API, watchdog)
        if let Some(rx) = source.update_receiver() {
            if rx.try_recv().is_ok() {
                let (new_pattern, new_invert) = source.current();

                // Check if pattern actually changed
                if !patterns_equal(&current_pattern, &new_pattern) || current_invert != new_invert {
                    current_pattern = new_pattern;
                    current_invert = new_invert;

                    if !current_pattern.is_animated() {
                        img = render_pattern(&current_pattern, current_invert)?;
                        texture
                            .update(None, img.as_raw(), (config.width() * 3) as usize)
                            .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;
                        // Stream the rendered image
                        source.on_rendered(&img);
                    }
                }
            }
        }

        // Handle animated patterns (render every frame)
        if current_pattern.is_animated() {
            current_pattern.generate_into_buffer(&mut buffer, display_size);

            let buffer_to_use: &[u8] = if current_invert {
                // Invert in place for animated patterns
                for b in buffer.iter_mut() {
                    *b = 255 - *b;
                }
                &buffer
            } else {
                &buffer
            };

            texture
                .update(None, buffer_to_use, (config.width() * 3) as usize)
                .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;

            // Stream the rendered image (rate-limited in on_rendered)
            if let Some(anim_img) =
                ImageBuffer::from_raw(config.width(), config.height(), buffer_to_use.to_vec())
            {
                source.on_rendered(&anim_img);
            }
        }

        // Render
        canvas.clear();
        canvas
            .copy(&texture, None, None)
            .map_err(|e| anyhow::anyhow!("Failed to copy texture: {e}"))?;
        canvas.present();

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

/// Check if two patterns are equal (for change detection).
/// This is a simple discriminant check - animated patterns always "change".
fn patterns_equal(a: &PatternConfig, b: &PatternConfig) -> bool {
    use std::mem::discriminant;

    // Different variants = different patterns
    if discriminant(a) != discriminant(b) {
        return false;
    }

    // Animated patterns are never "equal" - they need continuous updates
    if a.is_animated() {
        return false;
    }

    // For non-animated patterns, compare the actual values
    match (a, b) {
        (PatternConfig::Usaf, PatternConfig::Usaf) => true,
        (PatternConfig::Pixel, PatternConfig::Pixel) => true,
        (PatternConfig::Check { checker_size: a }, PatternConfig::Check { checker_size: b }) => {
            a == b
        }
        (PatternConfig::Uniform { level: a }, PatternConfig::Uniform { level: b }) => a == b,
        (PatternConfig::PixelGrid { spacing: a }, PatternConfig::PixelGrid { spacing: b }) => {
            a == b
        }
        (PatternConfig::SiemensStar { spokes: a }, PatternConfig::SiemensStar { spokes: b }) => {
            a == b
        }
        _ => false,
    }
}

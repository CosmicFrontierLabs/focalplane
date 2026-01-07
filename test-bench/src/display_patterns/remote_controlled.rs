//! Remote-controlled pattern that receives commands via ZMQ REQ/REP.
//!
//! Displays spots, grids, or uniform colors based on commands from
//! a desktop calibration controller.

use std::sync::Mutex;
use std::time::Instant;

use shared::image_size::PixelShape;
use shared::pattern_command::PatternCommand;

use super::shared::{compute_normalization_factor, render_gaussian_spot, BlendMode};

/// State for the remote-controlled pattern.
/// Commands are set externally via the ZMQ REP handler.
pub struct RemotePatternState {
    /// Currently active command
    current_command: PatternCommand,
    /// Time of last command update
    last_update: Instant,
    /// Precomputed normalization factor for current FWHM
    cached_norm_factor: f64,
    /// FWHM that the cached norm factor was computed for
    cached_fwhm: f64,
}

impl Default for RemotePatternState {
    fn default() -> Self {
        Self::new()
    }
}

impl RemotePatternState {
    /// Create a new remote pattern state.
    pub fn new() -> Self {
        Self {
            current_command: PatternCommand::default(),
            last_update: Instant::now(),
            cached_norm_factor: compute_normalization_factor(5.0, 255.0),
            cached_fwhm: 5.0,
        }
    }

    /// Set the current command (called from ZMQ REP handler).
    pub fn set_command(&mut self, cmd: PatternCommand) {
        self.current_command = cmd;
        self.last_update = Instant::now();
    }

    /// Get the current pattern command.
    pub fn current(&self) -> &PatternCommand {
        &self.current_command
    }

    /// Get time since last command update.
    pub fn time_since_update(&self) -> std::time::Duration {
        self.last_update.elapsed()
    }

    /// Get normalization factor for the given FWHM, using cache if possible.
    fn get_norm_factor(&mut self, fwhm: f64, intensity: f64) -> f64 {
        if (fwhm - self.cached_fwhm).abs() > 0.001 {
            self.cached_fwhm = fwhm;
            self.cached_norm_factor = compute_normalization_factor(fwhm, 255.0);
        }
        self.cached_norm_factor * intensity
    }
}

/// Render the current pattern command into a buffer.
pub fn generate_into_buffer(
    buffer: &mut [u8],
    size: PixelShape,
    state: &Mutex<RemotePatternState>,
) {
    let mut state = state.lock().unwrap();

    // Clear buffer to black first
    buffer.fill(0);

    // Clone the command to avoid borrow issues
    let cmd = state.current().clone();

    match cmd {
        PatternCommand::Spot {
            x,
            y,
            fwhm,
            intensity,
        } => {
            let norm = state.get_norm_factor(fwhm, intensity);
            render_gaussian_spot(
                buffer,
                size.width as u32,
                size.height as u32,
                x,
                y,
                fwhm,
                norm,
                BlendMode::Overwrite,
            );
        }

        PatternCommand::SpotGrid {
            positions,
            fwhm,
            intensity,
        } => {
            let norm = state.get_norm_factor(fwhm, intensity);
            for (x, y) in &positions {
                render_gaussian_spot(
                    buffer,
                    size.width as u32,
                    size.height as u32,
                    *x,
                    *y,
                    fwhm,
                    norm,
                    BlendMode::Additive,
                );
            }
        }

        PatternCommand::Uniform { level } => {
            buffer.fill(level);
        }

        PatternCommand::Clear => {
            // Already cleared above
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_remote_pattern_spot() {
        let state = Arc::new(Mutex::new(RemotePatternState::new()));

        // Set spot command
        state.lock().unwrap().set_command(PatternCommand::Spot {
            x: 512.0,
            y: 512.0,
            fwhm: 5.0,
            intensity: 1.0,
        });

        // Generate pattern
        let size = PixelShape {
            width: 1024,
            height: 1024,
        };
        let mut buffer = vec![0u8; 1024 * 1024 * 3];
        generate_into_buffer(&mut buffer, size, &state);

        // Center pixel should be bright
        let center_offset = (512 * 1024 + 512) * 3;
        assert!(buffer[center_offset] > 200, "Center pixel should be bright");
    }

    #[test]
    fn test_remote_pattern_uniform() {
        let state = Arc::new(Mutex::new(RemotePatternState::new()));

        state
            .lock()
            .unwrap()
            .set_command(PatternCommand::Uniform { level: 128 });

        let size = PixelShape {
            width: 100,
            height: 100,
        };
        let mut buffer = vec![0u8; 100 * 100 * 3];
        generate_into_buffer(&mut buffer, size, &state);

        // All pixels should be 128
        assert!(buffer.iter().all(|&b| b == 128));
    }

    #[test]
    fn test_remote_pattern_clear() {
        let state = Arc::new(Mutex::new(RemotePatternState::new()));

        // First set uniform
        state
            .lock()
            .unwrap()
            .set_command(PatternCommand::Uniform { level: 255 });

        // Then clear
        state.lock().unwrap().set_command(PatternCommand::Clear);

        let size = PixelShape {
            width: 100,
            height: 100,
        };
        let mut buffer = vec![255u8; 100 * 100 * 3];
        generate_into_buffer(&mut buffer, size, &state);

        // All pixels should be 0
        assert!(buffer.iter().all(|&b| b == 0));
    }
}

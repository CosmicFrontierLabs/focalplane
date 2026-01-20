//! Gyro data emitter for motion patterns.
//!
//! Emits simulated Exail Astrix NS gyro packets via FTDI RS-422 at 500Hz.
//! The gyro data is synchronized with motion patterns by querying their
//! MotionTrajectory for position, then converting to angle via plate scale.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytemuck::bytes_of;
use hardware::ftdi::{build_raw_packet, list_ftdi_devices};
use libftd2xx::{Ftdi, FtdiCommon};
use tracing::{debug, error, info, warn};

use crate::display_patterns::{MotionTrajectory, Position2D};

/// Gyro emitter configuration.
#[derive(Debug, Clone)]
pub struct GyroEmitterConfig {
    /// FTDI device index (0 = first device)
    pub device_index: i32,
    /// Baud rate in bits per second (default: 1_000_000 for RS-422)
    pub baud_rate: u32,
    /// Packet emission rate in Hz (default: 500)
    pub rate_hz: u32,
    /// Remote terminal address for Exail protocol
    pub address: u8,
    /// Use BASE variant message format
    pub base_variant: bool,
}

impl Default for GyroEmitterConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            baud_rate: 1_000_000,
            rate_hz: 500,
            address: 18,
            base_variant: false,
        }
    }
}

/// Runtime parameters for gyro emission.
#[derive(Debug, Clone, Copy)]
pub struct GyroEmissionParams {
    /// Plate scale in arcsec/pixel for converting position to angle
    pub plate_scale_arcsec_per_px: f64,
    /// Display width for position calculation
    pub display_width: u32,
    /// Display height for position calculation
    pub display_height: u32,
}

impl Default for GyroEmissionParams {
    fn default() -> Self {
        Self {
            plate_scale_arcsec_per_px: 1.0,
            display_width: 1920,
            display_height: 1080,
        }
    }
}

/// Trait object wrapper for MotionTrajectory to allow dynamic dispatch.
pub trait PositionSource: Send + Sync {
    fn position_at(&self, elapsed: Duration, width: u32, height: u32) -> Position2D;
}

impl<T: MotionTrajectory + Send + Sync> PositionSource for T {
    fn position_at(&self, elapsed: Duration, width: u32, height: u32) -> Position2D {
        MotionTrajectory::position_at(self, elapsed, width, height)
    }
}

/// Handle to control the gyro emitter thread.
pub struct GyroEmitterHandle {
    /// Enable/disable emission
    pub enabled: Arc<AtomicBool>,
    /// Current emission parameters
    pub params: Arc<std::sync::Mutex<GyroEmissionParams>>,
    /// Position source (pattern)
    pub position_source: Arc<std::sync::Mutex<Option<Arc<dyn PositionSource>>>>,
    /// Start time for elapsed calculation
    pub start_time: Arc<std::sync::Mutex<Instant>>,
    /// Thread join handle
    join_handle: Option<std::thread::JoinHandle<()>>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

impl GyroEmitterHandle {
    /// Enable gyro emission.
    pub fn enable(&self) {
        // Reset start time when enabling
        *self.start_time.lock().unwrap() = Instant::now();
        self.enabled.store(true, Ordering::SeqCst);
        info!("Gyro emission enabled");
    }

    /// Disable gyro emission.
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
        info!("Gyro emission disabled");
    }

    /// Check if emission is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Set the position source (pattern).
    pub fn set_position_source(&self, source: Arc<dyn PositionSource>) {
        *self.position_source.lock().unwrap() = Some(source);
    }

    /// Update emission parameters.
    pub fn set_params(&self, params: GyroEmissionParams) {
        *self.params.lock().unwrap() = params;
        debug!(
            "Gyro params: plate_scale={:.3} arcsec/px, display={}x{}",
            params.plate_scale_arcsec_per_px, params.display_width, params.display_height
        );
    }

    /// Signal shutdown and wait for thread to finish.
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for GyroEmitterHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

/// Spawn the gyro emitter thread.
pub fn spawn_gyro_emitter(config: GyroEmitterConfig) -> anyhow::Result<GyroEmitterHandle> {
    let devices = list_ftdi_devices()?;
    if devices.is_empty() {
        anyhow::bail!("No FTDI devices found");
    }
    if config.device_index as usize >= devices.len() {
        anyhow::bail!(
            "FTDI device index {} out of range (found {} devices)",
            config.device_index,
            devices.len()
        );
    }

    let selected = &devices[config.device_index as usize];
    info!(
        "Gyro emitter using FTDI device {} - Serial: {:?}, Description: {:?}",
        config.device_index, selected.serial_number, selected.description
    );

    let enabled = Arc::new(AtomicBool::new(false));
    let params = Arc::new(std::sync::Mutex::new(GyroEmissionParams::default()));
    let position_source: Arc<std::sync::Mutex<Option<Arc<dyn PositionSource>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let start_time = Arc::new(std::sync::Mutex::new(Instant::now()));
    let shutdown = Arc::new(AtomicBool::new(false));

    let enabled_clone = enabled.clone();
    let params_clone = params.clone();
    let position_source_clone = position_source.clone();
    let start_time_clone = start_time.clone();
    let shutdown_clone = shutdown.clone();

    let join_handle = std::thread::spawn(move || {
        if let Err(e) = run_gyro_emitter_loop(
            config,
            enabled_clone,
            params_clone,
            position_source_clone,
            start_time_clone,
            shutdown_clone,
        ) {
            error!("Gyro emitter error: {e}");
        }
    });

    Ok(GyroEmitterHandle {
        enabled,
        params,
        position_source,
        start_time,
        join_handle: Some(join_handle),
        shutdown,
    })
}

/// Main gyro emitter loop.
fn run_gyro_emitter_loop(
    config: GyroEmitterConfig,
    enabled: Arc<AtomicBool>,
    params: Arc<std::sync::Mutex<GyroEmissionParams>>,
    position_source: Arc<std::sync::Mutex<Option<Arc<dyn PositionSource>>>>,
    start_time: Arc<std::sync::Mutex<Instant>>,
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let mut ft = Ftdi::with_index(config.device_index)
        .map_err(|e| anyhow::anyhow!("Failed to open FTDI device: {e}"))?;

    info!("Setting FTDI baud rate to {} bps", config.baud_rate);
    ft.set_baud_rate(config.baud_rate)
        .map_err(|e| anyhow::anyhow!("Failed to set baud rate: {e}"))?;

    ft.set_latency_timer(Duration::from_millis(1))
        .map_err(|e| anyhow::anyhow!("Failed to set latency timer: {e}"))?;
    ft.set_usb_parameters(64)
        .map_err(|e| anyhow::anyhow!("Failed to set USB parameters: {e}"))?;
    ft.set_timeouts(Duration::from_millis(100), Duration::from_millis(100))
        .map_err(|e| anyhow::anyhow!("Failed to set timeouts: {e}"))?;

    let interval = Duration::from_secs_f64(1.0 / config.rate_hz as f64);

    // Conversion factor: arcsec per LSB for Exail angle encoding
    const ARCSEC_PER_LSB: f64 = 0.00153;

    let loop_start = Instant::now();
    let mut time_counter: u32 = 0;
    let mut packet_count: u64 = 0;
    let mut next_send = Instant::now();

    info!("Gyro emitter ready at {}Hz", config.rate_hz);

    while !shutdown.load(Ordering::SeqCst) {
        let now = Instant::now();

        if now >= next_send {
            if enabled.load(Ordering::SeqCst) {
                let p = *params.lock().unwrap();
                let source = position_source.lock().unwrap().clone();
                let motion_start = *start_time.lock().unwrap();

                // Get position from pattern
                let position = if let Some(ref src) = source {
                    let elapsed = now.duration_since(motion_start);
                    src.position_at(elapsed, p.display_width, p.display_height)
                } else {
                    Position2D { x: 0.0, y: 0.0 }
                };

                // Convert position (pixels) to angle (arcsec), then to LSB
                // Exail gyro reports integrated angle in X, Y, Z
                // We map pattern X -> gyro Y (pitch), pattern Y -> gyro X (roll)
                // Z (yaw) stays at 0 for pure translational pattern motion
                let angle_x_arcsec = position.y * p.plate_scale_arcsec_per_px;
                let angle_y_arcsec = position.x * p.plate_scale_arcsec_per_px;

                let angle_x = (angle_x_arcsec / ARCSEC_PER_LSB) as i32 as u32;
                let angle_y = (angle_y_arcsec / ARCSEC_PER_LSB) as i32 as u32;
                let angle_z: u32 = 0;

                let packet = build_raw_packet(
                    config.address,
                    config.base_variant,
                    time_counter,
                    angle_x,
                    angle_y,
                    angle_z,
                );

                let bytes = bytes_of(&packet);
                if let Err(e) = ft.write_all(bytes) {
                    warn!("Failed to write gyro packet: {e}");
                }

                debug!(
                    "Gyro packet: time={}, pos=({:.1}, {:.1})px, angles=({}, {}, {})",
                    time_counter, position.x, position.y, angle_x, angle_y, angle_z
                );

                time_counter = time_counter.wrapping_add(1);
                packet_count += 1;

                if packet_count % 500 == 0 {
                    let elapsed = loop_start.elapsed().as_secs_f64();
                    let rate = packet_count as f64 / elapsed;
                    info!("Gyro: {packet_count} packets in {elapsed:.2}s ({rate:.1} Hz)");
                }
            }

            next_send += interval;

            if next_send < now {
                next_send = now + interval;
            }
        }

        std::hint::spin_loop();
    }

    info!("Gyro emitter shutdown");
    Ok(())
}

/// List available FTDI devices (for CLI help).
pub fn list_ftdi_devices_info() -> anyhow::Result<Vec<String>> {
    let devices = list_ftdi_devices()?;
    Ok(devices
        .iter()
        .enumerate()
        .map(|(i, dev)| {
            format!(
                "{}: Serial={:?}, Description={:?}",
                i, dev.serial_number, dev.description
            )
        })
        .collect())
}

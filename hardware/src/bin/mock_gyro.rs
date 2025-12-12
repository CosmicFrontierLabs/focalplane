//! Mock Exail Asterix NS gyro over FTDI RS-422
//!
//! Simulates gyro packet output at 500 Hz (2ms intervals) over a 1 Mbit/s RS-422 link.
//! Uses libftd2xx for low-latency USB-serial communication.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bytemuck::bytes_of;
use clap::Parser;
use libftd2xx::{Ftdi, FtdiCommon};
use log::{debug, info};

use hardware::ftdi::{build_full_packet, build_raw_packet, list_ftdi_devices, print_device_info};

#[derive(Parser, Debug)]
#[command(name = "mock_gyro")]
#[command(about = "Mock Exail Asterix NS gyro transmitter")]
struct Args {
    /// List connected FTDI devices and exit
    #[arg(short, long)]
    list: bool,

    /// FTDI device index (0 = first device)
    #[arg(short, long, default_value = "0")]
    device: i32,

    /// Baud rate in bits per second
    #[arg(short, long, default_value = "1000000")]
    baud: u32,

    /// Packet interval in milliseconds
    #[arg(short, long, default_value = "2")]
    interval_ms: u64,

    /// Message type to send: "raw", "filtered", or "full"
    #[arg(short, long, default_value = "full")]
    message_type: String,

    /// Remote terminal address (start_word)
    #[arg(long, default_value = "18")]
    address: u8,

    /// Use BASE variant (includes time_base field)
    #[arg(long)]
    base_variant: bool,

    /// Number of packets to send (0 = infinite)
    #[arg(short, long, default_value = "0")]
    count: u64,

    /// Simulated angular rate in arcsec/s for each axis
    #[arg(long, default_value = "0.0")]
    rate_arcsec_s: f64,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    let devices = list_ftdi_devices()?;

    if args.list {
        println!("Found {} FTDI device(s):", devices.len());
        for (i, dev) in devices.iter().enumerate() {
            print_device_info(i, dev);
        }
        return Ok(());
    }

    if devices.is_empty() {
        anyhow::bail!("No FTDI devices found");
    }

    if args.device as usize >= devices.len() {
        anyhow::bail!(
            "Device index {} out of range (found {} devices)",
            args.device,
            devices.len()
        );
    }

    let selected = &devices[args.device as usize];
    info!(
        "Opening FTDI device {} - Serial: {:?}, Description: {:?}",
        args.device, selected.serial_number, selected.description
    );

    let mut ft = Ftdi::with_index(args.device).context("Failed to open FTDI device")?;

    info!("Setting baud rate to {} bps", args.baud);
    ft.set_baud_rate(args.baud)
        .context("Failed to set baud rate")?;

    info!("Configuring for low latency");
    ft.set_latency_timer(Duration::from_millis(1))
        .context("Failed to set latency timer")?;
    ft.set_usb_parameters(64)
        .context("Failed to set USB parameters")?;

    ft.set_timeouts(Duration::from_millis(100), Duration::from_millis(100))
        .context("Failed to set timeouts")?;

    let interval = Duration::from_millis(args.interval_ms);
    let mut time_counter: u32 = 0;
    let mut packet_count: u64 = 0;

    // Angle increment per packet (rate_arcsec_s * interval_s / arcsec_per_lsb)
    // arcsec_per_lsb = 0.000048 (from exail::ARCSECONDS_PER_LSB)
    let arcsec_per_lsb: f64 = 0.000048;
    let angle_increment_per_packet =
        (args.rate_arcsec_s * (args.interval_ms as f64 / 1000.0) / arcsec_per_lsb) as u32;

    let mut angle_x: u32 = 0;
    let mut angle_y: u32 = 0;
    let mut angle_z: u32 = 0;

    info!(
        "Starting {} packet transmission at {}ms intervals",
        args.message_type, args.interval_ms
    );
    info!(
        "Angle increment per packet: {} LSB ({} arcsec/s)",
        angle_increment_per_packet, args.rate_arcsec_s
    );

    let start_time = Instant::now();
    let mut next_send = Instant::now();

    loop {
        let now = Instant::now();

        if now >= next_send {
            let bytes: Vec<u8> = match args.message_type.as_str() {
                "full" => {
                    let packet = build_full_packet(
                        args.address,
                        args.base_variant,
                        time_counter,
                        angle_x,
                        angle_y,
                        angle_z,
                    );
                    bytes_of(&packet).to_vec()
                }
                "raw" | "filtered" => {
                    let packet = build_raw_packet(
                        args.address,
                        args.base_variant,
                        time_counter,
                        angle_x,
                        angle_y,
                        angle_z,
                    );
                    bytes_of(&packet).to_vec()
                }
                _ => {
                    anyhow::bail!("Unknown message type: {}", args.message_type);
                }
            };

            ft.write_all(&bytes).context("Failed to write packet")?;

            debug!(
                "Sent {} byte packet, time={}, angles=({}, {}, {})",
                bytes.len(),
                time_counter,
                angle_x,
                angle_y,
                angle_z
            );

            time_counter = time_counter.wrapping_add(1);
            angle_x = angle_x.wrapping_add(angle_increment_per_packet);
            angle_y = angle_y.wrapping_add(angle_increment_per_packet);
            angle_z = angle_z.wrapping_add(angle_increment_per_packet);

            packet_count += 1;
            next_send += interval;

            if args.count > 0 && packet_count >= args.count {
                break;
            }

            if packet_count % 500 == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let rate = packet_count as f64 / elapsed;
                info!("Sent {packet_count} packets in {elapsed:.2}s ({rate:.1} Hz)");
            }
        }

        // Spin-wait for tight timing - no sleep for sub-millisecond precision
        std::hint::spin_loop();
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let rate = packet_count as f64 / elapsed;
    info!("Complete: {packet_count} packets in {elapsed:.2}s ({rate:.1} Hz)");

    Ok(())
}

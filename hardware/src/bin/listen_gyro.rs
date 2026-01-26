//! Listen for Exail Asterix NS gyro data over FTDI RS-422 or serial port
//!
//! Receives gyro packets at up to 500 Hz over a 1 Mbit/s RS-422 link.
//! Validates checksums and reports packet rate and corruption statistics.
//! Supports both FTDI USB converters and direct serial ports (Neutralino differential pair).

use std::io::Read;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use libftd2xx::{Ftdi, FtdiCommon};
use log::{debug, info, warn};
use serialport::SerialPort;

use hardware::exail::{parse, verify_checksum_bytes, GyroMessage};

#[derive(Parser, Debug)]
#[command(name = "listen_gyro")]
#[command(about = "Exail Asterix NS gyro receiver and validator")]
struct Args {
    /// FTDI device options (use --list-ftdi to see available devices)
    #[command(flatten)]
    ftdi: hardware::ftdi::FtdiArgs,

    /// Serial port path (e.g., /dev/ttyTHS1). Mutually exclusive with --ftdi-device.
    #[arg(long, conflicts_with = "ftdi_device")]
    serial: Option<String>,

    /// Number of packets to receive (0 = infinite)
    #[arg(short, long, default_value = "0")]
    count: u64,

    /// Report statistics interval in packets
    #[arg(long, default_value = "500")]
    report_interval: u64,
}

/// Trait for reading bytes from either FTDI or serial port
trait ByteReader {
    fn read_bytes(&mut self, buffer: &mut [u8]) -> Result<usize>;
}

/// FTDI byte reader wrapper
struct FtdiReader {
    ftdi: Ftdi,
}

impl ByteReader for FtdiReader {
    fn read_bytes(&mut self, buffer: &mut [u8]) -> Result<usize> {
        self.ftdi
            .read(buffer)
            .context("Failed to read from FTDI device")
    }
}

/// Serial port byte reader wrapper
struct SerialReader {
    port: Box<dyn SerialPort>,
}

impl ByteReader for SerialReader {
    fn read_bytes(&mut self, buffer: &mut [u8]) -> Result<usize> {
        self.port
            .read(buffer)
            .context("Failed to read from serial port")
    }
}

/// Packet framing state machine
struct PacketFramer {
    buffer: Vec<u8>,
    max_packet_size: usize,
}

impl PacketFramer {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(128),
            max_packet_size: 66, // FullGyroData is largest at 66 bytes
        }
    }

    /// Add incoming bytes and try to extract complete packets.
    ///
    /// Returns a vector of complete packet byte slices.
    fn push_bytes(&mut self, data: &[u8]) -> Vec<Vec<u8>> {
        self.buffer.extend_from_slice(data);
        let mut packets = Vec::new();

        // Try to find and extract packets from buffer
        while let Some(packet) = self.try_extract_packet() {
            packets.push(packet);
        }

        // Prevent buffer from growing unbounded if we're getting garbage
        if self.buffer.len() > self.max_packet_size * 2 {
            warn!(
                "Buffer overflow ({} bytes), discarding oldest data",
                self.buffer.len()
            );
            self.buffer.drain(0..self.max_packet_size);
        }

        packets
    }

    /// Try to extract one complete packet from the buffer.
    ///
    /// Returns Some(packet_bytes) if a valid packet is found and removed from buffer.
    fn try_extract_packet(&mut self) -> Option<Vec<u8>> {
        if self.buffer.len() < 2 {
            return None;
        }

        // Scan for potential packet start
        // Frame ID is in byte 1, mask lower 5 bits
        for start_idx in 0..self.buffer.len() - 1 {
            let frame_id = self.buffer[start_idx + 1] & 0x1F;

            // Determine expected packet size based on frame ID
            let expected_size = match frame_id {
                // RAW_GYRO_BASE = 18, RAW_GYRO = 2
                // FILTERED_GYRO_BASE = 20, FILTERED_GYRO = 4
                2 | 4 | 18 | 20 => 26,
                // FULL_GYRO_BASE = 21, FULL_GYRO = 5
                5 | 21 => 66,
                _ => continue, // Unknown frame ID, keep scanning
            };

            // Check if we have enough bytes for this packet
            if start_idx + expected_size > self.buffer.len() {
                // Not enough data yet, but found valid header
                if start_idx == 0 {
                    return None; // Wait for more data
                }
                // Discard bytes before this potential packet and wait
                self.buffer.drain(0..start_idx);
                return None;
            }

            // Extract potential packet
            let packet_bytes = self.buffer[start_idx..start_idx + expected_size].to_vec();

            // Try to parse it
            if parse(&packet_bytes).is_ok() {
                // Valid packet! Remove it from buffer
                self.buffer.drain(0..start_idx + expected_size);
                return Some(packet_bytes);
            }
        }

        // No valid packet found, discard first byte and try again next time
        if !self.buffer.is_empty() {
            self.buffer.remove(0);
        }

        None
    }
}

/// Statistics tracker
struct Statistics {
    total_packets: u64,
    valid_checksums: u64,
    invalid_checksums: u64,
    raw_packets: u64,
    filtered_packets: u64,
    full_packets: u64,
    start_time: Instant,
    last_report_time: Instant,
    last_report_count: u64,
}

impl Statistics {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            total_packets: 0,
            valid_checksums: 0,
            invalid_checksums: 0,
            raw_packets: 0,
            filtered_packets: 0,
            full_packets: 0,
            start_time: now,
            last_report_time: now,
            last_report_count: 0,
        }
    }

    fn record_packet(&mut self, packet_bytes: &[u8], message: &GyroMessage) {
        self.total_packets += 1;

        // Check checksum
        if verify_checksum_bytes(packet_bytes) {
            self.valid_checksums += 1;
        } else {
            self.invalid_checksums += 1;
        }

        // Count packet type
        match message {
            GyroMessage::Raw(_) => self.raw_packets += 1,
            GyroMessage::Filtered(_) => self.filtered_packets += 1,
            GyroMessage::Full(_) => self.full_packets += 1,
        }
    }

    fn report(&mut self) {
        let now = Instant::now();
        let total_elapsed = self.start_time.elapsed().as_secs_f64();
        let interval_elapsed = now.duration_since(self.last_report_time).as_secs_f64();
        let interval_packets = self.total_packets - self.last_report_count;

        let total_rate = self.total_packets as f64 / total_elapsed;
        let interval_rate = interval_packets as f64 / interval_elapsed;
        let checksum_pass_pct = if self.total_packets > 0 {
            100.0 * self.valid_checksums as f64 / self.total_packets as f64
        } else {
            0.0
        };

        info!(
            "Packets: {} | Rate: {:.1} Hz (interval: {:.1} Hz) | Checksum pass: {:.1}% ({}/{}) | Types: R={} F={} Full={}",
            self.total_packets,
            total_rate,
            interval_rate,
            checksum_pass_pct,
            self.valid_checksums,
            self.total_packets,
            self.raw_packets,
            self.filtered_packets,
            self.full_packets
        );

        self.last_report_time = now;
        self.last_report_count = self.total_packets;
    }

    fn final_report(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.total_packets as f64 / elapsed;
        let checksum_pass_pct = if self.total_packets > 0 {
            100.0 * self.valid_checksums as f64 / self.total_packets as f64
        } else {
            0.0
        };

        info!("=== Final Statistics ===");
        info!("Total packets: {}", self.total_packets);
        info!("Duration: {elapsed:.2}s");
        info!("Average rate: {rate:.1} Hz");
        info!(
            "Checksum pass: {:.1}% ({}/{})",
            checksum_pass_pct, self.valid_checksums, self.total_packets
        );
        info!("Checksum fail: {}", self.invalid_checksums);
        info!(
            "Packet types: Raw={}, Filtered={}, Full={}",
            self.raw_packets, self.filtered_packets, self.full_packets
        );
    }
}

fn open_ftdi_reader(args: &Args) -> Result<FtdiReader> {
    let ftdi = args.ftdi.open_device_or_default()?;
    Ok(FtdiReader { ftdi })
}

fn open_serial_reader(path: &str, baud: u32) -> Result<SerialReader> {
    info!("Opening serial port: {path} at {baud} bps");

    let port = serialport::new(path, baud)
        .timeout(Duration::from_millis(100))
        .open()
        .with_context(|| format!("Failed to open serial port {path}"))?;

    Ok(SerialReader { port })
}

fn run_receiver(mut reader: impl ByteReader, args: &Args) -> Result<()> {
    info!("Starting gyro packet receiver...");
    info!(
        "Will report statistics every {} packets",
        args.report_interval
    );

    let mut framer = PacketFramer::new();
    let mut stats = Statistics::new();
    let mut read_buffer = vec![0u8; 1024];

    loop {
        // Read available data
        match reader.read_bytes(&mut read_buffer) {
            Ok(bytes_read) if bytes_read > 0 => {
                debug!("Read {bytes_read} bytes");

                // Process incoming bytes through framer
                let packets = framer.push_bytes(&read_buffer[..bytes_read]);

                for packet_bytes in packets {
                    match parse(&packet_bytes) {
                        Ok(message) => {
                            stats.record_packet(&packet_bytes, &message);

                            debug!(
                                "Received {:?} packet, {} bytes, checksum {}",
                                match message {
                                    GyroMessage::Raw(_) => "Raw",
                                    GyroMessage::Filtered(_) => "Filtered",
                                    GyroMessage::Full(_) => "Full",
                                },
                                packet_bytes.len(),
                                if verify_checksum_bytes(&packet_bytes) {
                                    "OK"
                                } else {
                                    "FAIL"
                                }
                            );

                            // Report statistics periodically
                            if stats.total_packets % args.report_interval == 0 {
                                stats.report();
                            }

                            // Check if we've hit the count limit
                            if args.count > 0 && stats.total_packets >= args.count {
                                stats.final_report();
                                return Ok(());
                            }
                        }
                        Err(e) => {
                            warn!("Failed to parse packet: {e:?}");
                        }
                    }
                }
            }
            Ok(_) => {
                // No data, brief sleep to avoid busy-wait
                std::thread::sleep(Duration::from_micros(100));
            }
            Err(e) => {
                warn!("Read error: {e:?}");
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    // Handle --list-ftdi flag
    if args.ftdi.handle_list_ftdi()? {
        return Ok(());
    }

    // Determine input source and run receiver
    if let Some(serial_path) = &args.serial {
        let reader = open_serial_reader(serial_path, args.ftdi.ftdi_baud)?;
        run_receiver(reader, &args)
    } else {
        let reader = open_ftdi_reader(&args)?;
        run_receiver(reader, &args)
    }
}

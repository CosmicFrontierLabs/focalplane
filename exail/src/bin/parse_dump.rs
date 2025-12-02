use std::env;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

use exail::{verify_checksum, FullGyroData, GyroData, TemperatureSensor};
use log::{error, info, warn};

const MSG_SIZE: usize = 66;
const SYNC_PATTERN: [u8; 2] = [0x87, 0x15];

/// Parsed record - either valid data or skipped bytes
enum Record {
    Data(FullGyroData),
    Skipped(Vec<u8>),
}

/// Pending skip event waiting for next valid timetag
struct PendingSkip {
    offset: usize,
    bytes: Vec<u8>,
    good_records_since_last_skip: usize,
    last_valid_timetag: Option<u16>,
}

/// Search forward from `start` for the sync pattern 0x87 0x15
fn find_sync(data: &[u8], start: usize) -> Option<usize> {
    (start..data.len().saturating_sub(1))
        .find(|&i| data[i] == SYNC_PATTERN[0] && data[i + 1] == SYNC_PATTERN[1])
}

/// Log skipped bytes with offset, timetags, and count of good records
fn log_skipped(skip: &PendingSkip, next_valid_timetag: Option<u16>, segment_number: usize) {
    let offset = skip.offset;
    let end_offset = offset + skip.bytes.len();
    let hex: String = skip
        .bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ");

    let last_time_str = skip
        .last_valid_timetag
        .map(|t| format!("{t:>10}"))
        .unwrap_or_else(|| format!("{:>10}", "none"));
    let next_time_str = next_valid_timetag
        .map(|t| format!("{t:>10}"))
        .unwrap_or_else(|| format!("{:>10}", "none"));

    let delta_str = match (skip.last_valid_timetag, next_valid_timetag) {
        (Some(last), Some(next)) => {
            let delta = next as i32 - last as i32;
            format!("{delta:>12}")
        }
        _ => format!("{:>12}", "n/a"),
    };

    warn!(
        "Corrupted segment {:>3}: skipped {:>3} bytes at offset {:>12} (0x{:>8x}) - {:>12} (0x{:>8x}) after {:>7} good records, timetag {} -> {} (delta {}): {}",
        segment_number,
        skip.bytes.len(),
        offset,
        offset,
        end_offset,
        end_offset,
        skip.good_records_since_last_skip,
        last_time_str,
        next_time_str,
        delta_str,
        hex
    );
}

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <dump_file> <output_csv>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    let output_path = &args[2];

    let mut file = File::open(path).expect("Failed to open file");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read file");

    info!("Read {} bytes from {}", data.len(), path);

    let mut records: Vec<Record> = Vec::new();

    // Find first sync and collect any leading bytes as skipped
    let Some(first_sync) = find_sync(&data, 0) else {
        error!("No sync pattern found in file");
        std::process::exit(1);
    };
    let mut good_records_since_last_skip = 0usize;
    let mut last_valid_timetag: Option<u16> = None;
    let mut pending_skip: Option<PendingSkip> = None;
    let mut corrupted_segment_number = 0usize;

    if first_sync > 0 {
        pending_skip = Some(PendingSkip {
            offset: 0,
            bytes: data[0..first_sync].to_vec(),
            good_records_since_last_skip,
            last_valid_timetag,
        });
        records.push(Record::Skipped(data[0..first_sync].to_vec()));
    }
    let mut pos = first_sync;

    while pos + MSG_SIZE <= data.len() {
        let msg_data = &data[pos..pos + MSG_SIZE];

        if !verify_checksum(msg_data) {
            // If there's already a pending skip, log it now with no next timetag (back-to-back skips)
            if let Some(prev_skip) = pending_skip.take() {
                log_skipped(&prev_skip, None, corrupted_segment_number);
                corrupted_segment_number += 1;
            }

            // Search forward for next sync pattern
            let search_start = pos + 1;
            if let Some(next_sync) = find_sync(&data, search_start) {
                pending_skip = Some(PendingSkip {
                    offset: pos,
                    bytes: data[pos..next_sync].to_vec(),
                    good_records_since_last_skip,
                    last_valid_timetag,
                });
                records.push(Record::Skipped(data[pos..next_sync].to_vec()));
                good_records_since_last_skip = 0;
                pos = next_sync;
                continue;
            } else {
                // No more syncs, collect remaining as skipped - log immediately with no next timetag
                let skip = PendingSkip {
                    offset: pos,
                    bytes: data[pos..].to_vec(),
                    good_records_since_last_skip,
                    last_valid_timetag,
                };
                log_skipped(&skip, None, corrupted_segment_number);
                corrupted_segment_number += 1;
                records.push(Record::Skipped(data[pos..].to_vec()));
                break;
            }
        }

        let msg: FullGyroData = *bytemuck::from_bytes(msg_data);
        let (current_timetag, _timebase) = msg.gyro_time.as_gyro_time_tag_and_base();

        // If there's a pending skip, log it now that we have the next valid timetag
        if let Some(skip) = pending_skip.take() {
            log_skipped(&skip, Some(current_timetag), corrupted_segment_number);
            corrupted_segment_number += 1;
        }

        records.push(Record::Data(msg));
        last_valid_timetag = Some(current_timetag);
        good_records_since_last_skip += 1;
        pos += MSG_SIZE;
    }

    // Collect trailing bytes if any
    if pos < data.len() {
        let skip = PendingSkip {
            offset: pos,
            bytes: data[pos..].to_vec(),
            good_records_since_last_skip,
            last_valid_timetag,
        };
        log_skipped(&skip, None, corrupted_segment_number);
        records.push(Record::Skipped(data[pos..].to_vec()));
    }

    // Count stats
    let data_count = records
        .iter()
        .filter(|r| matches!(r, Record::Data(_)))
        .count();
    let skip_count = records
        .iter()
        .filter(|r| matches!(r, Record::Skipped(_)))
        .count();
    let skip_bytes: usize = records
        .iter()
        .filter_map(|r| match r {
            Record::Skipped(b) => Some(b.len()),
            _ => None,
        })
        .sum();

    info!(
        "Parsed {data_count} data records, {skip_count} skip records ({skip_bytes} bytes skipped)"
    );

    // Write CSV with buffered I/O
    let file = File::create(output_path).expect("Failed to create output file");
    let mut out = BufWriter::new(file);

    // Header - temperature fields in Celsius, angles in arcseconds
    writeln!(out, "type,start_word,message_id,gyro_time,raw_ang_x_arcsec,raw_ang_y_arcsec,raw_ang_z_arcsec,fil_ang_x_arcsec,fil_ang_y_arcsec,fil_ang_z_arcsec,so_in_cur,cur_com,pow_meas_x,pow_meas_y,pow_meas_z,vpi_x,vpi_y,vpi_z,ramp_x,ramp_y,ramp_z,board_temp_c,sia_fil_temp_c,org_fil_temp_c,inter_temp_c,health_status,checksum,skipped_hex").unwrap();

    for record in &records {
        match record {
            Record::Data(msg) => {
                // Copy all packed fields to locals
                let start_word = { msg.start_word };
                let message_id = msg.message_id();
                let gyro_time = { msg.gyro_time }.as_time_tag();
                let so_in_cur = { msg.so_in_cur };
                let cur_com = { msg.cur_com };
                let pow_meas_x = { msg.pow_meas_x };
                let pow_meas_y = { msg.pow_meas_y };
                let pow_meas_z = { msg.pow_meas_z };
                let vpi_x = { msg.vpi_x };
                let vpi_y = { msg.vpi_y };
                let vpi_z = { msg.vpi_z };
                let ramp_x = { msg.ramp_x };
                let ramp_y = { msg.ramp_y };
                let ramp_z = { msg.ramp_z };
                let health_status = { msg.health_status }.bits();
                let checksum = { msg.checksum };

                // Decode angles using GyroData trait
                let raw_ang_x_arcsec = msg
                    .raw_angle_data()
                    .map(|a| format!("{:.6}", a.x))
                    .unwrap_or_default();
                let raw_ang_y_arcsec = msg
                    .raw_angle_data()
                    .map(|a| format!("{:.6}", a.y))
                    .unwrap_or_default();
                let raw_ang_z_arcsec = msg
                    .raw_angle_data()
                    .map(|a| format!("{:.6}", a.z))
                    .unwrap_or_default();

                let fil_ang_x_arcsec = msg
                    .filtered_angle_data()
                    .map(|a| format!("{:.6}", a.x))
                    .unwrap_or_default();
                let fil_ang_y_arcsec = msg
                    .filtered_angle_data()
                    .map(|a| format!("{:.6}", a.y))
                    .unwrap_or_default();
                let fil_ang_z_arcsec = msg
                    .filtered_angle_data()
                    .map(|a| format!("{:.6}", a.z))
                    .unwrap_or_default();

                // Decode temperatures using GyroData trait
                let temp_readings = msg.temperature_readings();

                // Extract temperatures by sensor (will be empty string if not present)
                let mut board_temp_c = String::new();
                let mut sia_fil_temp_c = String::new();
                let mut org_fil_temp_c = String::new();
                let mut inter_temp_c = String::new();

                for reading in temp_readings {
                    let temp_str = match reading.celsius {
                        Some(celsius) => format!("{celsius:.2}"),
                        None => String::from("NaN"),
                    };

                    match reading.sensor {
                        TemperatureSensor::Board => board_temp_c = temp_str,
                        TemperatureSensor::SiaFilter => sia_fil_temp_c = temp_str,
                        TemperatureSensor::Organizer => org_fil_temp_c = temp_str,
                        TemperatureSensor::Interface => inter_temp_c = temp_str,
                    }
                }

                writeln!(
                    out,
                    "data,{start_word},{message_id},{gyro_time},{raw_ang_x_arcsec},{raw_ang_y_arcsec},{raw_ang_z_arcsec},{fil_ang_x_arcsec},{fil_ang_y_arcsec},{fil_ang_z_arcsec},{so_in_cur},{cur_com},{pow_meas_x},{pow_meas_y},{pow_meas_z},{vpi_x},{vpi_y},{vpi_z},{ramp_x},{ramp_y},{ramp_z},{board_temp_c},{sia_fil_temp_c},{org_fil_temp_c},{inter_temp_c},{health_status},{checksum},",
                ).unwrap();
            }
            Record::Skipped(bytes) => {
                let hex: String = bytes
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                writeln!(out, "skipped,,,,,,,,,,,,,,,,,,,,,,,,,,,{hex}").unwrap();
            }
        }
    }

    info!("Wrote {output_path}");
}

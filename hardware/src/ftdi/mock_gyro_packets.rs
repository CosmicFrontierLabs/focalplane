//! Mock Exail Asterix NS gyro packet generation
//!
//! Generates valid gyro packets for hardware-in-the-loop testing.

use bytemuck::bytes_of;

use crate::exail::messages::frame_id;
use crate::exail::{compute_checksum, FullGyroData, GyroTime, HealthStatus, RawGyroInertialData};

/// Health status indicating all channels valid and operational
pub const HEALTH_OK: HealthStatus = HealthStatus::VALIDITY_X
    .union(HealthStatus::VALIDITY_Y)
    .union(HealthStatus::VALIDITY_Z)
    .union(HealthStatus::FOG_VALIDITY);

/// Build a FullGyroData packet with the given parameters.
///
/// Creates a complete 66-byte packet with valid checksum.
/// Raw and filtered angles are set to the same values.
/// Temperature sensors are set to nominal values (16384 = ~25°C).
///
/// # Arguments
/// * `address` - Remote terminal address (start_word)
/// * `base_variant` - If true, use FULL_GYRO_BASE message ID, otherwise FULL_GYRO
/// * `time_counter` - Time value to encode in the packet
/// * `angle_x` - Raw angle measurement for X axis
/// * `angle_y` - Raw angle measurement for Y axis
/// * `angle_z` - Raw angle measurement for Z axis
pub fn build_full_packet(
    address: u8,
    base_variant: bool,
    time_counter: u32,
    angle_x: u32,
    angle_y: u32,
    angle_z: u32,
) -> FullGyroData {
    let message_id = if base_variant {
        frame_id::FULL_GYRO_BASE
    } else {
        frame_id::FULL_GYRO
    };

    let mut packet = FullGyroData {
        start_word: address,
        message_id,
        gyro_time: GyroTime::from_bytes(time_counter.to_le_bytes()),
        raw_ang_x: angle_x,
        raw_ang_y: angle_y,
        raw_ang_z: angle_z,
        fil_ang_x: angle_x,
        fil_ang_y: angle_y,
        fil_ang_z: angle_z,
        so_in_cur: 0x1000,
        cur_com: 0x0800,
        pow_meas_x: 0x2000,
        pow_meas_y: 0x2000,
        pow_meas_z: 0x2000,
        vpi_x: 0,
        vpi_y: 0,
        vpi_z: 0,
        ramp_x: 0,
        ramp_y: 0,
        ramp_z: 0,
        board_temp: 16384,
        sia_fil_temp: 16384,
        org_fil_temp: 16384,
        inter_temp: 16384,
        health_status: HEALTH_OK,
        checksum: 0,
    };

    let bytes = bytes_of(&packet);
    let checksum = compute_checksum(&bytes[..bytes.len() - 2]);
    packet.checksum = checksum;

    packet
}

/// Build a RawGyroInertialData packet with the given parameters.
///
/// Creates a complete 26-byte packet with valid checksum.
/// Temperature sensor is set to nominal value (16384 = ~25°C).
///
/// # Arguments
/// * `address` - Remote terminal address (start_word)
/// * `base_variant` - If true, use RAW_GYRO_BASE message ID, otherwise RAW_GYRO
/// * `time_counter` - Time value to encode in the packet
/// * `angle_x` - Raw angle measurement for X axis
/// * `angle_y` - Raw angle measurement for Y axis
/// * `angle_z` - Raw angle measurement for Z axis
pub fn build_raw_packet(
    address: u8,
    base_variant: bool,
    time_counter: u32,
    angle_x: u32,
    angle_y: u32,
    angle_z: u32,
) -> RawGyroInertialData {
    let message_id = if base_variant {
        frame_id::RAW_GYRO_BASE
    } else {
        frame_id::RAW_GYRO
    };

    let mut packet = RawGyroInertialData {
        start_word: address,
        message_id,
        gyro_time: GyroTime::from_bytes(time_counter.to_le_bytes()),
        angle_x,
        angle_y,
        angle_z,
        sia_fil_temp: 16384,
        health_status: HEALTH_OK,
        checksum: 0,
    };

    let bytes = bytes_of(&packet);
    let checksum = compute_checksum(&bytes[..bytes.len() - 2]);
    packet.checksum = checksum;

    packet
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exail::{verify_checksum, GyroData};

    #[test]
    fn test_build_full_packet_size() {
        let packet = build_full_packet(0x12, false, 0, 0, 0, 0);
        let bytes = bytes_of(&packet);
        assert_eq!(bytes.len(), FullGyroData::PACKET_SIZE);
        assert_eq!(bytes.len(), 66);
    }

    #[test]
    fn test_build_full_packet_checksum_valid() {
        let packet = build_full_packet(0x12, false, 12345, 100, 200, 300);
        let bytes = bytes_of(&packet);
        assert!(verify_checksum(bytes));
    }

    #[test]
    fn test_build_full_packet_message_id_regular() {
        let packet = build_full_packet(0x12, false, 0, 0, 0, 0);
        assert_eq!(packet.frame_id(), frame_id::FULL_GYRO);
        assert!(!packet.is_base_variant());
    }

    #[test]
    fn test_build_full_packet_message_id_base() {
        let packet = build_full_packet(0x12, true, 0, 0, 0, 0);
        assert_eq!(packet.frame_id(), frame_id::FULL_GYRO_BASE);
        assert!(packet.is_base_variant());
    }

    #[test]
    fn test_build_full_packet_address() {
        let packet = build_full_packet(0x18, false, 0, 0, 0, 0);
        assert_eq!(packet.start_word, 0x18);
    }

    #[test]
    fn test_build_full_packet_angles() {
        let packet = build_full_packet(0x12, false, 0, 1000, 2000, 3000);
        // Copy from packed struct to avoid alignment issues
        let raw_x = packet.raw_ang_x;
        let raw_y = packet.raw_ang_y;
        let raw_z = packet.raw_ang_z;
        let fil_x = packet.fil_ang_x;
        let fil_y = packet.fil_ang_y;
        let fil_z = packet.fil_ang_z;
        assert_eq!(raw_x, 1000);
        assert_eq!(raw_y, 2000);
        assert_eq!(raw_z, 3000);
        assert_eq!(fil_x, 1000);
        assert_eq!(fil_y, 2000);
        assert_eq!(fil_z, 3000);
    }

    #[test]
    fn test_build_full_packet_time() {
        let packet = build_full_packet(0x12, false, 0xDEADBEEF, 0, 0, 0);
        assert_eq!(packet.gyro_time.as_time_tag(), 0xDEADBEEF);
    }

    #[test]
    fn test_build_full_packet_health_status() {
        let packet = build_full_packet(0x12, false, 0, 0, 0, 0);
        // Copy from packed struct to avoid alignment issues
        let health = packet.health_status;
        assert!(health.is_fog_valid());
        assert!(health.is_x_valid());
        assert!(health.is_y_valid());
        assert!(health.is_z_valid());
        assert!(health.is_gyro_mode());
    }

    #[test]
    fn test_build_raw_packet_size() {
        let packet = build_raw_packet(0x12, false, 0, 0, 0, 0);
        let bytes = bytes_of(&packet);
        assert_eq!(bytes.len(), RawGyroInertialData::PACKET_SIZE);
        assert_eq!(bytes.len(), 26);
    }

    #[test]
    fn test_build_raw_packet_checksum_valid() {
        let packet = build_raw_packet(0x12, false, 12345, 100, 200, 300);
        let bytes = bytes_of(&packet);
        assert!(verify_checksum(bytes));
    }

    #[test]
    fn test_build_raw_packet_message_id_regular() {
        let packet = build_raw_packet(0x12, false, 0, 0, 0, 0);
        assert_eq!(packet.frame_id(), frame_id::RAW_GYRO);
        assert!(!packet.is_base_variant());
    }

    #[test]
    fn test_build_raw_packet_message_id_base() {
        let packet = build_raw_packet(0x12, true, 0, 0, 0, 0);
        assert_eq!(packet.frame_id(), frame_id::RAW_GYRO_BASE);
        assert!(packet.is_base_variant());
    }

    #[test]
    fn test_build_raw_packet_angles() {
        let packet = build_raw_packet(0x12, false, 0, 1000, 2000, 3000);
        // Copy from packed struct to avoid alignment issues
        let x = packet.angle_x;
        let y = packet.angle_y;
        let z = packet.angle_z;
        assert_eq!(x, 1000);
        assert_eq!(y, 2000);
        assert_eq!(z, 3000);
    }

    #[test]
    fn test_build_raw_packet_time() {
        let packet = build_raw_packet(0x12, false, 0xCAFEBABE, 0, 0, 0);
        assert_eq!(packet.gyro_time.as_time_tag(), 0xCAFEBABE);
    }

    #[test]
    fn test_packets_parseable_by_exail_parser() {
        use crate::exail::{parse, GyroMessage};

        let full_packet = build_full_packet(0x12, false, 42, 100, 200, 300);
        let full_bytes = bytes_of(&full_packet);

        match parse(full_bytes) {
            Ok(GyroMessage::Full(parsed)) => {
                // Copy from packed struct to avoid alignment issues
                let x = parsed.raw_ang_x;
                let y = parsed.raw_ang_y;
                let z = parsed.raw_ang_z;
                assert_eq!(x, 100);
                assert_eq!(y, 200);
                assert_eq!(z, 300);
            }
            other => panic!("Expected Full packet, got {:?}", other),
        }

        let raw_packet = build_raw_packet(0x12, false, 42, 400, 500, 600);
        let raw_bytes = bytes_of(&raw_packet);

        match parse(raw_bytes) {
            Ok(GyroMessage::Raw(parsed)) => {
                // Copy from packed struct to avoid alignment issues
                let x = parsed.angle_x;
                let y = parsed.angle_y;
                let z = parsed.angle_z;
                assert_eq!(x, 400);
                assert_eq!(y, 500);
                assert_eq!(z, 600);
            }
            other => panic!("Expected Raw packet, got {:?}", other),
        }
    }
}

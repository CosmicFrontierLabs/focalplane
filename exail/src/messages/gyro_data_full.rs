//! Full Gyro Data message

use bytemuck::{Pod, Zeroable};

use crate::health_status::HealthStatus;
use crate::time::GyroTime;

/// Full Gyro Data packet from Exail Asterix NS
///
/// Total packet size: 66 bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, packed)]
pub struct FullGyroData {
    /// Address of Remote Terminal (1 byte)
    pub start_word: u8,

    /// Message ID for TM Full Gyro Data Base (1 byte)
    pub message_id: u8,

    /// Time field (4 bytes) - can be interpreted as TimeTag (u32) or (GyroTimeTag, TimeBase) pair
    pub gyro_time: GyroTime,

    /// Raw angle measurement, X axis (4 bytes)
    pub raw_ang_x: u32,

    /// Raw angle measurement, Y axis (4 bytes)
    pub raw_ang_y: u32,

    /// Raw angle measurement, Z axis (4 bytes)
    pub raw_ang_z: u32,

    /// Filtered angle measurement, X axis (4 bytes)
    pub fil_ang_x: u32,

    /// Filtered angle measurement, Y axis (4 bytes)
    pub fil_ang_y: u32,

    /// Filtered angle measurement, Z axis (4 bytes)
    pub fil_ang_z: u32,

    /// Source input current measurement (2 bytes)
    pub so_in_cur: u16,

    /// Current command sent to source (2 bytes)
    pub cur_com: u16,

    /// Optical power measurement, channel X (2 bytes)
    pub pow_meas_x: u16,

    /// Optical power measurement, channel Y (2 bytes)
    pub pow_meas_y: u16,

    /// Optical power measurement, channel Z (2 bytes)
    pub pow_meas_z: u16,

    /// VPI data channel X (2 bytes, S12 - signed 12-bit)
    pub vpi_x: i16,

    /// VPI data channel Y (2 bytes, S12 - signed 12-bit)
    pub vpi_y: i16,

    /// VPI data channel Z (2 bytes, S12 - signed 12-bit)
    pub vpi_z: i16,

    /// Modulation ramp data channel X (2 bytes, S16)
    pub ramp_x: i16,

    /// Modulation ramp data channel Y (2 bytes, S16)
    pub ramp_y: i16,

    /// Modulation ramp data channel Z (2 bytes, S16)
    pub ramp_z: i16,

    /// Board temperature (2 bytes)
    pub board_temp: u16,

    /// SIA filter temperature (2 bytes)
    pub sia_fil_temp: u16,

    /// Organizer temperature (2 bytes)
    pub org_fil_temp: u16,

    /// Interface temperature (2 bytes)
    pub inter_temp: u16,

    /// Health status register (4 bytes)
    pub health_status: HealthStatus,

    /// CRC checksum (2 bytes)
    pub checksum: u16,
}

impl FullGyroData {
    /// Expected packet size in bytes
    pub const PACKET_SIZE: usize = 66;
}

// SAFETY: FullGyroData is repr(C, packed) and all fields are Pod
unsafe impl Pod for FullGyroData {}
// SAFETY: FullGyroData is repr(C, packed) and all fields are Zeroable
unsafe impl Zeroable for FullGyroData {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_size() {
        assert_eq!(std::mem::size_of::<FullGyroData>(), 66);
        assert_eq!(FullGyroData::PACKET_SIZE, 66);
    }
}

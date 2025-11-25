//! Frame parsing for Exail Asterix NS messages

use bytemuck::from_bytes;

use crate::messages::{FilteredGyroInertialData, FullGyroData, RawGyroInertialData};

/// Mask for extracting the 5-bit frame ID from the message_id byte
pub const FRAME_ID_MASK: u8 = 0x1F;

/// Frame ID constants (5-bit values)
pub mod frame_id {
    pub const RAW_GYRO_BASE: u8 = 17;
    pub const RAW_GYRO: u8 = 18;
    pub const FILTERED_GYRO_BASE: u8 = 19;
    pub const FILTERED_GYRO: u8 = 20;
    pub const FULL_GYRO_BASE: u8 = 21;
    pub const FULL_GYRO: u8 = 22;
}

/// Parsed message variants
#[derive(Debug, Clone, Copy)]
pub enum GyroMessage {
    Raw(RawGyroInertialData),
    Filtered(FilteredGyroInertialData),
    Full(FullGyroData),
}

/// Parse error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    TooShort,
    UnknownFrameId(u8),
    WrongLength { expected: usize, got: usize },
}

/// Parse a message from bytes.
///
/// The frame ID is extracted using only the lower 5 bits of byte 1,
/// but full byte values are preserved in the struct for CRC computation.
pub fn parse(data: &[u8]) -> Result<GyroMessage, ParseError> {
    if data.len() < 2 {
        return Err(ParseError::TooShort);
    }

    // Mask to get 5-bit frame ID for dispatch only
    let frame_id = data[1] & FRAME_ID_MASK;

    match frame_id {
        frame_id::RAW_GYRO_BASE | frame_id::RAW_GYRO => {
            if data.len() != 26 {
                return Err(ParseError::WrongLength {
                    expected: 26,
                    got: data.len(),
                });
            }
            Ok(GyroMessage::Raw(*from_bytes(data)))
        }
        frame_id::FILTERED_GYRO_BASE | frame_id::FILTERED_GYRO => {
            if data.len() != 26 {
                return Err(ParseError::WrongLength {
                    expected: 26,
                    got: data.len(),
                });
            }
            Ok(GyroMessage::Filtered(*from_bytes(data)))
        }
        frame_id::FULL_GYRO_BASE | frame_id::FULL_GYRO => {
            if data.len() != 66 {
                return Err(ParseError::WrongLength {
                    expected: 66,
                    got: data.len(),
                });
            }
            Ok(GyroMessage::Full(*from_bytes(data)))
        }
        id => Err(ParseError::UnknownFrameId(id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_raw() {
        let mut data = [0u8; 26];
        data[1] = frame_id::RAW_GYRO;
        assert!(matches!(parse(&data), Ok(GyroMessage::Raw(_))));
    }

    #[test]
    fn test_parse_filtered() {
        let mut data = [0u8; 26];
        data[1] = frame_id::FILTERED_GYRO;
        assert!(matches!(parse(&data), Ok(GyroMessage::Filtered(_))));
    }

    #[test]
    fn test_parse_full() {
        let mut data = [0u8; 66];
        data[1] = frame_id::FULL_GYRO;
        assert!(matches!(parse(&data), Ok(GyroMessage::Full(_))));
    }

    #[test]
    fn test_unknown_id() {
        let mut data = [0u8; 26];
        data[1] = 15; // 15 masked is still 15, not a valid frame ID
        assert!(matches!(parse(&data), Err(ParseError::UnknownFrameId(15))));
    }

    #[test]
    fn test_parse_with_high_bits_set() {
        // Frame ID 21 with high bits set (0x80 | 21 = 0x95)
        let mut data = [0u8; 66];
        data[1] = 0x80 | frame_id::FULL_GYRO_BASE; // 0x95
        assert!(matches!(parse(&data), Ok(GyroMessage::Full(_))));
    }
}

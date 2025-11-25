//! Time representation for Exail Asterix NS frames
//!
//! A 4-byte field that can be interpreted as either a single TimeTag (u32)
//! or as a pair of GyroTimeTag and TimeBase (u16, u16).

use bytemuck::{Pod, Zeroable};

/// 4-byte time field with dual interpretation
///
/// Can be read as:
/// - Single `u32` TimeTag via [`as_time_tag`](Self::as_time_tag)
/// - Pair of `u16` (GyroTimeTag, TimeBase) via [`as_gyro_time_tag_and_base`](Self::as_gyro_time_tag_and_base)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct GyroTime([u8; 4]);

impl GyroTime {
    /// Create from raw bytes
    pub fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }

    /// Interpret as single u32 TimeTag (little-endian)
    pub fn as_time_tag(&self) -> u32 {
        u32::from_le_bytes(self.0)
    }

    /// Interpret as pair of u16s: (GyroTimeTag, TimeBase) (little-endian)
    pub fn as_gyro_time_tag_and_base(&self) -> (u16, u16) {
        let gyro_time_tag = u16::from_le_bytes([self.0[0], self.0[1]]);
        let time_base = u16::from_le_bytes([self.0[2], self.0[3]]);
        (gyro_time_tag, time_base)
    }

    /// Raw bytes access
    pub fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_time_tag() {
        let time = GyroTime::from_bytes([0x12, 0x34, 0x56, 0x78]);
        assert_eq!(time.as_time_tag(), 0x78563412);
    }

    #[test]
    fn test_as_gyro_time_tag_and_base() {
        let time = GyroTime::from_bytes([0x12, 0x34, 0x56, 0x78]);
        let (gyro_time_tag, time_base) = time.as_gyro_time_tag_and_base();
        assert_eq!(gyro_time_tag, 0x3412);
        assert_eq!(time_base, 0x7856);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let bytes = [0xAA, 0xBB, 0xCC, 0xDD];
        let time = GyroTime::from_bytes(bytes);
        assert_eq!(time.as_bytes(), &bytes);
    }
}

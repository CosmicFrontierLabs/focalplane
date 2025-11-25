//! Checksum computation for Exail Asterix NS frames
//!
//! The checksum is the sum of all 16-bit DataWords (little-endian),
//! truncated on overflow. Each DataWord consists of two bytes with
//! LSB sent first, followed by MSB.

/// Compute checksum over a byte slice containing little-endian 16-bit words.
///
/// The input length must be even (multiple of 2 bytes).
/// Returns the wrapping sum of all 16-bit little-endian words.
pub fn compute_checksum(data: &[u8]) -> u16 {
    debug_assert!(
        data.len() % 2 == 0,
        "Data length must be even for checksum computation"
    );

    data.chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .fold(0u16, |acc, word| acc.wrapping_add(word))
}

/// Verify that a frame's checksum is valid.
///
/// The frame should include all bytes up to and including the checksum.
/// The checksum of the entire frame (including the checksum field) should
/// equal the checksum field itself when computed correctly.
///
/// For a frame of N bytes where the last 2 bytes are the checksum:
/// - Computes sum of bytes 0..N-2
/// - Compares against the checksum stored in bytes N-2..N
pub fn verify_checksum(frame: &[u8]) -> bool {
    if frame.len() < 4 || frame.len() % 2 != 0 {
        return false;
    }

    let data_end = frame.len() - 2;
    let computed = compute_checksum(&frame[..data_end]);
    let stored = u16::from_le_bytes([frame[data_end], frame[data_end + 1]]);

    computed == stored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checksum_from_datasheet_example() {
        // Example from datasheet section 4.2.3:
        // FRAME: 12 02 F1 4C 0F B3 00 00 06 07 1E 00 FA 06 41 01 06 F9 BE FE 00 00 00 00 35 09
        // Checksum should be 0x0935
        let frame: [u8; 26] = [
            0x12, 0x02, 0xF1, 0x4C, 0x0F, 0xB3, 0x00, 0x00, 0x06, 0x07, 0x1E, 0x00, 0xFA, 0x06,
            0x41, 0x01, 0x06, 0xF9, 0xBE, 0xFE, 0x00, 0x00, 0x00, 0x00, 0x35, 0x09,
        ];

        // Verify the stored checksum
        let stored_checksum = u16::from_le_bytes([frame[24], frame[25]]);
        assert_eq!(stored_checksum, 0x0935);

        // Compute checksum of data portion (excluding checksum bytes)
        let computed = compute_checksum(&frame[..24]);
        assert_eq!(computed, 0x0935);

        // Verify full frame
        assert!(verify_checksum(&frame));
    }

    #[test]
    fn test_checksum_computation_step_by_step() {
        // Verify individual DataWord parsing from the example
        // First DW: 12 02 -> 0x0212
        assert_eq!(u16::from_le_bytes([0x12, 0x02]), 0x0212);
        // Second DW: F1 4C -> 0x4CF1
        assert_eq!(u16::from_le_bytes([0xF1, 0x4C]), 0x4CF1);
        // Third DW: 0F B3 -> 0xB30F
        assert_eq!(u16::from_le_bytes([0x0F, 0xB3]), 0xB30F);
    }

    #[test]
    fn test_checksum_wrapping() {
        // Test that overflow wraps correctly
        let data: [u8; 4] = [0xFF, 0xFF, 0x02, 0x00]; // 0xFFFF + 0x0002 = 0x0001 (wrapped)
        assert_eq!(compute_checksum(&data), 0x0001);
    }

    #[test]
    fn test_verify_checksum_invalid() {
        let mut frame: [u8; 26] = [
            0x12, 0x02, 0xF1, 0x4C, 0x0F, 0xB3, 0x00, 0x00, 0x06, 0x07, 0x1E, 0x00, 0xFA, 0x06,
            0x41, 0x01, 0x06, 0xF9, 0xBE, 0xFE, 0x00, 0x00, 0x00, 0x00, 0x35, 0x09,
        ];

        // Corrupt the checksum
        frame[24] = 0x00;
        assert!(!verify_checksum(&frame));
    }

    #[test]
    fn test_verify_checksum_short_frame() {
        assert!(!verify_checksum(&[0x00, 0x00])); // Too short
        assert!(!verify_checksum(&[0x00, 0x00, 0x00])); // Odd length
    }
}

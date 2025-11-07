/// COBS (Consistent Overhead Byte Stuffing) encoding and decoding
///
/// COBS removes all zero bytes from data, allowing 0x00 to be used as a frame delimiter.
/// For packets <= 24 bytes, overhead is exactly 2 bytes (1 code byte + 1 delimiter).
///
/// Encode data using COBS algorithm with compile-time size checking
///
/// Input must be <= 24 bytes. Output must be exactly N + 2 bytes (1 code + data + delimiter).
/// Format: [code byte][encoded data...][0x00 delimiter]
pub fn encode<const N: usize, const OUT: usize>(input: &[u8; N], output: &mut [u8; OUT]) {
    const {
        assert!(
            N <= 24,
            "Input must be <= 24 bytes for fixed 2-byte overhead"
        );
        assert!(OUT == N + 2, "Output size must be exactly N + 2");
    };

    if N == 0 {
        output[0] = 0x01;
        output[1] = 0x00;
        return;
    }

    let mut out_idx = 1;
    let mut code_idx = 0;
    let mut code = 0x01;

    for &byte in input.iter() {
        if byte == 0 {
            output[code_idx] = code;
            code_idx = out_idx;
            out_idx += 1;
            code = 0x01;
        } else {
            output[out_idx] = byte;
            out_idx += 1;
            code += 1;
        }
    }

    output[code_idx] = code;
    output[out_idx] = 0x00;
}

/// Decode COBS-encoded data with compile-time size checking
///
/// Input must be M bytes (encoded), output must be M - 2 bytes (original data).
/// Input must be at least 2 bytes and <= 26 bytes (24 data + 2 overhead).
/// Expected format: [code byte][encoded data...][0x00 delimiter]
pub fn decode<const M: usize, const OUT: usize>(
    input: &[u8; M],
    output: &mut [u8; OUT],
) -> Result<(), &'static str> {
    const {
        assert!(M >= 2, "Encoded input must be at least 2 bytes");
        assert!(
            M <= 26,
            "Encoded input must be <= 26 bytes (24 data + 2 overhead)"
        );
        assert!(OUT == M - 2, "Output size must be exactly M - 2");
    };

    if input[M - 1] != 0x00 {
        return Err("Invalid COBS encoding: missing zero delimiter");
    }

    if M == 2 {
        if input[0] != 0x01 {
            return Err("Invalid COBS encoding for empty payload");
        }
        return Ok(());
    }

    let mut out_idx = 0;
    let mut in_idx = 0;

    while in_idx < M - 1 {
        let code = input[in_idx];

        if code == 0 {
            return Err("Invalid COBS encoding: found zero byte before delimiter");
        }

        in_idx += 1;
        let copy_len = (code as usize) - 1;

        if in_idx + copy_len > M - 1 {
            return Err("Invalid COBS encoding: code extends beyond input");
        }

        if copy_len > 0 {
            output[out_idx..out_idx + copy_len].copy_from_slice(&input[in_idx..in_idx + copy_len]);
            out_idx += copy_len;
            in_idx += copy_len;
        }

        if code != 0xFF && in_idx < M - 1 {
            if out_idx >= OUT {
                return Err("Invalid COBS encoding: output overflow");
            }
            output[out_idx] = 0;
            out_idx += 1;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_empty() {
        let input = [];
        let mut output = [0u8; 2];
        encode(&input, &mut output);
        assert_eq!(output, [0x01, 0x00]);
    }

    #[test]
    fn test_encode_no_zeros() {
        let input = [0x11, 0x22, 0x33, 0x44];
        let mut output = [0u8; 6];
        encode(&input, &mut output);
        assert_eq!(output, [0x05, 0x11, 0x22, 0x33, 0x44, 0x00]);
    }

    #[test]
    fn test_encode_with_zeros() {
        let input = [0x11, 0x00, 0x00, 0x11];
        let mut output = [0u8; 6];
        encode(&input, &mut output);
        assert_eq!(output, [0x02, 0x11, 0x01, 0x02, 0x11, 0x00]);
    }

    #[test]
    fn test_encode_leading_zero() {
        let input = [0x00, 0x11, 0x22];
        let mut output = [0u8; 5];
        encode(&input, &mut output);
        assert_eq!(output, [0x01, 0x03, 0x11, 0x22, 0x00]);
    }

    #[test]
    fn test_encode_trailing_zero() {
        let input = [0x11, 0x22, 0x00];
        let mut output = [0u8; 5];
        encode(&input, &mut output);
        assert_eq!(output, [0x03, 0x11, 0x22, 0x01, 0x00]);
    }

    #[test]
    fn test_decode_empty() {
        let input = [0x01, 0x00];
        let mut output = [0u8; 0];
        decode(&input, &mut output).unwrap();
        assert_eq!(output, []);
    }

    #[test]
    fn test_decode_no_zeros() {
        let input = [0x05, 0x11, 0x22, 0x33, 0x44, 0x00];
        let mut output = [0u8; 4];
        decode(&input, &mut output).unwrap();
        assert_eq!(output, [0x11, 0x22, 0x33, 0x44]);
    }

    #[test]
    fn test_decode_with_zeros() {
        let input = [0x02, 0x11, 0x01, 0x02, 0x11, 0x00];
        let mut output = [0u8; 4];
        decode(&input, &mut output).unwrap();
        assert_eq!(output, [0x11, 0x00, 0x00, 0x11]);
    }

    #[test]
    fn test_roundtrip_simple() {
        let original = [0x11, 0x22, 0x33, 0x44];
        let mut encoded = [0u8; 6];
        encode(&original, &mut encoded);

        let mut decoded = [0u8; 4];
        decode(&encoded, &mut decoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_with_zeros() {
        let original = [0x00, 0x11, 0x00, 0x22, 0x33, 0x00, 0x44];
        let mut encoded = [0u8; 9];
        encode(&original, &mut encoded);

        let mut decoded = [0u8; 7];
        decode(&encoded, &mut decoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_max_payload_size() {
        let input = [0x42; 24];
        let mut output = [0u8; 26];
        encode(&input, &mut output);
        assert_eq!(output.len(), 26);

        let mut decoded = [0u8; 24];
        decode(&output, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_decode_invalid_zero() {
        let input = [0x02, 0x00, 0x11, 0x00];
        let mut output = [0u8; 2];
        let result = decode(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_missing_delimiter() {
        let input = [0x05, 0x11, 0x22, 0x33, 0x44, 0x55];
        let mut output = [0u8; 4];
        let result = decode(&input, &mut output);
        assert!(result.is_err());
    }
}

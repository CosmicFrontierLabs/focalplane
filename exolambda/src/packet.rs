use crate::cobs;

const CRC_ALGO: crc::Crc<u32> = crc::Crc::<u32>::new(&crc::CRC_32_CKSUM);
const SEQUENCE_MASK: u8 = 0b11111;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FSMCommandType {
    FeedbackMode = 0b000,
    HoldCurrent = 0b001,
    IdleOff = 0b010,
    FeedForward = 0b100,
}

impl FSMCommandType {
    pub fn from_bits(bits: u8) -> Result<Self, &'static str> {
        match bits {
            0b000 => Ok(FSMCommandType::FeedbackMode),
            0b001 => Ok(FSMCommandType::HoldCurrent),
            0b010 => Ok(FSMCommandType::IdleOff),
            0b100 => Ok(FSMCommandType::FeedForward),
            _ => Err("Invalid command type"),
        }
    }

    pub fn to_bits(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FSMCommand {
    pub cmd: FSMCommandType,
    pub sequence: u8,
    pub x_setpoint: f32,
    pub y_setpoint: f32,
}

impl FSMCommand {
    const RAW_SIZE: usize = 13;
    const ENCODED_SIZE: usize = 15;

    pub fn new(
        cmd: FSMCommandType,
        sequence: u8,
        x_setpoint: f32,
        y_setpoint: f32,
    ) -> Result<Self, &'static str> {
        if sequence > SEQUENCE_MASK {
            return Err("sequence must be 5 bits (0-31)");
        }
        Ok(Self {
            cmd,
            sequence,
            x_setpoint,
            y_setpoint,
        })
    }

    pub fn from_raw(encoded: &[u8; Self::ENCODED_SIZE]) -> Result<Self, &'static str> {
        let mut raw = [0u8; Self::RAW_SIZE];
        cobs::decode(encoded, &mut raw)?;

        let header = raw[0];
        let cmd_bits = (header >> 5) & 0b111;
        let cmd = FSMCommandType::from_bits(cmd_bits)?;
        let sequence = header & SEQUENCE_MASK;

        let x_setpoint = f32::from_le_bytes([raw[1], raw[2], raw[3], raw[4]]);
        let y_setpoint = f32::from_le_bytes([raw[5], raw[6], raw[7], raw[8]]);

        let received_crc = u32::from_le_bytes([raw[9], raw[10], raw[11], raw[12]]);

        let mut digest = CRC_ALGO.digest();
        digest.update(&raw[0..9]);
        let computed_crc = digest.finalize();

        if received_crc != computed_crc {
            return Err("CRC mismatch");
        }

        Ok(Self {
            cmd,
            sequence,
            x_setpoint,
            y_setpoint,
        })
    }

    pub fn to_raw(&self, encoded: &mut [u8; Self::ENCODED_SIZE]) {
        let header = (self.cmd.to_bits() << 5) | (self.sequence & SEQUENCE_MASK);

        let mut raw = [0u8; Self::RAW_SIZE];
        raw[0] = header;
        raw[1..5].copy_from_slice(&self.x_setpoint.to_le_bytes());
        raw[5..9].copy_from_slice(&self.y_setpoint.to_le_bytes());

        let mut digest = CRC_ALGO.digest();
        digest.update(&raw[0..9]);
        let crc = digest.finalize();
        raw[9..13].copy_from_slice(&crc.to_le_bytes());

        cobs::encode(&raw, encoded);
    }

    pub fn next_sequence_number(&self) -> u8 {
        (self.sequence + 1) & SEQUENCE_MASK
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FSMTelemetry {
    pub cmd: FSMCommandType,
    pub sequence: u8,
    pub x_position: f32,
    pub y_position: f32,
    pub x_dac_out: f32,
    pub y_dac_out: f32,
}

impl FSMTelemetry {
    const RAW_SIZE: usize = 21;
    const ENCODED_SIZE: usize = 23;

    pub fn new(
        cmd: FSMCommandType,
        sequence: u8,
        x_position: f32,
        y_position: f32,
        x_dac_out: f32,
        y_dac_out: f32,
    ) -> Result<Self, &'static str> {
        if sequence > SEQUENCE_MASK {
            return Err("sequence must be 5 bits (0-31)");
        }
        Ok(Self {
            cmd,
            sequence,
            x_position,
            y_position,
            x_dac_out,
            y_dac_out,
        })
    }

    pub fn from_raw(encoded: &[u8; Self::ENCODED_SIZE]) -> Result<Self, &'static str> {
        let mut raw = [0u8; Self::RAW_SIZE];
        cobs::decode(encoded, &mut raw)?;

        let header = raw[0];
        let cmd_bits = (header >> 5) & 0b111;
        let cmd = FSMCommandType::from_bits(cmd_bits)?;
        let sequence = header & SEQUENCE_MASK;

        let x_position = f32::from_le_bytes([raw[1], raw[2], raw[3], raw[4]]);
        let y_position = f32::from_le_bytes([raw[5], raw[6], raw[7], raw[8]]);
        let x_dac_out = f32::from_le_bytes([raw[9], raw[10], raw[11], raw[12]]);
        let y_dac_out = f32::from_le_bytes([raw[13], raw[14], raw[15], raw[16]]);

        let received_crc = u32::from_le_bytes([raw[17], raw[18], raw[19], raw[20]]);

        let mut digest = CRC_ALGO.digest();
        digest.update(&raw[0..17]);
        let computed_crc = digest.finalize();

        if received_crc != computed_crc {
            return Err("CRC mismatch");
        }

        Ok(Self {
            cmd,
            sequence,
            x_position,
            y_position,
            x_dac_out,
            y_dac_out,
        })
    }

    pub fn to_raw(&self, encoded: &mut [u8; Self::ENCODED_SIZE]) {
        let header = (self.cmd.to_bits() << 5) | (self.sequence & SEQUENCE_MASK);

        let mut raw = [0u8; Self::RAW_SIZE];
        raw[0] = header;
        raw[1..5].copy_from_slice(&self.x_position.to_le_bytes());
        raw[5..9].copy_from_slice(&self.y_position.to_le_bytes());
        raw[9..13].copy_from_slice(&self.x_dac_out.to_le_bytes());
        raw[13..17].copy_from_slice(&self.y_dac_out.to_le_bytes());

        let mut digest = CRC_ALGO.digest();
        digest.update(&raw[0..17]);
        let crc = digest.finalize();
        raw[17..21].copy_from_slice(&crc.to_le_bytes());

        cobs::encode(&raw, encoded);
    }

    pub fn next_sequence_number(&self) -> u8 {
        (self.sequence + 1) & SEQUENCE_MASK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsm_command_roundtrip() {
        let cmd = FSMCommand::new(FSMCommandType::IdleOff, 15, 1.5, -2.5).unwrap();
        let mut encoded = [0u8; FSMCommand::ENCODED_SIZE];
        cmd.to_raw(&mut encoded);

        let decoded = FSMCommand::from_raw(&encoded).unwrap();
        assert_eq!(decoded, cmd);
    }

    #[test]
    fn test_fsm_telemetry_roundtrip() {
        let telem = FSMTelemetry::new(FSMCommandType::FeedForward, 20, 1.0, 2.0, 3.0, 4.0).unwrap();
        let mut encoded = [0u8; FSMTelemetry::ENCODED_SIZE];
        telem.to_raw(&mut encoded);

        let decoded = FSMTelemetry::from_raw(&encoded).unwrap();
        assert_eq!(decoded, telem);
    }

    #[test]
    fn test_fsm_command_header() {
        let cmd = FSMCommand::new(FSMCommandType::FeedForward, 0b10110, 0.0, 0.0).unwrap();
        let mut encoded = [0u8; FSMCommand::ENCODED_SIZE];
        cmd.to_raw(&mut encoded);

        let decoded = FSMCommand::from_raw(&encoded).unwrap();
        assert_eq!(decoded.cmd, FSMCommandType::FeedForward);
        assert_eq!(decoded.sequence, 0b10110);
    }

    #[test]
    fn test_fsm_command_crc_validation() {
        let cmd = FSMCommand::new(FSMCommandType::HoldCurrent, 2, 3.0, 4.0).unwrap();
        let mut encoded = [0u8; FSMCommand::ENCODED_SIZE];
        cmd.to_raw(&mut encoded);

        let mut raw = [0u8; FSMCommand::RAW_SIZE];
        cobs::decode(&encoded, &mut raw).unwrap();
        raw[5] ^= 0xFF;

        let mut corrupted = [0u8; FSMCommand::ENCODED_SIZE];
        cobs::encode(&raw, &mut corrupted);

        let result = FSMCommand::from_raw(&corrupted);
        assert!(result.is_err());
    }

    #[test]
    fn test_sequence_bounds() {
        assert!(FSMCommand::new(FSMCommandType::FeedbackMode, 31, 0.0, 0.0).is_ok());
        assert!(FSMCommand::new(FSMCommandType::FeedbackMode, 32, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_all_command_types_roundtrip() {
        for cmd_type in [
            FSMCommandType::FeedbackMode,
            FSMCommandType::HoldCurrent,
            FSMCommandType::IdleOff,
            FSMCommandType::FeedForward,
        ] {
            let cmd = FSMCommand::new(cmd_type, 10, 1.0, 2.0).unwrap();
            let mut encoded = [0u8; FSMCommand::ENCODED_SIZE];
            cmd.to_raw(&mut encoded);

            let decoded = FSMCommand::from_raw(&encoded).unwrap();
            assert_eq!(decoded.cmd, cmd_type);
        }
    }

    #[test]
    fn test_sequence_rollover() {
        let cmd = FSMCommand::new(FSMCommandType::FeedbackMode, 31, 0.0, 0.0).unwrap();
        assert_eq!(cmd.next_sequence_number(), 0);

        let cmd2 = FSMCommand::new(FSMCommandType::FeedbackMode, 15, 0.0, 0.0).unwrap();
        assert_eq!(cmd2.next_sequence_number(), 16);

        let telem = FSMTelemetry::new(FSMCommandType::HoldCurrent, 31, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert_eq!(telem.next_sequence_number(), 0);

        let telem2 =
            FSMTelemetry::new(FSMCommandType::HoldCurrent, 20, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert_eq!(telem2.next_sequence_number(), 21);
    }
}

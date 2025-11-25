//! Exail Asterix NS Gyro protocol parsing
//!
//! This module provides structures and parsing for raw data packets
//! from the Exail Asterix NS inertial measurement unit.

mod checksum;
mod health_status;
pub mod messages;
mod parser;
mod temperature;
mod time;

pub use checksum::{compute_checksum, verify_checksum};
pub use health_status::HealthStatus;
pub use messages::{FilteredGyroInertialData, FullGyroData, RawGyroInertialData};
pub use parser::{frame_id, parse, GyroMessage, ParseError, FRAME_ID_MASK};
pub use temperature::{TempDecoder, BOARD_TEMP, SIA_FIL_TEMP};
pub use time::GyroTime;

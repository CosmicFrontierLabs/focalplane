//! I/O utilities for simulator data export and import

pub mod fits;

pub use fits::{read_fits_to_hashmap, write_typed_fits, FitsDataType, FitsError};

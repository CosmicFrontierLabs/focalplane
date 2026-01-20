//! Shared components and utilities for meter-sim modules.
//!
//! This module contains common types, traits, and utilities that are used
//! across multiple meter-sim components to avoid duplication and ensure
//! consistency.
//!
//! # Feature Flags
//!
//! - `full` (default): Enables all optional features
//! - `zmq-support`: ZeroMQ messaging support
//! - `tracking`: Tracking data collection and messages
//! - `pattern-client`: Pattern generator client (requires zmq-support)
//! - `frame-writer`: FITS file writing support
//! - `config-storage`: Configuration persistence
//! - `system-info`: System information types

pub mod algo;
pub mod bad_pixel_map;
pub mod barker;
pub mod cached_star_catalog;
pub mod camera_interface;
pub mod dark_frame;
pub mod image_proc;
pub mod image_size;
pub mod optical_alignment;
pub mod pattern_command;
pub use pattern_command::generate_centered_grid;
pub mod range_arg;
pub mod ring_buffer;
pub mod star_projector;
pub mod test_util;
pub mod units;
pub mod viz;

// Feature-gated modules
#[cfg(feature = "config-storage")]
pub mod config_storage;

#[cfg(feature = "frame-writer")]
pub mod frame_writer;

#[cfg(feature = "pattern-client")]
pub mod pattern_client;

#[cfg(feature = "system-info")]
pub mod system_info;

#[cfg(feature = "tracking")]
pub mod tracking_collector;

#[cfg(feature = "tracking")]
pub mod tracking_message;

#[cfg(feature = "zmq-support")]
pub mod zmq;

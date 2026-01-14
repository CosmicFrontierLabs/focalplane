//! Shared components and utilities for meter-sim modules.
//!
//! This module contains common types, traits, and utilities that are used
//! across multiple meter-sim components to avoid duplication and ensure
//! consistency.

pub mod algo;
pub mod bad_pixel_map;
pub mod barker;
pub mod cached_star_catalog;
pub mod camera_interface;
pub mod config_storage;
pub mod dark_frame;
pub mod frame_writer;
pub mod image_proc;
pub mod image_size;
pub mod optical_alignment;
pub mod pattern_client;
pub mod pattern_command;
pub mod range_arg;
pub mod star_projector;
pub mod system_info;
pub mod test_util;
pub mod tracking_collector;
pub mod tracking_message;
pub mod units;
pub mod viz;
pub mod zmq;

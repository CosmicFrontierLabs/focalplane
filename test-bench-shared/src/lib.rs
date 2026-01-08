//! Shared types for test-bench backend and frontend.
//!
//! This crate contains lightweight serialization types that can be used
//! by both the Rust backend (test-bench) and WASM frontend (test-bench-frontend).
//! All types here must be WASM-compatible (no threading, no C bindings).

use serde::{Deserialize, Serialize};

/// Current pattern configuration from calibrate_serve.
///
/// Returned via HTTP GET /config endpoint. Used by frontend to sync
/// state with server (e.g., after ZMQ commands or idle timeout).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatternConfigResponse {
    /// Pattern type identifier (e.g., "Crosshair", "Uniform")
    pub pattern_id: String,
    /// Pattern-specific parameter values (JSON object)
    pub values: serde_json::Value,
    /// Whether pattern colors are inverted
    pub invert: bool,
}

//! Calibration display module.
//!
//! Provides infrastructure for displaying calibration patterns via the web server.

mod display;
mod pattern;
mod schema;

pub use display::{run_display, DisplayConfig, DynamicPattern, PatternSource};
pub use pattern::PatternConfig;
pub use schema::{
    get_pattern_schemas, parse_pattern_request, pattern_to_dynamic, ControlSpec, PatternSpec,
    SchemaResponse,
};

use super::pattern::PatternConfig;

// Re-export shared types
pub use shared_wasm::{ControlSpec, PatternSpec, SchemaResponse};

/// Generate the schema for all web-accessible patterns.
pub fn get_pattern_schemas() -> SchemaResponse {
    SchemaResponse {
        patterns: vec![
            PatternSpec {
                id: "Check".into(),
                name: "Checkerboard".into(),
                controls: vec![ControlSpec::IntRange {
                    id: "checker_size".into(),
                    label: "Checker Size (px)".into(),
                    min: 10,
                    max: 500,
                    step: 10,
                    default: 100,
                }],
            },
            PatternSpec {
                id: "Usaf".into(),
                name: "USAF-1951 Target".into(),
                controls: vec![],
            },
            PatternSpec {
                id: "Static".into(),
                name: "Digital Static".into(),
                controls: vec![ControlSpec::IntRange {
                    id: "pixel_size".into(),
                    label: "Pixel Size (px)".into(),
                    min: 1,
                    max: 20,
                    step: 1,
                    default: 1,
                }],
            },
            PatternSpec {
                id: "Pixel".into(),
                name: "Center Pixel".into(),
                controls: vec![],
            },
            PatternSpec {
                id: "CirclingPixel".into(),
                name: "Circling Pixel".into(),
                controls: vec![
                    ControlSpec::IntRange {
                        id: "orbit_count".into(),
                        label: "Orbit Count".into(),
                        min: 1,
                        max: 10,
                        step: 1,
                        default: 1,
                    },
                    ControlSpec::IntRange {
                        id: "orbit_radius_percent".into(),
                        label: "Orbit Radius (% FOV)".into(),
                        min: 5,
                        max: 95,
                        step: 5,
                        default: 50,
                    },
                ],
            },
            PatternSpec {
                id: "Uniform".into(),
                name: "Uniform Screen".into(),
                controls: vec![ControlSpec::IntRange {
                    id: "level".into(),
                    label: "Brightness Level".into(),
                    min: 0,
                    max: 255,
                    step: 1,
                    default: 128,
                }],
            },
            PatternSpec {
                id: "WigglingGaussian".into(),
                name: "Wiggling Gaussian".into(),
                controls: vec![
                    ControlSpec::FloatRange {
                        id: "fwhm".into(),
                        label: "FWHM (px)".into(),
                        min: 1.0,
                        max: 100.0,
                        step: 1.0,
                        default: 47.0,
                    },
                    ControlSpec::FloatRange {
                        id: "wiggle_radius".into(),
                        label: "Wiggle Radius (px)".into(),
                        min: 0.0,
                        max: 50.0,
                        step: 0.5,
                        default: 3.0,
                    },
                    ControlSpec::FloatRange {
                        id: "intensity".into(),
                        label: "Intensity".into(),
                        min: 0.0,
                        max: 255.0,
                        step: 1.0,
                        default: 255.0,
                    },
                ],
            },
            PatternSpec {
                id: "PixelGrid".into(),
                name: "Pixel Grid".into(),
                controls: vec![ControlSpec::IntRange {
                    id: "spacing".into(),
                    label: "Grid Spacing (px)".into(),
                    min: 10,
                    max: 200,
                    step: 10,
                    default: 50,
                }],
            },
            PatternSpec {
                id: "SiemensStar".into(),
                name: "Siemens Star".into(),
                controls: vec![ControlSpec::IntRange {
                    id: "spokes".into(),
                    label: "Number of Spokes".into(),
                    min: 4,
                    max: 72,
                    step: 4,
                    default: 24,
                }],
            },
            PatternSpec {
                id: "RemoteControlled".into(),
                name: "Remote Controlled".into(),
                controls: vec![],
            },
        ],
        global_controls: vec![
            ControlSpec::Bool {
                id: "invert".into(),
                label: "Invert Colors".into(),
                default: false,
            },
            ControlSpec::Bool {
                id: "emit_gyro".into(),
                label: "Emit Gyro Data".into(),
                default: false,
            },
            ControlSpec::FloatRange {
                id: "plate_scale".into(),
                label: "Plate Scale (arcsec/px)".into(),
                min: 0.01,
                max: 100.0,
                step: 0.01,
                default: 1.0,
            },
        ],
    }
}

/// Convert pattern ID and values from web request to PatternConfig.
///
/// Note: RemoteControlled pattern is handled specially in calibrate_serve
/// since it needs the shared remote state.
pub fn parse_pattern_request(
    pattern_id: &str,
    values: &serde_json::Map<String, serde_json::Value>,
    _display_size: Option<(u32, u32)>,
) -> Result<PatternConfig, String> {
    let get_i64 = |key: &str, default: i64| -> i64 {
        values.get(key).and_then(|v| v.as_i64()).unwrap_or(default)
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        values.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    };

    match pattern_id {
        "Check" => Ok(PatternConfig::Check {
            checker_size: get_i64("checker_size", 100) as u32,
        }),
        "Usaf" => Ok(PatternConfig::Usaf),
        "Static" => Ok(PatternConfig::Static {
            pixel_size: get_i64("pixel_size", 1) as u32,
        }),
        "Pixel" => Ok(PatternConfig::Pixel),
        "CirclingPixel" => Ok(PatternConfig::CirclingPixel {
            orbit_count: get_i64("orbit_count", 1) as u32,
            orbit_radius_percent: get_i64("orbit_radius_percent", 50) as u32,
        }),
        "Uniform" => Ok(PatternConfig::Uniform {
            level: get_i64("level", 128) as u8,
        }),
        "WigglingGaussian" => Ok(PatternConfig::WigglingGaussian {
            fwhm: get_f64("fwhm", 47.0),
            wiggle_radius: get_f64("wiggle_radius", 3.0),
            intensity: get_f64("intensity", 255.0),
        }),
        "PixelGrid" => Ok(PatternConfig::PixelGrid {
            spacing: get_i64("spacing", 50) as u32,
        }),
        "SiemensStar" => Ok(PatternConfig::SiemensStar {
            spokes: get_i64("spokes", 24) as u32,
        }),
        // RemoteControlled is handled specially in calibrate_serve
        "RemoteControlled" => Err(
            "RemoteControlled pattern should be handled by calibrate_serve directly".to_string(),
        ),
        _ => Err(format!("Unknown pattern_id: {pattern_id}")),
    }
}

/// Convert PatternConfig to dynamic format for frontend.
pub fn pattern_to_dynamic(config: &PatternConfig) -> (String, serde_json::Value) {
    use serde_json::json;
    match config {
        PatternConfig::Check { checker_size } => {
            ("Check".into(), json!({"checker_size": checker_size}))
        }
        PatternConfig::Usaf => ("Usaf".into(), json!({})),
        PatternConfig::Static { pixel_size } => {
            ("Static".into(), json!({"pixel_size": pixel_size}))
        }
        PatternConfig::Pixel => ("Pixel".into(), json!({})),
        PatternConfig::CirclingPixel {
            orbit_count,
            orbit_radius_percent,
        } => (
            "CirclingPixel".into(),
            json!({"orbit_count": orbit_count, "orbit_radius_percent": orbit_radius_percent}),
        ),
        PatternConfig::Uniform { level } => ("Uniform".into(), json!({"level": level})),
        PatternConfig::WigglingGaussian {
            fwhm,
            wiggle_radius,
            intensity,
        } => (
            "WigglingGaussian".into(),
            json!({"fwhm": fwhm, "wiggle_radius": wiggle_radius, "intensity": intensity}),
        ),
        PatternConfig::PixelGrid { spacing } => ("PixelGrid".into(), json!({"spacing": spacing})),
        PatternConfig::SiemensStar { spokes } => ("SiemensStar".into(), json!({"spokes": spokes})),
        PatternConfig::RemoteControlled { .. } => ("RemoteControlled".into(), json!({})),
    }
}

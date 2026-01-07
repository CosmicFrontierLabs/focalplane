use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{Html, Response},
    routing::get,
    Json, Router,
};
use clap::Parser;
use image::{ImageBuffer, Rgb};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::mpsc;
use std::sync::Arc;
use test_bench::display_patterns as patterns;
use test_bench::display_utils::{
    get_display_resolution, list_displays, resolve_display_index, wait_for_oled_display,
};
use tokio::sync::RwLock;

#[derive(Parser, Debug)]
#[command(author, version, about = "Calibration pattern web server")]
struct Args {
    #[arg(short = 'p', long, default_value = "3001")]
    port: u16,

    #[arg(short = 'b', long, default_value = "0.0.0.0")]
    bind_address: String,

    #[arg(short, long, help = "Display index to use (0-based)")]
    display: Option<u32>,

    #[arg(short, long, help = "List available displays and exit")]
    list: bool,

    #[arg(long, help = "Poll until 2560x2560 OLED display is detected")]
    wait_for_oled: bool,

    #[arg(
        long,
        default_value = "5",
        help = "Seconds between OLED detection attempts"
    )]
    poll_interval: u64,

    #[arg(
        long,
        default_value = "300",
        help = "Seconds of inactivity before screen goes black (OLED burn-in protection)"
    )]
    idle_timeout: u64,
}

// ============================================================================
// Control Schema Types - Define UI controls that frontend renders dynamically
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ControlSpec {
    IntRange {
        id: String,
        label: String,
        min: i64,
        max: i64,
        step: i64,
        default: i64,
    },
    FloatRange {
        id: String,
        label: String,
        min: f64,
        max: f64,
        step: f64,
        default: f64,
    },
    Bool {
        id: String,
        label: String,
        default: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSpec {
    pub id: String,
    pub name: String,
    pub controls: Vec<ControlSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaResponse {
    pub patterns: Vec<PatternSpec>,
    pub global_controls: Vec<ControlSpec>,
}

fn get_pattern_schemas() -> SchemaResponse {
    SchemaResponse {
        patterns: vec![
            PatternSpec {
                id: "April".into(),
                name: "AprilTag Array".into(),
                controls: vec![],
            },
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
        ],
        global_controls: vec![ControlSpec::Bool {
            id: "invert".into(),
            label: "Invert Colors".into(),
            default: false,
        }],
    }
}

// ============================================================================
// Pattern Config - Internal representation for pattern generation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type")]
enum PatternConfig {
    Check {
        checker_size: u32,
    },
    Usaf,
    Static {
        pixel_size: u32,
    },
    Pixel,
    #[default]
    April,
    CirclingPixel {
        orbit_count: u32,
        orbit_radius_percent: u32,
    },
    Uniform {
        level: u8,
    },
    WigglingGaussian {
        fwhm: f64,
        wiggle_radius: f64,
        intensity: f64,
    },
    PixelGrid {
        spacing: u32,
    },
    SiemensStar {
        spokes: u32,
    },
}

struct AppState {
    pattern: Arc<RwLock<PatternConfig>>,
    width: u32,
    height: u32,
    invert: Arc<RwLock<bool>>,
    pattern_start_time: Arc<RwLock<std::time::Instant>>,
    display_update_tx: Option<mpsc::Sender<()>>,
    idle_timeout: std::time::Duration,
}

fn generate_pattern(
    config: &PatternConfig,
    width: u32,
    height: u32,
    invert: bool,
) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let mut img = match config {
        PatternConfig::Check { checker_size } => {
            patterns::checkerboard::generate(width, height, *checker_size)
        }
        PatternConfig::Usaf => patterns::usaf::generate(width, height)?,
        PatternConfig::Static { pixel_size } => {
            patterns::static_noise::generate(width, height, *pixel_size)
        }
        PatternConfig::Pixel => patterns::pixel::generate(width, height),
        PatternConfig::April => patterns::apriltag::generate(width, height)?,
        PatternConfig::CirclingPixel {
            orbit_count,
            orbit_radius_percent,
        } => patterns::circling_pixel::generate(width, height, *orbit_count, *orbit_radius_percent),
        PatternConfig::Uniform { level } => patterns::uniform::generate(width, height, *level),
        PatternConfig::WigglingGaussian {
            fwhm,
            wiggle_radius,
            intensity,
        } => {
            patterns::wiggling_gaussian::generate(width, height, *fwhm, *wiggle_radius, *intensity)
        }
        PatternConfig::PixelGrid { spacing } => {
            patterns::pixel_grid::generate(width, height, *spacing)
        }
        PatternConfig::SiemensStar { spokes } => {
            patterns::siemens_star::generate(width, height, *spokes)
        }
    };

    if invert {
        for pixel in img.pixels_mut() {
            pixel[0] = 255 - pixel[0];
            pixel[1] = 255 - pixel[1];
            pixel[2] = 255 - pixel[2];
        }
    }

    Ok(img)
}

async fn pattern_page(State(state): State<Arc<AppState>>) -> Html<String> {
    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Calibration Pattern Server</title>
    <link rel="stylesheet" href="/static/shared-styles.css" />
    <script type="module">
        import init from '/static/calibrate_wasm.js';
        init();
    </script>
</head>
<body>
    <div id="app" data-width="{}" data-height="{}"></div>
</body>
</html>"#,
        state.width, state.height
    );

    Html(html)
}

async fn jpeg_pattern_endpoint(State(state): State<Arc<AppState>>) -> Response {
    let elapsed = state.pattern_start_time.read().await.elapsed();
    let pattern_config = if elapsed > state.idle_timeout {
        PatternConfig::Uniform { level: 0 }
    } else {
        state.pattern.read().await.clone()
    };
    let invert = *state.invert.read().await;

    let img = match generate_pattern(&pattern_config, state.width, state.height, invert) {
        Ok(img) => img,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(format!("Failed to generate pattern: {e}")))
                .unwrap()
        }
    };

    let mut jpeg_bytes = Vec::new();
    if let Err(e) = img.write_to(
        &mut std::io::Cursor::new(&mut jpeg_bytes),
        image::ImageFormat::Jpeg,
    ) {
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(format!("Failed to encode JPEG: {e}")))
            .unwrap();
    }

    Response::builder()
        .header(header::CONTENT_TYPE, "image/jpeg")
        .body(Body::from(jpeg_bytes))
        .unwrap()
}

async fn get_schema() -> Json<SchemaResponse> {
    Json(get_pattern_schemas())
}

async fn get_pattern_config(State(state): State<Arc<AppState>>) -> Response {
    let pattern_config = state.pattern.read().await.clone();
    let invert = *state.invert.read().await;

    // Convert internal PatternConfig to dynamic format for frontend
    let (pattern_id, values) = pattern_config_to_dynamic(&pattern_config);

    let json = serde_json::json!({
        "pattern_id": pattern_id,
        "values": values,
        "invert": invert,
    });

    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&json).unwrap()))
        .unwrap()
}

fn pattern_config_to_dynamic(config: &PatternConfig) -> (String, serde_json::Value) {
    use serde_json::json;
    match config {
        PatternConfig::April => ("April".into(), json!({})),
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
    }
}

#[derive(Debug, Deserialize)]
struct DynamicPatternRequest {
    pattern_id: String,
    values: serde_json::Map<String, serde_json::Value>,
    invert: Option<bool>,
}

fn dynamic_to_pattern_config(
    pattern_id: &str,
    values: &serde_json::Map<String, serde_json::Value>,
) -> Result<PatternConfig, String> {
    let get_i64 = |key: &str, default: i64| -> i64 {
        values.get(key).and_then(|v| v.as_i64()).unwrap_or(default)
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        values.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    };

    match pattern_id {
        "April" => Ok(PatternConfig::April),
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
        _ => Err(format!("Unknown pattern_id: {pattern_id}")),
    }
}

async fn update_pattern_config(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DynamicPatternRequest>,
) -> Response {
    match dynamic_to_pattern_config(&req.pattern_id, &req.values) {
        Ok(pattern) => {
            *state.pattern.write().await = pattern;

            if let Some(invert) = req.invert {
                *state.invert.write().await = invert;
            }

            *state.pattern_start_time.write().await = std::time::Instant::now();

            if let Some(ref tx) = state.display_update_tx {
                let _ = tx.send(());
            }

            Response::builder()
                .status(StatusCode::OK)
                .body(Body::from("Pattern updated"))
                .unwrap()
        }
        Err(e) => Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from(e))
            .unwrap(),
    }
}

#[tokio::main]
async fn run_web_server(state: Arc<AppState>, port: u16, bind_address: String) -> Result<()> {
    use axum::routing::get_service;
    use tower_http::services::ServeDir;

    let app = Router::new()
        .route("/", get(pattern_page))
        .route("/schema", get(get_schema))
        .route("/jpeg", get(jpeg_pattern_endpoint))
        .route("/config", get(get_pattern_config))
        .route("/config", axum::routing::post(update_pattern_config))
        .nest_service(
            "/static",
            get_service(ServeDir::new("test-bench-frontend/dist/calibrate")),
        )
        .with_state(state.clone());

    let addr: SocketAddr = format!("{bind_address}:{port}")
        .parse()
        .context("Invalid bind address")?;

    tracing::info!("Starting calibration pattern server on http://{}", addr);
    tracing::info!("Pattern resolution: {}x{}", state.width, state.height);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server error")?;

    tracing::info!("Server shut down gracefully");
    Ok(())
}

fn run_sdl_display(
    sdl_context: sdl2::Sdl,
    display_index: u32,
    width: u32,
    height: u32,
    state: Arc<AppState>,
    update_rx: mpsc::Receiver<()>,
) -> Result<()> {
    let video_subsystem = sdl_context
        .video()
        .map_err(|e| anyhow::anyhow!("Video subsystem init failed: {e}"))?;

    let display_bounds = video_subsystem
        .display_bounds(display_index as i32)
        .map_err(|e| anyhow::anyhow!("Failed to get display bounds: {e}"))?;

    let window = video_subsystem
        .window("Calibration Pattern", width, height)
        .position(display_bounds.x(), display_bounds.y())
        .fullscreen_desktop()
        .build()
        .context("Failed to create window")?;

    let mut canvas = window
        .into_canvas()
        .build()
        .context("Failed to create canvas")?;

    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator
        .create_texture_streaming(sdl2::pixels::PixelFormatEnum::RGB24, width, height)
        .map_err(|e| anyhow::anyhow!("Failed to create texture: {e:?}"))?;

    let mut event_pump = sdl_context
        .event_pump()
        .map_err(|e| anyhow::anyhow!("Failed to get event pump: {e}"))?;

    let get_current_pattern = || -> (PatternConfig, bool) {
        let pattern = state.pattern.blocking_read().clone();
        let invert = *state.invert.blocking_read();
        let elapsed = state.pattern_start_time.blocking_read().elapsed();

        let pattern_config = if elapsed > state.idle_timeout {
            PatternConfig::Uniform { level: 0 }
        } else {
            pattern
        };

        (pattern_config, invert)
    };

    let render_static_pattern =
        |pattern_config: &PatternConfig, invert: bool| -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
            generate_pattern(pattern_config, width, height, invert)
        };

    let is_animated = |pattern_config: &PatternConfig| -> bool {
        matches!(
            pattern_config,
            PatternConfig::Static { .. }
                | PatternConfig::CirclingPixel { .. }
                | PatternConfig::WigglingGaussian { .. }
        )
    };

    let (mut current_pattern, mut current_invert) = get_current_pattern();
    let mut img = render_static_pattern(&current_pattern, current_invert)?;
    let mut buffer: Vec<u8> = vec![0; (width * height * 3) as usize];
    let animated = is_animated(&current_pattern);

    if !animated {
        texture
            .update(None, img.as_raw(), (width * 3) as usize)
            .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;
    }

    canvas.clear();
    canvas
        .copy(&texture, None, None)
        .map_err(|e| anyhow::anyhow!("Failed to copy texture: {e}"))?;
    canvas.present();

    loop {
        for event in event_pump.poll_iter() {
            use sdl2::event::Event;
            match event {
                Event::Quit { .. } | Event::KeyDown { .. } => {
                    return Ok(());
                }
                _ => {}
            }
        }

        let mut pattern_changed = false;
        if update_rx.try_recv().is_ok() {
            (current_pattern, current_invert) = get_current_pattern();
            pattern_changed = true;

            if !is_animated(&current_pattern) {
                img = render_static_pattern(&current_pattern, current_invert)?;
                texture
                    .update(None, img.as_raw(), (width * 3) as usize)
                    .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;
            }
        }

        if is_animated(&current_pattern) || pattern_changed {
            match &current_pattern {
                PatternConfig::Static { pixel_size } => {
                    patterns::static_noise::generate_into_buffer(
                        &mut buffer,
                        width,
                        height,
                        *pixel_size,
                    );
                }
                PatternConfig::CirclingPixel {
                    orbit_count,
                    orbit_radius_percent,
                } => {
                    patterns::circling_pixel::generate_into_buffer(
                        &mut buffer,
                        width,
                        height,
                        *orbit_count,
                        *orbit_radius_percent,
                    );
                }
                PatternConfig::WigglingGaussian {
                    fwhm,
                    wiggle_radius,
                    intensity,
                } => {
                    patterns::wiggling_gaussian::generate_into_buffer(
                        &mut buffer,
                        width,
                        height,
                        *fwhm,
                        *wiggle_radius,
                        *intensity,
                    );
                }
                _ => {}
            }

            if is_animated(&current_pattern) {
                let buffer_to_use = if current_invert {
                    buffer.iter().map(|&b| 255 - b).collect()
                } else {
                    buffer.clone()
                };

                texture
                    .update(None, &buffer_to_use, (width * 3) as usize)
                    .map_err(|e| anyhow::anyhow!("Failed to update texture: {e:?}"))?;
            }
        }

        canvas.clear();
        canvas
            .copy(&texture, None, None)
            .map_err(|e| anyhow::anyhow!("Failed to copy texture: {e}"))?;
        canvas.present();

        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}

fn check_frontend_files() -> Result<()> {
    let wasm_file = "test-bench-frontend/dist/calibrate/calibrate_wasm_bg.wasm";
    let js_file = "test-bench-frontend/dist/calibrate/calibrate_wasm.js";

    if !std::path::Path::new(wasm_file).exists() || !std::path::Path::new(js_file).exists() {
        anyhow::bail!(
            "Frontend WASM files not found!\n\n\
            The calibrate server requires compiled Yew frontend files.\n\n\
            To build the frontends, run:\n\
            \x20   ./scripts/build-yew-frontends.sh\n\n\
            Or if you don't have trunk installed:\n\
            \x20   cargo install --locked trunk\n\
            \x20   ./scripts/build-yew-frontends.sh\n\n\
            Missing files:\n\
            \x20   - {wasm_file}\n\
            \x20   - {js_file}"
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    check_frontend_files()?;

    // Either wait for OLED or initialize SDL normally
    let (sdl_context, video_subsystem, display_index) = if args.wait_for_oled {
        let poll_interval = std::time::Duration::from_secs(args.poll_interval);
        wait_for_oled_display(poll_interval)?
    } else {
        let sdl_context = sdl2::init().map_err(|e| anyhow::anyhow!("SDL init failed: {e}"))?;
        let video_subsystem = sdl_context
            .video()
            .map_err(|e| anyhow::anyhow!("Video subsystem init failed: {e}"))?;

        if args.list {
            return list_displays(&video_subsystem);
        }

        let display_index = resolve_display_index(&video_subsystem, args.display)?;
        (sdl_context, video_subsystem, display_index)
    };

    let (width, height) = get_display_resolution(&video_subsystem, display_index)?;

    tracing::info!("Using display {display_index}: {}x{}", width, height);

    let (display_update_tx, display_update_rx) = mpsc::channel::<()>();

    let state = Arc::new(AppState {
        pattern: Arc::new(RwLock::new(PatternConfig::default())),
        width,
        height,
        invert: Arc::new(RwLock::new(false)),
        pattern_start_time: Arc::new(RwLock::new(std::time::Instant::now())),
        display_update_tx: Some(display_update_tx),
        idle_timeout: std::time::Duration::from_secs(args.idle_timeout),
    });

    let state_clone = state.clone();
    let port = args.port;
    let bind_address = args.bind_address.clone();

    let web_thread = std::thread::spawn(move || run_web_server(state_clone, port, bind_address));

    run_sdl_display(
        sdl_context,
        display_index,
        width,
        height,
        state,
        display_update_rx,
    )?;

    drop(web_thread);
    tracing::info!("Shut down complete");
    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C, shutting down...");
        },
        _ = terminate => {
            tracing::info!("Received SIGTERM, shutting down...");
        },
    }
}

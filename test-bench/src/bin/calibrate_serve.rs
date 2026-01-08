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
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use test_bench::calibrate::{
    get_pattern_schemas, parse_pattern_request, pattern_to_dynamic, run_display, DisplayConfig,
    DynamicPattern, OledSafetyWatchdog, PatternConfig, SchemaResponse,
};
use test_bench::display_patterns::remote_controlled::RemotePatternState;
use test_bench::display_utils::{
    estimate_pixel_pitch_um, get_display_resolution, list_displays, resolve_display_index,
    wait_for_oled_display, OLED_HEIGHT, OLED_PIXEL_PITCH_UM, OLED_WIDTH,
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
        default_value = "600",
        help = "Seconds of inactivity before screen goes black (OLED burn-in protection)"
    )]
    idle_timeout: u64,

    #[arg(
        long,
        default_value = "tcp://*:5556",
        help = "ZMQ REP bind address for receiving pattern commands"
    )]
    zmq_bind: String,
}

struct AppState {
    pattern: Arc<RwLock<PatternConfig>>,
    width: u32,
    height: u32,
    invert: Arc<RwLock<bool>>,
    watchdog: OledSafetyWatchdog,
    display_update_tx: Option<mpsc::Sender<()>>,
    /// Shared remote pattern state for ZMQ-commanded patterns
    remote_state: Arc<Mutex<RemotePatternState>>,
    /// Display name from SDL
    display_name: String,
    /// Display index for DPI queries
    display_index: u32,
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
    let pattern_config = if state.watchdog.is_timed_out() {
        PatternConfig::Uniform { level: 0 }
    } else {
        state.pattern.read().await.clone()
    };
    let invert = *state.invert.read().await;

    let mut img = match pattern_config.generate(state.width, state.height) {
        Ok(img) => img,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(format!("Failed to generate pattern: {e}")))
                .unwrap()
        }
    };

    if invert {
        PatternConfig::apply_invert(&mut img);
    }

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

/// Get display info for auto-discovery.
///
/// Returns DisplayInfo with dimensions and pixel pitch.
/// For OLED displays (2560x2560), pixel pitch is hardcoded to 7.0um
/// because the firmware misreports it. For other displays, attempts
/// to estimate from DPI. Returns None for pixel_pitch_um if unavailable.
async fn get_display_info(
    State(state): State<Arc<AppState>>,
) -> Json<shared::system_info::DisplayInfo> {
    let is_oled = state.width == OLED_WIDTH && state.height == OLED_HEIGHT;

    let pixel_pitch_um = estimate_pixel_pitch_um(state.width, state.height, state.display_index);

    // Log what we found
    match (pixel_pitch_um, is_oled) {
        (Some(_), true) => {
            tracing::warn!(
                "OLED display detected ({}x{}): Using hardcoded pixel pitch of {}um \
                (firmware misreports this value)",
                state.width,
                state.height,
                OLED_PIXEL_PITCH_UM
            );
        }
        (Some(pitch), false) => {
            tracing::info!(
                "Display ({}x{}): Estimated pixel pitch {:.2}um from DPI",
                state.width,
                state.height,
                pitch
            );
        }
        (None, _) => {
            tracing::warn!(
                "Display ({}x{}): Could not determine pixel pitch (DPI unavailable)",
                state.width,
                state.height
            );
        }
    }

    let name = if is_oled {
        format!("OLED {}x{}", state.width, state.height)
    } else {
        state.display_name.clone()
    };

    Json(shared::system_info::DisplayInfo {
        width: state.width,
        height: state.height,
        pixel_pitch_um,
        name,
    })
}

async fn get_pattern_config(
    State(state): State<Arc<AppState>>,
) -> Json<test_bench_shared::PatternConfigResponse> {
    let pattern_config = state.pattern.read().await.clone();
    let invert = *state.invert.read().await;

    let (pattern_id, values) = pattern_to_dynamic(&pattern_config);

    Json(test_bench_shared::PatternConfigResponse {
        pattern_id,
        values,
        invert,
    })
}

#[derive(Debug, Deserialize)]
struct DynamicPatternRequest {
    pattern_id: String,
    values: serde_json::Map<String, serde_json::Value>,
    invert: Option<bool>,
}

async fn update_pattern_config(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DynamicPatternRequest>,
) -> Response {
    // Special handling for RemoteControlled - use shared state
    let pattern_result = if req.pattern_id == "RemoteControlled" {
        Ok(PatternConfig::RemoteControlled {
            state: state.remote_state.clone(),
            pattern_size: shared::image_size::PixelShape {
                width: state.width as usize,
                height: state.height as usize,
            },
        })
    } else {
        parse_pattern_request(
            &req.pattern_id,
            &req.values,
            Some((state.width, state.height)),
        )
    };

    match pattern_result {
        Ok(pattern) => {
            *state.pattern.write().await = pattern;

            if let Some(invert) = req.invert {
                *state.invert.write().await = invert;
            }

            state.watchdog.reset();

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
        .route("/info", get(get_display_info))
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
    let (sdl_context, display_index, width, height, display_name) = if args.wait_for_oled {
        let poll_interval = std::time::Duration::from_secs(args.poll_interval);
        let (sdl_context, video_subsystem, display_index) = wait_for_oled_display(poll_interval)?;
        let (width, height) = get_display_resolution(&video_subsystem, display_index)?;
        let display_name = video_subsystem
            .display_name(display_index as i32)
            .unwrap_or_else(|_| format!("Display {display_index}"));
        drop(video_subsystem);
        (sdl_context, display_index, width, height, display_name)
    } else {
        let sdl_context = sdl2::init().map_err(|e| anyhow::anyhow!("SDL init failed: {e}"))?;
        let video_subsystem = sdl_context
            .video()
            .map_err(|e| anyhow::anyhow!("Video subsystem init failed: {e}"))?;

        if args.list {
            return list_displays(&video_subsystem);
        }

        let display_index = resolve_display_index(&video_subsystem, args.display)?;
        let (width, height) = get_display_resolution(&video_subsystem, display_index)?;
        let display_name = video_subsystem
            .display_name(display_index as i32)
            .unwrap_or_else(|_| format!("Display {display_index}"));
        drop(video_subsystem);
        (sdl_context, display_index, width, height, display_name)
    };

    tracing::info!("Using display {display_index}: {}x{}", width, height);

    let (display_update_tx, display_update_rx) = mpsc::channel::<()>();

    // Create shared remote pattern state
    let remote_state = Arc::new(Mutex::new(RemotePatternState::new()));

    // Create watchdog for OLED burn-in protection
    let watchdog = OledSafetyWatchdog::new(std::time::Duration::from_secs(args.idle_timeout));

    // Bind ZMQ REP socket for receiving pattern commands
    let zmq_bind = args.zmq_bind.clone();
    let remote_state_zmq = remote_state.clone();
    let display_update_tx_zmq = display_update_tx.clone();
    let watchdog_zmq = watchdog.clone();

    std::thread::spawn(move || {
        let ctx = zmq::Context::new();
        let socket = match ctx.socket(zmq::REP) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Failed to create ZMQ REP socket: {e}");
                return;
            }
        };

        if let Err(e) = socket.bind(&zmq_bind) {
            tracing::error!("Failed to bind ZMQ REP socket to {}: {e}", zmq_bind);
            return;
        }

        tracing::info!("ZMQ REP socket listening on {}", zmq_bind);

        loop {
            // Receive command
            let msg = match socket.recv_string(0) {
                Ok(Ok(s)) => s,
                Ok(Err(_)) => {
                    tracing::warn!("Received non-UTF8 ZMQ message");
                    let _ = socket.send("error: invalid UTF-8", 0);
                    continue;
                }
                Err(e) => {
                    tracing::error!("ZMQ recv error: {e}");
                    break;
                }
            };

            // Parse command
            match serde_json::from_str::<shared::pattern_command::PatternCommand>(&msg) {
                Ok(cmd) => {
                    tracing::debug!("Received pattern command: {:?}", cmd);
                    remote_state_zmq.lock().unwrap().set_command(cmd);

                    // Reset idle timeout so ZMQ commands keep display active
                    watchdog_zmq.reset();

                    // Notify display to update
                    let _ = display_update_tx_zmq.send(());

                    // Send acknowledgment
                    let _ = socket.send("ok", 0);
                }
                Err(e) => {
                    tracing::warn!("Failed to parse pattern command: {e}");
                    let _ = socket.send(&format!("error: {e}"), 0);
                }
            }
        }
    });

    let state = Arc::new(AppState {
        pattern: Arc::new(RwLock::new(PatternConfig::default())),
        width,
        height,
        invert: Arc::new(RwLock::new(false)),
        watchdog: watchdog.clone(),
        display_update_tx: Some(display_update_tx),
        remote_state,
        display_name,
        display_index,
    });

    let state_clone = state.clone();
    let port = args.port;
    let bind_address = args.bind_address.clone();

    let web_thread = std::thread::spawn(move || run_web_server(state_clone, port, bind_address));

    let dynamic_source = DynamicPattern::new(
        state.pattern.clone(),
        state.invert.clone(),
        state.watchdog.clone(),
        display_update_rx,
    );

    let display_config = DisplayConfig {
        width,
        height,
        display_index,
        show_fps: false,
    };

    run_display(sdl_context, display_config, dynamic_source)?;

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

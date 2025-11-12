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
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::mpsc;
use std::sync::Arc;
use test_bench::display_patterns as patterns;
use test_bench::display_utils::{get_display_resolution, list_displays};
use tokio::sync::RwLock;

#[derive(RustEmbed)]
#[folder = "templates/"]
struct Templates;

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
}

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
    let template_content = Templates::get("calibrate_view.html")
        .map(|file| String::from_utf8_lossy(&file.data).to_string())
        .unwrap_or_else(|| "<html><body>Template not found</body></html>".to_string());

    let html = template_content
        .replace("{width}", &state.width.to_string())
        .replace("{height}", &state.height.to_string());

    Html(html)
}

async fn jpeg_pattern_endpoint(State(state): State<Arc<AppState>>) -> Response {
    let elapsed = state.pattern_start_time.read().await.elapsed();
    let pattern_config = if elapsed > std::time::Duration::from_secs(600) {
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

async fn get_pattern_config(State(state): State<Arc<AppState>>) -> Response {
    let pattern_config = state.pattern.read().await.clone();
    let invert = *state.invert.read().await;

    let json = serde_json::json!({
        "pattern": pattern_config,
        "invert": invert,
    });

    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&json).unwrap()))
        .unwrap()
}

#[derive(Debug, Deserialize)]
struct UpdatePatternRequest {
    pattern: PatternConfig,
    invert: Option<bool>,
}

async fn update_pattern_config(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UpdatePatternRequest>,
) -> Response {
    *state.pattern.write().await = req.pattern;

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

#[tokio::main]
async fn run_web_server(state: Arc<AppState>, port: u16, bind_address: String) -> Result<()> {
    let app = Router::new()
        .route("/", get(pattern_page))
        .route("/jpeg", get(jpeg_pattern_endpoint))
        .route("/config", get(get_pattern_config))
        .route("/config", axum::routing::post(update_pattern_config))
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

        let pattern_config = if elapsed > std::time::Duration::from_secs(600) {
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

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let sdl_context = sdl2::init().map_err(|e| anyhow::anyhow!("SDL init failed: {e}"))?;
    let video_subsystem = sdl_context
        .video()
        .map_err(|e| anyhow::anyhow!("Video subsystem init failed: {e}"))?;

    if args.list {
        return list_displays(&video_subsystem);
    }

    let display_index = args.display.unwrap_or(0);
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

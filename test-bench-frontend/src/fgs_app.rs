use std::collections::VecDeque;

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;

use crate::fgs::{
    histogram::render_histogram,
    views::{FsmView, StarDetectionSettingsView, StatsView, TrackingView, ZoomView},
    LogViewer,
};
use crate::ws_image_stream::WsImageStream;

pub use shared_wasm::CameraStats;
pub use shared_wasm::{
    CommandError, ExportSettings, ExportStatus, FgsWsCommand, FgsWsMessage, FsmMoveRequest,
    FsmStatus, StarDetectionSettings, TrackingEnableRequest, TrackingSettings, TrackingState,
    TrackingStatus,
};

/// Time window for sparkline plots in seconds.
const HISTORY_WINDOW_SECS: f64 = 10.0;

/// Hard cap on stored points to prevent unbounded growth.
const HISTORY_MAX_POINTS: usize = 10_000;

/// Single tracking data point with position, SNR, and timestamp.
#[derive(Clone, Copy, PartialEq)]
pub struct TrackingPoint {
    pub x: f64,
    pub y: f64,
    pub snr: f64,
    /// Timestamp in seconds (with fractional nanos)
    pub t: f64,
}

/// Rolling history buffer for tracking data visualization.
///
/// Retains points within a fixed time window rather than a fixed sample count,
/// so the sparkline duration stays consistent regardless of server update rate.
#[derive(Clone, PartialEq, Default)]
pub struct TrackingHistory {
    points: VecDeque<TrackingPoint>,
}

impl TrackingHistory {
    fn push(&mut self, x: f64, y: f64, snr: f64, timestamp_sec: u64, timestamp_nanos: u64) {
        let t = timestamp_sec as f64 + (timestamp_nanos as f64) / 1_000_000_000.0;
        self.points.push_back(TrackingPoint { x, y, snr, t });

        // Evict points outside the time window
        while self
            .points
            .front()
            .is_some_and(|oldest| t - oldest.t > HISTORY_WINDOW_SECS)
        {
            self.points.pop_front();
        }

        // Hard cap as safety bound
        while self.points.len() > HISTORY_MAX_POINTS {
            self.points.pop_front();
        }
    }

    pub fn points(&self) -> &VecDeque<TrackingPoint> {
        &self.points
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    fn clear(&mut self) {
        self.points.clear();
    }
}

#[derive(Properties, PartialEq)]
pub struct FgsFrontendProps {
    pub device: String,
    pub width: u32,
    pub height: u32,
}

pub struct FgsFrontend {
    stats: Option<CameraStats>,
    show_annotation: bool,
    log_scale_histogram: bool,
    overlay_refresh_handle: Option<gloo_timers::callback::Interval>,
    ws_connected: bool,
    ws_command_tx: Option<futures_channel::mpsc::UnboundedSender<String>>,
    // Zoom state
    zoom_center: Option<(u32, u32)>,
    // Tracking state
    tracking_available: bool,
    tracking_status: Option<TrackingStatus>,
    tracking_toggle_pending: bool,
    tracking_settings: Option<TrackingSettings>,
    tracking_settings_pending: bool,
    show_tracking_settings: bool,
    // Position and SNR history for plotting
    tracking_history: TrackingHistory,
    // Export state
    export_status: Option<ExportStatus>,
    export_settings_pending: bool,
    show_export_settings: bool,
    // FSM state
    fsm_status: Option<FsmStatus>,
    fsm_move_pending: bool,
    fsm_target_x: f64,
    fsm_target_y: f64,
    // Star detection state
    star_detection_settings: Option<StarDetectionSettings>,
    star_detection_pending: bool,
    show_star_detection_settings: bool,
    // SVG overlay for star detection (much faster than re-encoding full image)
    overlay_svg: Option<String>,
    // Current frame dimensions (from WebSocket stream)
    frame_size: Option<(u32, u32)>,
}

pub enum Msg {
    // WebSocket status messages
    FgsWs(FgsWsMessage),
    WsConnected(bool),
    WsCommandChannelReady(futures_channel::mpsc::UnboundedSender<String>),
    RefreshOverlay,
    ToggleAnnotation,
    ToggleLogScale,
    // Zoom messages
    ImageClicked(i32, i32),
    ClearZoom,
    // Tracking messages
    ToggleTracking,
    // Tracking settings messages
    ToggleTrackingSettings,
    UpdateTrackingSetting(String, f64),
    SaveTrackingSettings,
    // Export settings messages
    ToggleExportSettings,
    UpdateExportSetting(String, String),
    ToggleExportBool(String),
    SaveExportSettings,
    // FSM messages
    UpdateFsmTarget(String, f64),
    MoveFsm,
    // Star detection messages
    ToggleStarDetectionSettings,
    UpdateStarDetectionSetting(String, String),
    SaveStarDetectionSettings,
    // SVG overlay message
    OverlaySvgLoaded(String),
    // WebSocket stream frame size changed
    FrameSizeChanged(u32, u32),
}

impl FgsFrontend {
    fn send_command(&self, cmd: FgsWsCommand) {
        if let Some(ref tx) = self.ws_command_tx {
            if let Ok(json) = serde_json::to_string(&cmd) {
                let _ = tx.unbounded_send(json);
            }
        }
    }
}

impl Component for FgsFrontend {
    type Message = Msg;
    type Properties = FgsFrontendProps;

    fn create(ctx: &Context<Self>) -> Self {
        // Start WebSocket connection for status updates
        let link = ctx.link().clone();
        spawn_local(async move {
            connect_fgs_status_ws("/ws/status".to_string(), link).await;
        });

        Self {
            stats: None,
            show_annotation: false,
            log_scale_histogram: false,
            overlay_refresh_handle: None,
            ws_connected: false,
            ws_command_tx: None,
            zoom_center: None,
            tracking_available: false,
            tracking_status: None,
            tracking_toggle_pending: false,
            tracking_settings: None,
            tracking_settings_pending: false,
            show_tracking_settings: false,
            tracking_history: TrackingHistory::default(),
            export_status: None,
            export_settings_pending: false,
            show_export_settings: false,
            fsm_status: None,
            fsm_move_pending: false,
            fsm_target_x: 0.0,
            fsm_target_y: 0.0,
            star_detection_settings: None,
            star_detection_pending: false,
            show_star_detection_settings: false,
            overlay_svg: None,
            frame_size: None,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::FgsWs(ws_msg) => match ws_msg {
                FgsWsMessage::CameraStats(stats) => {
                    self.stats = Some(stats);
                    true
                }
                FgsWsMessage::TrackingStatus(status) => {
                    self.tracking_available = true;
                    self.tracking_toggle_pending = false;

                    // Check if we just left Tracking state - clear history
                    let was_tracking = matches!(
                        self.tracking_status.as_ref().map(|s| &s.state),
                        Some(TrackingState::Tracking { .. })
                    );
                    let is_tracking = matches!(&status.state, TrackingState::Tracking { .. });

                    if was_tracking && !is_tracking {
                        self.tracking_history.clear();
                    }

                    if let Some(ref pos) = status.position {
                        self.tracking_history.push(
                            pos.x,
                            pos.y,
                            pos.snr,
                            pos.timestamp_sec,
                            pos.timestamp_nanos,
                        );
                    }
                    self.tracking_status = Some(status);
                    true
                }
                FgsWsMessage::TrackingSettings(settings) => {
                    self.tracking_settings_pending = false;
                    self.tracking_settings = Some(settings);
                    true
                }
                FgsWsMessage::ExportStatus(status) => {
                    self.export_settings_pending = false;
                    self.export_status = Some(status);
                    true
                }
                FgsWsMessage::FsmStatus(status) => {
                    self.fsm_move_pending = false;
                    if self.fsm_status.is_none() {
                        self.fsm_target_x = status.x_urad;
                        self.fsm_target_y = status.y_urad;
                    }
                    self.fsm_status = Some(status);
                    true
                }
                FgsWsMessage::DetectionSettings(settings) => {
                    self.star_detection_pending = false;
                    if !settings.enabled {
                        self.overlay_svg = None;
                    }
                    self.star_detection_settings = Some(settings);
                    true
                }
                FgsWsMessage::CommandError(err) => {
                    match err.command.as_str() {
                        "SetTrackingEnabled" => self.tracking_toggle_pending = false,
                        "SetTrackingSettings" => self.tracking_settings_pending = false,
                        "SetDetectionSettings" => self.star_detection_pending = false,
                        "SetExportSettings" => self.export_settings_pending = false,
                        "MoveFsm" => self.fsm_move_pending = false,
                        _ => {}
                    }
                    web_sys::console::error_1(
                        &format!("Command error ({}): {}", err.command, err.message).into(),
                    );
                    true
                }
            },
            Msg::WsConnected(connected) => {
                self.ws_connected = connected;
                if !connected {
                    self.ws_command_tx = None;
                    // Reconnect after a delay
                    let link = ctx.link().clone();
                    gloo_timers::callback::Timeout::new(2000, move || {
                        spawn_local(async move {
                            connect_fgs_status_ws("/ws/status".to_string(), link).await;
                        });
                    })
                    .forget();
                }
                false
            }
            Msg::WsCommandChannelReady(tx) => {
                self.ws_command_tx = Some(tx);
                false
            }
            Msg::RefreshOverlay => {
                if self.show_annotation {
                    let link = ctx.link().clone();
                    spawn_local(async move {
                        if let Some(svg) = crate::fgs::api::fetch_text("/overlay-svg").await {
                            link.send_message(Msg::OverlaySvgLoaded(svg));
                        }
                    });
                }
                false
            }
            Msg::ToggleAnnotation => {
                self.show_annotation = !self.show_annotation;
                if self.show_annotation {
                    // Start overlay polling timer
                    let link = ctx.link().clone();
                    spawn_local(async move {
                        if let Some(svg) = crate::fgs::api::fetch_text("/overlay-svg").await {
                            link.send_message(Msg::OverlaySvgLoaded(svg));
                        }
                    });
                    let link2 = ctx.link().clone();
                    self.overlay_refresh_handle =
                        Some(gloo_timers::callback::Interval::new(500, move || {
                            link2.send_message(Msg::RefreshOverlay);
                        }));
                } else {
                    self.overlay_svg = None;
                    self.overlay_refresh_handle = None;
                }
                true
            }
            Msg::ToggleLogScale => {
                self.log_scale_histogram = !self.log_scale_histogram;
                true
            }
            Msg::ImageClicked(x, y) => {
                if let Some(window) = web_sys::window() {
                    if let Some(document) = window.document() {
                        if let Some(img) = document.get_element_by_id("camera-frame") {
                            if let Some(img) = img.dyn_ref::<web_sys::HtmlImageElement>() {
                                let natural_width = img.natural_width() as f64;
                                let natural_height = img.natural_height() as f64;
                                let display_width = img.client_width() as f64;
                                let display_height = img.client_height() as f64;

                                if natural_width > 0.0 && display_width > 0.0 {
                                    let scale_x = natural_width / display_width;
                                    let scale_y = natural_height / display_height;
                                    let real_x = (x as f64 * scale_x) as u32;
                                    let real_y = (y as f64 * scale_y) as u32;
                                    self.zoom_center = Some((real_x, real_y));
                                    return true;
                                }
                            }
                        }
                    }
                }
                false
            }
            Msg::ClearZoom => {
                self.zoom_center = None;
                true
            }
            Msg::ToggleTracking => {
                if self.tracking_toggle_pending {
                    return false;
                }
                let new_enabled = !self
                    .tracking_status
                    .as_ref()
                    .map(|s| s.enabled)
                    .unwrap_or(false);
                self.tracking_toggle_pending = true;
                self.send_command(FgsWsCommand::SetTrackingEnabled(TrackingEnableRequest {
                    enabled: new_enabled,
                }));
                true
            }
            Msg::ToggleTrackingSettings => {
                self.show_tracking_settings = !self.show_tracking_settings;
                true
            }
            Msg::UpdateTrackingSetting(field, value) => {
                if let Some(ref mut settings) = self.tracking_settings {
                    match field.as_str() {
                        "acquisition_frames" => settings.acquisition_frames = value as usize,
                        "roi_size" => settings.roi_size = value as usize,
                        "detection_threshold_sigma" => settings.detection_threshold_sigma = value,
                        "snr_min" => settings.snr_min = value,
                        "snr_dropout_threshold" => settings.snr_dropout_threshold = value,
                        "fwhm" => settings.fwhm = value,
                        _ => {}
                    }
                }
                true
            }
            Msg::SaveTrackingSettings => {
                if self.tracking_settings_pending {
                    return false;
                }
                self.tracking_settings_pending = true;
                if let Some(settings) = self.tracking_settings.clone() {
                    self.send_command(FgsWsCommand::SetTrackingSettings(settings));
                }
                true
            }
            Msg::ToggleExportSettings => {
                self.show_export_settings = !self.show_export_settings;
                true
            }
            Msg::UpdateExportSetting(field, value) => {
                if let Some(ref mut status) = self.export_status {
                    match field.as_str() {
                        "csv_filename" => status.settings.csv_filename = value,
                        "frames_directory" => status.settings.frames_directory = value,
                        _ => {}
                    }
                }
                true
            }
            Msg::ToggleExportBool(field) => {
                if let Some(ref mut status) = self.export_status {
                    match field.as_str() {
                        "csv_enabled" => status.settings.csv_enabled = !status.settings.csv_enabled,
                        "frames_enabled" => {
                            status.settings.frames_enabled = !status.settings.frames_enabled
                        }
                        _ => {}
                    }
                }
                true
            }
            Msg::SaveExportSettings => {
                if self.export_settings_pending {
                    return false;
                }
                self.export_settings_pending = true;
                if let Some(ref status) = self.export_status {
                    self.send_command(FgsWsCommand::SetExportSettings(status.settings.clone()));
                }
                true
            }
            Msg::UpdateFsmTarget(axis, value) => {
                match axis.as_str() {
                    "x" => self.fsm_target_x = value,
                    "y" => self.fsm_target_y = value,
                    _ => {}
                }
                true
            }
            Msg::MoveFsm => {
                if self.fsm_move_pending {
                    return false;
                }
                self.fsm_move_pending = true;
                self.send_command(FgsWsCommand::MoveFsm(FsmMoveRequest {
                    x_urad: self.fsm_target_x,
                    y_urad: self.fsm_target_y,
                }));
                true
            }
            // Star detection settings messages
            Msg::ToggleStarDetectionSettings => {
                self.show_star_detection_settings = !self.show_star_detection_settings;
                true
            }
            Msg::UpdateStarDetectionSetting(field, value) => {
                if let Some(ref mut settings) = self.star_detection_settings {
                    match field.as_str() {
                        "enabled" => settings.enabled = !settings.enabled,
                        "detection_sigma" => {
                            if let Ok(v) = value.parse() {
                                settings.detection_sigma = v;
                            }
                        }
                        "min_bad_pixel_distance" => {
                            if let Ok(v) = value.parse() {
                                settings.min_bad_pixel_distance = v;
                            }
                        }
                        "max_aspect_ratio" => {
                            if let Ok(v) = value.parse() {
                                settings.max_aspect_ratio = v;
                            }
                        }
                        "min_flux" => {
                            if let Ok(v) = value.parse() {
                                settings.min_flux = v;
                            }
                        }
                        _ => {}
                    }
                }
                true
            }
            Msg::SaveStarDetectionSettings => {
                if self.star_detection_pending {
                    return false;
                }
                self.star_detection_pending = true;
                if let Some(settings) = self.star_detection_settings.clone() {
                    self.send_command(FgsWsCommand::SetDetectionSettings(settings));
                }
                true
            }
            Msg::OverlaySvgLoaded(svg) => {
                self.overlay_svg = Some(svg);
                true
            }
            Msg::FrameSizeChanged(width, height) => {
                // WebSocket stream reported new frame dimensions
                // This triggers when the stream reconnects after server-side size change
                self.frame_size = Some((width, height));
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let props = ctx.props();

        // Callbacks for the WebSocket image stream component
        let onclick = ctx
            .link()
            .callback(|(x, y): (i32, i32)| Msg::ImageClicked(x, y));
        let ontouchstart = ctx
            .link()
            .callback(|(x, y): (i32, i32)| Msg::ImageClicked(x, y));
        let on_size_change = ctx
            .link()
            .callback(|(width, height): (u32, u32)| Msg::FrameSizeChanged(width, height));

        html! {
            <>
                <div class="column left-panel">
                    <h2>{"Camera Info"}</h2>
                    <div class="metadata-item">
                        <span class="metadata-label">{"Device:"}</span><br/>
                        {self.stats.as_ref()
                            .map(|s| s.device_name.as_str())
                            .unwrap_or(&props.device)}
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">{"Resolution:"}</span><br/>
                        {if let Some(stats) = &self.stats {
                            format!("{}x{}", stats.width, stats.height)
                        } else {
                            format!("{}x{}", props.width, props.height)
                        }}
                    </div>

                    <TrackingView
                        available={self.tracking_available}
                        status={self.tracking_status.clone()}
                        toggle_pending={self.tracking_toggle_pending}
                        on_toggle_tracking={ctx.link().callback(|_| Msg::ToggleTracking)}
                        history={self.tracking_history.clone()}
                        show_overlay={self.show_annotation}
                        on_toggle_overlay={ctx.link().callback(|_| Msg::ToggleAnnotation)}
                        show_settings={self.show_tracking_settings}
                        settings={self.tracking_settings.clone()}
                        settings_pending={self.tracking_settings_pending}
                        on_toggle_settings={ctx.link().callback(|_| Msg::ToggleTrackingSettings)}
                        on_update_setting={ctx.link().callback(|(field, val)| Msg::UpdateTrackingSetting(field, val))}
                        on_save_settings={ctx.link().callback(|_| Msg::SaveTrackingSettings)}
                        show_export={self.show_export_settings}
                        export_status={self.export_status.clone()}
                        export_pending={self.export_settings_pending}
                        on_toggle_export={ctx.link().callback(|_| Msg::ToggleExportSettings)}
                        on_update_export_string={ctx.link().callback(|(field, val)| Msg::UpdateExportSetting(field, val))}
                        on_toggle_export_bool={ctx.link().callback(Msg::ToggleExportBool)}
                        on_save_export={ctx.link().callback(|_| Msg::SaveExportSettings)}
                    />

                    <FsmView
                        status={self.fsm_status.clone()}
                        target_x={self.fsm_target_x}
                        target_y={self.fsm_target_y}
                        move_pending={self.fsm_move_pending}
                        on_update_target={ctx.link().callback(|(axis, val)| Msg::UpdateFsmTarget(axis, val))}
                        on_move={ctx.link().callback(|_| Msg::MoveFsm)}
                    />

                    <StarDetectionSettingsView
                        show={self.show_star_detection_settings}
                        settings={self.star_detection_settings.clone()}
                        pending={self.star_detection_pending}
                        on_toggle={ctx.link().callback(|_| Msg::ToggleStarDetectionSettings)}
                        on_update={ctx.link().callback(|(field, val)| Msg::UpdateStarDetectionSetting(field, val))}
                        on_save={ctx.link().callback(|_| Msg::SaveStarDetectionSettings)}
                    />

                    <h2 style="margin-top: 30px;">{"Endpoints"}</h2>
                    <div class="metadata-item">
                        <a href="/jpeg">{"JPEG Frame"}</a><br/>
                        <a href="/raw">{"Raw Frame"}</a><br/>
                        <a href="/annotated">{"Annotated Frame"}</a><br/>
                        <a href="/stats">{"Frame Stats (JSON)"}</a>
                    </div>
                </div>

                <div class="column center-panel">
                    <div class="frame-info">
                        <span id="update-time"></span><br/>
                        <span id="frame-timestamp" style="color: #00aa00; font-size: 0.9em;"></span>
                    </div>
                    <div class="image-container" style="position: relative;">
                        <WsImageStream
                            id={"camera-frame".to_string()}
                            class={"image-frame".to_string()}
                            onclick={Some(onclick)}
                            ontouchstart={Some(ontouchstart)}
                            on_size_change={Some(on_size_change)}
                        />
                        {if let Some(svg) = &self.overlay_svg {
                            // The SVG needs width/height 100% to scale with the image
                            let svg_with_style = svg.replacen(
                                r#"<svg "#,
                                r#"<svg style="width: 100%; height: 100%;" "#,
                                1
                            );
                            Html::from_html_unchecked(AttrValue::from(format!(
                                r#"<div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;">{svg_with_style}</div>"#
                            )))
                        } else {
                            html! {}
                        }}
                    </div>
                </div>

                <div class="column right-panel">
                    <h2>{"Statistics"}</h2>
                    <StatsView stats={self.stats.clone()} />

                    <h2 style="margin-top: 30px;">{"Zoom Region"}</h2>
                    <ZoomView
                        zoom_center={self.zoom_center}
                        on_clear={ctx.link().callback(|_| Msg::ClearZoom)}
                    />

                    <h2 style="margin-top: 30px;">{"Histogram"}</h2>
                    <div class="metadata-item">
                        <label style="cursor: pointer;">
                            <input
                                type="checkbox"
                                checked={self.log_scale_histogram}
                                onchange={ctx.link().callback(|_| Msg::ToggleLogScale)}
                                style="width: 20px; height: 20px; vertical-align: middle;"
                            />
                            <span style="margin-left: 5px;">{"Log Scale"}</span>
                        </label>
                    </div>
                    <canvas id="histogram-canvas" width="300" height="150" style="width: 100%;"></canvas>
                    <div id="histogram-info" style="font-size: 0.7em; color: #00aa00; margin-top: 5px;"></div>

                    <h2 style="margin-top: 30px;">{"Logs"}</h2>
                    <LogViewer max_height="200px" />
                </div>
            </>
        }
    }

    fn destroy(&mut self, _ctx: &Context<Self>) {
        self.overlay_refresh_handle = None;
    }

    fn rendered(&mut self, _ctx: &Context<Self>, _first_render: bool) {
        if let Some(ref stats) = self.stats {
            render_histogram(
                "histogram-canvas",
                "histogram-info",
                &stats.histogram,
                stats.histogram_mean,
                stats.histogram_max,
                self.log_scale_histogram,
            );
        }
    }
}

/// Connect to the FGS status WebSocket for bidirectional communication.
///
/// Reads `FgsWsMessage` status updates from the server and forwards them to
/// the component. Provides a command channel for sending `FgsWsCommand` messages.
async fn connect_fgs_status_ws(url: String, link: yew::html::Scope<FgsFrontend>) {
    use futures_util::{SinkExt, StreamExt};
    use gloo_net::websocket::{futures::WebSocket, Message};

    let ws_url = if url.starts_with("ws://") || url.starts_with("wss://") {
        url
    } else {
        let window = web_sys::window().expect("no window");
        let location = window.location();
        let protocol = if location.protocol().unwrap_or_default() == "https:" {
            "wss:"
        } else {
            "ws:"
        };
        let host = location.host().unwrap_or_default();
        format!("{protocol}//{host}{url}")
    };

    match WebSocket::open(&ws_url) {
        Ok(ws) => {
            link.send_message(Msg::WsConnected(true));

            let (mut write, mut read) = ws.split();
            let (cmd_tx, mut cmd_rx) = futures_channel::mpsc::unbounded::<String>();

            link.send_message(Msg::WsCommandChannelReady(cmd_tx));

            // Spawn write loop: forwards command channel messages to WebSocket
            spawn_local(async move {
                while let Some(text) = cmd_rx.next().await {
                    if write.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
            });

            while let Some(msg) = read.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        if let Ok(fgs_msg) = serde_json::from_str::<FgsWsMessage>(&text) {
                            link.send_message(Msg::FgsWs(fgs_msg));
                        }
                    }
                    Ok(Message::Bytes(_)) => {}
                    Err(e) => {
                        web_sys::console::log_1(
                            &format!("FGS status WebSocket error: {e:?}").into(),
                        );
                        break;
                    }
                }
            }

            link.send_message(Msg::WsConnected(false));
        }
        Err(e) => {
            web_sys::console::log_1(
                &format!("Failed to connect FGS status WebSocket: {e:?}").into(),
            );
            link.send_message(Msg::WsConnected(false));
        }
    }
}

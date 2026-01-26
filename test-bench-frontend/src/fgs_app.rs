use test_bench_shared::RingBuffer;
use wasm_bindgen::JsCast;
use yew::prelude::*;

use crate::fgs::{
    api::{calculate_backoff_delay, FgsError, FgsServerClient},
    histogram::render_histogram,
    views::{FsmView, StarDetectionSettingsView, StatsView, TrackingView, ZoomView},
};

pub use test_bench_shared::CameraStats;
pub use test_bench_shared::{
    ExportSettings, ExportStatus, FsmMoveRequest, FsmStatus, StarDetectionSettings,
    TrackingEnableRequest, TrackingSettings, TrackingState, TrackingStatus,
};

/// Maximum history entries for sparkline plots (at ~10Hz polling = ~10 seconds)
const HISTORY_MAX: usize = 100;

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
#[derive(Clone, PartialEq)]
pub struct TrackingHistory {
    points: RingBuffer<TrackingPoint>,
}

impl Default for TrackingHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackingHistory {
    fn new() -> Self {
        Self {
            points: RingBuffer::new(HISTORY_MAX),
        }
    }

    fn push(&mut self, x: f64, y: f64, snr: f64, timestamp_sec: u64, timestamp_nanos: u64) {
        let t = timestamp_sec as f64 + (timestamp_nanos as f64) / 1_000_000_000.0;
        self.points.push(TrackingPoint { x, y, snr, t });
    }

    pub fn points(&self) -> &std::collections::VecDeque<TrackingPoint> {
        self.points.as_deque()
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
    image_url: String,
    stats: Option<CameraStats>,
    show_annotation: bool,
    log_scale_histogram: bool,
    stats_refresh_handle: Option<gloo_timers::callback::Interval>,
    tracking_refresh_handle: Option<gloo_timers::callback::Interval>,
    stats_failure_count: u32,
    // Zoom state
    zoom_center: Option<(u32, u32)>,
    zoom_auto_update: bool,
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
    // Last known frame dimensions (for MJPEG stream restart on size change)
    last_frame_size: Option<(u32, u32)>,
}

pub enum Msg {
    RefreshStats,
    ToggleAnnotation,
    ToggleLogScale,
    StatsLoaded(CameraStats),
    StatsError,
    ResetStatsInterval,
    // Zoom messages
    ImageClicked(i32, i32),
    ClearZoom,
    ToggleZoomAutoUpdate,
    // Tracking messages
    RefreshTracking,
    TrackingStatusLoaded(TrackingStatus),
    TrackingNotAvailable,
    ToggleTracking,
    TrackingToggleComplete(TrackingStatus),
    TrackingToggleFailed,
    // Tracking settings messages
    TrackingSettingsLoaded(TrackingSettings),
    ToggleTrackingSettings,
    UpdateTrackingSetting(String, f64),
    SaveTrackingSettings,
    TrackingSettingsSaved(TrackingSettings),
    TrackingSettingsSaveFailed,
    // Export settings messages
    ExportStatusLoaded(ExportStatus),
    ToggleExportSettings,
    UpdateExportSetting(String, String),
    ToggleExportBool(String),
    SaveExportSettings,
    ExportSettingsSaved(ExportSettings),
    ExportSettingsSaveFailed,
    // FSM messages
    FsmStatusLoaded(FsmStatus),
    FsmNotAvailable,
    UpdateFsmTarget(String, f64),
    MoveFsm,
    FsmMoveComplete(FsmStatus),
    FsmMoveFailed,
    // Star detection messages
    StarDetectionSettingsLoaded(StarDetectionSettings),
    ToggleStarDetectionSettings,
    UpdateStarDetectionSetting(String, String),
    SaveStarDetectionSettings,
    StarDetectionSettingsSaved(StarDetectionSettings),
    StarDetectionSettingsSaveFailed,
    // SVG overlay message
    OverlaySvgLoaded(String),
    // MJPEG stream error (server disconnected due to frame size change)
    StreamError,
}

impl Component for FgsFrontend {
    type Message = Msg;
    type Properties = FgsFrontendProps;

    fn create(ctx: &Context<Self>) -> Self {
        let stats_link = ctx.link().clone();
        let stats_handle = gloo_timers::callback::Interval::new(1000, move || {
            stats_link.send_message(Msg::RefreshStats);
        });

        let tracking_link = ctx.link().clone();
        let tracking_handle = gloo_timers::callback::Interval::new(500, move || {
            tracking_link.send_message(Msg::RefreshTracking);
        });

        Self {
            image_url: "/mjpeg".to_string(),
            stats: None,
            show_annotation: false,
            log_scale_histogram: false,
            stats_refresh_handle: Some(stats_handle),
            tracking_refresh_handle: Some(tracking_handle),
            stats_failure_count: 0,
            zoom_center: None,
            zoom_auto_update: true,
            tracking_available: false,
            tracking_status: None,
            tracking_toggle_pending: false,
            tracking_settings: None,
            tracking_settings_pending: false,
            show_tracking_settings: false,
            tracking_history: TrackingHistory::new(),
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
            last_frame_size: None,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::RefreshStats => {
                let link = ctx.link().clone();
                let client = FgsServerClient::for_web();
                wasm_bindgen_futures::spawn_local(async move {
                    match client.get_camera_stats().await {
                        Ok(stats) => link.send_message(Msg::StatsLoaded(stats)),
                        Err(_) => link.send_message(Msg::StatsError),
                    }
                });
                false
            }
            Msg::StatsLoaded(stats) => {
                // Check for frame size changes (requires MJPEG stream restart)
                let new_size = (stats.width, stats.height);
                let size_changed = self
                    .last_frame_size
                    .map(|old| old != new_size)
                    .unwrap_or(false);

                if size_changed {
                    // Restart MJPEG stream by adding cache-busting timestamp
                    self.image_url = format!("/mjpeg?t={}", js_sys::Date::now());
                }
                self.last_frame_size = Some(new_size);

                self.stats = Some(stats);
                self.stats_failure_count = 0;
                ctx.link().send_message(Msg::ResetStatsInterval);
                true
            }
            Msg::ToggleAnnotation => {
                self.show_annotation = !self.show_annotation;
                true
            }
            Msg::ToggleLogScale => {
                self.log_scale_histogram = !self.log_scale_histogram;
                true
            }
            Msg::StatsError => {
                self.stats_failure_count += 1;
                ctx.link().send_message(Msg::ResetStatsInterval);
                false
            }
            Msg::ResetStatsInterval => {
                let delay = calculate_backoff_delay(self.stats_failure_count, 1000, 30000);
                let link = ctx.link().clone();
                self.stats_refresh_handle = None;
                self.stats_refresh_handle =
                    Some(gloo_timers::callback::Interval::new(delay, move || {
                        link.send_message(Msg::RefreshStats);
                    }));
                false
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
            Msg::ToggleZoomAutoUpdate => {
                self.zoom_auto_update = !self.zoom_auto_update;
                true
            }
            Msg::RefreshTracking => {
                let link = ctx.link().clone();
                let client = FgsServerClient::for_web();
                wasm_bindgen_futures::spawn_local(async move {
                    match client.get_tracking_status().await {
                        Ok(status) => link.send_message(Msg::TrackingStatusLoaded(status)),
                        Err(FgsError::ServerError { status: 404, .. }) => {
                            link.send_message(Msg::TrackingNotAvailable);
                        }
                        Err(_) => {}
                    }
                });

                // Also fetch settings if we don't have them yet
                if self.tracking_settings.is_none() {
                    let link2 = ctx.link().clone();
                    let client = FgsServerClient::for_web();
                    wasm_bindgen_futures::spawn_local(async move {
                        if let Ok(settings) = client.get_tracking_settings().await {
                            link2.send_message(Msg::TrackingSettingsLoaded(settings));
                        }
                    });
                }

                // Also fetch export status
                let link3 = ctx.link().clone();
                let client = FgsServerClient::for_web();
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(status) = client.get_export_status().await {
                        link3.send_message(Msg::ExportStatusLoaded(status));
                    }
                });

                // Also fetch FSM status
                let link4 = ctx.link().clone();
                let client = FgsServerClient::for_web();
                wasm_bindgen_futures::spawn_local(async move {
                    match client.get_fsm_status().await {
                        Ok(status) => link4.send_message(Msg::FsmStatusLoaded(status)),
                        Err(FgsError::ServerError { status: 404, .. }) => {
                            link4.send_message(Msg::FsmNotAvailable);
                        }
                        Err(_) => {}
                    }
                });

                // Fetch star detection settings if we don't have them yet
                if self.star_detection_settings.is_none() {
                    let link5 = ctx.link().clone();
                    let client = FgsServerClient::for_web();
                    wasm_bindgen_futures::spawn_local(async move {
                        if let Ok(settings) = client.get_star_detection_settings().await {
                            link5.send_message(Msg::StarDetectionSettingsLoaded(settings));
                        }
                    });
                }
                false
            }
            Msg::TrackingStatusLoaded(status) => {
                self.tracking_available = true;

                // Check if we just left the Tracking state - if so, clear history
                let was_tracking = matches!(
                    self.tracking_status.as_ref().map(|s| &s.state),
                    Some(TrackingState::Tracking { .. })
                );
                let is_tracking = matches!(&status.state, TrackingState::Tracking { .. });

                if was_tracking && !is_tracking {
                    self.tracking_history.clear();
                }

                // Update position and SNR history if we have a position
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
            Msg::TrackingNotAvailable => {
                self.tracking_available = false;
                self.tracking_status = None;
                false
            }
            Msg::ToggleTracking => {
                if self.tracking_toggle_pending {
                    return false;
                }
                self.tracking_toggle_pending = true;

                let new_enabled = !self
                    .tracking_status
                    .as_ref()
                    .map(|s| s.enabled)
                    .unwrap_or(false);

                let link = ctx.link().clone();
                let client = FgsServerClient::for_web();
                wasm_bindgen_futures::spawn_local(async move {
                    match client.set_tracking_enabled(new_enabled).await {
                        Ok(status) => link.send_message(Msg::TrackingToggleComplete(status)),
                        Err(_) => link.send_message(Msg::TrackingToggleFailed),
                    }
                });
                true
            }
            Msg::TrackingToggleComplete(status) => {
                self.tracking_toggle_pending = false;
                self.tracking_status = Some(status);
                true
            }
            Msg::TrackingToggleFailed => {
                self.tracking_toggle_pending = false;
                true
            }
            Msg::TrackingSettingsLoaded(settings) => {
                self.tracking_settings = Some(settings);
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
                    let link = ctx.link().clone();
                    let client = FgsServerClient::for_web();
                    wasm_bindgen_futures::spawn_local(async move {
                        match client.set_tracking_settings(&settings).await {
                            Ok(()) => link.send_message(Msg::TrackingSettingsSaved(settings)),
                            Err(_) => link.send_message(Msg::TrackingSettingsSaveFailed),
                        }
                    });
                }
                true
            }
            Msg::TrackingSettingsSaved(settings) => {
                self.tracking_settings_pending = false;
                self.tracking_settings = Some(settings);
                true
            }
            Msg::TrackingSettingsSaveFailed => {
                self.tracking_settings_pending = false;
                true
            }
            Msg::ExportStatusLoaded(status) => {
                self.export_status = Some(status);
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
                    let settings = status.settings.clone();
                    let link = ctx.link().clone();
                    let client = FgsServerClient::for_web();
                    wasm_bindgen_futures::spawn_local(async move {
                        match client.set_export_settings(&settings).await {
                            Ok(()) => link.send_message(Msg::ExportSettingsSaved(settings)),
                            Err(_) => link.send_message(Msg::ExportSettingsSaveFailed),
                        }
                    });
                }
                true
            }
            Msg::ExportSettingsSaved(settings) => {
                self.export_settings_pending = false;
                if let Some(ref mut status) = self.export_status {
                    status.settings = settings;
                }
                true
            }
            Msg::ExportSettingsSaveFailed => {
                self.export_settings_pending = false;
                true
            }
            Msg::FsmStatusLoaded(status) => {
                // Initialize target positions if this is first time getting status
                if self.fsm_status.is_none() {
                    self.fsm_target_x = status.x_urad;
                    self.fsm_target_y = status.y_urad;
                }
                self.fsm_status = Some(status);
                true
            }
            Msg::FsmNotAvailable => {
                self.fsm_status = None;
                false
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

                let x = self.fsm_target_x;
                let y = self.fsm_target_y;

                let link = ctx.link().clone();
                let client = FgsServerClient::for_web();
                wasm_bindgen_futures::spawn_local(async move {
                    match client.move_fsm(x, y).await {
                        Ok(()) => {
                            // Fetch updated status after move
                            if let Ok(status) = client.get_fsm_status().await {
                                link.send_message(Msg::FsmMoveComplete(status));
                            } else {
                                link.send_message(Msg::FsmMoveFailed);
                            }
                        }
                        Err(_) => link.send_message(Msg::FsmMoveFailed),
                    }
                });
                true
            }
            Msg::FsmMoveComplete(status) => {
                self.fsm_move_pending = false;
                self.fsm_status = Some(status);
                true
            }
            Msg::FsmMoveFailed => {
                self.fsm_move_pending = false;
                true
            }
            // Star detection settings messages
            Msg::StarDetectionSettingsLoaded(settings) => {
                self.star_detection_settings = Some(settings);
                true
            }
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
                    let link = ctx.link().clone();
                    let client = FgsServerClient::for_web();
                    wasm_bindgen_futures::spawn_local(async move {
                        match client.set_star_detection_settings(&settings).await {
                            Ok(saved) => link.send_message(Msg::StarDetectionSettingsSaved(saved)),
                            Err(_) => link.send_message(Msg::StarDetectionSettingsSaveFailed),
                        }
                    });
                }
                true
            }
            Msg::StarDetectionSettingsSaved(settings) => {
                self.star_detection_pending = false;
                // Clear overlay if detection was disabled
                if !settings.enabled {
                    self.overlay_svg = None;
                }
                self.star_detection_settings = Some(settings);
                true
            }
            Msg::StarDetectionSettingsSaveFailed => {
                self.star_detection_pending = false;
                true
            }
            Msg::OverlaySvgLoaded(svg) => {
                self.overlay_svg = Some(svg);
                true
            }
            Msg::StreamError => {
                // Server disconnected the MJPEG stream (likely due to frame size change)
                // Restart the stream with a cache-busting timestamp
                self.image_url = format!("/mjpeg?t={}", js_sys::Date::now());
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let props = ctx.props();

        let onclick = ctx.link().callback(|e: MouseEvent| {
            let target = e.target().unwrap();
            let element = target.dyn_ref::<web_sys::Element>().unwrap();
            let rect = element.get_bounding_client_rect();
            let x = e.client_x() - rect.left() as i32;
            let y = e.client_y() - rect.top() as i32;
            Msg::ImageClicked(x, y)
        });

        let ontouchstart = ctx.link().callback(|e: TouchEvent| {
            e.prevent_default();
            if let Some(touch) = e.touches().get(0) {
                let target = e.target().unwrap();
                let element = target.dyn_ref::<web_sys::Element>().unwrap();
                let rect = element.get_bounding_client_rect();
                let x = touch.client_x() - rect.left() as i32;
                let y = touch.client_y() - rect.top() as i32;
                return Msg::ImageClicked(x, y);
            }
            Msg::ImageClicked(0, 0)
        });

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

                    <h2 style="margin-top: 30px;">{"Display Options"}</h2>
                    <div class="metadata-item">
                        <label style="cursor: pointer;">
                            <input
                                type="checkbox"
                                checked={self.show_annotation}
                                onchange={ctx.link().callback(|_| Msg::ToggleAnnotation)}
                                style="width: 20px; height: 20px; vertical-align: middle;"
                            />
                            <span style="margin-left: 5px;">{"Show Analysis"}</span>
                        </label>
                    </div>

                    <TrackingView
                        available={self.tracking_available}
                        status={self.tracking_status.clone()}
                        toggle_pending={self.tracking_toggle_pending}
                        on_toggle_tracking={ctx.link().callback(|_| Msg::ToggleTracking)}
                        history={self.tracking_history.clone()}
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
                        <img
                            id="camera-frame"
                            class="image-frame"
                            src={self.image_url.clone()}
                            alt="Camera Frame"
                            onclick={onclick}
                            ontouchstart={ontouchstart}
                            onerror={ctx.link().callback(|_| Msg::StreamError)}
                            style="cursor: crosshair; touch-action: pinch-zoom; display: block;"
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
                        auto_update={self.zoom_auto_update}
                        on_clear={ctx.link().callback(|_| Msg::ClearZoom)}
                        on_toggle_auto={ctx.link().callback(|_| Msg::ToggleZoomAutoUpdate)}
                        use_annotated={self.star_detection_settings.as_ref().map(|s| s.enabled).unwrap_or(false)}
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
                </div>
            </>
        }
    }

    fn destroy(&mut self, _ctx: &Context<Self>) {
        self.stats_refresh_handle = None;
        self.tracking_refresh_handle = None;
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

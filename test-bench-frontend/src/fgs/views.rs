use std::collections::VecDeque;

use test_bench_shared::{
    CameraStats, ExportStatus, TrackingSettings, TrackingState, TrackingStatus,
};
use yew::prelude::*;

use crate::fgs_app::TrackingHistory;

use super::components::{
    ApplyButton, Checkbox, ErrorMessage, SettingsButton, SettingsPanel, Slider, SmallCheckbox,
    StatusCount, TextInput,
};

/// Configuration for a sparkline plot.
struct SparklineConfig {
    label: &'static str,
    color: &'static str,
    reference_line: Option<(f64, &'static str)>,
}

/// Render a sparkline SVG for the given data points.
fn render_sparkline(data: &VecDeque<f64>, config: SparklineConfig) -> Html {
    if data.is_empty() {
        return html! {};
    }

    let width = 200.0;
    let height = 40.0;
    let padding = 2.0;

    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(0.001);

    let points: Vec<String> = data
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let x = padding + (i as f64 / data.len().max(1) as f64) * (width - 2.0 * padding);
            let y = height - padding - ((val - min_val) / range) * (height - 2.0 * padding);
            format!("{x:.1},{y:.1}")
        })
        .collect();

    let path_d = if points.len() > 1 {
        format!("M {} L {}", points[0], points[1..].join(" L "))
    } else {
        format!("M {} L {}", points[0], points[0])
    };

    let reference_line_html = if let Some((ref_val, ref_color)) = config.reference_line {
        if max_val >= ref_val && min_val <= ref_val {
            let y = height - padding - ((ref_val - min_val) / range) * (height - 2.0 * padding);
            html! {
                <line
                    x1={padding.to_string()}
                    y1={y.to_string()}
                    x2={(width - padding).to_string()}
                    y2={y.to_string()}
                    stroke={ref_color}
                    stroke-width="1"
                    stroke-dasharray="3,3"
                />
            }
        } else {
            html! {}
        }
    } else {
        html! {}
    };

    html! {
        <div class="metadata-item" style="margin-top: 5px;">
            <span class="metadata-label">{config.label}</span>
            <svg width={width.to_string()} height={height.to_string()} style="background: #111; border: 1px solid #333;">
                { reference_line_html }
                <path d={path_d} fill="none" stroke={config.color} stroke-width="1.5"/>
            </svg>
            <div style="font-size: 0.7em; color: #666;">
                {format!("Range: {:.2} - {:.2}", min_val, max_val)}
            </div>
        </div>
    }
}

/// Props for stats view.
#[derive(Properties, PartialEq)]
pub struct StatsViewProps {
    pub stats: Option<CameraStats>,
}

/// Render the statistics panel.
#[function_component(StatsView)]
pub fn stats_view(props: &StatsViewProps) -> Html {
    if let Some(ref stats) = props.stats {
        let mut temps: Vec<_> = stats.temperatures.iter().collect();
        temps.sort_by_key(|(k, _)| *k);
        let temp_items: Html = temps
            .into_iter()
            .map(|(location, temp)| {
                let display_name = location
                    .chars()
                    .next()
                    .map(|c| c.to_uppercase().to_string())
                    .unwrap_or_default()
                    + &location[1..];
                html! {
                    <div style="padding-left: 10px;">{format!("{}: {:.1}°C", display_name, temp)}</div>
                }
            })
            .collect();

        html! {
            <div class="stats-placeholder">
                <div>{format!("FPS: {:.1}", stats.avg_fps)}</div>
                <div>{format!("Frames: {}", stats.total_frames)}</div>
                <div style="margin-top: 10px;"><strong>{"Temperatures"}</strong></div>
                { temp_items }
            </div>
        }
    } else {
        html! {
            <div class="stats-placeholder">
                <div>{"FPS: Calculating..."}</div>
                <div>{"Frames: 0"}</div>
                <div>{"Temperature: --°C"}</div>
            </div>
        }
    }
}

/// Props for zoom view.
#[derive(Properties, PartialEq)]
pub struct ZoomViewProps {
    pub zoom_center: Option<(u32, u32)>,
    pub auto_update: bool,
    pub on_clear: Callback<()>,
    pub on_toggle_auto: Callback<()>,
}

/// Render the zoom region panel.
#[function_component(ZoomView)]
pub fn zoom_view(props: &ZoomViewProps) -> Html {
    let zoom_size = 128;
    let zoom_url = if let Some((x, y)) = props.zoom_center {
        if props.auto_update {
            format!(
                "/zoom?x={}&y={}&size={}&t={}",
                x,
                y,
                zoom_size,
                js_sys::Date::now()
            )
        } else {
            format!("/zoom?x={x}&y={y}&size={zoom_size}")
        }
    } else {
        String::new()
    };

    let on_clear = props.on_clear.clone();
    let on_toggle_auto = props.on_toggle_auto.clone();

    html! {
        <div id="zoom-container">
            if let Some((x, y)) = props.zoom_center {
                <img
                    id="zoom-canvas"
                    src={zoom_url}
                    alt="Zoomed Region"
                    style="width: 100%; border: 1px solid #00ff00; background: #111; image-rendering: pixelated;"
                />
                <div id="zoom-info" style="font-size: 0.7em; color: #00aa00; margin-top: 5px;">
                    {format!("Center: ({}, {})", x, y)}
                </div>
                <div id="zoom-controls" style="margin-top: 10px;">
                    <button
                        onclick={Callback::from(move |_| on_clear.emit(()))}
                        style="background: #111; color: #00ff00; border: 1px solid #00ff00; padding: 5px 10px; cursor: pointer; font-family: 'Courier New', monospace;"
                    >
                        {"Clear Zoom"}
                    </button>
                    <label style="font-size: 0.8em; margin-left: 10px; cursor: pointer;">
                        <input
                            type="checkbox"
                            checked={props.auto_update}
                            onchange={Callback::from(move |_| on_toggle_auto.emit(()))}
                        />
                        {" Auto-update"}
                    </label>
                </div>
            } else {
                <div style="width: 100%; height: 150px; border: 1px solid #333; background: #111; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #666;">{"Click on image to zoom"}</span>
                </div>
            }
        </div>
    }
}

/// Props for tracking settings view.
#[derive(Properties, PartialEq)]
pub struct TrackingSettingsViewProps {
    pub show: bool,
    pub settings: Option<TrackingSettings>,
    pub pending: bool,
    pub on_toggle: Callback<()>,
    pub on_update: Callback<(String, f64)>,
    pub on_save: Callback<()>,
}

/// Render the tracking settings panel.
#[function_component(TrackingSettingsView)]
pub fn tracking_settings_view(props: &TrackingSettingsViewProps) -> Html {
    if !props.show {
        return html! {
            <SettingsButton
                icon="⚙"
                label="Settings"
                expanded={false}
                onclick={props.on_toggle.clone()}
            />
        };
    }

    let settings = match &props.settings {
        Some(s) => s.clone(),
        None => return html! { <div>{"Loading settings..."}</div> },
    };

    let make_slider_callback = |field: &'static str, on_update: Callback<(String, f64)>| {
        Callback::from(move |val: f64| {
            on_update.emit((field.to_string(), val));
        })
    };

    html! {
        <>
            <SettingsButton
                icon="⚙"
                label="Settings"
                expanded={true}
                onclick={props.on_toggle.clone()}
            />
            <SettingsPanel>
                <Slider
                    label="Acq. Frames"
                    value={settings.acquisition_frames as f64}
                    min={1.0}
                    max={20.0}
                    step={1.0}
                    decimals={0}
                    onchange={make_slider_callback("acquisition_frames", props.on_update.clone())}
                />
                <Slider
                    label="ROI Size (px)"
                    value={settings.roi_size as f64}
                    min={32.0}
                    max={256.0}
                    step={8.0}
                    decimals={0}
                    onchange={make_slider_callback("roi_size", props.on_update.clone())}
                />
                <Slider
                    label="Detection σ"
                    value={settings.detection_threshold_sigma}
                    min={2.0}
                    max={10.0}
                    step={0.5}
                    decimals={1}
                    onchange={make_slider_callback("detection_threshold_sigma", props.on_update.clone())}
                />
                <Slider
                    label="SNR Min"
                    value={settings.snr_min}
                    min={3.0}
                    max={500.0}
                    step={5.0}
                    decimals={0}
                    onchange={make_slider_callback("snr_min", props.on_update.clone())}
                />
                <Slider
                    label="SNR Dropout"
                    value={settings.snr_dropout_threshold}
                    min={1.0}
                    max={200.0}
                    step={5.0}
                    decimals={0}
                    onchange={make_slider_callback("snr_dropout_threshold", props.on_update.clone())}
                />
                <Slider
                    label="FWHM (px)"
                    value={settings.fwhm}
                    min={1.0}
                    max={20.0}
                    step={0.5}
                    decimals={1}
                    onchange={make_slider_callback("fwhm", props.on_update.clone())}
                />
                <ApplyButton pending={props.pending} onclick={props.on_save.clone()} />
            </SettingsPanel>
        </>
    }
}

/// Props for export settings view.
#[derive(Properties, PartialEq)]
pub struct ExportSettingsViewProps {
    pub show: bool,
    pub status: Option<ExportStatus>,
    pub pending: bool,
    pub on_toggle: Callback<()>,
    pub on_update_string: Callback<(String, String)>,
    pub on_toggle_bool: Callback<String>,
    pub on_save: Callback<()>,
}

/// Render the export settings panel.
#[function_component(ExportSettingsView)]
pub fn export_settings_view(props: &ExportSettingsViewProps) -> Html {
    if !props.show {
        return html! {
            <SettingsButton
                icon="💾"
                label="Export"
                expanded={false}
                onclick={props.on_toggle.clone()}
            />
        };
    }

    let status = match &props.status {
        Some(s) => s.clone(),
        None => return html! { <div>{"Loading export settings..."}</div> },
    };

    let csv_toggle = {
        let on_toggle = props.on_toggle_bool.clone();
        Callback::from(move |_| on_toggle.emit("csv_enabled".to_string()))
    };

    let frames_toggle = {
        let on_toggle = props.on_toggle_bool.clone();
        Callback::from(move |_| on_toggle.emit("frames_enabled".to_string()))
    };

    let csv_filename_change = {
        let on_update = props.on_update_string.clone();
        Callback::from(move |val: String| on_update.emit(("csv_filename".to_string(), val)))
    };

    let frames_dir_change = {
        let on_update = props.on_update_string.clone();
        Callback::from(move |val: String| on_update.emit(("frames_directory".to_string(), val)))
    };

    html! {
        <>
            <SettingsButton
                icon="💾"
                label="Export"
                expanded={true}
                onclick={props.on_toggle.clone()}
            />
            <SettingsPanel>
                <SmallCheckbox
                    label="CSV Export"
                    checked={status.settings.csv_enabled}
                    onchange={csv_toggle}
                />
                <div class="metadata-item" style="margin-top: 3px;">
                    <TextInput
                        value={status.settings.csv_filename.clone()}
                        placeholder="tracking_data.csv"
                        onchange={csv_filename_change}
                    />
                </div>
                <StatusCount count={status.csv_records_written} label="records written" />

                <div style="margin-top: 10px;">
                    <SmallCheckbox
                        label="Frame Export"
                        checked={status.settings.frames_enabled}
                        onchange={frames_toggle}
                    />
                </div>
                <div class="metadata-item" style="margin-top: 3px;">
                    <TextInput
                        value={status.settings.frames_directory.clone()}
                        placeholder="frames"
                        onchange={frames_dir_change}
                    />
                </div>
                <StatusCount count={status.frames_exported} label="frames exported" />

                <ErrorMessage message={status.last_error.clone()} />
                <ApplyButton pending={props.pending} onclick={props.on_save.clone()} />
            </SettingsPanel>
        </>
    }
}

/// Props for main tracking view.
#[derive(Properties, PartialEq)]
pub struct TrackingViewProps {
    pub available: bool,
    pub status: Option<TrackingStatus>,
    pub toggle_pending: bool,
    pub on_toggle_tracking: Callback<()>,
    /// Position and SNR history for plotting
    pub history: TrackingHistory,
    // Tracking settings
    pub show_settings: bool,
    pub settings: Option<TrackingSettings>,
    pub settings_pending: bool,
    pub on_toggle_settings: Callback<()>,
    pub on_update_setting: Callback<(String, f64)>,
    pub on_save_settings: Callback<()>,
    // Export settings
    pub show_export: bool,
    pub export_status: Option<ExportStatus>,
    pub export_pending: bool,
    pub on_toggle_export: Callback<()>,
    pub on_update_export_string: Callback<(String, String)>,
    pub on_toggle_export_bool: Callback<String>,
    pub on_save_export: Callback<()>,
}

/// Render the main tracking panel with all sub-panels.
#[function_component(TrackingView)]
pub fn tracking_view(props: &TrackingViewProps) -> Html {
    if !props.available {
        return html! {};
    }

    let status = props.status.as_ref();
    let enabled = status.map(|s| s.enabled).unwrap_or(false);

    // Build common parts
    let tracking_checkbox = html! {
        <div class="metadata-item">
            <Checkbox
                label="Enable Tracking"
                checked={enabled}
                disabled={props.toggle_pending}
                onchange={props.on_toggle_tracking.clone()}
            />
        </div>
    };

    let settings_view = html! {
        <TrackingSettingsView
            show={props.show_settings}
            settings={props.settings.clone()}
            pending={props.settings_pending}
            on_toggle={props.on_toggle_settings.clone()}
            on_update={props.on_update_setting.clone()}
            on_save={props.on_save_settings.clone()}
        />
    };

    let export_view = html! {
        <ExportSettingsView
            show={props.show_export}
            status={props.export_status.clone()}
            pending={props.export_pending}
            on_toggle={props.on_toggle_export.clone()}
            on_update_string={props.on_update_export_string.clone()}
            on_toggle_bool={props.on_toggle_export_bool.clone()}
            on_save={props.on_save_export.clone()}
        />
    };

    match status.map(|s| &s.state) {
        Some(TrackingState::Acquiring { frames_collected }) => {
            html! {
                <>
                    <h2 style="margin-top: 30px;">{"Tracking"}</h2>
                    { tracking_checkbox }
                    <div class="metadata-item" style="margin-top: 10px;">
                        <span style="color: #ffaa00;">{format!("Acquiring... ({} frames)", frames_collected)}</span>
                    </div>
                    { settings_view }
                    { export_view }
                </>
            }
        }
        Some(TrackingState::Tracking { frames_processed }) => {
            let position_text = status
                .and_then(|s| s.position.as_ref())
                .map(|p| format!("({:.2}, {:.2})", p.x, p.y))
                .unwrap_or_else(|| "N/A".to_string());

            let snr_text = status
                .and_then(|s| s.position.as_ref())
                .map(|p| format!("{:.1}", p.snr))
                .unwrap_or_else(|| "N/A".to_string());

            // Render sparkline plots using extracted function
            let x_plot = render_sparkline(
                props.history.x_slice(),
                SparklineConfig {
                    label: "X Position:",
                    color: "#00aaff",
                    reference_line: None,
                },
            );

            let y_plot = render_sparkline(
                props.history.y_slice(),
                SparklineConfig {
                    label: "Y Position:",
                    color: "#ffaa00",
                    reference_line: None,
                },
            );

            let snr_plot = render_sparkline(
                props.history.snr_slice(),
                SparklineConfig {
                    label: "SNR:",
                    color: "#00ff00",
                    reference_line: Some((3.0, "#ff0000")),
                },
            );

            html! {
                <>
                    <h2 style="margin-top: 30px;">{"Tracking"}</h2>
                    { tracking_checkbox }
                    <div class="metadata-item" style="margin-top: 10px;">
                        <span style="color: #00ff00;">{"TRACKING"}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">{"Position:"}</span>
                        <span style="color: #00ff00;">{position_text}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">{"SNR:"}</span>
                        <span>{snr_text}</span>
                    </div>
                    { x_plot }
                    { y_plot }
                    { snr_plot }
                    <div class="metadata-item">
                        <span class="metadata-label">{"Frames:"}</span>
                        <span>{frames_processed.to_string()}</span>
                    </div>
                    { settings_view }
                    { export_view }
                </>
            }
        }
        Some(TrackingState::Reacquiring { attempts }) => {
            html! {
                <>
                    <h2 style="margin-top: 30px;">{"Tracking"}</h2>
                    { tracking_checkbox }
                    <div class="metadata-item" style="margin-top: 10px;">
                        <span style="color: #ff0000;">{format!("Reacquiring... (attempt {})", attempts)}</span>
                    </div>
                    { settings_view }
                    { export_view }
                </>
            }
        }
        _ => {
            let state_text = match status.map(|s| &s.state) {
                Some(TrackingState::Idle) => "Idle",
                Some(TrackingState::Calibrating) => "Calibrating",
                _ => "Unknown",
            };

            html! {
                <>
                    <h2 style="margin-top: 30px;">{"Tracking"}</h2>
                    { tracking_checkbox }
                    <div class="metadata-item" style="margin-top: 10px;">
                        <span class="metadata-label">{"State:"}</span>
                        <span>{state_text}</span>
                    </div>
                    { settings_view }
                    { export_view }
                </>
            }
        }
    }
}

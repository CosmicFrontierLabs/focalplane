use shared_wasm::{
    CameraStats, ExportStatus, FsmStatus, StarDetectionSettings, TrackingSettings, TrackingState,
    TrackingStatus,
};
use yew::prelude::*;

use crate::fgs_app::TrackingHistory;

use super::components::{
    ApplyButton, Checkbox, ErrorMessage, Select, SelectOption, SettingsButton, SettingsPanel,
    Slider, SmallCheckbox, StatusCount, TextInput,
};

use crate::fgs_app::TrackingPoint;

/// Configuration for a single sparkline within the aligned panel.
struct SparklineRow {
    label: &'static str,
    color: &'static str,
    /// Function to extract value from TrackingPoint
    extractor: fn(&TrackingPoint) -> f64,
    reference_line: Option<(f64, &'static str)>,
}

/// Render aligned sparklines for X, Y, and SNR with shared time axis.
/// All three plots share the same width and time scale for easy correlation.
/// Uses actual timestamps for accurate X positioning when there are gaps from tab backgrounding.
/// Labels and values are rendered as HTML elements to avoid SVG clipping issues.
fn render_aligned_sparklines(history: &TrackingHistory) -> Html {
    let points = history.points();

    if points.is_empty() {
        return html! {};
    }

    // Calculate time range from actual timestamps
    let time_min = points.front().map(|p| p.t).unwrap_or(0.0);
    let time_max = points.back().map(|p| p.t).unwrap_or(0.0);
    let time_range = (time_max - time_min).max(0.001);
    let time_span_secs = time_range;

    let rows = [
        SparklineRow {
            label: "X",
            color: "#00aaff",
            extractor: |p| p.x,
            reference_line: None,
        },
        SparklineRow {
            label: "Y",
            color: "#ffaa00",
            extractor: |p| p.y,
            reference_line: None,
        },
        SparklineRow {
            label: "SNR",
            color: "#44ddaa",
            extractor: |p| p.snr,
            reference_line: Some((3.0, "#ff0000")),
        },
    ];

    // Layout constants
    let plot_width = 200.0;
    let row_height = 48.0;
    let padding = 2.0;
    let time_axis_height = 16.0;
    let total_svg_height = (rows.len() as f64) * row_height + time_axis_height;

    // Build HTML rows with label, SVG plot, and value
    let html_rows: Html = rows
        .iter()
        .enumerate()
        .map(|(row_idx, row)| {
            let y_offset = row_idx as f64 * row_height;

            // Extract values using the row's extractor function
            let values: Vec<f64> = points.iter().map(row.extractor).collect();

            // Calculate min/max for this row's data
            let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (max_val - min_val).max(0.001);

            // Build path for this row using actual timestamps for X positioning
            let path_points: Vec<String> = points
                .iter()
                .zip(values.iter())
                .map(|(pt, &val)| {
                    // Use actual timestamp for X position (handles gaps from backgrounded tabs)
                    let t_normalized = (pt.t - time_min) / time_range;
                    let x = padding + t_normalized * (plot_width - 2.0 * padding);
                    let y = y_offset + padding + (row_height - 2.0 * padding)
                        - ((val - min_val) / range) * (row_height - 2.0 * padding);
                    format!("{x:.1},{y:.1}")
                })
                .collect();

            let path_d = if path_points.len() > 1 {
                format!("M {} L {}", path_points[0], path_points[1..].join(" L "))
            } else if !path_points.is_empty() {
                format!("M {} L {}", path_points[0], path_points[0])
            } else {
                String::new()
            };

            // Reference line (for SNR threshold)
            let ref_line = if let Some((ref_val, ref_color)) = row.reference_line {
                if max_val >= ref_val && min_val <= ref_val {
                    let y = y_offset + padding + (row_height - 2.0 * padding)
                        - ((ref_val - min_val) / range) * (row_height - 2.0 * padding);
                    html! {
                        <line
                            x1={padding.to_string()}
                            y1={y.to_string()}
                            x2={(plot_width - padding).to_string()}
                            y2={y.to_string()}
                            stroke={ref_color}
                            stroke-width="1"
                            stroke-dasharray="2,2"
                        />
                    }
                } else {
                    html! {}
                }
            } else {
                html! {}
            };

            // Row background and border in SVG
            let bg_rect = html! {
                <rect
                    x="0"
                    y={y_offset.to_string()}
                    width={plot_width.to_string()}
                    height={row_height.to_string()}
                    fill="#111"
                    stroke="#333"
                    stroke-width="0.5"
                />
            };

            // Vertical scale annotations (min at bottom, max at top), 3 decimal places
            let scale_x = plot_width - 3.0;
            let max_text = format!("{max_val:.3}");
            let min_text = format!("{min_val:.3}");
            let scale_elems = html! {
                <>
                    <text
                        x={scale_x.to_string()}
                        y={(y_offset + 10.0).to_string()}
                        fill="#00ff00"
                        font-size="9"
                        font-family="monospace"
                        text-anchor="end"
                    >
                        {max_text}
                    </text>
                    <text
                        x={scale_x.to_string()}
                        y={(y_offset + row_height - 3.0).to_string()}
                        fill="#00ff00"
                        font-size="9"
                        font-family="monospace"
                        text-anchor="end"
                    >
                        {min_text}
                    </text>
                </>
            };

            // Return just the SVG elements for this row (background, line, path, scale)
            html! {
                <>
                    { bg_rect }
                    { ref_line }
                    <path d={path_d} fill="none" stroke={row.color} stroke-width="1.5"/>
                    { scale_elems }
                </>
            }
        })
        .collect();

    // Time axis at bottom of SVG
    let time_y = (rows.len() as f64) * row_height + 12.0;
    let time_axis = html! {
        <>
            <line
                x1="0"
                y1={(time_y - 8.0).to_string()}
                x2={plot_width.to_string()}
                y2={(time_y - 8.0).to_string()}
                stroke="#444"
                stroke-width="1"
            />
            <line
                x1="0"
                y1={(time_y - 10.0).to_string()}
                x2="0"
                y2={(time_y - 6.0).to_string()}
                stroke="#444"
                stroke-width="1"
            />
            <line
                x1={plot_width.to_string()}
                y1={(time_y - 10.0).to_string()}
                x2={plot_width.to_string()}
                y2={(time_y - 6.0).to_string()}
                stroke="#444"
                stroke-width="1"
            />
            <text
                x="0"
                y={time_y.to_string()}
                fill="#00ff00"
                font-size="9"
                font-family="monospace"
            >
                {format!("-{:.0}s", time_span_secs)}
            </text>
            <text
                x={plot_width.to_string()}
                y={time_y.to_string()}
                fill="#00ff00"
                font-size="9"
                font-family="monospace"
                text-anchor="end"
            >
                {"now"}
            </text>
        </>
    };

    // viewBox dimensions for SVG coordinate system
    let view_box = format!("0 0 {plot_width} {total_svg_height}");

    html! {
        <div style="margin-top: 10px;">
            <div style="display: flex; align-items: stretch;">
                // Left column: labels + values combined
                <div style="display: flex; flex-direction: column; flex-shrink: 0; padding-right: 12px;">
                    {for rows.iter().map(|row| {
                        let values: Vec<f64> = points.iter().map(row.extractor).collect();
                        let current_val = values.last().copied().unwrap_or(0.0);
                        // Format: "SNR 1234.567" - label right-padded to 3 chars, value right-aligned
                        let combined_text = format!("{:<3} {:>10.3}", row.label, current_val);
                        html! {
                            <div style={format!("color: {}; font-weight: bold; font-family: monospace; font-size: 32px; height: 48px; display: flex; align-items: center; white-space: pre;", row.color)}>
                                {combined_text}
                            </div>
                        }
                    })}
                    <div style="height: 16px;"></div>
                </div>
                // Right: SVG sparklines (stretches to fill available width)
                <svg
                    viewBox={view_box}
                    preserveAspectRatio="none"
                    style="display: block; flex: 1; min-width: 100px;"
                    height={total_svg_height.to_string()}
                >
                    { html_rows }
                    { time_axis }
                </svg>
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
    #[prop_or_default]
    pub use_annotated: bool,
}

/// Render the zoom region panel.
#[function_component(ZoomView)]
pub fn zoom_view(props: &ZoomViewProps) -> Html {
    let zoom_size = 128;
    let endpoint = if props.use_annotated {
        "/zoom-annotated"
    } else {
        "/zoom"
    };
    let zoom_url = if let Some((x, y)) = props.zoom_center {
        if props.auto_update {
            format!(
                "{}?x={}&y={}&size={}&t={}",
                endpoint,
                x,
                y,
                zoom_size,
                js_sys::Date::now()
            )
        } else {
            format!("{endpoint}?x={x}&y={y}&size={zoom_size}")
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

/// Render the tracking settings panel (always visible).
#[function_component(TrackingSettingsView)]
pub fn tracking_settings_view(props: &TrackingSettingsViewProps) -> Html {
    let settings = match &props.settings {
        Some(s) => s.clone(),
        None => return html! { <div>{"Loading settings..."}</div> },
    };

    let make_slider_callback = |field: &'static str, on_update: Callback<(String, f64)>| {
        Callback::from(move |val: f64| {
            on_update.emit((field.to_string(), val));
        })
    };

    let make_select_callback = |field: &'static str, on_update: Callback<(String, f64)>| {
        Callback::from(move |val: usize| {
            on_update.emit((field.to_string(), val as f64));
        })
    };

    // NSV455 camera supported ROI sizes (square regions)
    let roi_options = vec![
        SelectOption {
            value: 128,
            label: "128×128".to_string(),
        },
        SelectOption {
            value: 256,
            label: "256×256".to_string(),
        },
        SelectOption {
            value: 512,
            label: "512×512".to_string(),
        },
        SelectOption {
            value: 1024,
            label: "1024×1024".to_string(),
        },
        SelectOption {
            value: 2048,
            label: "2048×2048".to_string(),
        },
        SelectOption {
            value: 4096,
            label: "4096×4096".to_string(),
        },
        SelectOption {
            value: 8096,
            label: "8096×6324 (Full)".to_string(),
        },
    ];

    let toggle_text = if props.show {
        "▼ Settings"
    } else {
        "▶ Settings"
    };

    html! {
        <>
            <div
                style="margin-top: 15px; color: #00ff00; font-size: 0.9em; cursor: pointer;"
                onclick={props.on_toggle.reform(|_| ())}
            >
                {toggle_text}
            </div>
            {if props.show {
                html! {
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
                        <Select
                            label="ROI Size"
                            value={settings.roi_size}
                            options={roi_options}
                            onchange={make_select_callback("roi_size", props.on_update.clone())}
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
                }
            } else {
                html! {}
            }}
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
    // Star detection overlay
    pub show_overlay: bool,
    pub on_toggle_overlay: Callback<()>,
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

    let overlay_checkbox = html! {
        <div class="metadata-item">
            <Checkbox
                label="Star Detection Overlay"
                checked={props.show_overlay}
                disabled={false}
                onchange={props.on_toggle_overlay.clone()}
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
                    { overlay_checkbox }
                    <div class="metadata-item" style="margin-top: 10px;">
                        <span style="color: #ffaa00;">{format!("Acquiring... ({} frames)", frames_collected)}</span>
                    </div>
                    { settings_view }
                    { export_view }
                </>
            }
        }
        Some(TrackingState::Tracking { frames_processed }) => {
            // Render aligned sparkline plots (values shown inline)
            let sparklines = render_aligned_sparklines(&props.history);

            html! {
                <>
                    <h2 style="margin-top: 30px;">{"Tracking"}</h2>
                    { tracking_checkbox }
                    { overlay_checkbox }
                    <div class="metadata-item" style="margin-top: 10px;">
                        <span style="color: #00ff00;">{"TRACKING"}</span>
                        <span style="margin-left: 10px; color: #666;">{format!("({} frames)", frames_processed)}</span>
                    </div>
                    { sparklines }
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
                    { overlay_checkbox }
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
                    { overlay_checkbox }
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

/// Props for FSM control view.
#[derive(Properties, PartialEq)]
pub struct FsmViewProps {
    pub status: Option<FsmStatus>,
    pub target_x: f64,
    pub target_y: f64,
    pub move_pending: bool,
    pub on_update_target: Callback<(String, f64)>,
    pub on_move: Callback<()>,
}

/// Render the FSM control panel with X/Y sliders.
#[function_component(FsmView)]
pub fn fsm_view(props: &FsmViewProps) -> Html {
    let status = match &props.status {
        Some(s) => s,
        None => return html! {}, // FSM not available
    };

    if !status.connected {
        return html! {
            <>
                <h2 style="margin-top: 30px;">{"FSM Control"}</h2>
                <div class="metadata-item" style="color: #ff0000;">
                    {"FSM disconnected"}
                </div>
            </>
        };
    }

    let x_callback = {
        let on_update = props.on_update_target.clone();
        Callback::from(move |val: f64| on_update.emit(("x".to_string(), val)))
    };

    let y_callback = {
        let on_update = props.on_update_target.clone();
        Callback::from(move |val: f64| on_update.emit(("y".to_string(), val)))
    };

    let on_move = props.on_move.clone();
    let move_handler = Callback::from(move |_: MouseEvent| on_move.emit(()));

    html! {
        <>
            <h2 style="margin-top: 30px;">{"FSM Control"}</h2>
            <div class="metadata-item">
                <span class="metadata-label">{"Current Position:"}</span><br/>
                <span style="color: #00ff00;">
                    {format!("X: {:.1} µrad, Y: {:.1} µrad", status.x_urad, status.y_urad)}
                </span>
            </div>
            <div style="border: 1px solid #333; padding: 10px; margin-top: 10px; background: #0a0a0a;">
                <div class="metadata-item" style="margin-top: 5px;">
                    <span style="font-size: 0.8em;">
                        {format!("X Target: {:.1} µrad", props.target_x)}
                    </span><br/>
                    <input
                        type="range"
                        min={status.x_min.to_string()}
                        max={status.x_max.to_string()}
                        step="1"
                        value={props.target_x.to_string()}
                        onchange={Callback::from({
                            let cb = x_callback.clone();
                            move |e: Event| {
                                let input: web_sys::HtmlInputElement = e.target_unchecked_into();
                                if let Ok(val) = input.value().parse::<f64>() {
                                    cb.emit(val);
                                }
                            }
                        })}
                        style="width: 100%; accent-color: #00aaff;"
                    />
                    <div style="font-size: 0.7em; color: #666;">
                        {format!("Range: {:.0} - {:.0} µrad", status.x_min, status.x_max)}
                    </div>
                </div>
                <div class="metadata-item" style="margin-top: 10px;">
                    <span style="font-size: 0.8em;">
                        {format!("Y Target: {:.1} µrad", props.target_y)}
                    </span><br/>
                    <input
                        type="range"
                        min={status.y_min.to_string()}
                        max={status.y_max.to_string()}
                        step="1"
                        value={props.target_y.to_string()}
                        onchange={Callback::from({
                            let cb = y_callback.clone();
                            move |e: Event| {
                                let input: web_sys::HtmlInputElement = e.target_unchecked_into();
                                if let Ok(val) = input.value().parse::<f64>() {
                                    cb.emit(val);
                                }
                            }
                        })}
                        style="width: 100%; accent-color: #ffaa00;"
                    />
                    <div style="font-size: 0.7em; color: #666;">
                        {format!("Range: {:.0} - {:.0} µrad", status.y_min, status.y_max)}
                    </div>
                </div>
                <div style="margin-top: 15px; text-align: center;">
                    <button
                        onclick={move_handler}
                        disabled={props.move_pending}
                        style="background: #00ff00; color: #000; border: none; padding: 8px 20px; cursor: pointer; font-family: 'Courier New', monospace; font-weight: bold;"
                    >
                        { if props.move_pending { "Moving..." } else { "Move FSM" } }
                    </button>
                </div>
            </div>
            if let Some(ref err) = status.last_error {
                <div style="font-size: 0.7em; color: #ff0000; margin-top: 5px;">
                    {format!("Error: {}", err)}
                </div>
            }
        </>
    }
}

// ==================== Star Detection Settings ====================

/// Props for star detection settings view.
#[derive(Properties, PartialEq)]
pub struct StarDetectionSettingsViewProps {
    pub show: bool,
    pub settings: Option<StarDetectionSettings>,
    pub pending: bool,
    pub on_toggle: Callback<()>,
    pub on_update: Callback<(String, String)>,
    pub on_save: Callback<()>,
}

/// Render the star detection settings panel.
#[function_component(StarDetectionSettingsView)]
pub fn star_detection_settings_view(props: &StarDetectionSettingsViewProps) -> Html {
    let settings = match &props.settings {
        Some(s) => s.clone(),
        None => return html! { <div>{"Loading detection settings..."}</div> },
    };

    let make_slider_callback = |field: &'static str, on_update: Callback<(String, String)>| {
        Callback::from(move |val: f64| {
            on_update.emit((field.to_string(), val.to_string()));
        })
    };

    let make_checkbox_callback = |field: &'static str, on_update: Callback<(String, String)>| {
        Callback::from(move |_: ()| {
            on_update.emit((field.to_string(), "toggle".to_string()));
        })
    };

    html! {
        <>
            <SettingsButton
                icon="⭐"
                label="Detection"
                expanded={props.show}
                onclick={props.on_toggle.clone()}
            />
            if props.show {
                <SettingsPanel>
                    <Checkbox
                        label="Enable Detection"
                        checked={settings.enabled}
                        onchange={make_checkbox_callback("enabled", props.on_update.clone())}
                    />
                    <Slider
                        label="Detection σ"
                        value={settings.detection_sigma}
                        min={3.0}
                        max={20.0}
                        step={0.5}
                        decimals={1}
                        onchange={make_slider_callback("detection_sigma", props.on_update.clone())}
                    />
                    <Slider
                        label="Bad Pixel Dist"
                        value={settings.min_bad_pixel_distance}
                        min={0.0}
                        max={20.0}
                        step={1.0}
                        decimals={0}
                        onchange={make_slider_callback("min_bad_pixel_distance", props.on_update.clone())}
                    />
                    <Slider
                        label="Max Aspect"
                        value={settings.max_aspect_ratio}
                        min={1.0}
                        max={5.0}
                        step={0.1}
                        decimals={1}
                        onchange={make_slider_callback("max_aspect_ratio", props.on_update.clone())}
                    />
                    <Slider
                        label="Min Flux"
                        value={settings.min_flux}
                        min={10.0}
                        max={10000.0}
                        step={10.0}
                        decimals={0}
                        onchange={make_slider_callback("min_flux", props.on_update.clone())}
                    />
                    <ApplyButton
                        pending={props.pending}
                        onclick={props.on_save.clone()}
                    />
                </SettingsPanel>
            }
        </>
    }
}

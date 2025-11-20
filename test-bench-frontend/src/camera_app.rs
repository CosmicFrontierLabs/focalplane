use gloo_net::http::Request;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};
use yew::prelude::*;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Stats {
    pub total_frames: u64,
    pub avg_fps: f32,
    pub temperatures: HashMap<String, f64>,
    pub histogram: Vec<u32>,
    pub histogram_mean: f64,
    pub histogram_max: u16,
}

#[derive(Properties, PartialEq)]
pub struct CameraFrontendProps {
    pub device: String,
    pub width: u32,
    pub height: u32,
}

pub struct CameraFrontend {
    image_url: String,
    stats: Option<Stats>,
    show_annotation: bool,
    log_scale_histogram: bool,
    connection_status: String,
    image_refresh_handle: Option<gloo_timers::callback::Interval>,
    stats_refresh_handle: Option<gloo_timers::callback::Interval>,
    image_failure_count: u32,
    stats_failure_count: u32,
    is_loading_image: bool,
}

pub enum Msg {
    RefreshImage,
    ImageLoaded(String),
    RefreshStats,
    ToggleAnnotation,
    ToggleLogScale,
    StatsLoaded(Stats),
    ImageError,
    StatsError,
    ResetImageInterval,
    ResetStatsInterval,
}

impl Component for CameraFrontend {
    type Message = Msg;
    type Properties = CameraFrontendProps;

    fn create(ctx: &Context<Self>) -> Self {
        let image_link = ctx.link().clone();
        let image_handle = gloo_timers::callback::Interval::new(100, move || {
            image_link.send_message(Msg::RefreshImage);
        });

        let stats_link = ctx.link().clone();
        let stats_handle = gloo_timers::callback::Interval::new(1000, move || {
            stats_link.send_message(Msg::RefreshStats);
        });

        Self {
            image_url: format!("/jpeg?t={}", js_sys::Date::now()),
            stats: None,
            show_annotation: false,
            log_scale_histogram: false,
            connection_status: "Connecting...".to_string(),
            image_refresh_handle: Some(image_handle),
            stats_refresh_handle: Some(stats_handle),
            image_failure_count: 0,
            stats_failure_count: 0,
            is_loading_image: false,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::RefreshImage => {
                if self.is_loading_image {
                    return false;
                }
                self.is_loading_image = true;
                let link = ctx.link().clone();
                let url = format!("/jpeg?t={}", js_sys::Date::now());
                wasm_bindgen_futures::spawn_local(async move {
                    match gloo_net::http::Request::get(&url).send().await {
                        Ok(response) if response.ok() => {
                            link.send_message(Msg::ImageLoaded(url));
                        }
                        _ => {
                            link.send_message(Msg::ImageError);
                        }
                    }
                });
                false
            }
            Msg::ImageLoaded(url) => {
                self.is_loading_image = false;
                self.image_url = url;
                self.connection_status = "Connected".to_string();
                self.image_failure_count = 0;
                ctx.link().send_message(Msg::ResetImageInterval);
                true
            }
            Msg::RefreshStats => {
                let link = ctx.link().clone();
                wasm_bindgen_futures::spawn_local(async move {
                    match Request::get("/stats").send().await {
                        Ok(response) => {
                            if let Ok(stats) = response.json::<Stats>().await {
                                link.send_message(Msg::StatsLoaded(stats));
                            } else {
                                link.send_message(Msg::StatsError);
                            }
                        }
                        Err(_) => {
                            link.send_message(Msg::StatsError);
                        }
                    }
                });
                false
            }
            Msg::StatsLoaded(stats) => {
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
            Msg::ImageError => {
                self.is_loading_image = false;
                self.connection_status = "Connection Error".to_string();
                self.image_failure_count += 1;
                ctx.link().send_message(Msg::ResetImageInterval);
                true
            }
            Msg::StatsError => {
                self.stats_failure_count += 1;
                ctx.link().send_message(Msg::ResetStatsInterval);
                false
            }
            Msg::ResetImageInterval => {
                let delay = Self::calculate_backoff_delay(self.image_failure_count, 100, 10000);
                let link = ctx.link().clone();
                self.image_refresh_handle = None;
                self.image_refresh_handle =
                    Some(gloo_timers::callback::Interval::new(delay, move || {
                        link.send_message(Msg::RefreshImage);
                    }));
                false
            }
            Msg::ResetStatsInterval => {
                let delay = Self::calculate_backoff_delay(self.stats_failure_count, 1000, 30000);
                let link = ctx.link().clone();
                self.stats_refresh_handle = None;
                self.stats_refresh_handle =
                    Some(gloo_timers::callback::Interval::new(delay, move || {
                        link.send_message(Msg::RefreshStats);
                    }));
                false
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let props = ctx.props();

        html! {
            <>
                <div class="column left-panel">
                    <h2>{"Camera Info"}</h2>
                    <div class="metadata-item">
                        <span class="metadata-label">{"Status:"}</span><br/>
                        <span class={if self.connection_status == "Connected" { "" } else { "error" }}>
                            {&self.connection_status}
                        </span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">{"Device:"}</span><br/>
                        {&props.device}
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">{"Resolution:"}</span><br/>
                        {format!("{}x{}", props.width, props.height)}
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
                    <div class="image-container">
                        <img
                            class="image-frame"
                            src={self.image_url.clone()}
                            alt="Camera Frame"
                        />
                    </div>
                </div>

                <div class="column right-panel">
                    <h2>{"Statistics"}</h2>
                    { self.view_stats() }

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
        self.image_refresh_handle = None;
        self.stats_refresh_handle = None;
    }

    fn rendered(&mut self, _ctx: &Context<Self>, _first_render: bool) {
        if let Some(ref stats) = self.stats {
            self.render_histogram(
                &stats.histogram,
                stats.histogram_mean,
                stats.histogram_max,
                self.log_scale_histogram,
            );
        }
    }
}

impl CameraFrontend {
    fn calculate_backoff_delay(failure_count: u32, base_delay: u32, max_delay: u32) -> u32 {
        if failure_count == 0 {
            base_delay
        } else {
            let exponential_delay = base_delay * 2_u32.pow(failure_count.min(10));
            exponential_delay.min(max_delay)
        }
    }

    fn render_histogram(&self, histogram: &[u32], mean: f64, max_bin: u16, log_scale: bool) {
        let window = match web_sys::window() {
            Some(w) => w,
            None => return,
        };
        let document = match window.document() {
            Some(d) => d,
            None => return,
        };

        let canvas = match document.get_element_by_id("histogram-canvas") {
            Some(el) => match el.dyn_into::<HtmlCanvasElement>() {
                Ok(c) => c,
                Err(_) => return,
            },
            None => return,
        };

        let ctx = match canvas.get_context("2d") {
            Ok(Some(ctx)) => match ctx.dyn_into::<CanvasRenderingContext2d>() {
                Ok(c) => c,
                Err(_) => return,
            },
            _ => return,
        };

        let width = canvas.width() as f64;
        let height = canvas.height() as f64;

        ctx.set_fill_style_str("#000000");
        ctx.fill_rect(0.0, 0.0, width, height);

        if histogram.is_empty() {
            return;
        }

        let transform_value = |v: f64| -> f64 {
            if log_scale {
                if v > 0.0 {
                    (v + 1.0).ln()
                } else {
                    0.0
                }
            } else {
                v
            }
        };

        let max_value = histogram
            .iter()
            .map(|&v| transform_value(v as f64))
            .fold(0.0_f64, f64::max);

        if max_value == 0.0 {
            return;
        }

        let num_bins = histogram.len();
        let bar_width = width / num_bins as f64;

        ctx.set_fill_style_str("#00aa00");

        for (i, &count) in histogram.iter().enumerate() {
            let transformed = transform_value(count as f64);
            let bar_height = (transformed / max_value) * height;
            let x = i as f64 * bar_width;
            let y = height - bar_height;
            ctx.fill_rect(x, y, bar_width.max(1.0), bar_height);
        }

        let scale_label = if log_scale { " (log)" } else { "" };
        if let Some(info_el) = document.get_element_by_id("histogram-info") {
            info_el.set_inner_html(&format!(
                "Mean: {mean:.1} | Max bin: {max_bin}{scale_label}"
            ));
        }
    }

    fn view_stats(&self) -> Html {
        if let Some(ref stats) = self.stats {
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
}

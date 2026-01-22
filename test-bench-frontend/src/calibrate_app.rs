use gloo_net::http::Request;
use std::collections::HashMap;
use web_sys::HtmlInputElement;
use yew::prelude::*;

// Re-export shared types
pub use test_bench_shared::{
    ControlSpec, DisplayInfo, PatternConfigResponse, PatternSpec, SchemaResponse,
};

// ============================================================================
// Local helper types and extensions for shared types
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum ControlValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
}

impl ControlValue {
    fn to_json(&self) -> serde_json::Value {
        match self {
            ControlValue::Int(v) => serde_json::json!(v),
            ControlValue::Float(v) => serde_json::json!(v),
            ControlValue::Bool(v) => serde_json::json!(v),
            ControlValue::Text(v) => serde_json::json!(v),
        }
    }
}

/// Extension trait for ControlSpec to add frontend-specific helper methods.
trait ControlSpecExt {
    fn id(&self) -> &str;
    fn default_value(&self) -> ControlValue;
}

impl ControlSpecExt for ControlSpec {
    fn id(&self) -> &str {
        match self {
            ControlSpec::IntRange { id, .. } => id,
            ControlSpec::FloatRange { id, .. } => id,
            ControlSpec::Bool { id, .. } => id,
            ControlSpec::Text { id, .. } => id,
        }
    }

    fn default_value(&self) -> ControlValue {
        match self {
            ControlSpec::IntRange { default, .. } => ControlValue::Int(*default),
            ControlSpec::FloatRange { default, .. } => ControlValue::Float(*default),
            ControlSpec::Bool { default, .. } => ControlValue::Bool(*default),
            ControlSpec::Text { default, .. } => ControlValue::Text(default.clone()),
        }
    }
}

// ============================================================================
// Component
// ============================================================================

#[derive(Properties, PartialEq)]
pub struct CalibrateFrontendProps {
    pub width: u32,
    pub height: u32,
}

pub struct CalibrateFrontend {
    schema: Option<SchemaResponse>,
    display_info: Option<DisplayInfo>,
    initial_config: Option<PatternConfigResponse>,
    selected_pattern_id: String,
    control_values: HashMap<String, ControlValue>,
    global_control_values: HashMap<String, ControlValue>,
    invert: bool,
    image_url: String,
    image_refresh_handle: Option<gloo_timers::callback::Timeout>,
    config_debounce_handle: Option<gloo_timers::callback::Timeout>,
    config_poll_handle: Option<gloo_timers::callback::Interval>,
    image_failure_count: u32,
    loading_schema: bool,
    loading_config: bool,
}

pub enum Msg {
    SchemaLoaded(SchemaResponse),
    SchemaError,
    DisplayInfoLoaded(DisplayInfo),
    DisplayInfoError,
    InitialConfigLoaded(PatternConfigResponse),
    InitialConfigError,
    SelectPattern(String),
    UpdateControl(String, ControlValue),
    UpdateGlobalControl(String, ControlValue),
    ToggleInvert,
    ApplyPattern,
    RefreshImage,
    ImageLoaded,
    ImageError,
    ScheduleNextRefresh,
    PollConfig,
    ConfigPolled(PatternConfigResponse),
}

impl Component for CalibrateFrontend {
    type Message = Msg;
    type Properties = CalibrateFrontendProps;

    fn create(ctx: &Context<Self>) -> Self {
        // Trigger first image load immediately
        ctx.link().send_message(Msg::RefreshImage);

        // Fetch schema on load
        let link = ctx.link().clone();
        wasm_bindgen_futures::spawn_local(async move {
            match Request::get("/schema").send().await {
                Ok(response) => {
                    if let Ok(schema) = response.json::<SchemaResponse>().await {
                        link.send_message(Msg::SchemaLoaded(schema));
                    } else {
                        link.send_message(Msg::SchemaError);
                    }
                }
                Err(_) => {
                    link.send_message(Msg::SchemaError);
                }
            }
        });

        // Fetch current config on load (don't overwrite server state)
        let link = ctx.link().clone();
        wasm_bindgen_futures::spawn_local(async move {
            match Request::get("/config").send().await {
                Ok(response) => {
                    if let Ok(config) = response.json::<PatternConfigResponse>().await {
                        link.send_message(Msg::InitialConfigLoaded(config));
                    } else {
                        link.send_message(Msg::InitialConfigError);
                    }
                }
                Err(_) => {
                    link.send_message(Msg::InitialConfigError);
                }
            }
        });

        // Fetch display info on load
        let link = ctx.link().clone();
        wasm_bindgen_futures::spawn_local(async move {
            match Request::get("/info").send().await {
                Ok(response) => {
                    if let Ok(info) = response.json::<DisplayInfo>().await {
                        link.send_message(Msg::DisplayInfoLoaded(info));
                    } else {
                        link.send_message(Msg::DisplayInfoError);
                    }
                }
                Err(_) => {
                    link.send_message(Msg::DisplayInfoError);
                }
            }
        });

        // Poll config every second to sync with server state (e.g., timeout blanking)
        let link = ctx.link().clone();
        let config_poll_handle = gloo_timers::callback::Interval::new(1000, move || {
            link.send_message(Msg::PollConfig);
        });

        Self {
            schema: None,
            display_info: None,
            initial_config: None,
            selected_pattern_id: "April".to_string(),
            control_values: HashMap::new(),
            global_control_values: HashMap::new(),
            invert: false,
            image_url: format!("/jpeg?t={}", js_sys::Date::now()),
            image_refresh_handle: None,
            config_debounce_handle: None,
            config_poll_handle: Some(config_poll_handle),
            image_failure_count: 0,
            loading_schema: true,
            loading_config: true,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SchemaLoaded(schema) => {
                self.schema = Some(schema);
                self.loading_schema = false;
                // If we already have initial config from server, use it to init UI
                if let Some(config) = &self.initial_config {
                    self.init_from_server_config(config.clone());
                }
                true
            }
            Msg::SchemaError => {
                self.loading_schema = false;
                true
            }
            Msg::InitialConfigLoaded(config) => {
                self.loading_config = false;
                self.initial_config = Some(config.clone());
                // If schema already loaded, use config to init UI
                if self.schema.is_some() {
                    self.init_from_server_config(config);
                }
                true
            }
            Msg::InitialConfigError => {
                self.loading_config = false;
                // If config fetch failed and schema is loaded, fall back to defaults
                if self.schema.is_some() {
                    self.init_pattern_defaults();
                }
                true
            }
            Msg::DisplayInfoLoaded(info) => {
                self.display_info = Some(info);
                true
            }
            Msg::DisplayInfoError => true,
            Msg::SelectPattern(pattern_id) => {
                self.selected_pattern_id = pattern_id;
                self.init_pattern_defaults();
                ctx.link().send_message(Msg::ApplyPattern);
                true
            }
            Msg::UpdateControl(id, value) => {
                self.control_values.insert(id, value);
                // Debounce: cancel existing timeout and start new one
                self.config_debounce_handle = None;
                let link = ctx.link().clone();
                self.config_debounce_handle =
                    Some(gloo_timers::callback::Timeout::new(150, move || {
                        link.send_message(Msg::ApplyPattern);
                    }));
                true
            }
            Msg::UpdateGlobalControl(id, value) => {
                self.global_control_values.insert(id, value);
                // Immediate apply for global controls (checkboxes)
                ctx.link().send_message(Msg::ApplyPattern);
                true
            }
            Msg::ToggleInvert => {
                self.invert = !self.invert;
                // Immediate apply for checkboxes (no dragging)
                ctx.link().send_message(Msg::ApplyPattern);
                true
            }
            Msg::ApplyPattern => {
                let pattern_id = self.selected_pattern_id.clone();
                let invert = self.invert;

                // Build values map
                let mut values = serde_json::Map::new();
                for (k, v) in &self.control_values {
                    values.insert(k.clone(), v.to_json());
                }

                // Extract global control values
                let emit_gyro = self
                    .global_control_values
                    .get("emit_gyro")
                    .and_then(|v| match v {
                        ControlValue::Bool(b) => Some(*b),
                        _ => None,
                    });
                let plate_scale =
                    self.global_control_values
                        .get("plate_scale")
                        .and_then(|v| match v {
                            ControlValue::Float(f) => Some(*f),
                            _ => None,
                        });

                wasm_bindgen_futures::spawn_local(async move {
                    let mut body = serde_json::json!({
                        "pattern_id": pattern_id,
                        "values": values,
                        "invert": invert,
                    });

                    // Add optional global controls
                    if let Some(emit) = emit_gyro {
                        body["emit_gyro"] = serde_json::json!(emit);
                    }
                    if let Some(scale) = plate_scale {
                        body["plate_scale"] = serde_json::json!(scale);
                    }

                    let _ = Request::post("/config").json(&body).unwrap().send().await;
                });
                true
            }
            Msg::RefreshImage => {
                // Just update URL - img element's onload/onerror handles success/failure
                self.image_url = format!("/jpeg?t={}", js_sys::Date::now());
                true
            }
            Msg::ImageLoaded => {
                self.image_failure_count = 0;
                ctx.link().send_message(Msg::ScheduleNextRefresh);
                false
            }
            Msg::ImageError => {
                self.image_failure_count += 1;
                ctx.link().send_message(Msg::ScheduleNextRefresh);
                false
            }
            Msg::ScheduleNextRefresh => {
                // Serial refresh: wait for delay, then trigger next image load
                let delay = Self::calculate_backoff_delay(self.image_failure_count, 500, 10000);
                let link = ctx.link().clone();
                self.image_refresh_handle = None;
                self.image_refresh_handle =
                    Some(gloo_timers::callback::Timeout::new(delay, move || {
                        link.send_message(Msg::RefreshImage);
                    }));
                false
            }
            Msg::PollConfig => {
                let link = ctx.link().clone();
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(response) = Request::get("/config").send().await {
                        if let Ok(config) = response.json::<PatternConfigResponse>().await {
                            link.send_message(Msg::ConfigPolled(config));
                        }
                    }
                });
                false
            }
            Msg::ConfigPolled(config) => {
                let mut changed = false;

                // Update pattern if changed
                if self.selected_pattern_id != config.pattern_id {
                    self.selected_pattern_id = config.pattern_id;
                    changed = true;
                }

                // Update invert if changed
                if self.invert != config.invert {
                    self.invert = config.invert;
                    changed = true;
                }

                // Update control values from server
                if let Some(values_map) = config.values.as_object() {
                    for (key, value) in values_map {
                        let new_value = match value {
                            serde_json::Value::Number(n) => n
                                .as_i64()
                                .map(ControlValue::Int)
                                .or_else(|| n.as_f64().map(ControlValue::Float)),
                            serde_json::Value::Bool(b) => Some(ControlValue::Bool(*b)),
                            _ => None,
                        };

                        if let Some(new_val) = new_value {
                            if self.control_values.get(key) != Some(&new_val) {
                                self.control_values.insert(key.clone(), new_val);
                                changed = true;
                            }
                        }
                    }
                }

                changed
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        if self.loading_schema || self.loading_config {
            return html! {
                <div class="loading">{"Loading..."}</div>
            };
        }

        let Some(schema) = &self.schema else {
            return html! {
                <div class="error">{"Failed to load schema from server"}</div>
            };
        };

        let current_pattern = schema
            .patterns
            .iter()
            .find(|p| p.id == self.selected_pattern_id);

        html! {
            <>
                <div class="column left-panel">
                    <h2>{"Pattern Selection"}</h2>
                    { self.view_pattern_selector(ctx, schema) }

                    <h2 style="margin-top: 20px;">{"Pattern Parameters"}</h2>
                    { self.view_pattern_controls(ctx, current_pattern) }

                    { self.view_global_controls(ctx, schema) }
                </div>

                <div class="column center-panel">
                    <div class="image-container">
                        <img
                            class="image-frame"
                            src={self.image_url.clone()}
                            alt="Calibration Pattern"
                            onload={ctx.link().callback(|_| Msg::ImageLoaded)}
                            onerror={ctx.link().callback(|_| Msg::ImageError)}
                        />
                    </div>
                </div>

                <div class="column right-panel">
                    <h2>{"Display Info"}</h2>
                    { self.view_display_info() }

                    <h2 style="margin-top: 20px;">{"Current Pattern"}</h2>
                    <div class="info-item">
                        <span class="status">{
                            current_pattern.map(|p| p.name.as_str()).unwrap_or("Unknown")
                        }</span>
                    </div>

                    <h2 style="margin-top: 30px;">{"Endpoints"}</h2>
                    <div class="info-item">
                        <a href="/jpeg">{"JPEG Pattern"}</a><br/>
                        <a href="/config">{"Config (JSON)"}</a><br/>
                        <a href="/schema">{"Schema (JSON)"}</a><br/>
                        <a href="/info">{"Display Info (JSON)"}</a>
                    </div>

                    <h2 style="margin-top: 30px;">{"Info"}</h2>
                    <div class="info-item" style="font-size: 0.8em; color: #00aa00;">
                        {"Controls are dynamically loaded from the server schema."}
                        <br/><br/>
                        {"Animated patterns (Static, Circling Pixel, Wiggling Gaussian) will continuously regenerate."}
                    </div>
                </div>
            </>
        }
    }

    fn destroy(&mut self, _ctx: &Context<Self>) {
        self.image_refresh_handle = None;
        self.config_poll_handle = None;
    }
}

impl CalibrateFrontend {
    fn calculate_backoff_delay(failure_count: u32, base_delay: u32, max_delay: u32) -> u32 {
        if failure_count == 0 {
            base_delay
        } else {
            let exponential_delay = base_delay * 2_u32.pow(failure_count.min(10));
            exponential_delay.min(max_delay)
        }
    }

    fn init_pattern_defaults(&mut self) {
        self.control_values.clear();

        if let Some(schema) = &self.schema {
            if let Some(pattern) = schema
                .patterns
                .iter()
                .find(|p| p.id == self.selected_pattern_id)
            {
                for control in &pattern.controls {
                    self.control_values
                        .insert(control.id().to_string(), control.default_value());
                }
            }
        }
    }

    /// Initialize UI state from server's current config (don't overwrite server state on load)
    fn init_from_server_config(&mut self, config: PatternConfigResponse) {
        self.selected_pattern_id = config.pattern_id;
        self.invert = config.invert;
        self.control_values.clear();

        // Parse control values from server response
        if let Some(values_map) = config.values.as_object() {
            for (key, value) in values_map {
                let control_value = match value {
                    serde_json::Value::Number(n) => n
                        .as_i64()
                        .map(ControlValue::Int)
                        .or_else(|| n.as_f64().map(ControlValue::Float)),
                    serde_json::Value::Bool(b) => Some(ControlValue::Bool(*b)),
                    serde_json::Value::String(s) => Some(ControlValue::Text(s.clone())),
                    _ => None,
                };

                if let Some(val) = control_value {
                    self.control_values.insert(key.clone(), val);
                }
            }
        }
    }

    fn view_pattern_selector(&self, ctx: &Context<Self>, schema: &SchemaResponse) -> Html {
        let onchange = ctx.link().callback(|e: Event| {
            let target: HtmlInputElement = e.target_unchecked_into();
            Msg::SelectPattern(target.value())
        });

        html! {
            <div class="control-group">
                <label class="control-label">{"Pattern Type:"}</label>
                <select id="pattern-type" {onchange}>
                    { for schema.patterns.iter().map(|p| {
                        let selected = p.id == self.selected_pattern_id;
                        html! {
                            <option value={p.id.clone()} {selected}>{&p.name}</option>
                        }
                    })}
                </select>
            </div>
        }
    }

    fn view_pattern_controls(&self, ctx: &Context<Self>, pattern: Option<&PatternSpec>) -> Html {
        let Some(pattern) = pattern else {
            return html! {};
        };

        if pattern.controls.is_empty() {
            return html! {
                <div class="control-group" style="color: #888;">
                    {"No parameters for this pattern"}
                </div>
            };
        }

        html! {
            <>
                { for pattern.controls.iter().map(|control| {
                    self.view_control(ctx, control)
                })}
            </>
        }
    }

    fn view_control(&self, ctx: &Context<Self>, control: &ControlSpec) -> Html {
        match control {
            ControlSpec::IntRange {
                id,
                label,
                min,
                max,
                step,
                ..
            } => {
                let current_value = self
                    .control_values
                    .get(id)
                    .and_then(|v| match v {
                        ControlValue::Int(i) => Some(*i),
                        _ => None,
                    })
                    .unwrap_or(*min);

                let id_clone = id.clone();
                let oninput = ctx.link().callback(move |e: InputEvent| {
                    let target: HtmlInputElement = e.target_unchecked_into();
                    let value = target.value().parse().unwrap_or(0);
                    Msg::UpdateControl(id_clone.clone(), ControlValue::Int(value))
                });

                html! {
                    <div class="control-group">
                        <label class="control-label">
                            {label}{": "}
                            <span class="range-value">{current_value}</span>
                        </label>
                        <input
                            type="range"
                            min={min.to_string()}
                            max={max.to_string()}
                            step={step.to_string()}
                            value={current_value.to_string()}
                            {oninput}
                        />
                    </div>
                }
            }
            ControlSpec::FloatRange {
                id,
                label,
                min,
                max,
                step,
                ..
            } => {
                let current_value = self
                    .control_values
                    .get(id)
                    .and_then(|v| match v {
                        ControlValue::Float(f) => Some(*f),
                        _ => None,
                    })
                    .unwrap_or(*min);

                let id_clone = id.clone();
                let oninput = ctx.link().callback(move |e: InputEvent| {
                    let target: HtmlInputElement = e.target_unchecked_into();
                    let value = target.value().parse().unwrap_or(0.0);
                    Msg::UpdateControl(id_clone.clone(), ControlValue::Float(value))
                });

                html! {
                    <div class="control-group">
                        <label class="control-label">
                            {label}{": "}
                            <span class="range-value">{format!("{:.1}", current_value)}</span>
                        </label>
                        <input
                            type="range"
                            min={min.to_string()}
                            max={max.to_string()}
                            step={step.to_string()}
                            value={current_value.to_string()}
                            {oninput}
                        />
                    </div>
                }
            }
            ControlSpec::Bool { id, label, .. } => {
                let current_value = self
                    .control_values
                    .get(id)
                    .and_then(|v| match v {
                        ControlValue::Bool(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(false);

                let id_clone = id.clone();
                let onchange = ctx.link().callback(move |_| {
                    Msg::UpdateControl(id_clone.clone(), ControlValue::Bool(!current_value))
                });

                html! {
                    <div class="control-group">
                        <label style="cursor: pointer;">
                            <input
                                type="checkbox"
                                checked={current_value}
                                {onchange}
                            />
                            <span style="margin-left: 5px;">{label}</span>
                        </label>
                    </div>
                }
            }
            ControlSpec::Text {
                id,
                label,
                default,
                placeholder,
            } => {
                let current_value = self
                    .control_values
                    .get(id)
                    .and_then(|v| match v {
                        ControlValue::Text(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| default.clone());

                let id_clone = id.clone();
                let oninput = ctx.link().callback(move |e: InputEvent| {
                    let target: HtmlInputElement = e.target_unchecked_into();
                    Msg::UpdateControl(id_clone.clone(), ControlValue::Text(target.value()))
                });

                html! {
                    <div class="control-group">
                        <label class="control-label">{label}{":"}</label>
                        <input
                            type="text"
                            value={current_value}
                            placeholder={placeholder.clone()}
                            {oninput}
                            style="width: 100%; padding: 5px; background: #333; color: #0f0; border: 1px solid #0f0; font-family: monospace;"
                        />
                    </div>
                }
            }
        }
    }

    fn view_display_info(&self) -> Html {
        match &self.display_info {
            Some(info) => {
                let pixel_pitch_text = match info.pixel_pitch_um {
                    Some(pitch) => format!("{pitch:.2} µm"),
                    None => "Unknown".to_string(),
                };

                html! {
                    <>
                        <div class="info-item">
                            <span class="info-label">{"Name:"}</span><br/>
                            {&info.name}
                        </div>
                        <div class="info-item">
                            <span class="info-label">{"Resolution:"}</span><br/>
                            {format!("{}x{}", info.width, info.height)}
                        </div>
                        <div class="info-item">
                            <span class="info-label">{"Pixel Pitch:"}</span><br/>
                            {pixel_pitch_text}
                        </div>
                    </>
                }
            }
            None => {
                html! {
                    <div class="info-item" style="color: #888;">
                        {"Loading..."}
                    </div>
                }
            }
        }
    }

    fn view_global_controls(&self, ctx: &Context<Self>, schema: &SchemaResponse) -> Html {
        html! {
            <>
                // Invert is handled specially (uses self.invert state)
                { schema.global_controls.iter().filter_map(|c| {
                    if let ControlSpec::Bool { id, .. } = c {
                        if id == "invert" {
                            return Some(html! {
                                <div class="control-group">
                                    <label style="cursor: pointer;">
                                        <input
                                            type="checkbox"
                                            checked={self.invert}
                                            onchange={ctx.link().callback(|_| Msg::ToggleInvert)}
                                        />
                                        <span style="margin-left: 5px;">{"Invert Colors"}</span>
                                    </label>
                                </div>
                            });
                        }
                    }
                    None
                }).collect::<Html>() }

                // Other global controls rendered dynamically
                { for schema.global_controls.iter().filter_map(|control| {
                    // Skip invert, it's handled above
                    if control.id() == "invert" {
                        return None;
                    }
                    Some(self.view_global_control(ctx, control))
                })}
            </>
        }
    }

    fn view_global_control(&self, ctx: &Context<Self>, control: &ControlSpec) -> Html {
        match control {
            ControlSpec::Bool { id, label, .. } => {
                let current_value = self
                    .global_control_values
                    .get(id)
                    .and_then(|v| match v {
                        ControlValue::Bool(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(false);

                let id_clone = id.clone();
                let onchange = ctx.link().callback(move |_| {
                    Msg::UpdateGlobalControl(id_clone.clone(), ControlValue::Bool(!current_value))
                });

                html! {
                    <div class="control-group">
                        <label style="cursor: pointer;">
                            <input
                                type="checkbox"
                                checked={current_value}
                                {onchange}
                            />
                            <span style="margin-left: 5px;">{label}</span>
                        </label>
                    </div>
                }
            }
            ControlSpec::FloatRange {
                id,
                label,
                min,
                max,
                step,
                default,
            } => {
                let current_value = self
                    .global_control_values
                    .get(id)
                    .and_then(|v| match v {
                        ControlValue::Float(f) => Some(*f),
                        _ => None,
                    })
                    .unwrap_or(*default);

                let id_clone = id.clone();
                let oninput = ctx.link().callback(move |e: InputEvent| {
                    let target: HtmlInputElement = e.target_unchecked_into();
                    let value = target.value().parse().unwrap_or(0.0);
                    Msg::UpdateGlobalControl(id_clone.clone(), ControlValue::Float(value))
                });

                html! {
                    <div class="control-group">
                        <label class="control-label">
                            {label}{": "}
                            <span class="range-value">{format!("{:.2}", current_value)}</span>
                        </label>
                        <input
                            type="range"
                            min={min.to_string()}
                            max={max.to_string()}
                            step={step.to_string()}
                            value={current_value.to_string()}
                            {oninput}
                        />
                    </div>
                }
            }
            // Int and Text global controls not expected but handle for completeness
            _ => html! {},
        }
    }
}

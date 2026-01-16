use yew::prelude::*;

/// Standard checkbox with label used throughout the FGS frontend.
#[derive(Properties, PartialEq)]
pub struct CheckboxProps {
    pub label: AttrValue,
    pub checked: bool,
    #[prop_or_default]
    pub disabled: bool,
    pub onchange: Callback<()>,
}

#[function_component(Checkbox)]
pub fn checkbox(props: &CheckboxProps) -> Html {
    let onchange = props.onchange.clone();
    let onchange_handler = Callback::from(move |_| onchange.emit(()));

    html! {
        <label style="cursor: pointer;">
            <input
                type="checkbox"
                checked={props.checked}
                disabled={props.disabled}
                onchange={onchange_handler}
                style="width: 20px; height: 20px; vertical-align: middle;"
            />
            <span style="margin-left: 5px;">{&props.label}</span>
        </label>
    }
}

/// Settings toggle button with icon and text.
#[derive(Properties, PartialEq)]
pub struct SettingsButtonProps {
    pub icon: AttrValue,
    pub label: AttrValue,
    #[prop_or_default]
    pub expanded: bool,
    pub onclick: Callback<()>,
}

#[function_component(SettingsButton)]
pub fn settings_button(props: &SettingsButtonProps) -> Html {
    let onclick = props.onclick.clone();
    let onclick_handler = Callback::from(move |_: MouseEvent| onclick.emit(()));

    let text = if props.expanded {
        format!("{} Hide {}", props.icon, props.label)
    } else {
        format!("{} {}", props.icon, props.label)
    };

    html! {
        <div class="metadata-item" style="margin-top: 10px;">
            <button
                onclick={onclick_handler}
                style="background: #111; color: #00ff00; border: 1px solid #00ff00; padding: 3px 8px; cursor: pointer; font-family: 'Courier New', monospace; font-size: 0.8em;"
            >
                {text}
            </button>
        </div>
    }
}

/// Settings panel wrapper with consistent styling.
#[derive(Properties, PartialEq)]
pub struct SettingsPanelProps {
    pub children: Children,
}

#[function_component(SettingsPanel)]
pub fn settings_panel(props: &SettingsPanelProps) -> Html {
    html! {
        <div style="border: 1px solid #333; padding: 10px; margin-top: 10px; background: #0a0a0a;">
            { for props.children.iter() }
        </div>
    }
}

/// Slider input for numeric settings.
#[derive(Properties, PartialEq)]
pub struct SliderProps {
    pub label: AttrValue,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    #[prop_or(1.0)]
    pub step: f64,
    #[prop_or(1)]
    pub decimals: usize,
    pub onchange: Callback<f64>,
}

#[function_component(Slider)]
pub fn slider(props: &SliderProps) -> Html {
    let onchange = props.onchange.clone();
    let onchange_handler = Callback::from(move |e: Event| {
        let input: web_sys::HtmlInputElement = e.target_unchecked_into();
        if let Ok(val) = input.value().parse::<f64>() {
            onchange.emit(val);
        }
    });

    let value_str = match props.decimals {
        0 => format!("{:.0}", props.value),
        1 => format!("{:.1}", props.value),
        _ => format!("{:.2}", props.value),
    };

    html! {
        <div class="metadata-item" style="margin-top: 5px;">
            <span style="font-size: 0.8em;">{format!("{}: {}", props.label, value_str)}</span><br/>
            <input
                type="range"
                min={props.min.to_string()}
                max={props.max.to_string()}
                step={props.step.to_string()}
                value={props.value.to_string()}
                onchange={onchange_handler}
                style="width: 100%; accent-color: #00ff00;"
            />
        </div>
    }
}

/// Apply button for settings forms.
#[derive(Properties, PartialEq)]
pub struct ApplyButtonProps {
    #[prop_or_default]
    pub pending: bool,
    pub onclick: Callback<()>,
}

#[function_component(ApplyButton)]
pub fn apply_button(props: &ApplyButtonProps) -> Html {
    let onclick = props.onclick.clone();
    let onclick_handler = Callback::from(move |_: MouseEvent| onclick.emit(()));

    html! {
        <div style="margin-top: 10px; text-align: center;">
            <button
                onclick={onclick_handler}
                disabled={props.pending}
                style="background: #00ff00; color: #000; border: none; padding: 5px 15px; cursor: pointer; font-family: 'Courier New', monospace;"
            >
                { if props.pending { "Saving..." } else { "Apply" } }
            </button>
        </div>
    }
}

/// Text input field for settings.
#[derive(Properties, PartialEq)]
pub struct TextInputProps {
    pub value: AttrValue,
    #[prop_or_default]
    pub placeholder: AttrValue,
    pub onchange: Callback<String>,
}

#[function_component(TextInput)]
pub fn text_input(props: &TextInputProps) -> Html {
    let onchange = props.onchange.clone();
    let onchange_handler = Callback::from(move |e: Event| {
        let input: web_sys::HtmlInputElement = e.target_unchecked_into();
        onchange.emit(input.value());
    });

    html! {
        <input
            type="text"
            value={props.value.clone()}
            onchange={onchange_handler}
            placeholder={props.placeholder.clone()}
            style="width: 100%; background: #111; color: #00ff00; border: 1px solid #333; padding: 3px; font-family: 'Courier New', monospace; font-size: 0.8em;"
        />
    }
}

/// Small checkbox for settings panels (smaller than main Checkbox).
#[derive(Properties, PartialEq)]
pub struct SmallCheckboxProps {
    pub label: AttrValue,
    pub checked: bool,
    pub onchange: Callback<()>,
}

#[function_component(SmallCheckbox)]
pub fn small_checkbox(props: &SmallCheckboxProps) -> Html {
    let onchange = props.onchange.clone();
    let onchange_handler = Callback::from(move |_| onchange.emit(()));

    html! {
        <div class="metadata-item" style="margin-top: 5px;">
            <label style="cursor: pointer; font-size: 0.8em;">
                <input
                    type="checkbox"
                    checked={props.checked}
                    onchange={onchange_handler}
                    style="width: 16px; height: 16px; vertical-align: middle;"
                />
                <span style="margin-left: 5px;">{&props.label}</span>
            </label>
        </div>
    }
}

/// Status text with optional count.
#[derive(Properties, PartialEq)]
pub struct StatusCountProps {
    pub count: u64,
    pub label: AttrValue,
}

#[function_component(StatusCount)]
pub fn status_count(props: &StatusCountProps) -> Html {
    if props.count == 0 {
        return html! {};
    }
    html! {
        <div style="font-size: 0.7em; color: #00aa00;">
            {format!("{} {}", props.count, props.label)}
        </div>
    }
}

/// Error message display.
#[derive(Properties, PartialEq)]
pub struct ErrorMessageProps {
    #[prop_or_default]
    pub message: Option<String>,
}

#[function_component(ErrorMessage)]
pub fn error_message(props: &ErrorMessageProps) -> Html {
    match &props.message {
        Some(err) => html! {
            <div style="font-size: 0.7em; color: #ff0000; margin-top: 5px;">
                {format!("Error: {}", err)}
            </div>
        },
        None => html! {},
    }
}

/// A single option for the Select component.
#[derive(Clone, PartialEq)]
pub struct SelectOption {
    pub value: usize,
    pub label: String,
}

/// Select dropdown for discrete value selection.
#[derive(Properties, PartialEq)]
pub struct SelectProps {
    pub label: AttrValue,
    pub value: usize,
    pub options: Vec<SelectOption>,
    pub onchange: Callback<usize>,
}

#[function_component(Select)]
pub fn select(props: &SelectProps) -> Html {
    let onchange = props.onchange.clone();
    let onchange_handler = Callback::from(move |e: Event| {
        let select: web_sys::HtmlSelectElement = e.target_unchecked_into();
        if let Ok(val) = select.value().parse::<usize>() {
            onchange.emit(val);
        }
    });

    let options_html: Html = props
        .options
        .iter()
        .map(|opt| {
            html! {
                <option value={opt.value.to_string()} selected={opt.value == props.value}>
                    {&opt.label}
                </option>
            }
        })
        .collect();

    html! {
        <div class="metadata-item" style="margin-top: 5px;">
            <span style="font-size: 0.8em;">{&props.label}</span><br/>
            <select
                onchange={onchange_handler}
                style="width: 100%; background: #111; color: #00ff00; border: 1px solid #333; padding: 3px; font-family: 'Courier New', monospace; font-size: 0.8em;"
            >
                { options_html }
            </select>
        </div>
    }
}

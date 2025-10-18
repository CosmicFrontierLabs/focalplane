use axum::{extract::State, http::StatusCode, response::Response, Json};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::gpio::{GpioController, ORIN_PX04_LINE};

use nsv455::hil::AppState;

// Barker-13 sequence for correlation
const BARKER_13: [bool; 13] = [
    true, true, true, true, true, false, false, true, true, false, true, false, true,
];

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum PatternType {
    Short,
    Long,
    Barker,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LatencyRequest {
    pub pattern: PatternType,
    pub gpio_chip: Option<String>,
    pub gpio_line: Option<u32>,
    pub alignment_delay_us: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyResponse {
    pub pattern_executed: String,
    pub nominal_timing_us: u64,
    pub frame_timestamp_us: u64,
    pub execution_time_us: u64,
}

fn hot_wait(duration: Duration) {
    let start = Instant::now();
    while start.elapsed() < duration {
        std::hint::spin_loop();
    }
}

pub async fn gpio_pattern_endpoint(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<LatencyRequest>,
) -> Result<Json<LatencyResponse>, Response> {
    let gpio_chip = request.gpio_chip.unwrap_or_else(|| "gpiochip0".to_string());
    let gpio_line = request.gpio_line.unwrap_or(ORIN_PX04_LINE);
    let alignment_delay_us = request.alignment_delay_us.unwrap_or(10000);

    // Initialize GPIO
    let mut gpio = match GpioController::new(&gpio_chip, gpio_line) {
        Ok(g) => g,
        Err(e) => {
            return Err(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(format!("Failed to initialize GPIO: {e}").into())
                .unwrap())
        }
    };

    // Configure as output
    if let Err(e) = gpio.request_output("hil_latency", 0) {
        return Err(Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(format!("Failed to set GPIO as output: {e}").into())
            .unwrap());
    }

    // Wait for alignment
    hot_wait(Duration::from_micros(alignment_delay_us));

    let start_time = Instant::now();
    let nominal_timing_us: u64;
    let pattern_name: String;

    // Execute the pattern
    match request.pattern {
        PatternType::Short => {
            pattern_name = "Short exponential (2 bits)".to_string();
            let on_us = 500;
            let off_us = 500;
            let bits = 2;

            for bit in 0..bits {
                let _ = gpio.set_value(1);
                hot_wait(Duration::from_micros(on_us));

                let _ = gpio.set_value(0);
                let off_duration = (1 << bit) * off_us;
                hot_wait(Duration::from_micros(off_duration));
            }

            nominal_timing_us = bits * on_us + (0..bits).map(|b| (1u64 << b) * off_us).sum::<u64>();
        }
        PatternType::Long => {
            pattern_name = "Long exponential (8 bits)".to_string();
            let on_us = 1000;
            let off_us = 1000;
            let bits = 8;

            for bit in 0..bits {
                let _ = gpio.set_value(1);
                hot_wait(Duration::from_micros(on_us));

                let _ = gpio.set_value(0);
                let off_duration = (1 << bit) * off_us;
                hot_wait(Duration::from_micros(off_duration));
            }

            nominal_timing_us = bits * on_us + (0..bits).map(|b| (1u64 << b) * off_us).sum::<u64>();
        }
        PatternType::Barker => {
            pattern_name = "Barker-13".to_string();
            let symbol_us = 1000;

            for &symbol in BARKER_13.iter() {
                let _ = gpio.set_value(if symbol { 1 } else { 0 });
                hot_wait(Duration::from_micros(symbol_us));
            }
            let _ = gpio.set_value(0);

            nominal_timing_us = symbol_us * BARKER_13.len() as u64;
        }
    }

    let execution_time_us = start_time.elapsed().as_micros() as u64;

    Ok(Json(LatencyResponse {
        pattern_executed: pattern_name,
        nominal_timing_us,
        frame_timestamp_us: 0, // Would need frame capture integration
        execution_time_us,
    }))
}

pub async fn latency_measurement_endpoint(
    State(state): State<Arc<AppState>>,
    Json(request): Json<LatencyRequest>,
) -> Result<Json<LatencyMeasurementResponse>, Response> {
    // Capture frame before pattern
    let pre_frame = {
        let mut session = state.session.lock().await;
        match session.capture_frame() {
            Ok((_, metadata)) => {
                metadata.timestamp.sec as u64 * 1_000_000 + metadata.timestamp.usec as u64
            }
            Err(e) => {
                return Err(Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(format!("Failed to capture pre-frame: {e}").into())
                    .unwrap())
            }
        }
    };

    // Execute GPIO pattern
    let pattern_result = gpio_pattern_endpoint(State(state.clone()), Json(request)).await?;

    // Capture frame after pattern with raw data
    let (post_frame, raw_frame_base64) = {
        let mut session = state.session.lock().await;
        match session.capture_frame() {
            Ok((frame_data, metadata)) => {
                let timestamp =
                    metadata.timestamp.sec as u64 * 1_000_000 + metadata.timestamp.usec as u64;
                let encoded = STANDARD.encode(&frame_data);
                (timestamp, encoded)
            }
            Err(e) => {
                return Err(Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(format!("Failed to capture post-frame: {e}").into())
                    .unwrap())
            }
        }
    };

    let response = LatencyMeasurementResponse {
        pattern_executed: pattern_result.0.pattern_executed,
        nominal_timing_us: pattern_result.0.nominal_timing_us,
        pre_frame_timestamp_us: pre_frame,
        post_frame_timestamp_us: post_frame,
        frame_delta_us: post_frame.saturating_sub(pre_frame),
        execution_time_us: pattern_result.0.execution_time_us,
        raw_frame_base64,
        frame_width: state.width,
        frame_height: state.height,
    };

    Ok(Json(response))
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyMeasurementResponse {
    pub pattern_executed: String,
    pub nominal_timing_us: u64,
    pub pre_frame_timestamp_us: u64,
    pub post_frame_timestamp_us: u64,
    pub frame_delta_us: u64,
    pub execution_time_us: u64,
    pub raw_frame_base64: String,
    pub frame_width: u32,
    pub frame_height: u32,
}

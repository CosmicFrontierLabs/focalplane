use crate::{errors::MonitoringError, hardware_monitor::*};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use prometheus::{Counter, Encoder, Gauge, GaugeVec, Registry, TextEncoder};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{error, info};

#[derive(Clone)]
pub struct PrometheusMetrics {
    registry: Registry,

    // Power metrics
    power_rail_voltage: GaugeVec,
    power_rail_current: GaugeVec,
    power_rail_power: GaugeVec,

    // Thermal metrics
    thermal_temperature: GaugeVec,

    // System metrics
    cpu_usage_percent: Gauge,
    memory_usage_mb: Gauge,
    memory_total_mb: Gauge,
    disk_usage_percent: Gauge,
    gpu_usage_percent: Gauge,
    uptime_seconds: Gauge,

    // Monitoring health
    metrics_collection_total: Counter,
    metrics_collection_errors: Counter,
    last_collection_timestamp: Gauge,
}

impl PrometheusMetrics {
    pub fn new() -> Result<Self, MonitoringError> {
        let registry = Registry::new();

        // Power rail metrics
        let power_rail_voltage = GaugeVec::new(
            prometheus::Opts::new(
                "jetson_power_rail_voltage_mv",
                "Power rail voltage in millivolts",
            ),
            &["rail_name"],
        )?;

        let power_rail_current = GaugeVec::new(
            prometheus::Opts::new(
                "jetson_power_rail_current_ma",
                "Power rail current in milliamperes",
            ),
            &["rail_name"],
        )?;

        let power_rail_power = GaugeVec::new(
            prometheus::Opts::new(
                "jetson_power_rail_power_mw",
                "Power rail power consumption in milliwatts",
            ),
            &["rail_name"],
        )?;

        // Thermal metrics
        let thermal_temperature = GaugeVec::new(
            prometheus::Opts::new(
                "jetson_thermal_temperature_celsius",
                "Thermal zone temperature in Celsius",
            ),
            &["zone_name"],
        )?;

        // System metrics
        let cpu_usage_percent = Gauge::new("jetson_cpu_usage_percent", "CPU usage percentage")?;
        let memory_usage_mb = Gauge::new("jetson_memory_usage_mb", "Memory usage in megabytes")?;
        let memory_total_mb = Gauge::new("jetson_memory_total_mb", "Total memory in megabytes")?;
        let disk_usage_percent = Gauge::new("jetson_disk_usage_percent", "Disk usage percentage")?;
        let gpu_usage_percent = Gauge::new("jetson_gpu_usage_percent", "GPU usage percentage")?;
        let uptime_seconds = Gauge::new("jetson_uptime_seconds", "System uptime in seconds")?;

        // Monitoring health metrics
        let metrics_collection_total = Counter::new(
            "jetson_metrics_collection_total",
            "Total number of metrics collections",
        )?;
        let metrics_collection_errors = Counter::new(
            "jetson_metrics_collection_errors_total",
            "Total number of metrics collection errors",
        )?;
        let last_collection_timestamp = Gauge::new(
            "jetson_last_collection_timestamp",
            "Timestamp of last successful metrics collection",
        )?;

        // Register all metrics
        registry.register(Box::new(power_rail_voltage.clone()))?;
        registry.register(Box::new(power_rail_current.clone()))?;
        registry.register(Box::new(power_rail_power.clone()))?;
        registry.register(Box::new(thermal_temperature.clone()))?;
        registry.register(Box::new(cpu_usage_percent.clone()))?;
        registry.register(Box::new(memory_usage_mb.clone()))?;
        registry.register(Box::new(memory_total_mb.clone()))?;
        registry.register(Box::new(disk_usage_percent.clone()))?;
        registry.register(Box::new(gpu_usage_percent.clone()))?;
        registry.register(Box::new(uptime_seconds.clone()))?;
        registry.register(Box::new(metrics_collection_total.clone()))?;
        registry.register(Box::new(metrics_collection_errors.clone()))?;
        registry.register(Box::new(last_collection_timestamp.clone()))?;

        Ok(Self {
            registry,
            power_rail_voltage,
            power_rail_current,
            power_rail_power,
            thermal_temperature,
            cpu_usage_percent,
            memory_usage_mb,
            memory_total_mb,
            disk_usage_percent,
            gpu_usage_percent,
            uptime_seconds,
            metrics_collection_total,
            metrics_collection_errors,
            last_collection_timestamp,
        })
    }

    pub fn update_metrics(&self, system_metrics: &SystemMetrics) -> Result<(), MonitoringError> {
        // Update power rail metrics
        for rail in &system_metrics.power_rails {
            if let Some(voltage) = rail.voltage_mv {
                self.power_rail_voltage
                    .with_label_values(&[&rail.name])
                    .set(voltage);
            }

            if let Some(current) = rail.current_ma {
                self.power_rail_current
                    .with_label_values(&[&rail.name])
                    .set(current);
            }

            if let Some(power) = rail.power_mw {
                self.power_rail_power
                    .with_label_values(&[&rail.name])
                    .set(power);
            }
        }

        // Update thermal metrics
        for thermal in &system_metrics.thermal_zones {
            self.thermal_temperature
                .with_label_values(&[&thermal.name])
                .set(thermal.temperature_celsius);
        }

        // Update system metrics
        self.cpu_usage_percent.set(system_metrics.cpu_usage_percent);
        self.memory_usage_mb.set(system_metrics.memory_usage_mb);
        self.memory_total_mb.set(system_metrics.memory_total_mb);
        self.disk_usage_percent
            .set(system_metrics.disk_usage_percent);

        if let Some(gpu_usage) = system_metrics.gpu_usage_percent {
            self.gpu_usage_percent.set(gpu_usage);
        }

        self.uptime_seconds
            .set(system_metrics.uptime_seconds as f64);

        // Update collection health metrics
        self.metrics_collection_total.inc();
        self.last_collection_timestamp
            .set(system_metrics.timestamp as f64);

        Ok(())
    }

    pub fn increment_error_counter(&self) {
        self.metrics_collection_errors.inc();
    }

    pub fn render_metrics(&self) -> Result<String, MonitoringError> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut output = Vec::new();

        encoder
            .encode(&metric_families, &mut output)
            .map_err(|e| MonitoringError::SystemError(format!("Failed to encode metrics: {e}")))?;

        String::from_utf8(output).map_err(|e| {
            MonitoringError::SystemError(format!("Failed to convert metrics to UTF-8: {e}"))
        })
    }
}

pub struct MonitoringServer {
    monitor: JetsonOrinMonitor,
    metrics: Arc<PrometheusMetrics>,
}

impl MonitoringServer {
    pub fn new() -> Result<Self, MonitoringError> {
        let monitor = JetsonOrinMonitor::new()?;
        let metrics = Arc::new(PrometheusMetrics::new()?);

        Ok(Self { monitor, metrics })
    }

    pub async fn start_metrics_collection(&self, collection_interval_seconds: u64) {
        let monitor = self.monitor.clone();
        let metrics = Arc::clone(&self.metrics);

        let mut interval = interval(Duration::from_secs(collection_interval_seconds));

        tokio::spawn(async move {
            loop {
                interval.tick().await;

                match monitor.collect_metrics().await {
                    Ok(system_metrics) => {
                        if let Err(e) = metrics.update_metrics(&system_metrics) {
                            error!("Failed to update Prometheus metrics: {}", e);
                            metrics.increment_error_counter();
                        } else {
                            info!("Successfully collected and updated metrics");
                        }
                    }
                    Err(e) => {
                        error!("Failed to collect system metrics: {}", e);
                        metrics.increment_error_counter();
                    }
                }
            }
        });
    }

    pub async fn serve(self, bind_address: &str, port: u16) -> Result<(), MonitoringError> {
        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_handler))
            .with_state(Arc::clone(&self.metrics));

        let listener = tokio::net::TcpListener::bind(format!("{bind_address}:{port}"))
            .await
            .map_err(|e| {
                MonitoringError::SystemError(format!(
                    "Failed to bind to {bind_address}:{port}: {e}"
                ))
            })?;

        info!(
            "Prometheus metrics server starting on {}:{}",
            bind_address, port
        );
        info!("Metrics endpoint: http://{}:{}/metrics", bind_address, port);
        info!("Health endpoint: http://{}:{}/health", bind_address, port);

        // Start background metrics collection
        self.start_metrics_collection(1).await; // Collect every 1 second

        axum::serve(listener, app)
            .await
            .map_err(|e| MonitoringError::SystemError(format!("Server error: {e}")))?;

        Ok(())
    }
}

async fn metrics_handler(State(metrics): State<Arc<PrometheusMetrics>>) -> Response {
    match metrics.render_metrics() {
        Ok(metrics_output) => (StatusCode::OK, metrics_output).into_response(),
        Err(e) => {
            error!("Failed to render metrics: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {e}")).into_response()
        }
    }
}

async fn health_handler() -> Response {
    let health_info = serde_json::json!({
        "status": "healthy",
        "service": "jetson-orin-monitoring",
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });

    (StatusCode::OK, health_info.to_string()).into_response()
}

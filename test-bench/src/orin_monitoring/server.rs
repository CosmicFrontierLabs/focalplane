use super::prometheus::PrometheusMetrics;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use hardware::orin::{JetsonOrinMonitor, MonitoringError};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{error, info};

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
    let health_info = test_bench_shared::HealthInfo {
        status: "healthy".to_string(),
        service: "jetson-orin-monitoring".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    let json = serde_json::to_string(&health_info).unwrap();
    (StatusCode::OK, json).into_response()
}

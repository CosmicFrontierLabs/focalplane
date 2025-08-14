use thiserror::Error;

#[derive(Error, Debug)]
pub enum MonitoringError {
    #[error("I/O error reading hardware sensors: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse sensor value: {0}")]
    ParseError(String),

    #[error("Sensor not found: {0}")]
    SensorNotFound(String),

    #[error("Prometheus metrics error: {0}")]
    PrometheusError(#[from] prometheus::Error),

    #[error("System monitoring error: {0}")]
    SystemError(String),
}

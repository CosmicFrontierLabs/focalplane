use hardware::orin::MonitoringError;
use test_bench::orin_monitoring::MonitoringServer;
use tracing::{error, info, Level};

#[tokio::main]
async fn main() -> Result<(), MonitoringError> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("Starting Jetson Orin flight computer monitoring service");

    let server = MonitoringServer::new()?;

    let bind_address = std::env::var("BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "9090".to_string())
        .parse::<u16>()
        .unwrap_or(9090);

    info!("Configuration:");
    info!("  Bind address: {}", bind_address);
    info!("  Port: {}", port);

    if let Err(e) = server.serve(&bind_address, port).await {
        error!("Server failed to start: {}", e);
        return Err(e);
    }

    Ok(())
}

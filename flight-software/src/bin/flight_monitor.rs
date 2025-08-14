use flight_software::{MonitoringError, MonitoringServer};
use tracing::{error, info, Level};

#[tokio::main]
async fn main() -> Result<(), MonitoringError> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("Starting Jetson Orin flight computer monitoring service");

    // Create and start the monitoring server
    let server = MonitoringServer::new()?;

    let bind_address = std::env::var("BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "9090".to_string())
        .parse::<u16>()
        .unwrap_or(9090);

    info!("Configuration:");
    info!("  Bind address: {}", bind_address);
    info!("  Port: {}", port);

    // Start the server (this will run indefinitely)
    if let Err(e) = server.serve(&bind_address, port).await {
        error!("Server failed to start: {}", e);
        return Err(e);
    }

    Ok(())
}

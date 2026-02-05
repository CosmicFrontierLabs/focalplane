//! WebSocket-based log streaming for real-time log viewing.
//!
//! This module provides a tracing subscriber layer that broadcasts log messages
//! to multiple WebSocket clients. Logs are sent as JSON-encoded `LogEntry` structs.
//!
//! # Usage
//!
//! 1. Create a `LogBroadcaster` and get a reference to it
//! 2. Add the `WsLogLayer` to your tracing subscriber
//! 3. Register the `/logs` WebSocket endpoint with your router

use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use shared_wasm::{LogEntry, LogLevel};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

/// Broadcaster for log entries to multiple WebSocket clients.
pub struct LogBroadcaster {
    tx: broadcast::Sender<LogEntry>,
}

impl LogBroadcaster {
    /// Create a new log broadcaster with the given buffer capacity.
    ///
    /// The capacity determines how many log entries can be buffered before
    /// slow clients start missing messages (they'll receive a Lagged error).
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Broadcast a log entry to all subscribers.
    pub fn broadcast(&self, entry: LogEntry) {
        // Ignore errors - no subscribers is fine
        let _ = self.tx.send(entry);
    }

    /// Subscribe to the log stream.
    pub fn subscribe(&self) -> broadcast::Receiver<LogEntry> {
        self.tx.subscribe()
    }
}

impl Default for LogBroadcaster {
    fn default() -> Self {
        Self::new(64)
    }
}

/// Visitor to extract the message from a tracing event.
struct MessageVisitor {
    message: String,
}

impl MessageVisitor {
    fn new() -> Self {
        Self {
            message: String::new(),
        }
    }
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // Only record if this is the message field or we don't have a message yet
        if field.name() == "message" || self.message.is_empty() {
            self.message = format!("{value:?}");
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        // Only record if this is the message field or we don't have a message yet
        if field.name() == "message" || self.message.is_empty() {
            self.message = value.to_string();
        }
    }
}

/// Tracing layer that broadcasts log events to WebSocket clients.
pub struct WsLogLayer {
    broadcaster: Arc<LogBroadcaster>,
    min_level: Level,
}

impl WsLogLayer {
    /// Create a new WebSocket log layer.
    pub fn new(broadcaster: Arc<LogBroadcaster>) -> Self {
        Self {
            broadcaster,
            min_level: Level::TRACE,
        }
    }
}

impl<S> Layer<S> for WsLogLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();

        // Skip if below minimum level
        if *metadata.level() > self.min_level {
            return;
        }

        // Convert tracing level to our LogLevel
        let level = match *metadata.level() {
            Level::TRACE => LogLevel::Trace,
            Level::DEBUG => LogLevel::Debug,
            Level::INFO => LogLevel::Info,
            Level::WARN => LogLevel::Warn,
            Level::ERROR => LogLevel::Error,
        };

        // Extract message from event
        let mut visitor = MessageVisitor::new();
        event.record(&mut visitor);

        // Get current timestamp
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let entry = LogEntry {
            timestamp_ms,
            level,
            target: metadata.target().to_string(),
            message: visitor.message,
        };

        self.broadcaster.broadcast(entry);
    }
}

/// Query parameters for the `/logs` WebSocket endpoint.
#[derive(Debug, Deserialize)]
pub struct LogStreamParams {
    /// Minimum log level to stream. Defaults to Info.
    #[serde(default = "default_log_level")]
    pub level: LogLevel,
}

fn default_log_level() -> LogLevel {
    LogLevel::Info
}

/// WebSocket handler for log streaming.
///
/// Clients connect and receive a stream of JSON-encoded `LogEntry` messages.
/// The `min_level` parameter controls per-client server-side filtering.
pub async fn ws_log_handler(ws: WebSocket, broadcaster: Arc<LogBroadcaster>, min_level: LogLevel) {
    let (mut sender, mut receiver) = ws.split();
    let mut rx = broadcaster.subscribe();

    // Spawn task to handle incoming messages (ping/pong, close)
    let mut recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Close(_)) => break,
                Ok(_) => {}
                Err(e) => {
                    tracing::debug!("Log WebSocket receive error: {}", e);
                    break;
                }
            }
        }
    });

    // Send log entries to client
    loop {
        tokio::select! {
            log_result = rx.recv() => {
                match log_result {
                    Ok(entry) => {
                        if !entry.level.passes_filter(&min_level) {
                            continue;
                        }
                        if let Ok(json) = serde_json::to_string(&entry) {
                            if let Err(e) = sender.send(Message::Text(json)).await {
                                tracing::debug!("Log WebSocket send error: {}", e);
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        // Send a notification about missed logs
                        let missed_msg = format!("{{\"type\":\"lagged\",\"missed\":{n}}}");
                        let _ = sender.send(Message::Text(missed_msg)).await;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
            _ = &mut recv_task => {
                // Receiver task finished (client disconnected)
                break;
            }
        }
    }

    recv_task.abort();
}

/// Initialize tracing with console output and WebSocket streaming.
///
/// Returns a `LogBroadcaster` that streams log entries to WebSocket clients.
/// Pass it to `run_server` to enable the `/logs` endpoint.
pub fn init_tracing() -> Arc<LogBroadcaster> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::EnvFilter;

    let log_broadcaster = Arc::new(LogBroadcaster::new(64));
    let log_layer = WsLogLayer::new(log_broadcaster.clone());

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(log_layer)
        .with(tracing_subscriber::fmt::layer().with_filter(env_filter))
        .init();

    log_broadcaster
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_color() {
        assert_eq!(LogLevel::Error.color(), "#ff4444");
        assert_eq!(LogLevel::Info.color(), "#00ff00");
    }

    #[test]
    fn test_log_level_passes_filter() {
        assert!(LogLevel::Error.passes_filter(&LogLevel::Warn));
        assert!(LogLevel::Warn.passes_filter(&LogLevel::Warn));
        assert!(!LogLevel::Info.passes_filter(&LogLevel::Warn));
        assert!(LogLevel::Trace.passes_filter(&LogLevel::Trace));
        assert!(!LogLevel::Debug.passes_filter(&LogLevel::Error));
    }

    #[test]
    fn test_broadcast_without_subscribers() {
        let broadcaster = LogBroadcaster::new(16);
        let entry = LogEntry {
            timestamp_ms: 12345,
            level: LogLevel::Info,
            target: "test".to_string(),
            message: "hello".to_string(),
        };
        // Should not panic
        broadcaster.broadcast(entry);
    }

    #[tokio::test]
    async fn test_broadcast_with_subscriber() {
        let broadcaster = LogBroadcaster::new(16);
        let mut rx = broadcaster.subscribe();

        let entry = LogEntry {
            timestamp_ms: 12345,
            level: LogLevel::Info,
            target: "test".to_string(),
            message: "hello".to_string(),
        };
        broadcaster.broadcast(entry);

        let received = rx.recv().await.unwrap();
        assert_eq!(received.message, "hello");
        assert_eq!(received.level, LogLevel::Info);
    }
}

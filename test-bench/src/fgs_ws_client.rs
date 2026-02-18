//! Native WebSocket client for bidirectional communication with fgs_server.
//!
//! Wraps a persistent `tokio_tungstenite` connection to the `/ws/status` endpoint,
//! providing typed send/receive of `FgsWsCommand` and `FgsWsMessage`.

use std::mem::discriminant;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use shared_wasm::{FgsWsCommand, FgsWsMessage, TrackingEnableRequest, TrackingState};
use tokio_tungstenite::tungstenite::Message;

/// Error from the FGS WebSocket client.
#[derive(Debug, thiserror::Error)]
pub enum FgsWsClientError {
    /// WebSocket connection failed
    #[error("connect: {0}")]
    Connect(String),
    /// Failed to send a message
    #[error("send: {0}")]
    Send(String),
    /// Failed to receive a message
    #[error("receive: {0}")]
    Receive(String),
    /// JSON serialization failed
    #[error("serialize: {0}")]
    Serialize(#[from] serde_json::Error),
    /// Connection closed unexpectedly
    #[error("connection closed")]
    ConnectionClosed,
    /// Timed out waiting for expected state
    #[error("timeout")]
    Timeout,
}

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Persistent WebSocket client for fgs_server's `/ws/status` endpoint.
pub struct FgsWsClient {
    write: futures::stream::SplitSink<WsStream, Message>,
    read: futures::stream::SplitStream<WsStream>,
}

impl FgsWsClient {
    /// Connect to an FGS server's WebSocket status endpoint.
    pub async fn connect(url: &str) -> Result<Self, FgsWsClientError> {
        let (ws_stream, _) = tokio_tungstenite::connect_async(url)
            .await
            .map_err(|e| FgsWsClientError::Connect(e.to_string()))?;

        let (write, read) = ws_stream.split();
        Ok(Self { write, read })
    }

    /// Send a command to the server (fire-and-forget).
    pub async fn send_command(&mut self, cmd: FgsWsCommand) -> Result<(), FgsWsClientError> {
        let json = serde_json::to_string(&cmd)?;
        self.write
            .send(Message::Text(json.into()))
            .await
            .map_err(|e| FgsWsClientError::Send(e.to_string()))
    }

    /// Read the next `FgsWsMessage` from the stream, skipping unrecognized frames.
    ///
    /// Messages that fail to deserialize (e.g. from an older server) are
    /// silently skipped so the client stays connected.
    pub async fn next_message(&mut self) -> Result<FgsWsMessage, FgsWsClientError> {
        loop {
            match self.read.next().await {
                None => return Err(FgsWsClientError::ConnectionClosed),
                Some(Err(e)) => return Err(FgsWsClientError::Receive(e.to_string())),
                Some(Ok(Message::Text(text))) => {
                    if let Ok(msg) = serde_json::from_str(&text) {
                        return Ok(msg);
                    }
                }
                Some(Ok(_)) => {}
            }
        }
    }

    /// Drain any buffered WebSocket messages, discarding them.
    ///
    /// Reads messages for a fixed 100ms window to clear the buffer without
    /// getting stuck in a loop when messages arrive continuously.
    async fn drain_buffered(&mut self) {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
        loop {
            let remaining = deadline - tokio::time::Instant::now();
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, self.next_message()).await {
                Err(_) => break,
                Ok(Err(_)) => break,
                Ok(Ok(_)) => {}
            }
        }
    }

    /// Send a command then wait for tracking to reach a specific state.
    ///
    /// Drains buffered messages first so only fresh state transitions are
    /// considered. Returns the matching `TrackingState` or times out.
    pub async fn send_command_and_wait(
        &mut self,
        cmd: FgsWsCommand,
        target_state: TrackingState,
        timeout: Duration,
    ) -> Result<TrackingState, FgsWsClientError> {
        self.drain_buffered().await;
        self.send_command(cmd).await?;

        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let msg = tokio::time::timeout_at(deadline, self.next_message()).await;
            match msg {
                Err(_) => return Err(FgsWsClientError::Timeout),
                Ok(Err(e)) => return Err(e),
                Ok(Ok(FgsWsMessage::TrackingStatus(status))) => {
                    if discriminant(&status.state) == discriminant(&target_state) {
                        return Ok(status.state);
                    }
                }
                Ok(Ok(_)) => {}
            }
        }
    }

    /// Enable tracking and wait for it to reach `Tracking` state.
    ///
    /// Returns the tracking state on success, or `Timeout` if lock-on
    /// doesn't happen within `timeout`.
    pub async fn enable_tracking(
        &mut self,
        timeout: Duration,
    ) -> Result<TrackingState, FgsWsClientError> {
        self.send_command_and_wait(
            FgsWsCommand::SetTrackingEnabled(TrackingEnableRequest { enabled: true }),
            TrackingState::Tracking {
                frames_processed: 0,
            },
            timeout,
        )
        .await
    }

    /// Disable tracking and wait for `Idle` state.
    pub async fn disable_tracking(&mut self) -> Result<(), FgsWsClientError> {
        self.send_command_and_wait(
            FgsWsCommand::SetTrackingEnabled(TrackingEnableRequest { enabled: false }),
            TrackingState::Idle,
            Duration::from_secs(5),
        )
        .await?;
        Ok(())
    }
}

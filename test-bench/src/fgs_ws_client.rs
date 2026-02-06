//! Native WebSocket client for bidirectional communication with fgs_server.
//!
//! Wraps a persistent `tokio_tungstenite` connection to the `/ws/status` endpoint,
//! providing typed send/receive of `FgsWsCommand` and `FgsWsMessage`.

use std::mem::discriminant;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use shared_wasm::{FgsWsCommand, FgsWsMessage, TrackingState};
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

    /// Read the next `FgsWsMessage` from the stream, skipping non-text frames.
    pub async fn next_message(&mut self) -> Result<FgsWsMessage, FgsWsClientError> {
        loop {
            match self.read.next().await {
                None => return Err(FgsWsClientError::ConnectionClosed),
                Some(Err(e)) => return Err(FgsWsClientError::Receive(e.to_string())),
                Some(Ok(Message::Text(text))) => {
                    return serde_json::from_str(&text).map_err(FgsWsClientError::from);
                }
                Some(Ok(_)) => {}
            }
        }
    }

    /// Send a command then wait for tracking to reach a specific state.
    ///
    /// Returns the matching `TrackingState` or times out.
    pub async fn send_command_and_wait(
        &mut self,
        cmd: FgsWsCommand,
        target_state: TrackingState,
        timeout: Duration,
    ) -> Result<TrackingState, FgsWsClientError> {
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
}

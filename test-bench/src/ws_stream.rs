//! WebSocket-based image streaming for camera feeds.
//!
//! This module provides WebSocket streaming as an alternative to MJPEG.
//! WebSocket offers better connection lifecycle management:
//! - Clean close events when the connection terminates
//! - Binary frame support for efficient JPEG transfer
//! - Metadata alongside image data
//!
//! # Protocol
//!
//! Each message is a binary frame containing:
//! - First 4 bytes: frame width as u32 little-endian
//! - Next 4 bytes: frame height as u32 little-endian
//! - Next 8 bytes: frame number as u64 little-endian
//! - Remaining bytes: JPEG image data
//!
//! The server sends a close frame when the stream needs to restart
//! (e.g., due to frame size change). Clients should reconnect when
//! they receive the close event.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::broadcast;

/// A frame ready for WebSocket streaming.
#[derive(Clone)]
pub struct WsFrame {
    /// JPEG-encoded image data
    pub jpeg_data: Bytes,
    /// Frame sequence number
    pub frame_number: u64,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
}

impl WsFrame {
    /// Serialize frame into binary message format.
    ///
    /// Format: width(u32) + height(u32) + frame_number(u64) + jpeg_data
    pub fn to_binary(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + self.jpeg_data.len());
        buf.extend_from_slice(&self.width.to_le_bytes());
        buf.extend_from_slice(&self.height.to_le_bytes());
        buf.extend_from_slice(&self.frame_number.to_le_bytes());
        buf.extend_from_slice(&self.jpeg_data);
        buf
    }
}

/// Broadcaster for WebSocket image frames to multiple clients.
///
/// Similar to MjpegBroadcaster but uses WebSocket for proper connection lifecycle.
/// When frame size changes, all clients receive a close frame and should reconnect.
pub struct WsBroadcaster {
    tx: std::sync::RwLock<broadcast::Sender<WsFrame>>,
    capacity: usize,
    last_size: std::sync::RwLock<Option<(u32, u32)>>,
}

impl WsBroadcaster {
    /// Create a new broadcaster with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self {
            tx: std::sync::RwLock::new(tx),
            capacity,
            last_size: std::sync::RwLock::new(None),
        }
    }

    /// Publish a new frame to all subscribers.
    ///
    /// Returns the number of active subscribers, or 0 if none.
    ///
    /// If the frame size has changed, the channel is recreated which
    /// causes existing subscribers to receive an error and disconnect.
    pub fn publish(&self, frame: WsFrame) -> usize {
        let new_size = (frame.width, frame.height);

        // Check if size changed
        let size_changed = {
            let last = self.last_size.read().unwrap();
            last.map(|old| old != new_size).unwrap_or(false)
        };

        if size_changed {
            // Recreate the channel to disconnect all subscribers
            let (new_tx, _) = broadcast::channel(self.capacity);
            let mut tx = self.tx.write().unwrap();
            *tx = new_tx;
            tracing::info!(
                "WS frame size changed to {}x{}, disconnected all subscribers",
                new_size.0,
                new_size.1
            );
        }

        // Update last known size
        {
            let mut last = self.last_size.write().unwrap();
            *last = Some(new_size);
        }

        // Publish to current channel
        let tx = self.tx.read().unwrap();
        tx.send(frame).unwrap_or(0)
    }

    /// Subscribe to the frame stream.
    pub fn subscribe(&self) -> broadcast::Receiver<WsFrame> {
        let tx = self.tx.read().unwrap();
        tx.subscribe()
    }

    /// Get the current number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        let tx = self.tx.read().unwrap();
        tx.receiver_count()
    }
}

impl Default for WsBroadcaster {
    fn default() -> Self {
        Self::new(4)
    }
}

/// Handle WebSocket connection for image streaming.
///
/// This is the handler that gets called for each WebSocket upgrade request.
/// It subscribes to the broadcaster and forwards frames to the client.
pub async fn ws_stream_handler(ws: WebSocket, broadcaster: Arc<WsBroadcaster>) {
    let (mut sender, mut receiver) = ws.split();
    let mut rx = broadcaster.subscribe();

    // Spawn task to handle incoming messages (ping/pong, close)
    let mut recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Close(_)) => break,
                Ok(Message::Ping(data)) => {
                    tracing::trace!("Received ping: {:?}", data);
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::debug!("WebSocket receive error: {}", e);
                    break;
                }
            }
        }
    });

    // Send frames to client
    loop {
        tokio::select! {
            frame_result = rx.recv() => {
                match frame_result {
                    Ok(frame) => {
                        let binary_data = frame.to_binary();
                        if let Err(e) = sender.send(Message::Binary(binary_data)).await {
                            tracing::debug!("WebSocket send error: {}", e);
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::debug!("WebSocket client lagged {} frames", n);
                        // Continue receiving - client will get next frame
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        // Channel was recreated (frame size changed)
                        // Send close frame to client
                        tracing::info!("WebSocket stream channel closed, sending close to client");
                        let _ = sender.send(Message::Close(Some(axum::extract::ws::CloseFrame {
                            code: 1000,
                            reason: "Stream restarting".into(),
                        }))).await;
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
    tracing::debug!("WebSocket connection closed");
}

/// Create a WebSocket upgrade handler for the image stream endpoint.
///
/// Use this to create the route handler - use `.route("/ws/frames", get(ws_stream_endpoint))`
pub async fn ws_stream_endpoint<T: Send + Sync + 'static>(
    State(broadcaster): State<Arc<WsBroadcaster>>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| ws_stream_handler(socket, broadcaster))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_frame_to_binary() {
        let frame = WsFrame {
            jpeg_data: Bytes::from_static(b"\xFF\xD8test"),
            frame_number: 42,
            width: 640,
            height: 480,
        };

        let binary = frame.to_binary();

        // Check header
        assert_eq!(&binary[0..4], &640u32.to_le_bytes());
        assert_eq!(&binary[4..8], &480u32.to_le_bytes());
        assert_eq!(&binary[8..16], &42u64.to_le_bytes());
        // Check JPEG data
        assert_eq!(&binary[16..], b"\xFF\xD8test");
    }

    #[test]
    fn test_broadcaster_creation() {
        let broadcaster = WsBroadcaster::new(4);
        assert_eq!(broadcaster.subscriber_count(), 0);
    }

    #[test]
    fn test_publish_without_subscribers() {
        let broadcaster = WsBroadcaster::new(4);
        let frame = WsFrame {
            jpeg_data: Bytes::from_static(b"test"),
            frame_number: 1,
            width: 640,
            height: 480,
        };
        // Should not panic, just return 0
        assert_eq!(broadcaster.publish(frame), 0);
    }

    #[test]
    fn test_frame_size_change_disconnects_subscribers() {
        let broadcaster = WsBroadcaster::new(4);

        // Publish first frame to establish size
        let frame1 = WsFrame {
            jpeg_data: Bytes::from_static(b"frame1"),
            frame_number: 1,
            width: 640,
            height: 480,
        };
        broadcaster.publish(frame1);

        // Subscribe after first frame
        let _rx = broadcaster.subscribe();
        assert_eq!(broadcaster.subscriber_count(), 1);

        // Publish frame with same size - subscriber should remain
        let frame2 = WsFrame {
            jpeg_data: Bytes::from_static(b"frame2"),
            frame_number: 2,
            width: 640,
            height: 480,
        };
        broadcaster.publish(frame2);
        assert_eq!(broadcaster.subscriber_count(), 1);

        // Publish frame with different size - subscriber should be disconnected
        let frame3 = WsFrame {
            jpeg_data: Bytes::from_static(b"frame3"),
            frame_number: 3,
            width: 320,
            height: 240,
        };
        broadcaster.publish(frame3);
        // After channel recreation, old subscriber is disconnected
        assert_eq!(broadcaster.subscriber_count(), 0);
    }
}

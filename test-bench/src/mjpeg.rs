//! MJPEG streaming support for camera feeds.
//!
//! This module provides utilities for creating Motion JPEG (MJPEG) streams
//! that can be consumed by web browsers. MJPEG uses `multipart/x-mixed-replace`
//! content type where each frame is sent as a separate JPEG with boundary markers.
//!
//! # Browser Compatibility
//!
//! Simply set `<img src="/mjpeg">` and the browser handles everything:
//! - Automatic frame updates without JavaScript polling
//! - Smooth playback at whatever frame rate the server provides
//! - Memory-efficient (browser replaces previous frame, doesn't accumulate)

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::Response,
};
use bytes::Bytes;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

/// Boundary string used to separate MJPEG frames.
/// Must be unique enough to not appear in JPEG data.
const MJPEG_BOUNDARY: &str = "frame_boundary_7f8c3d2e";

/// A frame ready for MJPEG streaming.
#[derive(Clone)]
pub struct MjpegFrame {
    /// JPEG-encoded image data
    pub jpeg_data: Bytes,
    /// Frame sequence number (for debugging/logging)
    pub frame_number: u64,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
}

/// Broadcaster for MJPEG frames to multiple HTTP clients.
///
/// Multiple clients can subscribe to the same frame source. Each frame
/// is broadcast to all connected clients. Slow clients that fall behind
/// will skip frames (they receive the latest available frame).
///
/// When the frame size changes (e.g., switching to/from ROI mode), the
/// broadcaster recreates the channel, causing all subscribers to receive
/// an error and disconnect. Clients should handle this by restarting the stream.
pub struct MjpegBroadcaster {
    tx: std::sync::RwLock<broadcast::Sender<MjpegFrame>>,
    capacity: usize,
    last_size: std::sync::RwLock<Option<(u32, u32)>>,
}

impl MjpegBroadcaster {
    /// Create a new broadcaster with the given channel capacity.
    ///
    /// Capacity determines how many frames can be buffered before slow
    /// receivers start missing frames. A capacity of 2-4 is usually sufficient.
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
    /// If the frame size has changed from the previous frame, all existing
    /// subscribers are disconnected (the channel is recreated). This allows
    /// clients to detect the size change via stream error and reconnect.
    pub fn publish(&self, frame: MjpegFrame) -> usize {
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
                "MJPEG frame size changed to {}x{}, disconnected all subscribers",
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

    /// Create a subscriber that receives frames from this broadcaster.
    pub fn subscribe(&self) -> MjpegSubscriber {
        let tx = self.tx.read().unwrap();
        MjpegSubscriber { rx: tx.subscribe() }
    }

    /// Get the current number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        let tx = self.tx.read().unwrap();
        tx.receiver_count()
    }
}

impl Default for MjpegBroadcaster {
    fn default() -> Self {
        Self::new(4)
    }
}

/// A subscriber to an MJPEG frame stream.
pub struct MjpegSubscriber {
    rx: broadcast::Receiver<MjpegFrame>,
}

impl MjpegSubscriber {
    /// Convert this subscriber into an Axum response that streams MJPEG.
    ///
    /// The response uses `multipart/x-mixed-replace` content type which
    /// browsers understand natively for continuous image updates.
    pub fn into_response(self) -> Response {
        let stream = BroadcastStream::new(self.rx).filter_map(|result| {
            match result {
                Ok(frame) => {
                    // Format: boundary + headers + data
                    let part = format!(
                        "--{boundary}\r\n\
                         Content-Type: image/jpeg\r\n\
                         Content-Length: {len}\r\n\
                         \r\n",
                        boundary = MJPEG_BOUNDARY,
                        len = frame.jpeg_data.len()
                    );

                    let mut bytes = Vec::with_capacity(part.len() + frame.jpeg_data.len() + 2);
                    bytes.extend_from_slice(part.as_bytes());
                    bytes.extend_from_slice(&frame.jpeg_data);
                    bytes.extend_from_slice(b"\r\n");

                    Some(Ok::<_, std::convert::Infallible>(Bytes::from(bytes)))
                }
                Err(_) => {
                    // BroadcastStreamRecvError wraps either Lagged or channel closed
                    // In either case, skip this frame and continue
                    None
                }
            }
        });

        let body = Body::from_stream(stream);

        Response::builder()
            .status(StatusCode::OK)
            .header(
                header::CONTENT_TYPE,
                format!("multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}"),
            )
            .header(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")
            .header(header::PRAGMA, "no-cache")
            .header(header::EXPIRES, "0")
            .body(body)
            .expect("Failed to build MJPEG response")
    }
}

/// Shared state for MJPEG streaming in a camera server.
///
/// Wrap this in `Arc` and share it between the capture loop (which publishes)
/// and HTTP handlers (which create subscribers).
pub struct MjpegState {
    /// Broadcaster for full camera frames
    pub camera_feed: MjpegBroadcaster,
    /// Broadcaster for ROI/zoom region (optional, used by fgs_server)
    pub roi_feed: Option<MjpegBroadcaster>,
}

impl MjpegState {
    /// Create a new MJPEG state with camera feed only.
    pub fn new() -> Self {
        Self {
            camera_feed: MjpegBroadcaster::new(4),
            roi_feed: None,
        }
    }

    /// Create a new MJPEG state with both camera and ROI feeds.
    pub fn with_roi() -> Self {
        Self {
            camera_feed: MjpegBroadcaster::new(4),
            roi_feed: Some(MjpegBroadcaster::new(4)),
        }
    }
}

impl Default for MjpegState {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to create an MJPEG endpoint handler.
///
/// Use this to create route handlers that stream from a broadcaster.
/// In your router, subscribe to the broadcaster and call this function.
pub async fn mjpeg_endpoint(subscriber: MjpegSubscriber) -> Response {
    subscriber.into_response()
}

/// Encode a grayscale frame as JPEG for MJPEG streaming.
///
/// This is a convenience function that handles the common case of
/// encoding u8 grayscale image data to JPEG bytes.
pub fn encode_gray_jpeg(data: &[u8], width: u32, height: u32, quality: u8) -> Option<Bytes> {
    use image::{GrayImage, ImageBuffer};

    let img: GrayImage = ImageBuffer::from_raw(width, height, data.to_vec())?;

    let mut jpeg_bytes = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_bytes, quality);

    encoder.encode_image(&img).ok()?;

    Some(Bytes::from(jpeg_bytes))
}

/// Encode a u16 grayscale frame as JPEG for MJPEG streaming.
///
/// Converts 16-bit to 8-bit using simple right shift (divide by 256).
pub fn encode_u16_gray_jpeg(data: &[u16], width: u32, height: u32, quality: u8) -> Option<Bytes> {
    let u8_data: Vec<u8> = data.iter().map(|&v| (v >> 8) as u8).collect();
    encode_gray_jpeg(&u8_data, width, height, quality)
}

/// Encode an ndarray frame as JPEG for MJPEG streaming.
pub fn encode_ndarray_jpeg(frame: &ndarray::Array2<u16>, quality: u8) -> Option<Bytes> {
    let height = frame.nrows() as u32;
    let width = frame.ncols() as u32;

    // Convert to row-major u8
    let u8_data: Vec<u8> = frame.iter().map(|&v| (v >> 8) as u8).collect();

    encode_gray_jpeg(&u8_data, width, height, quality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcaster_creation() {
        let broadcaster = MjpegBroadcaster::new(4);
        assert_eq!(broadcaster.subscriber_count(), 0);
    }

    #[test]
    fn test_publish_without_subscribers() {
        let broadcaster = MjpegBroadcaster::new(4);
        let frame = MjpegFrame {
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
        let broadcaster = MjpegBroadcaster::new(4);

        // Publish first frame to establish size
        let frame1 = MjpegFrame {
            jpeg_data: Bytes::from_static(b"frame1"),
            frame_number: 1,
            width: 640,
            height: 480,
        };
        broadcaster.publish(frame1);

        // Subscribe after first frame
        let _sub = broadcaster.subscribe();
        assert_eq!(broadcaster.subscriber_count(), 1);

        // Publish frame with same size - subscriber should remain
        let frame2 = MjpegFrame {
            jpeg_data: Bytes::from_static(b"frame2"),
            frame_number: 2,
            width: 640,
            height: 480,
        };
        broadcaster.publish(frame2);
        assert_eq!(broadcaster.subscriber_count(), 1);

        // Publish frame with different size - subscriber should be disconnected
        let frame3 = MjpegFrame {
            jpeg_data: Bytes::from_static(b"frame3"),
            frame_number: 3,
            width: 320,
            height: 240,
        };
        broadcaster.publish(frame3);
        // After channel recreation, old subscriber is disconnected
        assert_eq!(broadcaster.subscriber_count(), 0);
    }

    #[test]
    fn test_subscriber_count() {
        let broadcaster = MjpegBroadcaster::new(4);
        assert_eq!(broadcaster.subscriber_count(), 0);

        let _sub1 = broadcaster.subscribe();
        assert_eq!(broadcaster.subscriber_count(), 1);

        let _sub2 = broadcaster.subscribe();
        assert_eq!(broadcaster.subscriber_count(), 2);

        drop(_sub1);
        assert_eq!(broadcaster.subscriber_count(), 1);
    }

    #[test]
    fn test_encode_gray_jpeg() {
        // 2x2 gray image
        let data = vec![0u8, 64, 128, 255];
        let result = encode_gray_jpeg(&data, 2, 2, 80);
        assert!(result.is_some());

        let jpeg = result.unwrap();
        // JPEG magic bytes
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_encode_u16_gray_jpeg() {
        // 2x2 u16 image (values 0, 16384, 32768, 65535)
        let data = vec![0u16, 16384, 32768, 65535];
        let result = encode_u16_gray_jpeg(&data, 2, 2, 80);
        assert!(result.is_some());

        let jpeg = result.unwrap();
        // JPEG magic bytes
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }
}

//! Type-safe ZeroMQ pub/sub wrappers for JSON-serialized messages.
//!
//! Provides `TypedZmqPublisher` and `TypedZmqSubscriber` that wrap pre-configured
//! sockets and handle JSON serialization/deserialization.

use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;
use std::sync::Mutex;
use thiserror::Error;

/// Errors that can occur during ZMQ operations.
#[derive(Error, Debug)]
pub enum ZmqError {
    #[error("ZMQ socket error: {0}")]
    Socket(#[from] zmq::Error),
    #[error("JSON serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// A type-safe ZMQ publisher that serializes messages to JSON.
///
/// Thread-safe: can be shared across threads via `Arc<TypedZmqPublisher<T>>`.
/// The internal mutex ensures safe concurrent access.
pub struct TypedZmqPublisher<T> {
    socket: Mutex<zmq::Socket>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Serialize> TypedZmqPublisher<T> {
    /// Create a new typed publisher from a pre-configured socket.
    ///
    /// The socket should already be bound to an address.
    pub fn new(socket: zmq::Socket) -> Self {
        Self {
            socket: Mutex::new(socket),
            _marker: PhantomData,
        }
    }

    /// Send a message, serializing it to JSON.
    ///
    /// Thread-safe: acquires internal mutex.
    pub fn send(&self, msg: &T) -> Result<(), ZmqError> {
        let json = serde_json::to_string(msg)?;
        let socket = self.socket.lock().unwrap();
        socket.send(&json, 0)?;
        Ok(())
    }
}

// Safety: The Mutex protects the socket, and zmq::Socket has internal Arc to context
unsafe impl<T> Send for TypedZmqPublisher<T> {}
unsafe impl<T> Sync for TypedZmqPublisher<T> {}

/// A type-safe ZMQ subscriber that deserializes JSON messages.
///
/// NOT thread-safe by design - use from a single thread.
/// For multi-threaded use, wrap in your own synchronization.
pub struct TypedZmqSubscriber<T> {
    socket: zmq::Socket,
    _marker: PhantomData<fn() -> T>,
}

impl<T: DeserializeOwned> TypedZmqSubscriber<T> {
    /// Create a new typed subscriber from a pre-configured socket.
    ///
    /// The socket should already be connected and subscribed.
    /// Sets non-blocking mode (rcvtimeo=0) for use with try_recv/drain.
    pub fn new(socket: zmq::Socket) -> Self {
        // Set non-blocking for our try_recv/drain pattern
        let _ = socket.set_rcvtimeo(0);
        Self {
            socket,
            _marker: PhantomData,
        }
    }

    /// Try to receive a single message (non-blocking).
    ///
    /// Returns `None` if no message is available or if deserialization fails.
    pub fn try_recv(&self) -> Option<T> {
        match self.socket.recv_string(zmq::DONTWAIT) {
            Ok(Ok(json_str)) => serde_json::from_str(&json_str).ok(),
            _ => None,
        }
    }

    /// Drain all pending messages (non-blocking).
    ///
    /// Returns a Vec of successfully deserialized messages.
    /// Messages that fail to deserialize are silently dropped.
    pub fn drain(&self) -> Vec<T> {
        let mut messages = Vec::new();
        while let Some(msg) = self.try_recv() {
            messages.push(msg);
        }
        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestMessage {
        id: u32,
        value: f64,
    }

    #[test]
    fn test_publisher_creation() {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::PUB).unwrap();
        socket.bind("tcp://127.0.0.1:*").unwrap();
        let _publisher = TypedZmqPublisher::<TestMessage>::new(socket);
    }

    #[test]
    fn test_subscriber_creation() {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::SUB).unwrap();
        socket.connect("tcp://127.0.0.1:59999").unwrap();
        socket.set_subscribe(b"").unwrap();
        let _subscriber = TypedZmqSubscriber::<TestMessage>::new(socket);
    }

    #[test]
    fn test_subscriber_drain_empty() {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::SUB).unwrap();
        socket.connect("tcp://127.0.0.1:59998").unwrap();
        socket.set_subscribe(b"").unwrap();
        let subscriber = TypedZmqSubscriber::<TestMessage>::new(socket);
        let messages = subscriber.drain();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_pub_sub_communication() {
        let ctx = zmq::Context::new();

        // Create and bind publisher
        let pub_socket = ctx.socket(zmq::PUB).unwrap();
        pub_socket.bind("tcp://127.0.0.1:15555").unwrap();
        let publisher = TypedZmqPublisher::<TestMessage>::new(pub_socket);

        // Create and connect subscriber (rcvtimeo set automatically)
        let sub_socket = ctx.socket(zmq::SUB).unwrap();
        sub_socket.connect("tcp://127.0.0.1:15555").unwrap();
        sub_socket.set_subscribe(b"").unwrap();
        let subscriber = TypedZmqSubscriber::<TestMessage>::new(sub_socket);

        // ZMQ pub/sub has "slow joiner" problem - subscriber needs time to connect
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Send some messages
        let msg1 = TestMessage { id: 1, value: 1.5 };
        let msg2 = TestMessage { id: 2, value: 2.5 };
        let msg3 = TestMessage { id: 3, value: 3.5 };

        publisher.send(&msg1).unwrap();
        publisher.send(&msg2).unwrap();
        publisher.send(&msg3).unwrap();

        // Give messages time to arrive
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Drain and verify
        let received = subscriber.drain();
        assert_eq!(received.len(), 3);
        assert_eq!(received[0], msg1);
        assert_eq!(received[1], msg2);
        assert_eq!(received[2], msg3);
    }
}

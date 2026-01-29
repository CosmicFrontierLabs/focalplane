//! SSE-based tracking telemetry collection.
//!
//! Provides a simple interface for collecting tracking measurements via Server-Sent Events (SSE)
//! from an HTTP endpoint.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use shared::tracking_message::TrackingMessage;

/// Error type for collection operations.
#[derive(Debug, Clone)]
pub enum CollectError {
    /// Operation timed out before completing
    Timeout {
        got: usize,
        expected: usize,
        elapsed: Duration,
    },
    /// SSE connection was lost during collection
    Disconnected { got: usize, error: Option<String> },
}

impl std::fmt::Display for CollectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollectError::Timeout {
                got,
                expected,
                elapsed,
            } => {
                write!(
                    f,
                    "Timeout: collected {got}/{expected} messages in {elapsed:?}"
                )
            }
            CollectError::Disconnected { got, error } => {
                if let Some(e) = error {
                    write!(f, "Disconnected after {got} messages: {e}")
                } else {
                    write!(f, "Disconnected after {got} messages")
                }
            }
        }
    }
}

impl std::error::Error for CollectError {}

/// Shared state between the collector and background reader thread.
struct SharedState {
    buffer: VecDeque<TrackingMessage>,
    connected: bool,
    error: Option<String>,
}

impl SharedState {
    fn new() -> Self {
        Self {
            buffer: VecDeque::with_capacity(1024),
            connected: false,
            error: None,
        }
    }
}

/// Collects tracking measurements from an SSE endpoint.
///
/// Connects to an HTTP SSE endpoint and buffers incoming TrackingMessage events.
/// The connection runs in a background thread, allowing non-blocking polling.
pub struct TrackingCollector {
    state: Arc<Mutex<SharedState>>,
}

impl TrackingCollector {
    /// Connect to an SSE endpoint and start collecting messages.
    ///
    /// Waits up to `timeout` for the connection to establish. The endpoint should
    /// be a URL like `http://host:port/tracking/events`.
    ///
    /// # Errors
    /// Returns an error if the connection fails or times out.
    pub fn connect_with_timeout(endpoint: &str, timeout: Duration) -> Result<Self, String> {
        let state = Arc::new(Mutex::new(SharedState::new()));
        let signal: Arc<OnceLock<Result<(), String>>> = Arc::new(OnceLock::new());

        let state_clone = state.clone();
        let signal_clone = signal.clone();
        let endpoint = endpoint.to_string();

        thread::spawn(move || {
            Self::sse_reader_thread(endpoint, state_clone, signal_clone);
        });

        // Wait for initial connection result
        let start = Instant::now();
        while start.elapsed() < timeout {
            if let Some(result) = signal.get() {
                return match result {
                    Ok(()) => Ok(Self { state }),
                    Err(e) => Err(e.clone()),
                };
            }
            thread::sleep(Duration::from_millis(10));
        }

        Err(format!("Connection timed out after {timeout:?}"))
    }

    /// Connect to an SSE endpoint with default 5 second timeout.
    ///
    /// # Errors
    /// Returns an error if the connection fails or times out.
    pub fn connect(endpoint: &str) -> Result<Self, String> {
        Self::connect_with_timeout(endpoint, Duration::from_secs(5))
    }

    /// Poll for all currently available messages (non-blocking).
    ///
    /// Returns immediately with all pending messages, or an empty Vec if none.
    /// Also returns an error if the connection has been lost.
    pub fn poll(&self) -> Result<Vec<TrackingMessage>, CollectError> {
        let mut state = self.state.lock().unwrap();
        let messages: Vec<_> = state.buffer.drain(..).collect();

        if !state.connected && messages.is_empty() {
            return Err(CollectError::Disconnected {
                got: 0,
                error: state.error.clone(),
            });
        }

        Ok(messages)
    }

    /// Collect messages for a specified duration.
    ///
    /// Blocks until the duration elapses, collecting all messages received
    /// during that time. Returns error if connection is lost during collection.
    pub fn collect(&self, duration: Duration) -> Result<Vec<TrackingMessage>, CollectError> {
        let start = Instant::now();
        let mut messages = Vec::new();

        while start.elapsed() < duration {
            match self.poll() {
                Ok(msgs) => messages.extend(msgs),
                Err(CollectError::Disconnected { error, .. }) => {
                    return Err(CollectError::Disconnected {
                        got: messages.len(),
                        error,
                    });
                }
                Err(e) => return Err(e),
            }
            thread::sleep(Duration::from_millis(10));
        }

        // Final poll
        if let Ok(msgs) = self.poll() {
            messages.extend(msgs);
        }

        Ok(messages)
    }

    /// Collect exactly N messages or return an error.
    ///
    /// Blocks until `count` messages are collected or `timeout` elapses.
    ///
    /// # Errors
    /// Returns `CollectError::Timeout` if timeout elapses before collecting enough messages.
    /// Returns `CollectError::Disconnected` if the SSE connection is lost.
    pub fn collect_n(
        &self,
        count: usize,
        timeout: Duration,
    ) -> Result<Vec<TrackingMessage>, CollectError> {
        let start = Instant::now();
        let mut messages = Vec::with_capacity(count);

        while messages.len() < count && start.elapsed() < timeout {
            match self.poll() {
                Ok(msgs) => {
                    for msg in msgs {
                        messages.push(msg);
                        if messages.len() >= count {
                            break;
                        }
                    }
                }
                Err(CollectError::Disconnected { error, .. }) => {
                    return Err(CollectError::Disconnected {
                        got: messages.len(),
                        error,
                    });
                }
                Err(e) => return Err(e),
            }
            if messages.len() < count {
                thread::sleep(Duration::from_millis(10));
            }
        }

        if messages.len() < count {
            return Err(CollectError::Timeout {
                got: messages.len(),
                expected: count,
                elapsed: start.elapsed(),
            });
        }

        Ok(messages)
    }

    /// Wait for at least one message to arrive.
    ///
    /// Returns `Ok(true)` if a message was received within the timeout.
    /// Returns `Ok(false)` if timeout elapsed with no messages.
    /// Returns `Err` if connection was lost.
    pub fn wait_for_message(&self, timeout: Duration) -> Result<bool, CollectError> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            match self.poll() {
                Ok(msgs) if !msgs.is_empty() => return Ok(true),
                Ok(_) => {}
                Err(e) => return Err(e),
            }
            thread::sleep(Duration::from_millis(50));
        }
        Ok(false)
    }

    /// Check if the SSE connection is currently active.
    pub fn is_connected(&self) -> bool {
        self.state.lock().unwrap().connected
    }

    /// Get the last connection error, if any.
    pub fn last_error(&self) -> Option<String> {
        self.state.lock().unwrap().error.clone()
    }

    /// Clear all buffered messages.
    ///
    /// Use this before collecting samples to ensure only fresh messages
    /// are captured, discarding any that accumulated during moves or waits.
    pub fn clear(&self) {
        self.state.lock().unwrap().buffer.clear();
    }

    fn sse_reader_thread(
        endpoint: String,
        state: Arc<Mutex<SharedState>>,
        signal: Arc<OnceLock<Result<(), String>>>,
    ) {
        loop {
            match ureq::get(&endpoint).call() {
                Ok(response) => {
                    {
                        let mut s = state.lock().unwrap();
                        s.connected = true;
                        s.error = None;
                    }
                    let _ = signal.set(Ok(()));

                    let reader = BufReader::new(response.into_body().into_reader());
                    for line in reader.lines() {
                        match line {
                            Ok(line) => {
                                if let Some(data) = line.strip_prefix("data: ") {
                                    if let Ok(msg) = serde_json::from_str::<TrackingMessage>(data) {
                                        let mut s = state.lock().unwrap();
                                        s.buffer.push_back(msg);
                                        while s.buffer.len() > 10000 {
                                            s.buffer.pop_front();
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                let mut s = state.lock().unwrap();
                                s.connected = false;
                                s.error = Some(format!("SSE read error: {e}"));
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("SSE connection failed: {e}");
                    {
                        let mut s = state.lock().unwrap();
                        s.connected = false;
                        s.error = Some(err_msg.clone());
                    }
                    let _ = signal.set(Err(err_msg));
                }
            }

            thread::sleep(Duration::from_secs(1));
        }
    }
}
